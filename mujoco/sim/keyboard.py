#!/usr/bin/env python3
"""Velocity command for the PIE driver -- global hold-to-move keys.

Lifecycle ported from the gazebo keyboard_teleop: arrow held = fixed
velocity; released = zero, and the source stays "active" (flushing the
zero command downstream) for RELEASE_FLUSH_S, then goes IDLE -- in gazebo
that means the /cmd_vel stream stops (so it cannot fight an interleaved
autonomous controller); here the control loop always feeds the policy, so
idle still means zeros, but `active()` reports live/flush/idle and the
status line shows it. GLOBAL capture at the KERNEL input layer,
independent of display server (Wayland/X11) and window focus:

1. PRIMARY: /dev/input EVIOCGKEY state poll (50 Hz). Reads the kernel
   key bitmap directly -- the layer BEFORE the compositor, so it sees
   the physical keyboard no matter which window is focused (or none).
   Needs read access to /dev/input/event*: user in the `input` group
   (`sudo usermod -aG input $USER` + re-login; `sg input -c ...` for
   already-open shells). Zero dependencies (raw fcntl ioctl).
   NOTE: X-test synthetic keys (pyautogui) do NOT appear here; only
   real hardware keys do.
2. FALLBACK: XQueryKeymap poll (sees X/XWayland windows only) then the
   mujoco viewer callback alone.

The mujoco key_callback (sim/mj hosts) stays for Space (stop) and R
(reset edge) -- deliberately window-scoped so stray R presses elsewhere
cannot reset. hil.py binds NO callback at all: Reset/Stop are web GUI
buttons only, and KeyboardOverride layers the arrows over the stack's
DDS /cmd_vel with these same release semantics.
A callback arrow press has no keyup event, so it counts as "held" for
CALLBACK_HOLD_S (its analog of one hold), then exits through the same
release-flush-to-idle path.

    Up      vx forward              Down    vx back
    Left    wz left                 Right   wz right
    ,       vy left                 .       vy right   (strafe)
    Space   stop    R reset (viewer window / web GUI button)

Hosts with a policy (view, hil) call set_envelope() with its training
command range: a held key then drives `speed` (default SPEED, the web
panel's Keyboard speed) times that range's end in its direction. Without
an envelope the fixed VX_HOLD / VX_HOLD_BWD / WZ_HOLD / VY_HOLD apply.

ConstantCmd: scripted (vx, wz) for headless smokes, optionally segmented.
"""

from __future__ import annotations

import threading

import numpy as np

VX_HOLD = 0.5    # teleop LINEAR_VEL; PIE flat range (0, 1.5)
VX_HOLD_BWD = 0.2  # teleop backward cap; training band (-0.25, -0.05),
                  # tracked at ~72% median by the flat specialist (10496)
WZ_HOLD = 0.3    # teleop ANGULAR_VEL; user cap ~0.10 m/s at the nose
                 # (r~0.3 m) => wz <= 0.33; in-place band (0.10, 0.33)
VY_HOLD = 0.3    # strafe; go2_depth_v2 trains |vy| <= 0.5
SPEED = 0.5      # held keys = this share of the policy's command range
CALLBACK_HOLD_S = 0.8   # callback press = hold this long (no reliable repeat)
RELEASE_FLUSH_S = 1.0   # post-release zero-flush window (gazebo RELEASE_FLUSH_S)

_ARROW_ACTIONS = {265: "vx+", 264: "vx-", 263: "wz+", 262: "wz-",
                  44: "vy+", 46: "vy-"}

# kernel input.h key codes -> the GLFW-ish codes used internally
_KERNEL_TO_CODE = {103: 265, 108: 264, 105: 263, 106: 262,  # UP DOWN LEFT RIGHT
                   51: 44, 52: 46}                          # COMMA DOT
# A device counts as a keyboard when it has the arrow keys.
_KERNEL_ARROWS = (103, 108, 105, 106)


def _ioc(dir_bits: int, type_: int, nr: int, size: int) -> int:
    return (dir_bits << 30) | (size << 16) | (type_ << 8) | nr


_EVIOCGKEY = _ioc(2, 0x45, 0x18, 512)      # EVIOCGKEY(512), _IOC_READ=2
_EVIOCGBIT_KEY = _ioc(2, 0x45, 0x21, 512)  # EVIOCGBIT(EV_KEY=1, 512)


class InputPoll:
    """Global arrow key state from the kernel input layer."""

    def __init__(self) -> None:
        import glob
        import os
        self._os = os
        self._fcntl = __import__("fcntl")
        self._fds: list[int] = []
        for path in sorted(glob.glob("/dev/input/event*")):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                if self._has_arrows(fd):
                    self._fds.append(fd)
            except OSError:
                pass
            if fd not in self._fds:
                os.close(fd)
        if not self._fds:
            raise RuntimeError(
                "no readable keyboard under /dev/input "
                "(input group missing? `sudo usermod -aG input $USER`)")

    def _bits(self, fd: int, request: int) -> bytearray:
        buf = bytearray(512)
        self._fcntl.ioctl(fd, request, buf, True)
        return buf

    def _has_arrows(self, fd: int) -> bool:
        caps = self._bits(fd, _EVIOCGBIT_KEY)
        return all(caps[k >> 3] & (1 << (k & 7))
                   for k in _KERNEL_ARROWS)

    def arrows(self) -> set:
        """Codes currently physically held, OR-ed across keyboards."""
        out = set()
        for fd in self._fds:
            try:
                state = self._bits(fd, _EVIOCGKEY)
            except OSError:
                continue
            for k, code in _KERNEL_TO_CODE.items():
                if state[k >> 3] & (1 << (k & 7)):
                    out.add(code)
        return out

    def close(self) -> None:
        for fd in self._fds:
            try:
                self._os.close(fd)
            except OSError:
                pass
        self._fds = []


class HoldCmd:
    def __init__(self, use_global: bool = True) -> None:
        self._lock = threading.Lock()
        self.speed = SPEED
        self._envelope: tuple[np.ndarray, np.ndarray] | None = None
        # (vx forward, vx back, vy, wz) a held key commands
        self._hold = (VX_HOLD, VX_HOLD_BWD, VY_HOLD, WZ_HOLD)
        self._reset_pending = False
        self._cb_held: dict[int, float] = {}   # code -> last event time
        self._flush_until = 0.0    # gazebo RELEASE_FLUSH_S window after release
        self._was_held = False
        import time
        self._time = time.monotonic
        self._input_poll = None
        self._x_poll = None
        if use_global:
            try:
                self._input_poll = InputPoll()
                print(f"[KEY] /dev/input global poll active: "
                      f"{len(self._input_poll._fds)} keyboard(s)", flush=True)
            except Exception as exc:       # noqa: BLE001 -- degrade, not die
                print(f"[KEY] /dev/input poll unavailable ({exc})", flush=True)
                try:
                    from Xlib import display, XK
                    dpy = display.Display()
                    kc_map = {}
                    for name, code in (("Up", 265), ("Down", 264),
                                       ("Left", 263), ("Right", 262),
                                       ("comma", 44), ("period", 46)):
                        kc = dpy.keysym_to_keycode(XK.string_to_keysym(name))
                        if kc:
                            kc_map[int(kc)] = code
                    if not kc_map:
                        raise RuntimeError("no arrow keycodes resolved")
                    self._dpy = dpy
                    self._kc_map = kc_map
                    self._x_poll = self._poll_x
                    print("[KEY] X keymap poll fallback active "
                          "(X windows only)", flush=True)
                except Exception as exc2:  # noqa: BLE001
                    print(f"[KEY] X poll unavailable too ({exc2}); "
                          "mujoco-callback keys only", flush=True)

    def _poll_x(self) -> set:
        keymap = self._dpy.query_keymap()
        return {code for kc, code in self._kc_map.items()
                if keymap[kc >> 3] & (1 << (kc & 7))}

    # -- mujoco viewer key_callback ----------------------------------------
    def key(self, keycode: int) -> bool:
        k = int(keycode)
        now = self._time()
        if k in _ARROW_ACTIONS:
            with self._lock:
                self._cb_held[k] = now     # press (or repeat) = hold
            return True
        if k == 32:                        # Space: stop everything
            with self._lock:
                self._cb_held.clear()
            return True
        if k == 82:                        # R: reset edge
            self._reset_pending = True
            return True
        return False

    # -- polled by the control loop (50 Hz) --------------------------------
    def _held_set(self) -> set:
        held: set = set()
        if self._input_poll is not None:
            try:
                held |= self._input_poll.arrows()
            except Exception:              # noqa: BLE001 -- transient
                pass
        if self._x_poll is not None:
            try:
                held |= self._poll_x()
            except Exception:              # noqa: BLE001 -- transient
                pass
        now = self._time()
        with self._lock:
            live = {k: t for k, t in self._cb_held.items()
                    if now - t <= CALLBACK_HOLD_S}
            self._cb_held = live
        held |= set(live)
        return held

    def set_envelope(self, low, high) -> None:
        """The active policy's training command range (vx, vy, wz)."""
        with self._lock:
            self._envelope = (np.asarray(low, dtype=np.float32),
                              np.asarray(high, dtype=np.float32))
            self._update_hold()

    def set_speed(self, fraction: float) -> None:
        """Held keys drive this share of the command range (0.05..1)."""
        with self._lock:
            self.speed = float(np.clip(fraction, 0.05, 1.0))
            self._update_hold()

    def _update_hold(self) -> None:
        if self._envelope is None:
            return
        low, high = self._envelope
        s = self.speed
        self._hold = (s * float(high[0]), s * float(-low[0]),
                      s * float(min(high[1], -low[1])),
                      s * float(min(high[2], -low[2])))

    def hold_speeds(self) -> tuple[float, float, float, float]:
        """(vx forward, vx back, vy, wz) a held key commands."""
        with self._lock:
            return self._hold

    def vector(self) -> np.ndarray:
        held = self._held_set()
        fwd, back, vy_hold, wz_hold = self.hold_speeds()
        vx = (fwd if 265 in held else 0.0) - (back if 264 in held else 0.0)
        vy = (vy_hold if 44 in held else 0.0) - (vy_hold if 46 in held else 0.0)
        wz = (wz_hold if 263 in held else 0.0) - (wz_hold if 262 in held else 0.0)
        return np.asarray((vx, vy, wz), dtype=np.float32)

    def active(self) -> str:
        """Teleop lifecycle, gazebo teleop semantics: 'live' while an arrow
        is held, 'flush' for RELEASE_FLUSH_S after the last release (gazebo
        keeps publishing the zero there), then 'idle' (gazebo goes silent on
        /cmd_vel; here the loop keeps feeding zeros, so idle is informational
        and for the status line)."""
        held = self._held_set()
        now = self._time()
        with self._lock:
            if held:
                self._was_held = True
                self._flush_until = 0.0
                return "live"
            if self._was_held:      # held -> empty edge: arm the flush window
                self._was_held = False
                self._flush_until = now + RELEASE_FLUSH_S
                return "flush"
            return "flush" if now < self._flush_until else "idle"

    def consume_reset(self) -> bool:
        edge = self._reset_pending
        self._reset_pending = False
        return edge

    def start(self):
        return self

    def stop(self) -> None:
        if self._input_poll is not None:
            self._input_poll.close()


class KeyboardOverride:
    """Keyboard teleop layered over another command source (hil.py: the
    stack's DDS /cmd_vel). While an arrow is held ('live') or within the
    release flush window ('flush') the keyboard wins -- zeros during flush,
    the gazebo release semantics -- and 'idle' lets the base source flow
    again. `state` carries the last active() verdict for status lines."""

    def __init__(self, kb: HoldCmd, base) -> None:
        self.kb = kb
        self.base = base
        self.state = "idle"

    def vector(self) -> np.ndarray:
        self.state = self.kb.active()
        return self.kb.vector() if self.state != "idle" else self.base.vector()

    def consume_reset(self) -> bool:
        return self.base.consume_reset()

    def zero(self) -> None:
        self.base.zero()

    def start(self):
        return self

    def stop(self) -> None:
        self.kb.stop()


class ConstantCmd:
    """Scripted command source for headless runs."""

    def __init__(self, vx: float, wz: float = 0.0,
                 segments: list | None = None, vy: float = 0.0) -> None:
        self.vx, self.vy, self.wz = vx, vy, wz
        # [(duration_s, vx, wz), ...] or [(duration_s, vx, vy, wz), ...]
        self.segments = segments or []
        self._t0 = None

    def vector(self) -> np.ndarray:
        if self.segments:
            if self._t0 is None:
                self._t0 = 0.0
            elapsed = self._t0
            self._t0 += 0.02
            for dur, *cmd in self.segments:
                if elapsed < dur:
                    vx, vy, wz = cmd if len(cmd) == 3 else (cmd[0], 0.0, cmd[1])
                    return np.asarray((vx, vy, wz), np.float32)
                elapsed -= dur
        return np.asarray((self.vx, self.vy, self.wz), dtype=np.float32)

    def consume_reset(self) -> bool:
        return False

    def zero(self) -> None:
        pass  # scripted source: nothing to zero (interface parity with RosCmdSource)

    def start(self):
        return self

    def stop(self) -> None:
        pass
