#!/usr/bin/env python3
"""Gamepad teleop from the kernel input layer (/dev/input, zero deps).

Same capture model as sim/keyboard.py: the pad is read on the SIM host (USB
or Bluetooth, inside the mjsim container through the /dev/input mount),
independent of browser focus and display server. Any evdev device with a
left stick (ABS_X/ABS_Y) and a south button (BTN_SOUTH) counts as a pad;
pads plugged in later are picked up within RESCAN_S.

    left stick     vx (up: forward to the envelope max, down: back) and vy
    right stick X  wz            (ABS_RX; pads without it use ABS_Z)
    A / Cross      toggle gamepad control on/off
    B / Circle     stop: zero command while held
    X / Y          march in place on / off (consume_march; the host keeps the
                   state). Both: pads report the X label as BTN_X (0x133) or
                   BTN_Y (0x134) depending on the driver / mode

Stick deflection maps linearly onto the command envelope (the active
policy's training envelope; GamepadCmd.set_envelope), past a DEADZONE.
GamepadOverride layers the pad over another command source while gamepad
control is on -- centred sticks then mean "stand", not "hand back".
"""

from __future__ import annotations

import errno
import fcntl
import glob
import os
import struct
import threading
import time

import numpy as np

EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03
ABS_X, ABS_Y, ABS_Z, ABS_RX = 0x00, 0x01, 0x02, 0x03
ABS_RY = 0x04
BTN_SOUTH, BTN_EAST = 0x130, 0x131
BTN_NORTH, BTN_WEST = 0x133, 0x134   # input.h BTN_X, BTN_Y
MARCH_BUTTONS = (BTN_NORTH, BTN_WEST)

DEADZONE = 0.12
RESCAN_S = 2.0
# Depth-student training envelope (sim/contract_v2.py COMMAND_LOW/HIGH).
DEFAULT_LOW = (-0.5, -0.5, -1.2)
DEFAULT_HIGH = (1.0, 0.5, 1.2)

# struct input_event on 64-bit Linux: timeval (2 x long), type, code, value.
_EVENT = struct.Struct("llHHi")
_ABSINFO = struct.Struct("6i")    # value, minimum, maximum, fuzz, flat, resolution


def _ioc(dir_bits: int, type_: int, nr: int, size: int) -> int:
    return (dir_bits << 30) | (size << 16) | (type_ << 8) | nr


_EVIOCGBIT_KEY = _ioc(2, 0x45, 0x20 + EV_KEY, 96)
_EVIOCGBIT_ABS = _ioc(2, 0x45, 0x20 + EV_ABS, 8)
_EVIOCGNAME = _ioc(2, 0x45, 0x06, 256)


def _eviocgabs(code: int) -> int:
    return _ioc(2, 0x45, 0x40 + code, _ABSINFO.size)


def _has(bits: bytes, code: int) -> bool:
    return bool(bits[code >> 3] & (1 << (code & 7)))


def deadzone(x: float, dz: float = DEADZONE) -> float:
    """0 inside the dead band, rescaled to reach +-1 at full deflection."""
    if abs(x) < dz:
        return 0.0
    return (x - np.sign(x) * dz) / (1.0 - dz)


class Pad:
    """One evdev pad: axis/button state kept current from its event stream."""

    def __init__(self, fd: int, name: str, absinfo: dict[int, tuple[int, int, int]]) -> None:
        self.fd = fd
        self.name = name
        self._abs = {code: info[0] for code, info in absinfo.items()}
        self._range = {code: (info[1], info[2]) for code, info in absinfo.items()}
        self.buttons: set[int] = set()
        self.right_x = ABS_RX if ABS_RX in absinfo and ABS_RY in absinfo else ABS_Z

    @classmethod
    def open(cls, path: str) -> Pad | None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return None
        try:
            keys, axes = bytearray(96), bytearray(8)
            fcntl.ioctl(fd, _EVIOCGBIT_KEY, keys, True)
            fcntl.ioctl(fd, _EVIOCGBIT_ABS, axes, True)
            if not (_has(keys, BTN_SOUTH) and _has(axes, ABS_X) and _has(axes, ABS_Y)):
                os.close(fd)
                return None
            absinfo = {}
            for code in range(0x40):
                if _has(axes, code):
                    buf = bytearray(_ABSINFO.size)
                    fcntl.ioctl(fd, _eviocgabs(code), buf, True)
                    absinfo[code] = _ABSINFO.unpack(buf)[:3]
            name = bytearray(256)
            fcntl.ioctl(fd, _EVIOCGNAME, name, True)
            return cls(fd, name.split(b"\0", 1)[0].decode(errors="replace"), absinfo)
        except OSError:
            os.close(fd)
            return None

    def feed(self, data: bytes) -> None:
        """Apply raw input_event records (EV_ABS axes, EV_KEY buttons)."""
        for i in range(len(data) // _EVENT.size):
            _, _, type_, code, value = _EVENT.unpack_from(data, i * _EVENT.size)
            if type_ == EV_ABS:
                self._abs[code] = value
            elif type_ == EV_KEY:
                if value:
                    self.buttons.add(code)
                else:
                    self.buttons.discard(code)

    def read(self) -> bool:
        """Drain pending events; False once the device is gone."""
        while True:
            try:
                data = os.read(self.fd, _EVENT.size * 64)
            except BlockingIOError:
                return True
            except OSError as exc:
                return exc.errno not in (errno.ENODEV, errno.EIO, errno.EBADF)
            if not data:
                return True
            self.feed(data)

    def axis(self, code: int) -> float:
        """Axis in [-1, 1] (0 = centre)."""
        lo, hi = self._range.get(code, (0, 0))
        if hi <= lo:
            return 0.0
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        return float(np.clip((self._abs.get(code, mid) - mid) / half, -1.0, 1.0))

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


class GamepadCmd:
    """Command source from the first connected pad (vector / consume_reset
    surface of the other sources). Thread-safe: hosts poll it from the
    control loop and flip `enabled` from GUI callbacks."""

    def __init__(self, enabled: bool = False, scan: bool = True) -> None:
        self._lock = threading.Lock()
        self.enabled = enabled
        self.pad: Pad | None = None
        self._scan = scan
        self._next_scan = 0.0
        self._prev_a = False
        self._prev_march = False
        self._march_pending = False   # X / Y pressed (edge), taken by consume_march
        self.stop_held = False
        self.set_envelope(DEFAULT_LOW, DEFAULT_HIGH)

    def set_envelope(self, low, high) -> None:
        with self._lock:
            self._low = np.asarray(low, dtype=np.float32)
            self._high = np.asarray(high, dtype=np.float32)

    @property
    def name(self) -> str:
        return self.pad.name if self.pad is not None else ""

    def _rescan(self) -> None:
        for path in sorted(glob.glob("/dev/input/event*")):
            pad = Pad.open(path)
            if pad is not None:
                self.pad = pad
                print(f"[PAD] {pad.name} ({path})", flush=True)
                return

    def poll(self) -> None:
        """Read pending pad events (A toggles control); rescan when absent."""
        with self._lock:
            if self.pad is None:
                now = time.monotonic()
                if self._scan and now >= self._next_scan:
                    self._next_scan = now + RESCAN_S
                    self._rescan()
                if self.pad is None:
                    return
            if not self.pad.read():
                print(f"[PAD] {self.pad.name} disconnected", flush=True)
                self.pad.close()
                self.pad = None
                return
            a = BTN_SOUTH in self.pad.buttons
            if a and not self._prev_a:
                self.enabled = not self.enabled
                print(f"[PAD] control {'ON' if self.enabled else 'off'}", flush=True)
            self._prev_a = a
            march = any(b in self.pad.buttons for b in MARCH_BUTTONS)
            if march and not self._prev_march:
                self._march_pending = True
            self._prev_march = march
            self.stop_held = BTN_EAST in self.pad.buttons

    def vector(self) -> np.ndarray:
        self.poll()
        with self._lock:
            pad = self.pad
            if pad is None or self.stop_held:
                return np.zeros(3, dtype=np.float32)
            ly, lx = deadzone(pad.axis(ABS_Y)), deadzone(pad.axis(ABS_X))
            rx = deadzone(pad.axis(pad.right_x))
            # Stick up / left read negative; forward, left and CCW are +.
            vx = -ly * (self._high[0] if ly < 0 else -self._low[0])
            vy = -lx * (self._high[1] if lx < 0 else -self._low[1])
            wz = -rx * (self._high[2] if rx < 0 else -self._low[2])
            return np.asarray((vx, vy, wz), dtype=np.float32)

    @property
    def connected(self) -> bool:
        return self.pad is not None

    def consume_march(self) -> bool:
        """X / Y pressed since the last call (toggle marching in place)."""
        with self._lock:
            edge = self._march_pending
            self._march_pending = False
        return edge

    def consume_reset(self) -> bool:
        return False

    def zero(self) -> None:
        pass

    def start(self):
        return self

    def stop(self) -> None:
        with self._lock:
            if self.pad is not None:
                self.pad.close()
                self.pad = None


class GamepadOverride:
    """Pad layered over another command source (hil: the stack's /cmd_vel,
    view: zeros): while gamepad control is on and a pad is connected the pad
    drives, otherwise the base source does. Wrap the result in
    sim.keyboard.KeyboardOverride so held arrows still win."""

    def __init__(self, pad: GamepadCmd, base) -> None:
        self.pad = pad
        self.base = base

    @property
    def driving(self) -> bool:
        return self.pad.enabled and self.pad.connected

    def vector(self) -> np.ndarray:
        cmd = self.pad.vector()    # also polls: A toggles even while off
        return cmd if self.driving else self.base.vector()

    def consume_reset(self) -> bool:
        return self.base.consume_reset()

    def zero(self) -> None:
        self.base.zero()

    def start(self):
        return self

    def stop(self) -> None:
        self.pad.stop()
        self.base.stop()


__all__ = ["GamepadCmd", "GamepadOverride", "Pad", "deadzone"]
