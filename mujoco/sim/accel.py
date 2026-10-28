#!/usr/bin/env python3
"""Maximum acceleration for the teleop command (keyboard and gamepad).

A held arrow or a pushed stick asks for its speed at once; AccelLimit hands
the policy a command that moves towards it by at most `linear` (m/s², vx and
vy together as one planar vector) and `angular` (rad/s², wz) per 50 Hz
control tick -- speeding up and braking alike. The pad's B is the exception:
while it is held and the command asked for is zero (the pad drives), the
command is zero at once.

It wraps a host's teleop chain (KeyboardOverride over GamepadOverride over
the base source). `engaged()` says whether the keys or the pad drive; while
they do, and afterwards until the command has caught up with the base
source, it ramps; otherwise the base source passes untouched -- hil's
/cmd_vel from the stack is not limited. Without `engaged` every command
ramps (view.py: the base source is zeros). The web panel's Max accel / Max
yaw accel sliders set the two limits (webpanel.py).
"""

from __future__ import annotations

import numpy as np

LINEAR = 1.0      # m/s²: a held arrow at 50% of go2-blind's range (0.75 m/s) in 0.75 s
ANGULAR = 2.0     # rad/s²: its turn (0.75 rad/s) in 0.4 s
CONTROL_DT = 0.02   # s: one vector() call per control tick


class AccelLimit:
    """Command source: `source`'s command, rate limited. engaged() and brake()
    are read after each source.vector() call (the overrides update who drives
    in there); brake() = the pad's B held."""

    def __init__(self, source, engaged=None, brake=None, linear: float = LINEAR,
                 angular: float = ANGULAR, dt: float = CONTROL_DT) -> None:
        self.source = source
        self._engaged = engaged
        self._brake = brake
        self.linear, self.angular = float(linear), float(angular)
        self.dt = float(dt)
        self.command = np.zeros(3, dtype=np.float32)   # last one handed on
        self._ramping = False    # teleop drove and the base source is not caught up

    def set_limits(self, linear: float, angular: float) -> None:
        """m/s² and rad/s² (the panel's sliders, from viser's thread)."""
        self.linear, self.angular = float(linear), float(angular)

    def vector(self) -> np.ndarray:
        target = np.asarray(self.source.vector(), dtype=np.float32).reshape(3)
        teleop = self._engaged is None or self._engaged()
        if teleop:
            self._ramping = True
        if not self._ramping or (self._brake is not None and self._brake()
                                 and not target.any()):
            self.command = target.copy()
        else:
            self.command = self._towards(target)
            if not teleop and np.array_equal(self.command, target):
                self._ramping = False
        return self.command.copy()

    def _towards(self, target: np.ndarray) -> np.ndarray:
        """One tick from self.command towards target (exactly onto it once
        within reach, float32 rounding included)."""
        step = target - self.command
        out = target.copy()
        reach = self.linear * self.dt
        planar = float(np.hypot(step[0], step[1]))
        if planar > reach + 1e-6:
            out[:2] = self.command[:2] + step[:2] * (reach / planar)
        turn = self.angular * self.dt
        if abs(float(step[2])) > turn + 1e-6:
            out[2] = self.command[2] + np.copysign(turn, step[2])
        return out

    def reset(self) -> None:
        """The robot stands again (Reset): a command still asked for ramps up
        from zero."""
        self.command = np.zeros(3, dtype=np.float32)
        self._ramping = False

    def consume_reset(self) -> bool:
        edge = self.source.consume_reset()
        if edge:
            self.reset()
        return edge

    def zero(self) -> None:
        self.source.zero()

    def start(self):
        return self

    def stop(self) -> None:
        self.source.stop()


__all__ = ["ANGULAR", "LINEAR", "AccelLimit"]
