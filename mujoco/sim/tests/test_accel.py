"""Teleop acceleration limit unit tests: scripted command sources (no keyboard
or pad needed). Run from mujoco/:  python3 -m unittest sim.tests.test_accel -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from sim import keyboard as K  # noqa: E402
from sim.accel import AccelLimit  # noqa: E402
from sim.tests.test_keyboard import BLIND_HIGH, BLIND_LOW, FakePoll  # noqa: E402


class Scripted:
    """Command source returning whatever the test sets."""

    def __init__(self) -> None:
        self.cmd = (0.0, 0.0, 0.0)
        self.reset_edge = False

    def vector(self) -> np.ndarray:
        return np.asarray(self.cmd, dtype=np.float32)

    def consume_reset(self) -> bool:
        edge, self.reset_edge = self.reset_edge, False
        return edge


def ticks(limit: AccelLimit, n: int) -> np.ndarray:
    for _ in range(n):
        out = limit.vector()
    return out


class TestAccelLimit(unittest.TestCase):
    def test_a_step_ramps_at_the_limit_both_ways(self):
        src = Scripted()
        limit = AccelLimit(src)                      # 1.0 m/s², 50 Hz
        src.cmd = (0.75, 0.0, 0.0)
        self.assertAlmostEqual(float(limit.vector()[0]), 0.02, places=6)
        self.assertAlmostEqual(float(ticks(limit, 36)[0]), 0.74, places=5)
        self.assertEqual(float(limit.vector()[0]), 0.75)   # 38 ticks = 0.76 s
        src.cmd = (0.0, 0.0, 0.0)                    # release: braking ramps too
        self.assertAlmostEqual(float(ticks(limit, 37)[0]), 0.01, places=5)
        self.assertEqual(limit.vector().tolist(), [0.0, 0.0, 0.0])

    def test_vx_and_vy_share_the_linear_limit(self):
        src = Scripted()
        limit = AccelLimit(src)
        src.cmd = (1.0, 1.0, 0.0)
        out = limit.vector()
        self.assertAlmostEqual(float(np.hypot(out[0], out[1])), 0.02, places=6)
        self.assertAlmostEqual(float(out[0]), float(out[1]), places=7)

    def test_wz_has_its_own_limit(self):
        src = Scripted()
        limit = AccelLimit(src)                      # 2.0 rad/s²
        src.cmd = (0.01, 0.0, -1.0)
        out = limit.vector()
        self.assertAlmostEqual(float(out[2]), -0.04, places=6)
        self.assertAlmostEqual(float(out[0]), 0.01, places=7)   # within reach
        limit.set_limits(2.0, 4.0)
        out = limit.vector()
        self.assertAlmostEqual(float(out[2]), -0.12, places=6)

    def test_the_pads_b_stops_at_once(self):
        src = Scripted()
        held = [False]
        limit = AccelLimit(src, brake=lambda: held[0])
        src.cmd = (1.0, 0.0, 0.5)
        ticks(limit, 100)
        held[0] = True                               # B held, a key still asks
        src.cmd = (0.5, 0.0, 0.0)
        self.assertAlmostEqual(float(limit.vector()[0]), 0.98, places=5)
        src.cmd = (0.0, 0.0, 0.0)                    # B held, the pad asks zero
        self.assertEqual(limit.vector().tolist(), [0.0, 0.0, 0.0])

    def test_the_base_source_passes_untouched(self):
        src = Scripted()
        limit = AccelLimit(src, engaged=lambda: False)
        src.cmd = (0.8, 0.0, -0.6)                   # hil: the stack's /cmd_vel
        np.testing.assert_allclose(limit.vector(), (0.8, 0.0, -0.6), atol=1e-7)

    def test_catches_up_with_the_base_source_after_teleop(self):
        src = Scripted()
        teleop = [True]
        limit = AccelLimit(src, engaged=lambda: teleop[0])
        src.cmd = (1.0, 0.0, 0.0)
        ticks(limit, 25)                             # 0.5 m/s
        teleop[0] = False                            # handed back to the stack
        src.cmd = (0.2, 0.0, 0.0)
        self.assertAlmostEqual(float(limit.vector()[0]), 0.48, places=5)
        self.assertAlmostEqual(float(ticks(limit, 14)[0]), 0.2, places=5)
        src.cmd = (0.9, 0.0, 0.0)                    # caught up: untouched again
        self.assertAlmostEqual(float(limit.vector()[0]), 0.9, places=6)

    def test_reset_ramps_up_from_zero(self):
        src = Scripted()
        limit = AccelLimit(src)
        src.cmd = (1.0, 0.0, 0.0)
        ticks(limit, 60)
        src.reset_edge = True
        self.assertTrue(limit.consume_reset())
        self.assertAlmostEqual(float(limit.vector()[0]), 0.02, places=6)


class TestWithTheKeyboard(unittest.TestCase):
    def test_a_held_arrow_ramps_and_its_release_brakes(self):
        kb = K.HoldCmd(use_global=False)
        kb._input_poll = poll = FakePoll()
        kb.set_envelope(BLIND_LOW, BLIND_HIGH)       # Up at 50%: 0.75 m/s
        mux = K.KeyboardOverride(kb, K.ConstantCmd(vx=0.0))
        limit = AccelLimit(mux, engaged=lambda: mux.state != "idle")
        poll.held = {265}
        self.assertAlmostEqual(float(ticks(limit, 10)[0]), 0.2, places=5)
        self.assertEqual(float(ticks(limit, 30)[0]), 0.75)
        poll.held = set()                            # flush: zeros, braking
        self.assertAlmostEqual(float(ticks(limit, 10)[0]), 0.55, places=5)
        self.assertEqual(mux.state, "flush")


if __name__ == "__main__":
    unittest.main()
