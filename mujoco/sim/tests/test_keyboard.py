"""Keyboard source unit tests: held-key speeds with and without a policy
command range (no keyboard needed). Run from mujoco/:
  python3 -m unittest sim.tests.test_keyboard -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sim import keyboard as K  # noqa: E402

BLIND_LOW, BLIND_HIGH = (-1.0, -0.6, -1.5), (1.5, 0.6, 1.5)


class TestHoldSpeeds(unittest.TestCase):
    def test_fixed_holds_without_a_command_range(self):
        kb = K.HoldCmd(use_global=False)
        self.assertEqual(kb.hold_speeds(),
                         (K.VX_HOLD, K.VX_HOLD_BWD, K.VY_HOLD, K.WZ_HOLD))

    def test_share_of_the_policy_command_range(self):
        kb = K.HoldCmd(use_global=False)
        kb.set_envelope(BLIND_LOW, BLIND_HIGH)
        for got, want in zip(kb.hold_speeds(), (0.75, 0.5, 0.3, 0.75)):
            self.assertAlmostEqual(got, want, places=5)
        kb.set_speed(1.0)
        for got, want in zip(kb.hold_speeds(), (1.5, 1.0, 0.6, 1.5)):
            self.assertAlmostEqual(got, want, places=5)
        kb.set_speed(0.0)
        self.assertAlmostEqual(kb.speed, 0.05)


class FakePoll:
    """InputPoll stand-in: the codes the test says are held."""

    def __init__(self) -> None:
        self.held: set = set()

    def arrows(self) -> set:
        return set(self.held)

    def close(self) -> None:
        pass


class TestToggles(unittest.TestCase):
    def test_m_toggles_marching_once_per_press_and_moves_nothing(self):
        kb = K.HoldCmd(use_global=False)
        kb._input_poll = poll = FakePoll()
        self.assertFalse(kb.consume_march())
        poll.held = {K.MARCH_KEY}                        # M pressed (global poll)
        self.assertEqual(kb.active(), "idle")            # not a move key
        self.assertEqual(kb.vector().tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(kb.consume_march())
        self.assertFalse(kb.consume_march())             # held: one press
        poll.held = set()
        self.assertFalse(kb.consume_march())
        poll.held = {K.MARCH_KEY, 265}                   # M with Up: both work
        self.assertTrue(kb.consume_march())
        self.assertGreater(float(kb.vector()[0]), 0.0)

    def test_viewer_keys(self):
        kb = K.HoldCmd(use_global=False)
        self.assertTrue(kb.key(K.MARCH_KEY))
        self.assertTrue(kb.consume_march())
        self.assertFalse(kb.consume_stop())
        kb.key(32)                                       # Space / GUI Stop
        self.assertTrue(kb.consume_stop())
        self.assertFalse(kb.consume_stop())


if __name__ == "__main__":
    unittest.main()
