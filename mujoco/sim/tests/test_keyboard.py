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


if __name__ == "__main__":
    unittest.main()
