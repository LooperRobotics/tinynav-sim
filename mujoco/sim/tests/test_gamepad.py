"""Gamepad source unit tests: synthetic input_event records through a pipe
(no device needed). Run from mujoco/:  python3 -m unittest sim.tests.test_gamepad -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from sim import gamepad as G  # noqa: E402
from sim.keyboard import ConstantCmd  # noqa: E402

# A typical pad: sticks 0..255 centred at 128 (ABS_RX present).
ABSINFO = {code: (128, 0, 255) for code in (G.ABS_X, G.ABS_Y, G.ABS_RX, G.ABS_RY)}


def event(type_: int, code: int, value: int) -> bytes:
    return G._EVENT.pack(0, 0, type_, code, value)


class PipePad:
    """A Pad reading from a pipe the test writes input_events into."""

    def __init__(self, absinfo=ABSINFO) -> None:
        r, self.w = os.pipe()
        os.set_blocking(r, False)
        self.pad = G.Pad(r, "test pad", absinfo)

    def send(self, *events: tuple[int, int, int]) -> None:
        os.write(self.w, b"".join(event(*e) for e in events)
                 + event(G.EV_SYN, 0, 0))

    def close(self) -> None:
        os.close(self.w)
        self.pad.close()


def source(pipe: PipePad, enabled: bool = True) -> G.GamepadCmd:
    cmd = G.GamepadCmd(enabled=enabled, scan=False)
    cmd.pad = pipe.pad
    return cmd


class TestGamepad(unittest.TestCase):
    def setUp(self) -> None:
        self.pipe = PipePad()

    def tearDown(self) -> None:
        self.pipe.close()

    def test_centred_sticks_are_zero(self):
        np.testing.assert_allclose(source(self.pipe).vector(), 0.0)

    def test_full_deflection_hits_the_envelope(self):
        cmd = source(self.pipe)
        self.pipe.send((G.EV_ABS, G.ABS_Y, 0))           # left stick up
        np.testing.assert_allclose(cmd.vector(), (1.0, 0.0, 0.0), atol=1e-6)
        self.pipe.send((G.EV_ABS, G.ABS_Y, 255))         # down: back, capped
        self.assertAlmostEqual(float(cmd.vector()[0]), -0.5, places=2)
        self.pipe.send((G.EV_ABS, G.ABS_Y, 128), (G.EV_ABS, G.ABS_X, 0),
                       (G.EV_ABS, G.ABS_RX, 255))        # strafe left, turn right
        vx, vy, wz = cmd.vector()
        self.assertAlmostEqual(float(vx), 0.0)
        self.assertAlmostEqual(float(vy), 0.5, places=6)
        self.assertAlmostEqual(float(wz), -1.2, places=2)

    def test_deadzone(self):
        cmd = source(self.pipe)
        self.pipe.send((G.EV_ABS, G.ABS_Y, 128 - 10))   # ~8% deflection
        np.testing.assert_allclose(cmd.vector(), 0.0)

    def test_envelope_follows_the_policy(self):
        cmd = source(self.pipe)
        cmd.set_envelope((-0.2, 0.0, -0.3), (0.5, 0.0, 0.3))
        self.pipe.send((G.EV_ABS, G.ABS_Y, 0), (G.EV_ABS, G.ABS_X, 0))
        np.testing.assert_allclose(cmd.vector(), (0.5, 0.0, 0.0), atol=1e-6)

    def test_a_toggles_and_b_stops(self):
        cmd = source(self.pipe, enabled=False)
        self.pipe.send((G.EV_KEY, G.BTN_SOUTH, 1))
        cmd.poll()
        self.assertTrue(cmd.enabled)
        self.pipe.send((G.EV_KEY, G.BTN_SOUTH, 0))
        cmd.poll()
        self.assertTrue(cmd.enabled)                     # release: no toggle
        self.pipe.send((G.EV_ABS, G.ABS_Y, 0), (G.EV_KEY, G.BTN_EAST, 1))
        np.testing.assert_allclose(cmd.vector(), 0.0)    # B held: stop
        self.pipe.send((G.EV_KEY, G.BTN_EAST, 0))
        self.assertGreater(float(cmd.vector()[0]), 0.9)

    def test_x_or_y_toggles_marching_once_per_press(self):
        cmd = source(self.pipe, enabled=False)           # works with control off
        self.assertFalse(cmd.consume_march())
        self.pipe.send((G.EV_KEY, G.BTN_NORTH, 1))
        cmd.poll()
        self.assertTrue(cmd.consume_march())
        self.assertFalse(cmd.consume_march())            # taken
        cmd.poll()                                       # still held: no new press
        self.assertFalse(cmd.consume_march())
        self.pipe.send((G.EV_KEY, G.BTN_NORTH, 0))
        cmd.poll()
        self.assertFalse(cmd.consume_march())            # release
        self.pipe.send((G.EV_KEY, G.BTN_WEST, 1))         # Y on other mappings
        cmd.poll()
        self.assertTrue(cmd.consume_march())
        self.assertFalse(cmd.enabled)                    # A's toggle untouched

    def test_right_stick_falls_back_to_abs_z(self):
        pipe = PipePad({code: (128, 0, 255) for code in (G.ABS_X, G.ABS_Y, G.ABS_Z)})
        try:
            cmd = source(pipe)
            pipe.send((G.EV_ABS, G.ABS_Z, 0))
            self.assertAlmostEqual(float(cmd.vector()[2]), 1.2, places=6)
        finally:
            pipe.close()

    def test_override_hands_back_when_off(self):
        cmd = source(self.pipe)
        mux = G.GamepadOverride(cmd, ConstantCmd(vx=0.3))
        self.pipe.send((G.EV_ABS, G.ABS_Y, 0))
        self.assertAlmostEqual(float(mux.vector()[0]), 1.0, places=6)
        cmd.enabled = False
        self.assertAlmostEqual(float(mux.vector()[0]), 0.3, places=6)

    def test_unplug_drops_the_pad(self):
        cmd = source(self.pipe)
        os.close(self.pipe.pad.fd)                       # EBADF on read
        self.pipe.pad.fd = os.open(os.devnull, os.O_WRONLY)   # read -> EBADF
        cmd.poll()
        self.assertFalse(cmd.connected)


if __name__ == "__main__":
    unittest.main()
