"""Wall guard unit tests: navcore frames ray cast against boxes and planes in
numpy (no renderer needed). Run from mujoco/:
  python3 -m unittest sim.tests.test_guard -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from sim import contract_v2 as V  # noqa: E402
from sim import guard as G  # noqa: E402

SPEC = V.DepthSpec("navcore", 68, 80, 38.76875, True, 3.0,
                   min_range_m=V.NAVCORE_MIN_RANGE_M)
CAMERA_HEIGHT = 0.43    # Go2's lens above its feet, standing


def camera_xmat(pitch_deg: float = 0.0) -> np.ndarray:
    """Camera looking along world +x (nose up = positive pitch), MuJoCo axes:
    columns x right, y up, z backwards."""
    p = math.radians(pitch_deg)
    ahead = np.array((math.cos(p), 0.0, math.sin(p)))
    up = np.array((-math.sin(p), 0.0, math.cos(p)))
    right = np.cross(ahead, up)
    return np.stack((right, up, -ahead), axis=1)


def frame(boxes, pitch_deg: float = 0.0) -> np.ndarray:
    """The policy frame (metres, misses = max range) of a camera at the
    origin over a floor CAMERA_HEIGHT below, among axis-aligned boxes
    ((x0, y0, z0), (x1, y1, z1))."""
    xmat = camera_xmat(pitch_deg)
    u = (np.arange(SPEC.width) - 0.5 * (SPEC.width - 1)) / SPEC.focal_px
    v = (np.arange(SPEC.height) - 0.5 * (SPEC.height - 1)) / SPEC.focal_px
    x, y = np.meshgrid(u, v)
    optical = np.stack((x, y, np.ones_like(x)), axis=-1)        # z = 1: t is optical Z
    rays = (optical * (1.0, -1.0, -1.0)) @ xmat.T
    t = np.full(x.shape, np.inf)
    down = rays[..., 2] < 0.0
    t[down] = -CAMERA_HEIGHT / rays[..., 2][down]
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (np.asarray(lo) - 0.0) / rays
            b = (np.asarray(hi) - 0.0) / rays
        near = np.nanmax(np.minimum(a, b), axis=-1)
        far = np.nanmin(np.maximum(a, b), axis=-1)
        hit = (near <= far) & (far > 0.0)
        t = np.where(hit & (near < t), np.maximum(near, 0.0), t)
    z = np.where(np.isfinite(t), t, 0.0).astype(np.float32)
    return SPEC.preprocess(z)[0] * SPEC.max_m


def wall(d: float, width: float = 4.0):
    return ((d, -0.5 * width, -CAMERA_HEIGHT), (d + 0.1, 0.5 * width, 2.0))


def stairs(first: float, rise: float, tread: float, n: int = 8):
    return [((first + i * tread, -0.5, -CAMERA_HEIGHT), (first + 4.0, 0.5, -CAMERA_HEIGHT + (i + 1) * rise))
            for i in range(n)]


def distance(boxes, pitch_deg: float = 0.0):
    return G.wall_distance(frame(boxes, pitch_deg), SPEC, np.zeros(3), camera_xmat(pitch_deg),
                           (1.0, 0.0))


class TestWallDistance(unittest.TestCase):
    def test_a_wall_ahead(self):
        for d in (0.2, 0.45, 0.8, 1.3):
            self.assertAlmostEqual(distance([wall(d)]), d, delta=0.015)

    def test_a_wall_seen_while_pitched(self):
        self.assertAlmostEqual(distance([wall(0.5)], pitch_deg=25.0), 0.5, delta=0.015)
        self.assertAlmostEqual(distance([wall(0.5)], pitch_deg=-20.0), 0.5, delta=0.015)

    def test_floor_and_stairs_are_no_wall(self):
        self.assertIsNone(distance([]))
        # map3's steepest (19 cm / 22 cm) and a shallow flight, from the landing
        # and with the body pitched up the way it climbs
        for rise, tread in ((0.19, 0.22), (0.14, 0.30)):
            for first in (0.1, 0.3, 0.6):
                for pitch in (0.0, 30.0):
                    self.assertIsNone(distance(stairs(first, rise, tread), pitch),
                                      (rise, tread, first, pitch))

    def test_side_walls_of_a_flight_are_no_wall(self):
        sides = [((-1.0, 0.40, -CAMERA_HEIGHT), (3.0, 0.50, 2.0)),
                 ((-1.0, -0.50, -CAMERA_HEIGHT), (3.0, -0.40, 2.0))]
        self.assertIsNone(distance(sides))
        self.assertAlmostEqual(distance(sides + [wall(0.7)]), 0.7, delta=0.015)

    def test_a_wall_under_the_minimum_range_is_unseen(self):
        self.assertIsNone(distance([wall(0.12)]))


def fake_runtime(frame_m, distance_walked=0.0):
    """What WallGuard reads of a PieRuntime: base at x = distance_walked facing
    +x, the camera over the base origin."""
    qpos = np.zeros(7)
    qpos[0] = distance_walked
    qpos[3] = 1.0
    data = SimpleNamespace(qpos=qpos, cam_xpos=np.zeros((1, 3)),
                           cam_xmat=camera_xmat().reshape(1, 9))
    return SimpleNamespace(depth_spec=SPEC, uses_depth=True, data=data,
                           b=SimpleNamespace(root_qpos_adr=0, depth_camera_id=0),
                           latest_raw_depth=object(),
                           depth_frame=lambda: frame_m / SPEC.max_m)


class Scripted:
    def __init__(self, cmd) -> None:
        self.cmd = cmd

    def vector(self) -> np.ndarray:
        return np.asarray(self.cmd, dtype=np.float32)


class TestWallGuard(unittest.TestCase):
    def test_slows_down_and_stops_in_front_of_a_wall(self):
        rt = fake_runtime(frame([wall(0.8)]))
        src = Scripted((1.0, 0.2, 0.5))
        guard = G.WallGuard(src, lambda: rt)
        out = guard.vector()
        self.assertAlmostEqual(float(out[0]), G.GAIN * (0.8 - G.STOP_M), delta=0.03)
        np.testing.assert_allclose(out[1:], (0.2, 0.5))          # strafe and turn untouched
        self.assertTrue(guard.limiting)
        rt.data.qpos[0] = 0.8 - (G.STOP_M + 0.1)                 # walked on, same frame
        self.assertAlmostEqual(float(guard.vector()[0]), G.GAIN * 0.1, delta=0.03)
        rt.data.qpos[0] = 0.8 - G.STOP_M + 0.05                  # past the stop
        self.assertEqual(float(guard.vector()[0]), 0.0)
        src.cmd = (-0.3, 0.0, 0.0)                               # backing off is free
        self.assertAlmostEqual(float(guard.vector()[0]), -0.3)
        self.assertFalse(guard.limiting)

    def test_off_when_disabled_not_engaged_or_blind(self):
        rt = fake_runtime(frame([wall(0.4)]))
        src = Scripted((1.0, 0.0, 0.0))
        guard = G.WallGuard(src, lambda: rt, engaged=lambda: False)
        self.assertEqual(float(guard.vector()[0]), 1.0)          # hil: the stack drives
        guard = G.WallGuard(src, lambda: rt)
        guard.enabled = False
        self.assertEqual(float(guard.vector()[0]), 1.0)
        rt.uses_depth = False                                    # the blind policy
        guard = G.WallGuard(src, lambda: rt)
        self.assertEqual(float(guard.vector()[0]), 1.0)
        self.assertIsNone(guard.distance)


if __name__ == "__main__":
    unittest.main()
