#!/usr/bin/env python3
"""Render a chase-cam video of the go2 walking the floorplan lap with the
real policy (no DDS, no viewer): the PieRuntime control loop drives the
physics exactly like the emu does, a pure-pursuit source feeds the four
platform corners WN->EN->ES->WS->WN, and every control tick one chase-cam
frame goes to stdout as raw RGB24 -- host-side ffmpeg turns that into the
mp4:

    docker exec mjsim-dog bash -c 'source /opt/ros/jazzy/setup.bash && \
        python3 /tmp/lap_video.py' > /tmp/lap.raw
    ffmpeg -f rawvideo -pix_fmt rgb24 -s 854x480 -r 50 -i /tmp/lap.raw \
        -c:v libx264 -crf 26 -preset fast lap_video.mp4
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim.plant import build_model, get_scene, reset_to_spawn  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402
from sim.policy import load_policy  # noqa: E402
import sim.plant as _plant  # noqa: E402

_HERE = Path(_plant.__file__).resolve().parents[1]  # the mujoco/ dir

W, H, FPS = 854, 480, 50
VX, WZ_GAIN, WZ_MAX, ARRIVE = 0.35, 1.8, 0.7, 0.30
TIME_LIMIT = 90.0


class PursuitSource:
    """Pure pursuit around the four platform corners; the protocol
    PieRuntime needs is just vector() + consume_reset()."""

    def __init__(self, corners):
        self.corners = list(corners)
        self.i = 0
        self.done = False
        self.state = "pp"

    def consume_reset(self) -> bool:
        return False

    def vector(self) -> np.ndarray:
        p = self._pos()
        if p is None:
            return np.zeros(3)
        tx, ty = self.corners[self.i]
        if math.hypot(p[0] - tx, p[1] - ty) < ARRIVE:
            self.i += 1
            if self.i >= len(self.corners):
                self.done = True
        if self.done:
            return np.zeros(3)
        q = self._quat()
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                         1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        err = math.atan2(ty - p[1], tx - p[0]) - yaw
        err = math.atan2(math.sin(err), math.cos(err))
        wz = max(-WZ_MAX, min(WZ_MAX, WZ_GAIN * err))
        slow = max(0.25, 1.0 - abs(err) / 1.2)
        return np.array([VX * slow, 0.0, wz])

    def _pos(self):
        return self._rt.data.qpos[0:3] if hasattr(self, "_rt") else None

    def _quat(self):
        return self._rt.data.qpos[3:7]


def main() -> None:
    model, b = build_model("floorplan")
    data = mujoco.MjData(model)
    wps = [(-2.5, 0.6), (2.5, 0.6), (2.5, -0.6), (-2.5, -0.6), (-2.5, 0.6)]
    spawn = (-2.5, 0.0, 0.30)
    reset_to_spawn(model, data, b, settle=True, pos=spawn, yaw_deg=90.0)

    policy = load_policy(
        _HERE / "assets" / "policy" / get_scene("floorplan").hil_policy,
        provider="cpu")
    depth_cam = DepthCamera(model, b.depth_camera_id)
    cmd = PursuitSource(wps)
    rt = PieRuntime(model, b, data, policy, depth_cam, cmd,
                    spawn_pos=spawn, spawn_yaw_deg=90.0)
    cmd._rt = rt

    rend = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    out = sys.stdout.buffer
    ctrl_dt = 1.0 / 50.0
    next_ctrl = time.perf_counter()
    t_start = time.perf_counter()
    n_frames = 0
    print(f"walking lap, {len(wps)} corners, limit {TIME_LIMIT}s",
          file=sys.stderr, flush=True)
    while not cmd.done and time.perf_counter() - t_start < TIME_LIMIT:
        now = time.perf_counter()
        if now >= next_ctrl:
            rt.tick()
            next_ctrl = max(next_ctrl + ctrl_dt, now)

            # orthographic-equivalent BEV: long lens (65 m out, 3.2 deg)
            # kills the perspective -- wall tops land exactly on their
            # footprints and the dog silhouette is true to its xy pose
            pos = np.array(data.qpos[0:3], float)
            cam.lookat[:] = [0.0, 0.0, 0.0]
            cam.distance = 4.5
            cam.azimuth = 90.0
            cam.elevation = -90.0
            r = rend
            r.update_scene(data, camera=cam)
            # true orthographic BEV: MjvCamera.orthographic is not honored
            # by the render path -- force it on the GL cameras and set the
            # planar frustum extents (world metres) explicitly
            for ci in (0, 1):
                g = r.scene.camera[ci]
                g.orthographic = 1
                # frustum_width is a HALF-extent like frustum_top/bottom:
                # 2.0 vertical + 16:9 aspect -> half-width 3.56 (7.11 m span)
                g.frustum_bottom, g.frustum_top = -2.0, 2.0
                g.frustum_width = 3.56
                g.frustum_near, g.frustum_far = 0.05, 20.0
            out.write(np.ascontiguousarray(r.render()).tobytes())
            n_frames += 1
            if n_frames % (FPS * 5) == 0:
                print(f"  t={data.time:5.1f}s corner {cmd.i + 1}/{len(wps)} "
                      f"frames={n_frames}", file=sys.stderr, flush=True)
        wake = next_ctrl - time.perf_counter()
        if wake > 0:
            time.sleep(wake)
    print(f"done: {n_frames} frames, sim t={data.time:.1f}s, "
          f"corner {cmd.i}/{len(wps)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
