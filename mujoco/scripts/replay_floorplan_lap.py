#!/usr/bin/env python3
"""Replay a recorded /cmd_vel sequence into the in-process policy+physics
(PieRuntime) and render the orthographic BEV with CONTACT MARKERS -- the
same walk as the live stack, with real gait, and every dog/wall contact
drawn as a red dot.  Raw RGB24 on stdout (host ffmpeg -> mp4/webm).

    python3 replay_floorplan_lap.py /tmp/cmdvel.jsonl > /tmp/replay.raw
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim.plant as _plant  # noqa: E402
from sim.plant import build_model, get_scene, reset_to_spawn  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.robots import get_robot  # noqa: E402

_HERE = Path(_plant.__file__).resolve().parents[1]
W, H = 854, 480
PPM = 120.0  # ortho pixels per metre (frustum 7.11 x 4.0)


class PlaybackSource:
    """Replays a (t_ns, vx, vy, wz) log against the wall clock; protocol
    PieRuntime needs: vector() + consume_reset()."""

    def __init__(self, rows):
        self.rows = rows
        self.t0 = None
        self.done = False
        self.state = "replay"
        self._rt = None

    def consume_reset(self) -> bool:
        return False

    def vector(self) -> np.ndarray:
        # playback clock starts when the replay starts, NOT at the recorded
        # wall-clock origin (which is minutes in the past and would make the
        # whole log "already elapsed" on the first call)
        if self.t0 is None:
            self.t0 = time.time_ns()
        el = time.time_ns() - self.t0
        cur = (0.0, 0.0, 0.0)
        for t, vx, vy, wz in self.rows:
            if t - self.rows[0][0] <= el:
                cur = (vx, vy, wz)
            else:
                break
        if el > self.rows[-1][0] - self.rows[0][0] + 2e9:
            self.done = True
        return np.array(cur)

    def _pos(self):
        return self._rt.data.qpos[0:3] if hasattr(self, "_rt") else None

    def _quat(self):
        return self._rt.data.qpos[3:7]


def main() -> None:
    log_path = sys.argv[1]
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                t, vx, vy, wz = json.loads(line)
                rows.append((t, vx, vy, wz))
    span = (rows[-1][0] - rows[0][0]) / 1e9
    print(f"replaying {len(rows)} cmd samples, {span:.1f}s span",
          file=sys.stderr, flush=True)

    model, b = build_model("floorplan")
    data = mujoco.MjData(model)
    reset_to_spawn(model, data, b, settle=True,
                   pos=(-2.5, 0.0, 0.30), yaw_deg=90.0)
    policy = PieOnnxPolicy(get_robot("go2").policy_path("blind"),
                           provider="cpu")
    depth_cam = DepthCamera(model, b.depth_camera_id)
    cmd = PlaybackSource(rows)
    rt = PieRuntime(model, b, data, policy, depth_cam, cmd,
                    spawn_pos=(-2.5, 0.0, 0.30), spawn_yaw_deg=90.0)
    cmd._rt = rt

    rend = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = 4.5
    cam.azimuth = 90.0
    cam.elevation = -90.0

    import cv2
    out = sys.stdout.buffer
    ctrl_dt = 1.0 / 50.0
    next_ctrl = time.perf_counter()
    t_start = time.perf_counter()
    n_frames = 0
    n_contacts = 0
    while not cmd.done and time.perf_counter() - t_start < span + 10.0:
        now = time.perf_counter()
        if now >= next_ctrl:
            rt.tick()
            next_ctrl = max(next_ctrl + ctrl_dt, now)

            rend.update_scene(data, camera=cam)
            for ci in (0, 1):
                g = rend.scene.camera[ci]
                g.orthographic = 1
                g.frustum_bottom, g.frustum_top = -2.0, 2.0
                g.frustum_width = 3.56
                g.frustum_near, g.frustum_far = 0.05, 20.0
            frame = np.ascontiguousarray(rend.render())

            # contact markers: dog-vs-something above knee height, drawn at
            # the contact's world xy (ortho maps linearly: u = W/2 + x*PPM,
            # v = H/2 - y*PPM)
            for i in range(data.ncon):
                c = data.contact[i]
                if c.dist > 0.005 or c.pos[2] < 0.06:
                    continue
                n_contacts += 1
                u = int(W / 2 + c.pos[0] * PPM)
                v = int(H / 2 - c.pos[1] * PPM)
                cv2.circle(frame, (u, v), 7, (60, 60, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (u, v), 9, (255, 255, 255), 2, cv2.LINE_AA)
            out.write(frame.tobytes())
            n_frames += 1
            if n_frames % (25 * 5) == 0:
                print(f"  sim t={data.time:5.1f}s frames={n_frames} "
                      f"contacts={n_contacts}", file=sys.stderr, flush=True)
        wake = next_ctrl - time.perf_counter()
        if wake > 0:
            time.sleep(wake)
    print(f"done: {n_frames} frames, sim t={data.time:.1f}s, "
          f"total contact ticks={n_contacts}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
