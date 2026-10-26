#!/usr/bin/env python3
"""Record an orthographic-BEV video of the LIVE emu walk.

Attaches dog.py's state ring (/dev/shm/mjsim_emu_state.bin: seq, t, pos,
quat_wxyz, gyro, acc -- base pose only), places the floorplan model's base
at the latest ring pose each frame, and renders the fixed orthographic BEV.
Run it while nav_cycle drives the real stack; raw RGB24 goes to stdout:

    python3 record_live_bev.py --secs 150 > /tmp/lap_live.raw
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim.plant as _plant  # noqa: E402
from sim.plant import build_model  # noqa: E402

_HERE = Path(_plant.__file__).resolve().parents[1]
W, H, FPS = 854, 480, 30.0
_REC = struct.Struct("<Q d 3d 4d 3d 3d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=150.0)
    ap.add_argument("--shm", default="/dev/shm/mjsim_emu_state.bin")
    args = ap.parse_args()

    fd = os.open(args.shm, os.O_RDONLY)
    mm = mmap.mmap(fd, _REC.size, prot=mmap.PROT_READ)

    model, b = build_model("floorplan")
    data = mujoco.MjData(model)

    rend = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.0]
    cam.distance = 4.5
    cam.azimuth = 90.0
    cam.elevation = -90.0

    out = sys.stdout.buffer
    dt = 1.0 / FPS
    t_end = time.perf_counter() + args.secs
    n = 0
    stale = 0
    print(f"recording {args.secs:.0f}s ...", file=sys.stderr, flush=True)
    while time.perf_counter() < t_end:
        t_frame = time.perf_counter()
        raw = mm[:]
        vals = _REC.unpack(raw)
        if vals[0] == 0:
            stale += 1
        else:
            data.qpos[0:3] = vals[2:5]
            data.qpos[3:7] = vals[5:9]
            mujoco.mj_forward(model, data)
        rend.update_scene(data, camera=cam)
        for ci in (0, 1):
            g = rend.scene.camera[ci]
            g.orthographic = 1
            g.frustum_bottom, g.frustum_top = -2.0, 2.0
            g.frustum_width = 3.56
            g.frustum_near, g.frustum_far = 0.05, 20.0
        out.write(np.ascontiguousarray(rend.render()).tobytes())
        n += 1
        if n % (FPS * 5) == 0:
            print(f"  {n} frames, ring seq={vals[0]}", file=sys.stderr,
                  flush=True)
        wake = t_frame + dt - time.perf_counter()
        if wake > 0:
            time.sleep(wake)
    print(f"done: {n} frames ({stale} empty)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
