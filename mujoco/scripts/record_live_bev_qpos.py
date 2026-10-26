#!/usr/bin/env python3
"""Record the LIVE emu walk as an orthographic BEV with real gait.

dog.py now writes every joint to /dev/shm/mjsim_emu_qpos.bin each control
tick; this script maps that file, and instead of replaying recorded
commands open-loop (which drifts -- the commands were closed-loop answers
to another trajectory) it shows the live state itself.  The current goal
(/control/target_pose + /control/target_heading) is drawn as a ring with
a heading arrow, and the DWA-selected trajectory (/planning/trajectory_path,
planning_node's argmin lattice row, baselink positions in the world frame)
as a magenta polyline -- so a stall can be read off directly: the dog sits
still while the line points where the planner wants it to go.

    python3 record_live_bev_qpos.py --secs 150 > /tmp/live.raw
"""
from __future__ import annotations

import argparse
import math
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
QPOS_PATH = "/dev/shm/mjsim_emu_qpos.bin"
_QHDR = struct.Struct("<QdI")
W, H, FPS = 854, 480, 30.0
PPM = 120.0  # ortho: 4.0 m vertical over 480 px
GOAL_BGR = (40, 220, 255)   # yellow
TRAJ_BGR = (255, 80, 255)   # magenta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=150.0)
    ap.add_argument("--qpos", default=QPOS_PATH)
    ap.add_argument("--traj-topic", default="/planning/trajectory_path")
    ap.add_argument("--traj-ttl", type=float, default=1.0,
                    help="hide the DWA line if no Path arrived for this long")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry, Path as PathMsg
    from std_msgs.msg import Float64
    import cv2

    rclpy.init()
    node = Node("bev_recorder")
    goal = {"xy": None, "yaw": None, "t": 0.0}
    traj = {"xy": [], "t": 0.0}

    def on_pose(m):
        goal["xy"] = (float(m.pose.pose.position.x),
                      float(m.pose.pose.position.y))
        goal["t"] = time.monotonic()

    def on_yaw(m):
        goal["yaw"] = float(m.data)
        goal["t"] = time.monotonic()

    def on_traj(m):
        pts = [(float(p.pose.position.x), float(p.pose.position.y),
                float(p.pose.orientation.x), float(p.pose.orientation.y),
                float(p.pose.orientation.z), float(p.pose.orientation.w))
               for p in m.poses]
        if len(pts) >= 2:
            traj["xy"] = pts
            traj["t"] = time.monotonic()

    node.create_subscription(Odometry, "/control/target_pose", on_pose, 10)
    node.create_subscription(Float64, "/control/target_heading", on_yaw, 10)
    node.create_subscription(PathMsg, args.traj_topic, on_traj, 10)

    model, b = build_model("floorplan")
    data = mujoco.MjData(model)

    size = _QHDR.size + 8 * model.nq
    fd = os.open(args.qpos, os.O_RDONLY)
    mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)

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
    empty = 0
    print(f"recording {args.secs:.0f}s (live qpos) ...", file=sys.stderr,
          flush=True)
    while time.perf_counter() < t_end:
        t_frame = time.perf_counter()
        rclpy.spin_once(node, timeout_sec=0.0)
        seq, t_emu, nq = _QHDR.unpack(mm[: _QHDR.size])
        if nq != model.nq:
            nq = min(nq, model.nq)
        qpos = np.frombuffer(mm[_QHDR.size: _QHDR.size + 8 * nq],
                             dtype=np.float64).copy()
        if seq == 0:
            empty += 1
        else:
            data.qpos[:nq] = qpos
            mujoco.mj_forward(model, data)

        rend.update_scene(data, camera=cam)
        for ci in (0, 1):
            g = rend.scene.camera[ci]
            g.orthographic = 1
            g.frustum_bottom, g.frustum_top = -2.0, 2.0
            g.frustum_width = 3.56
            g.frustum_near, g.frustum_far = 0.05, 20.0
        frame = np.ascontiguousarray(rend.render())

        if traj["xy"] and time.monotonic() - traj["t"] < args.traj_ttl:
            px = np.array([[W / 2 + p[0] * PPM, H / 2 - p[1] * PPM]
                           for p in traj["xy"]], dtype=np.int32)
            cv2.polylines(frame, [px], False, TRAJ_BGR, 2, cv2.LINE_AA)
            for u, v in px:
                cv2.circle(frame, (int(u), int(v)), 2, TRAJ_BGR, -1,
                           cv2.LINE_AA)
            # End-of-trajectory heading. Path poses carry the body +Z-forward
            # convention (planning_node's _world_heading), not the ROS x-forward
            # one, so the yaw comes from rotating +Z by the quaternion.
            _, _, qx, qy, qz, qw = traj["xy"][-1]
            hx = 2.0 * (qx * qz + qw * qy)
            hy = 2.0 * (qy * qz - qw * qx)
            yaw = math.atan2(hy, hx)
            u, v = int(px[-1][0]), int(px[-1][1])
            cv2.arrowedLine(frame, (u, v),
                            (u + int(22 * math.cos(yaw)),
                             v - int(22 * math.sin(yaw))),
                            TRAJ_BGR, 2, cv2.LINE_AA, tipLength=0.4)

        if goal["xy"] is not None and time.monotonic() - goal["t"] < 2.5:
            gx, gy = goal["xy"]
            u = int(W / 2 + gx * PPM)
            v = int(H / 2 - gy * PPM)
            cv2.circle(frame, (u, v), 11, GOAL_BGR, 2, cv2.LINE_AA)
            cv2.drawMarker(frame, (u, v), GOAL_BGR,
                           cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            if goal["yaw"] is not None:
                L = 34
                du = int(L * math.cos(goal["yaw"]))
                dv = int(-L * math.sin(goal["yaw"]))
                cv2.arrowedLine(frame, (u, v), (u + du, v + dv),
                                GOAL_BGR, 3, cv2.LINE_AA,
                                tipLength=0.35)

        cv2.putText(frame, "goal", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    GOAL_BGR, 1, cv2.LINE_AA)
        cv2.putText(frame, "dwa", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    TRAJ_BGR, 1, cv2.LINE_AA)
        out.write(frame.tobytes())
        n += 1
        if n % (FPS * 5) == 0:
            print(f"  {n} frames, emu t={t_emu:.1f} seq={seq}",
                  file=sys.stderr, flush=True)
        wake = t_frame + dt - time.perf_counter()
        if wake > 0:
            time.sleep(wake)
    print(f"done: {n} frames ({empty} empty reads)", file=sys.stderr,
          flush=True)


if __name__ == "__main__":
    main()
