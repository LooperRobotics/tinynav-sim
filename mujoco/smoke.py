#!/usr/bin/env python3
"""Headless smoke ladder for the PIE driver.

Modes (all headless, realtime pacing unless --no-realtime):
  flat   stand 1s + walk vx=0.5 for 10s on a ground plane (ladder step 2)
  map3   stand 10s at the F1 spawn, then walk toward the L0 stair base;
         also reports per-tick physics/policy timing (the 200 Hz probe)
  down   spawn on landing_F4_east facing down the R2b flight (8 treads,
         159 mm rise each) and descend with vx=--vx
  floorplan  stand then walk on the parametric flat ring (planning scene)

--robot picks the body (sim/robots.py) and its registry perception policy;
--onnx overrides the policy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sim import contract as C  # noqa: E402
from sim.keyboard import ConstantCmd  # noqa: E402
from sim.plant import (  # noqa: E402
    DOWN_SPAWN,
    DOWN_YAW_DEG,
    build_flat_model,
    build_map3_model,
    build_model,
    reset_to_spawn,
)
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.robots import ROBOTS, get_robot  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402

CONTROL_DT = C.PHYSICS_DT * C.PHYSICS_STEPS_PER_CONTROL

def run(mode: str, stand_s: float, walk_s: float, vx: float, wz: float,
        realtime: bool, onnx_path: str | None = None, vy: float = 0.0,
        robot: str = "go2") -> int:
    if mode == "flat":
        model, b = build_flat_model(robot)
    elif mode == "floorplan":
        model, b = build_model("floorplan", robot)
    else:
        model, b = build_map3_model(robot)
    data = mujoco_data(model)
    if mode == "down":
        reset_to_spawn(model, data, b, settle=True, pos=DOWN_SPAWN,
                       yaw_deg=DOWN_YAW_DEG)
    else:
        reset_to_spawn(model, data, b, settle=True)
    print(f"[{mode}] spawn base={np.round(data.qpos[b.root_qpos_adr:][:3], 3)} "
          f"z-ground unknown (settled)")

    if onnx_path is None:
        onnx_path = get_robot(robot).policy_path("perception")
    policy = PieOnnxPolicy(Path(onnx_path), provider="cpu")
    print(f"[{mode}] robot={robot} policy={Path(onnx_path).name} "
          f"contract={policy.contract.name}")
    cam = DepthCamera(model, b.depth_camera_id)
    cmd = ConstantCmd(vx=0.0, wz=0.0)
    rt = PieRuntime(model, b, data, policy, cam, cmd)

    phases = [("stand", stand_s, (0.0, 0.0, 0.0))]
    if walk_s > 0:
        phases.append(("walk", walk_s, (vx, vy, wz)))
    t_all0 = time.perf_counter()
    for phase, dur, (cvx, cvy, cwz) in phases:
        cmd.vx, cmd.vy, cmd.wz = cvx, cvy, cwz
        p0 = rt.base_pos().copy()
        n = int(dur / CONTROL_DT)
        for _ in range(n):
            t0 = time.perf_counter()
            rt.tick()
            if realtime:
                wait = CONTROL_DT - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)
        p1 = rt.base_pos()
        d = p1 - p0
        speed = np.linalg.norm(d[:2]) / dur
        print(f"[{mode}] {phase} {dur:.0f}s: disp=({d[0]:+.2f},{d[1]:+.2f},"
              f"{d[2]:+.2f}) speed={speed:.3f} m/s upright="
              f"{rt.base_uprightness():.2f} fallen={rt.is_fallen()} "
              f"physics={rt.physics_ms:.2f}ms policy={rt.policy_ms:.2f}ms/tick")
        if mode == "map3":
            print(f"[{mode}]   base_xyz={np.round(p1, 3)}")
    cam.close()
    wall = time.perf_counter() - t_all0
    sim = float(data.time)
    print(f"[{mode}] done: sim {sim:.1f}s in {wall:.1f}s wall "
          f"({sim / wall:.2f}x realtime)")
    return 0


def mujoco_data(model):
    import mujoco
    return mujoco.MjData(model)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("flat", "map3", "down", "floorplan"))
    ap.add_argument("--stand", type=float, default=1.0)
    ap.add_argument("--walk", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--robot", default="go2", choices=sorted(ROBOTS))
    ap.add_argument("--onnx", default=None,
                    help="policy file override (default: the robot's "
                         "registry perception policy)")
    args = ap.parse_args()
    return run(args.mode, args.stand, args.walk, args.vx, args.wz,
               realtime=not args.no_realtime, onnx_path=args.onnx, vy=args.vy,
               robot=args.robot)


if __name__ == "__main__":
    sys.exit(main())
