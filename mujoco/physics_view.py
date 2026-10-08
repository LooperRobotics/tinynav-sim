#!/usr/bin/env python3
"""Standalone MuJoCo host: PIE policy + map3 + viewer window only.

No wgpu splat, no OpenCV window — the viewer is the single X client
(besides the EGL offscreen depth camera). Space stop / R reset live in
the viewer callback; arrows are the global /dev/input poll. Run:

  .venv*/bin/python physics_view.py [--spawn down|f4|flat1]
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

try:
    ctypes.CDLL("libX11.so.6").XInitThreads()
except OSError:
    pass

import mujoco
import mujoco.viewer
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sim.keyboard import HoldCmd  # noqa: E402
from sim.plant import (  # noqa: E402
    SCENES,
    build_model,
    get_scene,
    reset_to_spawn,
)
from sim.policy import PieOnnxPolicy  # noqa: E402
from sim.runtime import DepthCamera, PieRuntime  # noqa: E402

CONTROL_DT = 0.02
SYNC_HZ = 30.0


def main() -> int:
    scene_name = "map3"
    if "--scene" in sys.argv:
        scene_name = sys.argv[sys.argv.index("--scene") + 1]
    sc = get_scene(scene_name)
    spawn = sc.default_spawn
    if "--spawn" in sys.argv:
        spawn = sys.argv[sys.argv.index("--spawn") + 1]
    if spawn not in sc.spawns:
        raise SystemExit(f"unknown spawn {spawn!r}; choose {sorted(sc.spawns)}")
    pos, yaw_deg = sc.spawns[spawn]

    model, b = build_model(scene_name)
    model.vis.quality.shadowsize = 0          # shadows off (viewer + renders)
    data = mujoco.MjData(model)

    policy = PieOnnxPolicy(_HERE / "assets" / "policy" / sc.hil_policy)
    cmd = HoldCmd(use_global=True)
    cam = DepthCamera(model, b.depth_camera_id)
    reset_to_spawn(model, data, b, settle=True, pos=pos, yaw_deg=yaw_deg)
    rt = PieRuntime(model, b, data, policy, cam, cmd,
                    spawn_pos=pos, spawn_yaw_deg=yaw_deg)

    quit_req = False

    def key_cb(keycode: int) -> bool:
        nonlocal quit_req
        if int(keycode) in (ord("Q"), 27):
            quit_req = True
            return True
        return cmd.key(keycode)

    viewer = None
    holder: dict = {}
    ready = threading.Event()
    ui = os.environ.get("MJUI", "both")   # both | left | right | none

    def _open_viewer():
        try:
            holder["v"] = mujoco.viewer.launch_passive(
                model, data, key_callback=key_cb,
                show_left_ui=ui in ("both", "left"),
                show_right_ui=ui in ("both", "right"))
        except Exception as exc:      # noqa: BLE001
            print(f"[VIEWER] launch failed: {exc}", flush=True)
        finally:
            ready.set()

    threading.Thread(target=_open_viewer, daemon=True).start()
    if not ready.wait(timeout=12.0):
        print("[VIEWER] window not mapped in 12s — exiting", flush=True)
        return 1
    viewer = holder.get("v")
    if viewer is None:
        return 1
    try:
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    except Exception:      # noqa: BLE001 -- older mujoco without handle
        pass
    print(f"viewer up  spawn={spawn}  arrows=global hold  "
          f"Space stop / R reset / q quit (viewer focused)", flush=True)

    next_ctrl = time.perf_counter()
    n_sync = 0
    t_last = time.perf_counter()
    n_ticks = 0
    while viewer.is_running() and not quit_req:
        now = time.perf_counter()
        if now >= next_ctrl:
            if cmd.consume_reset():
                reset_to_spawn(model, data, b, settle=True,
                               pos=pos, yaw_deg=yaw_deg)
                print("[RESET] back to spawn", flush=True)
            rt.tick()
            n_ticks += 1
            next_ctrl = max(next_ctrl + CONTROL_DT, time.perf_counter())
        if now - t_last >= 2.0:
            v = cmd.vector()
            print(f"{n_ticks * CONTROL_DT:6.1f}s  "
                  f"vx={v[0]:+.2f} wz={v[2]:+.2f} "
                  f"up={rt.base_uprightness():.2f} "
                  f"phy={rt.physics_ms:.1f} pol={rt.policy_ms:.1f}ms "
                  f"xyz={np.round(rt.base_pos(), 1)}", flush=True)
            t_last = now
        if n_sync * (1 / SYNC_HZ) <= now:
            viewer.sync()
            n_sync += 1
        wake = min(next_ctrl, time.perf_counter() + 0.002) - time.perf_counter()
        if wake > 0:
            time.sleep(wake)

    cmd.stop()
    cam.close()
    print("bye", flush=True)
    os._exit(0)   # skip atexit: mixed GL teardown segfaults (see docs §4.6)


if __name__ == "__main__":
    sys.exit(main())
