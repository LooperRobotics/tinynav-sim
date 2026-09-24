#!/usr/bin/env python3
"""Standalone viewer window for a running gsplat session (decoupled process).

Why a separate process instead of the server's embedded --window mode: the
container has no GPU GL/Vulkan for wgpu (Mesa llvmpipe only), and the sim's
three camera renders already eat ~0.85 of every wall second — put the viewport
in that same process and rtf collapses to ~0.5 with a choppy window. Here the
viewport costs ~8 ms/frame on its own core (measured 117 fps standalone) while
the simulation keeps full-rate sensors.

Shows the SAME scene + robot MJCF as the running server, re-posed from the
state slot (gsplat/server/gs_state_shm.py: sim_time + full dof_pos, 50 Hz), so
what you see is the live session's collision geometry — mouse orbit/zoom works
as in the upstream demo. It does NOT load gaussians (that would double the
8 GB VRAM footprint); the 3DGS imagery lives in the ROS topics.

Usage (inside the container, alongside a running server):
    .venv_gs/bin/python gsplat/tools/view_window.py            # follows the default slot
    ... --state /dev/shm/gsplay_state.bin --scene church --fps 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# wgpu must NOT see the NVIDIA EGL trio (Mesa llvmpipe drives the viewport;
# NVIDIA EGL makes wgpu's khronos-egl probe panic). /root/.bashrc bakes them
# for rviz/gz, so drop them before importing the renderer.
for _v in ("__GLX_VENDOR_LIBRARY_NAME", "__NV_PRIME_RENDER_OFFLOAD",
           "__EGL_VENDOR_LIBRARY_FILENAMES"):
    os.environ.pop(_v, None)
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
os.environ.setdefault("WAYLAND_DISPLAY", "")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("ALSOFT_DRIVERS", "null")

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent / "server"
sys.path.insert(0, SERVER_DIR.as_posix())

from gs_state_shm import StateShmReader, DEFAULT_PATH  # noqa: E402

from motrixsim import SceneData, forward_kinematic, msd  # noqa: E402
from motrixsim.render import RenderApp  # noqa: E402


def gspg_root() -> Path:
    env = os.environ.get("GS_PLAYGROUND_ROOT")
    if env:
        return Path(env)
    for cand in ("/workspace/github/simulation/gs_playground",
                 "/home/dm/workspace/github/simulation/gs_playground"):
        if Path(cand, "demo", "navigation").is_dir():
            return Path(cand)
    raise SystemExit("gs_playground checkout not found; set GS_PLAYGROUND_ROOT")


GSPG_NAV = gspg_root() / "demo" / "navigation"
SCENES = {                      # mirrors gsplat/configs/*.json
    "church": ("church_scene/mjcf/scene.xml", "church_scene"),
    "nav1": ("nav_scene_1/mjcf/scene.xml", "nav_scene_1"),
}
RIG_REL = "models/robots/navigation/go2/go2_sensor_rig.xml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=DEFAULT_PATH)
    ap.add_argument("--scene", default="church", help="church | nav1 | path to a scene xml")
    ap.add_argument("--rig", default="", help="rig MJCF (default: the GSPG go2 sensor rig)")
    ap.add_argument("--fps", type=float, default=30.0, help="viewer frame cap")
    ap.add_argument("--no-input", action="store_true",
                    help="skip the per-frame ESC poll (X11 round trip); close the "
                         "window or Ctrl+C instead")
    ap.add_argument("--stale-stop", type=float, default=2.0,
                    help="greys the viewport title state if the server slot stalls this long")
    args = ap.parse_args()

    scene_rel = SCENES.get(args.scene, (args.scene, None))[0]
    scene_file = Path(scene_rel)
    if not scene_file.is_absolute():
        scene_file = GSPG_NAV / scene_file
    rig_file = Path(args.rig) if args.rig else (GSPG_NAV / RIG_REL)
    if not scene_file.is_file() or not rig_file.is_file():
        raise SystemExit(f"scene/rig missing:\n  {scene_file}\n  {rig_file}")

    print(f"viewer: scene {scene_file}\n        rig   {rig_file}", flush=True)
    scene = msd.from_file(scene_file.as_posix())
    rig = msd.from_file(rig_file.as_posix())
    scene.attach(rig)
    model = scene.build()
    data = SceneData(model)
    n_dof = int(np.asarray(data.dof_pos).size)

    while True:                 # the server may still be booting
        try:
            state = StateShmReader(args.state)
            break
        except Exception as exc:
            print(f"waiting for state slot {args.state}: {exc}", flush=True)
            time.sleep(1.0)
    print(f"state slot {args.state} ({n_dof} dof); waiting for the first frame...",
          flush=True)

    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    n_applied = n_frames = 0
    t_last_state = time.monotonic()
    t_report = time.monotonic()
    fk_ms = sync_ms = read_ms = 0.0

    with RenderApp() as render:
        render.launch(model)
        # Same group visibility as the sim's viewer: 2/3 are the collision
        # groups the MJCF carries for the scene and the robot.
        render.opt.set_group_vis(2, True)
        render.opt.set_group_vis(3, True)
        print("window up: mouse orbit/zoom; close the window or Ctrl+C to quit",
              flush=True)
        while not render.is_closed:
            t0 = time.monotonic()
            tt = time.perf_counter()
            st = state.read()
            read_ms += time.perf_counter() - tt
            if st is not None:
                if st.dof_pos.size == n_dof:
                    tt = time.perf_counter()
                    data.set_dof_pos(st.dof_pos, model)
                    forward_kinematic(model, data)
                    fk_ms += time.perf_counter() - tt
                    n_applied += 1
                t_last_state = t0
            tt = time.perf_counter()
            render.sync(data)
            sync_ms += time.perf_counter() - tt
            n_frames += 1
            if not args.no_input:
                for key in ("esc", "escape"):
                    if render.input.is_key_just_pressed(key):
                        raise KeyboardInterrupt
            if frame_dt > 0.0:
                slack = frame_dt - (time.monotonic() - t0)
                if slack > 0:
                    time.sleep(slack)
            if t0 - t_report > 10.0:
                age = t0 - t_last_state
                print(f"  viewer {n_frames/(t0 - t_report):.1f} Hz | applied "
                      f"{n_applied} states | read {1000*read_ms/max(n_frames,1):.2f} ms "
                      f"| fk {1000*fk_ms/max(n_applied,1):.2f} ms "
                      f"| sync {1000*sync_ms/max(n_frames,1):.1f} ms | state age {age:.2f}s"
                      + ("  [STALE: server slot quiet]" if age > args.stale_stop else ""),
                      flush=True)
                t_report = t0
                n_applied = n_frames = 0
                fk_ms = sync_ms = read_ms = 0.0
    state.close()
    print("viewer closed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted — closing viewer", flush=True)
