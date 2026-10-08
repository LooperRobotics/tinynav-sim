#!/usr/bin/env python3
"""OPEN-1 probe v3: EGL leg of the viewer freeze. v2 showed viewer+wgpu
pipeline+first render does NOT freeze; hil.py additionally creates
mujoco.Renderer (EGL, MUJOCO_GL=egl -> NVIDIA in container) for the policy
depth camera right AFTER the wgpu pipeline -- the old forensics death time
also matches this step. This probe isolates it.

Modes:
  eglonly   viewer first; at t+2s create mujoco.Renderer (EGL) + one render.
  egl       viewer first; wgpu Pipeline -> first render -> EGL Renderer.
  wgpuEGLfirst  wgpu Pipeline + render + EGL Renderer BEFORE viewer opens.

Run: docker exec mjsim-hil /opt/mjsim/bin/python /tmp/probe_open1_c.py egl
"""

import sys
import threading
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "egl"
DUR = 14
NPZ = "/workspace/tinynav-sim/mujoco/assets/splat/w1_static.npz"

import mujoco  # noqa: E402

XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 1.5"><freejoint/><geom type="sphere" size=".12" mass="1"/></body>
  </worldbody>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)


def build_pipeline():
    import numpy as np
    from splatsense.backends.webgpu import Pipeline
    t0 = time.perf_counter()
    d = dict(np.load(NPZ))
    d["W"] = __import__("numpy").int32(640)
    d["H"] = __import__("numpy").int32(544)
    d["fovy"] = __import__("numpy").float32(93.168)
    d["cam_pos"] = __import__("numpy").zeros((2, 3), np.float32)
    d["cam_xmat"] = __import__("numpy").eye(3, dtype=np.float32)[None].repeat(2, 0)
    print(f"[wgpu] npz loaded in {time.perf_counter()-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    pipe = Pipeline(d, "5070")
    print(f"[wgpu] Pipeline built in {time.perf_counter()-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    pipe.render_frame()
    print(f"[wgpu] first render in {time.perf_counter()-t0:.2f}s", flush=True)
    return pipe


def build_egl_renderer():
    t0 = time.perf_counter()
    r = mujoco.Renderer(model, height=64, width=106)
    r.enable_depth_rendering()
    r.update_scene(data)
    img = r.render()
    print(f"[egl] Renderer created + rendered {img.shape} in "
          f"{time.perf_counter()-t0:.2f}s", flush=True)
    return r


if MODE == "wgpuEGLfirst":
    build_pipeline()
    build_egl_renderer()

import mujoco.viewer  # noqa: E402

viewer = mujoco.viewer.launch_passive(model, data)
print("[viewer] launched at t=0", flush=True)

stamps = []
deadline = time.time() + DUR


def frame_loop():
    while viewer.is_running() and time.time() < deadline:
        mujoco.mj_step(model, data)
        viewer.sync()
        stamps.append(time.time())


threading.Thread(target=frame_loop, daemon=True).start()

t0 = time.time()
fired = False


def report(tag):
    if stamps:
        recent = sum(1 for t in stamps if t > time.time() - 1.0)
        print(f"[{time.time()-t0:5.1f}s] {tag}: frames_last_1s={recent} "
              f"total={len(stamps)}", flush=True)
    else:
        print(f"[{time.time()-t0:5.1f}s] {tag}: NO FRAMES YET", flush=True)


for i in range(DUR):
    time.sleep(1)
    if i == 2 and not fired and MODE in ("eglonly", "egl"):
        fired = True
        if MODE == "egl":
            build_pipeline()
            report("after-wgpu-pipeline+render")
        build_egl_renderer()
        report("after-egl-renderer")
    else:
        report("tick")

last = stamps[-1] - t0 if stamps else None
alive = bool(stamps and stamps[-1] > time.time() - 1.2)
print("RESULT:", {"mode": MODE, "last_frame_at": None if last is None else round(last, 2),
                  "loop_alive": alive, "total_frames": len(stamps)}, flush=True)
