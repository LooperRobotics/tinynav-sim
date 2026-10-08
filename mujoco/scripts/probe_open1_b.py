#!/usr/bin/env python3
"""OPEN-1 probe v2: localize which splatsense pipeline step kills the mujoco
viewer repaint loop in the mjsim container, and test mitigation orders.

stageA bare wgpu enumerate+device does NOT freeze the viewer (v1 result).
v2 escalates to the real init chain hil.py uses.

Modes:
  pipeline      viewer first; at t+2s: np.load -> Pipeline(d) -> first render.
                Frame reports bracket each step -> localizes the kill point.
  wgpubirst     npz+Pipeline+first render BEFORE the viewer opens.
  xinitpipe     XInitThreads() first, then the "pipeline" flow.

Exit prints RESULT: {last_frame_at, loop_alive, total_frames}.

Run: docker exec mjsim-hil /opt/mjsim/bin/python /tmp/probe_open1_b.py pipeline
"""

import sys
import threading
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
DUR = 14

if MODE == "xinitpipe":
    import ctypes
    ctypes.CDLL("libX11.so.6").XInitThreads()
    print("[xinit] XInitThreads done", flush=True)

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

if MODE == "wgpubirst":
    from splatsense.backends.webgpu import Pipeline
    t0 = time.perf_counter()
    d = dict(__import__("numpy").load(
        "/opt/mjsim/mujoco/assets/splat/w1_static.npz"))
    print(f"[wgpu] npz loaded in {time.perf_counter()-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    pipe = Pipeline(d, "5070")
    print(f"[wgpu] Pipeline built in {time.perf_counter()-t0:.2f}s", flush=True)
    t0 = time.perf_counter()
    rgb, dep = pipe.render_frame()
    print(f"[wgpu] first render {rgb.shape}/{dep.shape} in "
          f"{time.perf_counter()-t0:.2f}s", flush=True)
    del pipe

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
    if MODE in ("pipeline", "xinitpipe") and i == 2 and not fired:
        fired = True
        import numpy as np
        from splatsense.backends.webgpu import Pipeline
        t0b = time.perf_counter()
        d = dict(np.load("/opt/mjsim/mujoco/assets/splat/w1_static.npz"))
        print(f"[wgpu] npz loaded in {time.perf_counter()-t0b:.2f}s", flush=True)
        report("after-npz")
        t0b = time.perf_counter()
        pipe = Pipeline(d, "5070")
        print(f"[wgpu] Pipeline built in {time.perf_counter()-t0b:.2f}s", flush=True)
        report("after-pipeline")
        t0b = time.perf_counter()
        rgb, dep = pipe.render_frame()
        print(f"[wgpu] first render in {time.perf_counter()-t0b:.2f}s", flush=True)
        report("after-first-render")
    else:
        report("tick")

last = stamps[-1] - t0 if stamps else None
alive = bool(stamps and stamps[-1] > time.time() - 1.2)
print("RESULT:", {"mode": MODE, "last_frame_at": None if last is None else round(last, 2),
                  "loop_alive": alive, "total_frames": len(stamps)}, flush=True)
