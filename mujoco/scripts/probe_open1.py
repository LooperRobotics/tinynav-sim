#!/usr/bin/env python3
"""OPEN-1 probe: localize when in-process wgpu init kills the mujoco viewer
repaint loop, and test candidate mitigations. Runs inside the mjsim container
(DISPLAY=:1, mujoco+wgpu in the /opt/mjsim venv).

Modes:
  stageA        viewer first; at t+2s enumerate_adapters, then request_device.
                Reports frames/1s around each step -> localizes the kill point.
  wgpubirst     full wgpu adapter+device init BEFORE the viewer opens.
  xinitthreads  XInitThreads() before anything, then stageA flow.
  intel         stageA flow but request_device on a non-NVIDIA adapter
                (needs mesa-vulkan-drivers installed in the container).

Exit prints RESULT: {last_frame_at, loop_alive, total_frames}; loop_alive
distinguishes "sync() deadlocked" from "present silently dropped".

Run: docker exec mjsim-hil /opt/mjsim/bin/python /tmp/probe_open1.py stageA
"""

import sys
import threading
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "stageA"
DUR = 12

if MODE == "xinitthreads":
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


def wgpu_init(prefer_nvidia=True):
    import wgpu
    t0 = time.perf_counter()
    ads = wgpu.gpu.enumerate_adapters()
    print(f"[wgpu] enumerate {len(ads)} adapters in "
          f"{time.perf_counter()-t0:.2f}s: {[a.summary for a in ads]}", flush=True)
    pred = (lambda a: "nvidia" in a.summary.lower()) if prefer_nvidia \
        else (lambda a: "nvidia" not in a.summary.lower())
    cands = [a for a in ads if pred(a)]
    if not cands:
        raise SystemExit(f"[wgpu] no adapter matching prefer_nvidia={prefer_nvidia}")
    t0 = time.perf_counter()
    dev = cands[0].request_device()
    print(f"[wgpu] device ready ({cands[0].summary}) in "
          f"{time.perf_counter()-t0:.2f}s", flush=True)
    return dev


if MODE == "wgpubirst":
    wgpu_init(prefer_nvidia=True)

import mujoco.viewer  # noqa: E402

viewer = mujoco.viewer.launch_passive(model, data)
print(f"[viewer] launched at t=0", flush=True)

stamps = []
deadline = time.time() + DUR


def frame_loop():
    while viewer.is_running() and time.time() < deadline:
        mujoco.mj_step(model, data)
        viewer.sync()
        stamps.append(time.time())


threading.Thread(target=frame_loop, daemon=True).start()

t0 = time.time()
device_fired = False


def report(tag):
    if stamps:
        recent = sum(1 for t in stamps if t > time.time() - 1.0)
        print(f"[{time.time()-t0:5.1f}s] {tag}: frames_last_1s={recent} "
              f"total={len(stamps)}", flush=True)
    else:
        print(f"[{time.time()-t0:5.1f}s] {tag}: NO FRAMES YET", flush=True)


for i in range(DUR):
    time.sleep(1)
    if MODE in ("stageA", "xinitthreads", "intel") and i == 2:
        wgpu_init(prefer_nvidia=(MODE != "intel"))
        device_fired = True
        report("1s-after-wgpu-device")
    elif device_fired and i == 3:
        report("2s-after-wgpu-device")
    else:
        report("tick")

last = stamps[-1] - t0 if stamps else None
alive = bool(stamps and stamps[-1] > time.time() - 1.2)
print("RESULT:", {"mode": MODE, "last_frame_at": None if last is None else round(last, 2),
                  "loop_alive": alive, "total_frames": len(stamps)}, flush=True)
