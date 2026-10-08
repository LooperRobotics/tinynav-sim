#!/usr/bin/env python3
"""GLFW/MuJoCo viewer health probe for launch.sh.

Creates a minimal viewer window and prints VIEWER-MAPPED once the
compositor maps it. Used to detect the X-Wayland wedge where stale
surfaces from kill -9'd GL processes make glfwCreateWindow block
forever (launch_passive never returns). Run under `timeout 8`: a hang
produces no marker, which launch.sh treats as failure.

A segfault during teardown after the marker is NOT a failure — only the
marker matters.
"""
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>"
)
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d) as v:
    print("VIEWER-MAPPED", flush=True)
    t0 = time.time()
    while v.is_running() and time.time() - t0 < 2:
        v.sync()
        time.sleep(0.05)
print("VIEWER-CLOSED", flush=True)
