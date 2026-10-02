#!/usr/bin/env python3
"""Fused-process probe: rclpy + torch + gsplat + numpy 2.x in ONE interpreter.

Question answered: can gsplat's sensor_server publish ROS topics directly and
drop the /dev/shm ring? Requires rclpy (apt, python3-rclpy via
--system-site-packages) to coexist with the gsplat stack in /opt/venv_gs.

Run inside the probe container:
  /opt/venv_gs/bin/python /tmp/fused_pub.py
"""
import time
from array import array

import numpy as np
import rclpy
import torch
from gsplat.cuda._backend import _C  # noqa: F401  (CUDA extension loads in-process)
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

print(f"numpy={np.__version__} torch={torch.__version__} "
      f"cuda_ok={torch.cuda.is_available()} gsplat_C={_C is not None}")

rclpy.init()
node = Node("fused_probe")
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=5)
pub = node.create_publisher(Image, "/fused_probe/image", qos)

h, w = 480, 544  # gsplat sim frame size
msg = Image(height=h, width=w, encoding="mono8", is_bigendian=0, step=w)
msg.data = array("B")
arr = np.random.randint(0, 255, (h, w), dtype=np.uint8)
raw = np.ascontiguousarray(arr).tobytes()

def frame_payload():
    msg.data.frombytes(raw)
    msg.header.stamp = node.get_clock().now().to_msg()

# -- phase A: payload build only (what the README's 0.12 ms claim covers)
frame_payload()
N = 300
t0 = time.perf_counter()
for _ in range(N):
    frame_payload()
t1 = time.perf_counter()
print(f"payload-only : {(t1-t0)/N*1000:.3f} ms/frame")

# -- phase B: publish only, no spinning
pub.publish(msg)
rclpy.spin_once(node, timeout_sec=0.1)
time.sleep(1.0)  # let discovery match
t0 = time.perf_counter()
for _ in range(N):
    pub.publish(msg)
t1 = time.perf_counter()
print(f"publish-only : {(t1-t0)/N*1000:.3f} ms/frame")

# -- phase C: the real loop — payload + publish + housekeeping spin
t0 = time.perf_counter()
for i in range(N):
    frame_payload()
    pub.publish(msg)
    if i % 10 == 0:
        rclpy.spin_once(node, timeout_sec=0.0)
t1 = time.perf_counter()
print(f"full loop    : {(t1-t0)/N*1000:.3f} ms/frame "
      f"(15 Hz budget is 66.7 ms)")
node.destroy_node()
rclpy.shutdown()
