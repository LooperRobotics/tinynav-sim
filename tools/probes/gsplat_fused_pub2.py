#!/usr/bin/env python3
"""Sustained-rate publisher probe: publish Image at --rate Hz for --seconds,
optionally with --heavy (import torch + init CUDA + gsplat first, i.e. the
fused single-process scenario).

Usage:
  fused_pub2.py --topic /t --size 544x480 --rate 15 --seconds 20 [--heavy]
"""
import argparse
import time
from array import array

ap = argparse.ArgumentParser()
ap.add_argument("--topic", default="/fused_probe2/image")
ap.add_argument("--size", default="544x480")
ap.add_argument("--rate", type=float, default=15.0)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--heavy", action="store_true",
                help="import torch + init CUDA + gsplat before publishing")
ap.add_argument("--rgb", action="store_true",
                help="rgb8 payload (783 KB for 544x480, the real color frame)")
a = ap.parse_args()
w, h = (int(v) for v in a.size.split("x"))
channels = 3 if a.rgb else 1

if a.heavy:
    import torch  # noqa: F401
    from gsplat.cuda._backend import _C  # noqa: F401
    torch.zeros(1, device="cuda")

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

rclpy.init()
node = Node("fused_probe2")
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                 history=HistoryPolicy.KEEP_LAST, depth=5)
pub = node.create_publisher(Image, a.topic, qos)

msg = Image(height=h, width=w,
            encoding="rgb8" if a.rgb else "mono8", is_bigendian=0,
            step=w * channels)
msg.data = array("B")
msg.data.frombytes(np.zeros(h * w * channels, dtype=np.uint8).tobytes())

period = 1.0 / a.rate
next_t = time.perf_counter() + 1.0  # discovery grace
n = 0
t_end = next_t + a.seconds
send_cost = 0.0
while time.perf_counter() < t_end:
    msg.header.stamp = node.get_clock().now().to_msg()
    s0 = time.perf_counter()
    pub.publish(msg)
    send_cost += time.perf_counter() - s0
    n += 1
    next_t += period
    dt = next_t - time.perf_counter()
    if dt > 0:
        time.sleep(dt)
    else:
        next_t = time.perf_counter()
    rclpy.spin_once(node, timeout_sec=0.0)
print(f"PUB2 heavy={a.heavy} {a.size}@{a.rate}Hz x {a.seconds}s: "
      f"{n} frames, publish cost {send_cost/n*1000:.3f} ms/frame")
node.destroy_node()
rclpy.shutdown()
