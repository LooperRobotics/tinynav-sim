#!/usr/bin/env python3
"""Rate-counting subscriber: report frames received over --seconds window."""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

ap = argparse.ArgumentParser()
ap.add_argument("--topic", default="/fused_probe2/image")
ap.add_argument("--seconds", type=float, default=25.0)
a = ap.parse_args()

rclpy.init()
n = Node("fused_sub_rate")
count = 0
first = None
last = None


def cb(m):
    global count, first, last
    count += 1
    now = n.get_clock().now()
    if first is None:
        first = now
    last = now


n.create_subscription(Image, a.topic, cb,
                      QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=20))
import time
t_end = time.monotonic() + a.seconds
while time.monotonic() < t_end and rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.1)
dur = (last - first).nanoseconds * 1e-9 if first and last and count > 1 else 0.0
rate = (count - 1) / dur if dur > 0 else 0.0
print(f"SUB_RATE {count} frames in window, sustained rate {rate:.1f} Hz")
