#!/usr/bin/env python3
"""Subscriber side for the fused-probe test: plain system python3 + rclpy
(receives what the venv_gs process publishes). Run in a second exec:
  source /opt/ros/humble/setup.bash && /usr/bin/python3 /tmp/fused_sub.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

rclpy.init()
n = Node("fused_sub")
got = []


def cb(m):
    got.append(m.header.stamp.sec)
    if len(got) in (1, 20):
        print(f"recv {len(got)} frame(s), {m.width}x{m.height} {m.encoding}",
              flush=True)


# match the publisher's best_effort sensor QoS (default reliable would never
# match and both sides would sit in incompatible-QoS silence)
n.create_subscription(Image, "/fused_probe/image", cb,
                      QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=10))
while len(got) < 20 and rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.1)
print(f"SUB_OK {len(got)} frames")
