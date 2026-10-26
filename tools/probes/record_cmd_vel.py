#!/usr/bin/env python3
"""Record /cmd_vel during a live nav lap for the in-process replay.

JSON lines of (t_ns, vx, vy, wz) on stdout; Ctrl-C / SIGTERM flushes.
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

rclpy.init()
node = Node("cmdvel_rec")
rows = []


def cb(m):
    t = time.time_ns()
    rows.append((t, m.linear.x, m.linear.y, m.angular.z))


node.create_subscription(Twist, "/cmd_vel", cb, 10)
end = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 200
try:
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
except KeyboardInterrupt:
    pass
with open("/dev/stdout", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"recorded {len(rows)} cmd_vel samples", file=sys.stderr)
