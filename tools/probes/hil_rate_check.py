#!/usr/bin/env python3
"""Measure real subscriber-side rate for the contract topics (rclpy, sensor QoS).

    python3 hil_rate_check.py [seconds]

The ros2 CLI is blind under Cyclone/Discovery-Server here; this counts actual
delivered messages on the same QoS the stack uses.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

TOPIcs = {
    "/camera/camera/infra1/image_rect_raw": (Image, "Image"),
    "/camera/camera/vio_image": (Image, "vio_image"),
    "/camera/camera/imu": (Imu, "Imu"),
}


def main() -> int:
    dur = float(sys.argv[1] if len(sys.argv) > 1 else 10.0)
    rclpy.init()
    node = Node("hil_rate_check")
    counts = {t: 0 for t in TOPIcs}
    last = {}

    def make_cb(topic):
        def cb(msg):
            counts[topic] += 1
            last[topic] = time.monotonic()
        return cb

    for topic, (msg_type, _kind) in TOPIcs.items():
        node.create_subscription(msg_type, topic, make_cb(topic),
                                 qos_profile_sensor_data)
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        rclpy.spin_once(node, timeout_sec=0.1)
    for topic, n in counts.items():
        print(f"{topic:45s} {n/dur:6.2f} Hz   ({n} msgs in {dur:.0f}s)")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
