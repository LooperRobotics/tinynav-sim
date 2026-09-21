#!/usr/bin/env python3
"""Data-plane publisher: publish N messages from a real rclpy node.

The ros2 CLI is blind under Fast DDS Discovery Server in Humble — including
its PUBLISHERS (`ros2 topic pub` never matches remote subscriptions, so
messages are silently dropped). Split-site goal publishing goes through
this node instead:

    python3 tools/probes/probe_pub_once.py /control/target_pose \
        nav_msgs/msg/Odometry 20 2 \
        '{header: {frame_id: world}, pose: {pose: {position: {x: 6.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}'

Args: topic short-or-full type count rate yaml-string. Prints per-publish
lines so the caller can tell delivery attempts from silence.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

SHORT_TYPES = {
    "Odometry": "nav_msgs/msg/Odometry",
    "Twist": "geometry_msgs/msg/Twist",
    "Empty": "std_msgs/msg/Empty",
}


def main() -> int:
    topic, type_arg = sys.argv[1], sys.argv[2]
    count, rate_hz = int(sys.argv[3]), float(sys.argv[4])
    yaml_str = sys.argv[5] if len(sys.argv) > 5 else "{}"
    type_name = SHORT_TYPES.get(type_arg, type_arg)

    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosidl_runtime_py import set_message_fields
    import yaml
    msg = get_message(type_name)()
    values = yaml.safe_load(yaml_str) or {}
    set_message_fields(msg, values)

    rclpy.init()
    node = Node("probe_pub_once")
    # RELIABLE by default: command topics (targets, resets) demand it; a
    # best_effort pub silently matches nothing on a reliable subscription.
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(msg.__class__, topic, qos)
    print(f"[pub] {count} x {type_name} at {rate_hz}Hz on {topic}", flush=True)
    sent = 0
    t0 = time.monotonic()
    while rclpy.ok() and sent < count:
        pub.publish(msg)
        sent += 1
        print(f"[pub] {sent}/{count}", flush=True)
        target = t0 + sent / rate_hz
        while time.monotonic() < target:
            rclpy.spin_once(node, timeout_sec=0.05)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
