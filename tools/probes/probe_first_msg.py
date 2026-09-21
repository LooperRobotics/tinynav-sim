#!/usr/bin/env python3
"""Data-plane probe: subscribe a topic, print the first message summary.

ros2 CLI graph/data tools (node list, topic hz) are BLIND under Fast DDS
Discovery Server client mode in Humble — they report "not published" while
real nodes exchange messages fine (verified 2026-09-20 with demo talker/
listener). Verification of the split-site link therefore goes through real
rclpy subscriptions; this is the one-liner for that.

    docker exec <site> bash -c 'source /opt/ros/humble/setup.bash && \
      python3 tools/probes/probe_first_msg.py /slam/odometry Odometry 15'

Type is a short name (Image/Odometry/Imu/CameraInfo/Clock) or a full path;
when omitted, the graph API is consulted once. Exits 0 on first message,
1 on timeout. (One type per process: rmw forbids same-topic subscriptions
with different types even across nodes of one process.)
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

SHORT_TYPES = {
    "Image": "sensor_msgs/msg/Image",
    "Odometry": "nav_msgs/msg/Odometry",
    "Imu": "sensor_msgs/msg/Imu",
    "CameraInfo": "sensor_msgs/msg/CameraInfo",
    "Clock": "rosgraph_msgs/msg/Clock",
}


def describe(msg) -> str:
    stamp = getattr(msg, "header", None)
    stamp_s = (f"{stamp.stamp.sec}.{stamp.stamp.nanosec:09d}"
               if stamp is not None else "-")
    detail = ""
    if hasattr(msg, "pose"):
        p = msg.pose.pose.position
        detail = f" pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})"
    elif hasattr(msg, "width") and hasattr(msg, "encoding"):
        detail = f" {msg.width}x{msg.height} {msg.encoding}"
    elif hasattr(msg, "linear_acceleration"):
        a = msg.linear_acceleration
        detail = f" acc=({a.x:.2f},{a.y:.2f},{a.z:.2f})"
    return f"stamp={stamp_s}{detail}"


def main() -> int:
    topic = sys.argv[1]
    type_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    timeout_s = float(sys.argv[3] if len(sys.argv) > 3 else 15)
    type_name = SHORT_TYPES.get(type_arg, type_arg)

    rclpy.init()
    node = Node("probe_first_msg")
    if not type_name:
        for name, types in node.get_topic_names_and_types():
            if name == topic and types:
                type_name = types[0]
                break
    if not type_name:
        print(f"[probe] {topic}: pass a type argument "
              f"({', '.join(SHORT_TYPES)})", flush=True)
        rclpy.shutdown()
        return 1

    from rosidl_runtime_py.utilities import get_message
    msg_cls = get_message(type_name)
    state = {"done": False}

    def on_msg(msg):
        if state["done"]:
            return
        state["done"] = True
        print(f"[probe] {topic} [{type_name}] {describe(msg)}", flush=True)

    node.create_subscription(msg_cls, topic, on_msg, qos_profile_sensor_data)
    print(f"[probe] waiting for {topic} ({type_name})", flush=True)
    t0 = time.monotonic()
    while rclpy.ok() and not state["done"]:
        if time.monotonic() - t0 > timeout_s:
            print(f"[probe] TIMEOUT on {topic}", flush=True)
            rclpy.shutdown()
            return 1
        rclpy.spin_once(node, timeout_sec=0.2)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
