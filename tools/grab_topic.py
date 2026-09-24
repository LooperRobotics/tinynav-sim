#!/usr/bin/env python3
"""One-shot ROS topic grabber — saves raw messages to disk for offline replay.

Born from the 2026-09-19 reloc session: `ros2 topic echo --field data --raw`
returned 0 bytes for keyframe images and the ad-hoc /tmp subscriber script was
thrown away, so the next session had to rewrite it. This is the keeper.

Generic for any type: serializes each message and writes <out>/msg_NNN.bin
(rclpy serialization, re-readable with rosbag2 tools). sensor_msgs/Image is
additionally decoded to <out>/img_NNN.npy + shape/encoding in meta.json —
numpy only, no cv2, so it runs on the bare ROS python:

    docker exec tinynav bash -c \
      'source /opt/ros/humble/setup.bash && \
       python3 /workspace/dm/tinynav-gazebo/tools/grab_topic.py \
       /slam/keyframe_image --count 1 --out /workspace/dm/tinynav-gazebo/fixtures/grab'

If fewer than --count messages arrive within --timeout, exits 1 and keeps
whatever it got.
"""
import argparse
import json
import os
import re
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message

try:
    import numpy as np
except ImportError:  # .bin dumps still work without numpy
    np = None


def safe_name(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", topic).strip("_")


def decode_image(msg):
    """sensor_msgs/Image -> 2D/3D array. dtype & channels from encoding/step."""
    enc = msg.encoding
    if "32F" in enc:
        dtype = np.float32
    elif "16U" in enc or "16S" in enc:
        dtype = np.uint16 if "16U" in enc else np.int16
    else:
        dtype = np.uint8
    itemsize = np.dtype(dtype).itemsize
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    row = msg.step // itemsize if msg.step else msg.width
    arr = arr[: msg.height * row].reshape(msg.height, row)
    chan = row // msg.width if msg.width else 1
    if chan > 1:
        arr = arr.reshape(msg.height, msg.width, chan)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="seconds to wait for the first message (default 10)")
    ap.add_argument("--max-wait", type=float, default=30.0,
                    help="overall seconds before giving up (default 30)")
    ap.add_argument("--out", default=None,
                    help="output dir (default ./grab_<topic>_<timestamp>)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("grab_topic")
    # DDS discovery needs a few seconds to converge in a fresh exec shell; a
    # single get_topic_names_and_types() snapshot here saw only the bridge
    # topics and reported live slam topics as "not in the graph" (2026-09-19).
    type_name = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        types = dict(node.get_topic_names_and_types())
        if args.topic in types:
            type_name = types[args.topic][0]
            break
        rclpy.spin_once(node, timeout_sec=0.3)
    if type_name is None:
        node.get_logger().error(
            f"topic {args.topic} not in the graph after 10s of discovery — "
            "is the stack publishing it?")
        rclpy.shutdown()
        return 1
    from rosidl_runtime_py.utilities import get_message
    msg_cls = get_message(type_name)

    out_dir = args.out or f"grab_{safe_name(args.topic)}_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    got = []
    is_image = type_name == "sensor_msgs/msg/Image"

    def on_msg(msg):
        stamp = ""
        hdr = getattr(msg, "header", None)
        if hdr is not None:
            stamp = f"{hdr.stamp.sec}.{hdr.stamp.nanosec:09d}"
        path_bin = os.path.join(out_dir, f"msg_{len(got):03d}.bin")
        with open(path_bin, "wb") as f:
            f.write(serialize_message(msg))
        entry = {"file": os.path.basename(path_bin), "stamp": stamp}
        if is_image and np is not None:
            arr = decode_image(msg)
            path_npy = os.path.join(out_dir, f"img_{len(got):03d}.npy")
            np.save(path_npy, arr)
            entry.update({"npy": os.path.basename(path_npy), "shape": list(arr.shape),
                          "encoding": msg.encoding, "width": msg.width,
                          "height": msg.height, "step": msg.step})
        got.append(entry)
        print(f"[grab] {len(got)}/{args.count} {type_name} stamp={stamp or '?'} -> {out_dir}",
              flush=True)

    sub = node.create_subscription(msg_cls, args.topic, on_msg,
                                   qos_profile_sensor_data)
    node.get_logger().info(f"waiting for {args.count} x {type_name} on {args.topic}")
    t0 = time.monotonic()
    while rclpy.ok() and len(got) < args.count:
        if not got and time.monotonic() - t0 > args.timeout:
            node.get_logger().error(f"no message within {args.timeout}s — topic alive?")
            break
        if time.monotonic() - t0 > args.max_wait:
            node.get_logger().error(f"got {len(got)}/{args.count} before max-wait")
            break
        rclpy.spin_once(node, timeout_sec=0.2)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"topic": args.topic, "type": type_name, "count": len(got),
                   "messages": got}, f, indent=2)
    print(f"[grab] saved {len(got)} message(s) to {out_dir}/meta.json", flush=True)
    rclpy.shutdown()
    return 0 if len(got) >= args.count else 1


if __name__ == "__main__":
    sys.exit(main())
