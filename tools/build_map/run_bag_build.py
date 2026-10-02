#!/usr/bin/env python3
"""Orchestrator: rosbag -> (perception if needed) + builder -> map v2 dir.

Topology mirrors scripts/run_rosbag_build_map.sh: a BagPlayer replays the bag
while perception_node computes VIO (/slam/*) live, and the builder consumes
keyframe topics. If the bag already contains /slam/keyframe_image (recorded
with perception running), perception is skipped unless --perception on.

Run inside the rig container (tinynav), from the repo root:
  /opt/venv/bin/python3 tools/build_map/run_bag_build.py \
      --bag <rosbag2 dir> --out <map output dir> [--perception auto]

Env requirements (gazebo/run_simulator.sh gotchas apply):
- PYTHONPATH must list, in order: reference/, the tinynav repo root (tool/
  package), tools/  — this script prepends them if absent.
- ROS setup sourced before starting (script falls back to bash -lc sourcing
  for the perception subprocess; the main process is expected to run under
  `bash -c "source /opt/ros/humble/setup.bash && ..."`).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # tinynav-sim/
_REFERENCE = os.path.join(_REPO, "reference")
_TINYNAV_REPO = "/workspace/dm/tinynav-pilot/tinynav"    # provides tool/ package
for p in (_REFERENCE, _TINYNAV_REPO, _HERE, os.path.join(_REPO, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # noqa: E402


def bag_topics(bag: str) -> set[str]:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id="sqlite3"),
                ConverterOptions(input_serialization_format="cdr",
                                 output_serialization_format="cdr"))
    topics = {t.name for t in reader.get_all_topics_and_types()}
    reader.close() if hasattr(reader, "close") else None
    return topics


def wait_for_topic(node, topic: str, timeout_s: float, msg_type) -> bool:
    got = {"ok": False}

    def _cb(_msg):
        got["ok"] = True

    sub = node.create_subscription(msg_type, topic, _cb, 1)
    deadline = time.time() + timeout_s
    while rclpy.ok() and not got["ok"] and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got["ok"]


def launch_perception() -> subprocess.Popen:
    env = dict(os.environ)
    pypath = ":".join(p for p in (_REFERENCE, _TINYNAV_REPO) if p)
    env["PYTHONPATH"] = pypath + ":" + env.get("PYTHONPATH", "")
    cmd = ("set +u; source /opt/ros/humble/setup.bash; "
           "source /3rdparty/message_filters_ws/install/local_setup.bash; "
           "export PYTHONPATH=\"" + pypath + ":$PYTHONPATH\"; "
           "exec /opt/venv/bin/python3 -u "
           "/workspace/dm/tinynav-sim/reference/tinynav/core/perception_node.py")
    proc = subprocess.Popen(["bash", "-c", cmd], env=env,
                            start_new_session=True)
    print(f"[orchestrator] perception_node launched (pid {proc.pid})")
    return proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--out", required=True, help="map v2 output directory")
    ap.add_argument("--play-rate", type=float, default=1.0)
    ap.add_argument("--perception", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--siglip2-plan", default=None,
                    help="override siglip2 image plan path")
    ap.add_argument("--global-frames-ratio", type=float, default=1.1)
    args = ap.parse_args()

    topics = bag_topics(args.bag)
    print(f"[orchestrator] bag topics ({len(topics)}): {sorted(topics)[:12]}...")
    has_slam = "/slam/keyframe_image" in topics
    need_perception = (args.perception == "on") or (args.perception == "auto" and not has_slam)
    if not has_slam and not need_perception:
        print("[orchestrator] WARNING: bag has no /slam/* topics and --perception off; "
              "builder will see nothing")

    from sensor_msgs.msg import Image
    from tinynav.core.build_map_node import BagPlayer
    from build_map.builder_node import BuilderNode, DEFAULT_SIGLIP2_PLAN

    rclpy.init()
    perception_proc = None
    try:
        if need_perception:
            perception_proc = launch_perception()
            probe = rclpy.create_node("build_probe")
            ok = wait_for_topic(probe, "/slam/keyframe_image", 300.0, Image)
            probe.destroy_node()
            if not ok:
                print("[orchestrator] WARNING: /slam/keyframe_image not seen within 300s; "
                      "continuing anyway (bag play may still succeed)")
            else:
                print("[orchestrator] perception is publishing keyframes")

        exec_ = SingleThreadedExecutor()
        player = BagPlayer(args.bag, play_rate=args.play_rate)
        builder = BuilderNode(
            args.out,
            siglip2_plan=args.siglip2_plan or DEFAULT_SIGLIP2_PLAN,
            global_frames_ratio=args.global_frames_ratio)
        exec_.add_node(player)
        exec_.add_node(builder)
        while rclpy.ok() and player.play_next():
            exec_.spin_once(timeout_sec=0.001)
        player._publish_percent(100.0)
        print("[orchestrator] bag playback finished, saving map ...")
        builder.save_mapping()
        print("[orchestrator] done")
    finally:
        if perception_proc is not None and perception_proc.poll() is None:
            os.killpg(perception_proc.pid, signal.SIGTERM)
            try:
                perception_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(perception_proc.pid, signal.SIGKILL)
            print("[orchestrator] perception stopped")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
