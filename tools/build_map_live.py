#!/usr/bin/env python3
"""Spin BuildMapNode against a LIVE stack — sim map-capture aid.

Production maps are built offline: rosbag-record the keyframe streams, then
`build_map_node.py --bag_file ...` replays them through the same node. This
runner skips the bag and consumes the live /slam/keyframe_* topics directly
(e.g. from the C++ rig). Same node, same save path. Ctrl+C or
/benchmark/stop=true triggers save_mapping (VLAD train + occupancy bake).

Usage: python3 tools/build_map_live.py --map_save_path output/map_build_test
"""
import argparse

import rclpy
from rclpy.executors import SingleThreadedExecutor

from tinynav.core.build_map_node import BuildMapNode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map_save_path", default="output/map_build_test")
    args = parser.parse_args()

    rclpy.init()
    node = BuildMapNode(args.map_save_path)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    print(f"[build_map_live] capturing keyframes into {args.map_save_path} "
          "(Ctrl+C to save and exit)")
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.save_mapping()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
