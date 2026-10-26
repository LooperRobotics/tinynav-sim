#!/usr/bin/env python3
"""Cycle the four floorplan platform task points around the ring.

All four task points sit on the platforms, each where a flight centreline
meets its platform centreline (the loop corners): WN -> EN -> ES -> WS.
Clockwise = 'down', counterclockwise = 'up'; adjacent points are joined by
the centrelines, so a lap is four straight legs and no routing vias exist
any more.

Run it where /control/target_pose is reachable (same DDS domain as the nav
stack; the mjsim container works).  Uses the emu's ground-truth VIO
/camera/camera/vio_100hz for arrival unless --odom delivers first.

    python3 tools/probes/nav_cycle.py --mode alternate --cycles 2
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "mujoco") not in sys.path:
    sys.path.insert(0, str(_ROOT / "mujoco"))

from sim.floorplan import (  # noqa: E402
    floorplan_loop,
    floorplan_spawns,
    load_params,
)


def _quat_yaw(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _legs(params, clockwise: bool):
    """All four task points in ring order (the loop wraps at the next lap)."""
    return floorplan_loop(params, clockwise=clockwise)


def _print_plan(params, args) -> None:
    print(f"floorplan: {params.summary()}", flush=True)
    for name, legs in (("down/CW", _legs(params, True)),
                       ("up/CCW", _legs(params, False))):
        seq = " -> ".join(
            f"{w['name']}({w['kind']})" for w in legs)
        print(f"  {name:7s}: {seq}", flush=True)


class _CycleNode:
    def __init__(self, args) -> None:
        import rclpy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool
        from rclpy.node import Node
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                               HistoryPolicy)

        self._rclpy = rclpy
        self._Odometry = Odometry
        self._Bool = Bool
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.node = Node("floorplan_cycle")
        self.pub = self.node.create_publisher(Odometry, args.topic, qos)
        # cmd_vel_control gates its passthrough on /nav/active; the backend
        # latches a one-shot false at startup and outlives any --once
        # publisher, so the gate must be held live for the whole run
        self.pub_active = self.node.create_publisher(
            Bool, "/nav/active",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST))
        self._pose = None                       # (x, y, wtime)
        self.node.create_subscription(
            Odometry, args.odom, self._on_odom, qos)
        self.node.create_subscription(
            PoseStamped, args.truth, self._on_truth, qos)
        self.args = args

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        bl = self._to_baselink(p.x, p.y, q.z, q.w)
        self._pose = (bl[0], bl[1], time.monotonic())

    def _on_truth(self, msg) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        bl = self._to_baselink(p.x, p.y, q.z, q.w)
        self._pose = (bl[0], bl[1], time.monotonic())

    def _to_baselink(self, cx: float, cy: float, qz: float, qw: float):
        """VIO/odometry poses are the left-camera optical centre; arrival is
        judged on the body centre.  Subtract the measured mount offset
        (body frame: fwd/left) rotated into the world by the pose yaw."""
        yaw = 2.0 * math.atan2(qz, qw)
        c, s = math.cos(yaw), math.sin(yaw)
        dx = self.args.cam_offset_fwd * c - self.args.cam_offset_left * s
        dy = self.args.cam_offset_fwd * s + self.args.cam_offset_left * c
        return cx - dx, cy - dy

    def latest_pose(self):
        if self._pose is None:
            return None
        if time.monotonic() - self._pose[2] > 2.0:
            return None
        return self._pose[:2]

    def spin_once(self, timeout: float = 0.02) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=timeout)

    def _goal(self, wp) -> object:
        msg = self._Odometry()
        msg.header.frame_id = "world"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(wp["pos"][0])
        msg.pose.pose.position.y = float(wp["pos"][1])
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = _quat_yaw(float(wp["yaw_deg"]))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        return msg

    def run_leg(self, wp) -> tuple[bool, float, float]:
        a = self.args
        goal = self._goal(wp)
        t0 = time.monotonic()
        last_pub = 0.0
        dist = float("nan")
        while self._rclpy.ok():
            self.spin_once()
            now = time.monotonic()
            if now - last_pub >= 1.0 / a.rate:
                goal.header.stamp = self.node.get_clock().now().to_msg()
                self.pub.publish(goal)
                self.pub_active.publish(self._Bool(data=True))
                last_pub = now
            p = self.latest_pose()
            if p is not None:
                dist = math.hypot(p[0] - wp["pos"][0],
                                  p[1] - wp["pos"][1])
                if dist <= a.arrive_radius:
                    return True, dist, now - t0
            if a.timeout > 0.0 and now - t0 > a.timeout:
                return False, dist, now - t0
        return False, dist, time.monotonic() - t0

    def dwell(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while self._rclpy.ok() and time.monotonic() < end:
            self.spin_once(0.05)

    def close(self) -> None:
        self.node.destroy_node()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="floorplan YAML (default assets/floorplan/default.yaml)")
    ap.add_argument("--mode", default="alternate",
                    choices=["down", "up", "alternate"],
                    help="down = clockwise loops, up = counterclockwise, "
                         "alternate = down then up")
    ap.add_argument("--cycles", type=int, default=2,
                    help="number of loops (0 = forever)")
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="seconds at each platform task point")
    ap.add_argument("--rate", type=float, default=2.0,
                    help="target publish rate (Hz)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-leg timeout (s, 0 = never)")
    ap.add_argument("--arrive-radius", type=float, default=0.4)
    ap.add_argument("--cam-offset-fwd", type=float, default=0.190,
                    help="camera forward offset from body centre (m); the "
                         "VIO pose is the camera, arrival is on the body")
    ap.add_argument("--cam-offset-left", type=float, default=0.026,
                    help="camera left offset from body centre (m)")
    ap.add_argument("--start-radius", type=float, default=1.0)
    ap.add_argument("--start-timeout", type=float, default=30.0)
    ap.add_argument("--skip-start-check", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true", default=True)
    ap.add_argument("--topic", default="/control/target_pose")
    ap.add_argument("--odom", default="/slam/odometry_visual")
    ap.add_argument("--truth", default="/camera/camera/vio_100hz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    params = load_params(args.config)
    _print_plan(params, args)
    if args.dry_run:
        return 0

    import rclpy

    rclpy.init()
    node = _CycleNode(args)
    try:
        # wait for a pose and check the robot starts on W
        t0 = time.monotonic()
        while rclpy.ok() and node.latest_pose() is None:
            if time.monotonic() - t0 > args.start_timeout:
                print("no pose from --odom/--truth; is the stack up?",
                      flush=True)
                return 2
            node.spin_once(0.1)
        w = floorplan_spawns(params)["w"][0]
        p = node.latest_pose()
        d0 = math.hypot(p[0] - w[0], p[1] - w[1])
        print(f"start pose ({p[0]:.2f},{p[1]:.2f}) -> W dist {d0:.2f} m",
              flush=True)
        if not args.skip_start_check and d0 > args.start_radius:
            print(f"robot is not on W (radius {args.start_radius:.2f} m); "
                  "reset to the floorplan W spawn first", flush=True)
            return 2

        down = _legs(params, True)
        up = _legs(params, False)
        total = 0
        failed = 0
        for i in (itertools.count() if args.cycles == 0
                  else range(args.cycles)):
            if args.mode == "down":
                direction, legs = "down/CW", down
            elif args.mode == "up":
                direction, legs = "up/CCW", up
            else:
                direction = "down/CW" if i % 2 == 0 else "up/CCW"
                legs = down if i % 2 == 0 else up
            print(f"[loop {i + 1}] {direction}", flush=True)
            for wp in legs:
                ok, dist, dt = node.run_leg(wp)
                total += 1
                print(f"  {wp['name']:1s} {wp['kind']:6s} "
                      f"{'OK ' if ok else 'FAIL'} {dt:6.1f}s "
                      f"dist={dist:.2f} m", flush=True)
                if not ok:
                    failed += 1
                    if args.stop_on_fail:
                        break
                    continue
                if wp["kind"] == "target":
                    node.dwell(args.dwell)
            if failed and args.stop_on_fail:
                break
        print(f"cycle done: {total - failed}/{total} legs OK", flush=True)
        return 0 if failed == 0 else 1
    finally:
        node.close()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
