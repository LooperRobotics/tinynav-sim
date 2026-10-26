#!/usr/bin/env python3
"""Send the four platform corner goals the way the frontend does -- direct
/control/target_pose publishing -- with arrival judged on the BODY CENTRE
and no early hand-off to the next goal.

Arrival, in order of preference:
  * body-centre distance <= --arrive-radius (0.20 m), or
  * the body has SETTLED (displacement < --settle-move over --settle-time)
    and is still within --settle-radius (0.35 m).

The second clause exists because the planner's near-goal dead zone stops
driving before the last centimetres; requiring the tight radius alone
would hang the leg.  Both clauses are measured on the body centre, so the
hand-off no longer happens while the camera (0.19 m ahead) is still short
of the goal -- that offset was buying a ~0.2 m early turn.

    python3 tools/probes/target_cycle.py --cycles 1
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "mujoco") not in sys.path:
    sys.path.insert(0, str(_ROOT / "mujoco"))

from sim.floorplan import floorplan_loop, load_params  # noqa: E402


class Cycle:
    def __init__(self, args) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Bool, Float64

        self._rclpy = rclpy
        self._Odometry = Odometry
        self._Bool = Bool
        self._Float64 = Float64
        self.args = args
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.node = Node("target_cycle")
        self.pub = self.node.create_publisher(Odometry, args.topic, qos)
        self.heading_pub = self.node.create_publisher(
            Float64, "/control/target_heading", 10)
        self.active_pub = self.node.create_publisher(
            Bool, "/nav/active",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       history=HistoryPolicy.KEEP_LAST))
        self.hist = collections.deque(maxlen=200)
        self.node.create_subscription(PoseStamped, args.truth,
                                      self._on_pose, qos)

    def _on_pose(self, msg) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        a = self.args
        c, s = math.cos(yaw), math.sin(yaw)
        dx = a.cam_offset_fwd * c - a.cam_offset_left * s
        dy = a.cam_offset_fwd * s + a.cam_offset_left * c
        self.hist.append((time.monotonic(), p.x - dx, p.y - dy, yaw))

    def pose(self):
        if not self.hist:
            return None
        t, x, y, _ = self.hist[-1]
        return (x, y) if time.monotonic() - t < 2.0 else None

    def yaw(self):
        if not self.hist:
            return None
        t, _, _, yaw = self.hist[-1]
        return yaw if time.monotonic() - t < 2.0 else None

    def settled(self) -> bool:
        a = self.args
        now = time.monotonic()
        window = [h for h in self.hist if now - h[0] <= a.settle_time]
        if len(window) < 5 or now - window[0][0] < a.settle_time * 0.8:
            return False
        xs = [h[1] for h in window]
        ys = [h[2] for h in window]
        yaws = [h[3] for h in window]
        # rotating in place holds the position still: a yaw sweep is not a
        # settle, or the endgame align phase would be read as arrival
        yaw_span = max(yaws) - min(yaws)
        return ((max(xs) - min(xs)) < a.settle_move
                and (max(ys) - min(ys)) < a.settle_move
                and yaw_span < a.settle_yaw)

    def spin(self, t=0.02) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=t)

    def goal(self, wp):
        m = self._Odometry()
        m.header.frame_id = "world"
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.pose.pose.position.x = float(wp["pos"][0])
        m.pose.pose.position.y = float(wp["pos"][1])
        yaw = math.radians(float(wp["yaw_deg"]))
        m.pose.pose.orientation.z = math.sin(yaw / 2.0)
        m.pose.pose.orientation.w = math.cos(yaw / 2.0)
        return m

    def run_leg(self, wp) -> tuple[bool, float, float, str]:
        a = self.args
        g = self.goal(wp)
        t0 = time.monotonic()
        last_pub = 0.0
        dist = float("nan")
        while self._rclpy.ok():
            self.spin()
            now = time.monotonic()
            if now - last_pub >= 0.5:
                g.header.stamp = self.node.get_clock().now().to_msg()
                self.pub.publish(g)
                self.heading_pub.publish(
                    self._Float64(data=math.radians(float(wp["yaw_deg"]))))
                self.active_pub.publish(self._Bool(data=True))
                last_pub = now
            p = self.pose()
            if p is None:
                continue
            dist = math.hypot(p[0] - wp["pos"][0], p[1] - wp["pos"][1])
            if dist <= a.arrive_radius:
                # let the controller finish its in-place heading alignment
                # before the next goal replaces this one
                yaw_goal = math.radians(float(wp["yaw_deg"]))
                t_arrive = now
                while self._rclpy.ok():
                    self.spin()
                    self.pub.publish(g)
                    self.heading_pub.publish(self._Float64(data=yaw_goal))
                    self.active_pub.publish(self._Bool(data=True))
                    y = self.yaw()
                    if y is not None:
                        err = math.atan2(math.sin(yaw_goal - y),
                                         math.cos(yaw_goal - y))
                        if abs(err) <= a.align_tol:
                            return True, dist, now - t0, "radius+aligned"
                    if time.monotonic() - t_arrive > a.align_wait:
                        return True, dist, now - t0, "radius+align-timeout"
                    time.sleep(0.05)
                return True, dist, now - t0, "radius"
            if dist <= a.settle_radius and self.settled():
                return True, dist, now - t0, "settled"
            if a.timeout > 0 and now - t0 > a.timeout:
                return False, dist, now - t0, "timeout"
        return False, dist, time.monotonic() - t0, "shutdown"

    def dwell(self, s: float) -> None:
        end = time.monotonic() + s
        while self._rclpy.ok() and time.monotonic() < end:
            self.active_pub.publish(self._Bool(data=True))
            self.spin(0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--mode", default="down", choices=["down", "up", "alternate"])
    ap.add_argument("--dwell", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--arrive-radius", type=float, default=0.20)
    ap.add_argument("--align-tol", type=float, default=6.0,
                    help="heading tolerance after the in-place turn (deg)")
    ap.add_argument("--align-wait", type=float, default=12.0,
                    help="max seconds to wait for the heading alignment")
    ap.add_argument("--settle-radius", type=float, default=0.35)
    ap.add_argument("--settle-move", type=float, default=0.02)
    ap.add_argument("--settle-yaw", type=float, default=0.05,
                    help="max yaw span (rad) over the settle window")
    ap.add_argument("--settle-time", type=float, default=1.0)
    ap.add_argument("--cam-offset-fwd", type=float, default=0.190)
    ap.add_argument("--cam-offset-left", type=float, default=0.026)
    ap.add_argument("--topic", default="/control/target_pose")
    ap.add_argument("--truth", default="/camera/camera/vio_100hz")
    args = ap.parse_args()

    p = load_params()
    import rclpy
    rclpy.init()
    c = Cycle(args)
    print(f"floorplan {p.summary()}", flush=True)
    try:
        t0 = time.monotonic()
        while rclpy.ok() and c.pose() is None:
            if time.monotonic() - t0 > 20:
                print("no vio pose", flush=True)
                return 2
            c.spin(0.1)
        print(f"start baselink ({c.pose()[0]:.2f},{c.pose()[1]:.2f})", flush=True)
        failed = 0
        for i in range(args.cycles):
            cw = (args.mode != "up") if args.mode != "alternate" else i % 2 == 0
            legs = floorplan_loop(p, clockwise=cw)
            print(f"[lap {i+1}] {'CW/down' if cw else 'CCW/up'}", flush=True)
            for wp in legs:
                ok, dist, dt, why = c.run_leg(wp)
                print(f"  {wp['name']} {'OK ' if ok else 'FAIL'} {dt:6.1f}s "
                      f"dist={dist:.2f} ({why})", flush=True)
                if not ok:
                    failed += 1
                    break
                c.dwell(args.dwell)
        print(f"done, failures={failed}", flush=True)
        return 1 if failed else 0
    finally:
        c.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
