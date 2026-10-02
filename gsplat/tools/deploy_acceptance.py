#!/usr/bin/env python3
"""Deployment-surface acceptance for a stairs policy checkpoint (Layer 3).

Runs the REAL deployment chain (onnx -> gs_ros_bridge -> cmd file ->
sensor_server -> MotrixSim physics) through a scripted /cmd_vel sequence
while recording the color stream and the GT trace. This is the layer that
caught the v12 freeze: training-env eval passed, deployment froze, because
the contact channels / PD rate / cmd-file path differ from training.

Run INSIDE the tinynav container, sim up (bash gsplat/run_gsplat.sh ...):
  /opt/venv/bin/python gsplat/tools/deploy_acceptance.py --out /tmp/accept_flat \
      --scenario flat
  /opt/venv/bin/python gsplat/tools/deploy_acceptance.py --out /tmp/accept_stairs \
      --scenario stairs
Prereqs: dog spawned at the scripted start pose (stairs scenario: an
up-flight mid-tread spawn, e.g. gsplat/tools/pick_spawn.py), supervisor on.

Artifacts: <out>/color.mp4, <out>/gt_trace.json, console summary
(distance, yaw turned, mean speed). No summary numbers -> no acceptance.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# (cmd (vx, wz), seconds) — mirrors record_scenarios.py's flat script.
SEGMENTS = {
    "flat": [
        ((0.0, 0.0), 3.0), ((0.4, 0.0), 5.0), ((0.0, 0.0), 2.0),
        ((0.0, 0.4), 4.0), ((0.0, -0.4), 4.0), ((-0.3, 0.0), 4.0),
        ((0.0, 0.0), 2.0),
    ],
    # stairs: slow fwd up the flight, pause, continue; caller spawns the dog
    # on an up-facing mid-tread.
    "stairs": [
        ((0.0, 0.0), 2.0), ((0.35, 0.0), 10.0), ((0.0, 0.0), 3.0),
        ((0.35, 0.0), 8.0), ((0.0, 0.0), 2.0),
    ],
}


class Acceptance(Node):
    def __init__(self, out_dir, segments, label):
        super().__init__("deploy_acceptance")
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self.on_image, qos)
        self.create_subscription(Odometry, "/sim/gt_pose", self.on_gt, qos)
        self.bridge = CvBridge()
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.out / "color.mp4"), fourcc, 15.0,
                                      (544, 480))
        self.gt = []
        self.segments = segments
        self.label = label
        self.t0 = time.monotonic()
        self.create_timer(0.05, self.tick)

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.writer.write(frame)

    def on_gt(self, msg):
        p = msg.pose.pose
        yaw = 2 * math.atan2(p.orientation.z, p.orientation.w)
        self.gt.append([time.monotonic() - self.t0, p.position.x, p.position.y,
                        p.position.z, yaw])

    def tick(self):
        el = time.monotonic() - self.t0
        acc = 0.0
        cmd = (0.0, 0.0)
        for c, secs in self.segments:
            acc += secs
            if el < acc:
                cmd = c
                break
        msg = Twist()
        msg.linear.x, msg.angular.z = cmd
        self.pub.publish(msg)
        if el > acc + 0.5:
            self.writer.release()
            self.summarize()
            # hard exit: rclpy.shutdown() from inside a timer callback leaves
            # the process hung in context teardown (observed:
            # artifacts written, python alive 10 min). The artifacts are the
            # deliverable; skip the clean teardown.
            rclpy.shutdown()
            os._exit(0)

    def summarize(self):
        g = np.asarray(self.gt)
        trace = self.out / "gt_trace.json"
        trace.write_text(json.dumps({"label": self.label, "gt": g.tolist()}))
        if len(g) < 10:
            print("FAIL: gt trace too short — /sim/gt_pose silent?")
            return
        dx = g[-1, 1] - g[0, 1]
        dy = g[-1, 2] - g[0, 2]
        dist = math.hypot(dx, dy)
        yaw_deg = math.degrees(g[-1, 4] - g[0, 4])
        dur = g[-1, 0] - g[0, 0]
        print(f"== deploy acceptance [{self.label}] ==")
        print(f"  duration {dur:.1f}s, gt samples {len(g)}")
        print(f"  displacement {dist:.2f} m, yaw {yaw_deg:+.1f} deg")
        print(f"  mean |speed| {dist / max(dur, 1e-6):.2f} m/s")
        print(f"  wrote {self.out / 'color.mp4'} and {trace}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", choices=sorted(SEGMENTS), default="flat")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    rclpy.init()
    node = Acceptance(a.out, SEGMENTS[a.scenario],
                      a.label or a.scenario)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
