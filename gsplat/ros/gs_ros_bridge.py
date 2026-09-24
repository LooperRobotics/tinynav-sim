#!/usr/bin/env python3
"""ROS 2 bridge: shared-memory sensor ring -> /camera/camera/... topics.

Ported from gs_playground/demo/navigation/ros/gs_ros_bridge.py (bring-up era).
Now reads ring v2 and additionally republishes
the ground-truth base pose as /sim/gt_pose (nav_msgs/Odometry, MJCF world
frame) -- the gsplat counterpart of gz's /world/<w>/dynamic_pose/info that
gazebo/dog_state.sh and sim_gt_reloc consume.

Publishes exactly the sensor face that `gazebo/robots/go2/go2_tinynav.xacro`
+ `gazebo/scene/camera_info_publisher.py` define, so the tinynav stack cannot
tell this simulator apart from gz:

    /camera/camera/infra1/image_rect_raw   sensor_msgs/Image  mono8  544x480
    /camera/camera/infra2/image_rect_raw   sensor_msgs/Image  mono8  544x480
    /camera/camera/color/image_raw         sensor_msgs/Image  rgb8   544x480
    /camera/camera/{infra1,infra2,color}/camera_info  sensor_msgs/CameraInfo
    /camera/camera/imu                     sensor_msgs/Imu           200 Hz
    /sim/gt_pose                           nav_msgs/Odometry          50 Hz
    /clock                                 rosgraph_msgs/Clock
and subscribes /cmd_vel -> writes vx,vy,wz into --cmd-file for the simulator.

Usage (inside the container):
    source /opt/ros/humble/setup.bash
    python3 gsplat/ros/gs_ros_bridge.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
import zlib
import time
from pathlib import Path

import numpy as np

SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, SERVER_DIR.as_posix())
from gs_sensor_ring import SensorRingReader, DEFAULT_PATH, DEFAULT_CMD_FILE  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, Imu  # noqa: E402

FX = FY = 272.0
CX, CY = 272.0, 240.0
BASELINE = 0.051
D435I_STD = dict(gyro=2.0e-4, accel=2.0e-3, quat=1.0e-3)   # plausible D435i covariances

FRAMES = {
    "infra1": "vehicle_blue/infra1_link/infra1",
    "infra2": "vehicle_blue/infra2_link/infra2",
    "color": "vehicle_blue/color_link/color",
}
IMU_FRAME = "vehicle_blue/camera_imu_link"      # matches the gz xacro link name
GT_FRAME = "gs_world"                           # MJCF world == gazebo's world analog
GT_CHILD = "base"


def cam_info(frame: str, right: bool = False) -> CameraInfo:
    m = CameraInfo()
    m.header.frame_id = frame
    m.height, m.width = 480, 544
    m.distortion_model = "plumb_bob"
    m.d = [0.0] * 5
    m.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    tx = -FX * BASELINE if right else 0.0
    m.p = [FX, 0.0, CX, tx, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
    m.binning_x = m.binning_y = 0
    m.roi.do_rectify = False
    return m


def stamp(sec: float) -> "object":
    from builtin_interfaces.msg import Time
    t = Time()
    t.sec = int(sec)
    t.nanosec = int((sec - int(sec)) * 1e9)
    return t


class Bridge(Node):
    def __init__(self, ring_path: str, cmd_file: str | None,
                 dump_dir: str | None = None, dump_n: int = 3):
        super().__init__("gs_playground_sensor_bridge")
        self.dump_dir = dump_dir
        self.dump_n = dump_n
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        clock_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST)
        self.pub = {
            "infra1": self.create_publisher(Image, "/camera/camera/infra1/image_rect_raw", qos),
            "infra2": self.create_publisher(Image, "/camera/camera/infra2/image_rect_raw", qos),
            "color": self.create_publisher(Image, "/camera/camera/color/image_raw", qos),
        }
        self.pub_info = {
            "infra1": self.create_publisher(CameraInfo, "/camera/camera/infra1/camera_info", qos),
            "infra2": self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", qos),
            "color": self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", qos),
        }
        self.pub_imu = self.create_publisher(Imu, "/camera/camera/imu", qos)
        self.pub_gt = self.create_publisher(Odometry, "/sim/gt_pose", qos)
        self.pub_clock = self.create_publisher(Clock, "/clock", clock_qos)
        # GT is a STATE, not a stream: publish the ring slot's current content
        # on a steady 50 Hz timer. The server writes the slot at 50 Hz sim time,
        # but bursts it during physics catch-up right after each ~55 ms render
        # stall -- a seq-gated "publish only new" reader coalesces those bursts
        # (measured 18 Hz instead of 50). Latest-value-at-fixed-rate has no
        # such dependency on poll cadence.
        self._gt_pub_seq = 0
        self.create_timer(0.02, self.publish_gt)
        self.info = {k: cam_info(FRAMES[k], right=(k == "infra2")) for k in ("infra1", "infra2", "color")}
        self.cmd_file = cmd_file
        if cmd_file:
            self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, qos)
        while True:                     # the simulator may still be loading: wait for the ring
            try:
                self.ring = SensorRingReader(ring_path)
                break
            except Exception as exc:
                self.get_logger().warn(f"waiting for sensor ring {ring_path}: {exc}")
                _time.sleep(1.0)
        self.n_cam = self.n_imu = self.n_gt = self.n_img = 0
        self.t_img = self.t_imu = self.t_imu_all = self.t_gt = 0.0
        self.t_spin = self.t_readcam = self.t_build = 0.0
        self.max_img = 0.0
        self.n_loop = 0
        self.t_start = time.monotonic()
        self.t_last = self.t_start
        self.get_logger().info(
            f"ring {ring_path} ({self.ring.w}x{self.ring.h}); publishing "
            f"infra1/infra2/color + camera_info + imu + gt_pose"
            + (f"; /cmd_vel -> {cmd_file}" if cmd_file else ""))

    # ------------------------------------------------------------------ #
    def on_cmd_vel(self, msg: Twist):
        if not self.cmd_file:
            return
        try:
            # atomic replace: a plain write_text is O_TRUNC-then-write, and the
            # simulator polling every 20 ms can read the truncated-but-empty
            # file in between (crashed the server within a minute of closed-loop
            # cmd_vel; bring-up's manual pokes never hit the window)
            tmp = self.cmd_file + ".tmp"
            with open(tmp, "w") as f:
                f.write(f"{msg.linear.x:.4f} {msg.linear.y:.4f} {msg.angular.z:.4f}\n")
            os.replace(tmp, self.cmd_file)
        except OSError as exc:
            self.get_logger().warn(f"cmd file write failed: {exc}")

    def tick(self) -> bool:
        did = False
        t0 = time.perf_counter()
        frame = self.ring.read_cameras()
        self.t_readcam += time.perf_counter() - t0
        if frame is not None:
            t = stamp(frame.sim_time)
            for name, arr in (("infra1", frame.infra1), ("infra2", frame.infra2),
                              ("color", frame.color)):
                img = Image()
                img.header.stamp = t
                img.header.frame_id = FRAMES[name]
                img.height, img.width = arr.shape[0], arr.shape[1]
                img.encoding = "rgb8" if name == "color" else "mono8"
                img.is_bigendian = 0
                img.step = arr.shape[1] * (3 if name == "color" else 1)
                t0 = time.perf_counter()
                # NB: `img.data = bytes` costs ~10 ms (mono) / ~33 ms (rgb) in rclpy
                # because the sequence<uint8> is filled element by element.
                # array.frombytes is a bulk C copy: ~0.1 / ~0.5 ms.
                img.data.frombytes(np.ascontiguousarray(arr).tobytes())
                self.t_build += time.perf_counter() - t0
                t0 = time.perf_counter()
                self.pub[name].publish(img)
                dt = time.perf_counter() - t0
                self.t_img += dt
                self.max_img = max(self.max_img, dt)
                self.n_img += 1
                info = self.info[name]
                info.header.stamp = t
                self.pub_info[name].publish(info)
            clk = Clock()
            clk.clock = t
            self.pub_clock.publish(clk)
            if self.n_cam < self.dump_n:
                # self-check: finger-print the payload we are about to publish
                c = np.ascontiguousarray(frame.color)
                i1 = np.ascontiguousarray(frame.infra1)
                self.get_logger().info(
                    f"frame {frame.sim_time:.3f} crc color {zlib.crc32(c.tobytes()):08x} "
                    f"infra1 {zlib.crc32(i1.tobytes()):08x}")
            if self.dump_dir and self.n_cam < self.dump_n:
                d = Path(self.dump_dir)
                d.mkdir(parents=True, exist_ok=True)
                tag = f"{frame.sim_time:.3f}"
                for nm, ar in (("color", frame.color), ("infra1", frame.infra1)):
                    (d / f"bridge_{nm}_{tag}.npy").write_bytes(
                        np.ascontiguousarray(ar).tobytes())
            self.n_cam += 1
            did = True

        t0 = time.perf_counter()
        imu_batch = self.ring.read_imu()
        self.t_imu_all += time.perf_counter() - t0
        for s in imu_batch:
            m = Imu()
            m.header.stamp = stamp(s.t)
            m.header.frame_id = IMU_FRAME
            m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = [float(v) for v in s.quat]
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = [float(v) for v in s.gyro]
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = [float(v) for v in s.accel]
            for i in range(3):
                m.orientation_covariance[i * 4] = D435I_STD["quat"]
                m.angular_velocity_covariance[i * 4] = D435I_STD["gyro"]
                m.linear_acceleration_covariance[i * 4] = D435I_STD["accel"]
            t0 = time.perf_counter()
            self.pub_imu.publish(m)
            self.t_imu += time.perf_counter() - t0
            clk = Clock()
            clk.clock = m.header.stamp
            self.pub_clock.publish(clk)
            self.n_imu += 1
            did = True
        return did

    def publish_gt(self):
        """Timer callback (50 Hz): publish the GT slot's latest content."""
        import struct
        t0 = time.perf_counter()
        v = struct.unpack_from("<d3d4d", self.ring.buf, self.ring.o["gt_base"])
        seq = struct.unpack_from("<Q", self.ring.buf, 56)[0]
        if seq == 0:
            return
        o = Odometry()
        o.header.stamp = stamp(v[0])
        o.header.frame_id = GT_FRAME
        o.child_frame_id = GT_CHILD
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = \
            [float(x) for x in v[1:4]]
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, \
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = \
            [float(x) for x in v[4:8]]
        self.pub_gt.publish(o)
        self.t_gt += time.perf_counter() - t0
        self.n_gt += 1

    def report(self):
        img_ms = 1000 * self.t_img / max(self.n_img, 1)
        imu_ms = 1000 * self.t_imu / max(self.n_imu, 1)
        wall = time.monotonic() - self.t_start
        self.get_logger().info(
            f"published {self.n_cam} camera frames, {self.n_imu} imu samples, "
            f"{self.n_gt} gt poses in {wall:.1f}s "
            f"({self.n_cam/wall:.1f} cam Hz, {self.n_imu/wall:.1f} imu Hz, "
            f"{self.n_gt/wall:.1f} gt Hz) | "
            f"loop {self.n_loop/wall:.0f} Hz | per-image avg {img_ms:.2f} max {1000*self.max_img:.1f} ms, "
            f"per-imu {imu_ms:.3f} ms | phases ms: spin {1000*self.t_spin/max(self.n_loop,1):.2f} "
            f"readcam {1000*self.t_readcam/max(self.n_cam,1):.2f} build {1000*self.t_build/max(self.n_cam,1):.2f} "
            f"pubcam {1000*self.t_img/max(self.n_cam,1):.2f} imu {1000*self.t_imu_all/max(self.n_loop,1):.2f} "
            f"gt {1000*self.t_gt/max(self.n_loop,1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", default=DEFAULT_PATH)
    ap.add_argument("--cmd-file", default=DEFAULT_CMD_FILE,
                    help="simulator reads vx,vy,wz from here; '' disables /cmd_vel")
    ap.add_argument("--report-every", type=float, default=10.0)
    ap.add_argument("--dump-dir", default=None,
                    help="save the first --dump-n camera frames read from the ring (self-check)")
    ap.add_argument("--dump-n", type=int, default=3)
    args = ap.parse_args()

    rclpy.init()
    node = Bridge(args.ring, args.cmd_file or None, args.dump_dir, args.dump_n)
    t_last = time.monotonic()
    try:
        while rclpy.ok():
            t0 = time.perf_counter()
            rclpy.spin_once(node, timeout_sec=0.0)
            node.t_spin += time.perf_counter() - t0
            node.tick()
            node.n_loop += 1
            time.sleep(0.002)                      # poll the ring at ~500 Hz
            if time.monotonic() - t_last > args.report_every:
                t_last = time.monotonic()
                node.report()
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception as exc:      # already shut down by a signal handler
            print(f"shutdown: {exc}", flush=True)


if __name__ == "__main__":
    main()
