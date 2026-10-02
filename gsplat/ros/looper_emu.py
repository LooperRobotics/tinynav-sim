#!/usr/bin/env python3
"""Looper-contract emulator: gsplat sensor ring -> the REAL looper's DDS face.

HIL route B (docs/hil-looper-emu-plan.md): the x86 workstation impersonates the
looper camera box so the navcore's pilot stack runs UNCHANGED. This node reads
the gsplat shared-memory ring directly (not the gz-parity ROS face -- running
gs_ros_bridge.py alongside would double-publish /camera/camera/infra1/* with
sim-time stamps and break the bridge's ExactTime sync) and publishes the
contract sampled from a live looper (docs/looper-contract.md):

    /camera/camera/infra1/image_rect_raw   Image        mono8    per camera frame
    /camera/camera/depth/image_rect_raw    Image        mono16mm per camera frame
    /camera/camera/vio_image               PoseStamped  T_world_camera, SAME stamp
    /camera/camera/vio_100hz               PoseStamped  T_world_camera @ ~100 Hz
    /camera/camera/vio_status              String       TRACKING_GOOD, latched 1 Hz
    /camera/camera/infra1/camera_info      CameraInfo   20 Hz
    /camera/camera/infra2/camera_info      CameraInfo   20 Hz
    /tf_static                             TFMessage    oracle transforms, latched

Hard requirements baked in (see the contract doc for the why):
  * node name MUST be `insight_full` -- pilot's sensor_source_ready() waits
    for exactly this name in the ROS graph;
  * depth / vio_image / infra1 carry IDENTICAL stamps (looper_bridge uses
    message_filters.TimeSynchronizer = ExactTime; unequal stamps = 0 hits);
  * stamps are x86 wall clock (the real looper stamps from its own boot clock;
    nothing on the navcore reasons about absolute stamp values under the
    GT-pose route, monotonicity + triple equality is all that is consumed);
  * QoS mirrored from the oracle (RELIABLE everywhere; vio_status/tf_static
    TRANSIENT_LOCAL).

vio_100hz freshness: camera frames only come at the render rate (~12-15 Hz),
so between frames the base-link GT (50 Hz ring slot) is composed with the
camera mount T_base_cam learned from the first frames (the rig is rigid; the
robot is still at startup, so the gt/cam pairing skew is harmless).

Also subscribes /cmd_vel and mirrors it into the simulator's cmd file (what
gs_ros_bridge.py does for the gz-parity face) -- the whole HIL control loop:
pilot -> /cmd_vel -> here -> cmd file -> RL policy -> sim dog -> ring -> here.

Usage (inside the tinynav container, DDS whitelisted onto the USB link):
    source /opt/ros/humble/setup.bash
    FASTDDS_DEFAULT_PROFILES_FILE=<x86 hil xml> python3 gsplat/ros/looper_emu.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

SERVER_DIR = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, SERVER_DIR.as_posix())
from gs_sensor_ring import SensorRingReader, DEFAULT_PATH, DEFAULT_CMD_FILE  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (QoSProfile, ReliabilityPolicy,  # noqa: E402
                       DurabilityPolicy, HistoryPolicy)
from geometry_msgs.msg import Twist, PoseStamped  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, CompressedImage  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402

# --------------------------------------------------------------------------- #
# oracle constants (docs/looper-contract.md; sampled from the live looper)
# --------------------------------------------------------------------------- #
FX = 302.77899169921875          # oracle fy=fx; keep in sync with sensor_server --cam-fy
BASELINE = 0.0996236577630043    # oracle tf_static left->right x

FRAMES = {
    "infra1": "camera_camera_left",
    "infra2": "camera_camera_right",
    "depth": "camera_camera_depth",
    "color": "camera_camera_rgb",
}
POSE_FRAME = "world"             # vio_image / vio_100hz frame_id (oracle)

# the five oracle /tf_static transforms, verbatim
ORACLE_TF_STATIC = [
    ("camera_camera_left", "camera_camera_depth", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ("camera_camera_left", "camera_camera_imu_optical", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ("camera_camera_left", "camera_camera_right", (BASELINE, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ("camera_camera_left", "camera_camera_rgb",
     (0.0507298048449633, -0.0009218509497213958, 0.0004383270107799681),
     (-0.0037471051339531022, -0.001724894113248995, 0.0005698719111198014, 0.9999913295571211)),
    ("camera_camera_imu", "camera_camera_left",
     (-0.03977611170763239, -0.02518247779983906, 0.027494611044653186), (0.0, 0.0, 0.0, 1.0)),
]


def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n < 1e-12 else 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ])


def mat_to_quat_xyzw(R: np.ndarray):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rigid_mul(a, b):
    """a @ b for rigid (R|t) pairs given as (R 3x3, t 3)."""
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def rigid_inv(a):
    R, t = a
    return R.T, -R.T @ t


class LooperEmu(Node):
    def __init__(self, ring_path: str, cmd_file: str | None, fx: float,
                 vio_hz: float, calib_frames: int = 10):
        # THE hard requirement: pilot's sensor_source_ready() greps for this
        # exact node name (pilot/backend/ops/sensor.py).
        super().__init__("insight_full")
        self.fx = fx
        self.calib_frames = calib_frames

        # depth 200: 12Hz 下 ~17s 的 writer 历史 —— 重传恢复窗口必须远大于
        # RTT×重传轮数,否则样本被覆盖成永久丢帧(实测 20→200:
        # 录制帧率 40%→满帧)
        img_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        fast_qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        latched_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST)

        self.pub_infra1 = self.create_publisher(Image, "/camera/camera/infra1/image_rect_raw", img_qos)
        self.pub_depth = self.create_publisher(Image, "/camera/camera/depth/image_rect_raw", img_qos)
        self.pub_vio_image = self.create_publisher(PoseStamped, "/camera/camera/vio_image", pose_qos)
        self.pub_vio_100hz = self.create_publisher(PoseStamped, "/camera/camera/vio_100hz", fast_qos)
        self.pub_vio_status = self.create_publisher(String, "/camera/camera/vio_status", latched_qos)
        self.pub_ci1 = self.create_publisher(CameraInfo, "/camera/camera/infra1/camera_info", pose_qos)
        self.pub_ci2 = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", pose_qos)
        # 非契约但建图必需:pilot recorder 录它、build_map_node 的 4 路同步
        # (keyframe_image/odom/depth + color image_raw) 吃它——ImageTransportsNode
        # 从压缩流转 raw,走生产同款路径(oracle tf_static: left→rgb 外参存在)
        self.pub_color_jpg = self.create_publisher(
            CompressedImage, "/camera/camera/color/image_rect_raw/compressed", img_qos)
        self.pub_color_ci = self.create_publisher(
            CameraInfo, "/camera/camera/color/camera_info", pose_qos)
        self.pub_tf_static = self.create_publisher(TFMessage, "/tf_static", latched_qos)

        self.cmd_file = cmd_file
        if cmd_file:
            self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, pose_qos)

        self.create_timer(1.0 / max(vio_hz, 1.0), self.publish_vio_100hz)
        self.create_timer(1.0, self.publish_vio_status)
        self.create_timer(5.0, self.publish_tf_static)

        self._tf_msg = self._build_tf_static()
        self._ci1, self._ci2 = None, None          # filled once the ring gives w/h
        self._mount_sum = np.zeros((3, 4))          # learned T_base_camera
        self._mount_n = 0
        self._mount = None                          # (R, t) frozen after calib_frames
        self._last_base = None                      # latest (R, t) world->base from gt slot
        self._last_cam = None                       # latest (pos, quat) from camera frames
        self.n_frames = self.n_vio = 0
        self.t0 = time.monotonic()

        while True:                                 # the simulator may still be loading
            try:
                self.ring = SensorRingReader(ring_path)
                break
            except Exception as exc:
                self.get_logger().warn(f"waiting for sensor ring {ring_path}: {exc}")
                time.sleep(1.0)
        self.get_logger().info(
            f"ring {ring_path} ({self.ring.w}x{self.ring.h}); looper contract face "
            f"fx={fx:.3f}; node insight_full" + (f"; /cmd_vel -> {cmd_file}" if cmd_file else ""))
        self.publish_tf_static()
        self.publish_vio_status()

    # ------------------------------------------------------------------ #
    def on_cmd_vel(self, msg: Twist):
        if not self.cmd_file:
            return
        try:
            # atomic replace (see gs_ros_bridge.py: O_TRUNC race vs the 20 ms poll)
            tmp = self.cmd_file + ".tmp"
            with open(tmp, "w") as f:
                f.write(f"{msg.linear.x:.4f} {msg.linear.y:.4f} {msg.angular.z:.4f}\n")
            os.replace(tmp, self.cmd_file)
        except OSError as exc:
            self.get_logger().warn(f"cmd file write failed: {exc}")

    def _build_tf_static(self) -> TFMessage:
        m = TFMessage()
        for parent, child, trans, rot in ORACLE_TF_STATIC:
            t = TransformStamped()
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = trans
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, \
                t.transform.rotation.w = rot
            m.transforms.append(t)
        return m

    def publish_tf_static(self):
        self._tf_msg.transforms[0].header.stamp = self.get_clock().now().to_msg()
        for t in self._tf_msg.transforms[1:]:
            t.header.stamp = self._tf_msg.transforms[0].header.stamp
        self.pub_tf_static.publish(self._tf_msg)

    def publish_vio_status(self):
        s = String()
        s.data = "TRACKING_GOOD"
        self.pub_vio_status.publish(s)

    def _pose_msg(self, pos, quat_xyzw, stamp=None) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        m.header.frame_id = POSE_FRAME
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = \
            (float(v) for v in quat_xyzw)
        return m

    def publish_vio_100hz(self):
        """~100 Hz cadence: freshest camera pose available.

        Between camera frames (render is ~12-15 Hz) that is base-GT (50 Hz slot)
        composed with the learned mount; falls back to the last frame pose
        until the mount is calibrated."""
        if self._mount is not None and self._last_base is not None:
            R, t = rigid_mul(self._last_base, self._mount)
            pos, quat = t, mat_to_quat_xyzw(R)
        elif self._last_cam is not None:
            pos, quat = self._last_cam
        else:
            return
        self.pub_vio_100hz.publish(self._pose_msg(pos, quat))
        self.n_vio += 1

    def _cam_info(self, frame: str, right: bool) -> CameraInfo:
        m = CameraInfo()
        m.header.frame_id = frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.height, m.width = self.ring.h, self.ring.w
        m.distortion_model = "plumb_bob"
        m.d = [0.0] * 5
        cx, cy = self.ring.w / 2.0, self.ring.h / 2.0   # batch_render's centred model
        m.k = [self.fx, 0.0, cx, 0.0, self.fx, cy, 0.0, 0.0, 1.0]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        tx = -self.fx * BASELINE if right else 0.0
        m.p = [self.fx, 0.0, cx, tx, 0.0, self.fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        m.binning_x = m.binning_y = 0
        m.roi.do_rectify = False
        return m

    def _image(self, arr: np.ndarray, encoding: str, frame: str, stamp) -> Image:
        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = frame
        img.height, img.width = arr.shape[0], arr.shape[1]
        img.encoding = encoding
        img.is_bigendian = 0
        img.step = arr.shape[1] * (2 if encoding == "mono16" else 1)
        img.data.frombytes(np.ascontiguousarray(arr).tobytes())   # bulk C copy
        return img

    def tick(self) -> bool:
        """One poll: new camera frame -> same-stamp triple + camera_info."""
        gt = self.ring.read_gt()
        if gt is not None:
            self._last_base = (quat_xyzw_to_mat(gt.quat), np.asarray(gt.pos, float))
        frame = self.ring.read_cameras()
        if frame is None:
            return False
        stamp = self.get_clock().now().to_msg()          # ONE stamp for the triple

        # mount calibration: robot is still at startup, so gt-slot vs camera-tick
        # skew (~<=20 ms) is harmless; freeze after calib_frames.
        if self._mount is None and self._last_base is not None:
            T_wb = self._last_base
            T_wc = (quat_xyzw_to_mat(frame.cam_quat), np.asarray(frame.cam_pos, float))
            R_bc, t_bc = rigid_mul(rigid_inv(T_wb), T_wc)
            self._mount_sum += np.hstack([R_bc, t_bc.reshape(3, 1)])
            self._mount_n += 1
            if self._mount_n >= self.calib_frames:
                avg = self._mount_sum / self._mount_n
                self._mount = (avg[:, :3], avg[:, 3])
                self.get_logger().info(
                    f"mount T_base_camera learned from {self._mount_n} frames: "
                    f"t={np.round(self._mount[1], 4).tolist()}")

        self._last_cam = (frame.cam_pos, frame.cam_quat)

        # the ExactTime triple -- identical stamps, no exceptions
        self.pub_infra1.publish(self._image(frame.infra1, "mono8", FRAMES["infra1"], stamp))
        self.pub_depth.publish(self._image(frame.depth, "mono16", FRAMES["depth"], stamp))
        self.pub_vio_image.publish(self._pose_msg(frame.cam_pos, frame.cam_quat, stamp))

        self.pub_ci1.publish(self._cam_info(FRAMES["infra1"], right=False))
        self.pub_ci2.publish(self._cam_info(FRAMES["infra2"], right=True))
        if frame.color.any():              # server 没渲彩色时 ring 给全零,跳过
            import cv2
            jpg = CompressedImage()
            jpg.header.stamp = stamp
            jpg.header.frame_id = FRAMES["color"]
            jpg.format = "jpeg"
            jpg.data.frombytes(
                cv2.imencode(".jpg", frame.color,
                             [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
            self.pub_color_jpg.publish(jpg)
            self.pub_color_ci.publish(self._cam_info(FRAMES["color"], right=False))
        self.n_frames += 1
        return True

    def report(self):
        wall = time.monotonic() - self.t0
        self.get_logger().info(
            f"{self.n_frames} contract frames ({self.n_frames/wall:.1f} Hz), "
            f"{self.n_vio} vio_100hz ({self.n_vio/wall:.1f} Hz) in {wall:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", default=DEFAULT_PATH)
    ap.add_argument("--cmd-file", default=DEFAULT_CMD_FILE,
                    help="simulator reads vx,vy,wz from here; '' disables /cmd_vel")
    ap.add_argument("--fx", type=float, default=FX,
                    help="fx=fy to declare in camera_info; MUST equal the server's "
                         "--cam-fy (batch_render's centred model)")
    ap.add_argument("--vio-hz", type=float, default=100.0)
    ap.add_argument("--report-every", type=float, default=15.0)
    args = ap.parse_args()

    rclpy.init()
    node = LooperEmu(args.ring, args.cmd_file or None, args.fx, args.vio_hz)
    t_last = time.monotonic()
    try:
        while rclpy.ok():
            t0 = time.perf_counter()
            rclpy.spin_once(node, timeout_sec=0.0)
            node.tick()
            time.sleep(max(0.0, 0.002 - (time.perf_counter() - t0)))   # ~500 Hz poll
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
        except Exception as exc:
            print(f"shutdown: {exc}", flush=True)


if __name__ == "__main__":
    main()
