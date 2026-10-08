"""rclpy transport face: publishes the looper contract as native ROS 2
messages straight into the DDS domain. Node identity is real: the node is
named `insight_full`, which is what pilot's readiness check greps for.
Runs inside the overall-sim container (jazzy, py3.12); rclpy is imported
lazily so hosts without ROS can still import this package.

QoS mirrors the oracle sampling (tinynav-sim/docs/looper-contract.md):
RELIABLE everywhere except imu (BEST_EFFORT); vio_status/tf_static
TRANSIENT_LOCAL. The host loop stamps every frame message with the wall
time at which the pose was sampled; vio_100hz/imu carry their own substep
sample stamps.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import (BASELINE, CONTRACT_H, CONTRACT_W, Face, FRAMES, POSE_FRAME,
               TOPICS, VIO_STATUS, mat_to_quat_xyzw)


class RosCmdSource:
    """/cmd_vel subscription with hold semantics: latest twist wins, 0.5 s
    silence watchdog zeroes."""

    WATCHDOG_S = 0.5

    def __init__(self, node) -> None:
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        self._lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._t_last = 0.0
        self._clipped = 0
        node.create_subscription(
            __import__("geometry_msgs.msg", fromlist=["Twist"]).Twist,
            TOPICS["cmd_vel"], self._on_cmd,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def _on_cmd(self, msg) -> None:
        vx = float(np.clip(msg.linear.x, -0.5, 1.5))
        vy = float(np.clip(msg.linear.y, -0.5, 0.5))
        wz = float(np.clip(msg.angular.z, -1.2, 1.2))
        with self._lock:
            self._cmd = np.asarray((vx, vy, wz), dtype=np.float32)
            self._t_last = time.monotonic()
            if (vx, vy, wz) != (msg.linear.x, msg.linear.y, msg.angular.z):
                self._clipped += 1

    def vector(self) -> np.ndarray:
        with self._lock:
            if self._t_last and time.monotonic() - self._t_last > self.WATCHDOG_S:
                self._cmd = np.zeros(3, dtype=np.float32)
                self._t_last = 0.0
            return self._cmd

    def zero(self) -> None:
        with self._lock:
            self._cmd = np.zeros(3, dtype=np.float32)
            self._t_last = 0.0

    def consume_reset(self) -> bool:
        return False

    @property
    def clipped(self) -> int:
        return self._clipped

    def start(self):
        return self

    def stop(self) -> None:
        pass


def _stamp_msg(header, t: float) -> None:
    header.stamp.sec = int(t)
    header.stamp.nanosec = int((t - int(t)) * 1e9)


class RosFace(Face):
    """The Face protocol over rclpy publishers. The fast thread drains the
    TelemetrySink; rclpy.spin runs on its own thread so /cmd_vel callbacks
    fire while the control loop blocks."""

    def __init__(self, sink, vio_hz: float, imu_every: int) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = Node("insight_full")

        img_qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST)
        pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        fast_qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        sensor_be = QoSProfile(depth=50,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST)

        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, Image
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage

        self.msg_types = {"image": Image, "cam": CameraInfo,
                          "pose": PoseStamped, "comp": CompressedImage,
                          "imu": Imu, "str": String, "tf": TFMessage}

        def pub(topic, mtype, qos):
            return self.node.create_publisher(mtype, topic, qos)

        K = "/camera/camera"
        self.pub = {
            "infra1": pub(f"{K}/infra1/image_rect_raw", Image, img_qos),
            "infra2": pub(f"{K}/infra2/image_rect_raw", Image, img_qos),
            "depth": pub(f"{K}/depth/image_rect_raw", Image, img_qos),
            "color": pub(f"{K}/color/image_rect_raw/compressed",
                         CompressedImage, img_qos),
            "vio_image": pub(f"{K}/vio_image", PoseStamped, pose_qos),
            "vio_100hz": pub(f"{K}/vio_100hz", PoseStamped, fast_qos),
            "vio_status": pub(f"{K}/vio_status", String, latched),
            "ci1": pub(f"{K}/infra1/camera_info", CameraInfo, pose_qos),
            "ci2": pub(f"{K}/infra2/camera_info", CameraInfo, pose_qos),
            "ci_color": pub(f"{K}/color/camera_info", CameraInfo, pose_qos),
            "tf": pub("/tf_static", TFMessage, latched),
            "imu": pub(f"{K}/imu", Imu, sensor_be),
        }

        self.cmd = RosCmdSource(self.node)
        self._sink = sink
        self._vio_period = 1.0 / max(vio_hz, 1.0)
        self._imu_every = max(int(imu_every), 1)
        self._stop = threading.Event()
        self.n_imu = 0
        self.n_vio = 0
        self._spin = threading.Thread(target=self._spin_node, daemon=True,
                                      name="ros-spin")
        self._spin.start()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ros-fast-pub")
        self._thread.start()

    def _spin_node(self) -> None:
        try:
            self._rclpy.spin(self.node)
            print("[ros-spin] spin returned", flush=True)
        except Exception as exc:   # noqa: BLE001 -- surfaced, never swallowed
            print(f"[ros-spin] exited: {exc!r}", flush=True)

    def _run(self) -> None:
        next_vio = time.perf_counter()
        imu_i = 0
        while not self._stop.is_set():
            n = 0
            for t, gyro, acc, quat in self._sink.drain_imu():
                imu_i += 1
                if imu_i % self._imu_every:
                    continue
                self.pub["imu"].publish(self._imu_msg(gyro, acc, quat, t))
                n += 1
            self.n_imu += n
            now = time.perf_counter()
            if now >= next_vio:
                t, pos, quat = self._sink.latest_pose()
                if t > 0.0:
                    R, tv = self._sink.cam_pose_world(pos, quat)
                    self.pub["vio_100hz"].publish(
                        self._pose_msg(tv, mat_to_quat_xyzw(R), t))
                    self.n_vio += 1
                next_vio = max(next_vio + self._vio_period, time.perf_counter())
            if n == 0:
                time.sleep(0.002)

    # -- Face protocol ---------------------------------------------------
    def publish_frame(self, stamp, gray_l, gray_r, depth, pose_R, pose_t,
                      jpg=None) -> None:
        self.pub["infra1"].publish(
            self._image(gray_l, "mono8", FRAMES["infra1"], stamp))
        self.pub["infra2"].publish(
            self._image(gray_r, "mono8", FRAMES["infra2"], stamp))
        self.pub["depth"].publish(
            self._image(depth, "mono16", FRAMES["depth"], stamp))
        self.pub["vio_image"].publish(self._pose_msg(pose_t, pose_R, stamp))
        if jpg is not None:
            m = self.msg_types["comp"]()
            _stamp_msg(m.header, stamp)
            m.header.frame_id = FRAMES["color"]
            m.format = "jpeg"
            # Humble's generated asserts want a sequence of ints: bytes works,
            # numpy uint8 arrays do not
            m.data = jpg if isinstance(jpg, bytes) else bytes(jpg)
            self.pub["color"].publish(m)

    def publish_camera_info(self, stamp, fx, cx, cy) -> None:
        for key, frame, right in (("ci1", FRAMES["infra1"], False),
                                  ("ci2", FRAMES["infra2"], True),
                                  ("ci_color", FRAMES["color"], False)):
            m = self.msg_types["cam"]()
            _stamp_msg(m.header, stamp)
            m.header.frame_id = frame
            m.height, m.width = CONTRACT_H, CONTRACT_W
            m.distortion_model = "plumb_bob"
            m.d = [0.0] * 5
            tx = -float(fx) * BASELINE if right else 0.0
            m.k = [float(fx), 0.0, float(cx), 0.0, float(fx), float(cy),
                   0.0, 0.0, 1.0]
            m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            m.p = [float(fx), 0.0, float(cx), tx,
                   0.0, float(fx), float(cy), 0.0,
                   0.0, 0.0, 1.0, 0.0]
            m.binning_x = m.binning_y = 0
            m.roi.do_rectify = False
            self.pub[key].publish(m)

    def publish_tf(self, t) -> None:
        from . import ORACLE_TF_STATIC
        m = self.msg_types["tf"]()
        for parent, child, trans, rot in ORACLE_TF_STATIC:
            from geometry_msgs.msg import TransformStamped
            ts = TransformStamped()
            _stamp_msg(ts.header, t)
            ts.header.frame_id = parent
            ts.child_frame_id = child
            ts.transform.translation.x, ts.transform.translation.y, \
                ts.transform.translation.z = trans
            ts.transform.rotation.x, ts.transform.rotation.y, \
                ts.transform.rotation.z, ts.transform.rotation.w = rot
            m.transforms.append(ts)
        self.pub["tf"].publish(m)

    def publish_status(self, t) -> None:
        m = self.msg_types["str"]()
        m.data = VIO_STATUS
        self.pub["vio_status"].publish(m)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cmd.stop()
        try:
            self.node.destroy_node()
            self._rclpy.shutdown(context=self.node.context)
        except Exception:   # noqa: BLE001
            pass

    # -- message builders --------------------------------------------------
    def _image(self, arr: np.ndarray, encoding: str, frame: str, t: float):
        m = self.msg_types["image"]()
        _stamp_msg(m.header, t)
        m.header.frame_id = frame
        m.height, m.width = arr.shape[:2]
        m.encoding = encoding
        m.is_bigendian = 0
        m.step = arr.shape[1] * (2 if encoding == "mono16" else 1)
        m.data = np.ascontiguousarray(arr).tobytes()
        return m

    def _pose_msg(self, pos, quat_xyzw, t: float):
        m = self.msg_types["pose"]()
        _stamp_msg(m.header, t)
        m.header.frame_id = POSE_FRAME
        pos = np.asarray(pos, dtype=np.float64).ravel()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = \
            float(pos[0]), float(pos[1]), float(pos[2])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, \
            m.pose.orientation.w = (float(v) for v in quat_xyzw)
        return m

    def _imu_msg(self, gyro, acc, quat_wxyz, t: float):
        w, x, y, z = (float(v) for v in quat_wxyz)   # MuJoCo qpos order is wxyz
        m = self.msg_types["imu"]()
        _stamp_msg(m.header, t)
        m.header.frame_id = FRAMES["imu"]
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = \
            x, y, z, w
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = \
            (float(v) for v in gyro)
        m.linear_acceleration.x, m.linear_acceleration.y, \
            m.linear_acceleration.z = (float(v) for v in acc)
        return m
