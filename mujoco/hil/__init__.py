"""HIL sensor face: shared contract constants, the Face protocol, and the
telemetry sampling slot.

One host loop (mujoco/hil.py) drives the transport face (hil.ros.RosFace).
This package stays importable on plain python3.10 + numpy (no
mujoco/wgpu/rclpy) so the ROS face can be exercised inside a stock ROS
container.
"""
from __future__ import annotations

import collections
import threading

import numpy as np

# --------------------------------------------------------------------- #
# sensor contract (tinynav-sim/docs/looper-contract.md oracle + extensions)
# --------------------------------------------------------------------- #
CONTRACT_W, CONTRACT_H = 544, 640
CONTRACT_FX = 302.77899169921875
BASELINE = 0.0996236577630043        # oracle tf_static left->right x
CONTRACT_FOVY_DEG = 93.168           # from CONTRACT_H and CONTRACT_FX

FRAMES = {
    "infra1": "camera_camera_left",
    "infra2": "camera_camera_right",
    "depth": "camera_camera_depth",
    "color": "camera_camera_rgb",
    "imu": "camera_camera_imu",
}
POSE_FRAME = "world"
VIO_STATUS = "TRACKING_GOOD"
TOPICS = {
    "infra1": f"/camera/camera/{'infra1'}/image_rect_raw",
    "infra2": "/camera/camera/infra2/image_rect_raw",
    "depth": "/camera/camera/depth/image_rect_raw",
    "color": "/camera/camera/color/image_rect_raw/compressed",
    "vio_image": "/camera/camera/vio_image",
    "vio_100hz": "/camera/camera/vio_100hz",
    "vio_status": "/camera/camera/vio_status",
    "imu": "/camera/camera/imu",
    "cmd_vel": "/cmd_vel",
}

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
        w, x, y, z = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rigid_mul(a, b):
    """a @ b for rigid (R|t) pairs."""
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def rigid_inv(a):
    R, t = a
    return R.T, -R.T @ t


def optical(T_wc_mjc):
    """MuJoCo camera frame (x right, y up, -z fwd) -> optical (z fwd, y down)."""
    R, t = T_wc_mjc
    return R @ np.diag([1.0, -1.0, -1.0]), t


# --------------------------------------------------------------------- #
# telemetry: substep samples -> fast transport threads
# --------------------------------------------------------------------- #
class TelemetrySink:
    """Consumed by the control thread (substep hook), drained by the face's
    fast publisher thread. IMU keeps every sample (bounded queue, drop-oldest
    counting); the pose slot keeps only the freshest sample for vio_100hz."""

    def __init__(self, gyro_adr: int, acc_adr: int, root_qpos_adr: int,
                 t_base_cam) -> None:
        self._gyro_adr = int(gyro_adr)
        self._acc_adr = int(acc_adr)
        self._root_qpos_adr = int(root_qpos_adr)
        self._t_base_cam = t_base_cam            # (R 3x3, t 3): base -> left cam
        self._imu_q: collections.deque = collections.deque(maxlen=1024)
        self._pose = (0.0, np.zeros(3), np.array((0.0, 0.0, 0.0, 1.0)))
        self._lock = threading.Lock()
        self.dropped = 0

    def sample(self, data) -> None:
        """substep hook: runs at the physics rate inside the control thread.

        `data` is an MjData (duck-typed: needs .sensordata and .qpos) so this
        module stays importable without mujoco."""
        import time
        t = time.time()
        gyro = data.sensordata[self._gyro_adr:self._gyro_adr + 3]
        acc = data.sensordata[self._acc_adr:self._acc_adr + 3]
        root = self._root_qpos_adr
        quat = data.qpos[root + 3:root + 7]
        pos = data.qpos[root:root + 3]
        if len(self._imu_q) == self._imu_q.maxlen:
            self.dropped += 1
        self._imu_q.append((t, gyro, acc, quat))
        with self._lock:
            self._pose = (t, pos, quat)

    def drain_imu(self):
        out = []
        while self._imu_q:
            out.append(self._imu_q.popleft())
        return out

    def latest_pose(self):
        with self._lock:
            return self._pose

    def cam_pose_world(self, base_pos, base_quat):
        """T_world_camera = T_world_base x T_base_camera (left optical)."""
        return rigid_mul((quat_xyzw_to_mat(base_quat), np.asarray(base_pos, float)),
                         self._t_base_cam)


class Face:
    """Transport face protocol. Implementations serialize + publish the
    contract over their wire; they also expose the command source protocol
    (vector/zero/start/stop) used by the control loop."""

    def vector(self) -> np.ndarray:
        """Current (vx, vy, wz) command fed to the policy."""
        raise NotImplementedError

    def zero(self) -> None:
        """Safety stop: zero the command now."""
        raise NotImplementedError

    def publish_frame(self, stamp, gray_l, gray_r, depth, pose_R, pose_t,
                      jpg=None) -> None:
        """One camera frame set, all messages stamped `stamp`."""
        raise NotImplementedError

    def publish_camera_info(self, stamp, fx, cx, cy) -> None:
        raise NotImplementedError

    def publish_tf(self, t) -> None:
        raise NotImplementedError

    def publish_status(self, t) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass
