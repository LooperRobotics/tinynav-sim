#!/usr/bin/env python3
"""Shared-memory ring for the gsplat simulator -> ROS bridge.

Written during bring-up (lived in gs_playground/demo/navigation/sensors/),
moved into tinynav-sim. Version 2 adds the ground-truth
channel (a single base-pose slot the simulator refreshes at the control rate)
so `gs_state.sh` and a future gs_gt_reloc have a truth source the way gz has
`/world/<w>/dynamic_pose/info`.

Both sides run in the tinynav container now; the ring stays because decoupling
the render loop from DDS pacing is still right (a slow subscriber must never
stretch the 15 Hz camera cadence) and the restart/torn-frame protocol is
already debugged.

Layout (little endian; header is 128 B, fixed byte offsets below):

    off  0  magic 'GSPG'      off 40  w          (u32)
    off  4  version  (u32=2)  off 44  h          (u32)
    off  8  cam_seq  (u64)    off 48  n_imu      (u32)
    off 16  imu_seq  (u64)    off 52  imu_stride (u32)
    off 24  cam_time (f64)    off 56  gt_seq     (u64)   [v2]
    off 32  imu_time (f64)    off 64  gt_time    (f64)   [v2]
                               off 128 cam slots

    cam slots : N_SLOT x [ infra1(w*h) | infra2(w*h) | color(w*h*3) ]   uint8
    imu ring  : N_IMU x [ t(f64) | gyro(3xf32) | accel(3xf32) | quat(4xf32) ]
    gt slot   : [ t(f64) | pos(3xf64) | quat(4xf64) ]  (64 B, single slot)

Publish protocol: fill the payload, then bump the seq (written last).  A
reader copies the payload, re-reads the seq and accepts only if unchanged ->
no torn frames, no locks.  Readers detect a writer restart via seq < last.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"GSPG"
VERSION = 2
HDR = 128
N_SLOT = 4
N_IMU = 512
IMU_STRIDE = 8 + 3 * 4 + 3 * 4 + 4 * 4  # 48 B
GT_STRIDE = 8 + 3 * 8 + 4 * 8          # 64 B
DEFAULT_PATH = "/dev/shm/gsplay_sensors.bin"
DEFAULT_CMD_FILE = "/dev/shm/gsplay_cmd.txt"

OFF_MAGIC, OFF_VERSION = 0, 4
OFF_CAM_SEQ, OFF_IMU_SEQ = 8, 16
OFF_CAM_TIME, OFF_IMU_TIME = 24, 32
OFF_W, OFF_H, OFF_NIMU, OFF_STRIDE = 40, 44, 48, 52
OFF_GT_SEQ, OFF_GT_TIME = 56, 64


def _layout(w: int, h: int, n_slot: int, n_imu: int) -> dict:
    slot = w * h * 2 + w * h * 3
    cam_base = HDR
    imu_base = HDR + slot * n_slot
    gt_base = imu_base + IMU_STRIDE * n_imu
    return {"w": w, "h": h, "slot": slot, "n_slot": n_slot, "n_imu": n_imu,
            "cam_base": cam_base, "imu_base": imu_base, "gt_base": gt_base,
            "size": gt_base + GT_STRIDE}


def ring_size(w: int, h: int, n_slot: int = N_SLOT, n_imu: int = N_IMU) -> int:
    return _layout(w, h, n_slot, n_imu)["size"]


# --------------------------------------------------------------------------- #
# writer (simulator side)
# --------------------------------------------------------------------------- #
class SensorRingWriter:
    def __init__(self, path: str = DEFAULT_PATH, w: int = 544, h: int = 480,
                 n_slot: int = N_SLOT, n_imu: int = N_IMU):
        self.path = path
        self.o = _layout(w, h, n_slot, n_imu)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(self.o["size"])
        self.buf = np.memmap(path, dtype=np.uint8, mode="r+", shape=(self.o["size"],))
        self.buf[OFF_MAGIC:OFF_MAGIC + 4] = np.frombuffer(MAGIC, np.uint8)
        struct.pack_into("<I", self.buf, OFF_VERSION, VERSION)
        struct.pack_into("<IIII", self.buf, OFF_W, w, h, n_imu, IMU_STRIDE)
        struct.pack_into("<dd", self.buf, OFF_CAM_TIME, 0.0, 0.0)
        struct.pack_into("<Q", self.buf, OFF_GT_SEQ, 0)
        struct.pack_into("<d", self.buf, OFF_GT_TIME, 0.0)
        self.cam_seq = 0
        self.imu_seq = 0
        self.gt_seq = 0
        struct.pack_into("<QQ", self.buf, OFF_CAM_SEQ, 0, 0)
        self.buf.flush()

    def publish_cameras(self, infra1: np.ndarray, infra2: np.ndarray,
                        color: np.ndarray, sim_time: float) -> None:
        o = self.o
        w, h = o["w"], o["h"]
        n1 = w * h
        base = o["cam_base"] + (self.cam_seq % o["n_slot"]) * o["slot"]
        assert infra1.nbytes == n1 and infra2.nbytes == n1 and color.nbytes == n1 * 3
        self.buf[base:base + n1] = infra1.reshape(-1)
        self.buf[base + n1:base + 2 * n1] = infra2.reshape(-1)
        self.buf[base + 2 * n1:base + 5 * n1] = color.reshape(-1)
        struct.pack_into("<d", self.buf, OFF_CAM_TIME, sim_time)
        self.cam_seq += 1
        struct.pack_into("<Q", self.buf, OFF_CAM_SEQ, self.cam_seq)   # publish

    def publish_imu(self, sim_time: float, gyro, accel, quat) -> None:
        o = self.o
        idx = self.imu_seq % o["n_imu"]
        p = o["imu_base"] + idx * IMU_STRIDE
        struct.pack_into("<7f4f", self.buf, p, sim_time, *gyro, *accel, *quat)
        struct.pack_into("<d", self.buf, OFF_IMU_TIME, sim_time)
        self.imu_seq += 1
        struct.pack_into("<Q", self.buf, OFF_IMU_SEQ, self.imu_seq)

    def publish_gt(self, sim_time: float, pos, quat) -> None:
        """Base-link ground truth (MJCF world frame, xyzw quaternion)."""
        p = self.o["gt_base"]
        struct.pack_into("<d3d4d", self.buf, p, sim_time, *pos, *quat)
        struct.pack_into("<d", self.buf, OFF_GT_TIME, sim_time)
        self.gt_seq += 1
        struct.pack_into("<Q", self.buf, OFF_GT_SEQ, self.gt_seq)

    def close(self, unlink: bool = False) -> None:
        self.buf.flush()
        del self.buf
        if unlink:
            try:
                os.unlink(self.path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# reader (ros bridge / gs_state side)
# --------------------------------------------------------------------------- #
@dataclass
class CameraFrame:
    seq: int
    sim_time: float
    infra1: np.ndarray
    infra2: np.ndarray
    color: np.ndarray


@dataclass
class ImuSample:
    t: float
    gyro: np.ndarray
    accel: np.ndarray
    quat: np.ndarray


@dataclass
class GtPose:
    t: float
    pos: np.ndarray
    quat: np.ndarray


class SensorRingReader:
    def __init__(self, path: str = DEFAULT_PATH):
        self.buf = np.memmap(path, dtype=np.uint8, mode="r")
        if bytes(self.buf[OFF_MAGIC:OFF_MAGIC + 4]) != MAGIC:
            raise RuntimeError(f"{path}: not a gs sensor ring")
        ver = struct.unpack_from("<I", self.buf, OFF_VERSION)[0]
        if ver != VERSION:
            raise RuntimeError(f"{path}: ring version {ver} != {VERSION}")
        self.w, self.h, n_imu, _stride, = struct.unpack_from("<IIII", self.buf, OFF_W)
        self.o = _layout(self.w, self.h, N_SLOT, n_imu)
        self._last_cam = 0
        self._last_imu = 0
        self._last_gt = 0

    def read_cameras(self) -> CameraFrame | None:
        seq = struct.unpack_from("<Q", self.buf, OFF_CAM_SEQ)[0]
        if seq < self._last_cam:          # writer restarted -> re-initialised ring
            self._reset()
        if seq == 0 or seq == self._last_cam:
            return None
        o = self.o
        n1 = self.w * self.h
        base = o["cam_base"] + ((seq - 1) % o["n_slot"]) * o["slot"]
        blob = self.buf[base:base + 5 * n1].tobytes()        # one copy, 1.3 MB
        if struct.unpack_from("<Q", self.buf, OFF_CAM_SEQ)[0] != seq:
            return None                                       # torn, retry next poll
        self._last_cam = seq
        t = struct.unpack_from("<d", self.buf, OFF_CAM_TIME)[0]
        f1 = np.frombuffer(blob, np.uint8, n1, 0).reshape(self.h, self.w).copy()
        f2 = np.frombuffer(blob, np.uint8, n1, n1).reshape(self.h, self.w).copy()
        f3 = np.frombuffer(blob, np.uint8, 3 * n1, 2 * n1).reshape(self.h, self.w, 3).copy()
        return CameraFrame(seq, t, f1, f2, f3)

    def read_imu(self) -> list[ImuSample]:
        total = struct.unpack_from("<Q", self.buf, OFF_IMU_SEQ)[0]
        if total < self._last_imu:        # writer restarted -> re-initialised ring
            self._reset()
            total = struct.unpack_from("<Q", self.buf, OFF_IMU_SEQ)[0]
        if total <= self._last_imu:
            return []
        first = max(self._last_imu, total - self.o["n_imu"])   # skip what we missed
        out = []
        for s in range(first, total):
            p = self.o["imu_base"] + (s % self.o["n_imu"]) * IMU_STRIDE
            v = struct.unpack_from("<7f4f", self.buf, p)
            out.append(ImuSample(v[0], np.array(v[1:4], np.float32),
                                 np.array(v[4:7], np.float32), np.array(v[7:11], np.float32)))
        self._last_imu = total
        return out

    def read_gt(self) -> GtPose | None:
        seq = struct.unpack_from("<Q", self.buf, OFF_GT_SEQ)[0]
        if seq < self._last_gt:
            self._reset()
        if seq == 0 or seq == self._last_gt:
            return None
        v = struct.unpack_from("<d3d4d", self.buf, self.o["gt_base"])
        if struct.unpack_from("<Q", self.buf, OFF_GT_SEQ)[0] != seq:
            return None                                       # torn, retry next poll
        self._last_gt = seq
        return GtPose(v[0], np.array(v[1:4]), np.array(v[4:8]))

    def _reset(self):
        self._last_cam = 0
        self._last_imu = 0
        self._last_gt = 0

    def close(self) -> None:
        del self.buf
