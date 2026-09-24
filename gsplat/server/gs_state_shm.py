#!/usr/bin/env python3
"""Robot-state slot for the decoupled viewer.

The sensor server (one process) writes the robot's full dof_pos — the floating
base pose is dof_pos[0:7] — into a small /dev/shm file; the viewer process
(gsplat/tools/view_window.py) reads it and re-poses its own copy of the MJCF.
That keeps the viewer's llvmpipe rendering (~8 ms/frame) off the simulation's
main thread: the sim keeps full-rate sensors, the window keeps a usable frame
rate. Measured split (church, 1280x720): viewer 117 fps standalone vs rtf 0.5
when the same render ran inside the sim process.

Layout (little endian, 160 B, seq written last — same torn-read protocol as the
sensor ring):

    off  0  magic 'GSPG'    off 16  sim_time (f64)
    off  4  version (u32=1) off 24  n_dof    (u32)
    off  8  seq     (u64)   off 32  dof_pos  (32 x f32, zero-padded)
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"GSPG"
VERSION = 1
SIZE = 160
NDOF_MAX = 32
DEFAULT_PATH = "/dev/shm/gsplay_state.bin"

OFF_MAGIC, OFF_VERSION, OFF_SEQ = 0, 4, 8
OFF_TIME, OFF_NDOF, OFF_DOF = 16, 24, 32


class StateShmWriter:
    def __init__(self, path: str = DEFAULT_PATH, n_dof: int = 19):
        self.path = path
        assert n_dof <= NDOF_MAX, f"n_dof {n_dof} > {NDOF_MAX}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(SIZE)
        self.buf = np.memmap(path, dtype=np.uint8, mode="r+", shape=(SIZE,))
        self.buf[OFF_MAGIC:OFF_MAGIC + 4] = np.frombuffer(MAGIC, np.uint8)
        struct.pack_into("<I", self.buf, OFF_VERSION, VERSION)
        struct.pack_into("<I", self.buf, OFF_NDOF, n_dof)
        struct.pack_into("<Q", self.buf, OFF_SEQ, 0)
        self.n_dof = n_dof
        self.seq = 0
        self.buf.flush()

    def publish(self, sim_time: float, dof_pos) -> None:
        q = np.asarray(dof_pos, np.float32).ravel()
        assert q.size == self.n_dof, f"dof_pos size {q.size} != {self.n_dof}"
        struct.pack_into("<d", self.buf, OFF_TIME, sim_time)
        self.buf[OFF_DOF:OFF_DOF + 4 * self.n_dof] = q.view(np.uint8)
        self.seq += 1
        struct.pack_into("<Q", self.buf, OFF_SEQ, self.seq)   # publish

    def close(self, unlink: bool = True) -> None:
        self.buf.flush()
        del self.buf
        if unlink:
            try:
                os.unlink(self.path)
            except OSError:
                pass


@dataclass
class RobotState:
    seq: int
    sim_time: float
    dof_pos: np.ndarray


class StateShmReader:
    def __init__(self, path: str = DEFAULT_PATH):
        self.buf = np.memmap(path, dtype=np.uint8, mode="r")
        if bytes(self.buf[OFF_MAGIC:OFF_MAGIC + 4]) != MAGIC:
            raise RuntimeError(f"{path}: not a gs state slot")
        ver = struct.unpack_from("<I", self.buf, OFF_VERSION)[0]
        if ver != VERSION:
            raise RuntimeError(f"{path}: state version {ver} != {VERSION}")
        self.n_dof = struct.unpack_from("<I", self.buf, OFF_NDOF)[0]
        self._last = 0

    def read(self) -> RobotState | None:
        seq = struct.unpack_from("<Q", self.buf, OFF_SEQ)[0]
        if seq < self._last:              # writer restarted
            self._last = 0
        if seq == 0 or seq == self._last:
            return None
        raw = self.buf[OFF_DOF:OFF_DOF + 4 * self.n_dof].tobytes()
        if struct.unpack_from("<Q", self.buf, OFF_SEQ)[0] != seq:   # torn
            return None
        self._last = seq
        t = struct.unpack_from("<d", self.buf, OFF_TIME)[0]
        return RobotState(seq, t, np.frombuffer(raw, np.float32).copy())

    def close(self) -> None:
        del self.buf
