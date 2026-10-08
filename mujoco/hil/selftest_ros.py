#!/usr/bin/env python3
"""RosFace self-test with synthetic frames -- no mujoco/wgpu needed.

Runs inside a stock ROS 2 container (Humble py3.10 works; the production
face runs on jazzy py3.12): publishes the full contract at the real rates
from moving synthetic images, so message construction, QoS, stamps and the
fast thread are all exercised end-to-end on the DDS wire.

    python3 hil/selftest_ros.py [--seconds 8]
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mujoco/ on path

import numpy as np

from hil import CONTRACT_H, CONTRACT_W, TelemetrySink, mat_to_quat_xyzw
from hil.ros import RosFace


class FakeData:
    """Duck-typed MjData: sensordata = [gyro(3) @0, acc(3) @3], qpos pos+quat."""
    def __init__(self, t):
        self.sensordata = np.zeros(6, dtype=np.float64)
        self.sensordata[0] = 0.01 * np.sin(t)
        self.sensordata[3] = -9.81
        self.qpos = np.zeros(7, dtype=np.float64)
        self.qpos[0] = 0.05 * t                      # x creeps forward
        self.qpos[2] = 0.3
        yaw = 0.1 * t
        self.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


def main() -> int:
    seconds = 8.0
    if "--seconds" in sys.argv:
        seconds = float(sys.argv[sys.argv.index("--seconds") + 1])
    sink = TelemetrySink(gyro_adr=0, acc_adr=3, root_qpos_adr=0,
                         t_base_cam=(np.eye(3), np.array([0.3, 0.0, 0.05])))
    face = RosFace(sink, vio_hz=100.0, imu_every=1)

    stop = threading.Event()

    def imu_feed():                                   # 200 Hz substep feed
        t = 0.0
        while not stop.is_set():
            sink.sample(FakeData(t))
            t += 0.005
            time.sleep(0.005)
    threading.Thread(target=imu_feed, daemon=True).start()

    frame = np.zeros((CONTRACT_H, CONTRACT_W, 3), np.uint8)
    n = 0
    t0 = time.perf_counter()
    last_print = t0
    while time.perf_counter() - t0 < seconds:
        t = time.perf_counter() - t0
        # moving synthetic image (visible in rqt)
        frame[:] = (np.arange(CONTRACT_W)[None, :, None] + 4 * n) % 256
        stamp = time.time()
        data = FakeData(t)
        sink.sample(data)                             # camera-stamp top-up
        gray_l = frame[:, :, 0].copy()
        gray_r = np.roll(frame[:, :, 0], 8, axis=1).copy()
        depth = (500 + 100 * np.sin(t * 2)).astype(np.float64) * \
            np.ones((CONTRACT_H, CONTRACT_W), np.float64)
        R = np.eye(3, dtype=np.float64)
        tv = np.array([data.qpos[0] + 0.3, 0.0, 0.05])
        try:
            import cv2
            ok, jpg = cv2.imencode(".jpg", frame[:, :, ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg = jpg.tobytes() if ok else None
        except ImportError:
            jpg = None
        face.publish_frame(stamp, gray_l, gray_r, depth.astype(np.uint16),
                           mat_to_quat_xyzw(R), tv, jpg)
        face.publish_camera_info(stamp, 302.779, CONTRACT_W / 2, CONTRACT_H / 2)
        n += 1
        if time.perf_counter() - last_print >= 1.0:
            print(f"[selftest] frames {n} imu {face.n_imu} vio {face.n_vio} "
                  f"dropped {sink.dropped}", flush=True)
            face.n_imu = face.n_vio = 0
            last_print = time.perf_counter()
        time.sleep(max(0.0, 0.1 - (time.perf_counter() - t0 - n * 0.1)))
    stop.set()
    time.sleep(0.3)
    face.publish_status(time.time())
    face.close()
    print(f"[selftest] done: {n} frames in {seconds:.0f}s "
          f"({n / seconds:.1f} Hz)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
