#!/usr/bin/env python3
"""Generate `go2_sensor_rig.xml`: the stock go2 MJCF + D435i-like sensor head.

Ported from gs_playground/demo/navigation/sensors/make_sensor_rig.py.
Reads AND writes inside the gs_playground checkout (GS_PLAYGROUND_ROOT): the
generated rig MUST sit next to go2_mjx.xml because MotrixSim resolves the
MJCF's relative mesh paths (`assets/*.obj`) against the file's own directory.
The generator itself is the git-tracked part; the xml is derived.

Adds, without touching the stock go2_mjx.xml:
  * three cameras on the `base` body (forward-looking, xyaxes="0 -1 0 0 0 1"):
      infra1 (y=+0.0255), infra2 (y=-0.0255), color (y=0)   [51 mm stereo baseline]
  * an `imu_optical` site in the camera/optical convention (z = forward, y = down),
    matching tinynav's "body Z = camera forward" IMU rig
  * gyro / accelerometer / framequat sensors bound to that site

MotrixSim ignores MJCF `focal`/`sensorsize` (fovy is always 45 deg), so the
sensor server passes the D435i fovy explicitly to batch_render instead.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

GSPG_NAV = Path(os.environ.get(
    "GS_PLAYGROUND_ROOT", "/workspace/github/simulation/gs_playground"),
    "demo/navigation")
GO2_DIR = GSPG_NAV / "models" / "robots" / "navigation" / "go2"
SRC = GO2_DIR / "go2_mjx.xml"
DST = GO2_DIR / "go2_sensor_rig.xml"

HEAD_X, HEAD_Z = 0.19, 0.09        # D435i head, matches tinynav-sim xacro head_x/head_z
IMU_X = 0.15                       # IMU sits 4 cm behind the camera head (same xacro)
CAM_Y = 0.0255                     # 51 mm / 2
CAM_AXES = "0 -1 0 0 0 1"          # look along body +X, image up = body +Z, right = body -Y

CAMS = [("infra1", CAM_Y), ("infra2", -CAM_Y), ("color", 0.0)]

# optical/RGB-D frame: z = forward (body +X), y = down (body -Z), x = y x z = body -Y
R_OPTICAL = np.stack([np.array([0., -1, 0]),   # x
                      np.array([0., 0, -1]),   # y
                      np.array([1., 0, 0])],   # z
                     axis=1)
QUAT_OPTICAL = Rotation.from_matrix(R_OPTICAL).as_quat()   # xyzw
SITE_QUAT = " ".join(f"{v:.8f}" for v in QUAT_OPTICAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    args = ap.parse_args()

    text = Path(args.src).read_text()

    cam_lines = "\n".join(
        f'      <camera name="{n}" pos="{HEAD_X} {y} {HEAD_Z}" xyaxes="{CAM_AXES}"/>'
        for n, y in CAMS)
    anchor = '      <camera name="head_camera"'
    i = text.index(anchor)
    j = text.index("\n", i) + 1
    text = text[:j] + cam_lines + "\n" + text[j:]

    imu_anchor = '      <site name="imu" '
    i = text.index(imu_anchor)
    j = text.index("\n", i) + 1
    text = text[:j] + (
        f'      <!-- optical-convention IMU frame (z=forward, y=down); 51 mm head, 0.15 m ahead -->\n'
        f'      <site name="imu_optical" pos="{IMU_X} 0 {HEAD_Z}" quat="{SITE_QUAT}"/>\n'
    ) + text[j:]

    text = text.replace("  </sensor>", (
        '    <gyro name="imu_gyro" site="imu_optical"/>\n'
        '    <accelerometer name="imu_accel" site="imu_optical"/>\n'
        '    <framequat name="imu_quat" objtype="site" objname="imu_optical"/>\n'
        "  </sensor>"))

    Path(args.dst).write_text(text)
    print(f"wrote {args.dst}")
    print(f"  cameras : {', '.join(n for n, _ in CAMS)}  at x={HEAD_X} z={HEAD_Z}, "
          f"stereo baseline {2*CAM_Y*1000:.0f} mm")
    print(f"  imu site: imu_optical quat(xyzw) = {SITE_QUAT}")


if __name__ == "__main__":
    main()
