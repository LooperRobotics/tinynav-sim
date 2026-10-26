#!/usr/bin/env python3
"""Synthesize a rosbag2 of one (or more) floorplan laps for tinynav mapping.

Kinematically walks the ring loop (W platform -> N flight -> E platform ->
S flight -> W) at fixed speed with in-place corner turns, renders the
contract cameras at each tick, and writes the exact topics the jd mapper
consumes: infra1/infra2 mono8, depth mono16, color JPEG, vio_image +
vio_100hz PoseStamped (left-camera optical pose, the live face's
convention), imu, camera_info x3, /tf_static, vio_status.  Output is a
hand-rolled rosbag2 sqlite3 bag (rclpy CDR blobs) + metadata.yaml, so no
rosbag2_py is needed in the sim container.

    python3 gen_floorplan_bag.py --out /tmp/synth_lap [--laps 1] [--speed 0.5]

Then on the mapping host:
    build_map_offline.py --bag_file <dir> --map_save_path <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

import mujoco

from sim.plant import build_model
from sim.render_contract import MujocoContractRenderer

from hil import (BASELINE, CONTRACT_H, CONTRACT_W, FRAMES, ORACLE_TF_STATIC,
                 POSE_FRAME, VIO_STATUS, mat_to_quat_xyzw, optical)

K = "/camera/camera"
IMG_QOS = ("- history: 3\n  depth: 4\n  reliability: 1\n  durability: 2\n"
           "  deadline:\n    sec: 9223372036854775807\n  lifespan:\n"
           "    sec: 9223372036854775807\n  liveliness: 1\n"
           "  liveliness_lease_duration:\n    sec: 9223372036854775807\n"
           "  avoid_ros_namespace_conventions: false")
LATCH_QOS = ("- history: 3\n  depth: 1\n  reliability: 1\n  durability: 1\n"
             "  deadline:\n    sec: 9223372036854775807\n  lifespan:\n"
             "    sec: 9223372036854775807\n  liveliness: 1\n"
             "  liveliness_lease_duration:\n    sec: 9223372036854775807\n"
             "  avoid_ros_namespace_conventions: false")

# topic -> (type string, qos string)
TOPICS = {
    f"{K}/infra1/image_rect_raw": ("sensor_msgs/msg/Image", IMG_QOS),
    f"{K}/infra2/image_rect_raw": ("sensor_msgs/msg/Image", IMG_QOS),
    f"{K}/depth/image_rect_raw": ("sensor_msgs/msg/Image", IMG_QOS),
    f"{K}/color/image_rect_raw/compressed":
        ("sensor_msgs/msg/CompressedImage", IMG_QOS),
    f"{K}/vio_image": ("geometry_msgs/msg/PoseStamped", IMG_QOS),
    f"{K}/vio_100hz": ("geometry_msgs/msg/PoseStamped", IMG_QOS),
    f"{K}/vio_status": ("std_msgs/msg/String", LATCH_QOS),
    f"{K}/infra1/camera_info": ("sensor_msgs/msg/CameraInfo", IMG_QOS),
    f"{K}/infra2/camera_info": ("sensor_msgs/msg/CameraInfo", IMG_QOS),
    f"{K}/color/camera_info": ("sensor_msgs/msg/CameraInfo", IMG_QOS),
    "/tf_static": ("tf2_msgs/msg/TFMessage", LATCH_QOS),
    f"{K}/imu": ("sensor_msgs/msg/Imu", IMG_QOS),
}


def build_trajectory(waypoints, yaw0, speed, corner_rate, laps, dwell):
    """(t, x, y, yaw_rad, yaw_rate) rows: legs + in-place 90 deg corners."""
    rows = []
    t = 0.0
    x, y = waypoints[0]
    yaw = yaw0
    for _ in range(dwell):
        rows.append((t, x, y, yaw, 0.0))
        t += 0.01

    def leg_pts(ax, ay, bx, by):
        L = math.hypot(bx - ax, by - ay)
        n = max(2, int(L / speed / 0.01))
        heading = math.atan2(by - ay, bx - ax)
        for i in range(n + 1):
            s = i / n
            rows.append((t + s * L / speed, ax + (bx - ax) * s,
                         ay + (by - ay) * s, heading, 0.0))
        return L / speed

    def turn(a0, a1):
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        dur = abs(d) / corner_rate
        n = max(2, int(dur / 0.01))
        sgn = math.copysign(1.0, d)
        for i in range(n + 1):
            s = i / n
            rows.append((t + s * dur, x, y,
                         a0 + sgn * abs(d) * s, sgn * corner_rate))
        return dur

    path = waypoints[:]
    prev = yaw0
    for lap in range(laps):
        for i in range(len(path) - 1):
            t += leg_pts(*path[i], *path[i + 1])
            x, y = path[i + 1]
            if i + 1 < len(path) - 1 or lap < laps - 1:
                nxt = path[i + 2] if i + 2 < len(path) else path[1]
                target = math.atan2(nxt[1] - y, nxt[0] - x)
                t += turn(prev, target)
                prev = target
    for _ in range(50):
        rows.append((t, x, y, prev, 0.0))
        t += 0.01
    rows.sort(key=lambda r: r[0])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--cam-hz", type=float, default=10.0)
    ap.add_argument("--vio-hz", type=float, default=100.0)
    ap.add_argument("--corner-rate", type=float,
                    default=45.0, help="deg/s in-place turn")
    ap.add_argument("--clockwise", action="store_true", default=True)
    args = ap.parse_args()

    from sim import floorplan as fp
    from sim.plant import _ACTIVE_PARAMS

    p = _ACTIVE_PARAMS or fp.load_params()
    wx = -(p.free_len_x / 2.0 - p.platform_len / 2.0)
    ex = -wx
    cy = p.free_len_y / 2.0 - p.flight_width / 2.0
    # clockwise ring: W platform -> N flight -> E -> S -> back
    wps = [(wx, 0.0), (wx, cy), (ex, cy), (ex, -cy), (wx, -cy), (wx, 0.0)]
    traj = build_trajectory(wps, math.radians(90.0), args.speed,
                            math.radians(args.corner_rate), args.laps, 100)
    dur = traj[-1][0]
    print(f"trajectory: {dur:.1f}s, {len(traj)} ticks, "
          f"bbox x[{wps[0][0]:.2f},{ex:.2f}] y[{-cy:.2f},{cy:.2f}]")

    model, b = build_model("floorplan")
    data = mujoco.MjData(model)
    rend = MujocoContractRenderer(
        model, b.contract_left_camera_id, b.contract_right_camera_id)

    # T_base_camera (mujoco convention) measured once at the spawn pose
    yaw0 = math.radians(90.0)
    data.qpos[0:3] = [wps[0][0], wps[0][1], 0.30]
    data.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
    mujoco.mj_forward(model, data)
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, data.qpos[3:7])
    Rb = Rb.reshape(3, 3)
    Rcam = data.cam_xmat[b.contract_left_camera_id].reshape(3, 3)
    pcam = data.cam_xpos[b.contract_left_camera_id]
    pbase = data.qpos[0:3]
    R_bc = Rb.T @ Rcam
    t_bc = Rb.T @ (pcam - pbase)

    from rclpy.serialization import serialize_message
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, Image
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    t0 = time.time_ns() // 1000 * 1000

    def stamp_msg(h, t):
        s = t0 + int(t * 1e9)
        h.stamp.sec = s // 10**9
        h.stamp.nanosec = s % 10**9
        return s

    out = []

    def add(topic, t, msg):
        out.append((t0 + int(t * 1e9), topic, bytes(serialize_message(msg))))

    def img_msg(arr, enc, frame, t):
        m = Image()
        m.header.stamp.sec, m.header.stamp.nanosec = (
            lambda s: (s // 10**9, s % 10**9))(stamp_msg(m.header, t))
        m.header.frame_id = frame
        m.height, m.width = arr.shape[:2]
        m.encoding = enc
        m.is_bigendian = 0
        m.step = arr.shape[1] * (2 if enc == "mono16" else 1)
        m.data = np.ascontiguousarray(arr).tobytes()
        return m

    def pose_msg(t, pos, quat_xyzw, frame=POSE_FRAME):
        m = PoseStamped()
        s = stamp_msg(m.header, t)
        m.header.frame_id = frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = \
            float(pos[0]), float(pos[1]), float(pos[2])
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = \
            (float(v) for v in quat_xyzw)
        return m, s

    fx, cx, cy_px = 302.7782745, 272.0, 320.0

    def cam_info(t, frame, right):
        m = CameraInfo()
        stamp_msg(m.header, t)
        m.header.frame_id = frame
        m.height, m.width = CONTRACT_H, CONTRACT_W
        m.distortion_model = "plumb_bob"
        m.d = [0.0] * 5
        tx = -fx * BASELINE if right else 0.0
        m.k = [fx, 0.0, cx, 0.0, fx, cy_px, 0.0, 0.0, 1.0]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        m.p = [fx, 0.0, cx, tx, 0.0, fx, cy_px, 0.0, 0.0, 0.0, 1.0, 0.0]
        return m

    def static_msgs(t):
        tf = TFMessage()
        for parent, child, trans, rot in ORACLE_TF_STATIC:
            ts = TransformStamped()
            stamp_msg(ts.header, t)
            ts.header.frame_id = parent
            ts.child_frame_id = child
            (ts.transform.translation.x, ts.transform.translation.y,
             ts.transform.translation.z) = trans
            (ts.transform.rotation.x, ts.transform.rotation.y,
             ts.transform.rotation.z, ts.transform.rotation.w) = rot
            tf.transforms.append(ts)
        st = String()
        st.data = VIO_STATUS
        return tf, st

    cam_dt = 1.0 / args.cam_hz
    vio_dt = 1.0 / args.vio_hz
    next_cam = 0.0
    next_vio = 0.0
    next_5s = 0.0
    n_frames = 0
    for t, x, y, yaw, wz in traj:
        if t >= next_5s:
            tf, st = static_msgs(t)
            add("/tf_static", t, tf)
            add(f"{K}/vio_status", t, st)
            add(f"{K}/infra1/camera_info", t, cam_info(t, FRAMES["infra1"], False))
            add(f"{K}/infra2/camera_info", t, cam_info(t, FRAMES["infra2"], True))
            add(f"{K}/color/camera_info", t, cam_info(t, FRAMES["color"], False))
            next_5s += 5.0
        if t >= next_vio:
            q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            Rb = np.zeros(9)
            mujoco.mju_quat2Mat(Rb, q)
            Rb = Rb.reshape(3, 3)
            Rwc = Rb @ R_bc
            twc = Rb @ t_bc + np.array([x, y, 0.30])
            R_opt, t_opt = optical((Rwc, twc))
            m, _ = pose_msg(t, t_opt, mat_to_quat_xyzw(R_opt))
            add(f"{K}/vio_100hz", t, m)
            imu = Imu()
            stamp_msg(imu.header, t)
            imu.header.frame_id = FRAMES["imu"]
            (imu.orientation.x, imu.orientation.y,
             imu.orientation.z, imu.orientation.w) = \
                float(q[1]), float(q[2]), float(q[3]), float(q[0])
            imu.angular_velocity.z = float(wz)
            imu.linear_acceleration.z = 9.81
            add(f"{K}/imu", t, imu)
            next_vio += vio_dt
        if t >= next_cam:
            data.qpos[0:3] = [x, y, 0.30]
            data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(model, data)
            g_l, g_r, depth, R_opt, t_opt, jpg = rend.render(
                [x, y, 0.30],
                [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            s = t0 + int(t * 1e9)
            add(f"{K}/infra1/image_rect_raw", t,
                img_msg(g_l, "mono8", FRAMES["infra1"], t))
            add(f"{K}/infra2/image_rect_raw", t,
                img_msg(g_r, "mono8", FRAMES["infra2"], t))
            add(f"{K}/depth/image_rect_raw", t,
                img_msg(depth, "mono16", FRAMES["depth"], t))
            cm = PoseStamped()
            cm.header.stamp.sec, cm.header.stamp.nanosec = s // 10**9, s % 10**9
            cm.header.frame_id = POSE_FRAME
            cm.pose.position.x, cm.pose.position.y, cm.pose.position.z = \
                (float(v) for v in t_opt)
            (cm.pose.orientation.x, cm.pose.orientation.y,
             cm.pose.orientation.z, cm.pose.orientation.w) = \
                (float(v) for v in mat_to_quat_xyzw(R_opt))
            add(f"{K}/vio_image", t, cm)
            if jpg is not None:
                comp = CompressedImage()
                comp.header.stamp.sec = s // 10**9
                comp.header.stamp.nanosec = s % 10**9
                comp.header.frame_id = FRAMES["color"]
                comp.format = "jpeg"
                comp.data = bytes(jpg)
                add(f"{K}/color/image_rect_raw/compressed", t, comp)
            n_frames += 1
            next_cam += cam_dt

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    db_path = outdir / "synth_lap_0.db3"
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT NOT NULL, "
                "type TEXT NOT NULL, serialization_format TEXT NOT NULL, "
                "offered_qos_profiles TEXT NOT NULL, type_description_hash TEXT NOT NULL)")
    con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id "
                "INTEGER NOT NULL REFERENCES topics (id), timestamp INTEGER NOT NULL, "
                "data BLOB NOT NULL)")
    name_id = {}
    for i, (name, (typ, qos)) in enumerate(sorted(TOPICS.items())):
        con.execute("INSERT INTO topics VALUES (?,?,?,?,?,?)",
                    (i, name, typ, "cdr", qos, ""))
        name_id[name] = i
    out.sort(key=lambda r: (r[0], r[1]))
    for i, (ts, topic, blob) in enumerate(out):
        con.execute("INSERT INTO messages VALUES (?,?,?,?)",
                    (i, name_id[topic], ts, blob))
    con.commit()
    con.close()

    counts = {}
    for _, topic, _ in out:
        counts[topic] = counts.get(topic, 0) + 1
    meta = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "relative_file_paths": [db_path.name],
            "duration": {"nanoseconds": int(dur * 1e9)},
            "starting_time": {"nanoseconds_since_epoch": t0},
            "message_count": len(out),
        "compression_format": "",
        "compression_mode": "",
            "topics_with_message_count": [
                {"topic_metadata": {
                    "name": name,
                    "type": TOPICS[name][0],
                    "serialization_format": "cdr",
                    "offered_qos_profiles": TOPICS[name][1]},
                 "message_count": counts.get(name, 0)}
                for name in sorted(TOPICS)],
        }
    }
    import yaml

    # humble's v5 info schema wants a per-file section (the reader rejects
    # the bag without it: "invalid node; first invalid key: files")
    info = meta["rosbag2_bagfile_information"]
    db = info["relative_file_paths"][0]
    info["files"] = [{"path": db,
                      "starting_time": info["starting_time"],
                      "duration": info["duration"],
                      "message_count": info["message_count"]}]
    (outdir / "metadata.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False))
    size = db_path.stat().st_size / 1e6
    print(f"bag: {db_path} {size:.1f}MB, {len(out)} msgs, {n_frames} frames, "
          f"{dur:.1f}s  -> {outdir}")


if __name__ == "__main__":
    main()
