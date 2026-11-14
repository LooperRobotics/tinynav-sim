#!/usr/bin/env python3
"""Convert a MetaCam scene (ros2 bag + LIO trajectory + point cloud map) into a
nerfstudio-format dataset (transforms.json + images + masks + init point cloud).

Inputs (defaults assume the lightning-lm offline mapping layout, all under the
scene dir given by METACAM_SCENE env var or CWD):
  - trajectory:  data/new_map/traj_tum.txt      (gravity-aligned LIO poses; this
                 repo aligns world Z to gravity at IMU init, no manual de-tilt
                 stage is needed -- see README_METACAM.md section 5)
  - map cloud:   data/new_map/global.pcd        (points3d.ply init cloud source)
  - calibration: calib/calibration.json         (device file from the SD card)
  - bag:         ros2_bag/                      (convert_ros1_to_ros2_vec.py output)

Pipeline per image:
  1. interpolate T_world_body from the 10 Hz TUM trajectory (linear + slerp)
  2. T_world_cam = T_world_body @ T_body_lidar @ inv(T_cam_lidar)
  3. remap the distorted fisheye image to an *ideal equidistant* fisheye
  4. mask pixels outside the lens image circle
  5. write half/quarter scale variants (images_2/4, masks_2/4)
Also voxel-downsamples the map cloud into points3d.ply (world frame).

Deps: pip install rosbags numpy opencv-python open3d
"""

import os
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

MAP_DIR = Path(os.environ.get("METACAM_SCENE", ".")).resolve()  # 场景目录：METACAM_SCENE 环境变量或 CWD
LEFT_TOPIC = "/camera/left/jpeg"
RIGHT_TOPIC = "/camera/right/jpeg"

# per-process caches, filled lazily (or inherited from the parent via fork)
_CAMS = {}


# ---------------------------------------------------------------- poses ----
def load_trajectory(path):
    """TUM file: t tx ty tz qx qy qz qw -> (ts, T_world_body)."""
    data = np.loadtxt(path)
    ts, trans, quats = data[:, 0], data[:, 1:4], data[:, 4:8]
    mats = np.tile(np.eye(4), (len(ts), 1, 1))
    mats[:, :3, :3] = quat_to_mat(quats)  # qx qy qz qw
    mats[:, :3, 3] = trans
    return ts, mats


def quat_to_mat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = x * x + y * y + z * z + w * w
    s = 2.0 / np.where(n == 0, 1, n)
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - (yy + zz)
    m[:, 0, 1] = xy - wz
    m[:, 0, 2] = xz + wy
    m[:, 1, 0] = xy + wz
    m[:, 1, 1] = 1 - (xx + zz)
    m[:, 1, 2] = yz - wx
    m[:, 2, 0] = xz - wy
    m[:, 2, 1] = yz + wx
    m[:, 2, 2] = 1 - (xx + yy)
    return m


def mat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def interpolate_pose(ts, mats, t):
    """Interpolate T_world_body at time t (linear trans + slerp rot)."""
    i = np.searchsorted(ts, t)
    if i == 0 or i >= len(ts):
        return None
    t0, t1 = ts[i - 1], ts[i]
    u = (t - t0) / (t1 - t0)
    T = np.eye(4)
    T[:3, 3] = (1 - u) * mats[i - 1][:3, 3] + u * mats[i][:3, 3]
    q0 = mat_to_quat(mats[i - 1][:3, :3])
    q1 = mat_to_quat(mats[i][:3, :3])
    T[:3, :3] = quat_to_mat(slerp(q0, q1, u)[None])[0]
    return T


def rt_to_mat(rot, pos):
    T = np.eye(4)
    T[:3, :3] = np.asarray(rot)
    T[:3, 3] = np.asarray(pos)
    return T


# --------------------------------------------------------------- fisheye ----
def build_remap(fx, fy, cx, cy, w, h, k1, k2, k3, k4):
    """Map from ideal-equidistant dst pixels -> distorted src pixels.

    Rays beyond 90 deg off-axis (dz <= 0) do not exist in the source fisheye;
    they are mapped to -1 so the remap yields the border value and the mask
    excludes them.
    """
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.hypot(x, y)
    theta = r  # ideal equidistant: r_px = f * theta
    safe = np.maximum(r, 1e-9)
    dx = np.sin(theta) * x / safe
    dy = np.sin(theta) * y / safe
    dz = np.cos(theta)
    valid = dz > 1e-6
    zsafe = np.where(valid, dz, 1.0)
    xp = np.where(valid, dx / zsafe, 0.0)
    yp = np.where(valid, dy / zsafe, 0.0)
    rr = np.hypot(xp, yp)
    th = np.arctan(rr)
    th2 = th * th
    th_d = th * (1 + k1 * th2 + k2 * th2**2 + k3 * th2**3 + k4 * th2**4)
    scale = np.where(rr > 1e-9, th_d / np.maximum(rr, 1e-9), 1.0)
    map_x = np.where(valid, fx * xp * scale + cx, -1.0).astype(np.float32)
    map_y = np.where(valid, fy * yp * scale + cy, -1.0).astype(np.float32)
    return map_x, map_y


def _verify_remap(map_x, map_y, fx, fy, cx, cy, w, h, k1, k2, k3, k4):
    """Cross-check the analytic remap against cv2.fisheye.projectPoints."""
    us = np.linspace(0, w - 1, 64).astype(int)
    vs = np.linspace(0, h - 1, 64).astype(int)
    u, v = np.meshgrid(us, vs)
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.hypot(x, y)
    safe = np.maximum(r, 1e-9)
    dirs = np.stack([np.sin(r) * x / safe, np.sin(r) * y / safe, np.cos(r)], axis=-1)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.array([k1, k2, k3, k4])
    proj, _ = cv2.fisheye.projectPoints(dirs.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, D)
    proj = proj.reshape(-1, 2)
    ok = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    ui = np.clip(u, 0, w - 1)
    vi = np.clip(v, 0, h - 1)
    mine = np.stack([map_x[vi, ui], map_y[vi, ui]], axis=-1)
    ok_grid = ok[vi, ui].ravel()
    err = np.abs(mine.reshape(-1, 2)[ok_grid] - proj[ok_grid]).max()
    assert err < 0.05, f"remap mismatch vs cv2.fisheye: {err} px"


def estimate_circle_radius(gray, cx, cy):
    """Estimate the lens image-circle radius from the dark border."""
    h, w = gray.shape
    yy, xx = np.nonzero(np.ones_like(gray, dtype=bool))
    rad = np.hypot(xx - cx, yy - cy).astype(np.int32)
    bright = (gray > 10).astype(np.float32).ravel()
    r_max = int(min(np.hypot(0 - cx, 0 - cy), np.hypot(w - cx, 0 - cy),
                    np.hypot(0 - cx, h - cy), np.hypot(w - cx, h - cy)))
    frac = np.bincount(rad.ravel(), weights=bright, minlength=r_max + 2)[: r_max + 1]
    cnt = np.bincount(rad.ravel(), minlength=r_max + 2)[: r_max + 1]
    frac = np.divide(frac, np.maximum(cnt, 1))
    r_valid = 0
    run = 0
    for r in range(r_max, 0, -1):
        if frac[r] > 0.3:
            run += 1
            if run >= 20:
                r_valid = r + 20 - 1
                break
        else:
            run = 0
    return r_valid


def get_cam(name):
    """Lazily build remap maps + circle mask for a camera (per process)."""
    if name not in _CAMS:
        calib = json.load(open(MAP_DIR / "calib" / "calibration.json"))
        c = next(c for c in calib["cameras"] if c["name"] == name)
        k = c["distortion"]["params"]
        fx, fy = c["intrinsic"]["fl_x"], c["intrinsic"]["fl_y"]
        cx, cy = c["intrinsic"]["cx"], c["intrinsic"]["cy"]
        map_x, map_y = build_remap(fx, fy, cx, cy, c["width"], c["height"],
                                   k["k1"], k["k2"], k["k3"], k["k4"])
        _verify_remap(map_x, map_y, fx, fy, cx, cy, c["width"], c["height"],
                      k["k1"], k["k2"], k["k3"], k["k4"])
        valid = (map_x >= 0) & (map_x <= c["width"] - 1) & (map_y >= 0) & (map_y <= c["height"] - 1)
        _CAMS[name] = dict(
            T_lidar_cam=np.linalg.inv(rt_to_mat(c["transform_from_lidar"].get("rotation", None),
                                                c["transform_from_lidar"]["position"])),
            map_x=map_x, map_y=map_y,
            mask=(valid.astype(np.uint8)) * 255,
            fx=fx, fy=fy, cx=cx, cy=cy,
            w=c["width"], h=c["height"],
            r_valid=None,
        )
    return _CAMS[name]


def process_frame(job):
    """Decode + remap + mask + write one frame. Runs in worker processes."""
    name, t, jpeg = job
    cv2.setNumThreads(1)
    out = _OUT
    cam = get_cam(name)

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (cam["h"],cam["w"]), "Recorded JPEG size differs from adapted calibration"
    if cam["r_valid"] is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cam["r_valid"] = estimate_circle_radius(gray, cam["cx"], cam["cy"])

    stem = f"{int(t * 1000):013d}_{name}"
    fname = f"images/{stem}.png"
    remapped = cv2.remap(img, cam["map_x"], cam["map_y"], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ok = cv2.imwrite(str(out / fname), remapped, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    ok &= cv2.imwrite(str(out / f"masks/{stem}.png"), cam["mask"], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    for s, sub in ((2, "2"), (4, "4")):
        small = cv2.resize(remapped, (cam["w"] // s, cam["h"] // s), interpolation=cv2.INTER_AREA)
        ok &= cv2.imwrite(str(out / f"images_{sub}" / f"{stem}.png"), small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        small_m = cv2.resize(cam["mask"], (cam["w"] // s, cam["h"] // s), interpolation=cv2.INTER_NEAREST)
        ok &= cv2.imwrite(str(out / f"masks_{sub}" / f"{stem}.png"), small_m, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise IOError(f"failed to write {stem}")

    T_body_lidar = np.array(_T_BODY_LIDAR)
    T_world_cam = np.asarray(_traj_lookup(t)) @ T_body_lidar @ cam["T_lidar_cam"]
    # nerfstudio stores camera-to-world in the OpenGL convention (looks -z);
    # the calibration chain above is OpenCV (z forward). Flip y/z.
    T_world_cam[:, 1:3] *= -1
    return fname, f"masks/{stem}.png", name, T_world_cam.tolist(), cam["r_valid"]


# globals for workers (set before Pool fork)
_OUT = None
_TRAJ_TS = None
_TRAJ = None
_T_BODY_LIDAR = None


def _traj_lookup(t):
    T = interpolate_pose(_TRAJ_TS, _TRAJ, t)
    if T is None:
        raise KeyError(t)
    return T


# ------------------------------------------------------------------ pcd ----
def voxel_downsample(xyz, voxel):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cnt), 3), np.float64)
    np.add.at(sums, inv, xyz)
    return (sums / cnt[:, None]).astype(np.float32)


def crop_robust(xyz, pct=0.5, pad=2.0):
    """Drop through-glass outliers via a percentile bounding box + pad (m)."""
    lo, hi = np.percentile(xyz, [pct, 100 - pct], axis=0)
    keep = (xyz >= lo - pad).all(1) & (xyz <= hi + pad).all(1)
    return xyz[keep]


def write_ply(path, xyz):
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 256, (len(xyz), 3), dtype=np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["r"], data["g"], data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


# ------------------------------------------------------------------ main ----
def main():
    global _OUT, _TRAJ_TS, _TRAJ, _T_BODY_LIDAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MAP_DIR / "nerfstudio"))
    ap.add_argument("--traj", default=str(MAP_DIR / "data" / "new_map" / "traj_tum.txt"),
                    help="TUM 轨迹（本仓库重力对齐输出，默认 data/new_map/traj_tum.txt）")
    ap.add_argument("--cloud", default=str(MAP_DIR / "data" / "new_map" / "global.pcd"),
                    help="点云地图（默认 data/new_map/global.pcd），生成 points3d.ply 初始化点")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="process only first N frames (testing)")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("images", "images_2", "images_4", "masks", "masks_2", "masks_4"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    _OUT = out

    calib = json.load(open(MAP_DIR / "calib" / "calibration.json"))
    _T_BODY_LIDAR = rt_to_mat(calib["imu"]["transform_from_lidar"].get("rotation", None),
                              calib["imu"]["transform_from_lidar"]["position"])
    _TRAJ_TS, _TRAJ = load_trajectory(args.traj)
    print(f"trajectory: {len(_TRAJ_TS)} poses [{_TRAJ_TS[0]:.3f}, {_TRAJ_TS[-1]:.3f}]")

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    jobs = []
    with Reader(str(MAP_DIR / "ros2_bag")) as reader:
        conns = [c for c in reader.connections if c.topic in (LEFT_TOPIC, RIGHT_TOPIC)]
        for conn, _, raw in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            name = "left" if "left" in conn.topic else "right"
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t <= _TRAJ_TS[0] or t >= _TRAJ_TS[-1]:
                continue
            jobs.append((name, t, bytes(msg.data)))
    jobs.sort(key=lambda j: j[1])
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"frames in traj range: {len(jobs)}")

    t0 = time.time()
    ctx = mp.get_context("fork")
    frames, skipped, r_valids = [], 0, {}
    with ctx.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_frame, jobs, chunksize=4)):
            fname, mname, name, T, rv = res
            if name not in r_valids:
                r_valids[name] = rv
                cam = get_cam(name)
                print(f"\n{name}: lens circle radius = {rv} px "
                      f"(theta_max = {np.degrees(rv / cam['fx']):.1f} deg)")
            frames.append(dict(file_path=fname, mask_path=mname, camera=name, transform_matrix=T))
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                eta = el / (i + 1) * (len(jobs) - i - 1)
                print(f"\r{i + 1}/{len(jobs)} frames  ({el:.0f}s, eta {eta / 60:.1f}m)", end="", flush=True)
    print(f"\nconverted {len(frames)} frames in {time.time() - t0:.0f}s")

    # init point cloud from the rotated LIO map (world frame)
    import open3d as o3d
    xyz = np.asarray(o3d.io.read_point_cloud(args.cloud).points)
    print(f"map points: {len(xyz)}")
    xyz = crop_robust(xyz)
    print(f"robust-crop kept: {len(xyz)}")
    xyz = voxel_downsample(xyz, args.voxel)
    print(f"downsampled @{args.voxel}m: {len(xyz)}")
    write_ply(out / "points3d.ply", xyz)

    meta_cam = {c["name"]: c for c in calib["cameras"]}
    out_frames = []
    for f in sorted(frames, key=lambda f: f["file_path"]):
        c = meta_cam[f["camera"]]
        out_frames.append({
            "file_path": f["file_path"],
            "mask_path": f["mask_path"],
            "fl_x": c["intrinsic"]["fl_x"], "fl_y": c["intrinsic"]["fl_y"],
            "cx": c["intrinsic"]["cx"], "cy": c["intrinsic"]["cy"],
            "w": c["width"], "h": c["height"],
            "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
            "transform_matrix": f["transform_matrix"],
        })
    transforms = {
        "camera_model": "OPENCV_FISHEYE",
        "ply_file_path": "points3d.ply",
        "frames": out_frames,
    }
    with open(out / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"wrote {out / 'transforms.json'} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()
