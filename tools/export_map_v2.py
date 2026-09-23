#!/usr/bin/env python3
"""Export a Python-built TinyNav map directory to map format v2 (plain .npy + json).

The Python map stores exactly the structures map_node.py::load_map consumes:

  poses.npy                 pickled dict {timestamp_ns: 4x4 pose matrix}
  features.<dbm>            shelve {ts: {"kpts","descps","mask"}}   (IntKeyShelf)
  depths.<dbm>              shelve {ts: HxW float32}
  vlad_descriptors.<dbm>    shelve {ts: (d,)}
  metadata.<dbm>            shelve {"vlad_centres": (C, d)}
  intrinsics/occupancy_grid/occupancy_meta/sdf_map/path_speed/path_climb  plain .npy

v2 replaces the pickled/shelve parts with C-order .npy files a C++ reader can
load without pickle/shelve, all rows ordered by ascending timestamp:

  pose_timestamps.npy [N] i64
  pose_matrices.npy   [N,4,4] f64
  vlad_centres.npy    [C,d]
  vlad_descriptors.npy [N,d]
  feature_offsets.npy [N+1] i64 (rows of feature_kpts/descps/mask per keyframe)
  feature_kpts.npy    [M,2] f32  (canonical [N,2] (x,y) pixel coords)
  feature_descps.npy  [M,D] f32
  feature_mask.npy    [M] u1
  depth_images.npy    [N,H,W] u2 millimeters (0 = invalid; the C++ reader
                      converts back to f32 meters per candidate frame)
  meta.json           informational (count/dims); the C++ reader validates by shape
  + copies of the plain .npy files above.

Usage:  python3 tools/export_map_v2.py <python_map_dir> <v2_output_dir>
Runs with the image's python3 (numpy + shelve stdlib only, no ROS imports).
"""
import json
import os
import shutil
import shelve
import sys

import numpy as np


def canon_kpts(kpts: np.ndarray) -> np.ndarray:
    """features dict kpts -> [N,2] f32. The engine may emit [1,N,2] or [2,N]."""
    a = np.asarray(kpts)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    return a.astype(np.float32)


def canon_descps(d: np.ndarray) -> np.ndarray:
    a = np.asarray(d)
    if a.ndim == 3:
        a = a[0]
    return a.astype(np.float32)


def canon_mask(m) -> np.ndarray:
    a = np.asarray(m).reshape(-1)
    return a.astype(np.uint8)


def read_shelve(path: str) -> dict:
    db = shelve.open(path, flag="r")
    try:
        return dict(db)
    finally:
        db.close()


def main(src: str, dst: str) -> None:
    poses = np.load(os.path.join(src, "poses.npy"), allow_pickle=True).item()
    timestamps = sorted(int(t) for t in poses.keys())
    n = len(timestamps)
    if n == 0:
        sys.exit(f"{src}/poses.npy holds no poses")

    features_db = read_shelve(os.path.join(src, "features"))
    depths_db = read_shelve(os.path.join(src, "depths"))
    vlad_db = read_shelve(os.path.join(src, "vlad_descriptors"))
    metadata = read_shelve(os.path.join(src, "metadata"))
    if "vlad_centres" not in metadata:
        sys.exit("metadata shelve has no vlad_centres — rebuild the map with this branch")
    centres = np.asarray(metadata["vlad_centres"])

    pose_matrices = np.stack([np.asarray(poses[t], dtype=np.float64) for t in timestamps])
    vlad_descriptors = np.stack([np.asarray(vlad_db[str(t)]) for t in timestamps])

    # Pack features with an offsets table (rows of kpts/descps/mask per keyframe).
    # IntKeyShelf stores str(ts) keys.
    kpts_parts, desc_parts, mask_parts, offsets = [], [], [], [0]
    for t in timestamps:
        feat = features_db.get(str(t))
        if feat is None:
            offsets.append(offsets[-1])
            continue
        k = canon_kpts(feat["kpts"])
        d = canon_descps(feat["descps"])
        m = canon_mask(feat["mask"])
        if d.shape[0] != k.shape[0] or m.shape[0] != k.shape[0]:
            sys.exit(f"features for ts={t} inconsistent: kpts {k.shape}, descps {d.shape}, mask {m.shape}")
        kpts_parts.append(k)
        desc_parts.append(d)
        mask_parts.append(m)
        offsets.append(offsets[-1] + k.shape[0])
    feature_kpts = np.concatenate(kpts_parts, axis=0) if kpts_parts else np.zeros((0, 2), np.float32)
    feature_descps = np.concatenate(desc_parts, axis=0) if desc_parts else np.zeros((0, 256), np.float32)
    feature_mask = np.concatenate(mask_parts, axis=0) if mask_parts else np.zeros((0,), np.uint8)

    depth_frames = []
    depth_hw = None
    for t in timestamps:
        if str(t) not in depths_db:
            sys.exit(f"depths shelve missing keyframe ts={t}")
        dep = np.asarray(depths_db[str(t)], dtype=np.float32)
        if dep.ndim != 2:
            sys.exit(f"depth for ts={t} is not HxW: {dep.shape}")
        if depth_hw is None:
            depth_hw = dep.shape
        elif dep.shape != depth_hw:
            sys.exit(f"depth for ts={t} shape {dep.shape} != {depth_hw}")
        depth_frames.append(dep)
    # u16 millimeters: half the f32 footprint (the dog's 13G depths.db becomes
    # ~6.5G). 0 stays "invalid", negatives clamp to 0, >65.535m clamps — the
    # reloc PnP cutoff is 50m. The C++ reader converts back to f32 meters per
    # candidate frame.
    depth_images = np.clip(
        np.stack(depth_frames, axis=0).astype(np.float64) * 1000.0, 0.0,
        65535.0).round().astype(np.uint16)

    os.makedirs(dst, exist_ok=True)
    np.save(os.path.join(dst, "pose_timestamps.npy"), np.asarray(timestamps, dtype=np.int64))
    np.save(os.path.join(dst, "pose_matrices.npy"), pose_matrices)
    np.save(os.path.join(dst, "vlad_centres.npy"), centres)
    np.save(os.path.join(dst, "vlad_descriptors.npy"), vlad_descriptors)
    np.save(os.path.join(dst, "feature_offsets.npy"),
            np.asarray(offsets, dtype=np.int64))
    np.save(os.path.join(dst, "feature_kpts.npy"), feature_kpts)
    np.save(os.path.join(dst, "feature_descps.npy"), feature_descps)
    np.save(os.path.join(dst, "feature_mask.npy"), feature_mask)
    np.save(os.path.join(dst, "depth_images.npy"), depth_images)

    for name in ("intrinsics.npy", "occupancy_grid.npy", "occupancy_meta.npy",
                 "sdf_map.npy", "path_speed.npy", "path_climb.npy"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst, name))

    meta = {
        "format": "tinynav_map_v2",
        "version": 2,
        "depth_dtype": "u2_mm",
        "count": int(n),
        "depth_height": int(depth_hw[0]),
        "depth_width": int(depth_hw[1]),
        "desc_dim": int(feature_descps.shape[1]) if feature_descps.size else 0,
        "vlad_dim": int(centres.shape[1]) if centres.ndim == 2 else 0,
        "vlad_centres": int(centres.shape[0]) if centres.ndim == 2 else 0,
        "feature_rows": int(feature_kpts.shape[0]),
        "source": os.path.abspath(src),
    }
    with open(os.path.join(dst, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"map v2: {n} keyframes, {feature_kpts.shape[0]} feature rows, "
          f"vlad {vlad_descriptors.shape} centres {centres.shape}, depth {depth_images.shape}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
