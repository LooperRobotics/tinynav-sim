#!/usr/bin/env python3
"""Migrate a map directory to map format v3 (single-file SQLite).

Source layout is auto-detected:
  v2  pose_timestamps.npy present — plain-npy layout (tools/mapio/writer.py)
  v1  poses.npy present          — pickled poses + shelve dbs (the python
                                   TinyNavDB, read the same way as
                                   tools/export_map_v2.py)

v1 sources carry SigLIP embeddings in the semantic_embeddings shelve when the
map was built with the siglip2 builder branch; v2 sources carry them in the
semantic_embeddings.npy sidecar. Everything the v2 writer would have kept as
aux .npy files lands in the blobs table (vlad_centres/intrinsics/occupancy/
sdf/path_*/baseline/rgb_camera_intrinsics/...).

Usage:  /opt/venv/bin/python3 tools/migrate_map_to_v3.py <src_map_dir> <out_map.sqlite>
"""
import json
import os
import shelve
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mapio.sqlite_writer import SQLiteWriter  # noqa: E402

AUX_CORE_V2 = {
    "pose_timestamps.npy", "pose_matrices.npy", "vlad_centres.npy",
    "vlad_descriptors.npy", "feature_offsets.npy", "feature_kpts.npy",
    "feature_descps.npy", "feature_mask.npy", "depth_images.npy",
    "semantic_embeddings.npy",
}
AUX_CORE_V1 = {"poses.npy"}


def read_shelve(path: str) -> dict:
    db = shelve.open(path, flag="r")
    try:
        return dict(db)
    finally:
        db.close()


def detect_source(src: str) -> str:
    if os.path.exists(os.path.join(src, "pose_timestamps.npy")):
        return "v2"
    if os.path.exists(os.path.join(src, "poses.npy")):
        return "v1"
    sys.exit(f"{src}: neither pose_timestamps.npy (v2) nor poses.npy (v1) found")


def _writer_kwargs(src: str) -> dict:
    meta_path = os.path.join(src, "semantic_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            sm = json.load(fh)
        return {"semantic_model": sm.get("model", ""),
                "semantic_model_version": sm.get("model_version", "")}
    return {}


def migrate_v2(src: str, dst: str) -> None:
    timestamps = np.load(os.path.join(src, "pose_timestamps.npy"))
    poses = np.load(os.path.join(src, "pose_matrices.npy"))
    centres = np.load(os.path.join(src, "vlad_centres.npy"))
    vlads = np.load(os.path.join(src, "vlad_descriptors.npy"))
    offsets = np.load(os.path.join(src, "feature_offsets.npy"))
    kpts = np.load(os.path.join(src, "feature_kpts.npy"))
    descps = np.load(os.path.join(src, "feature_descps.npy"))
    mask = np.load(os.path.join(src, "feature_mask.npy"))
    depth = np.load(os.path.join(src, "depth_images.npy"))

    writer = SQLiteWriter(dst, **_writer_kwargs(src))
    writer.set_vlad_centres(np.asarray(centres))
    for i, ts in enumerate(timestamps):
        t = int(ts)
        writer.add_keyframe(t, poses[i])
        b, e = int(offsets[i]), int(offsets[i + 1])
        if e > b:
            writer.add_features(t, kpts[b:e], descps[b:e], mask[b:e])
        if depth.dtype == np.uint16:  # u2 millimeters -> f32 meters (lossless)
            writer.add_depth(t, depth[i].astype(np.float32) * 0.001)
        else:  # legacy f32 meters
            writer.add_depth(t, depth[i])
        writer.add_vlad(t, vlads[i])

    sem_path = os.path.join(src, "semantic_embeddings.npy")
    if os.path.exists(sem_path):
        sem = np.load(sem_path)
        for i, ts in enumerate(timestamps):
            if np.any(sem[i]):
                writer.add_semantic(int(ts), sem[i])

    for name in sorted(os.listdir(src)):
        if not name.endswith(".npy") or name in AUX_CORE_V2:
            continue
        writer.add_aux(name, np.load(os.path.join(src, name)))

    writer.finalize()


def migrate_v1(src: str, dst: str) -> None:
    poses = np.load(os.path.join(src, "poses.npy"), allow_pickle=True).item()
    timestamps = sorted(int(t) for t in poses.keys())
    if not timestamps:
        sys.exit(f"{src}/poses.npy holds no poses")

    features_db = read_shelve(os.path.join(src, "features"))
    depths_db = read_shelve(os.path.join(src, "depths"))
    vlad_db = read_shelve(os.path.join(src, "vlad_descriptors"))
    metadata = read_shelve(os.path.join(src, "metadata"))
    if "vlad_centres" not in metadata:
        sys.exit("metadata shelve has no vlad_centres — rebuild the map with this branch")

    writer = SQLiteWriter(dst)
    writer.set_vlad_centres(np.asarray(metadata["vlad_centres"]))
    for t in timestamps:
        writer.add_keyframe(t, np.asarray(poses[t], dtype=np.float64))
        feat = features_db.get(str(t))
        if feat is not None:
            writer.add_features(t, feat["kpts"], feat["descps"], feat["mask"])
        if str(t) not in depths_db:
            sys.exit(f"depths shelve missing keyframe ts={t}")
        writer.add_depth(t, np.asarray(depths_db[str(t)], dtype=np.float32))
        writer.add_vlad(t, np.asarray(vlad_db[str(t)]))

    sem_path = os.path.join(src, "semantic_embeddings")
    if os.path.exists(sem_path + ".db"):
        sem_db = read_shelve(sem_path)
        for t in timestamps:
            if str(t) in sem_db:
                writer.add_semantic(t, np.asarray(sem_db[str(t)], dtype=np.float32))

    for name in ("intrinsics.npy", "occupancy_grid.npy", "occupancy_meta.npy",
                 "sdf_map.npy", "path_speed.npy", "path_climb.npy"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            writer.add_aux(name, np.load(p))

    writer.finalize()


def main(src: str, dst: str) -> None:
    kind = detect_source(src)
    print(f"source: {kind} map at {src}")
    if kind == "v2":
        migrate_v2(src, dst)
    else:
        migrate_v1(src, dst)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
