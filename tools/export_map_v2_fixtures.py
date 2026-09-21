#!/usr/bin/env python3
"""Build the map-v2 alignment fixtures.

Creates a tiny synthetic map in the RAW Python format (pickled poses.npy +
TinyNavDB shelve databases — what map_node.py::load_map consumes), runs
tools/export_map_v2.py on it, and writes the expected v2 arrays for the C++
gtest (test_alignment_mapping.cpp) to compare against.

Layout (all under fixtures/map_v2/, gitignored):
  raw/          synthetic Python-format map (poses.npy + shelve dbs)
  v2/           exporter output the C++ reader must load
  expected/     ground-truth arrays the test compares against

Run inside the container:  python3 tools/export_map_v2_fixtures.py
"""
import importlib.util
import os
import shelve
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "fixtures", "map_v2")

N_KEYFRAMES = 5
N_FEATURES = 512
DESC_DIM = 256
DEPTH_H, DEPTH_W = 12, 16
VLAD_DIM = 8
VLAD_CENTRES = 4


def write_raw(raw_dir: str) -> list:
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir)

    rng = np.random.default_rng(7)
    timestamps = [1726000000000000000 + i * 500_000_000 for i in range(N_KEYFRAMES)]
    poses = {}
    for i, ts in enumerate(timestamps):
        pose = np.eye(4)
        pose[:3, 3] = [float(i) * 0.4, -0.1 * i, 0.3]
        angle = 0.1 * i
        pose[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        poses[ts] = pose
    np.save(os.path.join(raw_dir, "poses.npy"), poses, allow_pickle=True)

    features = shelve.open(os.path.join(raw_dir, "features"))
    depths = shelve.open(os.path.join(raw_dir, "depths"))
    vlad = shelve.open(os.path.join(raw_dir, "vlad_descriptors"))
    metadata = shelve.open(os.path.join(raw_dir, "metadata"))
    try:
        for i, ts in enumerate(timestamps):
            features[str(ts)] = {
                # engine layout: [1, N, 2]
                "kpts": rng.uniform(0, 847, (1, N_FEATURES, 2)).astype(np.float32),
                "descps": rng.uniform(-1, 1, (1, N_FEATURES, DESC_DIM)).astype(np.float32),
                "mask": rng.integers(0, 2, (1, N_FEATURES, 1)).astype(np.uint8),
            }
            depths[str(ts)] = rng.uniform(0.2, 9.0, (DEPTH_H, DEPTH_W)).astype(np.float32)
            vlad[str(ts)] = rng.uniform(-1, 1, (VLAD_DIM,)).astype(np.float32)
        metadata["vlad_centres"] = rng.uniform(-1, 1, (VLAD_CENTRES, VLAD_DIM)).astype(np.float32)
    finally:
        for db in (features, depths, vlad, metadata):
            db.close()
    return timestamps


def canon_kpts(kpts: np.ndarray) -> np.ndarray:
    a = np.asarray(kpts)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    return a.astype(np.float32)


def main() -> None:
    raw_dir = os.path.join(FIXTURES, "raw")
    v2_dir = os.path.join(FIXTURES, "v2")
    expected_dir = os.path.join(FIXTURES, "expected")
    if os.path.exists(expected_dir):
        shutil.rmtree(expected_dir)
    os.makedirs(expected_dir)

    timestamps = write_raw(raw_dir)
    if os.path.exists(v2_dir):
        shutil.rmtree(v2_dir)

    spec = importlib.util.spec_from_file_location(
        "export_map_v2", os.path.join(HERE, "export_map_v2.py"))
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    exporter.main(raw_dir, v2_dir)

    poses = np.load(os.path.join(raw_dir, "poses.npy"), allow_pickle=True).item()
    np.save(os.path.join(expected_dir, "pose_timestamps.npy"),
            np.asarray(timestamps, dtype=np.int64))
    np.save(os.path.join(expected_dir, "pose_matrices.npy"),
            np.stack([poses[t] for t in timestamps]).astype(np.float64))

    feats = shelve.open(os.path.join(raw_dir, "features"))
    depths = shelve.open(os.path.join(raw_dir, "depths"))
    vlad = shelve.open(os.path.join(raw_dir, "vlad_descriptors"))
    metadata = shelve.open(os.path.join(raw_dir, "metadata"))
    try:
        np.save(os.path.join(expected_dir, "vlad_centres.npy"),
                np.asarray(metadata["vlad_centres"]))
        np.save(os.path.join(expected_dir, "vlad_descriptors.npy"),
                np.stack([np.asarray(vlad[str(t)]) for t in timestamps]))
        np.save(os.path.join(expected_dir, "feature_kpts.npy"),
                np.concatenate([canon_kpts(feats[str(t)]["kpts"]) for t in timestamps]))
        np.save(os.path.join(expected_dir, "feature_descps.npy"),
                np.concatenate([np.asarray(feats[str(t)]["descps"])[0] for t in timestamps]))
        np.save(os.path.join(expected_dir, "feature_mask.npy"),
                np.concatenate([np.asarray(feats[str(t)]["mask"]).reshape(-1) for t in timestamps]))
        np.save(os.path.join(expected_dir, "depth_images.npy"),
                np.stack([np.asarray(depths[str(t)], dtype=np.float32) for t in timestamps]))
    finally:
        for db in (feats, depths, vlad, metadata):
            db.close()

    np.save(os.path.join(expected_dir, "feature_offsets.npy"),
            np.load(os.path.join(v2_dir, "feature_offsets.npy")))
    print(f"fixtures written under {FIXTURES}")


if __name__ == "__main__":
    sys.exit(main())
