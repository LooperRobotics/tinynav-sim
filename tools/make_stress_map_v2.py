#!/usr/bin/env python3
"""Generate a yishang-scale synthetic map-format-v2 directory for the laziness
regression test (test_alignment_mapping stress_map_stays_lazy).

Writes the v2 npys directly (no shelve roundtrip — the test exercises the C++
READER, so going through tools/export_map_v2.py would only add gigabytes of
scratch). Depth planes dominate on purpose: at the dog's real scale (9.6k
keyframes of 544x640 f32) they are the 13G that OOMed the eager loader.

Usage:
  python3 tools/make_stress_map_v2.py                    # fixtures/map_v2_stress/v2
  python3 tools/make_stress_map_v2.py --keyframes 3000 --out fixtures/map_v2_stress/v2
"""
import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fixtures/map_v2_stress/v2")
    parser.add_argument("--keyframes", type=int, default=4000)
    parser.add_argument("--h", type=int, default=544)
    parser.add_argument("--w", type=int, default=640)
    parser.add_argument("--feats-per-kf", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = args.keyframes
    k = args.feats_per_kf
    m = n * k

    def raw_npy(path, dtype, shape):
        """Create a writable npy via open_memmap and return the array view."""
        a = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.dtype(dtype), shape=shape)
        return a

    # timestamps: 15 Hz spacing
    base_ns = 1_726_000_000_000_000_000
    ts = (base_ns + np.arange(n, dtype=np.int64) * 66_000_000)
    np.save(os.path.join(args.out, "pose_timestamps.npy"), ts)

    # poses: 0.3 m walk along +x with identity rotation
    poses = np.zeros((n, 4, 4), dtype="<f8")
    poses[:, 0, 0] = 1.0
    poses[:, 1, 1] = 1.0
    poses[:, 2, 2] = 1.0
    poses[:, 3, 3] = 1.0
    poses[:, 0, 3] = np.arange(n, dtype=np.float64) * 0.3
    np.save(os.path.join(args.out, "pose_matrices.npy"), poses)

    centres = rng.random((4, 8)).astype("<f4") + 0.5
    descriptors = rng.random((n, 8)).astype("<f4")
    np.save(os.path.join(args.out, "vlad_centres.npy"), centres)
    np.save(os.path.join(args.out, "vlad_descriptors.npy"), descriptors)

    offsets = (np.arange(n + 1, dtype=np.int64) * k)
    np.save(os.path.join(args.out, "feature_offsets.npy"), offsets)
    kpts = np.empty((m, 2), dtype="<f4")
    kpts[:, 0] = rng.random(m) * args.w
    kpts[:, 1] = rng.random(m) * args.h
    np.save(os.path.join(args.out, "feature_kpts.npy"), kpts)
    descps = raw_npy(os.path.join(args.out, "feature_descps.npy"), "<f4", (m, 256))
    for i in range(n):
        descps[i * k:(i + 1) * k] = rng.random((k, 256))
    descps.flush()
    del descps
    np.save(os.path.join(args.out, "feature_mask.npy"), np.ones(m, dtype="|u1"))

    # u16 millimeters — the format the writer/reader pair actually uses.
    depth = raw_npy(
        os.path.join(args.out, "depth_images.npy"), "<u2", (n, args.h, args.w))
    for i in range(n):
        depth[i] = (rng.random((args.h, args.w)) * 20000.0).astype("<u2")
        if i % 500 == 0:
            depth.flush()
    depth.flush()
    del depth

    depth_gb = n * args.h * args.w * 2 / 1024**3
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"format": "tinynav_map_v2", "depth_dtype": "u2_mm",
                   "keyframes": n, "depth_gb": round(depth_gb, 2)}, f)
    print(f"wrote {args.out}: {n} keyframes, depth {depth_gb:.2f}G (u16 mm) — "
          f"eager load would need ~{depth_gb:.1f}G RSS, lazy stays ~flat")


if __name__ == "__main__":
    main()
