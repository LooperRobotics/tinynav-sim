#!/usr/bin/env python3
"""Convert a map-format-v2 directory back to the v1 (python) layout.

Bridge direction v2 -> v1 (the v1 -> v2 direction is tools/export_map_v2.py).
Value loop: sim/C++-era maps become consumable by the main-line python
toolchain (map_node.load_map, retrieval_map_by_language, convert_to_colmap).

Written v1 pieces (main-line consumers verified against
reference/tinynav/core/map_node.py::load_map + tool/retrieval_map_by_language.py):
  poses.npy               pickled dict {ts_ns: 4x4}        (load_map)
  intrinsics.npy          plain copy
  metadata db             shelve {"vlad_centres"}          (TinyNavDB.metadata)
  vlad_descriptors db     shelve {str(ts): (d,)}
  features db             shelve {str(ts): {"kpts"[1,N,2], "descps"[1,N,D], "mask"[1,N,1]}}
  depths db               shelve {str(ts): HxW f32 meters} (u16 mm -> meters)
  semantic_embeddings db  shelve {str(ts): (768,)}         (retrieval tool)
  occupancy_grid/meta, sdf_map, path_speed, path_climb    plain copies

NOT written (v2 has no data for them): rgb_images_db / infra1_images_db
(VideoDB), embeddings.db (DINOv2 global), patch_tokens.db. Consequences:
retrieval ranking works but saving a result image needs images; reloop-style
tools that read map DINO embeddings are not covered.

Usage:  python3 tools/convert_v2_to_v1.py <v2_map_dir> <v1_output_dir>
"""
import os
import shelve
import shutil
import sys

import numpy as np


def _open_shelf(path_no_suffix: str) -> shelve.Shelf:
    return shelve.open(path_no_suffix)


def main(src: str, dst: str) -> None:
    ts = np.load(os.path.join(src, "pose_timestamps.npy"))
    poses = np.load(os.path.join(src, "pose_matrices.npy"))
    n = len(ts)
    os.makedirs(dst, exist_ok=True)

    np.save(os.path.join(dst, "poses.npy"),
            {int(t): poses[i] for i, t in enumerate(ts)}, allow_pickle=True)

    vlad = np.load(os.path.join(src, "vlad_descriptors.npy"))
    centres = np.load(os.path.join(src, "vlad_centres.npy"))
    meta = _open_shelf(os.path.join(dst, "metadata"))
    meta["vlad_centres"] = centres
    meta.close()

    vlad_db = _open_shelf(os.path.join(dst, "vlad_descriptors"))
    for i, t in enumerate(ts):
        vlad_db[str(int(t))] = vlad[i]
    vlad_db.close()

    offsets = np.load(os.path.join(src, "feature_offsets.npy"))
    kpts = np.load(os.path.join(src, "feature_kpts.npy"))
    descps = np.load(os.path.join(src, "feature_descps.npy"))
    mask = np.load(os.path.join(src, "feature_mask.npy"))
    feats_db = _open_shelf(os.path.join(dst, "features"))
    for i, t in enumerate(ts):
        r0, r1 = int(offsets[i]), int(offsets[i + 1])
        d = descps.shape[1]
        feats_db[str(int(t))] = {
            "kpts": kpts[r0:r1].reshape(1, r1 - r0, 2),
            "descps": descps[r0:r1].reshape(1, r1 - r0, d),
            "mask": mask[r0:r1].reshape(1, r1 - r0, 1),
        }
    feats_db.close()

    depth = np.load(os.path.join(src, "depth_images.npy"))
    depth_db = _open_shelf(os.path.join(dst, "depths"))
    for i, t in enumerate(ts):
        depth_db[str(int(t))] = depth[i].astype(np.float32) / 1000.0
    depth_db.close()

    sem_path = os.path.join(src, "semantic_embeddings.npy")
    if os.path.exists(sem_path):
        sem = np.load(sem_path)
        norms = np.linalg.norm(sem, axis=1)
        sem_db = _open_shelf(os.path.join(dst, "semantic_embeddings"))
        n_sem = 0
        for i, t in enumerate(ts):
            if norms[i] > 0.0:
                sem_db[str(int(t))] = sem[i]
                n_sem += 1
        sem_db.close()
    else:
        n_sem = 0

    for name in ("intrinsics.npy", "occupancy_grid.npy", "occupancy_meta.npy",
                 "sdf_map.npy", "path_speed.npy", "path_climb.npy",
                 "baseline.npy", "T_rgb_to_infra1.npy", "rgb_camera_intrinsics.npy",
                 "mapping_continuous_odom.npy"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst, name))

    print(f"v1 map: {n} keyframes ({n_sem} semantic), vlad {vlad.shape}, "
          f"depth {depth.shape} -> {dst}")
    if n_sem == 0:
        print("WARNING: no semantic embeddings carried over")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
