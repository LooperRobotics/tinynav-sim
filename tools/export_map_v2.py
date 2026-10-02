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
load without pickle/shelve, all rows ordered by ascending timestamp. The
canonical writer is tools/mapio/writer.py (V2NpyWriter); this tool is a thin
v1-reader front end over it. Output layout, including the semantic sidecar
(semantic_embeddings.npy + semantic_meta.json), is documented there.

Usage:  python3 tools/export_map_v2.py <python_map_dir> <v2_output_dir>
Runs with the image's python3 (numpy + shelve stdlib only, no ROS imports).
"""
import json
import os
import shelve
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mapio.writer import V2NpyWriter  # noqa: E402


def read_shelve(path: str) -> dict:
    db = shelve.open(path, flag="r")
    try:
        return dict(db)
    finally:
        db.close()


def main(src: str, dst: str) -> None:
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

    writer = V2NpyWriter(dst)
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
    # Optional: carry SigLIP semantic embeddings over as the v2 sidecar.
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
