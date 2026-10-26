#!/usr/bin/env python3
"""Decimate robot visual meshes (OBJ/STL) into MuJoCo binary MSH.

Same recipe as the go2 set in the model/ bundle (model/go2/README.md):
open3d quadric decimation keeping RATIO of the triangles, boundary_weight 10
(holds thin shell borders -- fast-simplification punches holes there), no
normals in the MSH (MuJoCo computes them). Meshes under MIN_FACES triangles
are only re-encoded. Collision never uses these meshes (collision geoms are
primitives in the robot MJCF), so this is purely visual.

    python decimate_mesh.py <src dir> <dst dir> [--ratio 0.15]

OBJ geometry is parsed here (v/f records, n-gons fan-triangulated like
MuJoCo's loader; open3d's OBJ reader drops the quads some exports carry),
STL goes through open3d. Needs open3d (not a sim dependency: run it in any
venv that has it).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import open3d as o3d

MIN_FACES = 2000


def write_msh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    v = vertices[used].astype(np.float32)
    f = remap[faces].astype(np.int32)
    with open(path, "wb") as out:
        out.write(struct.pack("<4i", len(v), 0, 0, len(f)))
        out.write(v.tobytes())
        out.write(f.tobytes())


def read_obj(path: Path) -> o3d.geometry.TriangleMesh:
    vertices, faces = [], []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            vertices.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
            faces += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices)),
                                     o3d.utility.Vector3iVector(np.asarray(faces)))


def decimate(src: Path, dst: Path, ratio: float) -> tuple[int, int]:
    if src.suffix.lower() == ".obj":
        mesh = read_obj(src)
    else:
        mesh = o3d.io.read_triangle_mesh(str(src))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    n = len(mesh.triangles)
    if n > MIN_FACES:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=max(MIN_FACES, int(n * ratio)),
            boundary_weight=10.0)
    write_msh(dst, np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return n, len(mesh.triangles)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--ratio", type=float, default=0.15)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.src.iterdir()):
        if path.suffix.lower() not in (".obj", ".stl"):
            continue
        before, after = decimate(path, args.dst / (path.stem + ".msh"), args.ratio)
        print(f"{path.name}: {before} -> {after} faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
