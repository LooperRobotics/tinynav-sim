#!/usr/bin/env python3
"""Generate COLMAP match pairs from the de-tilted LIO poses (transforms.json).

Pair types (plan §3.1):
  - stereo:     left-right of the same frame (21 ms exposure offset, natural
                short baseline)
  - temporal:   same camera, index distance <= --temporal (2 Hz -> 0.5 s step)
  - spatial:    any pair with camera-center distance < --max-dist and viewing
                direction angle < --max-angle, capped at --topk nearest per
                image (covers multi-height loops / revisits that pure
                sequential matching would miss)

Output: colmap/pairs.txt with "name1 name2" lines (COLMAP match-file format).
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np

MAP_DIR = Path(os.environ.get("METACAM_SCENE", ".")).resolve()  # 场景目录：METACAM_SCENE 环境变量或 CWD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nerfstudio", default=str(MAP_DIR / "nerfstudio"))
    ap.add_argument("--out", default=str(MAP_DIR / "colmap" / "pairs.txt"))
    ap.add_argument("--temporal", type=int, default=10)
    ap.add_argument("--max-dist", type=float, default=2.0)
    ap.add_argument("--max-angle", type=float, default=30.0)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    meta = json.load(open(Path(args.nerfstudio) / "transforms.json"))
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    names = [f["file_path"].split("/")[-1] for f in frames]
    C = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames])  # cam centers
    # viewing direction: OpenGL c2w looks along -z
    D = np.array([-np.array(f["transform_matrix"])[:3, 2] for f in frames])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    n = len(frames)
    print(f"{n} frames")

    pairs = set()

    # stereo: nearest timestamp across sides (exposure offset ~21-41 ms)
    ts = np.array([int(name[:-4].split("_")[0]) / 1000.0 for name in names])
    by_side = {}
    for i, name in enumerate(names):
        by_side.setdefault(name[:-4].rsplit("_", 1)[1], []).append(i)
    for side, idx in by_side.items():
        idx = sorted(idx, key=lambda i: ts[i])
        t_side = ts[idx]
        other_side = "right" if side == "left" else "left"
        idx_o = sorted(by_side[other_side], key=lambda i: ts[i])
        t_o = ts[idx_o]
        for a, i in enumerate(idx):
            b = np.searchsorted(t_o, t_side[a])
            for bb in (b - 1, b):
                if 0 <= bb < len(idx_o) and abs(t_o[bb] - t_side[a]) < 0.2:
                    pairs.add((min(i, idx_o[bb]), max(i, idx_o[bb])))
    n_stereo = len(pairs)

    # temporal: same side, index +-k (by_side already built for stereo)
    for side, idx in by_side.items():
        for a, i in enumerate(idx):
            for j in idx[a + 1: a + 1 + args.temporal]:
                pairs.add((i, j))
    n_temporal = len(pairs) - n_stereo

    # spatial: distance + angle, chunked full scan, topk per image
    ang_th = np.cos(np.radians(args.max_angle))
    cand = [[] for _ in range(n)]
    for s in range(0, n, 512):
        e = min(s + 512, n)
        d = np.linalg.norm(C[:, None, :] - C[None, s:e, :], axis=-1)  # (n, e-s)
        cosang = np.clip(D @ D[s:e].T, -1, 1)
        ok = (d < args.max_dist) & (cosang > ang_th)
        for c in range(e - s):
            rows = np.nonzero(ok[:, c])[0]
            rows = rows[rows != s + c]
            if len(rows):
                for r in rows:
                    cand[r].append((d[r, c], s + c))
    n_spatial = 0
    for i, lst in enumerate(cand):
        lst.sort()
        for d, j in lst[: args.topk]:
            key = (min(i, j), max(i, j))
            if key not in pairs:
                pairs.add(key)
                n_spatial += 1
    print(f"pairs: stereo {n_stereo}, temporal +{n_temporal}, spatial +{n_spatial}, total {len(pairs)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, j in sorted(pairs):
            f.write(f"{names[i]} {names[j]}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
