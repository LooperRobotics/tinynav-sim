#!/usr/bin/env python3
"""Probe A+B: compare Python vs C++ SuperPointTRT on the same image.

B: keypoint count + spatial overlap.
A: descriptor equivalence + LightGlue cross-matching (py x py / py x cpp,
   img_shape 848x480 as hardcoded in map_node vs actual 544x480).
"""
import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))
from tinynav.core.models_trt import SuperPointTRT, LightGlueTRT  # noqa: E402


def as_xy(kpts: np.ndarray) -> np.ndarray:
    """Normalize wrapper kpts output ([1,2,N] or [1,N,2]) to (N,2) xy."""
    k = kpts[0]
    if k.shape[0] == 2 and k.ndim == 2:
        return k.T
    return k


def match_count(lg, f0, f1, shape):
    r = asyncio.run(lg.infer(f0["kpts"], f1["kpts"], f0["descps"], f1["descps"],
                             f0["mask"], f1["mask"], shape, shape))
    mi = r["match_indices"][0]
    return int((mi != -1).sum())


def main():
    img_path, out_dir = sys.argv[1], Path(sys.argv[2])
    img = cv2.imread(img_path, 0)
    assert img is not None
    h, w = img.shape
    print(f"image {w}x{h}")

    # ---- Python wrapper ----
    sp = SuperPointTRT()
    pres = asyncio.run(sp.infer(img))
    pk = as_xy(pres["kpts"])
    print(f"[B] python: N={len(pk)} kpts shape={pres['kpts'].shape} "
          f"descps shape={pres['descps'].shape} dtype={pres['descps'].dtype} "
          f"mask shape={pres['mask'].shape}")

    # ---- C++ wrapper output ----
    ck = np.load(out_dir / "cpp_kpts.npy")
    cd = np.load(out_dir / "cpp_descps.npy")
    cm = np.load(out_dir / "cpp_mask.npy")
    ck_xy = as_xy(ck)
    print(f"[B] cpp:    N={len(ck_xy)} kpts shape={ck.shape} "
          f"descps shape={cd.shape} mask shape={cm.shape}")

    # ---- B: spatial overlap ----
    from scipy.spatial import cKDTree
    tree = cKDTree(ck_xy)
    d, _ = tree.query(pk, k=1)
    for tol in (1.0, 2.0, 4.0):
        print(f"[B] python kpts with cpp neighbor within {tol}px: {(d <= tol).sum()}/{len(pk)}")
    tree2 = cKDTree(pk)
    d2, _ = tree2.query(ck_xy, k=1)
    print(f"[B] cpp kpts with python neighbor within 2px: {(d2 <= 2.0).sum()}/{len(ck_xy)}")

    # ---- A: descriptor equivalence on co-located pairs ----
    dd = cd[0] if cd.ndim == 3 else cd
    pd_ = pres["descps"][0] if pres["descps"].ndim == 3 else pres["descps"]
    _, nearest_idx = tree.query(pk, k=1)
    take = np.argsort(d)[: min(200, len(pk))]
    sims = []
    for i in take:
        a = pd_[i].astype(np.float32)
        b = dd[nearest_idx[i]].astype(np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            sims.append(float(a @ b / (na * nb)))
    if sims:
        sims = np.array(sims)
        print(f"[A] desc cosine sim on nearest pairs: mean={sims.mean():.4f} "
              f"min={sims.min():.4f} p5={np.percentile(sims, 5):.4f} "
              f"(n={len(sims)}, >0.99: {(sims > 0.99).sum()})")

    # ---- A: LightGlue cross-matching ----
    lg = LightGlueTRT()
    f_py = {"kpts": pres["kpts"], "descps": pres["descps"], "mask": pres["mask"]}
    f_cpp = {"kpts": ck, "descps": cd, "mask": cm.astype(bool)}
    s848 = np.array([848, 480], dtype=np.int64)
    s_real = np.array([w, h], dtype=np.int64)
    print(f"[A] LG py x py   shape=848x480: {match_count(lg, f_py, f_py, s848)}")
    print(f"[A] LG cpp x cpp shape=848x480: {match_count(lg, f_cpp, f_cpp, s848)}")
    print(f"[A] LG py x cpp  shape=848x480: {match_count(lg, f_py, f_cpp, s848)}  <- production config")
    print(f"[A] LG py x cpp  shape={w}x{h}: {match_count(lg, f_py, f_cpp, s_real)}")
    print(f"[A] LG py x py   shape={w}x{h}: {match_count(lg, f_py, f_py, s_real)}")


if __name__ == "__main__":
    main()
