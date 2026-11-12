#!/usr/bin/env python3
"""Stage 1: COLMAP prior-pose BA pipeline for map_2 (plan §3).

Sub-commands (run in order; each is checkpointed so steps can be re-run):
  init         build map_2/colmap/: images/ symlink farm (half-res) + sparse_init/
               (3.x-format binary model: calibrated OPENCV_FISHEYE k=0 cameras,
               LIO w2c poses, empty points3D)
  extract      SIFT extraction into colmap/database.db (pycolmap, CPU);
               --limit N for the timing bench
  match        match + geometric-verify pairs from colmap/pairs.txt;
               --bench N runs only the first N pairs (timing bench)
  triangulate  point_triangulator -> colmap/sparse_tri + stats
  ba           bundle adjustment (poses + points; intrinsics fixed) -> colmap/sparse_ba
               + pose-delta stats vs LIO + acceptance gates
  align        Umeyama sim3 BA->LIO on camera centers -> colmap/sparse_ba_aligned
               + nerfstudio_ba/transforms_ba.json (full-res intrinsics) + report

Half-resolution rationale: SIFT CPU cost ~4x lower; poses are
resolution-independent; training also runs at downscale 2.
"""

import os
import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MAP_DIR = Path(os.environ.get("METACAM_SCENE", ".")).resolve()  # 场景目录：METACAM_SCENE 环境变量或 CWD
COLMAP_DIR = MAP_DIR / "colmap"
NS_DIR = MAP_DIR / "nerfstudio"
DB_PATH = COLMAP_DIR / "database.db"
PAIRS_PATH = COLMAP_DIR / "pairs.txt"
OPENCV_FISHEYE_MODEL_ID = 5


# ------------------------------------------------------------ shared utils ----
def load_calib():
    calib = json.load(open(MAP_DIR / "calib" / "calibration.json"))
    cams = {}
    for c in calib["cameras"]:
        cams[c["name"]] = dict(
            fx=c["intrinsic"]["fl_x"] / 2, fy=c["intrinsic"]["fl_y"] / 2,
            cx=c["intrinsic"]["cx"] / 2, cy=c["intrinsic"]["cy"] / 2,
            w=c["width"] // 2, h=c["height"] // 2,
        )
    return cams


def frame_data():
    meta = json.load(open(NS_DIR / "transforms.json"))
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    names = [f["file_path"].split("/")[-1] for f in frames]
    # OpenGL c2w -> OpenCV c2w (flip y/z columns) -> w2c
    w2c = []
    for f in frames:
        T = np.array(f["transform_matrix"])
        T[:, 1:3] *= -1
        w2c.append(np.linalg.inv(T))
    return names, w2c


def quat_wxyz_from_mat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


def umeyama(src, dst):
    """sim3: dst ~ s @ R @ src + t. Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    S_ = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_[2, 2] = -1
    R = U @ S_ @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(S) @ S_) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def pose_deltas(rec, names, w2c):
    """BA vs LIO pose deltas: translation (m) and rotation (deg)."""
    dt, dr = [], []
    for name, T in zip(names, w2c):
        im = rec.find_image_with_name(name)
        if im is None or not im.has_pose:
            continue
        p = im.cam_from_world()
        T_ba = np.eye(4)
        T_ba[:3, :3] = p.rotation.matrix()
        T_ba[:3, 3] = p.translation
        dT = np.linalg.inv(T) @ T_ba
        dt.append(float(np.linalg.norm(dT[:3, 3])))
        c = np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)
        dr.append(float(np.degrees(np.arccos(c))))
    return np.array(dt), np.array(dr)


def print_gates(rec, dt, dr, half=True):
    unit = "px@half-res" if half else "px"
    reproj = rec.compute_mean_reprojection_error()
    n_pts = rec.num_points3D()
    trk = rec.compute_mean_track_length()
    print(f"\n===== acceptance gates (plan §3.4) =====")
    print(f"points3D: {n_pts}   mean track: {trk:.1f}   mean reproj: {reproj:.3f} {unit}")
    print(f"pose deltas vs LIO: n={len(dt)}  median {np.median(dt)*100:.1f} cm / {np.median(dr):.2f} deg  "
          f"p95 {np.percentile(dt,95)*100:.1f} cm / {np.percentile(dr,95):.2f} deg")
    print(f"  >10cm: {(dt>0.1).mean()*100:.1f}%   >5deg: {(dr>5).mean()*100:.1f}%")
    g1 = reproj < 1.0
    g2 = np.median(dt) < 0.05 and np.median(dr) < 1.0
    g3 = (dt > 0.1).mean() < 0.05 and (dr > 5).mean() < 0.05
    g4 = n_pts > 500_000
    for name, ok in [("reproj<1px", g1), ("median cm-level", g2), ("no pollution (<5%)", g3), ("tracks>500k", g4)]:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return dict(reproj=float(reproj), n_points=int(n_pts), track=float(trk),
                dt_med=float(np.median(dt)), dr_med=float(np.median(dr)),
                dt_p95=float(np.percentile(dt, 95)), dr_p95=float(np.percentile(dr, 95)),
                frac_dt_gt10cm=float((dt > 0.1).mean()), frac_dr_gt5deg=float((dr > 5).mean()),
                gates=dict(reproj=bool(g1), median=bool(g2), pollution=bool(g3), tracks=bool(g4)))


# ------------------------------------------------------------------ steps ----
def cmd_init(args):
    names, w2c = frame_data()
    img_dir = COLMAP_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        link = img_dir / name
        if not link.exists():
            link.symlink_to(NS_DIR / "images_2" / name)
    print(f"init: {len(names)} images -> {img_dir} (symlinks)")


def cmd_extract(args):
    import pycolmap
    cams = load_calib()
    c = cams["left"]  # camera 1 created by the reader; right camera patched in below
    opts = pycolmap.FeatureExtractionOptions()
    opts.num_threads = args.threads
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV_FISHEYE"
    reader.camera_params = f"{c['fx']:.6f},{c['fy']:.6f},{c['cx']:.6f},{c['cy']:.6f},0,0,0,0"
    names = getattr(args, "names", None) or frame_data()[0]
    if args.limit:
        names = names[: args.limit]
    t0 = time.time()
    pycolmap.extract_features(
        DB_PATH, COLMAP_DIR / "images", image_names=names,
        camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=reader,
        extraction_options=opts, device=pycolmap.Device.cpu,
    )
    el = time.time() - t0

    db = pycolmap.Database.open(str(DB_PATH))
    # PER_IMAGE mode gives every image its own camera row; overwrite each with
    # the exact per-side calibration (left/right params, k=0)
    cam_by_side = {}
    for im in db.read_all_images():
        cam_by_side[im.camera_id] = "right" if "_right." in im.name else "left"
    n_left = n_right = 0
    for cam in db.read_all_cameras():
        side = cam_by_side.get(cam.camera_id)
        if side is None:
            continue
        cc = cams[side]
        cam.model = pycolmap.CameraModelId.OPENCV_FISHEYE
        cam.width, cam.height = cc["w"], cc["h"]
        cam.params = np.array([cc["fx"], cc["fy"], cc["cx"], cc["cy"], 0.0, 0.0, 0.0, 0.0])
        db.update_camera(cam)
        if side == "left":
            n_left += 1
        else:
            n_right += 1
    n_imgs, kp = db.num_images(), db.num_keypoints()
    db.close()
    print(f"extracted: {n_imgs} images, {kp} keypoints; cameras patched: {n_left} left, {n_right} right")

    # rebuild sparse_init mirroring the db structure exactly (PER_IMAGE: one
    # camera + one trivial rig per image) — the 4.x triangulator checks that
    # rig ids agree between the reconstruction and the database
    build_sparse_init_from_db()
    if args.limit:
        print(f"BENCH: {el:.1f}s for {len(names)} imgs -> {el/len(names)*1000:.0f} ms/img "
              f"-> full 3682 est {el/len(names)*3682/60:.1f} min")


def build_sparse_init_from_db():
    """Write sparse_init/ (3.x bins) straight from the db: db image ids, db
    camera ids and rows (per-side calibrated params), LIO w2c poses, no points."""
    import pycolmap
    db = pycolmap.Database.open(str(DB_PATH))
    db_imgs = [(im.image_id, im.name, im.camera_id) for im in db.read_all_images()]
    db_cams = {c.camera_id: c for c in db.read_all_cameras()}
    db.close()
    names, w2c = frame_data()
    pose_by_name = {n: T for n, T in zip(names, w2c)}
    missing = [name for _, name, _ in db_imgs if name not in pose_by_name]
    if missing:
        print(f"WARNING: {len(missing)} db images without LIO pose")

    sparse = COLMAP_DIR / "sparse_init"
    sparse.mkdir(parents=True, exist_ok=True)
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(db_cams)))
        for cam_id in sorted(db_cams):
            c = db_cams[cam_id]
            f.write(struct.pack("<iiQQ", cam_id, int(c.model), c.width, c.height))
            f.write(struct.pack("<8d", *[float(p) for p in c.params]))
    rows = sorted((iid, name, cid) for iid, name, cid in db_imgs)
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rows)))
        for iid, name, cid in rows:
            T = pose_by_name[name]
            q = quat_wxyz_from_mat(T[:3, :3])
            f.write(struct.pack("<i", iid))
            f.write(struct.pack("<7d", *q, *T[:3, 3]))
            f.write(struct.pack("<i", cid))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 0))
    print(f"sparse_init rebuilt from db: {len(rows)} images, {len(db_cams)} cameras")


def cmd_match(args):
    import pycolmap
    pairs_path = PAIRS_PATH
    if args.bench:
        lines = open(pairs_path).read().splitlines()[: args.bench]
        pairs_path = COLMAP_DIR / "pairs_bench.txt"
        pairs_path.write_text("\n".join(lines) + "\n")
    pairing = pycolmap.ImportedPairingOptions()
    pairing.match_list_path = str(pairs_path)
    verif = pycolmap.TwoViewGeometryOptions()
    verif.min_num_inliers = 15
    verif.min_E_F_inlier_ratio = 0.25
    verif.detect_watermark = False
    verif.ransac.max_error = args.ransac_error
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.num_threads = args.threads
    match_opts.use_gpu = False
    t0 = time.time()
    pycolmap.match_image_pairs(DB_PATH, matching_options=match_opts,
                               pairing_options=pairing, verification_options=verif)
    el = time.time() - t0
    db = pycolmap.Database.open(str(DB_PATH))
    n_ver = db.num_verified_image_pairs()
    n_inl = db.num_inlier_matches()
    db.close()
    n_pairs = args.bench or sum(1 for _ in open(PAIRS_PATH))
    print(f"verified pairs: {n_ver}/{n_pairs}, total inliers: {n_inl}")
    if args.bench:
        print(f"BENCH: {el:.1f}s for {n_pairs} pairs -> {el/n_pairs*1000:.0f} ms/pair "
              f"-> full est {el/n_pairs*sum(1 for _ in open(PAIRS_PATH))/60:.1f} min")
    else:
        print(f"match+verify took {el/60:.1f} min")


def cmd_triangulate(args):
    import pycolmap
    rec = pycolmap.Reconstruction(str(COLMAP_DIR / "sparse_init"))
    print(f"init rec: {rec.num_images()} images, {rec.num_reg_frames()} reg frames")
    t0 = time.time()
    out = pycolmap.triangulate_points(
        rec, DB_PATH, COLMAP_DIR / "images", str(COLMAP_DIR / "sparse_tri"),
        clear_points=True, refine_intrinsics=False,
    )
    print(f"triangulation took {(time.time()-t0)/60:.1f} min")
    print(f"points3D: {out.num_points3D()}, mean track: {out.compute_mean_track_length():.1f}, "
          f"mean reproj: {out.compute_mean_reprojection_error():.3f} px, "
          f"reg images: {out.num_reg_images()}")


def cmd_ba(args):
    import pycolmap
    rec = pycolmap.Reconstruction(str(COLMAP_DIR / "sparse_tri"))
    names, w2c = frame_data()
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = False
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    opts.refine_sensor_from_rig = False
    opts.refine_rig_from_world = True
    opts.refine_points3D = True
    opts.min_track_length = 3
    opts.print_summary = True
    opts.ceres.solver_options.max_num_iterations = args.max_iters
    opts.ceres.solver_options.num_threads = args.threads
    t0 = time.time()
    pycolmap.bundle_adjustment(rec, opts)
    print(f"BA took {(time.time()-t0)/60:.1f} min")
    (COLMAP_DIR / "sparse_ba").mkdir(parents=True, exist_ok=True)
    rec.write(str(COLMAP_DIR / "sparse_ba"))
    dt, dr = pose_deltas(rec, names, w2c)
    stats = print_gates(rec, dt, dr)
    json.dump(stats, open(COLMAP_DIR / "ba_stats.json", "w"), indent=2)


def cmd_align(args):
    import pycolmap
    rec = pycolmap.Reconstruction(str(COLMAP_DIR / "sparse_ba"))
    # drop degenerate points (short tracks the BA excluded, runaway positions):
    # ~13% of the cloud at >10 px reproj destroys all mean statistics
    bad = [pid for pid, p in rec.points3D.items() if not np.isfinite(p.error) or p.error > 10.0]
    for pid in bad:
        rec.delete_point3D(pid)
    print(f"dropped {len(bad)} degenerate points, kept {rec.num_points3D()}")
    errs = np.array([p.error for p in rec.points3D.values()])
    print(f"kept points reproj: median {np.median(errs):.3f} px, mean {errs.mean():.3f} px")
    names, w2c = frame_data()
    # camera centers: BA vs LIO (center = -R^T t of the world-to-camera pose)
    src, dst = [], []
    for name, T in zip(names, w2c):
        im = rec.find_image_with_name(name)
        if im is None or not im.has_pose:
            continue
        p = im.cam_from_world()
        R = p.rotation.matrix()
        src.append(-R.T @ np.array(p.translation))
        dst.append(np.linalg.inv(T)[:3, 3])
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
    print(f"Umeyama: scale={s:.6f}, center-residual median={np.median(resid)*100:.2f} cm, "
          f"p95={np.percentile(resid,95)*100:.2f} cm  (pure gauge fit)")

    sim3 = pycolmap.Sim3d(float(s), pycolmap.Rotation3d(R), t)
    rec.transform(sim3)
    out_dir = COLMAP_DIR / "sparse_ba_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    dt, dr = pose_deltas(rec, names, w2c)
    print(f"after align: median delta {np.median(dt)*100:.1f} cm / {np.median(dr):.2f} deg, "
          f"p95 {np.percentile(dt,95)*100:.1f} cm / {np.percentile(dr,95):.2f} deg")

    # transforms_ba.json: swap poses, keep full-res intrinsics + LiDAR init ply
    meta = json.load(open(NS_DIR / "transforms.json"))
    meta["frames"] = sorted(meta["frames"], key=lambda f: f["file_path"])
    n_set = 0
    for f in meta["frames"]:
        name = f["file_path"].split("/")[-1]
        im = rec.find_image_with_name(name)
        assert im is not None and im.has_pose, name
        p = im.cam_from_world()
        T_w2c = np.eye(4)
        T_w2c[:3, :3] = p.rotation.matrix()
        T_w2c[:3, 3] = p.translation
        T_c2w = np.linalg.inv(T_w2c)
        T_c2w[:, 1:3] *= -1  # OpenCV -> OpenGL
        f["transform_matrix"] = T_c2w.tolist()
        n_set += 1
    ns_ba = MAP_DIR / "nerfstudio_ba"
    ns_ba.mkdir(exist_ok=True)
    json.dump(meta, open(ns_ba / "transforms_ba.json", "w"), indent=2)
    for sub in ("images_2", "masks_2", "points3d.ply"):
        link = ns_ba / sub
        if not link.exists():
            link.symlink_to(NS_DIR / sub)
    # rename for the fixed-name dataparser lookup
    if not (ns_ba / "transforms.json").exists():
        (ns_ba / "transforms_ba.json").rename(ns_ba / "transforms.json")
    json.dump(dict(scale=float(s), residual_med_cm=float(np.median(resid) * 100),
                   residual_p95_cm=float(np.percentile(resid, 95) * 100),
                   n_poses=n_set, pose_delta=dict(dt_med_cm=float(np.median(dt) * 100),
                                                  dr_med_deg=float(np.median(dr)))),
              open(COLMAP_DIR / "align_stats.json", "w"), indent=2)
    print(f"wrote {ns_ba}/transforms.json ({n_set} frames) + symlinks")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("extract"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--names-from-db", action="store_true", help="only images missing from db")
    p.add_argument("--threads", type=int, default=-1)
    p = sub.add_parser("match"); p.add_argument("--bench", type=int, default=0)
    p.add_argument("--ransac-error", type=float, default=3.0)
    p.add_argument("--threads", type=int, default=-1)
    sub.add_parser("triangulate")
    p = sub.add_parser("ba"); p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--threads", type=int, default=-1)
    sub.add_parser("align")
    args = ap.parse_args()

    if args.cmd == "extract" and getattr(args, "names_from_db", False):
        import pycolmap
        db = pycolmap.Database.open(str(DB_PATH))
        have = {im.name for im in db.read_all_images()}
        db.close()
        allnames = frame_data()[0]
        args.names = [n for n in allnames if n not in have]
        print(f"{len(args.names)} images missing from db")

    dict(init=cmd_init, extract=cmd_extract, match=cmd_match,
         triangulate=cmd_triangulate, ba=cmd_ba, align=cmd_align)[args.cmd](args)


if __name__ == "__main__":
    main()
