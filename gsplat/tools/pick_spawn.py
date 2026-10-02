"""Pick a spawn pose for a gsplat scene from the capture trajectory.

Reads the dataset transforms.json (NeRF-style c2w, OpenGL convention), infers
the walkable levels from camera-height clustering, and ranks candidate frames:

  density   frames within 2.5 m (visual coverage around the spawn)
  straight  heading consistency of the local stretch (a walkable corridor)
  eye bonus closeness of the CAMERA height above its level to the dog's eye
            height (0.45 m) -- scenes only reconstructed at capture height
            render mush from lower viewpoints (map_2 lesson: spawn on LOW
            capture stretches; scenes with none, like map_3, max out below 1)

    python3 gsplat/tools/pick_spawn.py <transforms.json> \
        [--target x y z] [--eye 0.45] [--level z] [--top 10]

Prints candidates with ready-to-paste initial_qpos (xyzw quaternion, z =
level + dog stand height 0.445). --target biases the heading toward a point
(e.g. a stair base read out of the collision xml).
"""

import argparse
import json

import numpy as np

STAND_H = 0.445  # go2 spawn base height above the floor it stands on


def load_frames(path):
    tf = json.load(open(path))
    pos, fwd = [], []
    for f in tf["frames"]:
        m = np.array(f["transform_matrix"], float)
        pos.append(m[:3, 3])
        k = -(m[:3, :3] @ np.array([0.0, 0.0, 1.0]))  # OpenGL -z forward
        fwd.append(k / np.linalg.norm(k))
    files = [f["file_path"] for f in tf["frames"]]
    return np.stack(pos), np.stack(fwd), files


def infer_levels(z, min_sep=1.5, bin_w=0.5):
    """Cluster camera z into floor levels (histogram peaks >= min_sep apart)."""
    lo, hi = z.min(), z.max()
    hist, edges = np.histogram(z, bins=np.arange(lo, hi + bin_w, bin_w))
    peaks = []
    for i in np.argsort(hist)[::-1]:
        c = (edges[i] + edges[i + 1]) / 2
        if all(abs(c - p) >= min_sep for p in peaks):
            peaks.append(c)
        if len(peaks) >= 8:
            break
    return np.sort(np.array(peaks)), hist.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transforms")
    ap.add_argument("--target", nargs=3, type=float, default=None)
    ap.add_argument("--eye", type=float, default=0.45)
    ap.add_argument("--level", type=float, default=None,
                    help="pin the level (floor z) instead of inferring")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    pos, fwd, files = load_frames(a.transforms)
    n = len(pos)
    if a.level is not None:
        levels = np.array([a.level])
        # pinned level: keep only frames WALKING on that floor
        # (camera ~0.3..2.5 m above it), not frames on higher floors above it
        on_lvl = (pos[:, 2] > a.level + 0.3) & (pos[:, 2] < a.level + 2.5)
    else:
        levels, peak_count = infer_levels(pos[:, 2])
        print(f"levels (camera z clusters, m): {levels.round(2).tolist()}"
              f"  [frames/bin max {peak_count}]")
        on_lvl = np.ones(n, bool)
    lvl = levels[np.argmin(np.abs(pos[:, 2][:, None] - levels[None, :]), axis=1)]
    eye_err = np.abs((pos[:, 2] - lvl) - a.eye)

    # trajectory tangent per frame: displacement over ~1 m of travel in time
    # (MetaCam cameras scan around while walking -- optical axes are NOT the
    # walking direction, so orientation must come from positions, not fwd)
    tangent = np.zeros((n, 3))
    for i in range(n):
        back = i
        while back > 0 and np.linalg.norm(pos[back] - pos[i]) < 1.0:
            back -= 1
        fwd_i = i
        while fwd_i < n - 1 and np.linalg.norm(pos[fwd_i] - pos[i]) < 1.0:
            fwd_i += 1
        t = pos[fwd_i] - pos[back]
        t[2] = 0
        if np.linalg.norm(t) > 1e-3:
            tangent[i] = t / np.linalg.norm(t)

    dens = np.zeros(n)
    straight = np.full(n, -1.0)
    has_tan = np.linalg.norm(tangent, axis=1) > 1e-3
    idx_arr = np.arange(n)
    for i in range(n):
        if not on_lvl[i] or not has_tan[i]:
            continue
        d = np.linalg.norm(pos - pos[i], axis=1)
        # density: spatial coverage around the spawn (any pass, any direction)
        dens[i] = ((d < 2.5) & (np.abs(idx_arr - i) > 3)).sum()
        # straightness: TEMPORAL stretch -- a corridor walked there-and-back
        # has spatial neighbors with opposite tangents; only the local slice
        # of trajectory says whether the path ahead is walkable-straight
        w = (np.abs(idx_arr - i) < 60) & (d < 3.0) & (np.abs(idx_arr - i) > 3) & has_tan
        if w.sum() < 10:
            continue
        cos = tangent[w] @ tangent[i]
        straight[i] = cos.mean() - 0.5 * cos.std()

    score = (dens / dens.max() + np.clip(straight, 0, 1)
             + np.clip(1.0 - eye_err, 0, 1))
    # heading bonus toward target is folded into the per-candidate loop
    rows = []
    for i in range(n):
        if straight[i] < 0 or not on_lvl[i]:
            continue
        yaw = np.degrees(np.arctan2(tangent[i][1], tangent[i][0]))
        row = dict(i=i, file=files[i], pos=pos[i], lvl=lvl[i], eye_h=pos[i][2] - lvl[i],
                   dens=int(dens[i]), straight=round(float(straight[i]), 3),
                   yaw=round(yaw, 1), score=0.0)
        s = score[i]
        if a.target is not None:
            t = np.array(a.target); rel = t - pos[i]; rel[2] = 0
            nb = np.linalg.norm(rel)
            cosb = float(rel @ tangent[i] / nb) if nb > 1e-3 else 0.0
            s += cosb
            row["face_tgt_cos"] = round(cosb, 3)
            row["dist_tgt"] = round(float(nb), 1)
        row["score"] = round(float(s), 3)
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])

    print(f"\ntop {a.top} candidates (eye={a.eye} m, stand_h={STAND_H}):")
    for r in rows[:a.top]:
        p, yaw = r["pos"], r["yaw"]
        qz, qw = np.sin(np.radians(yaw) / 2), np.cos(np.radians(yaw) / 2)
        extra = ""
        if a.target is not None:
            extra = f"  tgt_cos {r['face_tgt_cos']:+.2f} dist {r['dist_tgt']}"
        print(f"  #{r['i']:4d} ({p[0]:7.2f},{p[1]:7.2f}) lvl {r['lvl']:5.2f} "
              f"eye_h {r['eye_h']:.2f}  dens {r['dens']:3d} straight {r['straight']:.2f}{extra}"
              f"  score {r['score']}")
    r = rows[0]
    p, yaw = r["pos"], r["yaw"]
    qz, qw = np.sin(np.radians(yaw) / 2), np.cos(np.radians(yaw) / 2)
    print(f"\nbest #{r['i']} ({r['file']}) initial_qpos:")
    print(f"  {p[0]:.2f}, {p[1]:.2f}, {r['lvl'] + STAND_H:.3f},")
    print(f"  0, 0, {qz:.4f}, {qw:.4f},")


if __name__ == "__main__":
    main()
