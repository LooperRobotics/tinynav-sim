"""Export golden fixtures from the Python reference for the core/mapping
kernels: estimate_pose + quat/rotvec math (math_utils), pose_graph_solve
(the image's pybind kernel, the very code the C++ port replaces), and the
capture-path priors (path_speed / path_climb).

Run inside the tinynav container:

    docker run --rm --gpus all --network host -v "$PWD":/ws -w /ws \\
      --entrypoint /bin/bash uniflexai/tinynav:latest -c \\
      'source /opt/ros/humble/setup.bash && \\
       PYTHONPATH=/ws/reference:$PYTHONPATH python3 tools/export_core_fixtures.py'

Writes fixtures/core/*.npy (gitignored).
"""

import os
import sys

sys.path.insert(0, "/ws/reference")

import numpy as np
import cv2

from tinynav.core import math_utils as mu
from tinynav.core import path_climb, path_speed

OUT = "/ws/fixtures/core"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)


def save(name, arr):
    np.save(f"{OUT}/{name}.npy", np.ascontiguousarray(arr))
    print(f"  {name}: {np.shape(arr)}")


print("== 1. quat/rotvec round trips ==")
quats = rng.normal(size=(12, 4))
quats /= np.linalg.norm(quats, axis=1, keepdims=True)
save("quat_in", quats)
save("quat_to_matrix_out", np.stack([mu.quat_to_matrix(q) for q in quats]))
mats = np.stack([mu.quat_to_matrix(q) for q in quats])
save("matrix_to_quat_out", np.stack([mu.matrix_to_quat(m) for m in mats]))
rotvecs = rng.uniform(-np.pi, np.pi, (12, 3))
save("rotvec_in", rotvecs)
save("rotvec_to_matrix_out", np.stack([mu.rotvec_to_matrix(rv) for rv in rotvecs]))
angles = rng.uniform(-7.0, 7.0, 20)
save("wrap_in", angles)
save("wrap_out", np.array([mu.wrap_angle(a) for a in angles]))

print("== 2. estimate_pose (synthetic, exact + noisy) ==")
# RANSAC samples OpenCV's global RNG, so per-call results are not reproducible
# across processes/versions; the fixtures pin the inputs and ground truth, and
# the C++/Python agreement is asserted against the ground truth instead.
cv2.setRNGSeed(0)
K = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]])
save("pnp_K", K)
for case, noise_px in {"exact": 0.0, "noisy": 0.7}.items():
    r = np.random.default_rng(7 if case == "exact" else 8)
    depth = r.uniform(1.0, 4.0, (480, 640))
    T_gt = np.eye(4)
    T_gt[:3, :3] = mu.rotvec_to_matrix(r.uniform(-0.06, 0.06, 3))
    T_gt[:3, 3] = r.uniform(-0.06, 0.06, 3)
    kpts_curr, kpts_prev = [], []
    for v in range(60, 420, 20):
        for u in range(60, 580, 20):
            z = depth[v, u]
            p3 = np.array([(u - 320.0) * z / 525.0, (v - 240.0) * z / 525.0, z])
            p_prev = T_gt[:3, :3] @ p3 + T_gt[:3, 3]
            kpts_prev.append([525.0 * p_prev[0] / p_prev[2] + 320.0,
                              525.0 * p_prev[1] / p_prev[2] + 240.0])
            kpts_curr.append([float(u), float(v)])
    kpts_curr = np.array(kpts_curr)
    kpts_prev = np.array(kpts_prev)
    if noise_px > 0:
        kpts_prev = kpts_prev + r.normal(0, noise_px, kpts_prev.shape)
        # a few gross outliers
        idx = r.integers(0, len(kpts_prev), 12)
        kpts_prev[idx] += r.uniform(-25, 25, (12, 2))
    state, T_est, _, _, inliers = mu.estimate_pose(kpts_prev, kpts_curr, depth, K)
    save(f"pnp_{case}_gt", T_gt)
    save(f"pnp_{case}_prev", kpts_prev)
    save(f"pnp_{case}_curr", kpts_curr)
    save(f"pnp_{case}_depth", depth)
    save(f"pnp_{case}_T", T_est)
    save(f"pnp_{case}_inliers", np.array(inliers, dtype=np.int64))
    save(f"pnp_{case}_state", np.array(1 if state else 0, dtype=np.int64))

print("== 3. pose_graph_solve (image pybind kernel) ==")
from tinynav import tinynav_cpp_bind as tcb  # noqa: E402  (the .so this port replaces)

for case, (seed, noise, n_poses) in {"a": (11, 0.01, 8), "b": (12, 0.02, 10)}.items():
    r = np.random.default_rng(seed)
    poses = {}
    T = np.eye(4)
    for i in range(n_poses):
        poses[i] = T.copy()
        step = np.eye(4)
        step[:3, :3] = mu.rotvec_to_matrix(r.uniform(-0.1, 0.1, 3))
        step[:3, 3] = r.uniform(0.1, 0.5, 3)
        T = T @ step
    # odometry-style relative constraints with translation noise (the odometry
    # rotation is kept exact; the loop closure below is the perturbed one)
    constraints = []
    for i in range(n_poses - 1):
        rel = np.linalg.inv(poses[i]) @ poses[i + 1]
        rel_noisy = rel.copy()
        rel_noisy[:3, 3] += r.normal(0, noise, 3)
        constraints.append((i + 1, i, rel_noisy, np.array([10.0, 10.0, 10.0]),
                            np.array([30.0, 30.0, 30.0])))
    # one loop closure
    rel = np.linalg.inv(poses[0]) @ poses[n_poses - 1]
    rel_noisy = rel.copy()
    rel_noisy[:3, 3] += r.normal(0, noise * 2, 3)
    constraints.append((n_poses - 1, 0, rel_noisy, np.array([10.0, 10.0, 10.0]),
                        np.array([30.0, 30.0, 30.0])))
    constant = {0: True}
    solved = tcb.pose_graph_solve(poses, constraints, constant, 100)
    save(f"pg_{case}_init", np.stack([poses[i] for i in range(n_poses)]))
    save(f"pg_{case}_n", np.array(n_poses, dtype=np.int64))
    flat = []
    for curr, prev, rel_c, tw, rw in constraints:
        flat.extend([curr, prev])
        flat.extend(np.asarray(rel_c).reshape(-1))
        flat.extend(tw)
        flat.extend(rw)
    save(f"pg_{case}_constraints", np.array(flat))
    save(f"pg_{case}_n_constraints", np.array(len(constraints), dtype=np.int64))
    save(f"pg_{case}_out", np.stack([solved[i] for i in range(n_poses)]))

print("== 4. capture-path priors ==")
for case, seed in {"a": 21, "b": 22}.items():
    r = np.random.default_rng(seed)
    poses = {}
    t_ns = 1_000_000_000
    T = np.eye(4)
    for i in range(120):
        poses[t_ns] = T.copy()
        t_ns += int(r.uniform(0.05, 0.15) * 1e9)
        step = np.eye(4)
        step[:3, 3] = np.array([r.uniform(0.05, 0.25), r.uniform(-0.05, 0.05),
                                r.uniform(-0.03, 0.12) if case == "b" else 0.0])
        T = T @ step
    speeds = path_speed.compute_path_speed(poses)
    climbs = path_climb.compute_path_climb(poses)
    save(f"prior_{case}_poses_ts", np.array(sorted(poses.keys()), dtype=np.int64))
    save(f"prior_{case}_poses_T", np.stack([poses[t] for t in sorted(poses.keys())]))
    save(f"prior_{case}_speed", speeds)
    save(f"prior_{case}_climb", climbs)

print(f"done -> {OUT}")
