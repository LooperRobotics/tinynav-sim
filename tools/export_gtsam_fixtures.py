#!/usr/bin/env python3
"""GTSAM refine alignment fixtures.

Builds a small deterministic scenario (4 window keyframes, 100 Hz IMU, 3
stereo tracks, one failed pair-PnP) and solves the perception window graph
with the python gtsam bindings — the same graph the C++ GtsamRefine impl
builds (tools-side port of the [ISAM Processing] block). The C++ test
(test_gtsam_refine.cpp) runs tinynav::gtsam::Refine::refine() on the same
scenario and compares poses/velocities/errors against the python result.

The IMU preintegration here uses the SAME batch convention as the C++ impl
(dt = stamp - previous stamp, per (t_i, t_j] slice) — see the RefineInput::imu
divergence note (the python node's per-frame drain double-counts the peeked
sample; that quirk is deliberately NOT replicated on either side).

Run inside the container:  python3 tools/export_gtsam_fixtures.py
"""
import os
import shutil
import sys

import numpy as np

import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import X, B, V

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(os.path.dirname(HERE), "fixtures", "gtsam_refine")

N_KF = 4
SAMPLES_PER_PAIR = 30  # 100 Hz imu, 0.3 s between keyframes


def scenario():
    rng = np.random.default_rng(3)
    kf_ts = np.array([i * 0.3 for i in range(N_KF)])
    imu_stamps, imu_accel, imu_gyro = [], [], []
    stamp = 0.0
    while stamp <= kf_ts[-1] + 1e-9:
        imu_stamps.append(stamp)
        imu_accel.append(rng.uniform(-0.4, 0.4, 3) + np.array([0, 0, -9.81]))
        imu_gyro.append(rng.uniform(-0.05, 0.05, 3))
        stamp += 0.01
    poses = []
    for i in range(N_KF):
        T = np.eye(4)
        angle = np.deg2rad(2.0 * i)
        T[:3, :3] = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        T[:3, 3] = [0.5 * i, 0.02 * i, 0.01 * i]
        poses.append(T)
    velocities = np.array([[0.4, 0.0, 0.0], [0.5, 0.01, 0.0],
                           [0.6, 0.02, 0.0], [0.7, 0.03, 0.0]])
    # tracks: (pose_idx, uL, uR, v) observations; parity only needs the SAME
    # numbers on both sides, not geometric validity
    tracks = [
        [(0, 220.5, 200.5, 120.0), (1, 218.1, 198.1, 121.0),
         (2, 215.7, 195.7, 122.0), (3, 213.3, 193.3, 123.0)],
        [(1, 380.2, 352.2, 300.5), (2, 378.9, 350.9, 301.5)],
        [(0, 90.1, 45.1, 400.2), (1, 89.5, 44.5, 400.9)],
    ]
    velocity_prior_pose_idx = [1]
    K = np.array([[272.0, 0.0, 272.0], [0.0, 272.0, 240.0], [0.0, 0.0, 1.0]])
    baseline = 0.05
    return (np.array(imu_stamps), np.array(imu_accel), np.array(imu_gyro),
            kf_ts, np.stack(poses), velocities, tracks, velocity_prior_pose_idx,
            K, baseline)


def preintegrate_pair(params, imu_stamps, imu_accel, imu_gyro, t_i, t_j):
    pim = gtsam.PreintegratedCombinedMeasurements(params, gtsam.imuBias.ConstantBias())
    prev = t_i
    for stamp, accel, gyro in zip(imu_stamps, imu_accel, imu_gyro):
        if stamp <= t_i:
            continue
        if stamp > t_j:
            break
        pim.integrateMeasurement(accel, gyro, stamp - prev)
        prev = stamp
    return pim


def matrix4x4_to_pose3(T):
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3])


def solve_reference(imu_stamps, imu_accel, imu_gyro, kf_ts, poses, velocities,
                    tracks, velocity_prior_pose_idx, K, baseline):
    params = gtsam.PreintegrationCombinedParams.MakeSharedU()
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    n = len(poses)
    for i in range(n):
        initial.insert(B(i), gtsam.imuBias.ConstantBias())
        graph.add(gtsam.PriorFactorConstantBias(
            B(i), gtsam.imuBias.ConstantBias(),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2] * 6))))
        initial.insert(V(i), velocities[i])
        initial.insert(X(i), matrix4x4_to_pose3(poses[i]))
        if i == 0:
            graph.add(gtsam.PriorFactorPose3(
                X(0), matrix4x4_to_pose3(poses[0]),
                gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-1] * 6))))
        if i != n - 1:
            graph.add(gtsam.CombinedImuFactor(
                X(i), V(i), X(i + 1), V(i + 1), B(i), B(i + 1),
                preintegrate_pair(params, imu_stamps, imu_accel, imu_gyro,
                                  kf_ts[i], kf_ts[i + 1])))
    for i in velocity_prior_pose_idx:
        graph.add(gtsam.PriorFactorVector(
            V(i), np.zeros(3),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([0.25] * 3))))
    for track in tracks:
        if len(track) < 2:
            continue
        noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam_unstable.SmartStereoProjectionPoseFactor(
            noise, gtsam.SmartProjectionParams())
        calib = gtsam.Cal3_S2Stereo(K[0, 0], K[1, 1], 0, K[0, 2], K[1, 2], baseline)
        for pose_idx, uL, uR, v in track:
            factor.add(gtsam.StereoPoint2(uL, uR, v), X(pose_idx), calib)
        graph.add(factor)

    lm = gtsam.LevenbergMarquardtOptimizer(graph, initial, _lm_params())
    result = lm.optimize()
    out_poses = np.stack([result.atPose3(X(i)).matrix() for i in range(n)])
    out_vel = np.stack([result.atVector(V(i)) for i in range(n)])
    return (float(graph.error(initial)), float(graph.error(result)),
            out_poses, out_vel, int(graph.size()), int(initial.size()))


def _lm_params():
    p = gtsam.LevenbergMarquardtParams()
    p.setMaxIterations(3)
    return p


def main():
    out_dir = FIXTURES
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    (imu_stamps, imu_accel, imu_gyro, kf_ts, poses, velocities, tracks,
     velocity_prior_pose_idx, K, baseline) = scenario()
    initial_error, final_error, out_poses, out_vel, num_factors, num_vars = \
        solve_reference(imu_stamps, imu_accel, imu_gyro, kf_ts, poses, velocities,
                        tracks, velocity_prior_pose_idx, K, baseline)

    np.save(os.path.join(out_dir, "imu_stamps.npy"), imu_stamps)
    np.save(os.path.join(out_dir, "imu_accel.npy"), imu_accel)
    np.save(os.path.join(out_dir, "imu_gyro.npy"), imu_gyro)
    np.save(os.path.join(out_dir, "kf_timestamps.npy"), kf_ts)
    np.save(os.path.join(out_dir, "init_poses.npy"), poses)
    np.save(os.path.join(out_dir, "init_velocities.npy"), velocities)
    np.save(os.path.join(out_dir, "K.npy"), K)
    np.save(os.path.join(out_dir, "baseline.npy"), np.array(baseline))
    np.save(os.path.join(out_dir, "expected_poses.npy"), out_poses)
    np.save(os.path.join(out_dir, "expected_velocities.npy"), out_vel)
    np.save(os.path.join(out_dir, "expected_errors.npy"),
            np.array([initial_error, final_error]))
    np.save(os.path.join(out_dir, "expected_counts.npy"),
            np.array([num_factors, num_vars, len(tracks)],
                     dtype=np.int64))
    np.save(os.path.join(out_dir, "velocity_prior_pose_idx.npy"),
            np.array(velocity_prior_pose_idx, dtype=np.int64))
    # ragged tracks -> flat + offsets
    flat = [obs for track in tracks for obs in track]
    offsets = np.cumsum([0] + [len(t) for t in tracks])
    np.save(os.path.join(out_dir, "track_offsets.npy"), offsets.astype(np.int64))
    np.save(os.path.join(out_dir, "track_observations.npy"),
            np.array(flat, dtype=np.float64))
    print(f"gtsam fixtures written to {out_dir}: "
          f"initial_error={initial_error:.4f} final_error={final_error:.4f} "
          f"factors={num_factors} vars={num_vars}")


if __name__ == "__main__":
    sys.exit(main())
