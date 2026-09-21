// Port of reference/tinynav/cpp/bundle_adjustment.cpp (pybind11 wrapper removed).
// Uses ceres (as the reference did) + Eigen; no other solver dependency.
#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "tinynav_cpp/kernels/pose_graph_solver.hpp"  // CameraPoses, ConstantPoseIndex, RelativePoseConstraint

namespace tinynav::kernels {

using Point3Ds = std::unordered_map<int64_t, Eigen::Vector3d>;  // 3x1

// (cam_idx, pt_idx, 2x1 observed keypoint)
struct Observation {
  int64_t cam_idx = 0;
  int64_t pt_idx = 0;
  Eigen::Vector2d keypoint = Eigen::Vector2d::Zero();
};

struct BaResult {
  CameraPoses camera_poses;
  Point3Ds point_3ds;
};

// Port of reference/tinynav/cpp/bundle_adjustment.cpp::ba_solve.
// NOTE: like the reference, `relative_pose_constraints` is accepted but unused
// (upstream left the pose-graph terms of BA unwired — the signature contract is
// kept so callers line up 1:1 with the Python).
BaResult ba_solve(
    const CameraPoses& camera_poses,
    const Point3Ds& point_3ds,
    const std::vector<Observation>& observations,
    const Eigen::Matrix3d& K,
    const ConstantPoseIndex& constant_pose_index,
    const std::vector<RelativePoseConstraint>& relative_pose_constraints);

}  // namespace tinynav::kernels
