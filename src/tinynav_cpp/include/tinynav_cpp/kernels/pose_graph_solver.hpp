// Port of reference/tinynav/cpp/pose_graph_solver.cpp (pybind11 wrapper removed).
// Uses ceres (as the reference did) + Eigen; no other solver dependency.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace tinynav::kernels {

using CameraPoses = std::unordered_map<int64_t, Eigen::Matrix4d>;  // 4x4
using ConstantPoseIndex = std::unordered_map<int64_t, bool>;

// (cam_idx_i, cam_idx_j, relative_pose_j_i, translation_weight, rotation_weight)
struct RelativePoseConstraint {
  int64_t cam_idx_i = 0;
  int64_t cam_idx_j = 0;
  Eigen::Matrix4d relative_pose_j_i = Eigen::Matrix4d::Identity();
  Eigen::Vector3d translation_weight = Eigen::Vector3d::Ones();
  Eigen::Vector3d rotation_weight = Eigen::Vector3d::Ones();
};

// Port of reference/tinynav/cpp/pose_graph_solver.cpp::pose_graph_solve.
CameraPoses pose_graph_solve(
    const CameraPoses& camera_poses,
    const std::vector<RelativePoseConstraint>& relative_pose_constraints,
    const ConstantPoseIndex& constant_pose_index,
    int64_t max_iteration_num);

}  // namespace tinynav::kernels
