// Port of reference/tinynav/cpp/pose_graph_solver.cpp (pybind11 wrapper removed).
#include "tinynav_cpp/kernels/pose_graph_solver.hpp"

#include <array>
#include <iostream>
#include <map>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace {

// T_j_i @ (T_w_i)^-1 @ T_w_j = I
class RelativePoseError {
 public:
  RelativePoseError(const Eigen::Matrix4d& relative_pose, Eigen::Vector3d translation_weight, Eigen::Vector3d rotation_weight)
      : relative_j_i_translation_(Eigen::Vector3d::Zero()), relative_j_i_rotation_lie_algebra_(Eigen::Vector3d::Zero()), translation_weight_(std::move(translation_weight)), rotation_weight_(std::move(rotation_weight)) {
    relative_j_i_translation_ = relative_pose.block<3, 1>(0, 3);
    Eigen::Matrix<double, 3, 3> relative_j_i_rotation = relative_pose.block<3, 3>(0, 0);
    ceres::RotationMatrixToAngleAxis(relative_j_i_rotation.data(), relative_j_i_rotation_lie_algebra_.data());
  }
  template <typename T>
  bool operator()(const T* camera_i, const T* camera_j, T* residuals) const {
    using TMatrix3 = Eigen::Matrix<T, 3, 3>;
    using TVector3 = Eigen::Matrix<T, 3, 1>;
    TVector3 translation_i = Eigen::Map<const TVector3>(camera_i);
    TVector3 rotation_i = Eigen::Map<const TVector3>(camera_i + 3);
    TVector3 translation_j = Eigen::Map<const TVector3>(camera_j);
    TVector3 rotation_j = Eigen::Map<const TVector3>(camera_j + 3);
    TMatrix3 R_i;
    ceres::AngleAxisToRotationMatrix(rotation_i.data(), R_i.data());
    TMatrix3 R_j;
    ceres::AngleAxisToRotationMatrix(rotation_j.data(), R_j.data());
    TMatrix3 relative_j_i_rotation;
    TVector3 relative_j_i_rotation_lie_algebra_T = relative_j_i_rotation_lie_algebra_.cast<T>();
    ceres::AngleAxisToRotationMatrix(relative_j_i_rotation_lie_algebra_T.data(), relative_j_i_rotation.data());
    TMatrix3 rotation_loss = relative_j_i_rotation * R_i.transpose() * R_j;
    ceres::RotationMatrixToAngleAxis(rotation_loss.data(), residuals + 3);
    Eigen::Map<TVector3> rotation_residuals_map(residuals + 3);
    rotation_residuals_map = rotation_weight_.cast<T>().asDiagonal() * rotation_residuals_map;
    TVector3 translation_residuals = relative_j_i_rotation * R_i.transpose() * (translation_j - translation_i) + relative_j_i_translation_;
    Eigen::Map<TVector3> translation_residuals_map(residuals);
    translation_residuals_map = translation_weight_.cast<T>().asDiagonal() * translation_residuals;
    return true;
  }
  Eigen::Vector3d relative_j_i_translation_;
  Eigen::Vector3d relative_j_i_rotation_lie_algebra_;
  Eigen::Vector3d translation_weight_;
  Eigen::Vector3d rotation_weight_;
};

}  // namespace

namespace tinynav::kernels {

CameraPoses pose_graph_solve(
    const CameraPoses& camera_poses,
    const std::vector<RelativePoseConstraint>& relative_pose_constraints,
    const ConstantPoseIndex& constant_pose_index,
    int64_t max_iteration_num) {
  ceres::Problem problem;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  std::map<int64_t, std::array<double, 6>> camera_parameters;

  for (const auto& [cam_idx, cam_pose] : camera_poses) {
    std::array<double, 6> camera_parameter;
    const Eigen::Matrix3d R = cam_pose.block<3, 3>(0, 0);
    const Eigen::Vector3d t = cam_pose.block<3, 1>(0, 3);
    camera_parameter[0] = t[0];
    camera_parameter[1] = t[1];
    camera_parameter[2] = t[2];
    ceres::RotationMatrixToAngleAxis(R.data(), camera_parameter.data() + 3);
    camera_parameters[cam_idx] = camera_parameter;
  }

  for (const auto& constraint : relative_pose_constraints) {
    ceres::CostFunction* relative_pose_error =
        new ceres::AutoDiffCostFunction<RelativePoseError, 6, 6, 6>(
            new RelativePoseError(constraint.relative_pose_j_i, constraint.translation_weight, constraint.rotation_weight));
    problem.AddResidualBlock(relative_pose_error, nullptr,
                             camera_parameters.at(constraint.cam_idx_i).data(),
                             camera_parameters.at(constraint.cam_idx_j).data());
  }

  for (const auto& [cam_idx, is_constant] : constant_pose_index) {
    if (is_constant) {
      if (!problem.HasParameterBlock(camera_parameters.at(cam_idx).data())) {
        problem.AddParameterBlock(camera_parameters.at(cam_idx).data(), 6);
      }
      problem.SetParameterBlockConstant(camera_parameters.at(cam_idx).data());
    }
  }

  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = static_cast<int>(max_iteration_num);
  options.num_threads = 1;

  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  CameraPoses optimized_camera_poses;
  for (const auto& [cam_idx, cam_parameter] : camera_parameters) {
    Eigen::Matrix4d cam_pose_eigen = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(cam_parameter.data() + 3, R.data());
    cam_pose_eigen.block<3, 3>(0, 0) = R;
    cam_pose_eigen.block<3, 1>(0, 3) = Eigen::Map<const Eigen::Vector3d>(cam_parameter.data());
    optimized_camera_poses[cam_idx] = cam_pose_eigen;
  }
  return optimized_camera_poses;
}

}  // namespace tinynav::kernels
