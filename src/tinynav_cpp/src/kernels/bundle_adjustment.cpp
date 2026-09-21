// Port of reference/tinynav/cpp/bundle_adjustment.cpp (pybind11 wrapper removed).
#include "tinynav_cpp/kernels/bundle_adjustment.hpp"

#include <array>
#include <iostream>
#include <map>

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace {

class ReprojectionError {
 public:
  ReprojectionError(Eigen::Vector2d observed_keypoint, Eigen::Matrix3d K)
      : observed_keypoint_(std::move(observed_keypoint)), K_(std::move(K)) {}
  template <typename T>
  bool operator()(const T* camera_parameters, const T* point_parameters, T* residuals) const {
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    Eigen::Map<const Eigen::Matrix<T, 6, 1>> camera(camera_parameters);
    Eigen::Map<const Vector3T> point_3d(point_parameters);
    Eigen::Matrix<T, 3, 1> phi = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(camera.data() + 3);
    Eigen::Matrix<T, 3, 1> translation = Eigen::Map<const Vector3T>(camera.data());
    Eigen::Matrix<T, 3, 3> rotation;
    ceres::AngleAxisToRotationMatrix(phi.data(), rotation.data());
    Vector3T point_3d_in_camera = rotation.transpose() * (point_3d - translation);
    Vector3T reprojection = K_.cast<T>() * point_3d_in_camera;
    residuals[0] = reprojection[0] / reprojection[2] - static_cast<T>(observed_keypoint_[0]);
    residuals[1] = reprojection[1] / reprojection[2] - static_cast<T>(observed_keypoint_[1]);
    return true;
  }

 private:
  Eigen::Vector2d observed_keypoint_;
  Eigen::Matrix3d K_;
};

}  // namespace

namespace tinynav::kernels {

BaResult ba_solve(
    const CameraPoses& camera_poses,
    const Point3Ds& point_3ds,
    const std::vector<Observation>& observations,
    const Eigen::Matrix3d& K,
    const ConstantPoseIndex& constant_pose_index,
    const std::vector<RelativePoseConstraint>& relative_pose_constraints) {
  (void)relative_pose_constraints;  // accepted but unused, exactly as the reference
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

  std::map<int64_t, std::array<double, 3>> point_parameters;
  for (const auto& [pt_idx, pt_3d] : point_3ds) {
    std::array<double, 3> point_parameter;
    point_parameter[0] = pt_3d[0];
    point_parameter[1] = pt_3d[1];
    point_parameter[2] = pt_3d[2];
    point_parameters[pt_idx] = point_parameter;
  }

  ceres::LossFunction* loss_function = new ceres::HuberLoss(2.0);
  for (const auto& obs : observations) {
    ceres::CostFunction* reprojection_error =
        new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(obs.keypoint, K));
    problem.AddResidualBlock(reprojection_error, loss_function,
                             camera_parameters.at(obs.cam_idx).data(),
                             point_parameters.at(obs.pt_idx).data());
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
  options.max_num_iterations = 1024;
  options.num_threads = 1;

  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  BaResult result;
  for (const auto& [cam_idx, cam_parameter] : camera_parameters) {
    Eigen::Matrix4d cam_pose_eigen = Eigen::Matrix4d::Identity();
    Eigen::Matrix<double, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(cam_parameter.data() + 3, R.data());
    cam_pose_eigen.block<3, 3>(0, 0) = R;
    cam_pose_eigen.block<3, 1>(0, 3) = Eigen::Map<const Eigen::Vector3d>(cam_parameter.data());
    result.camera_poses[cam_idx] = cam_pose_eigen;
  }
  for (const auto& [pt_idx, pt_parameter] : point_parameters) {
    result.point_3ds[pt_idx] = Eigen::Map<const Eigen::Vector3d>(pt_parameter.data());
  }
  return result;
}

}  // namespace tinynav::kernels
