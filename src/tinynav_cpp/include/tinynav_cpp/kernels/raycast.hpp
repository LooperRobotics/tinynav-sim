// Port of reference/tinynav/cpp/raycast.cpp::run_raycasting_cpp — pybind
// removed, pure C++ otherwise. This is the standalone kernel, NOT the njit
// run_raycasting_loopy planning_node.py actually calls (that one lives in
// planning/planning.hpp and differs: isfinite guard, filter_ground flag).
//
// Semantics kept from the reference:
//  - depth pixels: only `d <= 0` is skipped (no isfinite check here);
//  - interpolation along the ray uses std::nearbyint (banker's rounding, like
//    the reference's unaryExpr);
//  - returns an (nx, ny * nz) row-major matrix — the C-order flattening
//    ((x * ny + y) * nz + z), i.e. the reshape the Python caller did.
#pragma once

#include <Eigen/Dense>

namespace tinynav::kernels {

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
run_raycasting_cpp(const Eigen::MatrixXf& depth_image,
                   const Eigen::Matrix4d& T_cam_to_world,
                   const Eigen::Vector3i& grid_shape, double fx, double fy,
                   double cx, double cy, const Eigen::Vector3d& origin, int step,
                   double resolution);

}  // namespace tinynav::kernels
