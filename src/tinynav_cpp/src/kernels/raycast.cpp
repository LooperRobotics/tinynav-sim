// Port of reference/tinynav/cpp/raycast.cpp::run_raycasting_cpp (pybind
// removed); see raycast.hpp for the kept semantics.
#include "tinynav_cpp/kernels/raycast.hpp"

#include <algorithm>
#include <cmath>

namespace tinynav::kernels {

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
run_raycasting_cpp(const Eigen::MatrixXf& depth_image,
                   const Eigen::Matrix4d& T_cam_to_world,
                   const Eigen::Vector3i& grid_shape, double fx, double fy,
                   double cx, double cy, const Eigen::Vector3d& origin, int step,
                   double resolution) {
    const int depth_height = static_cast<int>(depth_image.rows());
    const int depth_width = static_cast<int>(depth_image.cols());
    const int nx = grid_shape[0], ny = grid_shape[1], nz = grid_shape[2];

    using GridFlat =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    GridFlat occupancy_grid = GridFlat::Zero(nx, ny * nz);

    const Eigen::Vector3d camera_origin = T_cam_to_world.topRightCorner<3, 1>();
    const Eigen::Vector3i start_voxel_base =
        ((camera_origin - origin) / resolution).array().floor().cast<int>();

    for (int v = 0; v < depth_height; v += step) {
        for (int u = 0; u < depth_width; u += step) {
            const float d = depth_image(v, u);
            if (d <= 0) continue;

            const double x = (u - cx) * static_cast<double>(d) / fx;
            const double y = (v - cy) * static_cast<double>(d) / fy;
            const double z = static_cast<double>(d);

            const Eigen::Vector4d point_cam(x, y, z, 1.0);
            const Eigen::Vector4d point_world_h = T_cam_to_world * point_cam;
            const Eigen::Vector3d point_world = point_world_h.head<3>();

            const Eigen::Vector3i end_voxel =
                ((point_world - origin) / resolution).array().floor().cast<int>();
            const Eigen::Vector3i diff = end_voxel - start_voxel_base;
            const int steps = diff.cwiseAbs().maxCoeff();
            if (steps == 0) continue;

            for (int i = 0; i <= steps; ++i) {
                const double t = static_cast<double>(i) / steps;
                const Eigen::Vector3i interp =
                    (start_voxel_base.cast<double>() + t * diff.cast<double>())
                        .unaryExpr([](double val) { return std::nearbyint(val); })
                        .cast<int>();
                if ((interp.array() < 0).any() ||
                    (interp.array() >= grid_shape.array()).any()) {
                    continue;
                }
                occupancy_grid(interp.x(), interp.y() * nz + interp.z()) -= 0.05;
            }

            if ((end_voxel.array() >= 0).all() &&
                (end_voxel.array() < grid_shape.array()).all()) {
                occupancy_grid(end_voxel.x(), end_voxel.y() * nz + end_voxel.z()) += 0.2;
            }
        }
    }

    occupancy_grid = occupancy_grid.cwiseMax(-0.1).cwiseMin(0.1);
    return occupancy_grid;
}

}  // namespace tinynav::kernels
