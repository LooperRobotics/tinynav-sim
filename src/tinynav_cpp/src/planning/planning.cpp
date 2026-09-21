// Port of reference/tinynav/core/planning_node.py — pure kernels; see
// planning.hpp for the semantics notes. Function order follows the Python file.
#include "tinynav_cpp/planning/planning.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tinynav::planning {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// numpy/numba np.linspace scalar row: y[i] = start + i * step, y[num-1] = stop
// (numpy pins the last sample exactly; numba matches).
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> y(static_cast<size_t>(std::max(num, 0)));
    if (num <= 0) return y;
    if (num == 1) {
        y[0] = start;
        return y;
    }
    const double step = (stop - start) / double(num - 1);
    for (int i = 0; i < num; ++i) y[i] = start + double(i) * step;
    y[num - 1] = stop;
    return y;
}

}  // namespace

// ---------------------------------------------------------------- OccupancyGrid3D
void OccupancyGrid3D::scale(double s) {
    for (double& v : data_) v *= s;
}

void OccupancyGrid3D::add(const OccupancyGrid3D& other) {
    const size_t n = std::min(data_.size(), other.data_.size());
    for (size_t i = 0; i < n; ++i) data_[i] += other.data_[i];
}

void OccupancyGrid3D::clip(double lo, double hi) {
    for (double& v : data_) v = std::min(std::max(v, lo), hi);
}

// ------------------------------------------------------- run_raycasting_loopy
OccupancyGrid3D run_raycasting_loopy(const Eigen::MatrixXf& depth_image,
                                     const Eigen::Matrix4d& T_cam_to_world,
                                     const Eigen::Vector3i& grid_shape, double fx,
                                     double fy, double cx, double cy,
                                     const Eigen::Vector3d& origin, int step,
                                     double resolution, bool filter_ground) {
    const int gx = grid_shape[0], gy = grid_shape[1], gz = grid_shape[2];
    OccupancyGrid3D occupancy_grid(gx, gy, gz);
    const int depth_height = static_cast<int>(depth_image.rows());
    const int depth_width = static_cast<int>(depth_image.cols());

    const double cam_orig_x = T_cam_to_world(0, 3);
    const double cam_orig_y = T_cam_to_world(1, 3);
    const double cam_orig_z = T_cam_to_world(2, 3);

    const int start_voxel_x =
        static_cast<int>(std::floor((cam_orig_x - origin[0]) / resolution));
    const int start_voxel_y =
        static_cast<int>(std::floor((cam_orig_y - origin[1]) / resolution));
    const int start_voxel_z =
        static_cast<int>(std::floor((cam_orig_z - origin[2]) / resolution));

    for (int v = 0; v < depth_height; v += step) {
        for (int u = 0; u < depth_width; u += step) {
            const float df = depth_image(v, u);
            if (!std::isfinite(df) || df <= 0.0f) continue;
            const double d = static_cast<double>(df);

            const double px = (u - cx) * d / fx;
            const double py = (v - cy) * d / fy;
            const double pz = d;
            const bool is_ground = py > 0;

            const double pw_x = T_cam_to_world(0, 0) * px + T_cam_to_world(0, 1) * py +
                                T_cam_to_world(0, 2) * pz + T_cam_to_world(0, 3);
            const double pw_y = T_cam_to_world(1, 0) * px + T_cam_to_world(1, 1) * py +
                                T_cam_to_world(1, 2) * pz + T_cam_to_world(1, 3);
            const double pw_z = T_cam_to_world(2, 0) * px + T_cam_to_world(2, 1) * py +
                                T_cam_to_world(2, 2) * pz + T_cam_to_world(2, 3);

            const int end_voxel_x =
                static_cast<int>(std::floor((pw_x - origin[0]) / resolution));
            const int end_voxel_y =
                static_cast<int>(std::floor((pw_y - origin[1]) / resolution));
            const int end_voxel_z =
                static_cast<int>(std::floor((pw_z - origin[2]) / resolution));

            const int diff_x = end_voxel_x - start_voxel_x;
            const int diff_y = end_voxel_y - start_voxel_y;
            const int diff_z = end_voxel_z - start_voxel_z;

            const int steps = std::max({std::abs(diff_x), std::abs(diff_y), std::abs(diff_z)});
            if (steps == 0) continue;

            for (int i = 0; i <= steps; ++i) {
                const double t = double(i) / double(steps);
                // Python round(): half to even.
                const int interp_x =
                    static_cast<int>(std::nearbyint(start_voxel_x + t * diff_x));
                const int interp_y =
                    static_cast<int>(std::nearbyint(start_voxel_y + t * diff_y));
                const int interp_z =
                    static_cast<int>(std::nearbyint(start_voxel_z + t * diff_z));
                if (0 <= interp_x && interp_x < gx && 0 <= interp_y && interp_y < gy &&
                    0 <= interp_z && interp_z < gz) {
                    occupancy_grid(interp_x, interp_y, interp_z) -= 0.05;
                }
            }

            if (0 <= end_voxel_x && end_voxel_x < gx && 0 <= end_voxel_y &&
                end_voxel_y < gy && 0 <= end_voxel_z && end_voxel_z < gz) {
                if (filter_ground && is_ground) {
                    // pass
                } else {
                    occupancy_grid(end_voxel_x, end_voxel_y, end_voxel_z) += 0.2;
                }
            }
        }
    }

    for (int i = 0; i < gx; ++i) {
        for (int j = 0; j < gy; ++j) {
            for (int k = 0; k < gz; ++k) {
                double& cell = occupancy_grid(i, j, k);
                if (cell < -0.1) {
                    cell = -0.1;
                } else if (cell > 0.1) {
                    cell = 0.1;
                }
            }
        }
    }
    return occupancy_grid;
}

// ---------------------------------------------------------- build_obstacle_map
Mask2D build_obstacle_map(const OccupancyGrid3D& occupancy_grid,
                          const Eigen::Vector3d& origin, double resolution,
                          double robot_z, const core::ObstacleConfig& config,
                          const Eigen::ArrayXXf* min_span_map) {
    const int h = occupancy_grid.nx();
    const int w = occupancy_grid.ny();
    const int z_dim = occupancy_grid.nz();

    std::vector<double> z_rel(static_cast<size_t>(z_dim));
    std::vector<int> band;  // k with robot_z_bottom <= z_rel[k] <= robot_z_top
    band.reserve(static_cast<size_t>(z_dim));
    for (int k = 0; k < z_dim; ++k) {
        z_rel[static_cast<size_t>(k)] = origin[2] + (k + 0.5) * resolution - robot_z;
        if (z_rel[static_cast<size_t>(k)] >= config.robot_z_bottom &&
            z_rel[static_cast<size_t>(k)] <= config.robot_z_top) {
            band.push_back(k);
        }
    }

    Mask2D obstacle = Mask2D::Constant(h, w, false);
    if (!band.empty()) {
        const int n_z = static_cast<int>(band.size());
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                bool has_occ = false;
                int occ_high = -1;
                int occ_low = n_z;
                for (int bi = 0; bi < n_z; ++bi) {
                    if (occupancy_grid(i, j, band[static_cast<size_t>(bi)]) >
                        config.occ_threshold) {
                        has_occ = true;
                        occ_high = std::max(occ_high, bi);
                        occ_low = std::min(occ_low, bi);
                    }
                }
                if (!has_occ) continue;
                // Python quirk kept: occ_high/occ_low come from a float32 z_idx,
                // so z_span and both span comparisons happen in float32 — a
                // span of exactly min_wall_span_m passes there and can fail in
                // double (0.2000000000000000111 < 0.20000000298).
                const float z_span =
                    static_cast<float>(occ_high - occ_low) * static_cast<float>(resolution);
                // relative height of the lowest occupied voxel in the cell
                const int low_idx = std::clamp(occ_low, 0, n_z - 1);
                const double low_z_rel = z_rel[static_cast<size_t>(band[static_cast<size_t>(low_idx)])];
                const bool near_ground =
                    low_z_rel <= config.robot_z_bottom + config.ground_band_m;
                const float min_span =
                    min_span_map != nullptr ? (*min_span_map)(i, j)
                                            : static_cast<float>(config.min_wall_span_m);
                // A relaxed cell keeps its threshold at any height; strict cells
                // anchored outside the ground band keep only the noise floor.
                const bool relaxed =
                    min_span_map != nullptr && min_span > static_cast<float>(config.min_wall_span_m);
                const bool span_ok = (near_ground || relaxed)
                                         ? (z_span >= min_span)
                                         : (z_span >= static_cast<float>(resolution));
                obstacle(i, j) = has_occ && span_ok;
            }
        }
    }

    if (config.dilation_cells > 0 && obstacle.any()) {
        obstacle = binary_dilation(obstacle, config.dilation_cells);
    }
    return obstacle;
}

// -------------------------------------------------------- roll_occupancy_grid
RolledGrid roll_occupancy_grid(const OccupancyGrid3D& occupancy_grid,
                               const Eigen::Vector3d& old_origin,
                               const Eigen::Vector3d& new_origin,
                               double resolution) {
    const Eigen::Vector3d shift_m = new_origin - old_origin;
    // np.round: half to even.
    Eigen::Vector3i shift_voxels;
    for (int a = 0; a < 3; ++a) {
        shift_voxels[a] = static_cast<int>(std::nearbyint(shift_m[a] / resolution));
    }
    if (shift_voxels[0] == 0 && shift_voxels[1] == 0 && shift_voxels[2] == 0) {
        return {occupancy_grid, old_origin};
    }
    const int nx = occupancy_grid.nx(), ny = occupancy_grid.ny(), nz = occupancy_grid.nz();
    OccupancyGrid3D rolled(nx, ny, nz);
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            for (int z = 0; z < nz; ++z) {
                // np.roll(a, -shift)[j] == a[j + shift] (wrapping), and the wrapped
                // part is zeroed; == gather with zero where the source leaves the grid.
                const int sx = x + shift_voxels[0];
                const int sy = y + shift_voxels[1];
                const int sz = z + shift_voxels[2];
                if (0 <= sx && sx < nx && 0 <= sy && sy < ny && 0 <= sz && sz < nz) {
                    rolled(x, y, z) = occupancy_grid(sx, sy, sz);
                }
            }
        }
    }
    const Eigen::Vector3d updated_origin =
        old_origin + shift_voxels.cast<double>() * resolution;
    return {rolled, updated_origin};
}

// ------------------------------------------------- trajectory library sampling
TrajectorySet generate_trajectory_library_3d(
    int num_samples, double duration, double dt, const Eigen::Vector3d& init_p,
    const Eigen::Vector4d& init_q, double max_linear_vel, double max_angular_vel,
    double max_path_len_m, double max_lat_acc, double min_linear_vel) {
    const int num_steps = static_cast<int>(duration / dt) + 1;

    const double vx_max = max_linear_vel;
    const int n_vx = std::max(3, static_cast<int>(num_samples / 2.0));
    const int n_omega = num_samples;
    const double vx_lo = min_linear_vel < vx_max ? min_linear_vel : vx_max;
    std::vector<double> vx_samples(static_cast<size_t>(n_vx));
    vx_samples[0] = 0.0;
    const std::vector<double> vx_rest = linspace(vx_lo, vx_max, n_vx - 1);
    for (int i = 1; i < n_vx; ++i) vx_samples[static_cast<size_t>(i)] = vx_rest[static_cast<size_t>(i - 1)];

    const int total = n_vx * n_omega;
    TrajectorySet out;
    out.poses.setZero(total * num_steps, 7);
    out.params.setZero(total, 2);
    out.num_trajectories = total;
    out.num_steps = num_steps;

    int k = -1;
    for (int i_vx = 0; i_vx < n_vx; ++i_vx) {
        const double vx = vx_samples[static_cast<size_t>(i_vx)];
        double omega_lim = max_angular_vel;
        if (vx > 1e-6 && max_lat_acc / vx < omega_lim) {
            omega_lim = max_lat_acc / vx;
        }
        const std::vector<double> omega_y_samples =
            linspace(-omega_lim, omega_lim, n_omega);
        const double step_len = vx * dt;
        for (int i_omega = 0; i_omega < n_omega; ++i_omega) {
            ++k;
            const double omega_y = omega_y_samples[static_cast<size_t>(i_omega)];
            Eigen::Vector3d p = init_p;
            Eigen::Matrix3d q = core::quat_to_matrix(init_q);
            double path_len = 0.0;
            for (int i = 0; i < num_steps; ++i) {
                if (path_len + step_len <= max_path_len_m) {
                    const Eigen::Matrix3d dq =
                        core::rotvec_to_matrix(Eigen::Vector3d(0.0, omega_y * dt, 0.0));
                    q = q * dq;
                    const Eigen::Vector3d v_world = q * Eigen::Vector3d(0.0, 0.0, vx);
                    p += v_world * dt;
                    path_len += step_len;
                }
                auto row = out.poses.row(k * num_steps + i);
                row.head<3>() = p;
                row.tail<4>() = core::matrix_to_quat(q);
            }
            // z-flatten hack
            for (int i = 0; i < num_steps; ++i) {
                out.poses(k * num_steps + i, 2) = out.poses(k * num_steps + 0, 2);
            }
            out.params(k, 0) = vx;
            out.params(k, 1) = omega_y;
        }
    }
    return out;
}

TrajectorySet generate_predefined_trajectory_vocabularies(
    double duration, double dt, const Eigen::Vector3d& init_p,
    const Eigen::Vector4d& init_q) {
    const int num_steps = static_cast<int>(duration / dt) + 1;
    const double reverse_speed = 0.3;
    Eigen::Vector3d p = init_p;
    const Eigen::Matrix3d q = core::quat_to_matrix(init_q);

    TrajectorySet out;
    out.poses.setZero(num_steps, 7);
    out.params.setZero(1, 2);
    out.num_trajectories = 1;
    out.num_steps = num_steps;
    for (int i = 0; i < num_steps; ++i) {
        const Eigen::Vector3d v_world = q * Eigen::Vector3d(0.0, 0.0, -reverse_speed);
        p += v_world * dt;
        auto row = out.poses.row(i);
        row.head<3>() = p;
        row.tail<4>() = core::matrix_to_quat(q);
    }
    for (int i = 0; i < num_steps; ++i) {
        out.poses(i, 2) = out.poses(0, 2);
    }
    out.params(0, 0) = -reverse_speed;
    out.params(0, 1) = 0.0;
    return out;
}

TrajectorySet concatenate_trajectories(const TrajectorySet& a, const TrajectorySet& b) {
    TrajectorySet out;
    out.num_steps = a.num_steps;
    out.num_trajectories = a.num_trajectories + b.num_trajectories;
    out.poses.resize(out.num_trajectories * out.num_steps, 7);
    out.params.resize(out.num_trajectories, 2);
    out.poses.topRows(a.num_trajectories * a.num_steps) = a.poses;
    out.poses.bottomRows(b.num_trajectories * b.num_steps) = b.poses;
    out.params.topRows(a.num_trajectories) = a.params;
    out.params.bottomRows(b.num_trajectories) = b.params;
    return out;
}

// ------------------------------------------------------------------- scoring
bool reverse_armed(double front_clearance, int n_fwd_ok, double resolution) {
    // Half a cell of slack: clearance lands on multiples of resolution and would
    // never hit the threshold exactly in metres (see the Python docstring).
    return front_clearance <= REVERSE_ENTER_M + resolution / 2 || n_fwd_ok == 0;
}

FootprintLattice footprint_lattice(double front_len, double rear_len, double half_w,
                                   double safety_radius) {
    const double pitch = safety_radius * std::sqrt(2.0);
    int n_long = static_cast<int>(std::ceil((front_len + rear_len) / pitch)) + 1;
    int n_lat = static_cast<int>(std::ceil((2.0 * half_w) / pitch)) + 1;
    if (n_long < 2) n_long = 2;
    if (n_lat < 2) n_lat = 2;
    FootprintLattice out;
    out.fwd.resize(static_cast<size_t>(n_long * n_lat) + 1);
    out.lat.resize(static_cast<size_t>(n_long * n_lat) + 1);
    out.fwd[0] = 0.0;  // index 0 is the centre and stays the centre
    out.lat[0] = 0.0;
    int n = 1;
    for (int a = 0; a < n_long; ++a) {
        const double f = -rear_len + (front_len + rear_len) * a / (n_long - 1);
        for (int b = 0; b < n_lat; ++b) {
            const double l = -half_w + (2.0 * half_w) * b / (n_lat - 1);
            if (f == 0.0 && l == 0.0) continue;  // already sample 0
            out.fwd[static_cast<size_t>(n)] = f;
            out.lat[static_cast<size_t>(n)] = l;
            ++n;
        }
    }
    out.fwd.resize(static_cast<size_t>(n));
    out.lat.resize(static_cast<size_t>(n));
    return out;
}

EsdfScoreResult score_trajectories_by_ESDF(
    const TrajectorySet& trajectories, const Eigen::ArrayXXf& ESDF_map,
    const Eigen::ArrayXXf& path_dist_map, const Eigen::ArrayXXf& remaining_map,
    const Eigen::ArrayXXf& route_heading_map, const Eigen::Vector3d& origin,
    double resolution, double safety_radius, double front_len, double rear_len,
    double half_w) {
    EsdfScoreResult res;
    const int ESDF_rows = static_cast<int>(ESDF_map.rows());
    const int ESDF_cols = static_cast<int>(ESDF_map.cols());

    const FootprintLattice lattice =
        footprint_lattice(front_len, rear_len, half_w, safety_radius);
    const int n_samp = static_cast<int>(lattice.fwd.size());
    const int n_traj = trajectories.num_trajectories;

    for (int t = 0; t < n_traj; ++t) {
        double min_dist_for_traj = kInf;
        int closest_step_for_traj = -1;
        double path_cost_max = 0.0;
        int path_cost_n = 0;
        double traveled_arc = 0.0;

        for (int i = 0; i < trajectories.num_steps; ++i) {
            const auto pose = trajectories.poses.row(t * trajectories.num_steps + i);
            const double x_world = pose[0], y_world = pose[1];
            if (i > 0) {
                const auto prev = trajectories.poses.row(t * trajectories.num_steps + i - 1);
                const double dx = x_world - prev[0];
                const double dy = y_world - prev[1];
                traveled_arc += std::sqrt(dx * dx + dy * dy);
            }
            const double qx = pose[3], qy = pose[4], qz = pose[5], qw = pose[6];

            // world XY forward from quaternion (body +Z forward)
            double fwd_x = 2.0 * (qx * qz + qw * qy);
            double fwd_y = 2.0 * (qy * qz - qw * qx);
            const double n = std::sqrt(fwd_x * fwd_x + fwd_y * fwd_y);
            if (n > 1e-6) {
                fwd_x /= n;
                fwd_y /= n;
            } else {
                fwd_x = 1.0;
                fwd_y = 0.0;
            }
            const double left_x = -fwd_y;
            const double left_y = fwd_x;

            for (int kk = 0; kk < n_samp; ++kk) {
                const double cx = x_world + fwd_x * lattice.fwd[static_cast<size_t>(kk)] +
                                  left_x * lattice.lat[static_cast<size_t>(kk)];
                const double cy = y_world + fwd_y * lattice.fwd[static_cast<size_t>(kk)] +
                                  left_y * lattice.lat[static_cast<size_t>(kk)];
                const int x_img = static_cast<int>((cx - origin[0]) / resolution);
                const int y_img = static_cast<int>((cy - origin[1]) / resolution);
                if (0 <= x_img && x_img < ESDF_rows && 0 <= y_img && y_img < ESDF_cols) {
                    const double dist = ESDF_map(x_img, y_img);
                    if (dist < min_dist_for_traj) {
                        min_dist_for_traj = dist;
                        closest_step_for_traj = i;
                    }
                    if (kk == 0) {  // route adherence is measured on the centre only
                        const double center_path_dist =
                            static_cast<double>(path_dist_map(x_img, y_img));
                        if (center_path_dist > path_cost_max) {
                            path_cost_max = center_path_dist;
                        }
                        ++path_cost_n;
                    }
                }
            }
        }

        res.path_costs.push_back(path_cost_n > 0 ? path_cost_max : 1e3);

        const auto end_pose = trajectories.poses.row(t * trajectories.num_steps +
                                                     trajectories.num_steps - 1);
        const int end_x_img = static_cast<int>((end_pose[0] - origin[0]) / resolution);
        const int end_y_img = static_cast<int>((end_pose[1] - origin[1]) / resolution);
        double end_remaining = 1e3;
        if (0 <= end_x_img && end_x_img < ESDF_rows && 0 <= end_y_img &&
            end_y_img < ESDF_cols) {
            end_remaining = static_cast<double>(remaining_map(end_x_img, end_y_img));
        }

        // 0 when the end left the grid: an off-grid end is already punished through
        // path_cost/remaining (see the Python comment).
        double end_heading_err = 0.0;
        if (0 <= end_x_img && end_x_img < ESDF_rows && 0 <= end_y_img &&
            end_y_img < ESDF_cols) {
            end_heading_err = angle_between(
                heading_of_pose7(end_pose),
                static_cast<double>(route_heading_map(end_x_img, end_y_img)));
        }
        res.end_heading_errs.push_back(end_heading_err);

        const int start_x_img = static_cast<int>((trajectories.poses(t * trajectories.num_steps, 0) - origin[0]) / resolution);
        const int start_y_img = static_cast<int>((trajectories.poses(t * trajectories.num_steps, 1) - origin[1]) / resolution);
        if (0 <= start_x_img && start_x_img < ESDF_rows && 0 <= start_y_img &&
            start_y_img < ESDF_cols) {
            const double start_remaining =
                static_cast<double>(remaining_map(start_x_img, start_y_img));
            if (start_remaining < 1e3) {
                end_remaining = std::max(end_remaining, start_remaining - traveled_arc);
            }
        }
        res.end_remainings.push_back(end_remaining);

        if (min_dist_for_traj < 1e-3) {  // collision
            res.scores.push_back(kInf);
        } else if (min_dist_for_traj != kInf) {
            if (min_dist_for_traj > safety_radius) {
                res.scores.push_back(0.0);
            } else {
                const int max_steps = trajectories.num_steps;
                const double decay_factor =
                    double(max_steps - closest_step_for_traj) / double(max_steps);
                const double base_score = 1.0 / (min_dist_for_traj + 1e-3);
                res.scores.push_back(decay_factor * base_score);
            }
        } else {
            res.scores.push_back(0.0);
        }
        res.occ_points.push_back(closest_step_for_traj);
    }
    return res;
}

EsdfScoreResult score_trajectories(const TrajectorySet& trajectories,
                                   const Eigen::MatrixX2d& params,
                                   const Eigen::ArrayXXf& ESDF_map,
                                   const Eigen::ArrayXXf& path_dist_map,
                                   const Eigen::ArrayXXf& remaining_map,
                                   const Eigen::ArrayXXf& route_heading_map,
                                   const Eigen::Vector3d& origin, double resolution,
                                   const core::RobotConfig& robot) {
    const auto [front_len, rear_len, half_w] = robot.footprint_from_control();
    EsdfScoreResult res = score_trajectories_by_ESDF(
        trajectories, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution, robot.safety_radius, front_len, rear_len, half_w);

    std::vector<int> back;
    for (int t = 0; t < trajectories.num_trajectories; ++t) {
        if (params(t, 0) < 0.0) back.push_back(t);
    }
    if (!back.empty() && trajectories.num_steps > 1) {
        // Score the reverse rows on the trajectory minus its first pose.
        TrajectorySet moved_set;
        moved_set.num_steps = trajectories.num_steps - 1;
        moved_set.num_trajectories = static_cast<int>(back.size());
        moved_set.poses.resize(moved_set.num_trajectories * moved_set.num_steps, 7);
        for (int j = 0; j < moved_set.num_trajectories; ++j) {
            const int t = back[static_cast<size_t>(j)];
            moved_set.poses.middleRows(j * moved_set.num_steps, moved_set.num_steps) =
                trajectories.poses.middleRows(t * trajectories.num_steps + 1,
                                              moved_set.num_steps);
        }
        EsdfScoreResult moved = score_trajectories_by_ESDF(
            moved_set, ESDF_map, path_dist_map, remaining_map, route_heading_map,
            origin, resolution, robot.safety_radius, front_len, rear_len, half_w);
        for (int j = 0; j < moved_set.num_trajectories; ++j) {
            const size_t t = static_cast<size_t>(back[static_cast<size_t>(j)]);
            res.scores[t] = moved.scores[static_cast<size_t>(j)];
            res.occ_points[t] = moved.occ_points[static_cast<size_t>(j)];
            res.path_costs[t] = moved.path_costs[static_cast<size_t>(j)];
            res.end_remainings[t] = moved.end_remainings[static_cast<size_t>(j)];
            res.end_heading_errs[t] = moved.end_heading_errs[static_cast<size_t>(j)];
        }
    }
    return res;
}

// --------------------------------------------------------------- route fields
namespace {

// np.interp(x, xp, fp) for a scalar x with monotone xp (endpoint clamping);
// zero-width intervals take the left value.
double interp_scalar(double x, const Eigen::VectorXd& xp, const Eigen::VectorXd& fp) {
    const int n = static_cast<int>(xp.size());
    if (n == 0) return 0.0;
    if (n == 1 || x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    // first j with xp[j] > x, then interpolate against j - 1 (np searchsorted 'right' - 1 style)
    int j = static_cast<int>(std::upper_bound(xp.data(), xp.data() + n, x) - xp.data());
    if (j >= n) return fp[n - 1];
    if (j <= 0) return fp[0];
    const double x0 = xp[j - 1], x1 = xp[j];
    if (x1 == x0) return fp[j - 1];
    const double slope = (fp[j] - fp[j - 1]) / (x1 - x0);
    return fp[j - 1] + (x - x0) * slope;
}

}  // namespace

RouteFields build_route_fields(const Eigen::MatrixX2d& route_xy,
                               const Eigen::Vector2i& shape,
                               const Eigen::Vector3d& origin, double resolution) {
    RouteFields out;
    const int rows = shape[0], cols = shape[1];
    out.path_dist_map = Eigen::ArrayXXf::Constant(rows, cols, 1e3f);
    out.remaining_map = Eigen::ArrayXXf::Constant(rows, cols, 1e3f);
    out.route_heading_map = Eigen::ArrayXXf::Zero(rows, cols);
    if (route_xy.rows() < 2) return out;

    const int n = static_cast<int>(route_xy.rows());
    Eigen::VectorXd node_arc(n);
    node_arc[0] = 0.0;
    for (int i = 1; i < n; ++i) {
        node_arc[i] = node_arc[i - 1] +
                      (route_xy.row(i) - route_xy.row(i - 1)).norm();
    }
    const double arc = node_arc[n - 1];
    if (arc < 1e-9) return out;

    // resample at half-cell steps so the rasterized line has no gaps for the EDT
    const int n_samples = static_cast<int>(std::ceil(arc / (0.5 * resolution))) + 1;
    Eigen::VectorXd sample_arc(n_samples);
    {
        const double step = arc / double(n_samples - 1);
        for (int k = 0; k < n_samples; ++k) sample_arc[k] = double(k) * step;
        sample_arc[n_samples - 1] = arc;  // numpy pins the last sample
    }
    Eigen::VectorXd sx(n_samples), sy(n_samples);
    Eigen::VectorXd xs = route_xy.col(0), ys = route_xy.col(1);
    for (int k = 0; k < n_samples; ++k) {
        sx[k] = interp_scalar(sample_arc[k], node_arc, xs);
        sy[k] = interp_scalar(sample_arc[k], node_arc, ys);
    }

    Mask2D route_mask = Mask2D::Constant(rows, cols, false);
    Eigen::ArrayXXf arc_map = Eigen::ArrayXXf::Zero(rows, cols);
    Eigen::ArrayXXf heading_map = Eigen::ArrayXXf::Zero(rows, cols);
    bool any_inside = false;
    Eigen::VectorXd seg_y(n - 1), seg_x(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        seg_x[i] = route_xy(i + 1, 0) - route_xy(i, 0);
        seg_y[i] = route_xy(i + 1, 1) - route_xy(i, 1);
    }
    // side='right' searchsorted - 1, clipped to [0, n-2]
    for (int k = 0; k < n_samples; ++k) {
        int j = static_cast<int>(std::upper_bound(node_arc.data(),
                                                  node_arc.data() + n, sample_arc[k]) -
                                 node_arc.data());
        j = std::clamp(j - 1, 0, n - 2);
        const double tang = std::atan2(seg_y[j], seg_x[j]);
        const int r = static_cast<int>((sx[k] - origin[0]) / resolution);  // astype(int64): truncate
        const int c = static_cast<int>((sy[k] - origin[1]) / resolution);
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            any_inside = true;
            route_mask(r, c) = true;
            arc_map(r, c) = static_cast<float>(sample_arc[k]);  // a cell crossed twice keeps the later arc
            heading_map(r, c) = static_cast<float>(tang);
        }
    }
    if (!any_inside) {
        // Python: `if not np.any(inside): return ..., False`.
        return out;
    }

    // One EDT, three fields: each off-route cell takes the arc and the direction
    // of its nearest route cell (a gather, not an average). scipy semantics:
    // the EDT's features are the ZERO cells of its input, so the input is
    // ~route_mask — the route cells are the features.
    Mask2D edt_input = Mask2D::Constant(rows, cols, true);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (route_mask(r, c)) edt_input(r, c) = false;
        }
    }
    const EdtResult edt = distance_transform_edt(edt_input);
    out.path_dist_map = (edt.distances * resolution).cast<float>();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int nr = edt.nearest_row(r, c);
            const int nc = edt.nearest_col(r, c);
            out.remaining_map(r, c) = static_cast<float>(arc - arc_map(nr, nc));
            out.route_heading_map(r, c) = heading_map(nr, nc);
        }
    }
    out.has_route = true;
    return out;
}

// ------------------------------------------------------------ DWA cost + pick
double end_heading_error(const Eigen::Matrix<double, 1, 7>& pose7,
                         const Eigen::Vector3d& goal) {
    return angle_between(std::atan2(goal[1] - pose7[1], goal[0] - pose7[0]),
                         heading_of_pose7(pose7));
}

double trajectory_cost(int i, const TrajectorySet& trajectories,
                       const Eigen::MatrixX2d& params,
                       const EsdfScoreResult& scored, bool has_route,
                       const std::optional<Eigen::Vector3d>& target,
                       const Eigen::Vector2d& last_param, bool should_reverse,
                       const DwaWeights& weights) {
    const double vx = params(i, 0);
    const double omega = params(i, 1);
    const double reverse_gate_penalty =
        ((vx < 0.0) == should_reverse) ? 0.0 : 1e9;
    const Eigen::Matrix<double, 1, 7> end_pose =
        trajectories.poses.row(i * trajectories.num_steps + trajectories.num_steps - 1);
    const Eigen::Vector3d traj_end(end_pose[0], end_pose[1], end_pose[2]);
    const Eigen::Vector3d target_end = target.value_or(traj_end);
    const double dist = (traj_end - target_end).norm();
    const double smooth = std::abs(last_param[0] - vx) + std::abs(last_param[1] - omega);

    // Skipped within 0.3 m of the goal, where the bearing is noise.
    double heading_penalty = 0.0;
    if (dist > 0.3) {
        const double to_goal = weights.w_route_heading * end_heading_error(end_pose, target_end);
        if (has_route) {
            const double fade = route_band_fade(scored.end_remainings[static_cast<size_t>(i)],
                                                weights.route_terminal_band);
            heading_penalty =
                route_heading_penalty(weights.w_route_heading,
                                      scored.end_heading_errs[static_cast<size_t>(i)],
                                      scored.end_remainings[static_cast<size_t>(i)],
                                      weights.route_terminal_band) +
                (1.0 - fade) * to_goal;
        } else {
            heading_penalty = to_goal;
        }
    }

    double positional;
    if (!has_route) {
        // No route this cycle: rank on the raw target.
        positional = 100.0 * dist;
    } else {
        double terminal = 0.0;
        if (target.has_value()) {
            terminal = weights.w_goal_terminal *
                       (1.0 - route_band_fade(scored.end_remainings[static_cast<size_t>(i)],
                                              weights.route_terminal_band)) *
                       (end_pose.head<2>().transpose() - target->head<2>()).norm();
        }
        positional = weights.w_route_progress * scored.end_remainings[static_cast<size_t>(i)] +
                     weights.w_path_follow * scored.path_costs[static_cast<size_t>(i)] +
                     terminal;
    }
    return scored.scores[static_cast<size_t>(i)] * weights.w_clearance + positional +
           10.0 * smooth + heading_penalty + reverse_gate_penalty;
}

int select_trajectory(const TrajectorySet& trajectories, const Eigen::MatrixX2d& params,
                      const EsdfScoreResult& scored, bool has_route,
                      const std::optional<Eigen::Vector3d>& target,
                      const Eigen::Vector2d& last_param, bool should_reverse,
                      const DwaWeights& weights) {
    double best_cost = kInf;
    int best = -1;
    for (int i = 0; i < trajectories.num_trajectories; ++i) {
        const double cost = trajectory_cost(i, trajectories, params, scored, has_route,
                                            target, last_param, should_reverse, weights);
        if (cost < best_cost) {  // strict: first minimal index on ties, like min()
            best_cost = cost;
            best = i;
        }
    }
    return best;
}

// ------------------------------------------------------------- node helpers
Eigen::Vector3d camera_to_robot_center(const Eigen::Matrix4d& T,
                                       const core::RobotConfig& robot) {
    return T.topRightCorner<3, 1>() -
           T.topLeftCorner<3, 3>() * robot.cam_offset_3d().cast<double>();
}

double front_obstacle_dist(const Eigen::Matrix4d& T, const Mask2D& obstacle_mask,
                           const Eigen::Vector3d& origin, double resolution,
                           const core::RobotConfig& robot, double max_dist) {
    const Eigen::Vector3d center = camera_to_robot_center(T, robot);
    const Eigen::Vector3d fwd_v = T.topLeftCorner<3, 3>() * Eigen::Vector3d(0.0, 0.0, 1.0);
    const double n = std::sqrt(fwd_v[0] * fwd_v[0] + fwd_v[1] * fwd_v[1]);
    const double fx = n > 1e-6 ? fwd_v[0] / n : 1.0;
    const double fy = n > 1e-6 ? fwd_v[1] / n : 0.0;
    const double lx = -fy, ly = fx;
    const auto [fl, rl, hw] = robot.footprint_from_control();
    (void)rl;
    const int rows = static_cast<int>(obstacle_mask.rows());
    const int cols = static_cast<int>(obstacle_mask.cols());
    const int steps = static_cast<int>(max_dist / resolution) + 1;
    for (int step = 0; step < steps; ++step) {
        const double d_from_face = step * resolution;
        const double d_from_center = fl + d_from_face;
        for (const double w : {(-hw), 0.0, hw}) {
            const int xi = static_cast<int>(
                (center[0] + fx * d_from_center + lx * w - origin[0]) / resolution);
            const int yi = static_cast<int>(
                (center[1] + fy * d_from_center + ly * w - origin[1]) / resolution);
            if (0 <= xi && xi < rows && 0 <= yi && yi < cols && obstacle_mask(xi, yi)) {
                return d_from_face;
            }
        }
    }
    return max_dist + 1.0;
}

FootprintHits footprint_hits(const Eigen::Matrix4d& T, const Mask2D& obstacle_mask,
                             const Eigen::Vector3d& origin, double resolution,
                             const core::RobotConfig& robot) {
    const Eigen::Vector3d center = camera_to_robot_center(T, robot);
    const Eigen::Vector3d fwd_v = T.topLeftCorner<3, 3>() * Eigen::Vector3d(0.0, 0.0, 1.0);
    const double n = std::sqrt(fwd_v[0] * fwd_v[0] + fwd_v[1] * fwd_v[1]);
    const double fx = n > 1e-6 ? fwd_v[0] / n : 1.0;
    const double fy = n > 1e-6 ? fwd_v[1] / n : 0.0;
    const double lx = -fy, ly = fx;
    const auto [fl, rl, hw] = robot.footprint_from_control();
    const FootprintLattice lattice =
        footprint_lattice(fl, rl, hw, robot.safety_radius);
    const int rows = static_cast<int>(obstacle_mask.rows());
    const int cols = static_cast<int>(obstacle_mask.cols());
    FootprintHits out;
    out.n_samples = static_cast<int>(lattice.fwd.size());
    for (int k = 0; k < out.n_samples; ++k) {
        const double x = center[0] + fx * lattice.fwd[static_cast<size_t>(k)] +
                         lx * lattice.lat[static_cast<size_t>(k)];
        const double y = center[1] + fy * lattice.fwd[static_cast<size_t>(k)] +
                         ly * lattice.lat[static_cast<size_t>(k)];
        const int xi = static_cast<int>((x - origin[0]) / resolution);
        const int yi = static_cast<int>((y - origin[1]) / resolution);
        if (0 <= xi && xi < rows && 0 <= yi && yi < cols && obstacle_mask(xi, yi)) {
            out.hits.emplace_back(lattice.fwd[static_cast<size_t>(k)],
                                  lattice.lat[static_cast<size_t>(k)]);
        }
    }
    return out;
}

double speed_from_clearance(double clearance_m, double v_prev, double v_open,
                            double vx_min, double clear_c0_m, double clear_open_m,
                            double t_react_s) {
    const double c_eff = std::max(0.0, clearance_m - v_prev * t_react_s);
    return interp_clamped(c_eff, {clear_c0_m, clear_open_m}, {vx_min, v_open});
}

double interp_clamped(double x, const std::vector<double>& xs,
                      const std::vector<double>& ys) {
    const int n = static_cast<int>(xs.size());
    if (n == 0) return 0.0;
    if (n == 1 || x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    int j = 1;
    while (j < n && xs[static_cast<size_t>(j)] < x) ++j;
    const double x0 = xs[static_cast<size_t>(j - 1)], x1 = xs[static_cast<size_t>(j)];
    if (x1 == x0) return ys[static_cast<size_t>(j - 1)];
    return ys[static_cast<size_t>(j - 1)] +
           (x - x0) * (ys[static_cast<size_t>(j)] - ys[static_cast<size_t>(j - 1)]) / (x1 - x0);
}

}  // namespace tinynav::planning
