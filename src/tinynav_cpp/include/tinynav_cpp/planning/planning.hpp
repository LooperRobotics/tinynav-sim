// Port of reference/tinynav/core/planning_node.py — the pure planning kernels:
// occupancy raycasting, obstacle map + ESDF, trajectory library, DWA scoring
// and selection. The ROS shell of PlanningNode (topics, QoS, timers, TF) is
// components/planning_component.cpp; every function here keeps its Python name
// (snake_case; the njit functions keep their exact spelling too).
//
// Semantics notes carried over from the Python:
//  - pose7 rows are [x, y, z, qx, qy, qz, qw] (see core/math.hpp).
//  - Image-index expressions like `int((x - origin) / resolution)` are C
//    truncation toward zero in BOTH languages, so static_cast<int> matches
//    exactly (including the negative-epsilon -> 0 quirk near the border).
//  - `round()` in the njit raycaster is banker's rounding (numpy/numba
//    round-half-to-even); std::nearbyint under the default FP rounding mode
//    matches it. np.floor is std::floor.
//  - The 2D maps (ESDF / route fields / obstacle mask) are indexed [row=x]
//    like the Python's (h, w) arrays whose row is the grid x axis.
#pragma once

#include <Eigen/Dense>

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "tinynav_cpp/core/math.hpp"
#include "tinynav_cpp/core/robot_specs.hpp"
#include "tinynav_cpp/planning/edt.hpp"

namespace tinynav::planning {

// ---------------------------------------------------------------------------
// Occupancy grid storage. Python: np.zeros(grid_shape) float64 indexed
// [x, y, z]; storage here is C-order ((x * ny + y) * nz + z), the same
// flattening reference/tinynav/cpp/raycast.cpp uses for its Eigen matrix.
class OccupancyGrid3D {
  public:
    OccupancyGrid3D() = default;
    OccupancyGrid3D(int nx, int ny, int nz)
        : nx_(nx), ny_(ny), nz_(nz), data_(size_t(nx) * ny * nz, 0.0) {}

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    Eigen::Vector3i shape() const { return {nx_, ny_, nz_}; }

    double& operator()(int x, int y, int z) {
        return data_[(size_t(x) * ny_ + y) * nz_ + z];
    }
    double operator()(int x, int y, int z) const {
        return data_[(size_t(x) * ny_ + y) * nz_ + z];
    }

    // Element-wise helpers for PlanningNode.update_occupancy_grid's
    // `grid *= 0.99; grid += new; grid = clip(grid, -0.2, 0.2)`.
    void scale(double s);
    void add(const OccupancyGrid3D& other);
    void clip(double lo, double hi);

  private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::vector<double> data_;
};

// ---------------------------------------------------------------------------
// Port of planning_node.py::run_raycasting_loopy (the njit kernel both
// planning_node and build_map_node use — NOT cpp/raycast.cpp, which is ported
// separately in kernels/raycast.hpp). depth is the 32FC1 image (float32).
OccupancyGrid3D run_raycasting_loopy(const Eigen::MatrixXf& depth_image,
                                     const Eigen::Matrix4d& T_cam_to_world,
                                     const Eigen::Vector3i& grid_shape, double fx,
                                     double fy, double cx, double cy,
                                     const Eigen::Vector3d& origin, int step,
                                     double resolution,
                                     bool filter_ground = false);

// Port of planning_node.py::build_obstacle_map. `min_span_map` mirrors the
// Python's optional (h, w) override array (nullptr == None).
Mask2D build_obstacle_map(const OccupancyGrid3D& occupancy_grid,
                          const Eigen::Vector3d& origin, double resolution,
                          double robot_z, const core::ObstacleConfig& config,
                          const Eigen::ArrayXXf* min_span_map = nullptr);

// Port of planning_node.py::roll_occupancy_grid. np.roll + zero-fill of the
// vacated band collapses to a gather with zero where the source index leaves
// the grid; identity when the shift is below half a voxel.
struct RolledGrid {
    OccupancyGrid3D grid;
    Eigen::Vector3d origin;
};
RolledGrid roll_occupancy_grid(const OccupancyGrid3D& occupancy_grid,
                               const Eigen::Vector3d& old_origin,
                               const Eigen::Vector3d& new_origin,
                               double resolution);

// ---------------------------------------------------------------------------
// Trajectory library. poses packs (num_trajectories, num_steps) pose7 rows
// row-major: trajectory t, step i at poses.row(t * num_steps + i).
struct TrajectorySet {
    Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor> poses;
    Eigen::MatrixX2d params;  // (num_trajectories, 2) [vx, omega_y]
    int num_trajectories = 0;
    int num_steps = 0;

    Eigen::Matrix<double, 1, 7> pose(int t, int i) const {
        return poses.row(t * num_steps + i);
    }
};

// Port of planning_node.py::generate_trajectory_library_3d (njit).
TrajectorySet generate_trajectory_library_3d(
    int num_samples = 15, double duration = 3.0, double dt = 0.1,
    const Eigen::Vector3d& init_p = Eigen::Vector3d::Zero(),
    const Eigen::Vector4d& init_q = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0),
    double max_linear_vel = 0.5, double max_angular_vel = M_PI / 3.0,
    double max_path_len_m = 1e9, double max_lat_acc = 1e9,
    double min_linear_vel = 0.0);

// Port of planning_node.py::generate_predefined_trajectory_vocabularies
// (the single fixed-speed reverse row, vx = -0.3).
TrajectorySet generate_predefined_trajectory_vocabularies(
    double duration = 3.0, double dt = 0.1,
    const Eigen::Vector3d& init_p = Eigen::Vector3d::Zero(),
    const Eigen::Vector4d& init_q = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0));

// np.concatenate over two TrajectorySets (same num_steps required).
TrajectorySet concatenate_trajectories(const TrajectorySet& a,
                                       const TrajectorySet& b);

// ---------------------------------------------------------------------------
// Port of planning_node.py::heading_of_pose7 (njit): world heading of a pose7
// row — body +Z projected onto world XY. Any 7-vector-like row works.
template <typename Derived>
double heading_of_pose7(const Eigen::DenseBase<Derived>& pose7) {
    const double qx = pose7[3], qy = pose7[4], qz = pose7[5], qw = pose7[6];
    return std::atan2(2.0 * (qy * qz - qw * qx), 2.0 * (qx * qz + qw * qy));
}

// Port of planning_node.py::angle_between (njit): |a - b| folded into [0, pi].
inline double angle_between(double a, double b) {
    const double d = a - b;
    return std::abs(std::atan2(std::sin(d), std::cos(d)));
}

// Port of planning_node.py::REVERSE_ENTER_M + reverse_armed.
inline constexpr double REVERSE_ENTER_M = 0.30;
bool reverse_armed(double front_clearance, int n_fwd_ok, double resolution);

// Port of planning_node.py::footprint_lattice (njit): body-frame (forward,
// left) samples covering the footprint, centre (index 0) first, pitch
// safety_radius * sqrt(2).
struct FootprintLattice {
    std::vector<double> fwd;
    std::vector<double> lat;
};
FootprintLattice footprint_lattice(double front_len, double rear_len,
                                   double half_w, double safety_radius);

// Port of planning_node.py::score_trajectories_by_ESDF (njit).
struct EsdfScoreResult {
    std::vector<double> scores;        // inf on collision
    std::vector<int> occ_points;       // step index of the min clearance
    std::vector<double> path_costs;    // worst centre-to-route distance (1e3 off-grid)
    std::vector<double> end_remainings;
    std::vector<double> end_heading_errs;
};
EsdfScoreResult score_trajectories_by_ESDF(
    const TrajectorySet& trajectories, const Eigen::ArrayXXf& ESDF_map,
    const Eigen::ArrayXXf& path_dist_map, const Eigen::ArrayXXf& remaining_map,
    const Eigen::ArrayXXf& route_heading_map, const Eigen::Vector3d& origin,
    double resolution, double safety_radius, double front_len, double rear_len,
    double half_w);

// Port of PlanningNode.score_trajectories: the ESDF scoring plus the reverse
// rows re-scored on the trajectory minus its first pose (a footprint sample
// underfoot must not veto the reverse family alike).
EsdfScoreResult score_trajectories(const TrajectorySet& trajectories,
                                   const Eigen::MatrixX2d& params,
                                   const Eigen::ArrayXXf& ESDF_map,
                                   const Eigen::ArrayXXf& path_dist_map,
                                   const Eigen::ArrayXXf& remaining_map,
                                   const Eigen::ArrayXXf& route_heading_map,
                                   const Eigen::Vector3d& origin,
                                   double resolution,
                                   const core::RobotConfig& robot);

// ---------------------------------------------------------------------------
// Port of planning_node.py::route_band_fade / route_heading_penalty.
inline double route_band_fade(double end_remaining_m, double terminal_band_m) {
    return std::min(1.0, end_remaining_m / std::max(terminal_band_m, 1e-6));
}
inline double route_heading_penalty(double weight, double heading_err_rad,
                                    double end_remaining_m,
                                    double terminal_band_m) {
    return weight * heading_err_rad * route_band_fade(end_remaining_m, terminal_band_m);
}

// Port of planning_node.py::build_route_fields. route_xy is (N, 2) world xy
// (0 rows when there is no route); shape is (rows, cols) of the ESDF map.
struct RouteFields {
    Eigen::ArrayXXf path_dist_map;
    Eigen::ArrayXXf remaining_map;
    Eigen::ArrayXXf route_heading_map;
    bool has_route = false;
};
RouteFields build_route_fields(const Eigen::MatrixX2d& route_xy,
                               const Eigen::Vector2i& shape,
                               const Eigen::Vector3d& origin, double resolution);

// ---------------------------------------------------------------------------
// The DWA cost of sync_callback's cost_function closure + argmin selection.
// Weights are plain members of the node in Python (not ROS parameters).
struct DwaWeights {
    double w_clearance = 200.0;
    double w_route_progress = 100.0;
    double w_path_follow = 80.0;
    double w_goal_terminal = 100.0;
    double route_terminal_band = 0.5;
    double w_route_heading = 60.0;
};

// |wrapped angle| between the pose's heading and the bearing from that pose to
// the goal (sync_callback's _end_heading_error).
double end_heading_error(const Eigen::Matrix<double, 1, 7>& pose7,
                         const Eigen::Vector3d& goal);

// Cost of candidate i; all indices/fields as produced above.
double trajectory_cost(int i, const TrajectorySet& trajectories,
                       const Eigen::MatrixX2d& params,
                       const EsdfScoreResult& scored, bool has_route,
                       const std::optional<Eigen::Vector3d>& target,
                       const Eigen::Vector2d& last_param, bool should_reverse,
                       const DwaWeights& weights);

// min(range(len), key=cost_function) — first minimal index on ties; -1 when empty.
int select_trajectory(const TrajectorySet& trajectories,
                      const Eigen::MatrixX2d& params,
                      const EsdfScoreResult& scored, bool has_route,
                      const std::optional<Eigen::Vector3d>& target,
                      const Eigen::Vector2d& last_param, bool should_reverse,
                      const DwaWeights& weights);

// ---------------------------------------------------------------------------
// Pure node-method helpers the component layer and the alignment tests share.

// Port of PlanningNode.camera_to_robot_center.
Eigen::Vector3d camera_to_robot_center(const Eigen::Matrix4d& T,
                                       const core::RobotConfig& robot);

// Port of PlanningNode._front_obstacle_dist: distance from the robot's front
// face to the nearest obstacle in the forward corridor; max_dist + 1.0 when clear.
double front_obstacle_dist(const Eigen::Matrix4d& T, const Mask2D& obstacle_mask,
                           const Eigen::Vector3d& origin, double resolution,
                           const core::RobotConfig& robot, double max_dist);

// Port of PlanningNode._footprint_hits: which footprint lattice samples stand
// on obstacle cells, as body-frame (forward, left) offsets.
struct FootprintHits {
    std::vector<std::pair<double, double>> hits;  // (off_fwd, off_lat)
    int n_samples = 0;
};
FootprintHits footprint_hits(const Eigen::Matrix4d& T, const Mask2D& obstacle_mask,
                             const Eigen::Vector3d& origin, double resolution,
                             const core::RobotConfig& robot);

// Port of PlanningNode._speed_from_clearance: linear peak-speed schedule from
// forward clearance, discounted for reaction latency; np.interp endpoint
// clamping outside [clear_c0_m, clear_open_m].
double speed_from_clearance(double clearance_m, double v_prev, double v_open,
                            double vx_min, double clear_c0_m,
                            double clear_open_m, double t_react_s);

// np.interp(x, xs, ys) for the scalar, monotone-xs case (endpoint clamping).
double interp_clamped(double x, const std::vector<double>& xs,
                      const std::vector<double>& ys);

}  // namespace tinynav::planning
