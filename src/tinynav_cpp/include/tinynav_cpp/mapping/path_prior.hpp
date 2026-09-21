// Port of reference/tinynav/core/path_prior.py + path_speed.py + path_climb.py —
// capture-path priors (climb region, capture speed). Pure Eigen/std; no ROS.
//
// The Python has PathSpeedIndex/PathClimbIndex subclass PathSampleIndex; here they
// compose one instead (the AGENTS.md composition rule).
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace tinynav::mapping {

// Nav-time trajectory-association radius: how close the robot must be to the capture
// path for that path point's label to apply (path_prior.py::ASSOC_M).
inline constexpr double ASSOC_M = 1.5;

// Nearest-neighbour index over 3D points. Internally a brute-force O(n) scan today —
// capture paths are a few hundred samples — with the interface shaped so a real
// kd-tree can replace the internals without touching callers.
class KdTree {
 public:
  KdTree() = default;
  explicit KdTree(const Eigen::MatrixXd& points);  // (N, 3)

  // 3D nearest neighbour; returns the index, or -1 when empty.
  int nearest(const Eigen::Vector3d& p, double& dist_out) const;
  // Indices within horizontal (xy) radius of p.
  std::vector<int> within_horizontal(const Eigen::Vector2d& p, double radius_m) const;
  int size() const { return static_cast<int>(points_.rows()); }

 private:
  Eigen::MatrixXd points_;  // (N, 3)
};

// Port of reference/tinynav/core/path_prior.py::is_stale (std::filesystem mtimes).
bool is_stale(const std::string& map_path, const std::string& prior_filename);

// Port of reference/tinynav/core/path_prior.py::poses_to_positions.
// poses: timestamp_ns -> 4x4 (std::map iterates in key order == capture order).
// Returns (N, 3).
Eigen::MatrixXd poses_to_positions(const std::map<int64_t, Eigen::Matrix4d>& poses);

// Port of reference/tinynav/core/path_prior.py::horizontal_arclength.
// Cumulative horizontal (xy) arclength along an ordered (N, 3) path; returns (N,).
Eigen::VectorXd horizontal_arclength(const Eigen::MatrixXd& pos);

// Minimal .npy reader for the (N, >=4) little-endian float32/float64 C-order arrays
// the prior bakes write (path_speed.npy / path_climb.npy). NOT a general numpy
// loader — poses.npy is a pickled dict and stays unreadable on purpose.
Eigen::MatrixXd load_npy_matrix(const std::string& npy_path);

// Port of reference/tinynav/core/path_prior.py::PathSampleIndex.
// pts is (N, >=4) with columns [x, y, z, label...].
class PathSampleIndex {
 public:
  explicit PathSampleIndex(const Eigen::MatrixXd& pts, double assoc_m = ASSOC_M);
  static PathSampleIndex load(const std::string& npy_path, double assoc_m = ASSOC_M);

  // Column-3 label of the nearest capture-path sample within assoc_m, or nullopt
  // when the index is empty or the robot is off the recorded path.
  std::optional<double> nearest_value(const Eigen::Vector3d& position_xyz) const;

  // Region form: the (M, 3) positions of samples whose label reaches min_label,
  // within radius_m horizontally.
  Eigen::MatrixXd samples_within(const Eigen::Vector3d& position_xyz,
                                 double radius_m, double min_label) const;

  const Eigen::MatrixXd& pts() const { return pts_; }
  double assoc_m() const { return assoc_m_; }

 private:
  Eigen::MatrixXd pts_;
  double assoc_m_;
  KdTree tree_;
};

// ------------------------------------------------------------------ path_speed
// Defaults from path_speed.py (tunable).
struct PathSpeedParams {
  double win_m = 1.0;       // half-window horizontal arclength (m)
  double pct = 75.0;        // percentile of in-window segment speeds
  double max_speed = 3.0;   // above this = VIO teleport / bad dt -> dropped
  double min_dt_s = 1e-3;   // segments with dt <= this are dropped
};

// Port of reference/tinynav/core/path_speed.py::compute_path_speed.
// Returns (N, 4) [x, y, z, speed]; speed is NaN where no valid segment is in range.
Eigen::MatrixXd compute_path_speed(const std::map<int64_t, Eigen::Matrix4d>& poses,
                                   const PathSpeedParams& params = {});

// Port of reference/tinynav/core/path_speed.py::PathSpeedIndex (composition).
class PathSpeedIndex {
 public:
  explicit PathSpeedIndex(PathSampleIndex index) : index_(std::move(index)) {}
  static PathSpeedIndex load(const std::string& npy_path, double assoc_m = ASSOC_M) {
    return PathSpeedIndex(PathSampleIndex::load(npy_path, assoc_m));
  }

  // Capture speed (m/s) at the nearest capture-path sample within assoc_m; +inf
  // when off-path or the sample has no valid speed ("no cap", the fail-safe).
  double speed_cap(const Eigen::Vector3d& position_xyz) const;

  const PathSampleIndex& index() const { return index_; }

 private:
  PathSampleIndex index_;
};

// ------------------------------------------------------------------ path_climb
// Defaults from path_climb.py (tunable).
struct PathClimbParams {
  double win_m = 1.0;        // half-window horizontal arclength (m)
  double min_rise = 0.12;    // min sustained net |dz| over the window (m)
  double consistency = 0.6;  // min fraction of moving in-window steps matching the net sign
  double max_step_dz = 0.5;  // single-step |dz| above this = VIO teleport
  double noise_dz = 0.02;    // |dz| below this is flat: excluded from the sign vote (m)
};

// Column 3 of a label array is a float; this is the one place that reads it as a flag.
inline constexpr double CLIMBING = 0.5;

// Port of reference/tinynav/core/path_climb.py::n_climbing.
int n_climbing(const Eigen::MatrixXd& labels);

// Port of reference/tinynav/core/path_climb.py::compute_path_climb.
// Returns (N, 4) [x, y, z, is_climbing].
Eigen::MatrixXd compute_path_climb(const std::map<int64_t, Eigen::Matrix4d>& poses,
                                   const PathClimbParams& params = {});

// Port of reference/tinynav/core/path_climb.py::PathClimbIndex (composition).
class PathClimbIndex {
 public:
  explicit PathClimbIndex(PathSampleIndex index) : index_(std::move(index)) {}
  static PathClimbIndex load(const std::string& npy_path, double assoc_m = ASSOC_M) {
    return PathClimbIndex(PathSampleIndex::load(npy_path, assoc_m));
  }

  // The (M, 3) positions of climbing samples within radius_m horizontally.
  Eigen::MatrixXd climbing_within(const Eigen::Vector3d& position_xyz, double radius_m) const;
  // Whether the nearest sample within assoc_m is climbing. Off the recorded path the
  // label is not trusted => flat/strict, the safe default.
  bool on_stairs(const Eigen::Vector3d& position_xyz) const;

  const PathSampleIndex& index() const { return index_; }

 private:
  PathSampleIndex index_;
};

}  // namespace tinynav::mapping
