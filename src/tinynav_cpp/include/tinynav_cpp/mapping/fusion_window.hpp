// Port of reference/tinynav/core/fusion_window.py — the relocalization fusion window.
// The Python reads TINYNAV_FUSE_WINDOW from the environment; here the window is a
// constructor argument (the component layer owns env parsing).
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "tinynav_cpp/kernels/pose_graph_solver.hpp"  // RelativePoseConstraint

namespace tinynav::mapping {

class FusionWindow {
 public:
  // Sized off the measured single-observation noise (p50 0.3m standing still),
  // not off the aliasing — see the reference module docstring. Default 5 matches
  // TINYNAV_FUSE_WINDOW's default.
  explicit FusionWindow(int fuse_window = 5) : fuse_window_(fuse_window) {}

  int fuse_window() const { return fuse_window_; }

  // Port of reference/tinynav/core/fusion_window.py::select_fusion_constraints.
  // The newest `fuse_window` constraints, newest last. `odom_poses[i]` is the odom
  // pose constraint i was observed at — unused here, kept because the caller in
  // map_node has it (see the reference docstring).
  std::vector<kernels::RelativePoseConstraint> select_fusion_constraints(
      const std::vector<kernels::RelativePoseConstraint>& constraints,
      const std::vector<Eigen::Matrix4d>& odom_poses) const;

 private:
  int fuse_window_;
};

}  // namespace tinynav::mapping
