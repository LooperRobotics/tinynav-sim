// Port of reference/tinynav/core/fusion_window.py.
#include "tinynav_cpp/mapping/fusion_window.hpp"

#include <algorithm>

namespace tinynav::mapping {

std::vector<kernels::RelativePoseConstraint> FusionWindow::select_fusion_constraints(
    const std::vector<kernels::RelativePoseConstraint>& constraints,
    const std::vector<Eigen::Matrix4d>& odom_poses) const {
  (void)odom_poses;  // unused, kept for signature parity with the Python
  if (constraints.empty() || fuse_window_ <= 0) {
    return constraints;
  }
  const size_t n = std::min(constraints.size(), static_cast<size_t>(fuse_window_));
  return std::vector<kernels::RelativePoseConstraint>(constraints.end() - n, constraints.end());
}

}  // namespace tinynav::mapping
