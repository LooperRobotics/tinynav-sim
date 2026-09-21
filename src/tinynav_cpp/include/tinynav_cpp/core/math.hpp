// Port of reference/tinynav/core/math_utils.py — pure-math subset.
// The ROS message adapters (np2msg / np2tf / tf2np / msg2np / pose_msg2np) are
// not ported: they belong to the component layer, not this pure library.
// The fufpy-backed union-find wrappers are re-implemented on a local UnionFind.
//
// Conventions, confirmed against the reference sources:
//  - quat is [x, y, z, w] everywhere (scipy Rotation.as_quat / ROS field order;
//    math_utils.py:30 quat_to_matrix docstring).
//  - pose7 is [x, y, z, qx, qy, qz, qw]: reference/tinynav/core/planning_node.py:305
//    `qx, qy, qz, qw = pose7[3], pose7[4], pose7[5], pose7[6]`, and
//    publish_selected_path's `x, y, z, qx, qy, qz, qw = trajectories[i][j]`.
//  - heading is camera-convention: body +Z is forward, so world heading is the
//    body +Z axis projected onto world XY (heading_of), NOT textbook body-X yaw.
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace tinynav::core {

// Port of reference/tinynav/core/math_utils.py::rotvec_to_matrix
// Rodrigues' formula; identity for |rv| < 1e-8.
Eigen::Matrix3d rotvec_to_matrix(const Eigen::Vector3d& rv);

// Port of reference/tinynav/core/math_utils.py::quat_to_matrix
// q = [x, y, z, w].
Eigen::Matrix3d quat_to_matrix(const Eigen::Vector4d& q);

// Port of reference/tinynav/core/math_utils.py::matrix_to_quat
// Hand-branched (trace / largest-diagonal) conversion, branches aligned
// line-by-line with the Python. Returns [x, y, z, w]; the sign is whatever the
// branch produces, so q and -q are both valid round-trips.
Eigen::Vector4d matrix_to_quat(const Eigen::Matrix3d& rot);

// Port of reference/tinynav/core/math_utils.py::rot_from_two_vector
// R such that R @ a = b (a, b need not be unit). Only the no-rotation case is
// special-cased; anti-parallel a, b degenerate exactly as the Python does.
Eigen::Matrix3d rot_from_two_vector(const Eigen::Vector3d& a, const Eigen::Vector3d& b);

// Port of reference/tinynav/core/math_utils.py::wrap_angle
// Angle folded into (-pi, pi]. Scalar form (the Python also vectorizes).
double wrap_angle(double a);

// Port of reference/tinynav/core/math_utils.py::heading_of
// World heading (rad): body +Z (this stack's forward) projected onto world XY.
// Like the Python's `np.asarray(rot)[:3, :3]`, any matrix >= 3x3 is accepted
// and only its top-left 3x3 is read.
template <typename Derived>
double heading_of(const Eigen::MatrixBase<Derived>& rot) {
    const Eigen::Vector3d fwd =
        rot.template topLeftCorner<3, 3>() * Eigen::Vector3d(0.0, 0.0, 1.0);
    return std::atan2(fwd.y(), fwd.x());
}

// Port of reference/tinynav/core/math_utils.py::depth_to_cloud
// Strided depth unprojection; rows in (v, u) scan order. 0x3 when empty.
Eigen::MatrixX3d depth_to_cloud(const Eigen::MatrixXd& depth, const Eigen::Matrix3d& K,
                                int step = 10, double max_dist = 1e9);

// Port of reference/tinynav/core/math_utils.py::process_keypoints
struct ProcessedKeypoints {
    Eigen::MatrixX3d points_3d;    // back-projected at kpts_curr, 0.1 < Z < 10
    Eigen::MatrixX2d points_2d;    // matching kpts_prev rows
    std::vector<int> valid_idx;    // matching idx_valid entries
};
ProcessedKeypoints process_keypoints(const Eigen::MatrixX2d& kpts_prev,
                                     const Eigen::MatrixX2d& kpts_curr,
                                     const std::vector<int>& idx_valid,
                                     const Eigen::MatrixXd& depth,
                                     const Eigen::Matrix3d& K);

// Port of reference/tinynav/core/math_utils.py::estimate_pose
// Same API semantics as the Python minus the lru_cache: failure is reported by
// success=false with pose=identity and empty inlier outputs. An empty idx_valid
// means "arange(N)", matching idx_valid=None in Python.
struct EstimatePoseResult {
    bool success = false;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    Eigen::MatrixX2d inliers_2d;
    Eigen::MatrixX3d inliers_3d;
    std::vector<int> inlier_idx_original;
};
EstimatePoseResult estimate_pose(const Eigen::MatrixX2d& kpts_prev,
                                 const Eigen::MatrixX2d& kpts_curr,
                                 const Eigen::MatrixXd& depth,
                                 const Eigen::Matrix3d& K,
                                 const std::vector<int>& idx_valid = {});

// Port of reference/tinynav/core/math_utils.py::rerank_by_pnp_inliers
// Defaults mirror the Python failure tuple (False, eye(4), -inf, -1, 0, 0).
struct PnpRerankResult {
    bool success = false;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    double inlier_ratio = -std::numeric_limits<double>::infinity();
    int best_candidate_index = -1;
    int best_inlier_count = 0;
    int best_point_count = 0;
};
PnpRerankResult rerank_by_pnp_inliers(
    const std::vector<std::pair<Eigen::MatrixX3d, Eigen::MatrixX2d>>& pnp_candidates,
    const Eigen::Matrix3d& K,
    int min_point_count = 80,
    int min_inlier_count = 50);

// Port of reference/tinynav/core/math_utils.py::uf_init / uf_union /
// uf_all_sets_list — fufpy replaced by a local path-compression union-find.
class UnionFind {
  public:
    explicit UnionFind(int n);
    int find(int x);
    int find_const(int x) const;  // no path compression; for const queries
    void unite(int a, int b);
    int size() const { return static_cast<int>(parent_.size()); }

  private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

UnionFind uf_init(int n);
void uf_union(int a, int b, UnionFind& uf);
// Components with size >= min_component_size, each sorted ascending.
std::vector<std::vector<int>> uf_all_sets_list(const UnionFind& uf,
                                               int min_component_size = 1);

}  // namespace tinynav::core
