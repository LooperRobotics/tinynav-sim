// Port of reference/tinynav/core/math_utils.py
#include "tinynav_cpp/core/math.hpp"

#include <cmath>
#include <numeric>
#include <unordered_map>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace tinynav::core {

namespace {

cv::Mat points3_to_cv(const Eigen::MatrixX3d& m) {
    cv::Mat out(static_cast<int>(m.rows()), 3, CV_64F);
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < 3; ++j) out.at<double>(i, j) = m(i, j);
    return out;
}

cv::Mat points2_to_cv(const Eigen::MatrixX2d& m) {
    cv::Mat out(static_cast<int>(m.rows()), 2, CV_64F);
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < 2; ++j) out.at<double>(i, j) = m(i, j);
    return out;
}

cv::Mat k_to_cv(const Eigen::Matrix3d& K) {
    cv::Mat out(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out.at<double>(i, j) = K(i, j);
    return out;
}

// T = [R|t] from an OpenCV rvec/tvec pair.
Eigen::Matrix4d rt_to_pose(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) T(r, c) = R_cv.at<double>(r, c);
    for (int i = 0; i < 3; ++i) T(i, 3) = tvec.at<double>(i);
    return T;
}

}  // namespace

Eigen::Matrix3d rotvec_to_matrix(const Eigen::Vector3d& rv) {
    const double theta = rv.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Vector3d axis = rv / theta;
    const double x = axis.x(), y = axis.y(), z = axis.z();
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double C = 1.0 - c;
    Eigen::Matrix3d R;
    R << c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C;
    return R;
}

Eigen::Matrix3d quat_to_matrix(const Eigen::Vector4d& q) {
    const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
    const double xx = x * x;
    const double yy = y * y;
    const double zz = z * z;
    const double xy = x * y;
    const double xz = x * z;
    const double yz = y * z;
    const double xw = x * w;
    const double yw = y * w;
    const double zw = z * w;
    Eigen::Matrix3d R;
    R(0, 0) = 1.0 - 2.0 * (yy + zz);
    R(0, 1) = 2.0 * (xy - zw);
    R(0, 2) = 2.0 * (xz + yw);
    R(1, 0) = 2.0 * (xy + zw);
    R(1, 1) = 1.0 - 2.0 * (xx + zz);
    R(1, 2) = 2.0 * (yz - xw);
    R(2, 0) = 2.0 * (xz - yw);
    R(2, 1) = 2.0 * (yz + xw);
    R(2, 2) = 1.0 - 2.0 * (xx + yy);
    return R;
}

Eigen::Vector4d matrix_to_quat(const Eigen::Matrix3d& R) {
    const double m00 = R(0, 0), m01 = R(0, 1), m02 = R(0, 2);
    const double m10 = R(1, 0), m11 = R(1, 1), m12 = R(1, 2);
    const double m20 = R(2, 0), m21 = R(2, 1), m22 = R(2, 2);
    const double trace = m00 + m11 + m22;
    double qw, qx, qy, qz;
    if (trace > 0) {
        const double S = std::sqrt(trace + 1.0) * 2.0;
        qw = 0.25 * S;
        qx = (m21 - m12) / S;
        qy = (m02 - m20) / S;
        qz = (m10 - m01) / S;
    } else if ((m00 > m11) && (m00 > m22)) {
        const double S = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        qw = (m21 - m12) / S;
        qx = 0.25 * S;
        qy = (m01 + m10) / S;
        qz = (m02 + m20) / S;
    } else if (m11 > m22) {
        const double S = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        qw = (m02 - m20) / S;
        qx = (m01 + m10) / S;
        qy = 0.25 * S;
        qz = (m12 + m21) / S;
    } else {
        const double S = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        qw = (m10 - m01) / S;
        qx = (m02 + m20) / S;
        qy = (m12 + m21) / S;
        qz = 0.25 * S;
    }
    return Eigen::Vector4d(qx, qy, qz, qw);
}

Eigen::Matrix3d rot_from_two_vector(const Eigen::Vector3d& a_in, const Eigen::Vector3d& b_in) {
    const Eigen::Vector3d a = a_in.normalized();
    const Eigen::Vector3d b = b_in.normalized();
    Eigen::Vector3d v = a.cross(b);
    const double c = a.dot(b);

    if (v.norm() < 1e-8 && std::abs(c - 1.0) < 1e-8) {
        return Eigen::Matrix3d::Identity();  // No rotation needed
    }

    const double s = v.norm();
    v /= s;
    const double vx = v.x(), vy = v.y(), vz = v.z();
    Eigen::Matrix3d R;
    R << c + vx * vx * (1 - c), vx * vy * (1 - c) - vz * s, vx * vz * (1 - c) + vy * s,
        vy * vx * (1 - c) + vz * s, c + vy * vy * (1 - c), vy * vz * (1 - c) - vx * s,
        vz * vx * (1 - c) - vy * s, vz * vy * (1 - c) + vx * s, c + vz * vz * (1 - c);
    return R;
}

double wrap_angle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

Eigen::MatrixX3d depth_to_cloud(const Eigen::MatrixXd& depth, const Eigen::Matrix3d& K,
                                int step, double max_dist) {
    const double fx = K(0, 0);
    const double fy = K(1, 1);
    const double cx = K(0, 2);
    const double cy = K(1, 2);

    std::vector<Eigen::Vector3d> pts;
    for (int v = 0; v < depth.rows(); v += step) {
        for (int u = 0; u < depth.cols(); u += step) {
            const double z = depth(v, u);
            if (z > 0.0 && z <= max_dist) {
                pts.emplace_back((u - cx) * z / fx, (v - cy) * z / fy, z);
            }
        }
    }
    Eigen::MatrixX3d out(static_cast<int>(pts.size()), 3);
    for (int i = 0; i < out.rows(); ++i) {
        out.row(i) = pts[i];
    }
    return out;
}

ProcessedKeypoints process_keypoints(const Eigen::MatrixX2d& kpts_prev,
                                     const Eigen::MatrixX2d& kpts_curr,
                                     const std::vector<int>& idx_valid,
                                     const Eigen::MatrixXd& depth,
                                     const Eigen::Matrix3d& K) {
    const int n = static_cast<int>(kpts_prev.rows());
    Eigen::MatrixX3d points_3d(n, 3);
    Eigen::MatrixX2d points_2d(n, 2);
    std::vector<int> valid_idx(n);
    int valid_count = 0;

    for (int i = 0; i < n; ++i) {
        const int u = static_cast<int>(kpts_curr(i, 0));
        const int v = static_cast<int>(kpts_curr(i, 1));
        if (v >= 0 && v < depth.rows() && u >= 0 && u < depth.cols()) {
            const double Z = depth(v, u);
            if (Z > 0.1 && Z < 10.0) {
                points_3d.row(valid_count) << (kpts_curr(i, 0) - K(0, 2)) * Z / K(0, 0),
                    (kpts_curr(i, 1) - K(1, 2)) * Z / K(1, 1), Z;
                points_2d.row(valid_count) = kpts_prev.row(i);
                valid_idx[valid_count] = idx_valid[i];
                ++valid_count;
            }
        }
    }

    ProcessedKeypoints out;
    out.points_3d = points_3d.topRows(valid_count);
    out.points_2d = points_2d.topRows(valid_count);
    out.valid_idx.assign(valid_idx.begin(), valid_idx.begin() + valid_count);
    return out;
}

EstimatePoseResult estimate_pose(const Eigen::MatrixX2d& kpts_prev,
                                 const Eigen::MatrixX2d& kpts_curr,
                                 const Eigen::MatrixXd& depth,
                                 const Eigen::Matrix3d& K,
                                 const std::vector<int>& idx_valid_in) {
    EstimatePoseResult result;
    std::vector<int> idx_valid = idx_valid_in;
    if (idx_valid.empty()) {
        idx_valid.resize(static_cast<size_t>(kpts_prev.rows()));
        std::iota(idx_valid.begin(), idx_valid.end(), 0);
    }

    const ProcessedKeypoints pk = process_keypoints(kpts_prev, kpts_curr, idx_valid, depth, K);
    if (pk.points_3d.rows() < 6) {
        return result;
    }
    cv::Mat rvec, tvec, inliers;
    const bool ok = cv::solvePnPRansac(points3_to_cv(pk.points_3d), points2_to_cv(pk.points_2d),
                                       k_to_cv(K), cv::noArray(), rvec, tvec,
                                       false, 100, 2.0, 0.999, inliers, cv::SOLVEPNP_EPNP);
    // Python only checks `not success` and would crash on inliers=None; an
    // empty inlier set is folded into the same failure return instead.
    if (!ok || inliers.empty()) {
        return result;
    }
    result.success = true;
    result.pose = rt_to_pose(rvec, tvec);
    const int n_in = inliers.rows;
    result.inliers_2d.resize(n_in, 2);
    result.inliers_3d.resize(n_in, 3);
    result.inlier_idx_original.resize(n_in);
    for (int i = 0; i < n_in; ++i) {
        const int j = inliers.at<int>(i);
        result.inliers_2d.row(i) = pk.points_2d.row(j);
        result.inliers_3d.row(i) = pk.points_3d.row(j);
        result.inlier_idx_original[i] = pk.valid_idx[j];
    }
    return result;
}

PnpRerankResult rerank_by_pnp_inliers(
    const std::vector<std::pair<Eigen::MatrixX3d, Eigen::MatrixX2d>>& pnp_candidates,
    const Eigen::Matrix3d& K,
    int min_point_count,
    int min_inlier_count) {
    PnpRerankResult best;

    for (size_t ci = 0; ci < pnp_candidates.size(); ++ci) {
        const Eigen::MatrixX3d& points_3d = pnp_candidates[ci].first;
        const Eigen::MatrixX2d& points_2d = pnp_candidates[ci].second;
        const int point_count = static_cast<int>(points_2d.rows());
        if (point_count <= min_point_count) {
            continue;
        }

        cv::Mat rvec, tvec, inliers;
        const bool ok = cv::solvePnPRansac(points3_to_cv(points_3d), points2_to_cv(points_2d),
                                           k_to_cv(K), cv::noArray(), rvec, tvec,
                                           false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
        const int inlier_count = inliers.empty() ? 0 : inliers.rows;
        if (!ok || inliers.empty() || inlier_count < min_inlier_count) {
            continue;
        }

        if (inlier_count > best.best_inlier_count) {
            best.best_candidate_index = static_cast<int>(ci);
            best.best_inlier_count = inlier_count;
            best.best_point_count = point_count;
            best.pose = rt_to_pose(rvec, tvec);
        }
    }

    if (best.best_candidate_index < 0) {
        return best;
    }
    best.success = true;
    best.inlier_ratio =
        static_cast<double>(best.best_inlier_count) / static_cast<double>(best.best_point_count);
    return best;
}

UnionFind::UnionFind(int n) : parent_(n), rank_(n, 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
}

int UnionFind::find(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]];
        x = parent_[x];
    }
    return x;
}

int UnionFind::find_const(int x) const {
    while (parent_[x] != x) {
        x = parent_[x];
    }
    return x;
}

void UnionFind::unite(int a, int b) {
    int ra = find(a);
    int rb = find(b);
    if (ra == rb) {
        return;
    }
    if (rank_[ra] < rank_[rb]) {
        std::swap(ra, rb);
    }
    parent_[rb] = ra;
    if (rank_[ra] == rank_[rb]) {
        ++rank_[ra];
    }
}

UnionFind uf_init(int n) {
    return UnionFind(n);
}

void uf_union(int a, int b, UnionFind& uf) {
    uf.unite(a, b);
}

std::vector<std::vector<int>> uf_all_sets_list(const UnionFind& uf, int min_component_size) {
    // Ascending scan groups each component under its root, so each part comes
    // out sorted and the outer order is by first (= smallest) member. fufpy's
    // own outer order is unspecified; the sorted parts match the Python.
    std::unordered_map<int, size_t> root_to_part;
    std::vector<std::vector<int>> parts;
    for (int x = 0; x < uf.size(); ++x) {
        const int root = uf.find_const(x);
        const auto [it, inserted] = root_to_part.emplace(root, parts.size());
        if (inserted) {
            parts.emplace_back();
        }
        parts[it->second].push_back(x);
    }
    std::vector<std::vector<int>> out;
    for (auto& part : parts) {
        if (static_cast<int>(part.size()) >= min_component_size) {
            out.push_back(std::move(part));
        }
    }
    return out;
}

}  // namespace tinynav::core
