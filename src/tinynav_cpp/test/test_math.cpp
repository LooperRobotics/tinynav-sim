// Smoke test for the math_utils port: quat -> matrix -> quat round trip and
// estimate_pose on a synthetic, all-inlier scene.
#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/core/math.hpp"

namespace tinynav::core {
namespace {

TEST(MathTest, QuatMatrixQuatRoundTrip) {
    Eigen::Vector4d quats[4];
    quats[0] = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);          // identity
    quats[1] = Eigen::Vector4d(0.5, 0.5, 0.5, 0.5);          // 120 deg about (1,1,1)
    quats[2] = Eigen::Vector4d(0.0, 0.0, std::sin(M_PI / 4), std::cos(M_PI / 4));
    quats[3] = Eigen::Vector4d(0.2, -0.3, 0.1, 0.9).normalized();
    for (const Eigen::Vector4d& q : quats) {
        const Eigen::Matrix3d R = quat_to_matrix(q);
        const Eigen::Vector4d q2 = matrix_to_quat(R);
        // Sign is branch-dependent; q2 == q or q2 == -q are both correct.
        EXPECT_NEAR(std::abs(q2.dot(q)), 1.0, 1e-12) << "q = " << q.transpose();
        const Eigen::Matrix3d R2 = quat_to_matrix(q2);
        EXPECT_TRUE(R2.isApprox(R, 1e-12)) << "q = " << q.transpose();
    }
}

TEST(MathTest, RotvecRoundTripAndHeading) {
    const Eigen::Vector3d rv(0.1, -0.2, 0.3);
    const Eigen::Matrix3d R = rotvec_to_matrix(rv);
    EXPECT_TRUE(R.isApprox(quat_to_matrix(matrix_to_quat(R)), 1e-12));
    // Camera convention: body +Z forward. Identity rotation -> heading 0.
    EXPECT_NEAR(heading_of(Eigen::Matrix3d::Identity()), 0.0, 1e-12);
    // (-pi, pi] fold; note atan2 puts exact +/-pi on the sign of sin(a).
    EXPECT_NEAR(wrap_angle(2.0 * M_PI + 0.3), 0.3, 1e-12);
    EXPECT_NEAR(wrap_angle(-0.7), -0.7, 1e-12);
}

TEST(MathTest, EstimatePoseSynthetic) {
    const Eigen::Matrix3d K =
        (Eigen::Matrix3d() << 525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0).finished();
    // Ground-truth object->camera motion, small enough to keep pixels in frame.
    const Eigen::Matrix3d R_true = rotvec_to_matrix(Eigen::Vector3d(0.03, -0.05, 0.08));
    const Eigen::Vector3d t_true(0.05, -0.04, 0.15);

    Eigen::MatrixXd depth = Eigen::MatrixXd::Zero(480, 640);
    std::vector<Eigen::Vector2d> prev, curr;
    for (int v = 60; v < 420; v += 40) {
        for (int u = 60; u < 580; u += 40) {
            const double z = 1.5 + ((u * 7 + v * 13) % 100) / 100.0 * 2.0;
            depth(v, u) = z;
            const Eigen::Vector3d p3((u - 320.0) * z / 525.0, (v - 240.0) * z / 525.0, z);
            const Eigen::Vector3d p_prev = R_true * p3 + t_true;
            prev.emplace_back(525.0 * p_prev.x() / p_prev.z() + 320.0,
                              525.0 * p_prev.y() / p_prev.z() + 240.0);
            curr.emplace_back(static_cast<double>(u), static_cast<double>(v));
        }
    }
    const int n = static_cast<int>(prev.size());
    ASSERT_GE(n, 6);
    Eigen::MatrixX2d kpts_prev(n, 2), kpts_curr(n, 2);
    for (int i = 0; i < n; ++i) {
        kpts_prev.row(i) = prev[i];
        kpts_curr.row(i) = curr[i];
    }

    const EstimatePoseResult result = estimate_pose(kpts_prev, kpts_curr, depth, K);
    ASSERT_TRUE(result.success);
    EXPECT_TRUE((result.pose.topLeftCorner<3, 3>().isApprox(R_true, 1e-6)));
    EXPECT_TRUE((result.pose.topRightCorner<3, 1>().isApprox(t_true, 1e-6)));
    // Clean synthetic data: every point is an inlier. Inlier order is OpenCV's,
    // so compare the index SET against arange(n).
    ASSERT_EQ(result.inliers_3d.rows(), n);
    ASSERT_EQ(result.inliers_2d.rows(), n);
    ASSERT_EQ(static_cast<int>(result.inlier_idx_original.size()), n);
    std::vector<int> idx = result.inlier_idx_original;
    std::sort(idx.begin(), idx.end());
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(idx[i], i);
    }
}

TEST(MathTest, UnionFindBasic) {
    UnionFind uf = uf_init(5);
    uf_union(0, 1, uf);
    uf_union(1, 2, uf);
    uf_union(3, 4, uf);
    const std::vector<std::vector<int>> sets = uf_all_sets_list(uf);
    ASSERT_EQ(sets.size(), 2u);
    EXPECT_EQ(sets[0], (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(sets[1], (std::vector<int>{3, 4}));
    // min_component_size filters the singleton-free case here: all merged.
    const auto big = uf_all_sets_list(uf, 3);
    ASSERT_EQ(big.size(), 1u);
    EXPECT_EQ(big[0], (std::vector<int>{0, 1, 2}));
}

}  // namespace
}  // namespace tinynav::core
