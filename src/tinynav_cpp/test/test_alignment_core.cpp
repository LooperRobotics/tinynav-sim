// Python-alignment tests for the core math and mapping kernels. Fixtures are
// exported by tools/export_core_fixtures.py from the UNMODIFIED reference:
// math_utils (the exact module the port copies), the image's pybind
// pose_graph_solve .so (the very kernel the C++ port replaces), and
// path_speed/path_climb. Tests SKIP when fixtures/ is absent.
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/core/imu.hpp"
#include "tinynav_cpp/core/math.hpp"
#include "tinynav_cpp/kernels/pose_graph_solver.hpp"
#include "tinynav_cpp/mapping/path_prior.hpp"

namespace tinynav {
namespace {

const char* kFixtureDir = "fixtures/core";

struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<double> data;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

NpyArray read_npy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[6] = {0};
    in.read(magic, 6);
    uint8_t ver[2] = {0, 0};
    in.read(reinterpret_cast<char*>(ver), 2);
    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    const bool is_f32 = header.find("<f4") != std::string::npos;
    const bool is_f64 = header.find("<f8") != std::string::npos;
    const bool is_u1 = header.find("u1") != std::string::npos;
    const bool is_i64 = header.find("i8") != std::string::npos;
    NpyArray out;
    const auto open = header.find('(');
    const auto close = header.find(')', open);
    std::stringstream ss(header.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) out.shape.push_back(std::stoll(token));
    }
    const int64_t total = out.numel();
    out.data.resize(static_cast<size_t>(total));
    if (is_f32) {
        std::vector<float> raw(static_cast<size_t>(total));
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total * 4));
        for (int64_t i = 0; i < total; ++i) out.data[static_cast<size_t>(i)] = raw[static_cast<size_t>(i)];
    } else if (is_f64) {
        in.read(reinterpret_cast<char*>(out.data.data()), static_cast<std::streamsize>(total * 8));
    } else if (is_u1) {
        std::vector<uint8_t> raw(static_cast<size_t>(total));
        in.read(reinterpret_cast<char*>(raw.data()), total);
        for (int64_t i = 0; i < total; ++i) out.data[static_cast<size_t>(i)] = raw[static_cast<size_t>(i)];
    } else if (is_i64) {
        std::vector<int64_t> raw(static_cast<size_t>(total));
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total * 8));
        for (int64_t i = 0; i < total; ++i) out.data[static_cast<size_t>(i)] = static_cast<double>(raw[static_cast<size_t>(i)]);
    }
    if (!in) throw std::runtime_error("short read " + path);
    return out;
}

bool fixtures_available() {
    std::ifstream in(std::string(kFixtureDir) + "/quat_in.npy", std::ios::binary);
    return in.good();
}

void expect_all_close(const NpyArray& got, const NpyArray& want, double tol,
                      const char* what) {
    ASSERT_EQ(got.numel(), want.numel()) << what;
    for (int64_t i = 0; i < want.numel(); ++i) {
        ASSERT_NEAR(got.data[static_cast<size_t>(i)], want.data[static_cast<size_t>(i)], tol)
            << what << " [" << i << "]";
    }
}

TEST(AlignmentCore, SkipWhenFixturesAbsent) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/core not exported; run tools/export_core_fixtures.py";
    }
}

TEST(AlignmentCore, QuatRotvecWrap) {
    if (!fixtures_available()) GTEST_SKIP();
    const NpyArray quats = read_npy(std::string(kFixtureDir) + "/quat_in.npy");
    const NpyArray want_q2m = read_npy(std::string(kFixtureDir) + "/quat_to_matrix_out.npy");
    const NpyArray want_m2q = read_npy(std::string(kFixtureDir) + "/matrix_to_quat_out.npy");
    NpyArray got;
    got.shape = {12, 3, 3};
    got.data.resize(12 * 9);
    for (int i = 0; i < 12; ++i) {
        const Eigen::Vector4d q(quats.data[static_cast<size_t>(i * 4)],
                                quats.data[static_cast<size_t>(i * 4 + 1)],
                                quats.data[static_cast<size_t>(i * 4 + 2)],
                                quats.data[static_cast<size_t>(i * 4 + 3)]);
        const Eigen::Matrix3d m = core::quat_to_matrix(q);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                got.data[static_cast<size_t>(i * 9 + r * 3 + c)] = m(r, c);
            }
        }
        // matrix_to_quat must reproduce the Python's branch-for-branch output
        const Eigen::Vector4d q2 = core::matrix_to_quat(m);
        for (int k = 0; k < 4; ++k) {
            ASSERT_NEAR(q2[k], want_m2q.data[static_cast<size_t>(i * 4 + k)], 1e-12)
                << "matrix_to_quat [" << i << "," << k << "]";
        }
    }
    expect_all_close(got, want_q2m, 1e-12, "quat_to_matrix");

    const NpyArray rotvecs = read_npy(std::string(kFixtureDir) + "/rotvec_in.npy");
    const NpyArray want_r2m = read_npy(std::string(kFixtureDir) + "/rotvec_to_matrix_out.npy");
    NpyArray got_r;
    got_r.shape = {12, 3, 3};
    got_r.data.resize(12 * 9);
    for (int i = 0; i < 12; ++i) {
        const Eigen::Matrix3d m = core::rotvec_to_matrix(Eigen::Vector3d(
            rotvecs.data[static_cast<size_t>(i * 3)], rotvecs.data[static_cast<size_t>(i * 3 + 1)],
            rotvecs.data[static_cast<size_t>(i * 3 + 2)]));
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                got_r.data[static_cast<size_t>(i * 9 + r * 3 + c)] = m(r, c);
            }
        }
    }
    expect_all_close(got_r, want_r2m, 1e-12, "rotvec_to_matrix");

    const NpyArray angles = read_npy(std::string(kFixtureDir) + "/wrap_in.npy");
    const NpyArray want_wrap = read_npy(std::string(kFixtureDir) + "/wrap_out.npy");
    for (int64_t i = 0; i < angles.numel(); ++i) {
        EXPECT_NEAR(core::wrap_angle(angles.data[static_cast<size_t>(i)]),
                    want_wrap.data[static_cast<size_t>(i)], 1e-12)
            << "wrap [" << i << "]";
    }
}

TEST(AlignmentCore, EstimatePose) {
    if (!fixtures_available()) GTEST_SKIP();
    const NpyArray K = read_npy(std::string(kFixtureDir) + "/pnp_K.npy");
    Eigen::Matrix3d k_mat;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) k_mat(r, c) = K.data[static_cast<size_t>(r * 3 + c)];
    }
    for (const char* c : {"exact", "noisy"}) {
        const std::string p = std::string(kFixtureDir) + "/pnp_" + c;
        const NpyArray prev = read_npy(p + "_prev.npy");
        const NpyArray curr = read_npy(p + "_curr.npy");
        const NpyArray depth = read_npy(p + "_depth.npy");
        const NpyArray want_T = read_npy(p + "_T.npy");
        const NpyArray want_inliers = read_npy(p + "_inliers.npy");
        const int n = static_cast<int>(prev.shape[0]);
        Eigen::MatrixX2d prev_m(n, 2), curr_m(n, 2);
        for (int i = 0; i < n; ++i) {
            prev_m(i, 0) = prev.data[static_cast<size_t>(i * 2)];
            prev_m(i, 1) = prev.data[static_cast<size_t>(i * 2 + 1)];
            curr_m(i, 0) = curr.data[static_cast<size_t>(i * 2)];
            curr_m(i, 1) = curr.data[static_cast<size_t>(i * 2 + 1)];
        }
        Eigen::MatrixXd depth_m(static_cast<int>(depth.shape[0]),
                                static_cast<int>(depth.shape[1]));
        for (int r = 0; r < depth_m.rows(); ++r) {
            for (int c2 = 0; c2 < depth_m.cols(); ++c2) {
                depth_m(r, c2) = depth.data[static_cast<size_t>(r * depth_m.cols() + c2)];
            }
        }
        const core::EstimatePoseResult got =
            core::estimate_pose(prev_m, curr_m, depth_m, k_mat, {});
        EXPECT_EQ(got.success, read_npy(p + "_state.npy").data[0] != 0) << c;
        if (!got.success) continue;
        // solvePnPRansac draws from OpenCV's GLOBAL RNG (and the pip cv2 4.11
        // differs from the system libopencv 4.5.4), so per-cell pose/inlier
        // equality is not reproducible across processes. The meaningful check
        // is ground-truth recovery: the ported call must refine to the same
        // accuracy class as the reference call did.
        const NpyArray gt = read_npy(p + "_gt.npy");
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner<3, 3>() = got.pose.topLeftCorner<3, 3>();
        T.topRightCorner<3, 1>() = got.pose.topRightCorner<3, 1>();
        for (int r = 0; r < 3; ++r) {
            for (int cc = 0; cc < 4; ++cc) {
                EXPECT_NEAR(T(r, cc), gt.data[static_cast<size_t>(r * 4 + cc)], 1e-3)
                    << c << " T-vs-gt[" << r << "," << cc << "]";
                // the reference estimate recorded in the fixture recovered the
                // same truth (this validates the fixture itself)
                ASSERT_NEAR(want_T.data[static_cast<size_t>(r * 4 + cc)],
                            gt.data[static_cast<size_t>(r * 4 + cc)], 1e-2)
                    << c << " fixture-vs-gt[" << r << "," << cc << "]";
            }
        }
        // inlier count within RANSAC noise
        EXPECT_LE(std::abs(static_cast<int>(got.inlier_idx_original.size()) -
                           static_cast<int>(want_inliers.numel())),
                  30)
            << c << " inlier count: " << got.inlier_idx_original.size() << " vs "
            << want_inliers.numel();
    }
}

TEST(AlignmentCore, PoseGraphSolve) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"a", "b"}) {
        const std::string p = std::string(kFixtureDir) + "/pg_" + c;
        const NpyArray init = read_npy(p + "_init.npy");
        const NpyArray n_arr = read_npy(p + "_n.npy");
        const NpyArray cons = read_npy(p + "_constraints.npy");
        const NpyArray n_cons = read_npy(p + "_n_constraints.npy");
        const NpyArray want = read_npy(p + "_out.npy");
        const int n = static_cast<int>(n_arr.data[0]);
        const int m = static_cast<int>(n_cons.data[0]);
        kernels::CameraPoses poses;
        for (int i = 0; i < n; ++i) {
            Eigen::Matrix4d T;
            for (int r = 0; r < 4; ++r) {
                for (int cc = 0; cc < 4; ++cc) {
                    T(r, cc) = init.data[static_cast<size_t>((i * 4 + r) * 4 + cc)];
                }
            }
            poses[i] = T;
        }
        std::vector<kernels::RelativePoseConstraint> constraints;
        int64_t idx = 0;
        for (int i = 0; i < m; ++i) {
            kernels::RelativePoseConstraint rc;
            rc.cam_idx_i = static_cast<int64_t>(cons.data[static_cast<size_t>(idx++)]);
            rc.cam_idx_j = static_cast<int64_t>(cons.data[static_cast<size_t>(idx++)]);
            for (int r = 0; r < 4; ++r) {
                for (int cc = 0; cc < 4; ++cc) {
                    rc.relative_pose_j_i(r, cc) = cons.data[static_cast<size_t>(idx++)];
                }
            }
            for (int k = 0; k < 3; ++k) rc.translation_weight[k] = cons.data[static_cast<size_t>(idx++)];
            for (int k = 0; k < 3; ++k) rc.rotation_weight[k] = cons.data[static_cast<size_t>(idx++)];
            constraints.push_back(rc);
        }
        kernels::ConstantPoseIndex constant;
        constant[0] = true;
        const kernels::CameraPoses got =
            kernels::pose_graph_solve(poses, constraints, constant, 100);
        for (int i = 0; i < n; ++i) {
            for (int r = 0; r < 4; ++r) {
                for (int cc = 0; cc < 4; ++cc) {
                    ASSERT_NEAR(got.at(i)(r, cc), want.data[static_cast<size_t>((i * 4 + r) * 4 + cc)],
                                1e-6)
                        << c << " solved[" << i << "](" << r << "," << cc << ")";
                }
            }
        }
    }
}

TEST(AlignmentCore, CapturePathPriors) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"a", "b"}) {
        const std::string p = std::string(kFixtureDir) + "/prior_" + c;
        const NpyArray ts = read_npy(p + "_poses_ts.npy");
        const NpyArray Ts = read_npy(p + "_poses_T.npy");
        const int n = static_cast<int>(ts.numel());
        std::map<int64_t, Eigen::Matrix4d> poses;
        for (int i = 0; i < n; ++i) {
            Eigen::Matrix4d T;
            for (int r = 0; r < 4; ++r) {
                for (int cc = 0; cc < 4; ++cc) {
                    T(r, cc) = Ts.data[static_cast<size_t>((i * 4 + r) * 4 + cc)];
                }
            }
            poses[static_cast<int64_t>(ts.data[static_cast<size_t>(i)])] = T;
        }
        const Eigen::MatrixXd got_speed = mapping::compute_path_speed(poses);
        const NpyArray want_speed = read_npy(p + "_speed.npy");
        ASSERT_EQ(got_speed.rows(), n);
        ASSERT_EQ(got_speed.cols(), 4);
        for (int r = 0; r < n; ++r) {
            for (int cc = 0; cc < 4; ++cc) {
                const double g = got_speed(r, cc);
                const double w = want_speed.data[static_cast<size_t>(r * 4 + cc)];
                if (std::isnan(w)) {
                    EXPECT_TRUE(std::isnan(g)) << c << " speed[" << r << "," << cc << "] should be NaN";
                } else {
                    EXPECT_NEAR(g, w, 1e-4) << c << " speed[" << r << "," << cc << "]";
                }
            }
        }
        const Eigen::MatrixXd got_climb = mapping::compute_path_climb(poses);
        const NpyArray want_climb = read_npy(p + "_climb.npy");
        for (int r = 0; r < n; ++r) {
            for (int cc = 0; cc < 4; ++cc) {
                EXPECT_NEAR(got_climb(r, cc), want_climb.data[static_cast<size_t>(r * 4 + cc)], 1e-6)
                    << c << " climb[" << r << "," << cc << "]";
            }
        }
    }
}

}  // namespace
}  // namespace tinynav
