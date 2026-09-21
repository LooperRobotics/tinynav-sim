// Python-alignment tests for the planning port. Each test replays a fixture
// case exported by tools/export_planning_fixtures.py from the UNMODIFIED
// reference (tinynav.core.planning_node / scipy) and compares the C++ port's
// output. Fixtures live in fixtures/ (gitignored); tests SKIP when they are
// absent so a bare checkout still runs the rest of the suite.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/kernels/raycast.hpp"
#include "tinynav_cpp/planning/planning.hpp"

namespace tinynav::planning {
namespace {

const char* kFixtureDir = "fixtures/planning";

struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<double> data;  // widened from f4/f8/u1

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
    if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("bad magic " + path);
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
    const bool is_i64 = header.find("i8") != std::string::npos;    // '<i8': 8-byte
    const bool is_i1 = header.find("i1") != std::string::npos;     // '|i1': 1-byte
    if (!is_f32 && !is_f64 && !is_u1 && !is_i64 && !is_i1) {
        throw std::runtime_error("unsupported dtype in " + path + ": " + header);
    }
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
    } else {  // i1
        std::vector<int8_t> raw(static_cast<size_t>(total));
        in.read(reinterpret_cast<char*>(raw.data()), total);
        for (int64_t i = 0; i < total; ++i) out.data[static_cast<size_t>(i)] = raw[static_cast<size_t>(i)];
    }
    if (!in) throw std::runtime_error("short read " + path);
    return out;
}

bool fixtures_available() {
    std::ifstream in(std::string(kFixtureDir) + "/raycast_a_out.npy", std::ios::binary);
    return in.good();
}

::testing::AssertionResult expect_close(const NpyArray& got, const NpyArray& want,
                                        double tol, const char* what) {
    if (got.numel() != want.numel()) {
        return ::testing::AssertionFailure()
               << what << ": element count " << got.numel() << " != " << want.numel();
    }
    double worst = 0.0;
    int64_t worst_i = -1;
    for (int64_t i = 0; i < want.numel(); ++i) {
        const double g = got.data[static_cast<size_t>(i)];
        const double w = want.data[static_cast<size_t>(i)];
        if (std::isinf(w)) {
            if (!std::isinf(g)) {
                return ::testing::AssertionFailure()
                       << what << ": [" << i << "] got " << g << " want inf";
            }
            continue;
        }
        const double diff = std::abs(g - w);
        if (diff > worst) {
            worst = diff;
            worst_i = i;
        }
    }
    if (worst > tol) {
        return ::testing::AssertionFailure()
               << what << ": worst diff " << worst << " at [" << worst_i << "] > " << tol;
    }
    return ::testing::AssertionSuccess();
}

// f32-valued fixture (widened to double on read) -> (rows, cols) float array.
Eigen::ArrayXXf npy_to_f32(const NpyArray& a) {
    Eigen::ArrayXXf out(static_cast<int>(a.shape[0]), static_cast<int>(a.shape[1]));
    for (int r = 0; r < out.rows(); ++r) {
        for (int c = 0; c < out.cols(); ++c) {
            out(r, c) = static_cast<float>(a.data[static_cast<size_t>(r * out.cols() + c)]);
        }
    }
    return out;
}

// Rasterize a route the way build_route_fields does (verified separately via
// the exact path_dist match) and expose the per-cell (later-wins) arc/heading,
// so the remaining/heading comparison can be tie-aware: cells equidistant
// between route cells may legitimately gather either neighbour's fields
// (scipy's EDT and our Felzenszwalb EDT tie-break differently — edt.hpp).
struct RouteRaster {
    std::vector<std::tuple<int, int, double, double>> cells;  // (r, c, arc, heading)
    double total_arc = 0.0;
};
RouteRaster rasterize_route(const Eigen::MatrixX2d& route, double origin_x,
                            double origin_y, double resolution) {
    RouteRaster out;
    const int n = static_cast<int>(route.rows());
    if (n < 2) return out;
    std::vector<double> node_arc(n, 0.0);
    for (int i = 1; i < n; ++i) {
        node_arc[static_cast<size_t>(i)] =
            node_arc[static_cast<size_t>(i - 1)] + (route.row(i) - route.row(i - 1)).norm();
    }
    const double arc = node_arc[static_cast<size_t>(n - 1)];
    if (arc < 1e-9) return out;
    out.total_arc = arc;
    const int n_samples = static_cast<int>(std::ceil(arc / (0.5 * resolution))) + 1;
    std::vector<double> sample_arc(static_cast<size_t>(n_samples));
    const double step = arc / double(n_samples - 1);
    for (int k = 0; k < n_samples; ++k) sample_arc[static_cast<size_t>(k)] = double(k) * step;
    sample_arc[static_cast<size_t>(n_samples - 1)] = arc;
    for (int k = 0; k < n_samples; ++k) {
        const double sa = sample_arc[static_cast<size_t>(k)];
        int j = 0;
        while (j < n && node_arc[static_cast<size_t>(j)] <= sa) ++j;
        j = std::clamp(j - 1, 0, n - 2);
        // np.interp / interp_scalar form (slope), NOT t*(delta) — the last-ulp
        // difference moves cells across int() boundaries.
        const double denom = node_arc[static_cast<size_t>(j + 1)] - node_arc[static_cast<size_t>(j)];
        const double dx = denom > 0 ? (route(j + 1, 0) - route(j, 0)) / denom : 0.0;
        const double dy = denom > 0 ? (route(j + 1, 1) - route(j, 1)) / denom : 0.0;
        const double sx = route(j, 0) + (sa - node_arc[static_cast<size_t>(j)]) * dx;
        const double sy = route(j, 1) + (sa - node_arc[static_cast<size_t>(j)]) * dy;
        const int r = static_cast<int>((sx - origin_x) / resolution);
        const int c = static_cast<int>((sy - origin_y) / resolution);
        const double tang = std::atan2(route(j + 1, 1) - route(j, 1),
                                       route(j + 1, 0) - route(j, 0));
        // the real rasterization only keeps in-grid samples
        if (r >= 0 && r < 80 && c >= 0 && c < 70) {
            out.cells.emplace_back(r, c, sa, tang);
        }
    }
    return out;
}

TrajectorySet npy_to_trajset(const NpyArray& trajs) {
    TrajectorySet set;
    set.num_trajectories = static_cast<int>(trajs.shape[0]);
    set.num_steps = static_cast<int>(trajs.shape[1]);
    set.poses.resize(set.num_trajectories * set.num_steps, 7);
    set.params.setZero(set.num_trajectories, 2);
    for (int64_t i = 0; i < trajs.numel(); ++i) {
        set.poses.data()[static_cast<size_t>(i)] = trajs.data[static_cast<size_t>(i)];
    }
    return set;
}

Eigen::MatrixX2d npy_to_matx2d(const NpyArray& a) {
    Eigen::MatrixX2d out(static_cast<int>(a.shape[0]), 2);
    for (int r = 0; r < out.rows(); ++r) {
        out(r, 0) = a.data[static_cast<size_t>(r * 2)];
        out(r, 1) = a.data[static_cast<size_t>(r * 2 + 1)];
    }
    return out;
}

Eigen::Matrix4d mat4(const NpyArray& a) {
    Eigen::Matrix4d m;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            m(r, c) = a.data[static_cast<size_t>(r * 4 + c)];
        }
    }
    return m;
}

TEST(Alignment, SkipWhenFixturesAbsent) {
    if (!fixtures_available()) {
        GTEST_SKIP() << "fixtures/planning not exported; run tools/export_planning_fixtures.py";
    }
}

TEST(AlignmentPlanning, RunRaycastingLoopy) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"a", "b", "c"}) {
        const std::string p = std::string(kFixtureDir) + "/raycast_" + c;
        const NpyArray depth = read_npy(p + "_depth.npy");
        const Eigen::Matrix4d T = mat4(read_npy(p + "_T.npy"));
        const NpyArray origin = read_npy(p + "_origin.npy");
        const NpyArray cfg = read_npy(p + "_cfg.npy");
        const NpyArray want = read_npy(p + "_out.npy");
        Eigen::MatrixXf depth_f(static_cast<int>(depth.shape[0]),
                                static_cast<int>(depth.shape[1]));
        for (int64_t i = 0; i < depth.numel(); ++i) {
            depth_f(static_cast<int64_t>(i / depth.shape[1]),
                    static_cast<int64_t>(i % depth.shape[1])) =
                static_cast<float>(depth.data[static_cast<size_t>(i)]);
        }
        const OccupancyGrid3D got = run_raycasting_loopy(
            depth_f, T, Eigen::Vector3i(30, 30, 10), cfg.data[0], cfg.data[1],
            cfg.data[2], cfg.data[3], Eigen::Vector3d(origin.data.data()), static_cast<int>(cfg.data[4]),
            cfg.data[5]);
        NpyArray got_arr;
        got_arr.shape = {30, 30, 10};
        got_arr.data.resize(got_arr.data.size());
        got_arr.data.assign(static_cast<size_t>(30 * 30 * 10), 0.0);
        for (int x = 0; x < 30; ++x) {
            for (int y = 0; y < 30; ++y) {
                for (int z = 0; z < 10; ++z) {
                    got_arr.data[static_cast<size_t>((x * 30 + y) * 10 + z)] = got(x, y, z);
                }
            }
        }
        EXPECT_TRUE(expect_close(got_arr, want, 1e-9, (std::string("raycast_") + c).c_str()));
    }
}

TEST(AlignmentPlanning, GenerateTrajectoryLibrary3d) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"default", "vxmin", "capped"}) {
        const std::string p = std::string(kFixtureDir) + "/traj_" + c;
        const NpyArray want_trajs = read_npy(p + "_out.npy");
        const NpyArray want_params = read_npy(p + "_params.npy");
        TrajectorySet got;
        if (std::string(c) == "default") {
            got = generate_trajectory_library_3d();
        } else if (std::string(c) == "vxmin") {
            got = generate_trajectory_library_3d(15, 3.0, 0.1, Eigen::Vector3d::Zero(),
                                                 Eigen::Vector4d(0, 0, 0, 1), 0.6, 0.75, 2.5, 0.5, 0.2);
        } else {
            got = generate_trajectory_library_3d(15, 3.0, 0.1, Eigen::Vector3d::Zero(),
                                                 Eigen::Vector4d(0, 0, 0, 1), 0.5, M_PI / 3.0, 0.2, 0.05, 0.0);
        }
        NpyArray got_arr;
        got_arr.shape = want_trajs.shape;
        got_arr.data.assign(got.poses.data(), got.poses.data() + got.poses.size());
        EXPECT_TRUE(expect_close(got_arr, want_trajs, 1e-9, (std::string("traj_") + c + "_trajs").c_str()));
        NpyArray got_params;
        got_params.shape = want_params.shape;  // MatrixX2d is col-major; copy row-major
        got_params.data.resize(static_cast<size_t>(got.params.size()));
        for (int r = 0; r < got.params.rows(); ++r) {
            got_params.data[static_cast<size_t>(r * 2)] = got.params(r, 0);
            got_params.data[static_cast<size_t>(r * 2 + 1)] = got.params(r, 1);
        }
        EXPECT_TRUE(expect_close(got_params, want_params, 1e-9, (std::string("traj_") + c + "_params").c_str()));
    }
    // vocabulary
    const TrajectorySet got = generate_predefined_trajectory_vocabularies();
    NpyArray got_arr;
    got_arr.shape = {1, got.num_steps, 7};
    got_arr.data.assign(got.poses.data(), got.poses.data() + got.poses.size());
    EXPECT_TRUE(expect_close(got_arr, read_npy(std::string(kFixtureDir) + "/vocab_out.npy"), 1e-9, "vocab"));
}

TEST(AlignmentPlanning, BuildObstacleMap) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"strict", "dilate", "relax"}) {
        const std::string p = std::string(kFixtureDir) + "/obstacle_" + c;
        const NpyArray grid = read_npy(p + "_grid.npy");
        const NpyArray origin = read_npy(p + "_origin.npy");
        const double robot_z = read_npy(p + "_robot_z.npy").data[0];
        const NpyArray want = read_npy(p + "_out.npy");
        OccupancyGrid3D occ(40, 40, 12);
        for (int x = 0; x < 40; ++x) {
            for (int y = 0; y < 40; ++y) {
                for (int z = 0; z < 12; ++z) {
                    occ(x, y, z) = grid.data[static_cast<size_t>((x * 40 + y) * 12 + z)];
                }
            }
        }
        Eigen::ArrayXXf min_span;
        const bool has_min_span = c == std::string("relax");
        if (has_min_span) {
            min_span = npy_to_f32(read_npy(p + "_minspan.npy"));
        }
        core::ObstacleConfig config;  // defaults == the exporter's ObstacleConfig()
        if (c == std::string("dilate")) config.dilation_cells = 2;
        const Mask2D got = build_obstacle_map(
            occ, Eigen::Vector3d(origin.data.data()), 0.1, robot_z, config,
            has_min_span ? &min_span : nullptr);
        NpyArray got_arr;
        got_arr.shape = {40, 40};
        got_arr.data.resize(1600);
        for (int r = 0; r < 40; ++r) {
            for (int cc = 0; cc < 40; ++cc) {
                got_arr.data[static_cast<size_t>(r * 40 + cc)] = got(r, cc) ? 1.0 : 0.0;
            }
        }
        EXPECT_TRUE(expect_close(got_arr, want, 0.0, (std::string("obstacle_") + c).c_str()));
    }
}

TEST(AlignmentPlanning, RollOccupancyGrid) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"x", "xyz", "back", "identity"}) {
        const std::string p = std::string(kFixtureDir) + "/roll_" + c;
        const NpyArray grid = read_npy(p + "_grid.npy");
        const NpyArray old_o = read_npy(p + "_old.npy");
        const NpyArray new_o = read_npy(p + "_new.npy");
        const NpyArray want = read_npy(p + "_out.npy");
        const NpyArray want_origin = read_npy(p + "_out_origin.npy");
        OccupancyGrid3D occ(20, 24, 6);
        for (int x = 0; x < 20; ++x) {
            for (int y = 0; y < 24; ++y) {
                for (int z = 0; z < 6; ++z) {
                    occ(x, y, z) = grid.data[static_cast<size_t>((x * 24 + y) * 6 + z)];
                }
            }
        }
        const RolledGrid got = roll_occupancy_grid(
            occ, Eigen::Vector3d(old_o.data.data()), Eigen::Vector3d(new_o.data.data()), 0.1);
        NpyArray got_arr;
        got_arr.shape = {20, 24, 6};
        got_arr.data.resize(20 * 24 * 6);
        for (int x = 0; x < 20; ++x) {
            for (int y = 0; y < 24; ++y) {
                for (int z = 0; z < 6; ++z) {
                    got_arr.data[static_cast<size_t>((x * 24 + y) * 6 + z)] = got.grid(x, y, z);
                }
            }
        }
        EXPECT_TRUE(expect_close(got_arr, want, 1e-12, (std::string("roll_") + c).c_str()));
        NpyArray got_origin;
        got_origin.shape = {3};
        got_origin.data.assign(got.origin.data(), got.origin.data() + 3);
        EXPECT_TRUE(expect_close(got_origin, want_origin, 1e-12, (std::string("roll_origin_") + c).c_str()));
    }
}

TEST(AlignmentPlanning, FootprintLattice) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* name : {"go2", "b2", "g1"}) {
        const std::string p = std::string(kFixtureDir) + "/lattice_" + name;
        const NpyArray cfg = read_npy(p + "_cfg.npy");
        const FootprintLattice got = footprint_lattice(cfg.data[0], cfg.data[1], cfg.data[2], cfg.data[3]);
        NpyArray got_fwd;
        got_fwd.shape = {static_cast<int64_t>(got.fwd.size())};
        got_fwd.data = got.fwd;
        EXPECT_TRUE(expect_close(got_fwd, read_npy(p + "_fwd.npy"), 1e-12, (std::string("lattice_") + name).c_str()));
        NpyArray got_lat;
        got_lat.shape = {static_cast<int64_t>(got.lat.size())};
        got_lat.data = got.lat;
        EXPECT_TRUE(expect_close(got_lat, read_npy(p + "_lat.npy"), 1e-12, (std::string("lattice_lat_") + name).c_str()));
    }
}

TEST(AlignmentPlanning, ScoreTrajectoriesByEsdf) {
    if (!fixtures_available()) GTEST_SKIP();
    const std::string p = std::string(kFixtureDir) + "/score_";
    const NpyArray trajs = read_npy(p + "trajs.npy");
    const Eigen::ArrayXXf esdf_map = npy_to_f32(read_npy(p + "esdf.npy"));
    const Eigen::ArrayXXf path_dist = npy_to_f32(read_npy(p + "path_dist.npy"));
    const Eigen::ArrayXXf remaining = npy_to_f32(read_npy(p + "remaining.npy"));
    const Eigen::ArrayXXf heading = npy_to_f32(read_npy(p + "heading.npy"));
    const NpyArray origin = read_npy(p + "origin.npy");
    const NpyArray cfg = read_npy(p + "cfg.npy");
    const TrajectorySet set = npy_to_trajset(trajs);
    const int T = set.num_trajectories;
    const EsdfScoreResult got = score_trajectories_by_ESDF(
        set, esdf_map, path_dist, remaining, heading, Eigen::Vector3d(origin.data.data()),
        cfg.data[4], cfg.data[0], cfg.data[1], cfg.data[2], cfg.data[3]);
    NpyArray got_scores;
    got_scores.shape = {T};
    got_scores.data = got.scores;
    EXPECT_TRUE(expect_close(got_scores, read_npy(p + "out_scores.npy"), 1e-9, "scores"));
    NpyArray got_occ;
    got_occ.shape = {T};
    got_occ.data.assign(got.occ_points.begin(), got.occ_points.end());
    EXPECT_TRUE(expect_close(got_occ, read_npy(p + "out_occ.npy"), 0.0, "occ_points"));
    NpyArray got_pc;
    got_pc.shape = {T};
    got_pc.data = got.path_costs;
    EXPECT_TRUE(expect_close(got_pc, read_npy(p + "out_path_costs.npy"), 1e-6, "path_costs"));
    NpyArray got_er;
    got_er.shape = {T};
    got_er.data = got.end_remainings;
    EXPECT_TRUE(expect_close(got_er, read_npy(p + "out_end_remainings.npy"), 1e-6, "end_remainings"));
    NpyArray got_eh;
    got_eh.shape = {T};
    got_eh.data = got.end_heading_errs;
    EXPECT_TRUE(expect_close(got_eh, read_npy(p + "out_end_heading_errs.npy"), 1e-6, "end_heading_errs"));
}

TEST(AlignmentPlanning, BuildRouteFields) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"straight", "elbow", "zigzag", "offgrid", "degenerate"}) {
        const std::string p = std::string(kFixtureDir) + "/route_" + c;
        const NpyArray xy = read_npy(p + "_xy.npy");
        Eigen::MatrixX2d route(static_cast<int>(xy.shape[0]), 2);
        for (int i = 0; i < route.rows(); ++i) {
            route(i, 0) = xy.data[static_cast<size_t>(i * 2)];
            route(i, 1) = xy.data[static_cast<size_t>(i * 2 + 1)];
        }
        const RouteFields got = build_route_fields(route, Eigen::Vector2i(80, 70),
                                                   Eigen::Vector3d(-2.0, -2.0, 0.0), 0.05);
        const int64_t has = read_npy(p + "_has.npy").data[0];
        EXPECT_EQ(got.has_route, has != 0) << c;
        if (!got.has_route) continue;
        // path_dist is exact: distances are unique regardless of tie-breaking.
        {
            const NpyArray want = read_npy(p + "_path_dist.npy");
            NpyArray got_arr;
            got_arr.shape = want.shape;
            got_arr.data.resize(want.numel());
            for (int r = 0; r < 80; ++r) {
                for (int cc = 0; cc < 70; ++cc) {
                    got_arr.data[static_cast<size_t>(r * 70 + cc)] = got.path_dist_map(r, cc);
                }
            }
            EXPECT_TRUE(expect_close(got_arr, want, 2e-5, (std::string("route_") + c + "_path_dist").c_str()));
        }
        // remaining/heading: tie-aware. For each cell both implementations must
        // produce a value gathered from SOME nearest route cell.
        const RouteRaster raster = rasterize_route(route, -2.0, -2.0, 0.05);
        ASSERT_FALSE(raster.cells.empty());
        const NpyArray want_rem = read_npy(p + "_remaining.npy");
        const NpyArray want_head = read_npy(p + "_heading.npy");
        for (int r = 0; r < 80; ++r) {
            for (int cc = 0; cc < 70; ++cc) {
                double best = 1e18;
                for (const auto& [br, bc, arc, tang] : raster.cells) {
                    const double d = std::sqrt(double((r - br) * (r - br) + (cc - bc) * (cc - bc)));
                    if (d < best) best = d;
                }
                std::vector<double> rem_set, head_set;
                for (const auto& [br, bc, arc, tang] : raster.cells) {
                    const double d = std::sqrt(double((r - br) * (r - br) + (cc - bc) * (cc - bc)));
                    if (std::abs(d - best) <= 1e-12) {
                        rem_set.push_back(raster.total_arc - arc);
                        head_set.push_back(tang);
                    }
                }
                auto in_set = [](double v, const std::vector<double>& set) {
                    for (double s : set) {
                        if (std::abs(v - s) <= 2e-5) return true;
                    }
                    return false;
                };
                EXPECT_TRUE(in_set(got.remaining_map(r, cc), rem_set))
                    << c << " remaining at (" << r << "," << cc << "): " << got.remaining_map(r, cc);
                EXPECT_TRUE(in_set(want_rem.data[static_cast<size_t>(r * 70 + cc)], rem_set))
                    << c << " scipy remaining at (" << r << "," << cc << ")";
                EXPECT_TRUE(in_set(got.route_heading_map(r, cc), head_set))
                    << c << " heading at (" << r << "," << cc << ")";
                EXPECT_TRUE(in_set(want_head.data[static_cast<size_t>(r * 70 + cc)], head_set))
                    << c << " scipy heading at (" << r << "," << cc << ")";
            }
        }
    }
}

TEST(AlignmentPlanning, ScipyEdtDistances) {
    if (!fixtures_available()) GTEST_SKIP();
    // Distances must match scipy exactly (indices are only tie-checked).
    {
        const NpyArray mask = read_npy(std::string(kFixtureDir) + "/edt_random_mask.npy");
        Mask2D input = Mask2D::Constant(static_cast<int>(mask.shape[0]),
                                        static_cast<int>(mask.shape[1]), false);
        for (int64_t i = 0; i < mask.numel(); ++i) {
            // fixture mask is True at features; scipy computed EDT of ~mask, so
            // our input (whose FALSE cells are the features) is the inverted mask
            input(static_cast<int64_t>(i / mask.shape[1]),
                  static_cast<int64_t>(i % mask.shape[1])) = mask.data[static_cast<size_t>(i)] == 0;
        }
        const EdtResult got = distance_transform_edt(input);
        NpyArray got_arr;
        got_arr.shape = mask.shape;  // ArrayXXd is col-major; copy row-major
        got_arr.data.resize(static_cast<size_t>(input.rows() * input.cols()));
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.cols(); ++c) {
                got_arr.data[static_cast<size_t>(r * input.cols() + c)] = got.distances(r, c);
            }
        }
        EXPECT_TRUE(expect_close(got_arr, read_npy(std::string(kFixtureDir) + "/edt_random_dist.npy"),
                                 1e-9, "edt_random_dist"));
        // tie-aware index check: the reported nearest feature is A nearest feature
        const NpyArray inds = read_npy(std::string(kFixtureDir) + "/edt_random_inds.npy");
        for (int r = 0; r < input.rows(); ++r) {
            for (int c = 0; c < input.cols(); ++c) {
                const int nr = got.nearest_row(r, c);
                const int nc = got.nearest_col(r, c);
                ASSERT_FALSE(input(nr, nc)) << "nearest cell is not a feature at " << r << "," << c;
                const double d = std::sqrt(double((r - nr) * (r - nr) + (c - nc) * (c - nc)));
                EXPECT_NEAR(d, got.distances(r, c), 1e-9);
                (void)inds;
            }
        }
    }
    // symmetric fixture: indices may differ from scipy (tie), distances must not
    {
        const NpyArray mask = read_npy(std::string(kFixtureDir) + "/edt_tie_mask.npy");
        Mask2D input = Mask2D::Constant(9, 9, false);
        for (int64_t i = 0; i < mask.numel(); ++i) {
            input(static_cast<int64_t>(i / 9), static_cast<int64_t>(i % 9)) =
                mask.data[static_cast<size_t>(i)] == 0;
        }
        const EdtResult got = distance_transform_edt(input);
        NpyArray got_arr;
        got_arr.shape = {9, 9};
        got_arr.data.resize(81);
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                got_arr.data[static_cast<size_t>(r * 9 + c)] = got.distances(r, c);
            }
        }
        EXPECT_TRUE(expect_close(got_arr, read_npy(std::string(kFixtureDir) + "/edt_tie_dist.npy"),
                                 1e-9, "edt_tie_dist"));
    }
}

TEST(AlignmentPlanning, ScalarHelpers) {
    if (!fixtures_available()) GTEST_SKIP();
    const NpyArray cases = read_npy(std::string(kFixtureDir) + "/reverse_cases.npy");
    const NpyArray want = read_npy(std::string(kFixtureDir) + "/reverse_out.npy");
    for (int i = 0; i < cases.shape[0]; ++i) {
        const bool got = reverse_armed(cases.data[static_cast<size_t>(i * 3)],
                                       static_cast<int>(cases.data[static_cast<size_t>(i * 3 + 1)]),
                                       cases.data[static_cast<size_t>(i * 3 + 2)]);
        EXPECT_EQ(got, want.data[static_cast<size_t>(i)] != 0) << "case " << i;
    }
    const NpyArray scases = read_npy(std::string(kFixtureDir) + "/speed_cases.npy");
    const NpyArray swant = read_npy(std::string(kFixtureDir) + "/speed_out.npy");
    for (int i = 0; i < scases.shape[0]; ++i) {
        const double got = speed_from_clearance(
            scases.data[static_cast<size_t>(i * 3)], scases.data[static_cast<size_t>(i * 3 + 1)],
            scases.data[static_cast<size_t>(i * 3 + 2)], 0.2, 0.35, 1.0, 0.2);
        EXPECT_NEAR(got, swant.data[static_cast<size_t>(i)], 1e-12) << "case " << i;
    }
}

TEST(AlignmentPlanning, DwaCostAndSelection) {
    if (!fixtures_available()) GTEST_SKIP();
    for (const char* c : {"route", "noroute", "reverse"}) {
        const std::string p = std::string(kFixtureDir) + "/cost_" + c;
        const NpyArray trajs = read_npy(p + "_trajs.npy");
        const NpyArray params = read_npy(p + "_params.npy");
        const NpyArray scores = read_npy(p + "_scores.npy");
        const NpyArray pc = read_npy(p + "_path_costs.npy");
        const NpyArray er = read_npy(p + "_end_remainings.npy");
        const NpyArray eh = read_npy(p + "_end_heading_errs.npy");
        const NpyArray cfg = read_npy(p + "_cfg.npy");
        TrajectorySet set = npy_to_trajset(trajs);
        const int T = set.num_trajectories;
        set.params = npy_to_matx2d(params);
        EsdfScoreResult scored;
        scored.scores.assign(scores.data.begin(), scores.data.end());
        scored.path_costs.assign(pc.data.begin(), pc.data.end());
        scored.end_remainings.assign(er.data.begin(), er.data.end());
        scored.end_heading_errs.assign(eh.data.begin(), eh.data.end());
        scored.occ_points.assign(T, -1);
        const bool has_route = cfg.data[0] != 0;
        const Eigen::Vector3d target(cfg.data[1], cfg.data[2], cfg.data[3]);
        const Eigen::Vector2d last_param(cfg.data[4], cfg.data[5]);
        const bool should_reverse = cfg.data[6] != 0;
        const DwaWeights weights;
        NpyArray got_costs;
        got_costs.shape = {T};
        got_costs.data.resize(static_cast<size_t>(T));
        int best = -1;
        double best_cost = std::numeric_limits<double>::infinity();
        for (int i = 0; i < T; ++i) {
            const double cost = trajectory_cost(i, set, set.params, scored, has_route,
                                                std::optional<Eigen::Vector3d>(target),
                                                last_param, should_reverse, weights);
            got_costs.data[static_cast<size_t>(i)] = cost;
            if (cost < best_cost) {
                best_cost = cost;
                best = i;
            }
        }
        EXPECT_TRUE(expect_close(got_costs, read_npy(p + "_out.npy"), 1e-6, (std::string("cost_") + c).c_str()));
        const int want_argmin = static_cast<int>(read_npy(p + "_argmin.npy").data[0]);
        EXPECT_EQ(best, want_argmin) << c;
    }
}

}  // namespace
}  // namespace tinynav::planning
