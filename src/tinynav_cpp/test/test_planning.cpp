// Tests for the planning port: golden, hand-computed assertions (the Python
// numeric-alignment pass is a separate, fixture-driven stage) plus the scipy
// replacements in edt.cpp.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/kernels/raycast.hpp"
#include "tinynav_cpp/planning/edt.hpp"
#include "tinynav_cpp/planning/planning.hpp"

namespace tinynav::planning {
namespace {

TEST(EdtTest, SingleFeatureExactDistancesAndIndices) {
    Mask2D input = Mask2D::Constant(4, 5, true);
    input(1, 1) = false;  // the single feature cell
    const EdtResult edt = distance_transform_edt(input);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 5; ++c) {
            const double expect = std::sqrt(double((r - 1) * (r - 1) + (c - 1) * (c - 1)));
            EXPECT_NEAR(edt.distances(r, c), expect, 1e-12) << r << "," << c;
            EXPECT_EQ(edt.nearest_row(r, c), 1);
            EXPECT_EQ(edt.nearest_col(r, c), 1);
        }
    }
    EXPECT_EQ(edt.distances(1, 1), 0.0);
}

TEST(EdtTest, TwoFeaturesNearestWins) {
    Mask2D input = Mask2D::Constant(1, 9, true);
    input(0, 1) = false;
    input(0, 6) = false;
    const EdtResult edt = distance_transform_edt(input);
    EXPECT_NEAR(edt.distances(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(edt.distances(0, 3), 2.0, 1e-12);  // nearer to col 1 than col 6
    EXPECT_NEAR(edt.distances(0, 4), 2.0, 1e-12);  // nearer to col 6
    EXPECT_NEAR(edt.distances(0, 8), 2.0, 1e-12);
    EXPECT_EQ(edt.nearest_col(0, 0), 1);
    EXPECT_EQ(edt.nearest_col(0, 3), 1);
    EXPECT_EQ(edt.nearest_col(0, 4), 6);
}

TEST(EdtTest, FeaturelessInputReadsAllZero) {
    // scipy quirk planning_node.py documents via free_space_esdf.
    const Mask2D input = Mask2D::Constant(3, 3, true);
    const EdtResult edt = distance_transform_edt(input);
    EXPECT_TRUE(edt.distances.isZero(0.0));
}

TEST(EdtTest, MatchesBruteForce) {
    // Random-ish pattern; exact EDT must equal the brute-force minimum.
    Mask2D input = Mask2D::Constant(9, 11, true);
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 11; ++c) {
            if ((r * 7 + c * 13) % 5 == 0) input(r, c) = false;
        }
    }
    const EdtResult edt = distance_transform_edt(input);
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 11; ++c) {
            double best = 1e18;
            for (int rr = 0; rr < 9; ++rr) {
                for (int cc = 0; cc < 11; ++cc) {
                    if (!input(rr, cc)) {
                        best = std::min(best, std::sqrt(double((r - rr) * (r - rr) + (c - cc) * (c - cc))));
                    }
                }
            }
            EXPECT_NEAR(edt.distances(r, c), best, 1e-12) << r << "," << c;
            // the reported nearest feature really is one
            EXPECT_FALSE(input(edt.nearest_row(r, c), edt.nearest_col(r, c)));
            const double dr = double(r - edt.nearest_row(r, c));
            const double dc = double(c - edt.nearest_col(r, c));
            EXPECT_NEAR(std::sqrt(dr * dr + dc * dc), edt.distances(r, c), 1e-12);
        }
    }
}

TEST(DilationTest, CrossIterationsFormManhattanBall) {
    Mask2D mask = Mask2D::Constant(5, 5, false);
    mask(2, 2) = true;
    const Mask2D one = binary_dilation(mask, 1);
    EXPECT_TRUE(one(2, 2));
    EXPECT_TRUE(one(1, 2) && one(3, 2) && one(2, 1) && one(2, 3));
    EXPECT_FALSE(one(1, 1));  // corners untouched by the cross
    const Mask2D two = binary_dilation(mask, 2);
    EXPECT_TRUE(two(0, 2) && two(2, 0) && two(2, 4) && two(4, 2));
    EXPECT_TRUE(two(1, 1));  // Manhattan distance 2
    EXPECT_FALSE(two(0, 0));
}

TEST(MaxFilterTest, SquareWindowChebyshevGrowth) {
    Mask2D seed = Mask2D::Constant(7, 7, false);
    seed(3, 3) = true;
    const Mask2D grown = maximum_filter(seed, 5);  // Chebyshev radius 2
    EXPECT_TRUE(grown(1, 1) && grown(5, 5) && grown(3, 1));
    EXPECT_FALSE(grown(0, 1));
    EXPECT_FALSE(grown(6, 6));
}

TEST(FootprintLatticeTest, CentreFirstAndPitchCoversFootprint) {
    const core::RobotConfig& go2 = core::go2_config();
    const auto [fl, rl, hw] = go2.footprint_from_control();
    const FootprintLattice lattice = footprint_lattice(fl, rl, hw, go2.safety_radius);
    ASSERT_EQ(lattice.fwd.front(), 0.0);
    ASSERT_EQ(lattice.lat.front(), 0.0);
    // go2: pitch = 0.1*sqrt2; n_long = ceil(0.6/pitch)+1 = 6, n_lat = ceil(0.3/pitch)+1 = 4
    EXPECT_EQ(lattice.fwd.size(), 25u);
    // extremes reached exactly at the lattice boundary
    EXPECT_NEAR(*std::min_element(lattice.fwd.begin(), lattice.fwd.end()), -rl, 1e-12);
    EXPECT_NEAR(*std::max_element(lattice.fwd.begin(), lattice.fwd.end()), fl, 1e-12);
    EXPECT_NEAR(*std::min_element(lattice.lat.begin(), lattice.lat.end()), -hw, 1e-12);
    EXPECT_NEAR(*std::max_element(lattice.lat.begin(), lattice.lat.end()), hw, 1e-12);
}

TEST(TrajectoryLibraryTest, ShapesParamsAndForwardMotion) {
    // Defaults: 15 samples, 3 s, 0.1 s -> 31 steps; vx grid 7 x omega 15 = 105.
    const TrajectorySet lib = generate_trajectory_library_3d();
    EXPECT_EQ(lib.num_steps, 31);
    EXPECT_EQ(lib.num_trajectories, 105);
    EXPECT_EQ(lib.poses.rows(), 105 * 31);
    EXPECT_EQ(lib.params.rows(), 105);
    // First block is the vx=0 (turn-in-place) family, omega linspace(-pi/3, pi/3, 15).
    EXPECT_NEAR(lib.params(0, 0), 0.0, 1e-15);
    EXPECT_NEAR(lib.params(0, 1), -M_PI / 3.0, 1e-12);
    EXPECT_NEAR(lib.params(14, 1), M_PI / 3.0, 1e-12);
    // vx grid with the default min_linear_vel=0: [0] + linspace(0, 0.5, 6)
    // = [0, 0, 0.1, 0.2, 0.3, 0.4, 0.5] — the second family is ALSO vx=0.
    EXPECT_NEAR(lib.params(15, 0), 0.0, 1e-12);
    EXPECT_NEAR(lib.params(30, 0), 0.1, 1e-12);
    // Straight-forward row: identity quat, body +Z forward, so the internal
    // motion is +z 0.05/step — which the z-flatten hack then pins to step 0's z.
    int fast = -1;
    for (int t = 0; t < lib.num_trajectories; ++t) {
        if (std::abs(lib.params(t, 0) - 0.5) < 1e-12 && std::abs(lib.params(t, 1)) < 1e-12) {
            fast = t;
        }
    }
    ASSERT_GE(fast, 0);
    for (int i = 0; i < lib.num_steps; ++i) {
        const auto pose = lib.pose(fast, i);
        EXPECT_NEAR(pose[2], 0.05, 1e-12) << "step " << i;
        EXPECT_NEAR(pose[0], 0.0, 1e-12);
        EXPECT_NEAR(pose[1], 0.0, 1e-12);
    }
    // Turn-in-place row (vx=0, omega=-pi/3): rotation accumulates the full
    // duration — total angle = omega * dt * num_steps about body +Y.
    const double omega = lib.params(0, 1);
    const Eigen::Vector4d q_end = lib.pose(0, lib.num_steps - 1).tail<4>();
    const Eigen::Matrix3d R_end = core::quat_to_matrix(q_end);
    const double angle = std::acos(std::clamp((R_end.trace() - 1.0) / 2.0, -1.0, 1.0));
    const double total = std::abs(omega) * 0.1 * lib.num_steps;
    EXPECT_NEAR(angle, std::min(total, 2.0 * M_PI - total), 1e-9);  // trace folds > pi
    // z-flatten hack: every pose carries the z of step 0.
    for (int t = 0; t < lib.num_trajectories; ++t) {
        const double z0 = lib.pose(t, 0)[2];
        for (int i = 0; i < lib.num_steps; ++i) {
            EXPECT_EQ(lib.pose(t, i)[2], z0);
        }
    }
}

TEST(TrajectoryLibraryTest, PathLenCapFreezesArcNotSpeed) {
    // max_path_len_m = 0.2 with vx = 0.5: the pose advances 0.05 of arc/step
    // until the accumulated arc reaches the cap, then freezes for the remaining
    // steps while vx (the param) is untouched. The z-flatten hides straight-line
    // motion, so observe a TURNING row's xy arc (omega = pi/3, the max row).
    const TrajectorySet lib = generate_trajectory_library_3d(
        15, 3.0, 0.1, Eigen::Vector3d::Zero(), Eigen::Vector4d(0, 0, 0, 1),
        0.5, M_PI / 3.0, 0.2, 1e9, 0.0);
    int turn = -1;
    for (int t = 0; t < lib.num_trajectories; ++t) {
        if (std::abs(lib.params(t, 0) - 0.5) < 1e-12 &&
            std::abs(lib.params(t, 1) - M_PI / 3.0) < 1e-9) {
            turn = t;
        }
    }
    ASSERT_GE(turn, 0);
    // The turn about body +Y sweeps the forward axis through the x-z plane; the
    // z-flatten hides z, y never moves, so the arc is visible in x only.
    auto x_at = [&](int i) { return lib.pose(turn, i)[0]; };
    // Frozen tail (the exact freeze step sits on a double boundary Python and
    // C++ share, so shape assertions are the honest golden), and the capped
    // rollout stays near the start while the uncapped one sweeps ~1 m out.
    EXPECT_NEAR(x_at(lib.num_steps - 1), x_at(lib.num_steps - 2), 1e-12);
    EXPECT_LT(std::abs(x_at(lib.num_steps - 1)), 0.1);
    const TrajectorySet uncapped = generate_trajectory_library_3d(
        15, 3.0, 0.1, Eigen::Vector3d::Zero(), Eigen::Vector4d(0, 0, 0, 1),
        0.5, M_PI / 3.0, 1e9, 1e9, 0.0);
    EXPECT_GT(std::abs(uncapped.pose(turn, uncapped.num_steps - 1)[0]), 0.5);
}

TEST(TrajectoryLibraryTest, LatAccCapShrinksOmegaRange) {
    // max_lat_acc = 0.05, vx = 0.5 -> omega_lim = 0.1 for that speed only.
    const TrajectorySet lib = generate_trajectory_library_3d(
        15, 3.0, 0.1, Eigen::Vector3d::Zero(), Eigen::Vector4d(0, 0, 0, 1),
        0.5, M_PI / 3.0, 1e9, 0.05, 0.0);
    double min_omega_at_vmax = 1e9, max_omega_at_vmax = -1e9;
    double min_omega_at_rest = 1e9, max_omega_at_rest = -1e9;
    for (int t = 0; t < lib.num_trajectories; ++t) {
        if (std::abs(lib.params(t, 0) - 0.5) < 1e-12) {
            min_omega_at_vmax = std::min(min_omega_at_vmax, lib.params(t, 1));
            max_omega_at_vmax = std::max(max_omega_at_vmax, lib.params(t, 1));
        }
        if (lib.params(t, 0) == 0.0) {
            min_omega_at_rest = std::min(min_omega_at_rest, lib.params(t, 1));
            max_omega_at_rest = std::max(max_omega_at_rest, lib.params(t, 1));
        }
    }
    EXPECT_NEAR(max_omega_at_vmax, 0.1, 1e-12);
    EXPECT_NEAR(min_omega_at_vmax, -0.1, 1e-12);
    EXPECT_NEAR(max_omega_at_rest, M_PI / 3.0, 1e-12);  // unchanged at a standstill
}

TEST(TrajectoryVocabularyTest, SingleReverseRow) {
    const TrajectorySet vocab = generate_predefined_trajectory_vocabularies();
    ASSERT_EQ(vocab.num_trajectories, 1);
    EXPECT_EQ(vocab.num_steps, 31);
    EXPECT_NEAR(vocab.params(0, 0), -0.3, 1e-15);
    EXPECT_NEAR(vocab.params(0, 1), 0.0, 1e-15);
    // Moves along -z (body forward convention), z flattened to step 0's value.
    EXPECT_NEAR(vocab.pose(0, 0)[2], -0.03, 1e-12);
    for (int i = 0; i < vocab.num_steps; ++i) {
        EXPECT_EQ(vocab.pose(0, i)[2], vocab.pose(0, 0)[2]);
        EXPECT_NEAR(vocab.pose(0, i)[0], 0.0, 1e-12);
    }
    const TrajectorySet lib = generate_trajectory_library_3d();
    const TrajectorySet both = concatenate_trajectories(lib, vocab);
    EXPECT_EQ(both.num_trajectories, lib.num_trajectories + 1);
    EXPECT_TRUE(both.poses.row(lib.num_trajectories * lib.num_steps)
                    .isApprox(vocab.poses.row(0), 0.0));
    EXPECT_NEAR(both.params(both.num_trajectories - 1, 0), -0.3, 1e-15);
}

TEST(ReverseGateTest, HalfCellSlackAndNoWayForward) {
    const double res = 0.05;
    EXPECT_TRUE(reverse_armed(0.30, 5, res));    // 6 * 0.05 lands above 0.30: slack admits it
    EXPECT_TRUE(reverse_armed(0.32, 5, res));
    EXPECT_FALSE(reverse_armed(0.36, 5, res));
    EXPECT_TRUE(reverse_armed(10.0, 0, res));    // no forward trajectory at all
    EXPECT_FALSE(reverse_armed(10.0, 3, res));
}

TEST(ScoringTest, CollisionAndFreeScores) {
    // 10x10 ESDF at 0.1 m: everything 0.5 except cell (5,5) = 0 (an obstacle cell).
    Eigen::ArrayXXf esdf = Eigen::ArrayXXf::Constant(10, 10, 0.5f);
    esdf(5, 5) = 0.0f;
    Eigen::ArrayXXf path_dist = Eigen::ArrayXXf::Constant(10, 10, 1.0f);
    Eigen::ArrayXXf remaining = Eigen::ArrayXXf::Constant(10, 10, 10.0f);
    Eigen::ArrayXXf heading = Eigen::ArrayXXf::Zero(10, 10);
    const Eigen::Vector3d origin(0.0, 0.0, 0.0);

    TrajectorySet trajs;
    trajs.num_steps = 3;
    trajs.num_trajectories = 2;
    trajs.poses.setZero(6, 7);
    trajs.params.resize(2, 2);
    // Identity quat falls back to world +x forward, +y left.
    for (int i = 0; i < 3; ++i) trajs.poses.row(i) << 0.55, 0.55, 0.0, 0, 0, 0, 1;
    for (int i = 0; i < 3; ++i) trajs.poses.row(3 + i) << 5.5, 5.5, 0.0, 0, 0, 0, 1;
    trajs.params << 0.2, 0.0, 0.2, 0.0;

    const EsdfScoreResult res = score_trajectories_by_ESDF(
        trajs, esdf, path_dist, remaining, heading, origin, 0.1, 0.1, 0.0, 0.0, 0.0);
    // Trajectory 0 sits with its centre sample on the obstacle cell from step 0.
    EXPECT_EQ(res.scores[0], std::numeric_limits<double>::infinity());
    EXPECT_EQ(res.occ_points[0], 0);
    // Trajectory 1 never sees the map: every lookup off-grid -> min dist inf -> 0.0.
    EXPECT_EQ(res.scores[1], 0.0);
    EXPECT_EQ(res.occ_points[1], -1);
    // Route lookups: on-grid centre hits give path_cost 1.0, end/start remaining 10.
    EXPECT_NEAR(res.path_costs[0], 1.0, 1e-6);
    EXPECT_NEAR(res.end_remainings[0], 10.0, 1e-6);
    EXPECT_NEAR(res.end_heading_errs[0], 0.0, 1e-12);
    // Off-grid everywhere: path_cost falls back to 1e3, remaining to 1e3.
    EXPECT_NEAR(res.path_costs[1], 1e3, 1e-3);
    EXPECT_NEAR(res.end_remainings[1], 1e3, 1e-3);
}

TEST(ScoringTest, ClearanceDecayScore) {
    // A 20x20 map at 0.1 m with one obstacle at (0,0): ESDF = Euclidean distance
    // in metres. A trajectory whose centre passes exactly over cell (3,0)
    // (dist 0.3 < safety_radius 0.5) scores decay * 1/(0.3 + 1e-3).
    Eigen::ArrayXXf esdf(20, 20);
    for (int r = 0; r < 20; ++r) {
        for (int c = 0; c < 20; ++c) {
            esdf(r, c) = static_cast<float>(std::sqrt(double(r * r + c * c)) * 0.1);
        }
    }
    Eigen::ArrayXXf path_dist = Eigen::ArrayXXf::Constant(20, 20, 1.0f);
    Eigen::ArrayXXf remaining = Eigen::ArrayXXf::Constant(20, 20, 5.0f);
    Eigen::ArrayXXf heading = Eigen::ArrayXXf::Zero(20, 20);
    TrajectorySet trajs;
    trajs.num_steps = 2;
    trajs.num_trajectories = 1;
    trajs.poses.setZero(2, 7);
    trajs.params.resize(1, 2);
    trajs.poses.row(0) << 0.35, 0.05, 0.0, 0, 0, 0, 1;  // centre cell (3, 0)
    trajs.poses.row(1) << 1.35, 0.05, 0.0, 0, 0, 0, 1;
    trajs.params << 0.2, 0.0;
    const EsdfScoreResult res = score_trajectories_by_ESDF(
        trajs, esdf, path_dist, remaining, heading, Eigen::Vector3d::Zero(), 0.1,
        0.5, 0.0, 0.0, 0.0);
    // Only the centre sample is within the map at step 0 (others wrap off-grid at
    // negative indices -> skipped); step 1 cells are all >= 0.4 away... actually
    // (13,0) dist 1.3 etc., so min is 0.3 at step 0.
    EXPECT_EQ(res.occ_points[0], 0);
    const double expect = (2.0 - 0.0) / 2.0 * (1.0 / (0.3 + 1e-3));
    EXPECT_NEAR(res.scores[0], expect, 1e-6);  // esdf map is float32
}

TEST(RouteFieldsTest, StraightRouteAlongX) {
    Eigen::MatrixX2d route(2, 2);
    route << 0.05, 0.5, 0.95, 0.5;
    const RouteFields fields = build_route_fields(route, Eigen::Vector2i(10, 10),
                                                  Eigen::Vector3d::Zero(), 0.1);
    ASSERT_TRUE(fields.has_route);
    // Route cells are row r, col int(0.5/0.1) = 5 (0.5/0.1 rounds to exactly
    // 5.0 in IEEE; Python and C++ agree) for r = 0..9. The exact arc value per
    // cell sits at an int() truncation edge and is pinned by the Python
    // alignment fixtures, not here.
    EXPECT_NEAR(fields.path_dist_map(5, 5), 0.0, 1e-6);
    EXPECT_NEAR(fields.path_dist_map(5, 4), 0.1, 1e-6);
    EXPECT_NEAR(fields.path_dist_map(0, 0), 0.5, 1e-6);  // 5 cells from the route
    // Heading everywhere follows the route direction (+x): 0 rad.
    EXPECT_NEAR(fields.route_heading_map(0, 0), 0.0, 1e-6);
    EXPECT_NEAR(fields.route_heading_map(3, 7), 0.0, 1e-6);
    // remaining decreases along the route and stays inside [0, arc].
    const double r0 = fields.remaining_map(0, 5);
    const double r5 = fields.remaining_map(5, 5);
    const double r9 = fields.remaining_map(9, 5);
    EXPECT_GT(r0, r5);
    EXPECT_GT(r5, r9);
    EXPECT_GE(r9, -1e-6);
    EXPECT_LE(r0, 0.9 + 1e-6);
    // Off-route cells inherit the nearest route cell's fields.
    EXPECT_NEAR(fields.remaining_map(0, 0), fields.remaining_map(0, 5), 1e-6);
    EXPECT_NEAR(fields.route_heading_map(0, 0), fields.route_heading_map(0, 5), 1e-6);
}

TEST(RouteFieldsTest, DegenerateRoutes) {
    const RouteFields empty = build_route_fields(Eigen::MatrixX2d(0, 2),
                                                 Eigen::Vector2i(4, 4),
                                                 Eigen::Vector3d::Zero(), 0.1);
    EXPECT_FALSE(empty.has_route);
    EXPECT_TRUE((empty.path_dist_map == 1e3f).all());
    Eigen::MatrixX2d zero_len(2, 2);
    zero_len << 1.0, 1.0, 1.0, 1.0;
    const RouteFields flat = build_route_fields(zero_len, Eigen::Vector2i(4, 4),
                                                Eigen::Vector3d::Zero(), 0.1);
    EXPECT_FALSE(flat.has_route);
}

TEST(RollGridTest, ShiftGathersAndZeroesVacatedBand) {
    OccupancyGrid3D grid(3, 3, 1);
    double v = 1.0;
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            grid(x, y, 0) = v;
            v += 1.0;
        }
    }
    const Eigen::Vector3d old_origin(0.0, 0.0, 0.0);
    const RolledGrid rolled = roll_occupancy_grid(grid, old_origin,
                                                  Eigen::Vector3d(0.1, 0.0, 0.0), 0.1);
    EXPECT_EQ(rolled.grid(0, 0, 0), grid(1, 0, 0));
    EXPECT_EQ(rolled.grid(1, 0, 0), grid(2, 0, 0));
    EXPECT_EQ(rolled.grid(2, 0, 0), 0.0);  // vacated band
    EXPECT_NEAR(rolled.origin[0], 0.1, 1e-12);
    // Sub-half-voxel shift is an identity.
    const RolledGrid same = roll_occupancy_grid(grid, old_origin,
                                                Eigen::Vector3d(0.04, 0.0, 0.0), 0.1);
    EXPECT_EQ(same.origin[0], 0.0);
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            EXPECT_EQ(same.grid(x, y, 0), grid(x, y, 0));
        }
    }
}

TEST(ObstacleMapTest, ZBandSpanFilterAndDilation) {
    // 4x4x4 grid, resolution 0.1, origin (0,0,0), robot_z = 0.
    // z_world = (k+0.5)*0.1 in [z_bottom, z_top] = [-0.45, 0.2] -> band {0, 1}.
    const core::ObstacleConfig config;  // defaults
    OccupancyGrid3D grid(4, 4, 4);
    // Wall in cell (1,1): both band voxels occupied -> span 0.1 >= resolution
    // (floating: not near ground since low_z_rel 0.05 > -0.15).
    grid(1, 1, 0) = 0.2;
    grid(1, 1, 1) = 0.2;
    // Single-voxel noise in (2,2): span 0 -> rejected by the floating noise floor.
    grid(2, 2, 1) = 0.2;
    const Eigen::Vector3d origin(0.0, 0.0, 0.0);
    const Mask2D mask = build_obstacle_map(grid, origin, 0.1, 0.0, config);
    EXPECT_TRUE(mask(1, 1));
    EXPECT_FALSE(mask(2, 2));
    EXPECT_FALSE(mask(0, 0));

    // min_span_map override: a cell with a LARGER threshold (climb relaxation)
    // turns the same wall off; a strict cell keeps it.
    Eigen::ArrayXXf min_span = Eigen::ArrayXXf::Constant(4, 4, 0.05f);
    min_span(1, 1) = 0.2f;
    const Mask2D relaxed = build_obstacle_map(grid, origin, 0.1, 0.0, config, &min_span);
    EXPECT_FALSE(relaxed(1, 1));
    EXPECT_FALSE(relaxed(2, 2));

    // Dilation spreads the wall to its 4-neighbours.
    core::ObstacleConfig dilated = config;
    dilated.dilation_cells = 1;
    const Mask2D grown = build_obstacle_map(grid, origin, 0.1, 0.0, dilated);
    EXPECT_TRUE(grown(0, 1) && grown(2, 1) && grown(1, 0) && grown(1, 2));
    EXPECT_TRUE(grown(1, 1));
}

TEST(ObstacleMapTest, EmptyGridYieldsEmptyMask) {
    const core::ObstacleConfig config;
    const OccupancyGrid3D grid(4, 4, 4);
    const Mask2D mask = build_obstacle_map(grid, Eigen::Vector3d::Zero(), 0.1, 0.0, config);
    EXPECT_FALSE(mask.any());
}

TEST(RaycastingLoopyTest, SinglePixelSynthetic) {
    // Camera at origin (identity T), fx=fy=1, cx=cy=0; one depth pixel 0.15 at
    // (0,0). Grid 5x5x5 at 0.1, origin -0.25: start voxel (2,2,2), end voxel
    // floor((0.15+0.25)/0.1) = (2,2,4).
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::MatrixXf depth = Eigen::MatrixXf::Zero(1, 1);
    depth(0, 0) = 0.15f;
    const OccupancyGrid3D grid = run_raycasting_loopy(
        depth, T, Eigen::Vector3i(5, 5, 5), 1.0, 1.0, 0.0, 0.0,
        Eigen::Vector3d::Constant(-0.25), 1, 0.1);
    EXPECT_NEAR(grid(2, 2, 2), -0.05, 1e-15);
    EXPECT_NEAR(grid(2, 2, 3), -0.05, 1e-15);               // carved only
    EXPECT_NEAR(grid(2, 2, 4), 0.1, 1e-15);                 // 0.2 - 0.05, clipped
    // Everything else untouched.
    double abs_sum = 0.0;
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            for (int z = 0; z < 5; ++z) {
                if (!(x == 2 && y == 2 && z >= 2)) abs_sum += std::abs(grid(x, y, z));
            }
        }
    }
    EXPECT_NEAR(abs_sum, 0.0, 1e-15);
}

TEST(RaycastingLoopyTest, FilterGroundSkipsHitInsertion) {
    // py > 0 marks ground: with cy=0 that is v > 0. The v=1 ray must project
    // inside the grid for the difference to show: d = 0.2 -> world (0, 0.2, 0.2)
    // -> end voxel (2, 4, 4), in-grid.
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::MatrixXf depth = Eigen::MatrixXf::Zero(2, 1);
    depth(0, 0) = 0.15f;  // v=0: py = 0 -> not ground (strictly)
    depth(1, 0) = 0.2f;   // v=1: py = 0.2 > 0 -> ground
    const Eigen::Vector3d origin = Eigen::Vector3d::Constant(-0.25);
    const OccupancyGrid3D plain = run_raycasting_loopy(
        depth, T, Eigen::Vector3i(5, 5, 5), 1.0, 1.0, 0.0, 0.0, origin, 1, 0.1, false);
    const OccupancyGrid3D filtered = run_raycasting_loopy(
        depth, T, Eigen::Vector3i(5, 5, 5), 1.0, 1.0, 0.0, 0.0, origin, 1, 0.1, true);
    // v=0 ray: end voxel (2,2,4) takes its hit either way.
    EXPECT_NEAR(plain(2, 2, 4), 0.1, 1e-15);
    EXPECT_NEAR(filtered(2, 2, 4), 0.1, 1e-15);
    // v=1 ray (world (0, 0.2, 0.2) -> end voxel (2,4,4)): the hit lands only
    // without the ground filter.
    EXPECT_NEAR(plain(2, 4, 4), 0.1, 1e-15);
    EXPECT_NEAR(filtered(2, 4, 4), -0.05, 1e-15);  // carved, never hit
}

TEST(RaycastKernelTest, MatchesLoopyOnFiniteDepths) {
    // The de-pybind kernel and the njit port agree when depths are finite
    // (their only differences: the isfinite guard and filter_ground).
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(2, 3) = 0.1;
    Eigen::MatrixXf depth(4, 6);
    depth << 0.5f, 0.8f, 1.2f, 0.3f, 2.0f, 0.9f,
             0.4f, 0.0f, -1.0f, 1.5f, 0.7f, 0.2f,
             1.0f, 1.1f, 0.6f, 0.5f, 0.4f, 1.3f,
             0.2f, 0.9f, 1.4f, 0.8f, 0.3f, 0.6f;
    const Eigen::Vector3i shape(5, 5, 5);
    const Eigen::Vector3d origin = Eigen::Vector3d::Constant(-0.25);
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flat =
        kernels::run_raycasting_cpp(depth, T, shape, 60.0, 60.0, 2.5, 1.5, origin, 1, 0.1);
    const OccupancyGrid3D loopy = run_raycasting_loopy(
        depth, T, shape, 60.0, 60.0, 2.5, 1.5, origin, 1, 0.1);
    for (int x = 0; x < shape[0]; ++x) {
        for (int y = 0; y < shape[1]; ++y) {
            for (int z = 0; z < shape[2]; ++z) {
                EXPECT_NEAR(flat(x, y * shape[2] + z), loopy(x, y, z), 1e-12)
                    << x << "," << y << "," << z;
            }
        }
    }
}

TEST(NodeHelpersTest, CameraToRobotCenterAndFrontScan) {
    const core::RobotConfig& go2 = core::go2_config();
    // cam_offset_3d = [0, 0, 0.3] (left, up, forward); identity pose -> control
    // centre 0.3 behind the camera along body +Z.
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const Eigen::Vector3d center = camera_to_robot_center(T, go2);
    EXPECT_NEAR(center[2], -0.3, 1e-12);

    // Identity fwd falls back to mask +x (n < 1e-6), centre cell (50, 44) on a
    // 100x100 grid at 0.05 with origin (-2.5, -2.5).
    Mask2D mask = Mask2D::Constant(100, 100, false);
    const Eigen::Vector3d origin(-2.5, -2.5, 0.0);
    EXPECT_NEAR(front_obstacle_dist(T, mask, origin, 0.05, go2, 0.5), 1.5, 1e-12);
    // Pin the scanned cells with the same double expressions the Python and
    // C++ both evaluate. The control centre sits at (0, 0, -0.3): the 0.3 lead
    // is on the body-forward axis, and go2's lateral offset is 0, so the mask
    // (world-x, world-y) centre line is y = 0.
    auto cell_x = [&](int step) {
        // Mirror the scan exactly: x = fl + step * resolution, then int(/res).
        return static_cast<int>((0.0 + 0.25 + step * 0.05 + 2.5) / 0.05);
    };
    const int centre_y = static_cast<int>((0.0 + 2.5) / 0.05);  // l = 0 line
    mask(cell_x(0), centre_y) = true;
    EXPECT_NEAR(front_obstacle_dist(T, mask, origin, 0.05, go2, 0.5), 0.0, 1e-12);
    mask(cell_x(0), centre_y) = false;
    // NOTE: cell_x(1) quantises onto cell_x(0) at this resolution, so the next
    // distinct scanned cell is step 2 -> d_from_face 0.1.
    mask(cell_x(2), centre_y) = true;
    EXPECT_NEAR(front_obstacle_dist(T, mask, origin, 0.05, go2, 0.5), 0.1, 1e-12);

    // Footprint hits: an obstacle cell under a front-row lattice sample. The
    // go2 lattice samples l in {-0.15,-0.05,0.05,0.15}; front face f = 0.25
    // with l = -0.05 sits at world y = -0.05.
    const int front_l_x = static_cast<int>((0.25 + 2.5) / 0.05);
    const int front_l_y = static_cast<int>((0.0 - 0.05 + 2.5) / 0.05);
    mask(cell_x(2), centre_y) = false;
    mask(front_l_x, front_l_y) = true;
    const FootprintHits hits = footprint_hits(T, mask, origin, 0.05, go2);
    EXPECT_EQ(hits.n_samples, 25);
    ASSERT_EQ(hits.hits.size(), 1u);
    EXPECT_NEAR(hits.hits[0].first, 0.25, 1e-9);   // front_len: body fwd offset
    EXPECT_NEAR(hits.hits[0].second, -0.05, 1e-9);
}

TEST(NodeHelpersTest, SpeedFromClearanceSchedule) {
    // net <= 0.35 -> vx_min; >= 1.0 -> v_open; linear between, latency-discounted.
    EXPECT_NEAR(speed_from_clearance(1.0, 0.5, 0.8, 0.2, 0.35, 1.0, 0.2),
                0.2 + (0.9 - 0.35) / 0.65 * 0.6, 1e-12);
    EXPECT_NEAR(speed_from_clearance(0.2, 0.5, 0.8, 0.2, 0.35, 1.0, 0.2), 0.2, 1e-12);
    EXPECT_NEAR(speed_from_clearance(10.0, 0.5, 0.8, 0.2, 0.35, 1.0, 0.2), 0.8, 1e-12);
    EXPECT_NEAR(speed_from_clearance(0.0, 1.0, 0.8, 0.2, 0.35, 1.0, 0.2), 0.2, 1e-12);
}

TEST(SelectionTest, CostPicksForwardToGoalAndReverseGate) {
    TrajectorySet trajs;
    trajs.num_steps = 2;
    trajs.num_trajectories = 3;
    trajs.poses.setZero(6, 7);
    trajs.params.resize(3, 2);
    // All identity quats (heading 0), ends at x = 0 / 0.5 / 0.5 (reverse row).
    for (int i = 0; i < 2; ++i) trajs.poses.row(i) << 0.0, 0.0, 0.0, 0, 0, 0, 1;
    for (int i = 0; i < 2; ++i) trajs.poses.row(2 + i) << 0.5, 0.0, 0.0, 0, 0, 0, 1;
    for (int i = 0; i < 2; ++i) trajs.poses.row(4 + i) << 0.5, 0.0, 0.0, 0, 0, 0, 1;
    trajs.params << 0.0, 0.0, 0.3, 0.0, -0.3, 0.0;

    EsdfScoreResult scored;
    scored.scores = {0.0, 0.0, 0.0};
    scored.occ_points = {-1, -1, -1};
    scored.path_costs = {1e3, 1e3, 1e3};  // flat maps: no route
    scored.end_remainings = {1e3, 1e3, 1e3};
    scored.end_heading_errs = {0.0, 0.0, 0.0};

    // No route, target at (0.5, 0, 0), last command (0, 0), reverse not armed.
    const int pick = select_trajectory(
        trajs, trajs.params, scored, /*has_route=*/false,
        Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector2d(0.0, 0.0),
        /*should_reverse=*/false, DwaWeights{});
    EXPECT_EQ(pick, 1);  // T0 pays 100*0.5 + heading(0); T1 pays 10*smooth; T2 1e9 gate

    // With the reverse armed the gate flips and the reverse row wins the tie.
    const int pick_rev = select_trajectory(
        trajs, trajs.params, scored, false, Eigen::Vector3d(0.5, 0.0, 0.0),
        Eigen::Vector2d(-0.3, 0.0), /*should_reverse=*/true, DwaWeights{});
    EXPECT_EQ(pick_rev, 2);
}

}  // namespace
}  // namespace tinynav::planning
