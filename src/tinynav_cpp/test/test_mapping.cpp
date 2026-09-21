// Smoke tests for the mapping port: VLAD descriptor shape/norm on small synthetic
// features, and A* finding a known path on a small grid.
#include <cmath>

#include <gtest/gtest.h>

#include "tinynav_cpp/mapping/astar.hpp"
#include "tinynav_cpp/mapping/vlad.hpp"

namespace tinynav::mapping {
namespace {

TEST(Vlad, ComputeVladShapeAndNorm) {
  constexpr int K = 2, C = 4;
  Eigen::MatrixXd centres(K, C);
  centres << 1, 0, 0, 0,
             0, 1, 0, 0;
  Eigen::MatrixXd tokens(3, C);
  tokens << 1, 0.01, 0, 0,
            0.99, 0, 0, 0,
            0, 1, 0.02, 0;

  const Eigen::VectorXd desc = compute_vlad(tokens, centres);
  ASSERT_EQ(desc.size(), K * C);
  EXPECT_NEAR(desc.norm(), 1.0, 1e-9);

  // Empty token set -> zero descriptor (Python early return).
  const Eigen::VectorXd empty_desc = compute_vlad(Eigen::MatrixXd(0, C), centres);
  ASSERT_EQ(empty_desc.size(), K * C);
  EXPECT_EQ(empty_desc.norm(), 0.0);
}

TEST(Vlad, ComputeVladBatchShape) {
  constexpr int K = 2, C = 4;
  Eigen::MatrixXd centres(K, C);
  centres << 1, 0, 0, 0,
             0, 1, 0, 0;
  const std::vector<Eigen::MatrixXd> tokens_list = {
      Eigen::MatrixXd::Constant(5, C, 0.1),
      Eigen::MatrixXd(0, C),
  };
  const Eigen::MatrixXd descs = compute_vlad_batch(tokens_list, centres);
  ASSERT_EQ(descs.rows(), 2);
  ASSERT_EQ(descs.cols(), K * C);
  EXPECT_NEAR(descs.row(0).norm(), 1.0, 1e-9);
  EXPECT_EQ(descs.row(1).norm(), 0.0);
}

TEST(Vlad, TrainVocabularyStreaming) {
  // Two well-separated token clusters around e0 and e1; the learned centres must
  // land near them (up to permutation), with unit-norm rows.
  constexpr int C = 4;
  std::vector<Eigen::MatrixXd> frames;
  for (int f = 0; f < 8; ++f) {
    Eigen::MatrixXd tokens(16, C);
    for (int i = 0; i < 8; ++i) {
      tokens.row(i) << 1.0, 0.05 * (i % 3), 0, 0;
      tokens.row(8 + i) << 0.05 * (i % 2), 1.0, 0, 0;
    }
    frames.push_back(tokens);
  }
  TokenIteratorFactory factory = [&frames]() -> TokenIterator {
    return [i = size_t{0}, &frames](Eigen::MatrixXd& out) mutable -> bool {
      if (i >= frames.size()) return false;
      out = frames[i++];
      return true;
    };
  };

  const Eigen::MatrixXd centres =
      train_vocabulary_streaming(factory, /*vocab_size=*/2, /*epochs=*/2,
                                 /*batch_size=*/32, /*seed=*/42);
  ASSERT_EQ(centres.rows(), 2);
  ASSERT_EQ(centres.cols(), C);
  for (int k = 0; k < 2; ++k) {
    EXPECT_NEAR(centres.row(k).norm(), 1.0, 1e-9);
  }
  // Each learned centre is much closer to one of the two cluster directions.
  for (int k = 0; k < 2; ++k) {
    const double c0 = std::abs(centres(k, 0));
    const double c1 = std::abs(centres(k, 1));
    EXPECT_GT(std::max(c0, c1), 0.9);
  }
  // ...and the two centres picked different clusters.
  EXPECT_NE((centres(0, 0) > centres(0, 1)), (centres(1, 0) > centres(1, 1)));
}

TEST(Astar, WithinSdfFindsKnownPathAroundWall) {
  // 5x5x1 grid; a wall at x=2 for y in [0, 3] leaves only the (2, 4) gap.
  // sdf is high everywhere, so all cells sit in one bin queue and the search is
  // plain greedy best-first.
  SdfGrid sdf(5, 5, 1, 10.0);
  OccupancyGrid occ(5, 5, 1, 1);
  for (int y = 0; y < 4; ++y) {
    occ(2, y, 0) = 2;
  }
  const GridCell start{0, 0, 0};
  const GridCell goal{4, 4, 0};

  const std::vector<GridCell> path = search_within_sdf_map(start, goal, sdf, occ, 0.1);
  ASSERT_FALSE(path.empty());
  EXPECT_EQ(path.front(), start);
  EXPECT_EQ(path.back(), goal);
  for (size_t i = 1; i < path.size(); ++i) {
    const GridCell d{path[i].x - path[i - 1].x, path[i].y - path[i - 1].y,
                     path[i].z - path[i - 1].z};
    EXPECT_LE(std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)}), 1);
    EXPECT_NE(occ(path[i].x, path[i].y, path[i].z), 2);
  }
  // The only way across the wall is the gap at y = 4.
  for (const GridCell& cell : path) {
    if (cell.x == 2) {
      EXPECT_EQ(cell.y, 4);
    }
  }
}

TEST(Astar, WithinSdfUnreachableGoalReturnsEmpty) {
  // Wall across the whole grid: no path.
  SdfGrid sdf(4, 4, 1, 10.0);
  OccupancyGrid occ(4, 4, 1, 1);
  for (int y = 0; y < 4; ++y) {
    occ(1, y, 0) = 2;
  }
  const std::vector<GridCell> path =
      search_within_sdf_map({0, 0, 0}, {3, 3, 0}, sdf, occ, 0.1);
  EXPECT_TRUE(path.empty());
}

TEST(Astar, CloseToSdfWalksDownToCorridor) {
  // One low-sdf cell (the corridor) at (4, 4); the search must end there.
  SdfGrid sdf(5, 5, 1, 5.0);
  sdf(4, 4, 0) = 0.1;
  OccupancyGrid occ(5, 5, 1, 1);
  const std::vector<GridCell> path =
      search_close_to_sdf_map({0, 0, 0}, sdf, occ, 0.2);
  ASSERT_FALSE(path.empty());
  EXPECT_EQ(path.front(), (GridCell{0, 0, 0}));
  EXPECT_EQ(path.back(), (GridCell{4, 4, 0}));
}

}  // namespace
}  // namespace tinynav::mapping
