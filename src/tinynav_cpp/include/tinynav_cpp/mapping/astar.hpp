// Port of the global-path search extracted from reference/tinynav/core/map_node.py:
// heuristic / reconstruct_path_sdf / search_close_to_sdf_map / search_within_sdf_map.
// Algorithm only — none of the node state. The heapq open lists become
// std::priority_queue with a (cost, cell) ordering matching Python's tuple compare.
//
// Grid semantics (from build_map_node.py::generate_occupancy_map):
//   sdf_map   — distance (m) to the nearest CAPTURE-TRAJECTORY seed, not to the
//               nearest obstacle; low sdf == on the driven corridor. occupancy:
//               0 unknown, 1 free, 2 occupied; the searches treat != 2 as passable.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tinynav::mapping {

// Minimal dense 3D grid, C-order layout matching numpy [i][j][k]:
// offset = (i * ny + j) * nz + k.
template <typename T>
class Grid3 {
 public:
  Grid3() = default;
  Grid3(int nx, int ny, int nz, T fill = T{})
      : shape_{nx, ny, nz}, data_(static_cast<size_t>(nx) * ny * nz, fill) {}

  T& operator()(int i, int j, int k) {
    return data_[(static_cast<size_t>(i) * shape_[1] + j) * shape_[2] + k];
  }
  const T& operator()(int i, int j, int k) const {
    return data_[(static_cast<size_t>(i) * shape_[1] + j) * shape_[2] + k];
  }

  const std::array<int, 3>& shape() const { return shape_; }
  std::vector<T>& data() { return data_; }
  const std::vector<T>& data() const { return data_; }

 private:
  std::array<int, 3> shape_{0, 0, 0};
  std::vector<T> data_;
};

using SdfGrid = Grid3<double>;       // sdf_map.npy is float32; promoted per the double rule
using OccupancyGrid = Grid3<uint8_t>;

// Integer voxel index, the Python tuple coordinate.
struct GridCell {
  int x = 0, y = 0, z = 0;
  bool operator==(const GridCell& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  bool operator!=(const GridCell& other) const { return !(*this == other); }
};

struct GridCellHash {
  size_t operator()(const GridCell& c) const noexcept {
    size_t h = std::hash<int>()(c.x);
    h = h * 1000003u ^ std::hash<int>()(c.y);
    h = h * 1000003u ^ std::hash<int>()(c.z);
    return h;
  }
};

using CellParentMap = std::unordered_map<GridCell, GridCell, GridCellHash>;

// Port of reference/tinynav/core/map_node.py::heuristic.
double heuristic(const GridCell& start, const GridCell& goal, double resolution);

// Port of reference/tinynav/core/map_node.py::reconstruct_path_sdf.
// Walks the parent map from `current` back to the root, returns start-first.
std::vector<GridCell> reconstruct_path_sdf(const CellParentMap& parent,
                                           const GridCell& current);

// Port of reference/tinynav/core/map_node.py::search_close_to_sdf_map.
// Greedy best-first (priority = sdf value) from start_index to the first popped
// cell with sdf < stop_distance — i.e. it snaps a possibly off-corridor start/goal
// onto the capture path. Empty path when no such cell is reachable.
std::vector<GridCell> search_close_to_sdf_map(const GridCell& start_index,
                                              const SdfGrid& sdf_map,
                                              const OccupancyGrid& occupancy_map,
                                              double stop_distance);

// Port of reference/tinynav/core/map_node.py::search_within_sdf_map.
// Multi-queue best-first search: cells are bucketed by sdf into the bins
// [0.2, 0.5, 1.0, 2.0, 5.0, 10.0] and the lowest-sdf non-empty queue is popped
// first, so the path hugs the capture corridor; within a queue the priority is
// `heuristic(cell, goal, resolution)`. Empty path when goal is unreachable.
std::vector<GridCell> search_within_sdf_map(const GridCell& start,
                                            const GridCell& goal,
                                            const SdfGrid& sdf_map,
                                            const OccupancyGrid& occupancy_map,
                                            double resolution);

}  // namespace tinynav::mapping
