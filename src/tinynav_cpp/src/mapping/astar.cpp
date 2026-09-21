// Port of the global-path search extracted from reference/tinynav/core/map_node.py.
#include "tinynav_cpp/mapping/astar.hpp"

#include <cmath>
#include <queue>
#include <unordered_set>

namespace tinynav::mapping {
namespace {

// Min-heap matching heapq's (priority, cell_tuple) ordering: smallest priority
// first, ties broken by lexicographically smallest (x, y, z).
struct QueueEntry {
  double cost;
  GridCell cell;
};

struct QueueEntryGreater {
  bool operator()(const QueueEntry& a, const QueueEntry& b) const {
    if (a.cost != b.cost) return a.cost > b.cost;
    if (a.cell.x != b.cell.x) return a.cell.x > b.cell.x;
    if (a.cell.y != b.cell.y) return a.cell.y > b.cell.y;
    return a.cell.z > b.cell.z;
  }
};

using MinHeap = std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater>;
using CellSet = std::unordered_set<GridCell, GridCellHash>;

bool in_bounds(const GridCell& c, const std::array<int, 3>& shape) {
  return c.x >= 0 && c.x < shape[0] && c.y >= 0 && c.y < shape[1] && c.z >= 0 && c.z < shape[2];
}

template <typename F>
void for_each_neighbor26(const GridCell& c, F&& f) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        f(GridCell{c.x + dx, c.y + dy, c.z + dz});
      }
    }
  }
}

}  // namespace

double heuristic(const GridCell& start, const GridCell& goal, double resolution) {
  const double dx = static_cast<double>(start.x - goal.x) * resolution;
  const double dy = static_cast<double>(start.y - goal.y) * resolution;
  const double dz = static_cast<double>(start.z - goal.z) * resolution;
  return std::sqrt(dx * dx + dy * dy + dz * dz) + 20.0 * std::abs(dz);
}

std::vector<GridCell> reconstruct_path_sdf(const CellParentMap& parent,
                                           const GridCell& current) {
  std::vector<GridCell> path;
  GridCell cursor = current;
  while (parent.count(cursor) != 0) {
    path.push_back(cursor);
    const GridCell& next = parent.at(cursor);
    if (next == cursor) break;
    cursor = next;
  }
  return {path.rbegin(), path.rend()};
}

std::vector<GridCell> search_close_to_sdf_map(const GridCell& start_index,
                                              const SdfGrid& sdf_map,
                                              const OccupancyGrid& occupancy_map,
                                              double stop_distance) {
  MinHeap open_heap;
  open_heap.push({sdf_map(start_index.x, start_index.y, start_index.z), start_index});
  CellSet open_heap_set{start_index};
  CellParentMap parent{{start_index, start_index}};
  CellSet visited;
  while (!open_heap.empty()) {
    const QueueEntry top = open_heap.top();
    open_heap.pop();
    const double current_sdf = top.cost;
    const GridCell current = top.cell;
    open_heap_set.erase(current);
    visited.insert(current);
    if (current_sdf < stop_distance) {
      return reconstruct_path_sdf(parent, current);
    }
    for_each_neighbor26(current, [&](const GridCell& neighbor) {
      if (!in_bounds(neighbor, sdf_map.shape())) return;
      if (open_heap_set.count(neighbor) != 0 || visited.count(neighbor) != 0) return;
      if (occupancy_map(neighbor.x, neighbor.y, neighbor.z) == 2) return;
      open_heap_set.insert(neighbor);
      open_heap.push({sdf_map(neighbor.x, neighbor.y, neighbor.z), neighbor});
      parent[neighbor] = current;
    });
  }
  return {};
}

std::vector<GridCell> search_within_sdf_map(const GridCell& start,
                                            const GridCell& goal,
                                            const SdfGrid& sdf_map,
                                            const OccupancyGrid& occupancy_map,
                                            double resolution) {
  const double sdf_bins[] = {0.2, 0.5, 1.0, 2.0, 5.0, 10.0};
  constexpr int kNumQueues = 7;  // len(sdf_bins) + 1

  auto get_queue_index = [&](double sdf_value) {
    for (int idx = 0; idx < kNumQueues - 1; ++idx) {
      if (sdf_value < sdf_bins[idx]) return idx;
    }
    return kNumQueues - 1;
  };

  std::array<MinHeap, kNumQueues> open_heaps;
  std::array<CellSet, kNumQueues> open_sets;
  const int start_queue_idx = get_queue_index(sdf_map(start.x, start.y, start.z));
  open_heaps[start_queue_idx].push({heuristic(start, goal, resolution), start});
  open_sets[start_queue_idx].insert(start);
  CellParentMap parent{{start, start}};
  CellSet visited;

  while (true) {
    int queue_idx = -1;
    for (int i = 0; i < kNumQueues; ++i) {
      if (!open_heaps[i].empty()) {
        queue_idx = i;
        break;
      }
    }
    if (queue_idx == -1) break;

    const QueueEntry top = open_heaps[queue_idx].top();
    open_heaps[queue_idx].pop();
    const GridCell current = top.cell;
    open_sets[queue_idx].erase(current);
    if (visited.count(current) != 0) continue;
    visited.insert(current);
    if (current == goal) {
      return reconstruct_path_sdf(parent, current);
    }
    for_each_neighbor26(current, [&](const GridCell& neighbor) {
      if (!in_bounds(neighbor, sdf_map.shape())) return;
      if (visited.count(neighbor) != 0) return;
      if (occupancy_map(neighbor.x, neighbor.y, neighbor.z) == 2) return;
      const int neighbor_queue_idx =
          get_queue_index(sdf_map(neighbor.x, neighbor.y, neighbor.z));
      if (open_sets[neighbor_queue_idx].count(neighbor) != 0) return;
      open_sets[neighbor_queue_idx].insert(neighbor);
      open_heaps[neighbor_queue_idx].push({heuristic(neighbor, goal, resolution), neighbor});
      if (parent.count(neighbor) == 0) {
        parent[neighbor] = current;
      }
    });
  }
  return {};
}

}  // namespace tinynav::mapping
