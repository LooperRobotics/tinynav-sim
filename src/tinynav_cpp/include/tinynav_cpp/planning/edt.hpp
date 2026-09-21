// scipy.ndimage replacements used by the planning port (planning_node.py calls
// distance_transform_edt / binary_dilation / maximum_filter). Pure functions,
// no ROS. Semantics match scipy for the exact call forms used there:
//   - distance_transform_edt(~mask, return_indices=True): distance FROM each
//     nonzero cell TO the nearest zero cell (the zeros are the features).
//   - binary_dilation(mask, iterations=N): default cross structuring element
//     (generate_binary_structure(2, 1)) applied N times == 4-connected
//     Manhattan ball of radius N.
//   - maximum_filter(mask, size=2k+1, mode='constant'): square window, zero pad.
#pragma once

#include <Eigen/Dense>

namespace tinynav::planning {

using Mask2D = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>;

// Port of scipy.ndimage.distance_transform_edt for the 2D boolean case.
// Distances are in CELLS (float64, exact Euclidean — Felzenszwalb & Huttenlocher
// two-pass lower envelope), scipy returns float64 too; callers scale by
// resolution themselves. `nearest_row`/`nearest_col` carry the coordinates of the
// nearest feature cell (the return_indices=True gather source) for every cell,
// itself included for feature cells. SciPy quirk kept: an input with NO zero
// cells returns all-zero distances (noted in planning_node.py's
// free_space_esdf comment), with the nearest indices left at (0, 0).
//
// Tie-breaking note: distances are exact, but WHICH equidistant feature cell is
// reported can differ from scipy's implementation; downstream fields that gather
// through the indices (remaining_map / route_heading_map) are only ambiguous on
// exactly symmetric route layouts.
struct EdtResult {
    Eigen::ArrayXXd distances;  // in cells
    Eigen::ArrayXXi nearest_row;
    Eigen::ArrayXXi nearest_col;
};
EdtResult distance_transform_edt(const Mask2D& input);

// Port of scipy.ndimage.binary_dilation with the default cross structure.
Mask2D binary_dilation(const Mask2D& mask, int iterations);

// Port of scipy.ndimage.maximum_filter for a boolean input, odd square size,
// mode='constant' (zero padding). Output(r, c) = any seed within the window.
Mask2D maximum_filter(const Mask2D& seed, int size);

}  // namespace tinynav::planning
