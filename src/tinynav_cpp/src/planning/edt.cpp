// Port of the scipy.ndimage primitives planning_node.py relies on; see
// edt.hpp for the exact call forms matched.
#include "tinynav_cpp/planning/edt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tinynav::planning {
namespace {

// Squared-distance sentinel for "no feature seen yet". Large-but-finite so the
// lower-envelope slope arithmetic stays NaN-free (inf - inf would poison it);
// real squared distances on these grids are O(1e4), so 1e12 dominates safely.
constexpr double kBigSq = 1e12;

// 1D exact squared-distance transform along a scanline (Felzenszwalb &
// Huttenlocher lower envelope), tracking the argmin position per output cell.
// `f` holds squared distances to candidate features ON this line (kBigSq where
// none); on return `dist2`/`argmin` give, per position, the best (offset^2 + f)
// and the position achieving it.
void edt_1d(const std::vector<double>& f, std::vector<double>& dist2,
            std::vector<int>& argmin) {
    const int n = static_cast<int>(f.size());
    dist2.assign(n, 0.0);
    argmin.assign(n, 0);
    if (n == 0) return;
    std::vector<int> v(n);
    std::vector<double> z(n + 1);
    int k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < n; ++q) {
        double s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) /
                   (2.0 * (double(q) - double(v[k])));
        while (s <= z[k]) {
            --k;
            s = ((f[q] + double(q) * q) - (f[v[k]] + double(v[k]) * v[k])) /
                (2.0 * (double(q) - double(v[k])));
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = std::numeric_limits<double>::infinity();
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < double(q)) ++k;
        const int p = v[k];
        dist2[q] = (double(q) - double(p)) * (double(q) - double(p)) + f[p];
        argmin[q] = p;
    }
}

}  // namespace

EdtResult distance_transform_edt(const Mask2D& input) {
    const int rows = static_cast<int>(input.rows());
    const int cols = static_cast<int>(input.cols());
    EdtResult out;
    out.distances.setZero(rows, cols);
    out.nearest_row.setZero(rows, cols);
    out.nearest_col.setZero(rows, cols);
    if (rows == 0 || cols == 0) return out;

    bool any_feature = false;
    for (int r = 0; r < rows && !any_feature; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!input(r, c)) {  // features are the zero cells, scipy semantics
                any_feature = true;
                break;
            }
        }
    }
    if (!any_feature) {
        // SciPy reads a featureless input as all-zero distance (planning_node.py
        // leans on this being *replaced* by free_space_esdf, but match it anyway).
        return out;
    }

    // Pass 1, per row: squared distance to the nearest feature in the same row.
    Eigen::ArrayXXd row_dist2(rows, cols);   // g[r][c]
    Eigen::ArrayXXi row_arg(rows, cols);     // v[r][c]: nearest feature column
    std::vector<double> f(cols), dist2;
    std::vector<int> argmin;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            f[c] = input(r, c) ? kBigSq : 0.0;
        }
        edt_1d(f, dist2, argmin);
        for (int c = 0; c < cols; ++c) {
            row_dist2(r, c) = dist2[c];
            row_arg(r, c) = argmin[c];
        }
    }

    // Pass 2, per column: fold in the vertical offsets. The nearest feature of
    // (r, c) is then (best_row, row_arg(best_row, c)).
    std::vector<double> g(rows);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) g[r] = row_dist2(r, c);
        edt_1d(g, dist2, argmin);
        for (int r = 0; r < rows; ++r) {
            const int best_row = argmin[r];
            out.distances(r, c) = std::sqrt(std::max(0.0, dist2[r]));
            out.nearest_row(r, c) = best_row;
            out.nearest_col(r, c) = row_arg(best_row, c);
        }
    }
    return out;
}

Mask2D binary_dilation(const Mask2D& mask, int iterations) {
    Mask2D out = mask;
    if (iterations <= 0) return out;
    const int rows = static_cast<int>(mask.rows());
    const int cols = static_cast<int>(mask.cols());
    for (int it = 0; it < iterations; ++it) {
        Mask2D next = out;  // the SE includes the centre cell
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!out(r, c)) continue;
                if (r > 0) next(r - 1, c) = true;
                if (r + 1 < rows) next(r + 1, c) = true;
                if (c > 0) next(r, c - 1) = true;
                if (c + 1 < cols) next(r, c + 1) = true;
            }
        }
        out = std::move(next);
    }
    return out;
}

Mask2D maximum_filter(const Mask2D& seed, int size) {
    const int rows = static_cast<int>(seed.rows());
    const int cols = static_cast<int>(seed.cols());
    Mask2D out = Mask2D::Constant(rows, cols, false);
    if (size <= 1) return seed;
    const int k = size / 2;
    for (int r = 0; r < rows; ++r) {
        const int r_lo = std::max(0, r - k);
        const int r_hi = std::min(rows - 1, r + k);
        for (int c = 0; c < cols; ++c) {
            const int c_lo = std::max(0, c - k);
            const int c_hi = std::min(cols - 1, c + k);
            bool any = false;
            for (int rr = r_lo; rr <= r_hi && !any; ++rr) {
                for (int cc = c_lo; cc <= c_hi; ++cc) {
                    if (seed(rr, cc)) {
                        any = true;
                        break;
                    }
                }
            }
            out(r, c) = any;
        }
    }
    return out;
}

}  // namespace tinynav::planning
