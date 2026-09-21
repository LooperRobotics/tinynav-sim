// Port of reference/tinynav/core/path_prior.py + path_speed.py + path_climb.py.
#include "tinynav_cpp/mapping/path_prior.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tinynav::mapping {
namespace {

// np.percentile with the default 'linear' interpolation.
double percentile_linear(std::vector<double> w, double pct) {
  std::sort(w.begin(), w.end());
  if (w.size() == 1) {
    return w[0];
  }
  const double rank = pct / 100.0 * static_cast<double>(w.size() - 1);
  const auto i = static_cast<size_t>(std::floor(rank));
  const double frac = rank - static_cast<double>(i);
  if (i + 1 >= w.size()) {
    return w.back();
  }
  return w[i] + frac * (w[i + 1] - w[i]);
}

// np.searchsorted(side='left') / (side='right') over a sorted ascending vector.
int searchsorted_left(const Eigen::VectorXd& v, double x) {
  const double* begin = v.data();
  return static_cast<int>(std::lower_bound(begin, begin + v.size(), x) - begin);
}
int searchsorted_right(const Eigen::VectorXd& v, double x) {
  const double* begin = v.data();
  return static_cast<int>(std::upper_bound(begin, begin + v.size(), x) - begin);
}

int sign_of(double x) { return (x > 0.0) - (x < 0.0); }

std::string parse_npy_descr(const std::string& header) {
  const size_t key = header.find("'descr'");
  if (key == std::string::npos) {
    throw std::runtime_error("load_npy_matrix: no 'descr' in header");
  }
  const size_t colon = header.find(':', key);
  const size_t quote = colon == std::string::npos
                           ? std::string::npos
                           : header.find_first_of("'\"", colon);
  if (quote == std::string::npos) {
    throw std::runtime_error("load_npy_matrix: malformed 'descr' in header");
  }
  const size_t end = header.find(header[quote], quote + 1);
  if (end == std::string::npos) {
    throw std::runtime_error("load_npy_matrix: malformed 'descr' in header");
  }
  return header.substr(quote + 1, end - quote - 1);
}

std::vector<int64_t> parse_npy_shape(const std::string& header) {
  const size_t key = header.find("'shape'");
  if (key == std::string::npos) {
    throw std::runtime_error("load_npy_matrix: no 'shape' in header");
  }
  const size_t open = header.find('(', key);
  const size_t close = header.find(')', open);
  std::vector<int64_t> dims;
  size_t pos = open + 1;
  while (pos < close) {
    while (pos < close && (header[pos] == ' ' || header[pos] == ',')) ++pos;
    if (pos >= close) break;
    size_t next = pos;
    while (next < close && std::isdigit(header[next])) ++next;
    dims.push_back(std::stoll(header.substr(pos, next - pos)));
    pos = next;
  }
  return dims;
}

bool parse_npy_fortran_order(const std::string& header) {
  const size_t key = header.find("'fortran_order'");
  if (key == std::string::npos) {
    throw std::runtime_error("load_npy_matrix: no 'fortran_order' in header");
  }
  const size_t colon = header.find(':', key);
  const size_t value = colon == std::string::npos
                           ? std::string::npos
                           : header.find_first_not_of(' ', colon + 1);
  if (value == std::string::npos) {
    throw std::runtime_error("load_npy_matrix: malformed 'fortran_order' in header");
  }
  return header.compare(value, 4, "True") == 0;
}

}  // namespace

// ------------------------------------------------------------------ KdTree

KdTree::KdTree(const Eigen::MatrixXd& points) : points_(points) {}

int KdTree::nearest(const Eigen::Vector3d& p, double& dist_out) const {
  if (points_.rows() == 0) {
    return -1;
  }
  Eigen::Index idx = 0;
  const double d2 = (points_.rowwise() - p.transpose()).rowwise().squaredNorm().minCoeff(&idx);
  dist_out = std::sqrt(d2);
  return static_cast<int>(idx);
}

std::vector<int> KdTree::within_horizontal(const Eigen::Vector2d& p, double radius_m) const {
  std::vector<int> out;
  if (radius_m < 0.0) {
    return out;
  }
  const double r2 = radius_m * radius_m;
  for (Eigen::Index i = 0; i < points_.rows(); ++i) {
    const double dx = points_(i, 0) - p.x();
    const double dy = points_(i, 1) - p.y();
    if (dx * dx + dy * dy <= r2) {
      out.push_back(static_cast<int>(i));
    }
  }
  return out;
}

// ------------------------------------------------------------------ path_prior.py

bool is_stale(const std::string& map_path, const std::string& prior_filename) {
  namespace fs = std::filesystem;
  const fs::path out = fs::path(map_path) / prior_filename;
  if (!fs::exists(out)) {
    return true;
  }
  const fs::path poses = fs::path(map_path) / "poses.npy";
  if (!fs::exists(poses)) {
    return false;  // nothing to compare against; leave what is there
  }
  return fs::last_write_time(poses) > fs::last_write_time(out);
}

Eigen::MatrixXd poses_to_positions(const std::map<int64_t, Eigen::Matrix4d>& poses) {
  Eigen::MatrixXd pos(poses.size(), 3);
  Eigen::Index i = 0;
  for (const auto& [timestamp, T] : poses) {
    (void)timestamp;
    pos.row(i++) = T.block<3, 1>(0, 3).transpose();
  }
  return pos;
}

Eigen::VectorXd horizontal_arclength(const Eigen::MatrixXd& pos) {
  const Eigen::Index n = pos.rows();
  Eigen::VectorXd s(n);
  if (n == 0) {
    return s;
  }
  s[0] = 0.0;
  for (Eigen::Index i = 1; i < n; ++i) {
    s[i] = s[i - 1] + (pos.row(i).head<2>() - pos.row(i - 1).head<2>()).norm();
  }
  return s;
}

Eigen::MatrixXd load_npy_matrix(const std::string& npy_path) {
  std::ifstream f(npy_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("load_npy_matrix: cannot open " + npy_path);
  }
  char magic[6];
  f.read(magic, 6);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("load_npy_matrix: not a .npy file: " + npy_path);
  }
  uint8_t major = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.ignore(1);  // minor
  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t h = 0;
    f.read(reinterpret_cast<char*>(&h), 2);
    header_len = h;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!f) {
    throw std::runtime_error("load_npy_matrix: truncated header: " + npy_path);
  }

  const std::string descr = parse_npy_descr(header);
  if (parse_npy_fortran_order(header)) {
    throw std::runtime_error("load_npy_matrix: fortran_order arrays unsupported: " + npy_path);
  }
  const std::vector<int64_t> dims = parse_npy_shape(header);
  if (dims.empty() || dims.size() > 2) {
    throw std::runtime_error("load_npy_matrix: only 1D/2D arrays supported: " + npy_path);
  }
  const int64_t rows = dims[0];
  const int64_t cols = dims.size() == 2 ? dims[1] : 1;

  // Hosts are little-endian (x86_64 / aarch64), so '<' payloads read directly.
  Eigen::MatrixXd out(rows, cols);
  if (descr == "<f8") {
    std::vector<double> buf(static_cast<size_t>(rows * cols));
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(rows * cols * sizeof(double)));
    out = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        buf.data(), rows, cols);
  } else if (descr == "<f4") {
    std::vector<float> buf(static_cast<size_t>(rows * cols));
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(rows * cols * sizeof(float)));
    out = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
              buf.data(), rows, cols)
              .cast<double>();
  } else {
    throw std::runtime_error("load_npy_matrix: unsupported dtype '" + descr + "': " + npy_path);
  }
  if (!f) {
    throw std::runtime_error("load_npy_matrix: truncated data: " + npy_path);
  }
  return out;
}

PathSampleIndex::PathSampleIndex(const Eigen::MatrixXd& pts, double assoc_m)
    : pts_(pts), assoc_m_(assoc_m), tree_(pts.leftCols(3)) {
  if (pts.cols() < 4) {
    throw std::invalid_argument("PathSampleIndex: pts must be (N, >=4) [x, y, z, label...]");
  }
}

PathSampleIndex PathSampleIndex::load(const std::string& npy_path, double assoc_m) {
  return PathSampleIndex(load_npy_matrix(npy_path), assoc_m);
}

std::optional<double> PathSampleIndex::nearest_value(const Eigen::Vector3d& position_xyz) const {
  if (pts_.rows() == 0) {
    return std::nullopt;
  }
  double dist = 0.0;
  const int i = tree_.nearest(position_xyz, dist);
  if (i < 0 || dist > assoc_m_) {
    return std::nullopt;
  }
  return pts_(i, 3);
}

Eigen::MatrixXd PathSampleIndex::samples_within(const Eigen::Vector3d& position_xyz,
                                                double radius_m, double min_label) const {
  const std::vector<int> near = tree_.within_horizontal(position_xyz.head<2>(), radius_m);
  std::vector<int> kept;
  for (const int i : near) {
    if (pts_(i, 3) >= min_label) {
      kept.push_back(i);
    }
  }
  Eigen::MatrixXd out(kept.size(), 3);
  for (size_t r = 0; r < kept.size(); ++r) {
    out.row(r) = pts_.row(kept[r]).head<3>();
  }
  return out;
}

// ------------------------------------------------------------------ path_speed.py

Eigen::MatrixXd compute_path_speed(const std::map<int64_t, Eigen::Matrix4d>& poses,
                                   const PathSpeedParams& params) {
  const Eigen::Index n = static_cast<Eigen::Index>(poses.size());
  Eigen::MatrixXd out(n, 4);
  if (n == 0) {
    return out;
  }
  const Eigen::MatrixXd pos = poses_to_positions(poses);
  out.leftCols(3) = pos;
  out.col(3).setConstant(std::numeric_limits<double>::quiet_NaN());
  if (n < 2) {
    return out;
  }
  std::vector<double> t_s;
  t_s.reserve(n);
  for (const auto& [timestamp, T] : poses) {
    (void)T;
    t_s.push_back(static_cast<double>(timestamp) * 1e-9);  // ns -> s (capture order)
  }
  Eigen::VectorXd seg_speed(n - 1);
  std::vector<bool> valid(n - 1);
  for (Eigen::Index i = 0; i < n - 1; ++i) {
    const double seg_dist = (pos.row(i + 1) - pos.row(i)).norm();
    const double dt = t_s[i + 1] - t_s[i];
    // IEEE: dt == 0 gives inf/nan here, and the valid mask below excludes it.
    seg_speed[i] = seg_dist / dt;
    valid[i] = dt > params.min_dt_s && std::isfinite(seg_speed[i]) && seg_speed[i] <= params.max_speed;
  }
  // Window segments by the arclength of their midpoints; speed itself is from 3D motion.
  const Eigen::VectorXd s = horizontal_arclength(pos);
  Eigen::VectorXd s_mid(n - 1);
  for (Eigen::Index i = 0; i < n - 1; ++i) {
    s_mid[i] = 0.5 * (s[i] + s[i + 1]);
  }
  for (Eigen::Index i = 0; i < n; ++i) {
    const int lo = searchsorted_left(s_mid, s[i] - params.win_m);
    const int hi = searchsorted_right(s_mid, s[i] + params.win_m);
    std::vector<double> w;
    for (int j = lo; j < hi; ++j) {
      if (valid[j]) {
        w.push_back(seg_speed[j]);
      }
    }
    if (!w.empty()) {
      out(i, 3) = percentile_linear(std::move(w), params.pct);
    }
  }
  return out;
}

double PathSpeedIndex::speed_cap(const Eigen::Vector3d& position_xyz) const {
  const std::optional<double> v = index_.nearest_value(position_xyz);
  if (v.has_value() && std::isfinite(*v) && *v > 0.0) {
    return *v;
  }
  return std::numeric_limits<double>::infinity();
}

// ------------------------------------------------------------------ path_climb.py

int n_climbing(const Eigen::MatrixXd& labels) {
  int n = 0;
  for (Eigen::Index i = 0; i < labels.rows(); ++i) {
    if (labels(i, 3) >= CLIMBING) {
      ++n;
    }
  }
  return n;
}

Eigen::MatrixXd compute_path_climb(const std::map<int64_t, Eigen::Matrix4d>& poses,
                                   const PathClimbParams& params) {
  const Eigen::MatrixXd pos = poses_to_positions(poses);
  const Eigen::Index n = pos.rows();
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, 4);
  if (n == 0) {
    return out;
  }
  out.leftCols(3) = pos;
  if (n < 3) {
    return out;
  }
  const Eigen::VectorXd s = horizontal_arclength(pos);
  for (Eigen::Index i = 0; i < n; ++i) {
    const int j0 = searchsorted_left(s, s[i] - params.win_m);
    const int j1 = searchsorted_right(s, s[i] + params.win_m);
    if (j1 - j0 < 3) {
      continue;
    }
    double max_abs_step = 0.0;
    std::vector<double> steps;
    steps.reserve(j1 - j0 - 1);
    for (int j = j0; j + 1 < j1; ++j) {
      const double dz = pos(j + 1, 2) - pos(j, 2);
      max_abs_step = std::max(max_abs_step, std::abs(dz));
      steps.push_back(dz);
    }
    if (steps.empty() || max_abs_step > params.max_step_dz) {
      continue;  // teleport / discontinuity in window
    }
    const double net = pos(j1 - 1, 2) - pos(j0, 2);
    if (std::abs(net) < params.min_rise) {
      continue;
    }
    // Vote only over steps that actually move in z.
    int moving = 0;
    int same = 0;
    for (const double dz : steps) {
      if (std::abs(dz) >= params.noise_dz) {
        ++moving;
        if (sign_of(dz) == sign_of(net)) {
          ++same;
        }
      }
    }
    if (moving == 0) {
      continue;
    }
    if (static_cast<double>(same) / static_cast<double>(moving) >= params.consistency) {
      out(i, 3) = 1.0;
    }
  }
  return out;
}

Eigen::MatrixXd PathClimbIndex::climbing_within(const Eigen::Vector3d& position_xyz,
                                                double radius_m) const {
  return index_.samples_within(position_xyz, radius_m, CLIMBING);
}

bool PathClimbIndex::on_stairs(const Eigen::Vector3d& position_xyz) const {
  const std::optional<double> v = index_.nearest_value(position_xyz);
  return v.has_value() && *v >= CLIMBING;
}

}  // namespace tinynav::mapping
