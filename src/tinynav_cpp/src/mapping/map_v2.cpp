// Map format v2 loader: reads the plain-npy map written by tools/export_map_v2.py
// (poses + SuperPoint features + depths + DINOv2 patch-VLAD index that
// map_node.py otherwise loads through pickle and shelve).
// Port of reference/tinynav/core/map_node.py::load_map (the pickled/shelve part)
// and reference/tinynav/core/map_node.py::keypoint_with_depth_to_3d.
#include "tinynav_cpp/mapping/map_v2.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

namespace tinynav::mapping
{

namespace
{

// Minimal .npy reader: f32/f64/i64/u1, C-order only (the exporter's contract).
struct NpyArray
{
  enum class DType { F4, F8, I8, U1 };
  std::vector<int64_t> shape;
  DType dtype = DType::F4;
  std::vector<uint8_t> bytes;

  size_t count() const
  {
    size_t total = 1;
    for (int64_t dim : shape) {
      total *= static_cast<size_t>(dim);
    }
    return total;
  }
};

bool read_npy(const std::string & path, NpyArray & out, std::string & error)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  char magic[6] = {0};
  in.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    error = "not an npy file: " + path;
    return false;
  }
  uint8_t ver[2] = {0, 0};
  in.read(reinterpret_cast<char *>(ver), 2);
  uint16_t header_len = 0;
  if (ver[0] == 1) {
    in.read(reinterpret_cast<char *>(&header_len), 2);
  } else if (ver[0] == 2 || ver[0] == 3) {
    uint32_t len32 = 0;
    in.read(reinterpret_cast<char *>(&len32), 4);
    header_len = static_cast<uint16_t>(len32);
  } else {
    error = "unsupported npy version in " + path;
    return false;
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) {
    error = "truncated npy header: " + path;
    return false;
  }

  const auto descr_pos = header.find("'descr'");
  if (descr_pos == std::string::npos) {
    error = "npy header has no descr: " + path;
    return false;
  }
  if (header.find("<f4", descr_pos) != std::string::npos) {
    out.dtype = NpyArray::DType::F4;
  } else if (header.find("<f8", descr_pos) != std::string::npos) {
    out.dtype = NpyArray::DType::F8;
  } else if (header.find("<i8", descr_pos) != std::string::npos) {
    out.dtype = NpyArray::DType::I8;
  } else if (header.find("u1", descr_pos) != std::string::npos) {
    out.dtype = NpyArray::DType::U1;
  } else {
    error = "unsupported dtype in " + path + " (want <f4/<f8/<i8/u1)";
    return false;
  }
  if (header.find("True", header.find("'fortran_order'")) != std::string::npos) {
    error = "fortran-order npy not supported: " + path;
    return false;
  }

  const auto shape_pos = header.find("'shape'");
  const auto open = header.find('(', shape_pos);
  const auto close = header.find(')', shape_pos);
  if (shape_pos == std::string::npos || open == std::string::npos ||
    close == std::string::npos)
  {
    error = "npy header has no shape: " + path;
    return false;
  }
  out.shape.clear();
  std::stringstream ss(header.substr(open + 1, close - open - 1));
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (!token.empty()) {
      out.shape.push_back(std::stoll(token));
    }
  }

  size_t elem_size = 0;
  switch (out.dtype) {
    case NpyArray::DType::F4: elem_size = 4; break;
    case NpyArray::DType::F8: elem_size = 8; break;
    case NpyArray::DType::I8: elem_size = 8; break;
    case NpyArray::DType::U1: elem_size = 1; break;
  }
  out.bytes.resize(out.count() * elem_size);
  in.read(reinterpret_cast<char *>(out.bytes.data()),
    static_cast<std::streamsize>(out.bytes.size()));
  if (!in) {
    error = "truncated npy data: " + path;
    return false;
  }
  return true;
}

std::vector<double> as_doubles(const NpyArray & a)
{
  const size_t n = a.count();
  std::vector<double> out(n);
  if (a.dtype == NpyArray::DType::F8) {
    std::memcpy(out.data(), a.bytes.data(), n * sizeof(double));
  } else {
    const auto * src = reinterpret_cast<const float *>(a.bytes.data());
    for (size_t i = 0; i < n; ++i) {
      out[i] = static_cast<double>(src[i]);
    }
  }
  return out;
}

std::vector<int64_t> as_int64(const NpyArray & a)
{
  const size_t n = a.count();
  std::vector<int64_t> out(n);
  std::memcpy(out.data(), a.bytes.data(), n * sizeof(int64_t));
  return out;
}

std::vector<uint8_t> as_u8(const NpyArray & a)
{
  return a.bytes;
}

bool expect(const NpyArray & a, NpyArray::DType dtype, std::vector<int64_t> shape,
  const char * name, std::string & error)
{
  if (a.dtype != dtype) {
    error = std::string(name) + ": wrong dtype";
    return false;
  }
  if (a.shape != shape) {
    std::stringstream want, got;
    want << "(";
    got << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
      want << (i ? "," : "") << shape[i];
    }
    for (size_t i = 0; i < a.shape.size(); ++i) {
      got << (i ? "," : "") << a.shape[i];
    }
    want << ")";
    got << ")";
    error = std::string(name) + ": shape " + got.str() + " != " + want.str();
    return false;
  }
  return true;
}

}  // namespace

bool load_map_v2(const std::string & dir, MapV2 & out, std::string & error)
{
  out = MapV2{};
  NpyArray arr;

  if (!read_npy(dir + "/pose_timestamps.npy", arr, error) ||
    arr.dtype != NpyArray::DType::I8 || arr.shape.size() != 1)
  {
    error = "pose_timestamps.npy: " + error;
    return false;
  }
  out.timestamps = as_int64(arr);
  const int64_t n = static_cast<int64_t>(out.timestamps.size());
  if (n == 0) {
    error = "map holds no keyframes";
    return false;
  }

  if (!read_npy(dir + "/pose_matrices.npy", arr, error) ||
    !expect(arr, NpyArray::DType::F8, {n, 4, 4}, "pose_matrices.npy", error))
  {
    return false;
  }
  {
    const std::vector<double> data = as_doubles(arr);
    for (int64_t i = 0; i < n; ++i) {
      Eigen::Matrix4d pose;
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          pose(r, c) = data[static_cast<size_t>((i * 4 + r) * 4 + c)];
        }
      }
      out.poses[out.timestamps[static_cast<size_t>(i)]] = pose;
    }
  }

  if (!read_npy(dir + "/vlad_centres.npy", arr, error) || arr.shape.size() != 2 ||
    !(arr.dtype == NpyArray::DType::F4 || arr.dtype == NpyArray::DType::F8))
  {
    error = "vlad_centres.npy: " + error;
    return false;
  }
  {
    const std::vector<double> data = as_doubles(arr);
    const int64_t c = arr.shape[0], d = arr.shape[1];
    out.vlad_centres.resize(c, d);
    for (int64_t r = 0; r < c; ++r) {
      for (int64_t col = 0; col < d; ++col) {
        out.vlad_centres(r, col) = data[static_cast<size_t>(r * d + col)];
      }
    }
  }
  if (!read_npy(dir + "/vlad_descriptors.npy", arr, error) || arr.shape.size() != 2 ||
    arr.shape[0] != n ||
    !(arr.dtype == NpyArray::DType::F4 || arr.dtype == NpyArray::DType::F8))
  {
    error = "vlad_descriptors.npy: " + error;
    return false;
  }
  {
    const std::vector<double> data = as_doubles(arr);
    const int64_t d = arr.shape[1];
    out.vlad_descriptors.resize(n, d);
    for (int64_t r = 0; r < n; ++r) {
      for (int64_t col = 0; col < d; ++col) {
        out.vlad_descriptors(r, col) = data[static_cast<size_t>(r * d + col)];
      }
    }
  }

  if (!read_npy(dir + "/feature_offsets.npy", arr, error) ||
    !expect(arr, NpyArray::DType::I8, {n + 1}, "feature_offsets.npy", error))
  {
    return false;
  }
  const std::vector<int64_t> offsets = as_int64(arr);

  if (!read_npy(dir + "/feature_kpts.npy", arr, error) || arr.shape.size() != 2 ||
    arr.shape[1] != 2 || arr.dtype != NpyArray::DType::F4)
  {
    error = "feature_kpts.npy: " + error;
    return false;
  }
  const int64_t m = arr.shape[0];
  const std::vector<float> kpts = [&] {
    std::vector<float> v(m * 2);
    std::memcpy(v.data(), arr.bytes.data(), arr.bytes.size());
    return v;
  }();
  if (!read_npy(dir + "/feature_descps.npy", arr, error) || arr.shape.size() != 2 ||
    arr.shape[0] != m || arr.dtype != NpyArray::DType::F4)
  {
    error = "feature_descps.npy: " + error;
    return false;
  }
  const int64_t desc_dim = arr.shape[1];
  std::vector<float> descps(arr.count());
  std::memcpy(descps.data(), arr.bytes.data(), arr.bytes.size());
  if (!read_npy(dir + "/feature_mask.npy", arr, error) ||
    !expect(arr, NpyArray::DType::U1, {m}, "feature_mask.npy", error))
  {
    return false;
  }
  const std::vector<uint8_t> mask = as_u8(arr);

  for (int64_t i = 0; i < n; ++i) {
    const int64_t begin = offsets[static_cast<size_t>(i)];
    const int64_t end = offsets[static_cast<size_t>(i) + 1];
    if (begin < 0 || end < begin || end > m) {
      error = "feature_offsets inconsistent";
      return false;
    }
    const int rows = static_cast<int>(end - begin);
    MapV2Features feat;
    if (rows > 0) {
      const int sz_k[3] = {1, rows, 2};
      feat.kpts = cv::Mat(3, sz_k, CV_32F);
      std::memcpy(feat.kpts.data, kpts.data() + begin * 2,
        static_cast<size_t>(rows) * 2 * sizeof(float));
      const int sz_d[3] = {1, rows, static_cast<int>(desc_dim)};
      feat.descps = cv::Mat(3, sz_d, CV_32F);
      std::memcpy(feat.descps.data, descps.data() + begin * desc_dim,
        static_cast<size_t>(rows) * desc_dim * sizeof(float));
      const int sz_m[3] = {1, rows, 1};
      feat.mask = cv::Mat(3, sz_m, CV_8U);
      std::memcpy(feat.mask.data, mask.data() + begin,
        static_cast<size_t>(rows));
    }
    out.features[out.timestamps[static_cast<size_t>(i)]] = std::move(feat);
  }

  if (!read_npy(dir + "/depth_images.npy", arr, error) || arr.shape.size() != 3 ||
    arr.shape[0] != n || arr.dtype != NpyArray::DType::F4)
  {
    error = "depth_images.npy: " + error;
    return false;
  }
  {
    const int64_t h = arr.shape[1], w = arr.shape[2];
    const size_t frame_bytes = static_cast<size_t>(h * w) * sizeof(float);
    for (int64_t i = 0; i < n; ++i) {
      cv::Mat depth(static_cast<int>(h), static_cast<int>(w), CV_32F);
      std::memcpy(depth.data, arr.bytes.data() + static_cast<size_t>(i) * frame_bytes,
        frame_bytes);
      out.depth[out.timestamps[static_cast<size_t>(i)]] = std::move(depth);
    }
  }
  return true;
}

std::pair<Eigen::MatrixX3d, std::vector<bool>> keypoint_with_depth_to_3d(
  const Eigen::MatrixX2d & keypoints, const cv::Mat & depth,
  const Eigen::Matrix4d & pose_from_camera_to_world, const Eigen::Matrix3d & K)
{
  const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
  Eigen::MatrixX3d points_in_world(keypoints.rows(), 3);
  std::vector<bool> inliers;
  inliers.reserve(keypoints.rows());
  for (int i = 0; i < keypoints.rows(); ++i) {
    // python int() truncates toward zero — same as the static_cast.
    const int u = static_cast<int>(keypoints(i, 0));
    const int v = static_cast<int>(keypoints(i, 1));
    double x = 0, y = 0, z = 0;
    bool ok = false;
    if (!depth.empty() && v >= 0 && v < depth.rows && u >= 0 && u < depth.cols) {
      z = depth.at<float>(v, u);
      if (z > 0 && z < 50) {
        x = (u - cx) * z / fx;
        y = (v - cy) * z / fy;
        ok = true;
      }
    }
    Eigen::Vector3d point_in_camera(x, y, z);
    points_in_world.row(i) =
      (pose_from_camera_to_world.topLeftCorner<3, 3>() * point_in_camera).transpose() +
      pose_from_camera_to_world.topRightCorner<3, 1>().transpose();
    inliers.push_back(ok);
  }
  return {points_in_world, inliers};
}

}  // namespace tinynav::mapping
