#ifndef TINYNAV_CPP__MAPPING__MAP_V2_HPP_
#define TINYNAV_CPP__MAPPING__MAP_V2_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "tinynav_cpp/mapping/mapped_npy.hpp"

namespace tinynav::mapping
{

// One map keyframe's SuperPoint output in the internal cv::Mat layout the
// component's match_keypoints consumes: kpts [1,N,2] CV_32F, descps [1,N,D]
// CV_32F, mask [1,N,1] CV_8U.
struct MapV2Features
{
  cv::Mat kpts;
  cv::Mat descps;
  cv::Mat mask;
};

// Row-major f32: memcpy'd straight from the npy and scanned row-wise per
// reloc query. f32 on purpose — python keeps the index in f32, and dog-scale
// (9.6k keyframes x 24576 dims) is 0.94G here vs 1.9G in f64; the Orin Nano
// has 8G total. Query vectors cast to f32 once before the dot loop.
using VladIndex = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Map format v2: exactly what map_node.py::load_map reads through
// np.load(allow_pickle=True) and the TinyNavDB shelve, flattened to plain
// C-order .npy files a C++ reader can load without pickle/shelve
// (writer: tools/export_map_v2.py). Descriptors row i corresponds to
// timestamps[i].
//
// Memory model mirrors the Python split: the retrieval index + poses are
// small and eager; the big per-keyframe arrays (depth planes, SuperPoint
// features) are mmap views touched per candidate keyframe — python reads
// them from the shelve on demand, C++ maps the flat npys on demand. A
// yishang-scale map (9.6k keyframes, ~13G of depth) loads in tens of MB of
// RSS instead of materializing the whole index. The get_* views borrow the
// mapping — the MapV2 owner must outlive them (true at the only call site,
// where they die at the end of the reloc attempt).
struct MapV2
{
  std::vector<int64_t> timestamps;
  std::unordered_map<int64_t, Eigen::Matrix4d> poses;
  Eigen::MatrixXd vlad_centres;      // C x d
  VladIndex vlad_descriptors;        // N x d f32 row-major

  bool has_frame(int64_t ts) const;
  // Non-owning view into the feature npys; false when ts is unknown.
  bool get_features(int64_t ts, MapV2Features & out) const;
  // Non-owning CV_32F HxW view; empty Mat when ts is unknown.
  cv::Mat get_depth(int64_t ts) const;

  std::unordered_map<int64_t, uint32_t> row_of_;
  std::vector<int64_t> feature_offsets_;
  MappedNpy feature_kpts_, feature_descps_, feature_mask_;
  MappedNpy depth_images_;
  int64_t desc_dim_ = 0;
};

// Load a map-format-v2 directory. Returns false with `error` set on any
// missing/unreadable piece — the caller degrades relocalization, never crashes.
bool load_map_v2(const std::string & dir, MapV2 & out, std::string & error);

// Port of reference/tinynav/core/map_node.py::keypoint_with_depth_to_3d.
// Unprojects matched map keypoints through the map keyframe's depth into world
// points; second output is the per-keypoint validity mask (0 < Z < 50). Unlike
// the Python (which would IndexError), out-of-bounds pixels read as invalid.
std::pair<Eigen::MatrixX3d, std::vector<bool>> keypoint_with_depth_to_3d(
  const Eigen::MatrixX2d & keypoints, const cv::Mat & depth,
  const Eigen::Matrix4d & pose_from_camera_to_world, const Eigen::Matrix3d & K);

}  // namespace tinynav::mapping

#endif  // TINYNAV_CPP__MAPPING__MAP_V2_HPP_
