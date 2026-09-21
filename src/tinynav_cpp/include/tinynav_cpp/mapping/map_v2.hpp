#ifndef TINYNAV_CPP__MAPPING__MAP_V2_HPP_
#define TINYNAV_CPP__MAPPING__MAP_V2_HPP_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

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

// Map format v2: exactly what map_node.py::load_map reads through
// np.load(allow_pickle=True) and the TinyNavDB shelve, flattened to plain
// C-order .npy files a C++ reader can load without pickle/shelve
// (writer: tools/export_map_v2.py). Descriptors row i corresponds to
// timestamps[i].
struct MapV2
{
  std::vector<int64_t> timestamps;
  std::unordered_map<int64_t, Eigen::Matrix4d> poses;
  Eigen::MatrixXd vlad_centres;      // C x d
  Eigen::MatrixXd vlad_descriptors;  // N x d
  std::unordered_map<int64_t, MapV2Features> features;
  std::unordered_map<int64_t, cv::Mat> depth;  // CV_32F HxW
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
