// Standalone probe: load a map-format-v2 directory through the C++ reader
// (libmapping_component's load_map_v2) and sanity-check the consumption
// surface. Verifies the builder-written map (incl. the semantic sidecar
// files sitting in the same dir) loads with the unmodified reader.
//
//   g++ -std=c++17 -O2 -I src/tinynav_cpp/include -I /usr/include/opencv4 \
//       tools/probes/probe_map_v2.cpp -o tools/probes/probe_map_v2 \
//       -L build/tinynav_cpp -lmapping_component -lopencv_core
//   ./tools/probes/probe_map_v2 <map_v2_dir>
#include <cstdio>
#include <string>

#include "tinynav_cpp/mapping/map_v2.hpp"

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::fprintf(stderr, "usage: probe_map_v2 <map_v2_dir>\n");
    return 2;
  }
  tinynav::mapping::MapV2 map;
  std::string error;
  if (!tinynav::mapping::load_map_v2(argv[1], map, error)) {
    std::fprintf(stderr, "LOAD FAIL: %s\n", error.c_str());
    return 1;
  }
  std::printf("LOAD OK: %zu keyframes, %zu poses, vlad %zux%zu, centres %zux%zu\n",
    map.timestamps.size(), map.poses.size(),
    map.vlad_descriptors.rows(), map.vlad_descriptors.cols(),
    map.vlad_centres.rows(), map.vlad_centres.cols());
  int checked = 0;
  for (size_t i = 0; i < map.timestamps.size() && checked < 5; i += 7, ++checked) {
    const auto ts = map.timestamps[i];
    tinynav::mapping::MapV2Features feats;
    const bool has_feats = map.get_features(ts, feats);
    const cv::Mat depth = map.get_depth(ts);
    std::printf("  ts=%lld has_frame=%d feats=%dx%d depth=%dx%d\n",
      static_cast<long long>(ts), map.has_frame(ts),
      has_feats ? feats.kpts.size[1] : 0, has_feats ? feats.descps.size[2] : 0,
      depth.rows, depth.cols);
  }
  return 0;
}
