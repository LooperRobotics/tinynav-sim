// Standalone probe: load a map-format-v3 sqlite file through the C++ reader
// (libmapping_component's load_map_v3) and sanity-check the consumption
// surface — same statistics as probe_map_v2 for a side-by-side comparison.
//
//   g++ -std=c++17 -O2 -I src/tinynav_cpp/include -I /usr/include/opencv4 \
//       -I /usr/include/eigen3 tools/probes/probe_map_v3.cpp -o tools/probes/probe_map_v3 \
//       -L build/tinynav_cpp -Wl,--whole-archive -ltinynav_core -Wl,--no-whole-archive \
//       -lmapping_component -lsqlite3 -lceres -lglog -lopencv_core -lopencv_calib3d
//   ./tools/probes/probe_map_v3 <map_v3.sqlite>
// (tinynav_core is linked whole-archive: nothing inside the mapping component
// references load_map_v3 yet, so a plain archive link would drop map_v3.o)
#include <cstdio>
#include <string>

#include "tinynav_cpp/mapping/map_v3.hpp"

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::fprintf(stderr, "usage: probe_map_v3 <map_v3.sqlite>\n");
    return 2;
  }
  tinynav::mapping::MapV3 map;
  std::string error;
  if (!tinynav::mapping::load_map_v3(argv[1], map, error)) {
    std::fprintf(stderr, "LOAD FAIL: %s\n", error.c_str());
    return 1;
  }
  std::printf("LOAD OK: %zu keyframes, %zu poses, vlad %zux%zu, centres %zux%zu, "
    "semantic %zux%zu\n",
    map.timestamps.size(), map.poses.size(),
    map.vlad_descriptors.rows(), map.vlad_descriptors.cols(),
    map.vlad_centres.rows(), map.vlad_centres.cols(),
    map.semantic_embeddings.rows(), map.semantic_embeddings.cols());
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
