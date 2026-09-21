// Probe: run C++ LightGlueTRT on one fixed keypoint pair (raw f32 dumps),
// print match counts for both img_shape conventions (python hardcodes
// [848,480]; the sim rig once passed the live size [544,480] — that
// difference was the 2026-09-19 reloc root cause).
//
// Input layout is what the mapping component's dump channel
// (TINYNAV_RELOC_DUMP_DIR, see mapping_component.cpp) and probe_sp write:
// kpts [1,N,2] f32, descps [1,N,256] f32, mask [1,N,1] f32, map_* / live_*.
// N defaults to 512 (the SuperPoint dynamic-engine output size) and can be
// overridden: probe_lg <dir> [N].
#include <cstdio>
#include <fstream>
#include <array>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <tinynav_cpp/trt/trt_engine.hpp>
#include "tinynav_cpp/trt/models.hpp"

static cv::Mat load_f32(const std::string& path, int d0, int d1, int d2) {
  std::ifstream in(path, std::ios::binary);
  if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  cv::Mat m(3, std::vector<int>{d0, d1, d2}.data(), CV_32F);
  in.read(reinterpret_cast<char*>(m.data), static_cast<size_t>(d0) * d1 * d2 * 4);
  return m;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: probe_lg <dir> [N]\n"); return 1; }
  const std::string dir = argv[1];
  const int n = argc > 2 ? std::atoi(argv[2]) : 512;
  const cv::Mat mk = load_f32(dir + "/map_kpts.f32", 1, n, 2);
  const cv::Mat md = load_f32(dir + "/map_descps.f32", 1, n, 256);
  const cv::Mat mm = load_f32(dir + "/map_mask.f32", 1, n, 1);
  const cv::Mat lk = load_f32(dir + "/live_kpts.f32", 1, n, 2);
  const cv::Mat ld = load_f32(dir + "/live_descps.f32", 1, n, 256);
  const cv::Mat lm = load_f32(dir + "/live_mask.f32", 1, n, 1);

  tinynav::trt::LightGlueTRT lg("/tinynav/tinynav/models");
  for (const auto s : {std::array<int64_t, 2>{848, 480}, std::array<int64_t, 2>{544, 480}}) {
    tinynav::trt::TrtOutputMap out;
    if (!lg.infer(mk, lk, md, ld, mm, lm, s, s, out)) {
      fprintf(stderr, "infer failed\n");
      return 1;
    }
    const cv::Mat& mi = out.at("match_indices");
    int count = 0;
    for (int i = 0; i < mi.size[1]; ++i) {
      const long long v = mi.type() == CV_32S
                              ? mi.at<int32_t>(0, i)
                              : static_cast<long long>(mi.at<double>(0, i));
      if (v != -1) ++count;
    }
    printf("img_shape=[%ld,%ld] matches=%d\n", static_cast<long>(s[0]),
           static_cast<long>(s[1]), count);
  }
  return 0;
}
