// Probe: does concurrent in-process TRT execution degrade LightGlue results?
// Thread A runs the fixed pair N times (sequential baseline, then while a
// hammer thread drives a second LightGlueTRT instance).
#include <atomic>
#include <cstdio>
#include <fstream>
#include <thread>
#include <array>
#include <string>
#include <opencv2/core.hpp>
#include <tinynav_cpp/trt/trt_engine.hpp>
#include <tinynav_cpp/trt/models.hpp>

static cv::Mat load_f32(const std::string& path, int d0, int d1, int d2) {
  std::ifstream in(path, std::ios::binary);
  cv::Mat m(3, std::vector<int>{d0, d1, d2}.data(), CV_32F);
  in.read(reinterpret_cast<char*>(m.data), static_cast<size_t>(d0) * d1 * d2 * 4);
  return m;
}
static int count_matches(tinynav::trt::LightGlueTRT& lg, const cv::Mat* mk,
                         const std::array<int64_t, 2>& s) {
  tinynav::trt::TrtOutputMap out;
  if (!lg.infer(mk[0], mk[1], mk[2], mk[3], mk[4], mk[5], s, s, out)) return -1;
  const cv::Mat& mi = out.at("match_indices");
  int n = 0;
  for (int i = 0; i < mi.size[1]; ++i) {
    const bool ok = mi.type() == CV_32S ? mi.at<int32_t>(0, i) != -1
                                        : mi.at<double>(0, i) != -1;
    if (ok) ++n;
  }
  return n;
}
int main(int argc, char** argv) {
  const std::string dir = argv[1];
  const cv::Mat mk[6] = {load_f32(dir + "/map_kpts.f32", 1, 512, 2),
                         load_f32(dir + "/live_kpts.f32", 1, 512, 2),
                         load_f32(dir + "/map_descps.f32", 1, 512, 256),
                         load_f32(dir + "/live_descps.f32", 1, 512, 256),
                         load_f32(dir + "/map_mask.f32", 1, 512, 1),
                         load_f32(dir + "/live_mask.f32", 1, 512, 1)};
  const std::array<int64_t, 2> s{544, 480};
  tinynav::trt::LightGlueTRT lg_main("/tinynav/tinynav/models");
  printf("sequential:");
  for (int i = 0; i < 5; ++i) printf(" %d", count_matches(lg_main, mk, s));
  printf("\n");
  std::atomic<bool> stop{false};
  tinynav::trt::LightGlueTRT lg_hammer("/tinynav/tinynav/models");
  std::thread hammer([&] {
    while (!stop.load(std::memory_order_relaxed)) count_matches(lg_hammer, mk, s);
  });
  printf("concurrent :");
  for (int i = 0; i < 10; ++i) printf(" %d", count_matches(lg_main, mk, s));
  stop.store(true);
  hammer.join();
  printf("\n");
  return 0;
}
