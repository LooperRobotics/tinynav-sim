// Probe: is SuperPointTRT's output bit-stable when TRT engines execute
// concurrently in-process? Main thread runs SP on a fixed image and hashes
// descps; a hammer thread drives a LightGlueTRT instance concurrently.
#include <atomic>
#include <cstdio>
#include <fstream>
#include <thread>
#include <array>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tinynav_cpp/trt/trt_engine.hpp>
#include <tinynav_cpp/trt/models.hpp>

static std::string hash_bytes(const char* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) { h ^= static_cast<unsigned char>(p[i]); h *= 1099511628211ull; }
  char buf[32];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}
int main(int argc, char** argv) {
  const std::string img_path = argv[1];
  cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  if (img.empty()) { fprintf(stderr, "cannot read image\n"); return 1; }

  tinynav::trt::SuperPointTRT sp_main("/tinynav/tinynav/models");
  std::string seq_hash;
  {
    tinynav::trt::TrtOutputMap r;
    if (!sp_main.infer(img, r)) { fprintf(stderr, "sp infer failed\n"); return 1; }
    const cv::Mat& d = r.at("descps");
    seq_hash = hash_bytes(reinterpret_cast<const char*>(d.data), d.total() * d.elemSize());
    printf("sequential descps hash: %s\n", seq_hash.c_str());
  }
  // repeat sequential a few times: engine determinism alone
  for (int i = 0; i < 3; ++i) {
    tinynav::trt::TrtOutputMap r;
    sp_main.infer(img, r);
    const cv::Mat& d = r.at("descps");
    printf("  seq repeat %d: %s\n", i,
           (hash_bytes(reinterpret_cast<const char*>(d.data), d.total() * d.elemSize()) == seq_hash)
               ? "same" : "DIFFERENT");
  }
  std::atomic<bool> stop{false};
  tinynav::trt::LightGlueTRT lg_hammer("/tinynav/tinynav/models");
  std::thread hammer([&] {
    // need a valid pair to chew on; reuse the dumped inputs if present
    const std::string dir = argc > 2 ? argv[2] : "";
    if (dir.empty()) { return; }
    const auto load = [&](const std::string& n, int d0, int d1, int d2) {
      std::ifstream in(dir + "/" + n, std::ios::binary);
      cv::Mat m(3, std::vector<int>{d0, d1, d2}.data(), CV_32F);
      in.read(reinterpret_cast<char*>(m.data), (size_t)d0 * d1 * d2 * 4);
      return m;
    };
    const cv::Mat mk = load("map_kpts.f32", 1, 512, 2), lk = load("live_kpts.f32", 1, 512, 2);
    const cv::Mat md = load("map_descps.f32", 1, 512, 256), ld = load("live_descps.f32", 1, 512, 256);
    const cv::Mat mm = load("map_mask.f32", 1, 512, 1), lm = load("live_mask.f32", 1, 512, 1);
    const std::array<int64_t, 2> s{544, 480};
    tinynav::trt::TrtOutputMap out;
    while (!stop.load(std::memory_order_relaxed)) {
      if (!lg_hammer.infer(mk, lk, md, ld, mm, lm, s, s, out)) break;
    }
  });
  printf("concurrent SP runs:");
  for (int i = 0; i < 10; ++i) {
    tinynav::trt::TrtOutputMap r;
    if (!sp_main.infer(img, r)) { printf(" FAIL"); continue; }
    const cv::Mat& d = r.at("descps");
    const std::string h = hash_bytes(reinterpret_cast<const char*>(d.data), d.total() * d.elemSize());
    printf(" %s", h == seq_hash ? "." : "DIFF");
  }
  stop.store(true);
  hammer.join();
  printf("\n");
  return 0;
}
