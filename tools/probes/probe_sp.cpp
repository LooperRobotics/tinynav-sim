// Probe: run C++ SuperPointTRT on one image, dump kpts/descps/mask as .npy
// for cross-wrapper comparison with reference/tinynav/core/models_trt.py.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tinynav_cpp/trt/models.hpp"

static bool save_npy_f32(const std::string& path, const cv::Mat& m) {
  // minimal npy v1.0, little-endian f32, C-order, up to 3 dims
  cv::Mat f;
  m.convertTo(f, CV_32F);
  std::vector<int> shape;
  for (int i = 0; i < f.dims; ++i) shape.push_back(f.size[i]);
  if (shape.empty()) shape.push_back(1);
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    dict += std::to_string(shape[i]);
    if (i + 1 < shape.size()) dict += ", ";
  }
  if (shape.size() == 1) dict += ",";
  dict += "), }";
  size_t header_len = 10 + dict.size() + 1;
  size_t pad = (64 - header_len % 64) % 64;
  dict.append(pad, ' ');
  dict += "\n";
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(magic, 8);
  uint16_t hlen = static_cast<uint16_t>(dict.size());
  out.write(reinterpret_cast<const char*>(&hlen), 2);
  out.write(dict.data(), dict.size());
  // cv::Mat data is contiguous for continuous mats
  cv::Mat c = f.isContinuous() ? f : f.clone();
  out.write(reinterpret_cast<const char*>(c.ptr()), c.total() * sizeof(float));
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: probe_sp <image> <out_dir>\n"); return 1; }
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  if (img.empty()) { fprintf(stderr, "cannot read image %s\n", argv[1]); return 1; }
  fprintf(stderr, "image %dx%d\n", img.cols, img.rows);

  tinynav::trt::SuperPointTRT sp;
  tinynav::trt::TrtOutputMap res;
  if (!sp.infer(img, res, 5e-4f)) { fprintf(stderr, "C++ SuperPointTRT unavailable\n"); return 2; }

  std::string dir = argv[2];
  save_npy_f32(dir + "/cpp_kpts.npy", res.at("kpts"));
  save_npy_f32(dir + "/cpp_descps.npy", res.at("descps"));
  save_npy_f32(dir + "/cpp_mask.npy", res.at("mask"));
  fprintf(stderr, "kpts dims=%d shape=[", res.at("kpts").dims);
  for (int i = 0; i < res.at("kpts").dims; ++i) fprintf(stderr, "%d%s", res.at("kpts").size[i], i + 1 < res.at("kpts").dims ? "," : "");
  fprintf(stderr, "] descps dims=%d shape=[", res.at("descps").dims);
  for (int i = 0; i < res.at("descps").dims; ++i) fprintf(stderr, "%d%s", res.at("descps").size[i], i + 1 < res.at("descps").dims ? "," : "");
  fprintf(stderr, "]\n");
  return 0;
}
