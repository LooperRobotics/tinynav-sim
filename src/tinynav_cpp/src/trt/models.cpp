// Port of reference/tinynav/core/models_trt.py — the model wrapper classes.
#include "tinynav_cpp/trt/models.hpp"

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#if defined(__aarch64__)
#define TINYNAV_TRT_MAPPED_HOST 1
#else
#define TINYNAV_TRT_MAPPED_HOST 0
#endif

namespace tinynav::trt {
namespace {

// Same element reader as trt_engine.cpp (anonymous-namespace twin).
double mat_elem_as_double(const cv::Mat& m, size_t i) {
  switch (m.depth()) {
    case CV_8U: return m.data[i];
    case CV_8S: return reinterpret_cast<const int8_t*>(m.data)[i];
    case CV_16U: return reinterpret_cast<const uint16_t*>(m.data)[i];
    case CV_16S: return reinterpret_cast<const int16_t*>(m.data)[i];
    case CV_32S: return reinterpret_cast<const int32_t*>(m.data)[i];
    case CV_32F: return reinterpret_cast<const float*>(m.data)[i];
    case CV_64F: return reinterpret_cast<const double*>(m.data)[i];
    case CV_16F: {
      // Minimal half->float for reading engine-side fp16 tensors.
      uint16_t h = reinterpret_cast<const uint16_t*>(m.data)[i];
      uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
      uint32_t exp = (h >> 10) & 0x1fu;
      uint32_t mant = h & 0x3ffu;
      uint32_t x;
      if (exp == 0) {
        if (mant == 0) {
          x = sign;
        } else {
          exp = 127 - 15 + 1;
          while (!(mant & 0x400u)) {
            mant <<= 1;
            --exp;
          }
          mant &= 0x3ffu;
          x = sign | (exp << 23) | (mant << 13);
        }
      } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
      } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
      }
      float f;
      std::memcpy(&f, &x, 4);
      return f;
    }
    default: return 0.0;
  }
}

// Same-dims CV_32F copy of any-depth single-channel mat (cv::Mat::convertTo
// is not guaranteed to handle CV_16F in OpenCV 4.5).
void to_float32(const cv::Mat& m, cv::Mat& out) {
  cv::Mat cont = m.isContinuous() ? m : m.clone();
  std::vector<int> sizes(cont.dims);
  for (int i = 0; i < cont.dims; ++i) sizes[i] = cont.size[i];
  out.create(cont.dims, sizes.data(), CV_32F);
  const size_t n = cont.total();
  for (size_t i = 0; i < n; ++i) {
    reinterpret_cast<float*>(out.data)[i] =
        static_cast<float>(mat_elem_as_double(cont, i));
  }
}

bool flatten_to_float(const cv::Mat& m, std::vector<float>& out) {
  if (m.empty()) return false;
  cv::Mat cont = m.isContinuous() ? m : m.clone();
  out.resize(cont.total());
  for (size_t i = 0; i < cont.total(); ++i) {
    out[i] = static_cast<float>(mat_elem_as_double(cont, i));
  }
  return true;
}

// HWC 3-channel -> planar [1,3,h,w] CV_32F with (v/divisor - mean)/std.
cv::Mat to_planar_chw(const cv::Mat& img3, const double mean[3], const double std_[3],
                      double divisor) {
  cv::Mat f;
  img3.convertTo(f, CV_32F);
  const int h = f.rows;
  const int w = f.cols;
  int sizes[4] = {1, 3, h, w};
  cv::Mat out(4, sizes, CV_32F);
  float* dst = reinterpret_cast<float*>(out.data);
  for (int c = 0; c < 3; ++c) {
    const double inv = 1.0 / (divisor * std_[c]);
    const double shift = mean[c] / std_[c];
    for (int y = 0; y < h; ++y) {
      const cv::Vec3f* row = f.ptr<cv::Vec3f>(y);
      float* drow = dst + static_cast<size_t>(c) * h * w + static_cast<size_t>(y) * w;
      for (int x = 0; x < w; ++x) drow[x] = static_cast<float>(row[x][c] * inv - shift);
    }
  }
  return out;
}

// L2-normalize rows in place: row /= max(norm, 1e-8) (np.maximum semantics).
void l2_normalize_rows(cv::Mat& m) {
  for (int r = 0; r < m.rows; ++r) {
    cv::Mat row = m.row(r);
    const double n = std::max(cv::norm(row), 1e-8);
    row /= n;
  }
}

}  // namespace

// ---------------------------------------------------------------- SuperPoint
// Port of reference/tinynav/core/models_trt.py::SuperPointTRT.

SuperPointTRT::SuperPointTRT(std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "superpoint_fp16_dynamic", arch)) {}

bool SuperPointTRT::on_loaded(std::string& err) {
  // engine input [1,1,H,W] + threshold input
  if (inputs_.size() < 2 || inputs_[0]->shape.size() != 4) {
    err = "SuperPoint engine must have image input [1,1,H,W] + threshold input";
    return false;
  }
  net_h_ = static_cast<int>(inputs_[0]->shape[2]);
  net_w_ = static_cast<int>(inputs_[0]->shape[3]);
  return true;
}

bool SuperPointTRT::infer(const cv::Mat& image, TrtOutputMap& results, float threshold) {
  if (!ensure_loaded()) return false;
  if (image.empty() || image.channels() != 1) {
    set_error("SuperPointTRT::infer expects a single-channel image");
    return false;
  }
  const int h_in = image.rows;
  const int w_in = image.cols;

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(net_w_, net_h_));
  if (!copy_to_input(0, resized)) return false;
  cv::Mat thr(1, 1, CV_32F, cv::Scalar(threshold));
  if (!copy_to_input(1, thr)) return false;

  if (!run_graph(results)) return false;

  // Scale keypoints from network coords back to input image coords, per axis.
  auto it = results.find("kpts");
  if (it == results.end()) {
    set_error("SuperPoint engine output has no 'kpts' tensor");
    return false;
  }
  cv::Mat kf;
  to_float32(it->second, kf);
  const double scale_x = static_cast<double>(w_in) / net_w_;
  const double scale_y = static_cast<double>(h_in) / net_h_;
  int rows = 0;
  int cols = 0;
  float* plane = reinterpret_cast<float*>(kf.data);
  if (kf.dims == 3) {
    rows = kf.size[1];
    cols = kf.size[2];
  } else if (kf.dims == 2) {
    rows = kf.size[0];
    cols = kf.size[1];
  } else {
    set_error("unexpected kpts rank");
    return false;
  }
  if (rows == 2) {  // [2,N]: row 0 = x, row 1 = y
    for (int i = 0; i < cols; ++i) {
      plane[i] = static_cast<float>((plane[i] + 0.5) * scale_x - 0.5);
      plane[cols + i] = static_cast<float>((plane[cols + i] + 0.5) * scale_y - 0.5);
    }
  } else if (cols == 2) {  // [N,2]
    for (int i = 0; i < rows; ++i) {
      plane[2 * i] = static_cast<float>((plane[2 * i] + 0.5) * scale_x - 0.5);
      plane[2 * i + 1] = static_cast<float>((plane[2 * i + 1] + 0.5) * scale_y - 0.5);
    }
  }
  results["kpts"] = kf;

  auto mit = results.find("mask");
  if (mit != results.end() && mit->second.dims == 2) {
    const int sz[3] = {mit->second.size[0], mit->second.size[1], 1};
    mit->second = mit->second.reshape(1, 3, sz);
  }
  return true;
}

// ---------------------------------------------------------------- LightGlue
// Port of reference/tinynav/core/models_trt.py::LightGlueTRT.

LightGlueTRT::LightGlueTRT(std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "lightglue_fp16", arch)) {}

bool LightGlueTRT::infer(const cv::Mat& kpts0, const cv::Mat& kpts1, const cv::Mat& desc0,
                         const cv::Mat& desc1, const cv::Mat& mask0, const cv::Mat& mask1,
                         const std::array<int64_t, 2>& img_shape0,
                         const std::array<int64_t, 2>& img_shape1, TrtOutputMap& results,
                         float match_threshold) {
  if (!ensure_loaded()) return false;
  const cv::Mat* srcs[6] = {&kpts0, &kpts1, &desc0, &desc1, &mask0, &mask1};
  for (size_t i = 0; i < 6; ++i) {
    if (!copy_to_input(i, *srcs[i])) return false;
  }
  double shapes[4] = {static_cast<double>(img_shape0[0]), static_cast<double>(img_shape0[1]),
                      static_cast<double>(img_shape1[0]), static_cast<double>(img_shape1[1])};
  cv::Mat shape0(2, 1, CV_64F, &shapes[0]);
  cv::Mat shape1(2, 1, CV_64F, &shapes[2]);
  if (!copy_to_input(6, shape0)) return false;
  if (!copy_to_input(7, shape1)) return false;
  cv::Mat thr(1, 1, CV_32F, cv::Scalar(match_threshold));
  if (!copy_to_input(8, thr)) return false;
  return run_graph(results);
}

// ------------------------------------------------------------------ DINOv2
// Port of reference/tinynav/core/models_trt.py::Dinov2TRT.

Dinov2TRT::Dinov2TRT(std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "dinov2_base_224x224_fp16", arch)) {}

cv::Mat Dinov2TRT::preprocess_image(const cv::Mat& image, int target_size) {
  if (image.empty() || image.channels() != 1) {
    std::fprintf(stderr, "[tinynav::trt] Dinov2TRT::preprocess_image: bad input\n");
    return {};
  }
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  cv::resize(rgb, rgb, cv::Size(target_size, target_size), 0, 0, cv::INTER_CUBIC);
  const double mean[3] = {0.485, 0.456, 0.406};
  const double std_[3] = {0.229, 0.224, 0.225};
  return to_planar_chw(rgb, mean, std_, 255.0);
}

bool Dinov2TRT::run_hidden_state(const cv::Mat& image, cv::Mat& hidden_f32) {
  if (!ensure_loaded()) return false;
  cv::Mat in = preprocess_image(image);
  if (in.empty() || !copy_to_input(0, in)) return false;
  TrtOutputMap results;
  if (!run_graph(results)) return false;
  auto it = results.find("last_hidden_state");
  if (it == results.end()) {
    set_error("DINOv2 engine output has no 'last_hidden_state' tensor");
    return false;
  }
  to_float32(it->second, hidden_f32);
  if (hidden_f32.dims != 3 || hidden_f32.size[0] != 1) {
    set_error("unexpected last_hidden_state shape");
    return false;
  }
  return true;
}

bool Dinov2TRT::infer(const cv::Mat& image, std::vector<float>& embedding) {
  cv::Mat f;
  if (!run_hidden_state(image, f)) return false;
  const int d = f.size[2];
  const float* plane = reinterpret_cast<const float*>(f.data);
  embedding.assign(plane, plane + d);
  return true;
}

bool Dinov2TRT::infer_global_and_patch_tokens(const cv::Mat& image,
                                              std::vector<float>& global_embedding,
                                              cv::Mat& tokens) {
  cv::Mat f;
  if (!run_hidden_state(image, f)) return false;
  if (f.size[1] < 2) {
    set_error("unexpected last_hidden_state shape");
    return false;
  }
  const int rows = f.size[1];
  const int d = f.size[2];
  const float* plane = reinterpret_cast<const float*>(f.data);
  global_embedding.assign(plane, plane + d);
  tokens.create(rows - 1, d, CV_32F);
  std::memcpy(tokens.data, plane + d, static_cast<size_t>(rows - 1) * d * sizeof(float));
  l2_normalize_rows(tokens);
  return true;
}

bool Dinov2TRT::infer_patch_tokens(const cv::Mat& image, cv::Mat& tokens) {
  cv::Mat f;
  if (!run_hidden_state(image, f)) return false;
  if (f.size[1] < 2) {
    set_error("unexpected last_hidden_state shape");
    return false;
  }
  const int rows = f.size[1];
  const int d = f.size[2];
  const float* plane = reinterpret_cast<const float*>(f.data);
  tokens.create(rows - 1, d, CV_32F);
  std::memcpy(tokens.data, plane + d, static_cast<size_t>(rows - 1) * d * sizeof(float));
  l2_normalize_rows(tokens);
  return true;
}

// ------------------------------------------------------------- SigLIP image
// Port of reference/tinynav/core/models_trt.py::SigLIPImageTRT.

SigLIPImageTRT::SigLIPImageTRT(std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "siglip_vit_b_16_webli_image_fp16", arch)) {}

bool SigLIPImageTRT::on_loaded(std::string& err) {
  if (inputs_.size() != 1) {
    err = "SigLIP image engine must have 1 input, got " + std::to_string(inputs_.size());
    return false;
  }
  if (outputs_.empty() || inputs_[0]->shape.size() != 4) {
    err = "SigLIP image engine: bad IO shapes";
    return false;
  }
  output_name_ = outputs_[0]->name;
  net_h_ = static_cast<int>(inputs_[0]->shape[2]);
  net_w_ = static_cast<int>(inputs_[0]->shape[3]);
  return true;
}

cv::Mat SigLIPImageTRT::preprocess_image(const cv::Mat& image) {
  cv::Mat rgb;
  if (image.dims == 2 && image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  } else {
    std::fprintf(stderr, "[tinynav::trt] SigLIPImageTRT::preprocess_image: bad input\n");
    return {};
  }
  cv::resize(rgb, rgb, cv::Size(net_w_, net_h_), 0, 0, cv::INTER_CUBIC);
  const double mean[3] = {0.5, 0.5, 0.5};
  const double std_[3] = {0.5, 0.5, 0.5};
  return to_planar_chw(rgb, mean, std_, 255.0);
}

bool SigLIPImageTRT::infer(const cv::Mat& image, std::vector<float>& embedding) {
  if (!ensure_loaded()) return false;
  cv::Mat in = preprocess_image(image);
  if (in.empty() || !copy_to_input(0, in)) return false;
  TrtOutputMap results;
  if (!run_graph(results)) return false;
  auto it = results.find(output_name_);
  if (it == results.end() || !flatten_to_float(it->second, embedding)) {
    set_error("SigLIP image engine: missing output " + output_name_);
    return false;
  }
  return true;
}

// -------------------------------------------------------------- SigLIP text
// Port of reference/tinynav/core/models_trt.py::SigLIPTextTRT (tokenizer is
// pluggable instead of the HF `tokenizers` package).

SigLIPTextTRT::SigLIPTextTRT(std::shared_ptr<SiglipTokenizer> tokenizer,
                             std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "siglip_vit_b_16_webli_text_fp16", arch)),
      tokenizer_(std::move(tokenizer)) {}

bool SigLIPTextTRT::pre_load(std::string& err) {
  if (!tokenizer_ || !tokenizer_->available()) {
    err = "no SiglipTokenizer plugged in (HF tokenizers not ported); "
          "use PrecomputedTextEncoder or provide a SiglipTokenizer";
    return false;
  }
  return true;
}

bool SigLIPTextTRT::on_loaded(std::string& err) {
  if (inputs_.empty() || outputs_.empty() || inputs_[0]->shape.empty()) {
    err = "SigLIP text engine: bad IO shapes";
    return false;
  }
  output_name_ = outputs_[0]->name;
  context_length_ = static_cast<size_t>(inputs_[0]->shape.back());
  return true;
}

bool SigLIPTextTRT::encode(const std::string& text, std::vector<float>& embedding) {
  if (!ensure_loaded()) return false;
  std::vector<int64_t> input_ids(context_length_, 0);
  std::vector<int64_t> attention_mask(context_length_, 0);
  if (!tokenizer_->encode(text, input_ids.data(), attention_mask.data(), context_length_)) {
    set_error("SiglipTokenizer::encode failed");
    return false;
  }
  for (size_t i = 0; i < inputs_.size(); ++i) {
    const std::vector<int64_t>* src = nullptr;
    if (inputs_[i]->name == "input_ids") {
      src = &input_ids;
    } else if (inputs_[i]->name == "attention_mask") {
      src = &attention_mask;
    } else if (inputs_.size() == 1) {
      src = &input_ids;
    } else {
      set_error("Unsupported SigLIP text engine input: " + inputs_[i]->name);
      return false;
    }
    cv::Mat m(static_cast<int>(src->size()), 1, CV_64F);
    for (size_t j = 0; j < src->size(); ++j) {
      m.at<double>(static_cast<int>(j)) = static_cast<double>((*src)[j]);
    }
    if (!copy_to_input(i, m)) return false;
  }
  TrtOutputMap results;
  if (!run_graph(results)) return false;
  auto it = results.find(output_name_);
  if (it == results.end() || !flatten_to_float(it->second, embedding)) {
    set_error("SigLIP text engine: missing output " + output_name_);
    return false;
  }
  return true;
}

// --------------------------------------------------- Precomputed text store

bool PrecomputedTextEncoder::available() {
  std::error_code ec;
  return std::filesystem::is_directory(dir_, ec);
}

std::string PrecomputedTextEncoder::sanitize(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    const bool keep = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
    out.push_back(keep ? c : '_');
  }
  return out;
}

bool PrecomputedTextEncoder::encode(const std::string& text,
                                    std::vector<float>& embedding) {
  const std::string path = dir_ + "/" + sanitize(text) + ".f32";
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamoff size = f.tellg();
  if (size <= 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) return false;
  embedding.resize(static_cast<size_t>(size) / sizeof(float));
  f.seekg(0);
  return static_cast<bool>(
      f.read(reinterpret_cast<char*>(embedding.data()), size));
}

// ------------------------------------------------------------ SigLIP facade
// Port of reference/tinynav/core/models_trt.py::SigLIPTRT.

SigLIPTRT::SigLIPTRT(std::string model_dir, std::string arch,
                     std::shared_ptr<SiglipTokenizer> tokenizer)
    : model_dir_(std::move(model_dir)),
      arch_(std::move(arch)),
      tokenizer_(std::move(tokenizer)) {}

void SigLIPTRT::set_text_encoder(std::shared_ptr<SiglipTextEncoder> encoder) {
  text_encoder_ = std::move(encoder);
}

bool SigLIPTRT::encode_image(const cv::Mat& image, std::vector<float>& embedding) {
  if (!image_encoder_) {
    image_encoder_ = std::make_unique<SigLIPImageTRT>(model_dir_, arch_);
  }
  return image_encoder_->infer(image, embedding);
}

bool SigLIPTRT::encode_text(const std::string& text, std::vector<float>& embedding) {
  if (!text_encoder_) {
    text_encoder_ = std::make_shared<SigLIPTextTRT>(tokenizer_, model_dir_, arch_);
  }
  return text_encoder_->encode(text, embedding);
}

// --------------------------------------------------------- FoundationStereo
// Port of reference/tinynav/core/models_trt.py::FoundationStereoTRT.

FoundationStereoTRT::FoundationStereoTRT(std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "foundation_stereo_11-33-40_256x320_4", arch)) {}

bool FoundationStereoTRT::on_loaded(std::string& err) {
  if (inputs_.size() != 2) {
    err = "FoundationStereo engine must have 2 inputs, got " +
          std::to_string(inputs_.size());
    return false;
  }
  if (outputs_.size() != 1) {
    err = "FoundationStereo engine must have 1 output, got " +
          std::to_string(outputs_.size());
    return false;
  }
  left_idx_ = inputs_[0]->name == "left" ? 0 : 1;
  right_idx_ = 1 - left_idx_;
  output_name_ = outputs_[0]->name;
  if (inputs_[left_idx_]->shape.size() != 4) {
    err = "FoundationStereo engine: input must be [1,3,H,W]";
    return false;
  }
  net_h_ = static_cast<int>(inputs_[left_idx_]->shape[2]);
  net_w_ = static_cast<int>(inputs_[left_idx_]->shape[3]);
  return true;
}

cv::Mat FoundationStereoTRT::resize_for_engine(const cv::Mat& image) const {
  if (image.rows == net_h_ && image.cols == net_w_) return image;
  cv::Mat out;
  cv::resize(image, out, cv::Size(net_w_, net_h_), 0, 0, cv::INTER_LINEAR);
  return out;
}

cv::Mat FoundationStereoTRT::to_three_channel_float(const cv::Mat& image) const {
  cv::Mat img3;
  if (image.channels() == 1) {
    cv::cvtColor(image, img3, cv::COLOR_GRAY2BGR);  // repeats the channel
  } else if (image.channels() == 3) {
    img3 = image;  // first 3 channels, order preserved (no BGR->RGB swap)
  } else if (image.channels() == 4) {
    cv::cvtColor(image, img3, cv::COLOR_BGRA2BGR);
  } else {
    return {};
  }
  const double mean[3] = {0.0, 0.0, 0.0};
  const double std_[3] = {1.0, 1.0, 1.0};
  return to_planar_chw(img3, mean, std_, 1.0);  // cast to float, no scaling
}

bool FoundationStereoTRT::infer(const cv::Mat& left, const cv::Mat& right, double baseline,
                                double focal_length, cv::Mat& disp, cv::Mat& depth) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("Left/right shape mismatch");
  }
  if (!ensure_loaded()) return false;
  const int h_in = left.rows;
  const int w_in = left.cols;

  cv::Mat left_tensor = to_three_channel_float(resize_for_engine(left));
  cv::Mat right_tensor = to_three_channel_float(resize_for_engine(right));
  if (left_tensor.empty() || right_tensor.empty()) {
    set_error("FoundationStereoTRT: unsupported image shape");
    return false;
  }
  if (!copy_to_input(left_idx_, left_tensor)) return false;
  if (!copy_to_input(right_idx_, right_tensor)) return false;

  TrtOutputMap results;
  if (!run_graph(results)) return false;
  auto it = results.find(output_name_);
  if (it == results.end()) {
    set_error("FoundationStereo engine: missing output " + output_name_);
    return false;
  }
  cv::Mat dn;
  to_float32(it->second, dn);
  if (static_cast<int64_t>(dn.total()) !=
      static_cast<int64_t>(net_h_) * net_w_) {
    set_error("FoundationStereo engine: unexpected output size");
    return false;
  }
  cv::Mat disp_net(net_h_, net_w_, CV_32F);
  std::memcpy(disp_net.data, dn.data, static_cast<size_t>(net_h_) * net_w_ * sizeof(float));
  cv::max(disp_net, 0.0f, disp_net);

  if (h_in == net_h_ && w_in == net_w_) {
    disp = disp_net;
  } else {
    cv::resize(disp_net, disp, cv::Size(w_in, h_in), 0, 0, cv::INTER_LINEAR);
    disp *= static_cast<float>(w_in) / static_cast<float>(net_w_);
    cv::max(disp, 0.0f, disp);
  }

  // Port of the upstream hack: disp[300:,:] = 0.0 (no-op below 300 rows).
  if (disp.rows > 300) disp.rowRange(300, disp.rows).setTo(0.0f);

  depth = disparity_to_depth(disp, baseline, focal_length);
  return true;
}

// ---------------------------------------------------------------- Retinify
// Port of reference/tinynav/core/models_trt.py::RetinifyTRT.

RetinifyTRT::RetinifyTRT(std::string model_dir, std::string arch)
    : TRTBase(engine_path_for(model_dir, "retinify_0_1_5_dynamic", arch)) {}

std::vector<int64_t> RetinifyTRT::get_static_shape(const char* name) {
  // Port of RetinifyTRT._get_static_shape: the NHWC output shares the left
  // input's spatial resolution; derive its max shape from the "left" profile
  // because some TRT versions report empty/scalar dynamic output shapes.
  auto* engine = static_cast<nvinfer1::ICudaEngine*>(engine_);
  if (engine != nullptr &&
      engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
    const nvinfer1::Dims mx =
        engine->getProfileShape("left", 0, nvinfer1::OptProfileSelector::kMAX);
    if (mx.nbDims == 4) {
      return {1, static_cast<int64_t>(mx.d[1]), static_cast<int64_t>(mx.d[2]), 1};
    }
  }
  return TRTBase::get_static_shape(name);
}

bool RetinifyTRT::on_loaded(std::string& err) {
  if (inputs_.size() != 2) {
    err = "Retinify disp-only engine must have 2 inputs, got " +
          std::to_string(inputs_.size());
    return false;
  }
  if (outputs_.size() != 1) {
    err = "Retinify disp-only engine must have 1 output, got " +
          std::to_string(outputs_.size());
    return false;
  }
  output_name_ = outputs_[0]->name;
  input_dtype_ = inputs_[0]->dtype;
  return true;
}

bool RetinifyTRT::capture_graph(std::string& /*err*/) {
  // Port of RetinifyTRT.capture_graph: bind addresses only; the dynamic
  // shapes change per infer, so no graph is captured — enqueueV3 each call.
  auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
  for (const auto& t : tensors_) {
    context->setTensorAddress(t.name.c_str(), t.device);
  }
  return false;
}

bool RetinifyTRT::run_graph(TrtOutputMap& results) {
  auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  std::string err;

#if !TINYNAV_TRT_MAPPED_HOST
  for (int i = 0; i < 2; ++i) {
    const cudaError_t cerr =
        cudaMemcpyAsync(inputs_[i]->device, inputs_[i]->host, cur_input_nbytes_,
                        cudaMemcpyHostToDevice, stream);
    if (cerr != cudaSuccess) {
      set_error(std::string("H2D copy: ") + cudaGetErrorString(cerr));
      return false;
    }
  }
#endif

  if (!context->setOptimizationProfileAsync(0, stream)) {
    set_error("setOptimizationProfileAsync(0) failed");
    return false;
  }
  nvinfer1::Dims d;
  d.nbDims = 4;
  d.d[0] = 1;
  d.d[1] = cur_h_;
  d.d[2] = cur_w_;
  d.d[3] = 1;
  if (!context->setInputShape("left", d) || !context->setInputShape("right", d)) {
    set_error("setInputShape failed for left/right");
    return false;
  }
  if (!context->enqueueV3(stream)) {
    set_error("enqueueV3 failed");
    return false;
  }

  const size_t out_nbytes =
      static_cast<size_t>(cur_h_) * cur_w_ * dtype_size(outputs_[0]->dtype);
#if !TINYNAV_TRT_MAPPED_HOST
  for (const auto* out : outputs_) {
    const cudaError_t cerr = cudaMemcpyAsync(out->host, out->device, out_nbytes,
                                             cudaMemcpyDeviceToHost, stream);
    if (cerr != cudaSuccess) {
      set_error(std::string("D2H copy: ") + cudaGetErrorString(cerr));
      return false;
    }
  }
#endif
  const cudaError_t serr = cudaStreamSynchronize(stream);
  if (serr != cudaSuccess) {
    set_error(std::string("cudaStreamSynchronize: ") + cudaGetErrorString(serr));
    return false;
  }

  for (const auto* out : outputs_) {
    const int depth = cv_depth_for(out->dtype);
    if (depth < 0) {
      set_error("cannot map output dtype of " + out->name + " to a cv::Mat");
      return false;
    }
    cv::Mat m(cur_h_, cur_w_, depth);
    std::memcpy(m.data, out->host, out_nbytes);
    results[out->name] = m;
  }
  return true;
}

bool RetinifyTRT::infer(const cv::Mat& left, const cv::Mat& right, double baseline,
                        double focal_length, cv::Mat& disp, cv::Mat& depth) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("Left/right shape mismatch");
  }
  if (!ensure_loaded()) return false;
  if (left.channels() != 1 || right.channels() != 1) {
    set_error("RetinifyTRT::infer expects single-channel images");
    return false;
  }
  cur_h_ = left.rows;
  cur_w_ = left.cols;

  // Partial fill of the max-profile buffers (np.copyto(host.flat[:n], ...)).
  if (!copy_to_input(0, left)) return false;
  if (!copy_to_input(1, right)) return false;
  cur_input_nbytes_ =
      static_cast<size_t>(cur_h_) * cur_w_ * dtype_size(input_dtype_);

  TrtOutputMap results;
  if (!run_graph(results)) return false;
  auto it = results.find(output_name_);
  if (it == results.end()) {
    set_error("Retinify engine: missing output " + output_name_);
    return false;
  }
  if (it->second.rows != cur_h_ || it->second.cols != cur_w_) {
    set_error("RetinifyTRT output shape mismatch");
    return false;
  }
  to_float32(it->second, disp);
  depth = disparity_to_depth(disp, baseline, focal_length);
  return true;
}

}  // namespace tinynav::trt
