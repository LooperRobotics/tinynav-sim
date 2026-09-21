// Port of reference/tinynav/core/models_trt.py — the model wrapper classes.
//
// All wrappers share TRTBase's lazy/degrade contract: construction only
// records paths, the first infer() loads the engine, and any load failure
// marks the instance unavailable (infer returns false) instead of crashing.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "tinynav_cpp/trt/trt_engine.hpp"

namespace tinynav::trt {

// Port of reference/tinynav/core/models_trt.py::SuperPointTRT.
class SuperPointTRT : public TRTBase {
 public:
  explicit SuperPointTRT(std::string model_dir = default_model_dir(),
                         std::string arch = engine_arch());

  // image: HxW single-channel (any depth; cast to the engine dtype).
  // results: "kpts" CV_32F, [1,2,N] or [1,N,2], rescaled from network to
  //          input-image pixel coords ((k+0.5)*scale-0.5 per axis);
  //          "descps" as emitted by the engine; "mask" reshaped to [1,N,1].
  // threshold: detection threshold, engine input [1,1] (Python default 5e-4).
  bool infer(const cv::Mat& image, TrtOutputMap& results, float threshold = 5e-4f);

 protected:
  bool on_loaded(std::string& err) override;

 private:
  int net_h_ = 0;
  int net_w_ = 0;
};

// Port of reference/tinynav/core/models_trt.py::LightGlueTRT.
class LightGlueTRT : public TRTBase {
 public:
  explicit LightGlueTRT(std::string model_dir = default_model_dir(),
                        std::string arch = engine_arch());

  // Positional inputs mirror the Python copy order:
  //   kpts0, kpts1, desc0, desc1, mask0, mask1, img_shape0, img_shape1,
  //   match_threshold (inputs 0..8).
  // img_shape is (width, height) like Python's np.array([w, h], int64).
  // results: the engine outputs verbatim ("match_indices", ...).
  bool infer(const cv::Mat& kpts0, const cv::Mat& kpts1, const cv::Mat& desc0,
             const cv::Mat& desc1, const cv::Mat& mask0, const cv::Mat& mask1,
             const std::array<int64_t, 2>& img_shape0,
             const std::array<int64_t, 2>& img_shape1, TrtOutputMap& results,
             float match_threshold = 0.1f);
};

// Port of reference/tinynav/core/models_trt.py::Dinov2TRT.
class Dinov2TRT : public TRTBase {
 public:
  explicit Dinov2TRT(std::string model_dir = default_model_dir(),
                     std::string arch = engine_arch());

  // resize INTER_CUBIC -> GRAY2RGB -> CHW float /255 -> (x-mean)/std with
  // ImageNet mean/std; returns planar [1,3,target,target] CV_32F.
  cv::Mat preprocess_image(const cv::Mat& image, int target_size = 224);

  // CLS token of last_hidden_state: [D].
  bool infer(const cv::Mat& image, std::vector<float>& embedding);
  // CLS token [D] + L2-normalized patch tokens [N,D] CV_32F, one engine run.
  bool infer_global_and_patch_tokens(const cv::Mat& image,
                                     std::vector<float>& global_embedding,
                                     cv::Mat& tokens);
  // L2-normalized patch tokens [N,D] CV_32F (no CLS).
  bool infer_patch_tokens(const cv::Mat& image, cv::Mat& tokens);

 private:
  // One engine run; hidden_f32 gets last_hidden_state [1, 1+N, D] as CV_32F.
  bool run_hidden_state(const cv::Mat& image, cv::Mat& hidden_f32);
};

// Port of reference/tinynav/core/models_trt.py::SigLIPImageTRT.
class SigLIPImageTRT : public TRTBase {
 public:
  explicit SigLIPImageTRT(std::string model_dir = default_model_dir(),
                          std::string arch = engine_arch());

  // gray/(H,W,1)/BGR -> RGB -> resize INTER_CUBIC to the engine size ->
  // CHW float /255 -> (x-0.5)/0.5; returns planar [1,3,net_h,net_w] CV_32F.
  cv::Mat preprocess_image(const cv::Mat& image);

  // Image embedding, flattened engine output: [D].
  bool infer(const cv::Mat& image, std::vector<float>& embedding);

 protected:
  bool on_loaded(std::string& err) override;

 private:
  int net_h_ = 0;
  int net_w_ = 0;
  std::string output_name_;
};

// Pluggable tokenizer for the SigLIP text tower (see the porting notes: the
// HF `tokenizers` dependency is not ported). Contract of
// SigLIPTextTRT._tokenize: truncate to context_length, pad with the [PAD]
// id (0 when unknown), attention_mask 1 on real tokens / 0 on padding.
class SiglipTokenizer {
 public:
  virtual ~SiglipTokenizer() = default;
  virtual bool available() const = 0;
  virtual bool encode(const std::string& text, int64_t* input_ids,
                      int64_t* attention_mask, size_t context_length) const = 0;
};

// Text-embedding source for SigLIPTRT — either the TRT engine (with a
// tokenizer plugged in) or a precomputed-embedding store.
class SiglipTextEncoder {
 public:
  virtual ~SiglipTextEncoder() = default;
  virtual bool available() = 0;
  virtual bool encode(const std::string& text, std::vector<float>& embedding) = 0;
};

// Port of reference/tinynav/core/models_trt.py::SigLIPTextTRT.
// Without a usable tokenizer the encoder is unavailable (degrade, no crash).
class SigLIPTextTRT : public TRTBase, public SiglipTextEncoder {
 public:
  explicit SigLIPTextTRT(std::shared_ptr<SiglipTokenizer> tokenizer,
                         std::string model_dir = default_model_dir(),
                         std::string arch = engine_arch());

  bool available() override { return TRTBase::available(); }
  // Text embedding, flattened engine output: [D].
  bool encode(const std::string& text, std::vector<float>& embedding) override;
  bool infer(const std::string& text, std::vector<float>& embedding) {
    return encode(text, embedding);
  }

 protected:
  bool pre_load(std::string& err) override;
  bool on_loaded(std::string& err) override;

 private:
  std::shared_ptr<SiglipTokenizer> tokenizer_;
  size_t context_length_ = 0;
  std::string output_name_;
};

// SiglipTextEncoder backed by precomputed embeddings: reads raw
// little-endian float32 blobs from <dir>/<sanitize(text)>.f32; a missing
// file degrades to false. Drop-in text side when no tokenizer is available.
class PrecomputedTextEncoder : public SiglipTextEncoder {
 public:
  explicit PrecomputedTextEncoder(std::string dir) : dir_(std::move(dir)) {}

  bool available() override;
  bool encode(const std::string& text, std::vector<float>& embedding) override;

  // alnum, '-' and '_' are kept; everything else becomes '_'.
  static std::string sanitize(const std::string& text);

 private:
  std::string dir_;
};

// Port of reference/tinynav/core/models_trt.py::SigLIPTRT (facade; the
// Python version takes a tokenizer_path, this one takes a pluggable
// SiglipTokenizer / SiglipTextEncoder). Both encoders stay lazy.
class SigLIPTRT {
 public:
  explicit SigLIPTRT(std::string model_dir = default_model_dir(),
                     std::string arch = engine_arch(),
                     std::shared_ptr<SiglipTokenizer> tokenizer = nullptr);

  // Inject a text encoder (e.g. PrecomputedTextEncoder) replacing the
  // default engine+tokenizer path.
  void set_text_encoder(std::shared_ptr<SiglipTextEncoder> encoder);

  bool encode_image(const cv::Mat& image, std::vector<float>& embedding);
  bool encode_text(const std::string& text, std::vector<float>& embedding);

 private:
  std::string model_dir_;
  std::string arch_;
  std::shared_ptr<SiglipTokenizer> tokenizer_;
  std::unique_ptr<SigLIPImageTRT> image_encoder_;
  std::shared_ptr<SiglipTextEncoder> text_encoder_;
};

// Port of reference/tinynav/core/models_trt.py::FoundationStereoTRT.
class FoundationStereoTRT : public TRTBase {
 public:
  explicit FoundationStereoTRT(std::string model_dir = default_model_dir(),
                               std::string arch = engine_arch());

  // left/right: same-size images (gray or color). disp: CV_32F at input
  // resolution, clipped >= 0 (and rescaled on resize); rows 300+ are zeroed
  // (port of the upstream hack). depth: disparity_to_depth(disp, ...).
  // Throws std::invalid_argument on left/right shape mismatch (Python
  // raises ValueError); baseline/focal validation as in disparity_to_depth.
  bool infer(const cv::Mat& left, const cv::Mat& right, double baseline,
             double focal_length, cv::Mat& disp, cv::Mat& depth);

 protected:
  bool on_loaded(std::string& err) override;

 private:
  // _resize_for_engine / _to_three_channel_float: gray is repeated to 3
  // channels, color keeps its first 3 channels in order (no BGR->RGB swap),
  // cast to float without normalization; planar [1,3,h,w] CV_32F.
  cv::Mat resize_for_engine(const cv::Mat& image) const;
  cv::Mat to_three_channel_float(const cv::Mat& image) const;

  int left_idx_ = 0;
  int right_idx_ = 1;
  int net_h_ = 0;
  int net_w_ = 0;
  std::string output_name_;
};

// Port of reference/tinynav/core/models_trt.py::RetinifyTRT.
// Disp-only dynamic-shape engine with NHWC tensors; no CUDA graph (shapes
// are re-set per infer), buffers allocated at the max profile.
class RetinifyTRT : public TRTBase {
 public:
  explicit RetinifyTRT(std::string model_dir = default_model_dir(),
                       std::string arch = engine_arch());

  // left/right: same-size single-channel images, cast to the engine input
  // dtype (no resize, no normalization). disp: CV_32F at input resolution;
  // depth: disparity_to_depth(disp, ...).
  bool infer(const cv::Mat& left, const cv::Mat& right, double baseline,
             double focal_length, cv::Mat& disp, cv::Mat& depth);

 protected:
  std::vector<int64_t> get_static_shape(const char* name) override;
  bool on_loaded(std::string& err) override;
  bool capture_graph(std::string& err) override;
  bool run_graph(TrtOutputMap& results) override;

 private:
  int32_t input_dtype_ = 0;
  int cur_h_ = 0;
  int cur_w_ = 0;
  size_t cur_input_nbytes_ = 0;
  std::string output_name_;
};

// Python: StereoEngineTRT = RetinifyTRT
using StereoEngineTRT = RetinifyTRT;

}  // namespace tinynav::trt
