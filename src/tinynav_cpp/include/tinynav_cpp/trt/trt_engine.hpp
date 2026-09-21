// Port of reference/tinynav/core/models_trt.py — TRTBase + disparity_to_depth.
//
// TensorRT 10 engine wrapper: lazy engine deserialization, page-locked host
// buffers, CUDA Graph capture/replay with fallback to direct enqueueV3.
// Pure library: no ROS headers; this header exposes no TRT/CUDA types.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace tinynav::trt {

// Compile-time platform suffix used in engine file names, mirroring
// Python's platform.machine(). Overridable with -DTINYNAV_TRT_ARCH=...
std::string engine_arch();

// "/tinynav/tinynav/models" — engines shipped in the image.
std::string default_model_dir();

// <model_dir>/<file_stem>_<arch>.plan
std::string engine_path_for(const std::string& model_dir, const std::string& file_stem,
                            const std::string& arch);

// Results of one engine run, keyed by output tensor name; each cv::Mat keeps
// the engine's tensor shape and is a host-side copy. kINT64 outputs are
// delivered as CV_64F (exact for index values); everything else keeps the
// natural cv depth (kFLOAT→CV_32F, kHALF→CV_16F, kINT32→CV_32S, ...).
using TrtOutputMap = std::unordered_map<std::string, cv::Mat>;

// Port of reference/tinynav/core/models_trt.py::disparity_to_depth.
// depth = baseline * focal / disparity where disparity is finite and > 0,
// else 0. Throws std::invalid_argument on non-positive baseline/focal
// (Python raises ValueError).
cv::Mat disparity_to_depth(const cv::Mat& disparity, double baseline, double focal_length);

// One bound engine tensor (input or output), buffers allocated at the
// resolved static / max-profile shape. dtype stores an nvinfer1::DataType
// value as int32_t so this header stays TRT-free.
struct TrtTensor {
  std::string name;
  std::vector<int64_t> shape;
  int32_t dtype = 0;  // nvinfer1::DataType
  int64_t numel = 0;
  size_t nbytes = 0;
  void* host = nullptr;    // page-locked host buffer
  void* device = nullptr;  // device buffer (== host on aarch64 mapped memory)
  bool is_input = false;
};

// Port of reference/tinynav/core/models_trt.py::TRTBase.
//
// Lazy + degrade-never-crash: the constructor only records the engine path.
// The first infer() triggers load(); any failure (missing file, no GPU,
// deserialize error, buffer allocation error) is logged to stderr and the
// instance becomes permanently unavailable — available()/infer() return
// false and the process keeps running.
class TRTBase {
 public:
  explicit TRTBase(std::string engine_path);
  virtual ~TRTBase();

  TRTBase(const TRTBase&) = delete;
  TRTBase& operator=(const TRTBase&) = delete;

  const std::string& engine_path() const { return engine_path_; }

  // Triggers the lazy load on first call; true iff the engine is usable.
  virtual bool available();
  bool loaded() const;
  const std::string& last_error() const { return last_error_; }

  const std::vector<TrtTensor*>& inputs() const { return inputs_; }
  const std::vector<TrtTensor*>& outputs() const { return outputs_; }

 protected:
  // Hook run before any file/CUDA work; subclasses reject configurations
  // that can never load (e.g. SigLIP text without a tokenizer).
  virtual bool pre_load(std::string& err);
  // Hook run after buffers are allocated; subclasses validate IO counts and
  // cache shapes/names. Returning false marks the engine unavailable.
  virtual bool on_loaded(std::string& err);
  // Port of TRTBase::_get_static_shape: concrete shape for a tensor,
  // dynamic dims resolved from optimization profile 0's max shape.
  virtual std::vector<int64_t> get_static_shape(const char* name);
  // Port of TRTBase::capture_graph. Return false = run without a graph
  // (direct enqueueV3); capture failure also degrades to that path.
  virtual bool capture_graph(std::string& err);
  // Port of TRTBase::run_graph (synchronous: H2D -> graph/enqueueV3 -> D2H
  // -> stream sync; the Python version's asyncio event polling collapses
  // into the sync).
  virtual bool run_graph(TrtOutputMap& results);

  bool ensure_loaded();

  // numpy copyto(host, value.astype(host.dtype)) semantics: per-element cast
  // into the page-locked host buffer. src.total() must be <= tensor numel
  // (a short src fills only the leading flat region — Retinify relies on
  // this for its max-profile buffers).
  bool copy_to_input(size_t input_index, const cv::Mat& src);

  void set_error(std::string msg);

  size_t dtype_size(int32_t dtype) const;
  int cv_depth_for(int32_t dtype) const;  // -1 if unmappable

  std::string engine_path_;
  // Opaque TRT/CUDA handles: nvinfer1::IRuntime* / ICudaEngine* /
  // IExecutionContext*, cudaStream_t, cudaGraphExec_t.
  void* runtime_ = nullptr;
  void* engine_ = nullptr;
  void* context_ = nullptr;
  std::vector<TrtTensor> tensors_;  // engine IO order
  std::vector<TrtTensor*> inputs_;
  std::vector<TrtTensor*> outputs_;
  void* stream_ = nullptr;
  void* graph_exec_ = nullptr;
  bool use_graph_ = false;

 private:
  bool load(std::string& err);
  void release();

  enum class LoadState { kNotTried, kReady, kFailed };
  LoadState load_state_ = LoadState::kNotTried;
  std::string last_error_;
};

}  // namespace tinynav::trt
