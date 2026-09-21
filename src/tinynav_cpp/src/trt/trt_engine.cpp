// Port of reference/tinynav/core/models_trt.py — TRTBase + disparity_to_depth.
#include "tinynav_cpp/trt/trt_engine.hpp"

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#if defined(__aarch64__)
// Jetson: zero-copy mapped host memory, no explicit H2D/D2H copies.
#define TINYNAV_TRT_MAPPED_HOST 1
#else
#define TINYNAV_TRT_MAPPED_HOST 0
#endif

namespace tinynav::trt {
namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[tinynav::trt] TRT: %s\n", msg);
    }
  }
};

TrtLogger& trt_logger() {
  static TrtLogger logger;
  return logger;
}

bool cuda_ok(cudaError_t err, const std::string& what, std::string& out) {
  if (err == cudaSuccess) return true;
  out = what + ": " + cudaGetErrorString(err);
  return false;
}

// IEEE-754 binary16 <-> binary32, host-side (avoids a cuda_fp16.h include).
uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int exp = static_cast<int>((x >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    uint32_t m = mant >> (14 - exp);
    if ((mant >> (13 - exp)) & 1u) m += 1u;
    return static_cast<uint16_t>(sign | m);
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 1u : 0u));
  uint32_t h = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1u;
  return static_cast<uint16_t>(h);
}

float half_to_float(uint16_t h) {
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

double mat_elem_as_double(const cv::Mat& m, size_t i) {
  switch (m.depth()) {
    case CV_8U: return m.data[i];
    case CV_8S: return reinterpret_cast<const int8_t*>(m.data)[i];
    case CV_16U: return reinterpret_cast<const uint16_t*>(m.data)[i];
    case CV_16S: return reinterpret_cast<const int16_t*>(m.data)[i];
    case CV_32S: return reinterpret_cast<const int32_t*>(m.data)[i];
    case CV_32F: return reinterpret_cast<const float*>(m.data)[i];
    case CV_64F: return reinterpret_cast<const double*>(m.data)[i];
    case CV_16F: return half_to_float(reinterpret_cast<const uint16_t*>(m.data)[i]);
    default: return 0.0;
  }
}

void write_elem(void* host, int32_t dtype, size_t i, double v) {
  switch (static_cast<nvinfer1::DataType>(dtype)) {
    case nvinfer1::DataType::kFLOAT:
      reinterpret_cast<float*>(host)[i] = static_cast<float>(v);
      break;
    case nvinfer1::DataType::kHALF:
      reinterpret_cast<uint16_t*>(host)[i] = float_to_half(static_cast<float>(v));
      break;
    case nvinfer1::DataType::kINT8:
      reinterpret_cast<int8_t*>(host)[i] = static_cast<int8_t>(v);
      break;
    case nvinfer1::DataType::kUINT8:
      reinterpret_cast<uint8_t*>(host)[i] = static_cast<uint8_t>(v);
      break;
    case nvinfer1::DataType::kINT32:
      reinterpret_cast<int32_t*>(host)[i] = static_cast<int32_t>(v);
      break;
    case nvinfer1::DataType::kINT64:
      reinterpret_cast<int64_t*>(host)[i] = static_cast<int64_t>(v);
      break;
    case nvinfer1::DataType::kBOOL:
      reinterpret_cast<uint8_t*>(host)[i] = v != 0.0 ? 1 : 0;
      break;
    default:
      break;
  }
}

}  // namespace

std::string engine_arch() {
#if defined(TINYNAV_TRT_ARCH)
  return TINYNAV_TRT_ARCH;
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__aarch64__)
  return "aarch64";
#else
  return "unknown";
#endif
}

std::string default_model_dir() { return "/tinynav/tinynav/models"; }

std::string engine_path_for(const std::string& model_dir, const std::string& file_stem,
                            const std::string& arch) {
  return model_dir + "/" + file_stem + "_" + arch + ".plan";
}

cv::Mat disparity_to_depth(const cv::Mat& disparity, double baseline, double focal_length) {
  if (baseline <= 0.0) {
    throw std::invalid_argument("baseline must be positive, got " + std::to_string(baseline));
  }
  if (focal_length <= 0.0) {
    throw std::invalid_argument("focal_length must be positive, got " +
                                std::to_string(focal_length));
  }
  cv::Mat cont = disparity.isContinuous() ? disparity : disparity.clone();
  cv::Mat depth = cv::Mat::zeros(cont.size(), CV_32F);
  const size_t n = cont.total();
  for (size_t i = 0; i < n; ++i) {
    const double d = mat_elem_as_double(cont, i);
    if (std::isfinite(d) && d > 0.0) {
      reinterpret_cast<float*>(depth.data)[i] =
          static_cast<float>(baseline * focal_length / d);
    }
  }
  return depth;
}

TRTBase::TRTBase(std::string engine_path) : engine_path_(std::move(engine_path)) {}

TRTBase::~TRTBase() { release(); }

bool TRTBase::loaded() const { return load_state_ == LoadState::kReady; }

bool TRTBase::available() { return ensure_loaded(); }

void TRTBase::set_error(std::string msg) { last_error_ = std::move(msg); }

bool TRTBase::pre_load(std::string& /*err*/) { return true; }

bool TRTBase::on_loaded(std::string& /*err*/) { return true; }

bool TRTBase::ensure_loaded() {
  if (load_state_ == LoadState::kReady) return true;
  if (load_state_ == LoadState::kFailed) return false;
  std::string err;
  if (load(err)) {
    load_state_ = LoadState::kReady;
    std::fprintf(stderr, "[tinynav::trt] load %s done!\n", engine_path_.c_str());
    return true;
  }
  release();
  set_error(std::move(err));
  load_state_ = LoadState::kFailed;
  std::fprintf(stderr, "[tinynav::trt] %s unavailable: %s\n", engine_path_.c_str(),
               last_error_.c_str());
  return false;
}

size_t TRTBase::dtype_size(int32_t dtype) const {
  switch (static_cast<nvinfer1::DataType>(dtype)) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kFP8: return 1;
    default: return 0;
  }
}

int TRTBase::cv_depth_for(int32_t dtype) const {
  switch (static_cast<nvinfer1::DataType>(dtype)) {
    case nvinfer1::DataType::kFLOAT: return CV_32F;
    case nvinfer1::DataType::kHALF: return CV_16F;
    case nvinfer1::DataType::kINT8: return CV_8S;
    case nvinfer1::DataType::kUINT8: return CV_8U;
    case nvinfer1::DataType::kINT32: return CV_32S;
    case nvinfer1::DataType::kBOOL: return CV_8U;
    case nvinfer1::DataType::kINT64: return CV_64F;  // converted element-wise
    default: return -1;
  }
}

std::vector<int64_t> TRTBase::get_static_shape(const char* name) {
  auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
  auto* engine = static_cast<nvinfer1::ICudaEngine*>(engine_);
  const nvinfer1::Dims dims = context->getTensorShape(name);
  std::vector<int64_t> shape;
  if (dims.nbDims > 0) shape.assign(dims.d, dims.d + dims.nbDims);
  if (std::find(shape.begin(), shape.end(), -1) == shape.end()) return shape;

  // Resolve dynamic dims from optimization profile 0's max shape.
  const nvinfer1::Dims mx =
      engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
  if (mx.nbDims > 0) return std::vector<int64_t>(mx.d, mx.d + mx.nbDims);

  // Fallback: replace dynamic dims with 1 to avoid crashes.
  for (auto& d : shape) {
    if (d == -1) d = 1;
  }
  if (shape.empty()) shape.push_back(1);
  return shape;
}

bool TRTBase::load(std::string& err) {
  if (!pre_load(err)) return false;

  std::ifstream f(engine_path_, std::ios::binary | std::ios::ate);
  if (!f) {
    err = "cannot open engine file: " + engine_path_;
    return false;
  }
  const std::streamoff size = f.tellg();
  if (size <= 0) {
    err = "engine file is empty: " + engine_path_;
    return false;
  }
  std::string blob(static_cast<size_t>(size), '\0');
  f.seekg(0);
  if (!f.read(blob.data(), size)) {
    err = "failed to read engine file: " + engine_path_;
    return false;
  }

  auto* runtime = nvinfer1::createInferRuntime(trt_logger());
  if (runtime == nullptr) {
    err = "createInferRuntime failed (no CUDA/TensorRT runtime?)";
    return false;
  }
  runtime_ = runtime;

  auto* engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (engine == nullptr) {
    err = "deserializeCudaEngine failed: " + engine_path_;
    return false;
  }
  engine_ = engine;

  auto* context = engine->createExecutionContext();
  if (context == nullptr) {
    err = "createExecutionContext failed: " + engine_path_;
    return false;
  }
  context_ = context;

  cudaStream_t stream = nullptr;
  if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate", err)) return false;
  stream_ = stream;

  // Buffer allocation, one page-locked host buffer + one device buffer per IO
  // tensor (mapped host memory on aarch64 instead of two buffers).
  const int32_t num_io = engine->getNbIOTensors();
  tensors_.reserve(num_io);
  for (int32_t i = 0; i < num_io; ++i) {
    const char* name = engine->getIOTensorName(i);
    TrtTensor t;
    t.name = name;
    t.is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    t.dtype = static_cast<int32_t>(engine->getTensorDataType(name));
    t.shape = get_static_shape(name);
    t.numel = std::accumulate(t.shape.begin(), t.shape.end(), int64_t{1},
                              std::multiplies<int64_t>());
    t.nbytes = static_cast<size_t>(t.numel) * dtype_size(t.dtype);
    if (dtype_size(t.dtype) == 0) {
      err = "unsupported tensor dtype for " + t.name;
      return false;
    }
#if TINYNAV_TRT_MAPPED_HOST
    if (!cuda_ok(cudaHostAlloc(&t.host, t.nbytes, cudaHostAllocMapped),
                 "cudaHostAlloc", err)) {
      return false;
    }
    if (!cuda_ok(cudaHostGetDevicePointer(&t.device, t.host, 0),
                 "cudaHostGetDevicePointer", err)) {
      return false;
    }
#else
    if (!cuda_ok(cudaMallocHost(&t.host, t.nbytes), "cudaMallocHost", err)) return false;
    if (!cuda_ok(cudaMalloc(&t.device, t.nbytes), "cudaMalloc", err)) return false;
#endif
    tensors_.push_back(t);
  }
  for (auto& t : tensors_) {
    (t.is_input ? inputs_ : outputs_).push_back(&t);
  }

  if (!on_loaded(err)) return false;

  // CUDA Graph capture; failure degrades to direct enqueueV3.
  std::string graph_err;
  if (!capture_graph(graph_err)) {
    use_graph_ = false;
    if (!graph_err.empty()) {
      std::fprintf(stderr, "[tinynav::trt] %s: no CUDA graph (%s); using enqueueV3\n",
                   engine_path_.c_str(), graph_err.c_str());
    }
  }
  return true;
}

bool TRTBase::capture_graph(std::string& err) {
  auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);

  // Ensure dynamic input shapes are specified before first execution.
  for (const auto* in : inputs_) {
    nvinfer1::Dims d;
    d.nbDims = static_cast<int32_t>(in->shape.size());
    for (size_t i = 0; i < in->shape.size(); ++i) d.d[i] = static_cast<int32_t>(in->shape[i]);
    if (!context->setInputShape(in->name.c_str(), d)) {
      err = "setInputShape failed for " + in->name;
      return false;
    }
  }
  for (const auto& t : tensors_) {
    if (!context->setTensorAddress(t.name.c_str(), t.device)) {
      err = "setTensorAddress failed for " + t.name;
      return false;
    }
  }

  cudaError_t cerr =
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (cerr != cudaSuccess) {
    err = std::string("cudaStreamBeginCapture: ") + cudaGetErrorString(cerr);
    cudaGetLastError();  // clear sticky state so the stream stays usable
    return false;
  }
  const bool enqueued = context->enqueueV3(stream);
  cudaGraph_t graph = nullptr;
  cerr = cudaStreamEndCapture(stream, &graph);
  if (!enqueued || cerr != cudaSuccess || graph == nullptr) {
    err = std::string("graph capture of enqueueV3 failed: ") +
          (cerr == cudaSuccess ? "enqueueV3 returned false" : cudaGetErrorString(cerr));
    cudaGetLastError();
    return false;
  }
  cudaGraphExec_t exec = nullptr;
  cerr = cudaGraphInstantiate(&exec, graph, 0ULL);
  cudaGraphDestroy(graph);
  if (cerr != cudaSuccess || exec == nullptr) {
    err = std::string("cudaGraphInstantiate: ") + cudaGetErrorString(cerr);
    cudaGetLastError();
    return false;
  }
  cudaStreamSynchronize(stream);
  graph_exec_ = exec;
  use_graph_ = true;
  return true;
}

bool TRTBase::run_graph(TrtOutputMap& results) {
  auto* context = static_cast<nvinfer1::IExecutionContext*>(context_);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  std::string err;

#if !TINYNAV_TRT_MAPPED_HOST
  for (const auto* in : inputs_) {
    if (!cuda_ok(cudaMemcpyAsync(in->device, in->host, in->nbytes,
                                 cudaMemcpyHostToDevice, stream),
                 "H2D copy of " + in->name, err)) {
      set_error(err);
      return false;
    }
  }
#endif

  if (use_graph_) {
    const cudaError_t cerr =
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec_), stream);
    if (!cuda_ok(cerr, "cudaGraphLaunch", err)) {
      set_error(err);
      return false;
    }
  } else if (!context->enqueueV3(stream)) {
    set_error("enqueueV3 failed");
    return false;
  }

#if !TINYNAV_TRT_MAPPED_HOST
  for (const auto* out : outputs_) {
    if (!cuda_ok(cudaMemcpyAsync(out->host, out->device, out->nbytes,
                                 cudaMemcpyDeviceToHost, stream),
                 "D2H copy of " + out->name, err)) {
      set_error(err);
      return false;
    }
  }
#endif

  if (!cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize", err)) {
    set_error(err);
    return false;
  }

  for (const auto* out : outputs_) {
    const int depth = cv_depth_for(out->dtype);
    if (depth < 0) {
      set_error("cannot map output dtype of " + out->name + " to a cv::Mat");
      return false;
    }
    std::vector<int> sizes(out->shape.begin(), out->shape.end());
    if (sizes.empty()) sizes.push_back(1);
    const int nd = static_cast<int>(sizes.size());
    cv::Mat m;
    if (static_cast<nvinfer1::DataType>(out->dtype) == nvinfer1::DataType::kINT64) {
      m.create(nd, sizes.data(), CV_64F);
      const auto* src = reinterpret_cast<const int64_t*>(out->host);
      auto* dst = reinterpret_cast<double*>(m.data);
      for (int64_t i = 0; i < out->numel; ++i) dst[i] = static_cast<double>(src[i]);
    } else {
      m.create(nd, sizes.data(), depth);
      std::memcpy(m.data, out->host, out->nbytes);
    }
    results[out->name] = m;
  }
  return true;
}

bool TRTBase::copy_to_input(size_t input_index, const cv::Mat& src) {
  if (input_index >= inputs_.size()) {
    set_error("input index out of range");
    return false;
  }
  const TrtTensor& t = *inputs_[input_index];
  if (cv_depth_for(t.dtype) < 0) {
    set_error("copy_to_input: unsupported dtype for " + t.name);
    return false;
  }
  if (src.empty()) {
    set_error("copy_to_input: empty source for " + t.name);
    return false;
  }
  cv::Mat cont = src.isContinuous() ? src : src.clone();
  const int64_t n = static_cast<int64_t>(cont.total());
  if (n > t.numel) {
    set_error("copy_to_input: source has " + std::to_string(n) + " elements, tensor " +
              t.name + " holds " + std::to_string(t.numel));
    return false;
  }
  for (int64_t i = 0; i < n; ++i) {
    write_elem(t.host, t.dtype, static_cast<size_t>(i),
               mat_elem_as_double(cont, static_cast<size_t>(i)));
  }
  return true;
}

void TRTBase::release() {
  if (graph_exec_ != nullptr) {
    cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
    graph_exec_ = nullptr;
  }
  for (auto& t : tensors_) {
    if (t.host != nullptr) cudaFreeHost(t.host);
#if !TINYNAV_TRT_MAPPED_HOST
    if (t.device != nullptr) cudaFree(t.device);
#endif
    t.host = nullptr;
    t.device = nullptr;
  }
  tensors_.clear();
  inputs_.clear();
  outputs_.clear();
  if (stream_ != nullptr) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
  if (context_ != nullptr) {
    delete static_cast<nvinfer1::IExecutionContext*>(context_);
    context_ = nullptr;
  }
  if (engine_ != nullptr) {
    delete static_cast<nvinfer1::ICudaEngine*>(engine_);
    engine_ = nullptr;
  }
  if (runtime_ != nullptr) {
    delete static_cast<nvinfer1::IRuntime*>(runtime_);
    runtime_ = nullptr;
  }
  use_graph_ = false;
}

}  // namespace tinynav::trt
