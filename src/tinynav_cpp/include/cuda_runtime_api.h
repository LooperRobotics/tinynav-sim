// Compatibility shim, NOT a real CUDA header.
//
// In the uniflexai/tinynav image the CUDA headers live under
// /usr/local/cuda/include, which is not on g++'s default search path, while
// the system TensorRT headers (/usr/include/x86_64-linux-gnu/NvInfer*.h)
// hard-include <cuda_runtime_api.h>. This shim forwards to the real header so
// plain `g++ -Iinclude -I/usr/include/x86_64-linux-gnu` invocations (the
// syntax-check recipe in the porting notes) resolve it. When the CUDA include
// dir is on the search path after this one (the colcon build adds
// CUDAToolkit_INCLUDE_DIRS), __has_include_next finds the real header first
// and this file is inert. All of the real header's own includes are quoted,
// so they resolve relative to /usr/local/cuda/include either way.
#ifndef TINYNAV_CPP_CUDA_RUNTIME_API_SHIM_H
#define TINYNAV_CPP_CUDA_RUNTIME_API_SHIM_H

#if defined(__GNUC__) && defined(__has_include_next)
#  if __has_include_next(<cuda_runtime_api.h>)
#    include_next <cuda_runtime_api.h>
#    define TINYNAV_CPP_CUDA_SHIM_RESOLVED 1
#  endif
#endif

#ifndef TINYNAV_CPP_CUDA_SHIM_RESOLVED
#  include "/usr/local/cuda/include/cuda_runtime_api.h"
#endif

#endif  // TINYNAV_CPP_CUDA_RUNTIME_API_SHIM_H
