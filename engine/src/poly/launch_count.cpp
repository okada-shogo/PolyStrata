// src/poly/launch_count.cpp - the profiler's launch counter (profile.hpp launch_count()): the runtime's kernel launch
// calls wrapped at link time (-Wl,--wrap=..., Linux only; engine/cmake/strata_poly.cmake), so every launch the
// engine's own code makes adds one while the profiler is on.  With the profiler off the wrapper only reads a flag and
// calls the runtime.  The calls a BLAS library makes inside itself do not come through here.
#include "profile.hpp"

#include <cuda_runtime.h>

namespace {
const bool linked = (strata::poly::prof::launch_wrapper_linked().store(true), true);

inline void count() {
    if (strata::poly::prof::launch_counting().load(std::memory_order_relaxed))
        strata::poly::prof::launch_count().fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

extern "C" {
#if defined(STRATA_USE_HIP)
hipError_t __real_hipLaunchKernel(const void* f, dim3 grid, dim3 block, void** args, size_t shared, hipStream_t s);
hipError_t __wrap_hipLaunchKernel(const void* f, dim3 grid, dim3 block, void** args, size_t shared, hipStream_t s) {
    count();
    return __real_hipLaunchKernel(f, grid, block, args, shared, s);
}
#else
cudaError_t __real_cudaLaunchKernel(const void* f, dim3 grid, dim3 block, void** args, size_t shared, cudaStream_t s);
cudaError_t __wrap_cudaLaunchKernel(const void* f, dim3 grid, dim3 block, void** args, size_t shared, cudaStream_t s) {
    count();
    return __real_cudaLaunchKernel(f, grid, block, args, shared, s);
}
cudaError_t __real_cudaLaunchKernelExC(const cudaLaunchConfig_t* c, const void* f, void** args);
cudaError_t __wrap_cudaLaunchKernelExC(const cudaLaunchConfig_t* c, const void* f, void** args) {
    count();
    return __real_cudaLaunchKernelExC(c, f, args);
}
#endif
}
