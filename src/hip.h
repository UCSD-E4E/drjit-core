/*
    src/hip.h -- HIP (AMD ROCm) backend declarations

    Target: AMD Instinct MI210 -- CDNA2, gfx90a, wavefront 64.

    Structural reference is the METAL backend, not CUDA/OptiX: Metal is the
    only backend in this tree that emits *source text* and integrates ray
    tracing inline, which is the shape this backend needs. See
    tools/hip_validate/BACKEND_NOTES.md for the mapping, and PLAN.md §3.1/§3.2
    for why source emission and the jit_metal_ray_trace contract were chosen.

    The runtime layer is the exception: the HIP driver API is a near 1:1
    analogue of CUDA's, so hip_api / hip_core port mechanically from
    cuda_api.cpp / cuda_core.cpp (PLAN.md §4).
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_HIP)

/// Initialize the HIP backend. Returns false if no usable device or runtime is
/// present, in which case the backend simply stays unregistered.
extern bool jitc_hip_init();

/// Release all resources held by the HIP backend.
extern void jitc_hip_shutdown();

#if !defined(DRJIT_HIP_CUDA_SHIM)
/// Look up a kernel of the precompiled device library by name.
///
/// Returns a hipFunction_t (as void *), or nullptr if the library is
/// unavailable on this device -- which happens when the device's wavefront
/// width differs from the one the blob was built for, and is reported once by
/// the loader. Callers must handle nullptr rather than assume; see the six
/// utility-kernel methods in hip_ts.cpp.
extern void *jitc_hip_kernel(int device, const char *name);
#endif

#if defined(DRJIT_HIP_CUDA_SHIM)
/// Compile generated HIP C++ source into a loadable module.
///
/// Under the shim this runs NVRTC over the source to obtain PTX and then hands
/// that to jitc_cuda_compile(), so the resulting CUmodule lands in the same
/// kernel.cuda.mod slot CUDAThreadState::launch() reads. On real hardware the
/// equivalent is hiprtc -> hipModuleLoadData; the call SHAPE is identical,
/// which tools/hip_validate/hipnv_pipeline.cpp verifies against the real API.
///
/// Returns (module, cache_hit), matching jitc_cuda_compile().
extern std::pair<void *, bool> jitc_hip_compile(const char *source);
#endif

// NOTE: jitc_hip_assemble / _reset / jitc_hip_format are declared in eval.h,
// not here. That mirrors the Metal backend and is deliberate: those entry
// points take ScheduledGroup, and eval.cpp must be able to call them without
// pulling in a backend header full of codegen macros.

#endif // DRJIT_ENABLE_HIP
