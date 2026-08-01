/*
    src/hip_ts.h -- ThreadState for the HIP backend

    Two very different implementations behind one name, selected at build time.

    SHIM (-DDRJIT_HIP_CUDA_SHIM). HIPThreadState DERIVES FROM CUDAThreadState
    and overrides nothing. Allocation, launch, memcpy, reductions and every
    other ThreadState duty are inherited and really execute, on a real device
    (PLAN.md §0.3).

    That works because the two halves of a backend are separable: codegen
    decides WHAT to run, the ThreadState decides HOW to run it. Under the shim
    we swap the second half for one that is already written and tested, so
    hip_eval.cpp -- the 6-10 week block and the actual risk -- can be developed
    against real execution instead of against nothing.

    It is a scaffold, and its limits are exactly the CUDA runtime's: warp width
    is 32, kernels are PTX rather than gfx90a code objects, and NONE of the HIP
    API is exercised. The last point is why hipnv_pipeline.cpp exists
    separately -- it covers the real hipMalloc / hipModuleLaunchKernel surface
    that this deliberately bypasses (§3.5).

    REAL (Phase 1). A standalone implementation over the HIP driver API,
    structured as the mechanical cu* -> hip* port of cuda_ts.cpp that §4
    anticipates. Not written yet; the API surface it will target is already
    proven by tools/hip_validate/hipnv_pipeline.cpp.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_HIP)

#if defined(DRJIT_HIP_CUDA_SHIM)

#include "cuda_ts.h"

/// An ALIAS rather than a subclass, for two reasons. CUDAThreadState is
/// declared `final`, and un-finalising it would be an upstream change for a
/// dev scaffold's benefit -- exactly the kind of seam §9 says to avoid. An
/// alias also makes the situation honest: under the shim this IS a CUDA thread
/// state, not a HIP one wearing a hat.
///
/// Safe because cuda_ts.cpp never branches on `ts->backend`; it drives the
/// device through ts->stream / ts->context, which jitc_init_thread_state()
/// populates from the CUDA device.
using HIPThreadState = CUDAThreadState;

#else

// Phase 1 lands the real implementation here. Declaring it rather than
// defining it keeps the build honest -- enabling DRJIT_ENABLE_HIP without the
// shim fails to link rather than silently doing something wrong.
struct HIPThreadState;

#endif // DRJIT_HIP_CUDA_SHIM

#endif // DRJIT_ENABLE_HIP
