/*
    src/hip_ts.cpp -- ThreadState for the HIP backend (real runtime path)

    The mechanical cu* -> hip* port of cuda_ts.cpp that PLAN.md §4 anticipates.
    Compiled only when DRJIT_HIP_CUDA_SHIM is OFF; with the shim, hip_ts.h
    aliases HIPThreadState to CUDAThreadState and none of this is built.

    ------------------------------------------------------------------------
     What is and is not verifiable without an MI210
    ------------------------------------------------------------------------

    Every call below was checked for SHAPE -- signature, argument order, flag
    values, error convention -- by tools/hip_validate/hipnv_ts_calls.cpp, which
    compiles the same sequences against the real HIP headers and RUNS them
    through HIP-on-CUDA (§3.5). That catches the mechanical port's failure
    modes: a swapped argument, a CUDA flag value, a missing stream parameter.

    What it does not catch is anything AMD-specific about the runtime's
    behaviour. Those are on the same list as wave64 and fp16, and are why this
    file is Phase 1 rather than Phase 1-complete.

    ------------------------------------------------------------------------
     Two divergences from cuda_ts.cpp worth stating
    ------------------------------------------------------------------------

    1. hipMemcpyAsync takes an explicit DIRECTION argument; the CUDA driver API
       infers it from the pointers. hipMemcpyDefault (4) restores the inferring
       behaviour and is what unified addressing makes correct -- passing
       hipMemcpyHostToDevice by reflex would break device-to-device copies.

    2. The utility kernels (block_reduce, compress, mkperm, aggregate, ...) are
       NOT ported here. They are launches of precompiled kernels from
       resources/kernels.cu, which needs a gfx90a code object -- Phase 3. Each
       raises with that explanation instead of returning wrong data.
*/

#include "hip_ts.h"

#if defined(DRJIT_ENABLE_HIP) && !defined(DRJIT_HIP_CUDA_SHIM)

#include "hip_api.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "util.h"
#include "profile.h"
#include "malloc.h"

/// hipMemcpyKind. Only Default is used -- see divergence (1) above.
#define DR_HIP_MEMCPY_DEFAULT 4

/// Check a HIP return value and fail loudly with the runtime's own message.
#define hip_check(err)                                                         \
    do {                                                                       \
        hipError_t rv_ = (err);                                                \
        if (unlikely(rv_ != hipSuccess))                                       \
            jitc_hip_check_impl(rv_, #err, __FILE__, __LINE__);                \
    } while (0)

static void jitc_hip_check_impl(hipError_t rv, const char *expr,
                                const char *file, int line) {
    const char *msg = hipGetErrorString ? hipGetErrorString(rv) : nullptr;
    jitc_fail("hip_check(): API error %i (%s) in %s:%i -- \"%s\".", (int) rv,
              msg ? msg : "?", file, line, expr);
}

/// RAII guard that makes this thread state's context current.
struct scoped_set_hip_context {
    scoped_set_hip_context(void *ctx) {
        hip_check(hipCtxPushCurrent((hipCtx_t) ctx));
    }
    ~scoped_set_hip_context() {
        hipCtx_t unused;
        hip_check(hipCtxPopCurrent(&unused));
    }
};

// ---------------------------------------------------------------------------
//  Launch
// ---------------------------------------------------------------------------

Task *HIPThreadState::launch(Kernel kernel, KernelKey & /*key*/,
                             XXH128_hash_t /*hash*/, uint32_t size,
                             std::vector<void *> &kernel_params,
                             const std::vector<uint32_t> & /*param_ids*/,
                             KernelHistoryEntry *kernel_history_entry) {
    scoped_set_hip_context guard(hip_context);

    uint32_t block_size = kernel.hip.block_size,
             grid_size  = (size + block_size - 1) / block_size;

    // The kernel takes its Params struct BY VALUE (see hip_prologue.h), so the
    // launch passes a pointer to the packed parameter array as argument 0 --
    // matching how the CUDA path drives the same generated signature.
    void *args[] = { kernel_params.data() };

    if (unlikely(kernel_history_entry)) {
        hip_check(hipEventRecord((hipEvent_t) kernel_history_entry->event_start,
                                 (hipStream_t) hip_stream));
    }

    hip_check(hipModuleLaunchKernel((hipFunction_t) kernel.hip.func,
                                    grid_size, 1, 1,
                                    block_size, 1, 1,
                                    /* sharedMemBytes = */ 0,
                                    (hipStream_t) hip_stream,
                                    args, nullptr));

    if (unlikely(kernel_history_entry)) {
        hip_check(hipEventRecord((hipEvent_t) kernel_history_entry->event_end,
                                 (hipStream_t) hip_stream));
    }

    return nullptr;   // GPU backends are stream-ordered, not task-based
}

// ---------------------------------------------------------------------------
//  Memory
// ---------------------------------------------------------------------------

void HIPThreadState::memset_async(void *ptr, uint32_t size_, uint32_t isize,
                                  const void *src) {
    if (isize != 1 && isize != 2 && isize != 4 && isize != 8)
        jitc_raise("HIPThreadState::memset_async(): invalid element size %u!",
                   isize);

    jitc_trace("jit_memset_async(" DRJIT_PTR ", isize=%u, size=%u)",
               (uintptr_t) ptr, isize, size_);

    if (size_ == 0)
        return;

    scoped_set_hip_context guard(hip_context);

    // Broadcast the pattern to the widest native memset. 8-byte fills have no
    // direct form, so they go out as 4-byte fills when the two halves match
    // and fall back to a strided copy otherwise -- the same shape cuda_ts.cpp
    // uses, kept because the alternative silently truncates the pattern.
    switch (isize) {
        case 1:
            hip_check(hipMemsetD8Async((hipDeviceptr_t) ptr,
                                       ((uint8_t *) src)[0], size_,
                                       (hipStream_t) hip_stream));
            break;

        case 2:
            hip_check(hipMemsetD16Async((hipDeviceptr_t) ptr,
                                        ((uint16_t *) src)[0], size_,
                                        (hipStream_t) hip_stream));
            break;

        case 4:
            hip_check(hipMemsetD32Async((hipDeviceptr_t) ptr,
                                        ((int32_t *) src)[0], size_,
                                        (hipStream_t) hip_stream));
            break;

        case 8: {
            const uint32_t *v = (const uint32_t *) src;
            if (v[0] == v[1]) {
                hip_check(hipMemsetD32Async((hipDeviceptr_t) ptr, (int32_t) v[0],
                                            size_ * 2,
                                            (hipStream_t) hip_stream));
            } else {
                jitc_raise("HIPThreadState::memset_async(): 8-byte fills with a "
                           "non-uniform pattern need the utility kernels from "
                           "resources/kernels.cu (Phase 3).");
            }
            break;
        }
    }
}

void HIPThreadState::memcpy(void *dst, const void *src, size_t size) {
    scoped_set_hip_context guard(hip_context);
    hip_check(hipMemcpyAsync(dst, src, size, DR_HIP_MEMCPY_DEFAULT,
                             (hipStream_t) hip_stream));
    hip_check(hipStreamSynchronize((hipStream_t) hip_stream));
}

void HIPThreadState::memcpy_async(void *dst, const void *src, size_t size) {
    scoped_set_hip_context guard(hip_context);
    hip_check(hipMemcpyAsync(dst, src, size, DR_HIP_MEMCPY_DEFAULT,
                             (hipStream_t) hip_stream));
}

void HIPThreadState::poke(void *dst, const void *src, uint32_t size) {
    jitc_trace("jit_poke(" DRJIT_PTR ", size=%u)", (uintptr_t) dst, size);

    // The CUDA backend uses a tiny precompiled kernel to avoid a host round
    // trip. That kernel is Phase 3; until then a staged async copy is correct,
    // just slower. Staged rather than direct because `src` is caller-owned
    // host memory that may die before the copy retires.
    scoped_set_hip_context guard(hip_context);

    void *tmp = jitc_malloc(JitBackend::HIP, size, /*shared=*/true);
    std::memcpy(tmp, src, size);
    hip_check(hipMemcpyAsync(dst, tmp, size, DR_HIP_MEMCPY_DEFAULT,
                             (hipStream_t) hip_stream));
    jitc_free(tmp);
}

// ---------------------------------------------------------------------------
//  Ordering
// ---------------------------------------------------------------------------

void HIPThreadState::enqueue_host_func(void (*callback)(void *),
                                       void *payload) {
    scoped_set_hip_context guard(hip_context);
    hip_check(hipLaunchHostFunc((hipStream_t) hip_stream, callback, payload));
}

void HIPThreadState::barrier() {
    // Single-stream ordering already serialises launches; the only work here is
    // draining allocations parked for deferred release (mirrors CUDA).
    if (!free_later.empty())
        flush_deferred_free();
}

void HIPThreadState::flush_deferred_free() {
    if (void *batch = take_deferred_free()) {
        scoped_set_hip_context guard(hip_context);
        hip_check(hipLaunchHostFunc((hipStream_t) hip_stream,
                                    jitc_malloc_release_batch, batch));
    }
}

// ---------------------------------------------------------------------------
//  Phase 3 -- precompiled utility kernels
// ---------------------------------------------------------------------------
//
// These are not API calls. Each needs resources/kernels.cu built as a gfx90a
// code object and loaded like the CUDA backend loads its PTX blob. Raising is
// deliberate: returning zeros or an unmodified buffer would corrupt results
// far from here.

#define DR_HIP_PHASE3(name)                                                    \
    jitc_raise("HIPThreadState::" name "(): not implemented. This launches a "  \
               "precompiled utility kernel from resources/kernels.cu, which "   \
               "needs a gfx90a code object -- PLAN.md Phase 3.")

void HIPThreadState::aggregate(void *, AggregationEntry *, uint32_t) {
    DR_HIP_PHASE3("aggregate");
}

void HIPThreadState::block_reduce(VarType, ReduceOp, uint32_t, uint32_t,
                                  const void *, void *) {
    DR_HIP_PHASE3("block_reduce");
}

void HIPThreadState::block_prefix_reduce(VarType, ReduceOp, uint32_t, uint32_t,
                                         bool, bool, const void *, void *) {
    DR_HIP_PHASE3("block_prefix_reduce");
}

void HIPThreadState::reduce_dot(VarType, const void *, const void *, uint32_t,
                                void *) {
    DR_HIP_PHASE3("reduce_dot");
}

uint32_t HIPThreadState::compress(const uint8_t *, uint32_t, uint32_t *) {
    DR_HIP_PHASE3("compress");
}

uint32_t HIPThreadState::block_mkperm(const uint32_t *, uint32_t, uint32_t,
                                      uint32_t, uint32_t *, uint32_t *) {
    DR_HIP_PHASE3("block_mkperm");
}

// --- Off the critical path rather than deferred -----------------------------

void HIPThreadState::batched_gemm(VarType, bool, bool, uint32_t, uint32_t,
                                  uint32_t, const GemmBatch *, const void *,
                                  const void *, void *) {
    jitc_raise("HIPThreadState::batched_gemm(): not implemented. The CUDA "
               "backend routes this to cuBLAS; the HIP equivalent is rocBLAS, "
               "which no Mitsuba variant we target requires.");
}

void HIPThreadState::coop_vec_pack(uint32_t, const void *, const MatrixDescr *,
                                   void *, const MatrixDescr *) {
    jitc_raise("HIPThreadState::coop_vec_pack(): not implemented. Cooperative "
               "vectors are an OptiX/Metal feature with no HIP analogue.");
}

#endif // DRJIT_ENABLE_HIP && !DRJIT_HIP_CUDA_SHIM
