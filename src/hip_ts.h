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

    REAL (Phase 1). A standalone implementation over the HIP runtime API, the
    mechanical cu* -> hip* port of cuda_ts.cpp that §4 anticipates. The API
    surface it targets is proven by tools/hip_validate/hipnv_pipeline.cpp, and
    the exact call shapes it emits are checked by hipnv_ts_calls.cpp.

    What it deliberately does NOT implement: block_reduce, block_prefix_reduce,
    reduce_dot, compress, block_mkperm and aggregate. Those are not API calls,
    they are LAUNCHES OF PRECOMPILED UTILITY KERNELS -- resources/kernels.cu
    compiled to a gfx90a code object, which is Phase 3. Each raises with that
    explanation rather than silently returning wrong data.
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

#include "hip_api.h"

struct HIPThreadState final : ThreadState {
    // --- Execution -----------------------------------------------------------

    Task *launch(Kernel kernel, KernelKey &key, XXH128_hash_t hash,
                 uint32_t size, std::vector<void *> &kernel_params,
                 const std::vector<uint32_t> &kernel_param_ids,
                 KernelHistoryEntry *kernel_history_entry) override;

    // --- Memory --------------------------------------------------------------

    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override;
    void memcpy(void *dst, const void *src, size_t size) override;
    void memcpy_async(void *dst, const void *src, size_t size) override;
    void poke(void *dst, const void *src, uint32_t size) override;

    // --- Ordering ------------------------------------------------------------

    void enqueue_host_func(void (*callback)(void *), void *payload) override;
    void barrier() override;
    void flush_deferred_free() override;

    // --- Phase 3: these launch precompiled utility kernels ---------------------
    //
    // Not API calls -- they need resources/kernels.cu built as a gfx90a code
    // object. Each raises rather than returning wrong data, and says which
    // phase supplies it.

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override;
    void block_reduce(VarType vt, ReduceOp op, uint32_t size,
                      uint32_t block_size, const void *in, void *out) override;
    void block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                             uint32_t block_size, bool exclusive, bool reverse,
                             const void *in, void *out) override;
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override;
    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override;
    uint32_t block_mkperm(const uint32_t *values, uint32_t size,
                          uint32_t block_size, uint32_t bucket_count,
                          uint32_t *perm, uint32_t *offsets) override;

    // Not Phase 3 so much as not-on-the-path: matmul goes through rocBLAS and
    // cooperative vectors are an OptiX/Metal feature with no HIP analogue yet.
    // Both raise; neither is needed by any Mitsuba variant we target.
    void batched_gemm(VarType type, bool At, bool Bt, uint32_t M, uint32_t N,
                      uint32_t K, const GemmBatch *batch, const void *A,
                      const void *B, void *C) override;
    void coop_vec_pack(uint32_t count, const void *in, const MatrixDescr *in_d,
                       void *out, const MatrixDescr *out_d) override;
};

#endif // DRJIT_HIP_CUDA_SHIM

#endif // DRJIT_ENABLE_HIP
