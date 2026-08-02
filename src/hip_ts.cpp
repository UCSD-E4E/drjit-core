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

    2. The utility kernels (block_reduce, compress, mkperm, aggregate, ...) do
       not map to API calls at all -- they launch precompiled kernels from
       resources/kernels.cu. Their launch configurations mirror cuda_ts.cpp,
       except that every warp-size constant is read from HIPDevice instead of
       being the literal 32 the CUDA original can afford (PLAN.md §3.3).
*/

#include "hip_ts.h"

#if defined(DRJIT_ENABLE_HIP) && !defined(DRJIT_HIP_CUDA_SHIM)

#include "hip.h"
#include "hip_api.h"
#include "log.h"
#include "var.h"
#include "eval.h"
#include "util.h"
#include "profile.h"
#include "malloc.h"
#include <algorithm>

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
//  Precompiled utility kernels
// ---------------------------------------------------------------------------
//
// These are not API calls: each launches a kernel from resources/kernels.cu,
// shipped as the gfx90a code object that jitc_hip_kernel() resolves against.
// The launch configurations mirror cuda_ts.cpp, with one systematic change --
// every warp-size constant is read from HIPDevice rather than assumed to be
// 32 (PLAN.md §3.3). Those sites are marked below; each is a place where the
// CUDA original would compute a wrong answer rather than fail on gfx90a.

/// Sum/product/and/or over a signed type is bit-identical to the unsigned one,
/// so half the reduction kernels need not be instantiated (as in cuda_ts.cpp).
static VarType make_int_type_unsigned(VarType type) {
    switch (type) {
        case VarType::Int8:  return VarType::UInt8;
        case VarType::Int16: return VarType::UInt16;
        case VarType::Int32: return VarType::UInt32;
        case VarType::Int64: return VarType::UInt64;
        default: return type;
    }
}

/// Launch a device-library kernel, honouring KernelHistory and LaunchBlocking.
static void hip_submit(KernelType type, KernelRecordingMode recording_mode,
                       void *func, uint32_t block_count_x,
                       uint32_t thread_count, uint32_t shared_mem_bytes,
                       void *stream, void **args, uint32_t width,
                       uint32_t block_count_y = 1) {
    KernelHistoryEntry entry = {};
    uint32_t flags = jit_flags();

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        hip_check(hipEventCreateWithFlags((hipEvent_t *) &entry.event_start, 0));
        hip_check(hipEventCreateWithFlags((hipEvent_t *) &entry.event_end, 0));
        hip_check(hipEventRecord((hipEvent_t) entry.event_start,
                                 (hipStream_t) stream));
    }

    hip_check(hipModuleLaunchKernel((hipFunction_t) func, block_count_x,
                                    block_count_y, 1, thread_count, 1, 1,
                                    shared_mem_bytes, (hipStream_t) stream,
                                    args, nullptr));

    if (unlikely(flags & (uint32_t) JitFlag::LaunchBlocking))
        hip_check(hipStreamSynchronize((hipStream_t) stream));

    if (unlikely(flags & (uint32_t) JitFlag::KernelHistory)) {
        entry.backend = JitBackend::HIP;
        entry.type = type;
        entry.recording_mode = recording_mode;
        entry.size = width;
        entry.input_count = 1;
        entry.output_count = 1;
        hip_check(hipEventRecord((hipEvent_t) entry.event_end,
                                 (hipStream_t) stream));
        state.kernel_history.append(entry);
    }
}

/// Resolve a device-library kernel, or fail with the name that was missing.
static void *hip_kernel(int device, const char *name) {
    void *func = jitc_hip_kernel(device, name);
    if (unlikely(!func))
        jitc_raise("jit_hip(): the precompiled kernel \"%s\" is unavailable. "
                   "Either the device library does not match this device (see "
                   "the preceding warning) or it was built without this "
                   "variant.", name);
    return func;
}

void HIPThreadState::aggregate(void *dst, AggregationEntry *agg,
                               uint32_t size) {
    scoped_set_hip_context guard(hip_context);
    const HIPDevice &dev = state.hip_devices[device];
    void *func = hip_kernel(device, "aggregate");
    void *args[] = { &dst, &agg, &size };

    uint32_t block_count, thread_count;
    dev.get_launch_config(&block_count, &thread_count, size);

    jitc_log(InfoSym,
             "jit_aggregate(" DRJIT_PTR " -> " DRJIT_PTR
             ", size=%u, blocks=%u, threads=%u)",
             (uintptr_t) agg, (uintptr_t) dst, size, block_count, thread_count);

    hip_submit(KernelType::Aggregate, this->recording_mode, func, block_count,
               thread_count, 0, hip_stream, args, 1);
}

void HIPThreadState::block_reduce(VarType vt, ReduceOp op, uint32_t size,
                                  uint32_t block_size, const void *in,
                                  void *out) {
    // The documentation of resources/block_reduce.cuh explains the geometry.
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise("jit_block_reduce(): invalid block size (size=%u, "
                   "block_size=%u)!", size, block_size);
    }

    const HIPDevice &dev = state.hip_devices[device];
    uint32_t warp_size = dev.warp_size;   // §3.3

    uint32_t tsize = type_size[(int) vt];
    if (block_size == 1) {
        memcpy_async(out, in, size * tsize);
        return;
    }

    VarType vts = vt;
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And) {
        vt = make_int_type_unsigned(vt);
        if (vt == VarType::Float16)
            vts = VarType::Float32;
    }

    uint32_t block_count = ceil_div(size, block_size),
             chunk_size  = round_pow2(block_size);

    uint32_t thread_count, grid_dim_x, grid_dim_y, chunk_count;
    uint32_t chunks_per_block, chunks_per_thread_block;
    bool x_is_block_id = true;
    uint32_t vector_width = 1;

    if (chunk_size < 1024) {
        // §3.3: a partial wavefront must not enter the kernel's cross-lane
        // reduction, so the thread count is rounded to whole wavefronts.
        uint32_t min_threads = align_up(block_count * chunk_size, warp_size);
        thread_count = std::min(std::max(chunk_size, 128u), min_threads);

        chunk_count = block_count;
        chunks_per_block = 1;
        chunks_per_thread_block = thread_count / chunk_size;

        grid_dim_x = ceil_div(block_count, chunks_per_thread_block);
        grid_dim_y = 1;
    } else {
        if ((block_size * tsize) % 16 == 0 && (size * tsize) % 16 == 0 &&
            ((uintptr_t) in) % 16 == 0)
            vector_width = 16 / tsize;

        chunk_size = 1024;
        thread_count = chunk_size / vector_width;

        chunks_per_block = ceil_div(block_size, chunk_size);
        chunks_per_thread_block = 1;

        grid_dim_x = block_count;
        grid_dim_y = chunks_per_block;
        chunk_count = block_count * chunks_per_block;

        if (grid_dim_y > grid_dim_x) {
            std::swap(grid_dim_x, grid_dim_y);
            x_is_block_id = false;
        }
    }

    // §3.3: the kernel's first cross-lane pass collapses each wavefront, so
    // the shared-memory staging area is sized by the device width. Assuming 32
    // here would over-allocate on gfx90a rather than corrupt -- but it also
    // feeds the recursion bound below, which would then be wrong.
    uint32_t after_stage_1 =
        chunk_size / (std::min(chunk_size, warp_size) * vector_width);

    uint32_t smem_per_chunk =
                 (after_stage_1 == 1 ? 0 : after_stage_1) * type_size[(int) vts],
             smem_bytes = smem_per_chunk * chunks_per_thread_block;

    jitc_log(Debug,
             "jit_block_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, block_count=%u, "
             "chunk_size=%u, chunks_per_block=%u, vector_width=%u): launching "
             "a %u x %u grid with %u threads and %u bytes of shared memory per "
             "thread block.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, block_count, chunk_size,
             chunks_per_block, vector_width, grid_dim_x, grid_dim_y,
             thread_count, smem_bytes);

    char name[128];
    if (vector_width != 1)
        snprintf(name, sizeof(name), "block_reduce_%s_%s_vec_1024",
                 red_name[(int) op], type_name_short[(int) vt]);
    else
        snprintf(name, sizeof(name), "block_reduce_%s_%s_%u",
                 red_name[(int) op], type_name_short[(int) vt], chunk_size);

    void *func = hip_kernel(device, name);

    struct {
        const void *in;
        void *out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunks_per_block;
        uint32_t chunk_count;
        uint8_t x_is_block_id;
    } params;

    params.in = in;
    params.size = size / vector_width;
    params.block_size = block_size / vector_width;
    params.chunks_per_block = chunks_per_block;
    params.chunk_count = chunk_count;
    params.x_is_block_id = x_is_block_id;
    params.out = (chunks_per_block == 1)
                     ? out
                     : jitc_malloc(backend, chunk_count * tsize);

    {
        scoped_set_hip_context guard(hip_context);
        void *args[] = { &params };
        hip_submit(KernelType::BlockReduce, this->recording_mode, func,
                   grid_dim_x, thread_count, smem_bytes, hip_stream, args, size,
                   grid_dim_y);
    }

    if (chunks_per_block > 1) {
        block_reduce(vt, op, chunk_count, chunks_per_block, params.out, out);
        jitc_free(params.out);
    }
}

void HIPThreadState::block_prefix_reduce(VarType vt, ReduceOp op, uint32_t size,
                                         uint32_t block_size, bool exclusive,
                                         bool reverse, const void *in,
                                         void *out) {
    uint32_t tsize = type_size[(int) vt];
    if (size == 0) {
        return;
    } else if (block_size == 0 || block_size > size) {
        jitc_raise("jit_block_prefix_reduce(): invalid block size (size=%u, "
                   "block_size=%u)!", size, block_size);
    } else if (block_size == 1) {
        if (exclusive) {
            uint64_t ident = jitc_reduce_identity(vt, op);
            memset_async(out, size, tsize, &ident);
        } else if (in != out) {
            memcpy_async(out, in, size * tsize);
        }
        return;
    }

    const HIPDevice &dev = state.hip_devices[device];
    uint32_t warp_size = dev.warp_size;   // §3.3

    VarType vts = vt;
    if (op == ReduceOp::Add || op == ReduceOp::Mul ||
        op == ReduceOp::Or || op == ReduceOp::And) {
        vt = make_int_type_unsigned(vt);
        if (vt == VarType::Float16)
            vts = VarType::Float32;
    }

    uint32_t block_count = ceil_div(size, block_size),
             chunk_size  = round_pow2(block_size);

    uint32_t thread_count, grid_dim_x, grid_dim_y, chunk_count;
    uint32_t chunks_per_block, chunks_per_thread_block;
    bool x_is_block_id = true;

    if (chunk_size < 1024) {
        uint32_t min_threads = align_up(block_count * chunk_size, warp_size);
        thread_count = std::min(std::max(chunk_size, 128u), min_threads);

        chunk_count = block_count;
        chunks_per_block = 1;
        chunks_per_thread_block = thread_count / chunk_size;

        grid_dim_x = ceil_div(block_count, chunks_per_thread_block);
        grid_dim_y = 1;
    } else {
        chunk_size = thread_count = 1024;

        chunks_per_block = ceil_div(block_size, chunk_size);
        chunks_per_thread_block = 1;

        grid_dim_x = block_count;
        grid_dim_y = chunks_per_block;
        chunk_count = block_count * chunks_per_block;

        if (grid_dim_y > grid_dim_x) {
            std::swap(grid_dim_x, grid_dim_y);
            x_is_block_id = false;
        }
    }

    uint32_t smem_bytes = thread_count * type_size[(int) vts];

    jitc_log(Debug,
             "jit_block_prefix_reduce(" DRJIT_PTR " -> " DRJIT_PTR
             ", type=%s, op=%s, size=%u, block_size=%u, exclusive=%i, "
             "reverse=%i, block_count=%u, chunk_size=%u, chunks_per_block=%u): "
             "launching a %u x %u grid with %u threads and %u bytes of shared "
             "memory per thread block.",
             (uintptr_t) in, (uintptr_t) out, type_name[(int) vt],
             red_name[(int) op], size, block_size, exclusive, reverse,
             block_count, chunk_size, chunks_per_block, grid_dim_x, grid_dim_y,
             thread_count, smem_bytes);

    char name[128];
    snprintf(name, sizeof(name), "block_prefix_reduce_%s_%s_%u",
             red_name[(int) op], type_name_short[(int) vt], chunk_size);
    void *func = hip_kernel(device, name);

    struct {
        const void *in;
        void *scratch;
        void *out;
        uint32_t size;
        uint32_t block_size;
        uint32_t chunks_per_block;
        bool x_is_block_id;
        bool exclusive;
        bool reverse;
    } params;

    params.in = in;
    params.out = out;
    params.size = size;
    params.block_size = block_size;
    params.chunks_per_block = chunks_per_block;
    params.x_is_block_id = x_is_block_id;
    params.exclusive = exclusive;
    params.reverse = reverse;

    if (chunks_per_block > 1) {
        uint32_t scratch_size = chunk_count * 2,
                 vsize = type_size[(int) vts];
        params.scratch = jitc_malloc(backend, scratch_size * vsize);
        uint64_t z = 0;
        memset_async(params.scratch, scratch_size, vsize, &z);
    } else {
        params.scratch = nullptr;
    }

    {
        scoped_set_hip_context guard(hip_context);
        void *args[] = { &params };
        hip_submit(KernelType::BlockPrefixReduce, this->recording_mode, func,
                   grid_dim_x, thread_count, smem_bytes, hip_stream, args, size,
                   grid_dim_y);
    }

    if (chunks_per_block > 1)
        jitc_free(params.scratch);
}

void HIPThreadState::reduce_dot(VarType vt, const void *ptr_1,
                                const void *ptr_2, uint32_t size, void *out) {
    const HIPDevice &dev = state.hip_devices[device];

    char name[128];
    snprintf(name, sizeof(name), "reduce_dot_%s", type_name_short[(int) vt]);
    void *func = hip_kernel(device, name);

    uint32_t thread_count = 1024,
             tsize = type_size[(int) vt],
             shared_size = thread_count * tsize,
             block_count = (size + thread_count * 2 - 1) / (thread_count * 2);

    block_count = std::min(dev.sm_count * 4, block_count);

    jitc_log(Debug, "jit_reduce_dot(" DRJIT_PTR ", " DRJIT_PTR
             ", type=%s, size=%u, smem=%u, blocks=%u)",
             (uintptr_t) ptr_1, (uintptr_t) ptr_2, type_name[(int) vt],
             size, shared_size, block_count);

    scoped_set_hip_context guard(hip_context);
    if (block_count == 1) {
        void *args[] = { &ptr_1, &ptr_2, &size, &out };
        hip_submit(KernelType::Dot, this->recording_mode, func, 1, thread_count,
                   shared_size, hip_stream, args, size);
    } else {
        void *temp = jitc_malloc(backend, block_count * (size_t) tsize);

        void *args_1[] = { &ptr_1, &ptr_2, &size, &temp };
        hip_submit(KernelType::Dot, this->recording_mode, func, block_count,
                   thread_count, shared_size, hip_stream, args_1, size);

        block_reduce(vt, ReduceOp::Add, block_count, block_count, temp, out);
        jitc_free(temp);
    }
}

uint32_t HIPThreadState::compress(const uint8_t *in, uint32_t size,
                                  uint32_t *out) {
    if (size == 0)
        return 0;

    const HIPDevice &dev = state.hip_devices[device];
    uint32_t warp_size = dev.warp_size;   // §3.3
    scoped_set_hip_context guard(hip_context);

    uint32_t *count_out = (uint32_t *) jitc_malloc(
        backend, sizeof(uint32_t), /*shared=*/true);

    if (size <= 4096) {
        uint32_t items_per_thread = 4,
                 thread_count = round_pow2((size + items_per_thread - 1) /
                                           items_per_thread),
                 shared_size = thread_count * 2 * sizeof(uint32_t),
                 trailer = thread_count * items_per_thread - size;

        jitc_log(Debug,
                 "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, type=small, threads=%u, shared=%u)",
                 (uintptr_t) in, (uintptr_t) out, size, thread_count,
                 shared_size);

        if (trailer > 0)
            hip_check(hipMemsetD8Async((hipDeviceptr_t) (in + size), 0, trailer,
                                       (hipStream_t) hip_stream));

        void *func = hip_kernel(device, "compress_small");
        void *args[] = { &in, &out, &size, &count_out };
        hip_submit(KernelType::Compress, this->recording_mode, func, 1,
                   thread_count, shared_size, hip_stream, args, size);
    } else {
        uint32_t items_per_thread = 16,
                 thread_count = 128,
                 items_per_block = items_per_thread * thread_count,
                 block_count = (size + items_per_block - 1) / items_per_block,
                 shared_size = items_per_block * sizeof(uint32_t),
                 // §3.3: compress_large's decoupled lookback starts at
                 // `lane - WarpSize` and reads before block 0. The padding it
                 // reads instead is one wavefront wide, and
                 // compress_large_init marks exactly that many slots complete.
                 scratch_items = block_count + warp_size,
                 trailer = items_per_block * block_count - size;

        jitc_log(Debug,
                 "jit_compress(" DRJIT_PTR " -> " DRJIT_PTR
                 ", size=%u, type=large, blocks=%u, threads=%u, shared=%u, "
                 "scratch=%u)",
                 (uintptr_t) in, (uintptr_t) out, size, block_count,
                 thread_count, shared_size, scratch_items * 4);

        uint64_t *scratch = (uint64_t *) jitc_malloc(
            backend, scratch_items * sizeof(uint64_t));

        uint32_t block_count_init, thread_count_init;
        dev.get_launch_config(&block_count_init, &thread_count_init,
                              scratch_items);

        void *func_init = hip_kernel(device, "compress_large_init");
        void *args[] = { &scratch, &scratch_items };
        hip_submit(KernelType::Compress, this->recording_mode, func_init,
                   block_count_init, thread_count_init, 0, hip_stream, args,
                   scratch_items);

        if (trailer > 0)
            hip_check(hipMemsetD8Async((hipDeviceptr_t) (in + size), 0, trailer,
                                       (hipStream_t) hip_stream));

        void *func = hip_kernel(device, "compress_large");
        scratch += warp_size; // move beyond padding area
        void *args_2[] = { &in, &out, &scratch, &count_out };
        hip_submit(KernelType::Compress, this->recording_mode, func,
                   block_count, thread_count, shared_size, hip_stream, args_2,
                   scratch_items);
        scratch -= warp_size;

        jitc_free(scratch);
    }

    jitc_sync_thread(this);
    uint32_t count_out_v = *count_out;
    jitc_free(count_out);
    return count_out_v;
}

/// Transpose a batch of uint32 matrices in place of cuda_transpose().
static void hip_transpose(ThreadState *ts, const uint32_t *in, uint32_t *out,
                          uint32_t rows, uint32_t cols,
                          uint32_t num_batches = 1, uint32_t batch_stride = 0) {
    uint32_t blocks_x = (cols + 15u) / 16u,
             blocks_y = (rows + 15u) / 16u;

    scoped_set_hip_context guard(ts->hip_context);
    jitc_log(Debug,
             "jit_transpose(" DRJIT_PTR " -> " DRJIT_PTR
             ", rows=%u, cols=%u, blocks=%ux%u, batches=%u)",
             (uintptr_t) in, (uintptr_t) out, rows, cols, blocks_x, blocks_y,
             num_batches);

    void *args[] = { &in, &out, &rows, &cols, &batch_stride };

    // 16x16 threads and a Z grid dimension: hip_submit() only exposes a 1D
    // thread block and 2D grid, so this one launches directly.
    hip_check(hipModuleLaunchKernel(
        (hipFunction_t) hip_kernel(ts->device, "transpose"), blocks_x, blocks_y,
        num_batches, 16, 16, 1, 16 * 17 * sizeof(uint32_t),
        (hipStream_t) ts->hip_stream, args, nullptr));
}

uint32_t HIPThreadState::block_mkperm(const uint32_t *ptr, uint32_t size,
                                      uint32_t block_size,
                                      uint32_t bucket_count, uint32_t *perm,
                                      uint32_t *offsets) {
    if (size == 0)
        return 0;
    else if (unlikely(bucket_count == 0))
        jitc_fail("jit_block_mkperm(): bucket_count cannot be zero!");

    scoped_set_hip_context guard(hip_context);
    const HIPDevice &dev = state.hip_devices[device];

    uint32_t warp_size = dev.warp_size;   // §3.3

    uint32_t n_blocks = ceil_div(size, block_size);

    uint32_t gpu_blocks_per_group, thread_count;
    dev.get_launch_config(&gpu_blocks_per_group, &thread_count, block_size,
                          1024, 1);

    // Always launch full wavefronts: the kernels ballot across the whole
    // wavefront, so a partial one would vote with uninitialized lanes.
    uint32_t warp_count = (thread_count + warp_size - 1) / warp_size;
    thread_count = warp_count * warp_size;

    uint32_t gpu_block_count = n_blocks * gpu_blocks_per_group;

    uint32_t bucket_size_1 = bucket_count * sizeof(uint32_t),
             bucket_size_all = bucket_size_1 * gpu_block_count;

    uint32_t shared_size = 0;
    const char *variant = nullptr, *phase_1_name = nullptr,
               *phase_4_name = nullptr;
    bool initialize_buckets = false, is_tiny = false;

    if (bucket_size_1 * warp_count <= dev.shared_memory_bytes) {
        phase_1_name = "block_mkperm_phase_1_tiny";
        phase_4_name = "block_mkperm_phase_4_tiny";
        shared_size = bucket_size_1 * warp_count;
        bucket_size_all *= warp_count;
        variant = "tiny";
        is_tiny = true;
    } else if (bucket_size_1 <= dev.shared_memory_bytes) {
        phase_1_name = "block_mkperm_phase_1_small";
        phase_4_name = "block_mkperm_phase_4_small";
        shared_size = bucket_size_1;
        variant = "small";
    } else {
        phase_1_name = "block_mkperm_phase_1_large";
        phase_4_name = "block_mkperm_phase_4_large";
        variant = "large";
        initialize_buckets = true;
    }

    void *phase_1 = hip_kernel(device, phase_1_name),
         *phase_4 = hip_kernel(device, phase_4_name);

    uint32_t rows_per_group =
        is_tiny ? gpu_blocks_per_group * warp_count : gpu_blocks_per_group;

    bool needs_transpose = rows_per_group > 1;
    uint32_t *buckets_1, *buckets_2, *counter = nullptr;
    buckets_1 = buckets_2 = (uint32_t *) jitc_malloc(backend, bucket_size_all);

    if (needs_transpose)
        buckets_2 = (uint32_t *) jitc_malloc(backend, bucket_size_all);

    if (offsets) {
        counter = (uint32_t *) jitc_malloc(backend, sizeof(uint32_t));
        hip_check(hipMemsetD8Async((hipDeviceptr_t) counter, 0,
                                   sizeof(uint32_t), (hipStream_t) hip_stream));
    }

    if (initialize_buckets)
        hip_check(hipMemsetD8Async((hipDeviceptr_t) buckets_1, 0,
                                   bucket_size_all, (hipStream_t) hip_stream));

    uint32_t size_per_gpu_block =
        (block_size + gpu_blocks_per_group - 1) / gpu_blocks_per_group;

    jitc_log(Debug,
             "jit_block_mkperm(" DRJIT_PTR
             ", size=%u, block_size=%u, bucket_count=%u, gpu_block_count=%u, "
             "thread_count=%u, size_per_gpu_block=%u, variant=%s, "
             "shared_size=%u)",
             (uintptr_t) ptr, size, block_size, bucket_count, gpu_block_count,
             thread_count, size_per_gpu_block, variant, shared_size);

    // Phase 1: Count the number of occurrences per GPU block
    void *args_1[] = { &ptr, &buckets_1, &size, &size_per_gpu_block,
                       &bucket_count, &block_size };

    hip_submit(KernelType::MkPerm, this->recording_mode, phase_1,
               gpu_blocks_per_group, thread_count, shared_size, hip_stream,
               args_1, size, n_blocks);

    // Phase 2: exclusive prefix sum over transposed buckets
    if (needs_transpose)
        hip_transpose(this, buckets_1, buckets_2, rows_per_group, bucket_count,
                      n_blocks, rows_per_group * bucket_count);

    uint32_t psum_count = bucket_size_all / sizeof(uint32_t);
    uint32_t psum_block_size = rows_per_group * bucket_count;
    block_prefix_reduce(VarType::UInt32, ReduceOp::Add, psum_count,
                        psum_block_size, true, false, buckets_2, buckets_2);

    if (needs_transpose)
        hip_transpose(this, buckets_2, buckets_1, bucket_count, rows_per_group,
                      n_blocks, rows_per_group * bucket_count);

    // Phase 3: collect non-empty buckets (only for a single sorting group)
    if (likely(offsets) && n_blocks == 1) {
        uint32_t gpu_block_count_3, thread_count_3;
        dev.get_launch_config(&gpu_block_count_3, &thread_count_3,
                              bucket_count * gpu_block_count);

        uint32_t bucket_count_rounded =
            (bucket_count + thread_count_3 - 1) / thread_count_3 * thread_count_3;

        void *args_3[] = { &buckets_1, &bucket_count, &bucket_count_rounded,
                           &size,      &counter,      &offsets };

        hip_submit(KernelType::MkPerm, this->recording_mode,
                   hip_kernel(device, "block_mkperm_phase_3"),
                   gpu_block_count_3, thread_count_3,
                   sizeof(uint32_t) * thread_count_3, hip_stream, args_3, size);

        hip_check(hipMemcpyAsync(offsets + 4 * size_t(bucket_count), counter,
                                 sizeof(uint32_t), DR_HIP_MEMCPY_DEFAULT,
                                 (hipStream_t) hip_stream));

        hip_check(hipEventRecord((hipEvent_t) hip_event,
                                 (hipStream_t) hip_stream));
    }

    // Phase 4: write out permutation based on bucket counts
    void *args_4[] = { &ptr, &buckets_1, &perm, &size, &size_per_gpu_block,
                       &bucket_count, &block_size };

    hip_submit(KernelType::MkPerm, this->recording_mode, phase_4,
               gpu_blocks_per_group, thread_count, shared_size, hip_stream,
               args_4, size, n_blocks);

    if (likely(offsets) && n_blocks == 1) {
        unlock_guard guard_2(state.lock);
        hip_check(hipEventSynchronize((hipEvent_t) hip_event));
    }

    jitc_free(buckets_1);
    if (needs_transpose)
        jitc_free(buckets_2);
    jitc_free(counter);

    return (offsets && n_blocks == 1) ? offsets[4 * bucket_count] : 0u;
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
