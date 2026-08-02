#pragma once

#include <stdint.h>
#include <type_traits>
#include <limits>

/*
   These kernels compile for two targets: NVIDIA via nvcc (producing the
   committed resources/kernels_75.lz4) and AMD via hipcc (producing the
   committed gfx90a code object). The differences are confined to this header;
   the .cuh sources below are written against the primitives it defines.

   The single most important of those is the WAVEFRONT WIDTH. gfx90a is
   wave64-only and CUDA is warp-32, and cross-lane code written for one width
   does not fail to compile on the other -- it silently addresses the wrong
   lane. PLAN.md §3.3 therefore asks for the width as a single-definition
   parameter that every site reads, which is what `WarpSize` is here. The
   idioms that hide a width behind a literal -- `31 - __clz(ballot)`,
   `mask << (32 - lane)`, a 0xFFFFFFFF ballot mask -- have named replacements
   below, and the ones that remain are compile errors rather than silent
   miscomputation: ROCm's __ballot_sync static_asserts that its mask is 64-bit.
*/

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#  define DRJIT_KERNELS_HIP 1
#  include <hip/hip_runtime.h>
#  include <hip/hip_fp16.h>
#else
#  define DRJIT_KERNELS_HIP 0
#  include <cuda_fp16.h>
#endif

#define KERNEL extern "C" __global__
#define DEVICE __device__
#define FINLINE __forceinline__

#if DRJIT_KERNELS_HIP
   // HIP's `warpSize` is a runtime value, not a constant expression, so it
   // cannot size the templates below. Derive the width from the target instead:
   // every GFX9 part (Vega, CDNA1-3) is wave64 and cannot do wave32 at all.
   // A future RDNA target would land in the #error, which is the point -- it
   // defaults to wave32 and would otherwise take this path silently.
#  if defined(__AMDGCN_WAVEFRONT_SIZE__)
#    define WarpSize __AMDGCN_WAVEFRONT_SIZE__
#  elif defined(__GFX9__)
#    define WarpSize 64
#  elif defined(__HIP_DEVICE_COMPILE__)
#    error "Unknown AMD target: the wavefront width must be stated explicitly."
#  else
#    define WarpSize 64   // host pass; never used to generate code
#  endif
   using WarpMaskT = uint64_t;
#  define WarpMask (~0ull)
#else
#  define WarpSize 32
   using WarpMaskT = uint32_t;
#  define WarpMask 0xffffffffu
#endif

// ----------------------------------------------------------------------------
//  Cross-lane primitives, width-parametric
// ----------------------------------------------------------------------------

DEVICE FINLINE WarpMaskT ballot_(WarpMaskT mask, bool pred) {
    return (WarpMaskT) __ballot_sync(mask, pred);
}

DEVICE FINLINE bool any_(WarpMaskT mask, bool pred) {
    return __any_sync(mask, pred) != 0;
}

template <typename T>
DEVICE FINLINE T shfl_(WarpMaskT mask, T value, uint32_t src_lane) {
    return __shfl_sync(mask, value, src_lane);
}

template <typename T>
DEVICE FINLINE T shfl_xor_(WarpMaskT mask, T value, uint32_t lane_mask,
                           uint32_t width = WarpSize) {
    return __shfl_xor_sync(mask, value, lane_mask, width);
}

template <typename T>
DEVICE FINLINE T shfl_down_(WarpMaskT mask, T value, uint32_t delta,
                            uint32_t width = WarpSize) {
    return __shfl_down_sync(mask, value, delta, width);
}

DEVICE FINLINE void syncwarp_() { __syncwarp(); }

DEVICE FINLINE uint32_t popc_(WarpMaskT v) {
#if DRJIT_KERNELS_HIP
    return (uint32_t) __popcll((unsigned long long) v);
#else
    return (uint32_t) __popc(v);
#endif
}

/// One-based index of the lowest set lane, or 0 if `v` is empty (as __ffs).
DEVICE FINLINE uint32_t ffs_(WarpMaskT v) {
#if DRJIT_KERNELS_HIP
    return (uint32_t) __ffsll((unsigned long long) v);
#else
    return (uint32_t) __ffs(v);
#endif
}

/// Index of the HIGHEST set lane. Replaces the `31 - __clz(mask)` idiom, which
/// compiles fine at width 64 and then names a lane 32 places off.
DEVICE FINLINE uint32_t highest_lane_(WarpMaskT v) {
#if DRJIT_KERNELS_HIP
    return 63u - (uint32_t) __clzll((unsigned long long) v);
#else
    return 31u - (uint32_t) __clz(v);
#endif
}

/// Mask of the lanes strictly below `lane`. Replaces `peers << (32 - lane)`,
/// which needs a width-matched shift distance and is undefined at lane 0.
DEVICE FINLINE WarpMaskT lanemask_lt_(uint32_t lane) {
    return (((WarpMaskT) 1) << lane) - 1;
}

/// Mask of the lanes in this wavefront holding the same `value` as the caller.
DEVICE FINLINE WarpMaskT match_any_(WarpMaskT active, uint32_t value) {
#if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 700
    return (WarpMaskT) __match_any_sync(active, value);
#else
    /* Emulate __match_any_sync. Based on "Voting And Shuffling For Fewer
       Atomic Operations" by Elmar Westphal. */
    do {
        // Find lowest-numbered active lane, fetch its value, compare to ours
        bool match = (value == shfl_(active, value, ffs_(active) - 1));

        // Determine which lanes had a match
        WarpMaskT peers = ballot_(active, match);

        // Key of the current lane was chosen, return the active mask
        if (match)
            return peers;

        // Remove lanes with matching values from the pool
        active ^= peers;
    } while (true);
#endif
}

#if DRJIT_KERNELS_HIP
// CUDA-only integer min/max builtins. Clang resolves the signed 32-bit forms
// through <algorithm>, but the unsigned and 64-bit ones have no HIP spelling.
DEVICE FINLINE uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }
DEVICE FINLINE uint64_t ullmax(uint64_t a, uint64_t b) { return a > b ? a : b; }
DEVICE FINLINE int64_t llmax(int64_t a, int64_t b) { return a > b ? a : b; }
DEVICE FINLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }
DEVICE FINLINE uint64_t ullmin(uint64_t a, uint64_t b) { return a < b ? a : b; }
DEVICE FINLINE int64_t llmin(int64_t a, int64_t b) { return a < b ? a : b; }
#endif

template <typename T> struct SharedMemory {
    __device__ inline static T *get() {
        extern __shared__ int shared[];
        return (T *) shared;
    }
};

template <> struct SharedMemory<double> {
    __device__ inline static double *get() {
        extern __shared__ double shared_d[];
        return shared_d;
    }
};


DEVICE float fma_(float a, float b, float c) { return __fmaf_rn(a, b, c); }
DEVICE double fma_(double a, double b, double c) { return __fma_rn(a, b, c); }
DEVICE half fma_(half a, half b, half c) {
    #if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 600
        return __hfma(a, b, c);
    #else
        return __fma_rn((float) a, (float) b, (float) c);
    #endif
}

template <typename T> DEVICE T add_(T a, T b) { return a + b; }
DEVICE half add_(half a, half b) {
    #if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 600
        return __hadd(a, b);
    #else
        return (half) ((float) a + (float) b);
    #endif
}

template <typename T> DEVICE T mul_(T a, T b) { return a * b; }
DEVICE half mul_(half a, half b) {
    #if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 600
        return __hmul(a, b);
    #else
        return (half) ((float) a * (float) b);
    #endif
}

template <typename T> DEVICE T sub_(T a, T b) { return a - b; }
DEVICE half sub_(half a, half b) {
    #if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 600
        return __hsub(a, b);
    #else
        return (half) ((float) a - (float) b);
    #endif
}

DEVICE float max_(float a, float b) { return fmaxf(a, b); }
DEVICE double max_(double a, double b) { return fmax(a, b); }
DEVICE uint32_t max_(uint32_t a, uint32_t b) { return umax(a, b); }
DEVICE uint64_t max_(uint64_t a, uint64_t b) { return ullmax(a, b); }
DEVICE int32_t max_(int32_t a, int32_t b) { return max(a, b); }
DEVICE int64_t max_(int64_t a, int64_t b) { return llmax(a, b); }
DEVICE half max_(half a, half b) {
    #if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 800
        return __hmax(a, b);
    #else
        return (half) fmaxf((float) a, (float) b);
    #endif
}
DEVICE float min_(float a, float b) { return fminf(a, b); }
DEVICE double min_(double a, double b) { return fmin(a, b); }
DEVICE uint32_t min_(uint32_t a, uint32_t b) { return umin(a, b); }
DEVICE uint64_t min_(uint64_t a, uint64_t b) { return ullmin(a, b); }
DEVICE int32_t min_(int32_t a, int32_t b) { return min(a, b); }
DEVICE int64_t min_(int64_t a, int64_t b) { return llmin(a, b); }
DEVICE half min_(half a, half b) {
    #if DRJIT_KERNELS_HIP || __CUDA_ARCH__ >= 800
        return __hmin(a, b);
    #else
        return (half) fminf((float) a, (float) b);
    #endif
}

template <typename T> struct reduction_add {
    using Value = std::conditional_t<std::is_same<half, T>::value, float, T>;
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value a, Value b) const {
        return add_(a, b);
    }
};

template <typename T> struct reduction_mul {
    using Value = std::conditional_t<std::is_same<half, T>::value, float, T>;
    __device__ Value init() { return (Value) 1; }
    __device__ Value operator()(Value a, Value b) const {
        return mul_(a, b);
    }
};

template <typename T> struct reduction_max {
    using Value = T;
    __device__ Value init() {
        return std::is_integral<Value>::value
                   ?  std::numeric_limits<Value>::min()
                   : -std::numeric_limits<Value>::infinity();
    }
    __device__ Value operator()(Value a, Value b) const {
        return max_(a, b);
    }
};

template <> struct reduction_max<half> {
    using Value = half;
    __device__ half init() { return __ushort_as_half((unsigned short) 0xFC00U); }
    __device__ half operator()(half a, half b) const {
        return max_(a, b);
    }
};

template <typename T> struct reduction_min {
    using Value = T;
    __device__ Value init() {
        return std::is_integral<Value>::value
                   ? std::numeric_limits<Value>::max()
                   : std::numeric_limits<Value>::infinity();
    }
    __device__ Value operator()(Value a, Value b) const {
        return min_(a, b);
    }
};

template <> struct reduction_min<half> {
    using Value = half;
    __device__ half init() { return __ushort_as_half((unsigned short) 0x7C00U); }
    __device__ half operator()(half a, half b) const {
        return min_(a, b);
    }
};

template <typename T> struct reduction_or {
    using Value = T;
    __device__ Value init() { return (Value) 0; }
    __device__ Value operator()(Value a, Value b) const {
        return a | b;
    }
};

template <typename T> struct reduction_and {
    using Value = T;
    __device__ Value init() { return (Value) -1; }
    __device__ Value operator()(Value a, Value b) const {
        return a & b;
    }
};

template <size_t> struct uint_with_size;
template <> struct uint_with_size<2> { using type = uint16_t; };
template <> struct uint_with_size<4> { using type = uint32_t; };
template <> struct uint_with_size<8> { using type = uint64_t; };
template <size_t Size> using uint_with_size_t = typename uint_with_size<Size>::type;

template <typename Target, typename Source>
__device__ Target memcpy_cast(Source source) {
    static_assert(sizeof(Source) == sizeof(Target), "memcpy_cast: sizes must be identical!");
    Target target;
    memcpy(&target, &source, sizeof(Source));
    return target;
}

/* Helper routines to write tagged values while bypassing the L1 cache.

   The decoupled-lookback scans below depend on this: a thread block publishes
   (value, status) as one indivisible word and its successors spin on it. A
   read served from a stale, non-coherent L1 either spins forever or -- worse --
   pairs a "done" status with the value that preceded it.

   NVIDIA spells that `.cg` / `ld.volatile.global`. AMD has no cache-operator
   suffix; the equivalent is an atomic access at AGENT scope, which is defined
   to be visible across compute units and so cannot be served by the CU-private
   L1. Both forms are single instructions, and both are naturally atomic at the
   widths used here, which is what keeps the status tag from tearing away from
   its value. */
#if DRJIT_KERNELS_HIP
#  define DRJIT_STORE_CG(p, v)                                                 \
      __hip_atomic_store((p), (v), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#  define DRJIT_LOAD_CG(p)                                                     \
      __hip_atomic_load((p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
#endif

__device__ void store_with_status(uint16_t *p, uint16_t value, uint32_t status) {
    uint32_t v = ((uint32_t) value) | (((uint32_t) status) << 16);

#if DRJIT_KERNELS_HIP
    DRJIT_STORE_CG((uint32_t *) p, v);
#else
    asm("st.global.cg.u32 [%0], %1;"
        :
        : "l"(p)
          "r"(v)
        : "memory");
#endif
}

__device__ void store_with_status(uint32_t *p, uint32_t value, uint32_t status) {
    uint64_t v = ((uint64_t) value) | (((uint64_t) status) << 32);

#if DRJIT_KERNELS_HIP
    DRJIT_STORE_CG((uint64_t *) p, v);
#else
    asm("st.global.cg.u64 [%0], %1;"
        :
        : "l"(p)
          "l"(v)
        : "memory");
#endif
}

__device__ void store_with_status(uint64_t *p, uint64_t value, uint32_t status) {
    uint64_t status_shift = ((uint64_t) status) << 32,
             v0 = uint32_t(value)  | status_shift,
             v1 = (value >> 32)    | status_shift;

    // 64-bit payloads exceed the widest atomic word, so the status is
    // replicated into both halves and the loader rejects a mismatch.
#if DRJIT_KERNELS_HIP
    DRJIT_STORE_CG((uint64_t *) p, v0);
    DRJIT_STORE_CG((uint64_t *) p + 1, v1);
#else
    asm("st.global.cg.u64 [%0], %1;\n"
        "st.global.cg.u64 [%0 + 8], %2;"
        :
        : "l"(p)
          "l"(v0)
          "l"(v1)
        : "memory");
#endif
}

__device__ void load_with_status(volatile uint16_t *p, uint16_t &value, uint32_t &status) {
    uint32_t v;

#if DRJIT_KERNELS_HIP
    v = DRJIT_LOAD_CG((const uint32_t *) (const uint16_t *) p);
#else
    asm("ld.volatile.global.u32 %0, [%1];"
        : "=r"(v)
        : "l"(p)
        : "memory");
#endif

    value = (uint16_t) v;
    status = (uint32_t) (v >> 16);
}

__device__ void load_with_status(volatile uint32_t *p, uint32_t &value, uint32_t &status) {
    uint64_t v;

#if DRJIT_KERNELS_HIP
    v = DRJIT_LOAD_CG((const uint64_t *) (const uint32_t *) p);
#else
    asm("ld.volatile.global.u64 %0, [%1];"
        : "=l"(v)
        : "l"(p)
        : "memory");
#endif

    value = (uint32_t) v;
    status = (uint32_t) (v >> 32);
}

__device__ void load_with_status(volatile uint64_t *p, uint64_t &value, uint32_t &status) {
    uint64_t v0, v1;

#if DRJIT_KERNELS_HIP
    const uint64_t *pp = (const uint64_t *) (const uint64_t *) p;
    v0 = DRJIT_LOAD_CG(pp);
    v1 = DRJIT_LOAD_CG(pp + 1);
#else
    asm("ld.volatile.global.u64 %0, [%2];\n"
        "ld.volatile.global.u64 %1, [%2 + 8];"
        : "=l"(v0)
          "=l"(v1)
        : "l"(p)
        : "memory");
#endif

    uint32_t v0_lo = (uint32_t) v0,
             v1_lo = (uint32_t) v1,
             v0_hi = (uint32_t) (v0 >> 32),
             v1_hi = (uint32_t) (v1 >> 32);

    value = (uint64_t) v0_lo + (((uint64_t) v1_lo) << 32);
    status = v0_hi == v1_hi ? v0_hi : 0;
}


// Vectorized types for 128 bit loads
template <typename T> struct Vec2 {
    T data[2];

    template <typename Func> __device__ T reduce(Func f) {
        return f(data[0], data[1]);
    }
};

template <typename T> struct Vec4 {
    T data[4];

    template <typename Func> __device__ T reduce(Func f) {
        return f(f(data[0], data[1]), f(data[2], data[3]));
    }
};

template <typename T> struct Vec8 {
    T data[8];

    template <typename Func> __device__ T reduce(Func f) {
        return f(f(f(data[0], data[1]),
                   f(data[2], data[3])),
                 f(f(data[4], data[5]),
                   f(data[6], data[7])));
    }
};

// Precomputed integer division helper
// Based on libidivide (https://github.com/ridiculousfish/libdivide)
struct divisor {
    uint32_t magic;
    uint32_t shift;
    uint32_t value;

    static __device__ uint32_t mulhi(uint32_t a, uint32_t b) {
#if DRJIT_KERNELS_HIP
        return __umulhi(a, b);
#else
        uint32_t r;
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
#endif
    }

    __device__ void div_rem(uint32_t input, uint32_t *out_div, uint32_t *out_rem) const {
        uint32_t div = 0, rem = 0;

        if (!magic) {
            div = input >> shift;
            rem = input - (div << shift);
        } else {
            uint32_t hi = mulhi(input, magic);
            div = (((input - hi) >> 1) + hi) >> shift;
            rem = input - div * value;
        }

        *out_div = div;
        *out_rem = rem;
    }
};
