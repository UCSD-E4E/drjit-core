/*
    prelude_hip.h -- AMD/hipcc implementation of the prelude_portable.h contract.

    Prepended by hip_validate to the gfx-arm compile. See prelude_portable.h for
    the contract and the rationale.
*/

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define DRJIT_KERNEL extern "C" __global__
#define DRJIT_DEVICE __device__ __forceinline__

/* §3.3: wave width as a single-definition parameter, never a hardcoded 64.
   Mirrors HIP-RT's own arch dispatch in hiprt_common.h, which is the
   convention we deliberately follow rather than inventing one. */
#if defined(__gfx900__)  || defined(__gfx902__)  || defined(__gfx904__) || \
    defined(__gfx906__)  || defined(__gfx908__)  || defined(__gfx909__) || \
    defined(__gfx90a__)  || defined(__gfx90c__)  || \
    defined(__gfx940__)  || defined(__gfx941__)  || defined(__gfx942__)
#  define DRJIT_WARP_SIZE 64u
#else
#  define DRJIT_WARP_SIZE 32u
#endif

/* Half precision. gfx90a supports it natively, but the type is declared in a
   header rather than being a builtin -- hence the include above. */
#define DRJIT_HALF __half

/* Reinterpret 16 raw bits as a half. Paired with jitc_hip_literal(), which
   emits float constants as bit patterns rather than decimals. */
#define DRJIT_HALF_FROM_BITS(bits) __ushort_as_half((unsigned short) (bits))

typedef uint64_t DRJIT_LANE_MASK_T;

#define DRJIT_TID \
    ((uint32_t) (blockIdx.x * blockDim.x + threadIdx.x))

/* HIP's shuffle takes an explicit width and needs no mask argument. */
#define DRJIT_SHFL(v, lane) __shfl((v), (int) (lane), (int) DRJIT_WARP_SIZE)

/* Already 64-bit on HIP. */
#define DRJIT_BALLOT(pred)  ((DRJIT_LANE_MASK_T) __ballot((int) (pred)))

#define DRJIT_ACTIVEMASK()  ((DRJIT_LANE_MASK_T) __ballot(1))
