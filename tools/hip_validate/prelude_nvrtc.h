/*
    prelude_nvrtc.h -- NVIDIA/NVRTC implementation of the prelude_portable.h
    contract.

    Prepended by hip_validate to the execution arm. See prelude_portable.h for
    the contract and the rationale.

    NVRTC supplies the CUDA builtins implicitly, so there is deliberately no
    #include here -- pulling in <cuda_runtime.h> would fail under NVRTC.
*/

#pragma once

/* NVRTC provides no <stdint.h>, so the fixed-width types the emitter uses have
 * to be declared here. This must stay in step with type_name_hip in
 * src/hip_eval.h -- every spelling that table can emit must exist on BOTH arms
 * or generated code compiles for gfx90a and fails here.
 * kernels/smoke_types.hip exercises the full set to keep them aligned. */
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;

#define DRJIT_KERNEL extern "C" __global__
#define DRJIT_DEVICE __device__ __forceinline__

/* All current NVIDIA parts are warp-32.

   NOTE (PLAN.md §0.3, §7.2): this is precisely why the execution arm cannot
   verify wave64 *semantics*. Structure is checked here; width-64 behaviour is
   not. Anything whose correctness depends on the width being 64 must stay on
   the wave64-unverified list until MI210 access. */
#define DRJIT_WARP_SIZE 32u

/* Half precision.

   NVRTC does not ship cuda_fp16.h, and _Float16 is not reliably available on
   the device side here, so the execution arm cannot represent fp16 faithfully.
   Aliasing to float keeps fp16 kernels COMPILABLE and RUNNABLE on this arm --
   useful for checking surrounding logic -- but the values carry f32 precision.

   Consequence: fp16 numerics are verified by the gfx arm only, and belong on
   the same "unverified on NVIDIA" list as wave64 semantics (PLAN.md §0.3,
   §7.2). Do not use this arm to sign off rounding behaviour. */
#define DRJIT_HALF float

/* Reinterpret 16 raw bits as a half.
 *
 * DRJIT_HALF is `float` on this arm, so this is an actual conversion rather
 * than a bitcast. Written out longhand because NVRTC ships no fp16 header.
 * Handles zero, subnormals and inf/NaN explicitly so that literal payloads
 * survive instead of being flushed -- tests/hip_literal.cpp asserts that
 * jitc_hip_literal() preserves NaN payloads, and that guarantee would be
 * hollow if the prelude threw them away. */
DRJIT_DEVICE float DRJIT_HALF_FROM_BITS(unsigned int h) {
    unsigned int sign = (h & 0x8000u) << 16;
    int          exp  = (int) ((h >> 10) & 0x1fu);
    unsigned int man  = h & 0x3ffu;
    if (exp == 0) {
        if (man == 0)
            return __uint_as_float(sign);                 /* +/- zero */
        while (!(man & 0x400u)) { man <<= 1; exp--; }     /* normalise */
        exp++;
        man &= 0x3ffu;
    } else if (exp == 31) {
        return __uint_as_float(sign | 0x7f800000u | (man << 13)); /* inf/NaN */
    }
    return __uint_as_float(sign | ((unsigned int) (exp + 112) << 23) |
                           (man << 13));
}

/* Deliberately 64-bit even though CUDA ballots are 32-bit, so that emitted
   code never assumes 32. The widening is free and keeps one type across arms. */
typedef uint64_t DRJIT_LANE_MASK_T;

#define DRJIT_TID \
    ((uint32_t) (blockIdx.x * blockDim.x + threadIdx.x))

/* CUDA's sync-shuffle needs an explicit participating mask. */
#define DRJIT_SHFL(v, lane) __shfl_sync(0xffffffffu, (v), (int) (lane), (int) DRJIT_WARP_SIZE)

#define DRJIT_BALLOT(pred)  ((DRJIT_LANE_MASK_T) __ballot_sync(0xffffffffu, (int) (pred)))

#define DRJIT_ACTIVEMASK()  ((DRJIT_LANE_MASK_T) __activemask())
