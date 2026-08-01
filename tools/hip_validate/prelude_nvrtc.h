/*
    prelude_nvrtc.h -- NVIDIA/NVRTC implementation of the prelude_portable.h
    contract.

    Prepended by hip_validate to the execution arm. See prelude_portable.h for
    the contract and the rationale.

    NVRTC supplies the CUDA builtins implicitly, so there is deliberately no
    #include here -- pulling in <cuda_runtime.h> would fail under NVRTC.
*/

#pragma once

typedef unsigned int       uint32_t;
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

/* Deliberately 64-bit even though CUDA ballots are 32-bit, so that emitted
   code never assumes 32. The widening is free and keeps one type across arms. */
typedef uint64_t DRJIT_LANE_MASK_T;

#define DRJIT_TID \
    ((uint32_t) (blockIdx.x * blockDim.x + threadIdx.x))

/* CUDA's sync-shuffle needs an explicit participating mask. */
#define DRJIT_SHFL(v, lane) __shfl_sync(0xffffffffu, (v), (int) (lane), (int) DRJIT_WARP_SIZE)

#define DRJIT_BALLOT(pred)  ((DRJIT_LANE_MASK_T) __ballot_sync(0xffffffffu, (int) (pred)))

#define DRJIT_ACTIVEMASK()  ((DRJIT_LANE_MASK_T) __activemask())
