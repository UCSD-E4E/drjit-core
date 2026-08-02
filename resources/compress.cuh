/*
    kernels/compress.cuh -- CUDA kernels for converting a mask into a set of
    indices that can be used to compress an associated array.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

// See the DRJIT_STORE_CG / DRJIT_LOAD_CG commentary in common.h: these must
// bypass the (non-coherent) L1, or the lookback loop below spins forever.
DEVICE FINLINE void store_cg(volatile uint64_t *ptr, uint64_t val) {
#if DRJIT_KERNELS_HIP
    DRJIT_STORE_CG((uint64_t *) ptr, val);
#else
    asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val));
#endif
}

DEVICE FINLINE uint64_t load_cg(volatile uint64_t *ptr) {
#if DRJIT_KERNELS_HIP
    return DRJIT_LOAD_CG((const uint64_t *) ptr);
#else
    uint64_t retval;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(retval) : "l"(ptr) : "memory");
    return retval;
#endif
}

KERNEL void compress_small(const uint8_t *in, uint32_t *out, uint32_t size, uint32_t *count_out) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint8_t values_8[5];
    *(uint32_t *) values_8 = ((const uint32_t *) in)[threadIdx.x];
    values_8[4] = 0;

    // Unrolled exclusive scan
    uint32_t sum_local = 0;
    uint32_t values[5];
    for (int i = 0; i < 5; ++i) {
        uint32_t v = values_8[i];
        values[i] = sum_local;
        sum_local += v;
    }

    // Reduce using shared memory
    uint32_t si = threadIdx.x;
    shared[si] = 0;
    si += blockDim.x;
    shared[si] = sum_local;

    uint32_t sum_block = sum_local;
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        __syncthreads();
        sum_block = shared[si] + shared[si - offset];
        __syncthreads();
        shared[si] = sum_block;
    }

    if (threadIdx.x == blockDim.x - 1)
        *count_out = sum_block;

    sum_block -= sum_local;
    for (int i = 0; i < 5; ++i)
        values[i] += sum_block;

    for (int i = 0; i < 4; ++i) {
        if (values[i] != values[i + 1])
            out[values[i]] = threadIdx.x * 4 + i;
    }
}

KERNEL void compress_large(const uint8_t *in, uint32_t *out, volatile uint64_t *scratch, uint32_t *count_out) {
    uint32_t *shared = SharedMemory<uint32_t>::get();
    uint32_t thread_count = 128;

    uint8_t values_8[17];
    *(uint4 *) values_8 = ((const uint4 *) in)[blockIdx.x * thread_count + threadIdx.x];
    values_8[16] = 0;

    // Unrolled exclusive scan
    uint32_t sum_local = 0;
    uint32_t values[17];
    for (int i = 0; i < 17; ++i) {
        uint32_t v = values_8[i];
        values[i] = sum_local;
        sum_local += v;
    }

    // Block-level reduction of partial sum over 16 elements via shared memory
    uint32_t si = threadIdx.x;
    shared[si] = 0;
    si += thread_count;
    shared[si] = sum_local;

    uint32_t sum_block = sum_local;
    for (uint32_t offset = 1; offset < thread_count; offset <<= 1) {
        __syncthreads();
        sum_block = shared[si] + shared[si - offset];
        __syncthreads();
        shared[si] = sum_block;
    }

    // Store block-level partial inclusive scan value in global memory
    scratch += blockIdx.x;
    if (threadIdx.x == thread_count - 1)
        store_cg(scratch, (((uint64_t) sum_block) << 32) | 1ull);

    uint32_t lane = threadIdx.x & (WarpSize - 1);
    uint32_t prefix = 0;
    int32_t shift = (int32_t) lane - WarpSize;

    /* Compute prefix due to previous blocks using warp-level primitives.
       Based on "Single-pass Parallel Prefix Scan with Decoupled Look-back"
       by Duane Merrill and Michael Garland */
    while (true) {
        uint64_t temp = load_cg(scratch + shift);
        uint32_t flag = (uint32_t) temp;

        if (any_(WarpMask, flag == 0))
            continue;

        WarpMaskT mask = ballot_(WarpMask, flag == 2);
        uint32_t  value = (uint32_t) (temp >> 32);
        if (mask == 0) {
            prefix += value;
            shift -= WarpSize;
        } else {
            uint32_t index = highest_lane_(mask);
            if (lane >= index)
                prefix += value;
            break;
        }
    }

    // Warp-level reduction
    for (uint32_t offset = WarpSize / 2; offset > 0; offset /= 2)
        prefix += shfl_down_(WarpMask, prefix, offset, WarpSize);
    sum_block += shfl_(WarpMask, prefix, 0);

    // Store block-level complete inclusive scan value in global memory
    if (threadIdx.x == thread_count - 1) {
        store_cg(scratch, (((uint64_t) sum_block) << 32) | 2ull);

        if (blockIdx.x == gridDim.x - 1)
            *count_out = sum_block;
    }

    sum_block -= sum_local;
    for (int i = 0; i < 17; ++i)
        values[i] += sum_block;

    for (int i = 0; i < 16; ++i) {
        if (values[i] != values[i + 1])
            out[values[i]] = (blockIdx.x * thread_count + threadIdx.x) * 16 + i;
    }
}

KERNEL void compress_large_init(uint64_t *scratch, uint32_t size) {
    // The first WarpSize slots are sentinels marked "complete, contributes 0":
    // the lookback above starts at `lane - WarpSize` and would otherwise read
    // before the buffer for the leading blocks. The host allocates the padding
    // (see HIPThreadState::compress), so this bound must track the width.
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        scratch[i] = (i < WarpSize) ? 2 : 0;
}
