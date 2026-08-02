/*
    kernels/mkperm.cuh -- Efficient CUDA kernels for permuting arrays
    into groups of identical values

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "common.h"

/// Accumulate 'value' into histogram 'buckets', using a minimal number of memory operations
inline __device__ uint32_t reduce(WarpMaskT active, uint32_t value, uint32_t *buckets) {
    WarpMaskT peers = match_any_(active, value);

    // Thread's position within warp
    uint32_t lane_idx = threadIdx.x & (WarpSize - 1);

    // Designate a leader thread within the set of peers
    uint32_t leader_idx = ffs_(peers) - 1;

    // If the current thread is the leader, perform atomic op.
    uint32_t offset = 0;
    if (lane_idx == leader_idx) {
        offset = buckets[value];
        buckets[value] = offset + popc_(peers);
    }

    // Fetch offset into output array from leader
    offset = shfl_(peers, offset, leader_idx);

    // Determine current thread's position within peer group
    uint32_t rel_pos = popc_(peers & lanemask_lt_(lane_idx));

    return offset + rel_pos;
}

/// Atomically accumulate 'value' into histogram 'buckets', using a minimal number of atomic operations
inline __device__ uint32_t reduce_atomic(WarpMaskT active, uint32_t value, uint32_t *buckets) {
    WarpMaskT peers = match_any_(active, value);

    // Thread's position within warp
    uint32_t lane_idx = threadIdx.x & (WarpSize - 1);

    // Designate a leader thread within the set of peers
    uint32_t leader_idx = ffs_(peers) - 1;

    // If the current thread is the leader, perform atomic op.
    uint32_t offset = 0;
    if (lane_idx == leader_idx)
        offset = atomicAdd(buckets + value, popc_(peers));

    // Fetch offset into output array from leader
    offset = shfl_(peers, offset, leader_idx);

    // Determine current thread's position within peer group
    uint32_t rel_pos = popc_(peers & lanemask_lt_(lane_idx));

    return offset + rel_pos;
}

/// Add 'value' to histogram 'buckets' (one update per peer group). Used by the
/// counting phase, which does not need per-lane offsets -- avoiding the
/// convergent shuffle that the compiler cannot eliminate from 'reduce'.
inline __device__ void count(WarpMaskT active, uint32_t value, uint32_t *buckets) {
    WarpMaskT peers = match_any_(active, value);
    if ((threadIdx.x & (WarpSize - 1)) == ffs_(peers) - 1)
        buckets[value] += popc_(peers);
}

inline __device__ void count_atomic(WarpMaskT active, uint32_t value, uint32_t *buckets) {
    WarpMaskT peers = match_any_(active, value);
    if ((threadIdx.x & (WarpSize - 1)) == ffs_(peers) - 1)
        atomicAdd(buckets + value, popc_(peers));
}

/**
 * \brief Generate a histogram of values in the range (0 .. bucket_count - 1).
 *
 * "Tiny" variant, which uses per-warp shared memory histograms to produce a
 * stable permutation. Handles up to 512 buckets with 64KiB of shared memory.
 * Should be combined with \ref block_mkperm_phase_4_tiny.
 *
 * Each warp processes a contiguous range of elements so that warp ordering in
 * the subsequent prefix sum matches element ordering (required for stability).
 */
KERNEL void block_mkperm_phase_1_tiny(const uint32_t *values,
                                        uint32_t *buckets,
                                        uint32_t size,
                                        uint32_t size_per_block,
                                        uint32_t bucket_count,
                                        uint32_t block_size) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    // Grid: x = sub-block within a block, y = block
    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             group        = blockIdx.y,
             sub_block    = blockIdx.x,
             flat_block   = group * gridDim.x + sub_block,
             user_block_start = group * block_size,
             block_start  = user_block_start + sub_block * size_per_block,
             warp_count   = thread_count / WarpSize,
             warp_id      = thread_id / WarpSize,
             lane_id      = thread_id & (WarpSize - 1);

    // Clamp to the user block boundary
    uint32_t user_block_end = min(user_block_start + block_size, size);
    uint32_t block_end = min(block_start + size_per_block, user_block_end);

    for (uint32_t i = thread_id; i < bucket_count * warp_count; i += thread_count)
        shared[i] = 0;

    __syncthreads();

    uint32_t *shared_warp = shared + warp_id * bucket_count;

    // Each warp processes a contiguous range for stable ordering. The range is
    // rounded up to a full multiple of the warp size so that every lane reaches
    // the ballot below; out-of-range lanes are masked off via 'active'.
    uint32_t total = block_end > block_start ? block_end - block_start : 0;
    uint32_t elems_per_warp = ((total + warp_count - 1) / warp_count + WarpSize - 1) & ~(WarpSize - 1);
    uint32_t warp_start = block_start + warp_id * elems_per_warp;
    uint32_t warp_end   = warp_start + elems_per_warp;

    for (uint32_t i = warp_start + lane_id; i < warp_end; i += WarpSize) {
        bool active = i < block_end;

        WarpMaskT active_mask = ballot_(WarpMask, active);

        if (active)
            count(active_mask, values[i], shared_warp);

        syncwarp_();
    }

    __syncthreads();

    uint32_t *out = buckets + flat_block * bucket_count * warp_count;
    for (uint32_t i = thread_id; i < bucket_count * warp_count; i += thread_count)
        out[i] = shared[i];
}

/**
 * \brief Generate a histogram of values in the range (0 .. bucket_count - 1).
 *
 * "Small" variant, which uses shared memory atomics and handles up to 16K
 * buckets with 64KiB of shared memory. The permutation can be somewhat
 * unstable due to scheduling variations when performing atomic operations
 * (although some effort is made to keep it stable within each wavefront
 * of elements by performing an intra-warp reduction.) Should be combined with
 * \ref block_mkperm_phase_4_small.
 */
KERNEL void block_mkperm_phase_1_small(const uint32_t *values,
                                         uint32_t *buckets,
                                         uint32_t size,
                                         uint32_t size_per_block,
                                         uint32_t bucket_count,
                                         uint32_t block_size) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             group        = blockIdx.y,
             sub_block    = blockIdx.x,
             flat_block   = group * gridDim.x + sub_block,
             user_block_start = group * block_size,
             block_start  = user_block_start + sub_block * size_per_block;

    // Clamp to the user block boundary
    uint32_t user_block_end = min(user_block_start + block_size, size);
    uint32_t block_end = min(block_start + size_per_block, user_block_end);

    for (uint32_t i = thread_id; i < bucket_count; i += thread_count)
        shared[i] = 0;

    __syncthreads();

    // Warp-aligned upper bound so every lane uniformly reaches the ballot;
    // tail lanes past the data are masked off via 'active'.
    uint32_t iter_end = block_start + ((size_per_block + WarpSize - 1) & ~(WarpSize - 1));

    for (uint32_t i = block_start + thread_id; i < iter_end; i += thread_count) {
        bool active = i < block_end;

        WarpMaskT active_mask = ballot_(WarpMask, active);

        if (active)
            count_atomic(active_mask, values[i], shared);
    }

    __syncthreads();

    uint32_t *out = buckets + flat_block * bucket_count;
    for (uint32_t i = thread_id; i < bucket_count; i += thread_count)
        out[i] = shared[i];
}

/**
 * \brief Generate a histogram of values in the range (0 .. bucket_count - 1).
 *
 * "Large" variant, which uses global memory atomics and handles arbitrarily
 * many elements (though this is somewhat slower than the previous two shared
 * memory variants). The permutation can be somewhat unstable due to scheduling
 * variations when performing atomic operations (although some effort is made
 * to keep it stable within each wavefront of elements by performing an
 * intra-warp reduction.) Should be combined with \ref block_mkperm_phase_4_large.
 */
KERNEL void block_mkperm_phase_1_large(const uint32_t *values,
                                         uint32_t *buckets_, uint32_t size,
                                         uint32_t size_per_block,
                                         uint32_t bucket_count,
                                         uint32_t block_size) {
    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             group        = blockIdx.y,
             sub_block    = blockIdx.x,
             flat_block   = group * gridDim.x + sub_block,
             user_block_start = group * block_size,
             block_start  = user_block_start + sub_block * size_per_block;

    // Clamp to the user block boundary
    uint32_t user_block_end = min(user_block_start + block_size, size);
    uint32_t block_end = min(block_start + size_per_block, user_block_end);

    uint32_t *buckets = buckets_ + flat_block * bucket_count;

    // Warp-aligned upper bound so every lane uniformly reaches the ballot;
    // tail lanes past the data are masked off via 'active'.
    uint32_t iter_end = block_start + ((size_per_block + WarpSize - 1) & ~(WarpSize - 1));

    for (uint32_t i = block_start + thread_id; i < iter_end; i += thread_count) {
        bool active = i < block_end;

        WarpMaskT active_mask = ballot_(WarpMask, active);

        if (active)
            count_atomic(active_mask, values[i], buckets);
    }
}

/// Detect non-empty buckets and record their offsets
KERNEL void block_mkperm_phase_3(uint32_t *buckets,
                                   uint32_t bucket_count,
                                   uint32_t bucket_count_rounded,
                                   uint32_t perm_size,
                                   uint32_t *counter,
                                   uint4 *offsets) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    // Thread's position within warp
    uint32_t lane_idx = threadIdx.x & (WarpSize - 1);

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < bucket_count_rounded; i += blockDim.x * gridDim.x) {

        uint32_t offset_a, offset_b;

        offset_a = (i < bucket_count) ? buckets[i] : perm_size;
        shared[threadIdx.x] = offset_a;
        __syncthreads();

        if (threadIdx.x + 1 < blockDim.x)
            offset_b = shared[threadIdx.x + 1];
        else
            offset_b = (i + 1 < bucket_count) ? buckets[i + 1] : perm_size;

        // Did we find a non-empty bucket?
        bool found = offset_a != offset_b;

        // Peers within the same warp that also found one
        WarpMaskT peers = ballot_(WarpMask, found);

        if (found) {
            // Designate a leader thread within the set of peers
            uint32_t leader_idx = ffs_(peers) - 1;

            // If the current thread is the leader, perform atomic op.
            uint32_t offset = 0;
            if (lane_idx == leader_idx)
                offset = atomicAdd(counter, popc_(peers));

            // Fetch offset into output array from leader
            offset = shfl_(peers, offset, leader_idx);

            // Determine current thread's position within peer group
            offset += popc_(peers & lanemask_lt_(lane_idx));

            offsets[offset] = make_uint4(i, offset_a, offset_b - offset_a, 0);
        }
    }
}

/**
 * \brief Generate a sorting permutation based on offsets generated by
 * \ref block_mkperm_phase_1_tiny().
 *
 * Each warp processes a contiguous range of elements for stable ordering.
 */
KERNEL void block_mkperm_phase_4_tiny(const uint32_t *values,
                                        const uint32_t *buckets_,
                                        uint32_t *perm,
                                        uint32_t size,
                                        uint32_t size_per_block,
                                        uint32_t bucket_count,
                                        uint32_t block_size) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             group        = blockIdx.y,
             sub_block    = blockIdx.x,
             flat_block   = group * gridDim.x + sub_block,
             user_block_start = group * block_size,
             block_start  = user_block_start + sub_block * size_per_block,
             warp_count   = thread_count / WarpSize,
             warp_id      = thread_id / WarpSize,
             lane_id      = thread_id & (WarpSize - 1);

    // Clamp to the user block boundary
    uint32_t user_block_end = min(user_block_start + block_size, size);
    uint32_t block_end = min(block_start + size_per_block, user_block_end);

    const uint32_t *buckets = buckets_ + flat_block * bucket_count * warp_count;
    for (uint32_t i = thread_id; i < bucket_count * warp_count; i += thread_count)
        shared[i] = buckets[i];

    __syncthreads();

    uint32_t *shared_warp = shared + warp_id * bucket_count;

    // Each warp processes a contiguous range for stable ordering. The range is
    // rounded up to a full multiple of the warp size so that every lane reaches
    // the ballot below; out-of-range lanes are masked off via 'active'.
    uint32_t total = block_end > block_start ? block_end - block_start : 0;
    uint32_t elems_per_warp = ((total + warp_count - 1) / warp_count + WarpSize - 1) & ~(WarpSize - 1);
    uint32_t warp_start = block_start + warp_id * elems_per_warp;
    uint32_t warp_end   = warp_start + elems_per_warp;

    for (uint32_t i = warp_start + lane_id; i < warp_end; i += WarpSize) {
        bool active = i < block_end;

        WarpMaskT active_mask = ballot_(WarpMask, active);

        if (active) {
            uint32_t offset = reduce(active_mask, values[i], shared_warp);
            perm[user_block_start + offset] = i;
        }
    }
}

/**
 * \brief Generate a sorting permutation based on offsets generated by
 * \ref block_mkperm_phase_1_small().
 */
KERNEL void block_mkperm_phase_4_small(const uint32_t *values,
                                         const uint32_t *buckets_,
                                         uint32_t *perm,
                                         uint32_t size,
                                         uint32_t size_per_block,
                                         uint32_t bucket_count,
                                         uint32_t block_size) {
    uint32_t *shared = SharedMemory<uint32_t>::get();

    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             group        = blockIdx.y,
             sub_block    = blockIdx.x,
             flat_block   = group * gridDim.x + sub_block,
             user_block_start = group * block_size,
             block_start  = user_block_start + sub_block * size_per_block;

    // Clamp to the user block boundary
    uint32_t user_block_end = min(user_block_start + block_size, size);
    uint32_t block_end = min(block_start + size_per_block, user_block_end);

    const uint32_t *buckets = buckets_ + flat_block * bucket_count;
    for (uint32_t i = thread_id; i < bucket_count; i += thread_count)
        shared[i] = buckets[i];

    __syncthreads();

    // Warp-aligned upper bound so every lane uniformly reaches the ballot;
    // tail lanes past the data are masked off via 'active'.
    uint32_t iter_end = block_start + ((size_per_block + WarpSize - 1) & ~(WarpSize - 1));

    for (uint32_t i = block_start + thread_id; i < iter_end; i += thread_count) {
        bool active = i < block_end;

        WarpMaskT active_mask = ballot_(WarpMask, active);

        if (active) {
            uint32_t offset = reduce_atomic(active_mask, values[i], shared);
            perm[user_block_start + offset] = i;
        }
    }
}

/**
 * \brief Generate a sorting permutation based on offsets generated by
 * \ref block_mkperm_phase_1_large().
 */
KERNEL void block_mkperm_phase_4_large(const uint32_t *values,
                                         uint32_t *buckets_,
                                         uint32_t *perm,
                                         uint32_t size,
                                         uint32_t size_per_block,
                                         uint32_t bucket_count,
                                         uint32_t block_size) {
    uint32_t thread_id    = threadIdx.x,
             thread_count = blockDim.x,
             group        = blockIdx.y,
             sub_block    = blockIdx.x,
             flat_block   = group * gridDim.x + sub_block,
             user_block_start = group * block_size,
             block_start  = user_block_start + sub_block * size_per_block;

    // Clamp to the user block boundary
    uint32_t user_block_end = min(user_block_start + block_size, size);
    uint32_t block_end = min(block_start + size_per_block, user_block_end);

    uint32_t *buckets = buckets_ + flat_block * bucket_count;

    // Warp-aligned upper bound so every lane uniformly reaches the ballot;
    // tail lanes past the data are masked off via 'active'.
    uint32_t iter_end = block_start + ((size_per_block + WarpSize - 1) & ~(WarpSize - 1));

    for (uint32_t i = block_start + thread_id; i < iter_end; i += thread_count) {
        bool active = i < block_end;

        WarpMaskT active_mask = ballot_(WarpMask, active);

        if (active) {
            uint32_t offset = reduce_atomic(active_mask, values[i], buckets);
            perm[user_block_start + offset] = i;
        }
    }
}

KERNEL void transpose(const uint32_t *in,
                      uint32_t *out,
                      uint32_t i_rows,
                      uint32_t i_cols,
                      uint32_t batch_stride) {
    uint32_t *shared = SharedMemory<uint32_t>::get();
    const uint32_t dim = 16;

    uint32_t batch = batch_stride ? blockIdx.z : 0;
    const uint32_t *in_b  = in  + batch * batch_stride;
    uint32_t       *out_b = out + batch * batch_stride;

    uint32_t i_c = blockIdx.x * dim + threadIdx.x,
             i_r = blockIdx.y * dim + threadIdx.y;

    bool valid = i_r < i_rows && i_c < i_cols;

    if (valid)
        shared[threadIdx.y * (dim + 1) + threadIdx.x] = in_b[i_r * i_cols + i_c];

    __syncthreads();

    uint32_t o_rows = i_cols,
             o_cols = i_rows,
             o_c = blockIdx.y * dim + threadIdx.x,
             o_r = blockIdx.x * dim + threadIdx.y;

    valid = o_r < o_rows && o_c < o_cols;

    if (valid)
        out_b[o_r * o_cols + o_c] = shared[threadIdx.x * (dim + 1) + threadIdx.y];
}
