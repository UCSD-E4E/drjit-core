// Wave-width sensitive: a butterfly reduction across the wave.
//
// This is the case the NVIDIA arm can only PARTIALLY check. It runs at
// DRJIT_WARP_SIZE == 32 there and 64 on gfx90a, so structure is verified but
// width-64 semantics are not. Kernels of this shape belong on the
// wave64-unverified list (PLAN.md §0.3, §7.2).
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t i = DRJIT_TID;
    if (i >= n) return;
    float v = in[i];
    for (unsigned off = DRJIT_WARP_SIZE / 2; off > 0; off >>= 1)
        v += DRJIT_SHFL(v, (DRJIT_TID % DRJIT_WARP_SIZE) ^ off);
    out[i] = v;
}
