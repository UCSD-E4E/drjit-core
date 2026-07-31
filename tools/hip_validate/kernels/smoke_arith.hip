// Baseline: arithmetic only. Mirrors the fma/mul/add-dominated mix measured in
// the real Mitsuba raygen kernel (PLAN.md §0.2). No wave ops, so this one IS
// fully verified by the NVIDIA execution arm.
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t i = DRJIT_TID;
    if (i >= n) return;
    float v = in[i];
    for (int j = 0; j < 8; ++j)
        v = fmaf(v, 1.01f, 0.25f) * 0.999f;
    out[i] = v;
}
