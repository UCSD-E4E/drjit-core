// Deliberately broken -- proves the harness reports failure rather than
// silently passing. Not part of any suite; used to test the tester.
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t i = DRJIT_TID;
    if (i >= n) return
    out[i] = in[i] * undefined_symbol_xyz;
}
