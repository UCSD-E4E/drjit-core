// Verifies the Float16 -> DRJIT_HALF mapping asserted in tests/hip_types.cpp is
// actually usable, i.e. that the prelude pulls in whatever declares it.
// Written after the type table, to check the table is not lying.
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t i = DRJIT_TID;
    if (i >= n) return;
    DRJIT_HALF h = (DRJIT_HALF) in[i];
    out[i] = (float) h * 2.0f;
}
