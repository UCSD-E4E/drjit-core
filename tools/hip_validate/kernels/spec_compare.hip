// SPECIMEN: comparisons, select, and rounding. Machine-shaped SSA output.
// Self-checking; run with --expect-zero.
//
// Covers VarKind: Eq Neq Lt Le Gt Ge Select Ceil Floor Round Trunc
//
// Comparisons produce Bool, which type_name_hip spells `bool`. The emitter
// must NOT widen them to int: Select depends on the ?: operator seeing a real
// bool, and a mask stored as int would change the semantics of And/Or on masks.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    float r1 = in[r0];      // in [0, 1]
    float r2 = 0.5f;

    // --- Comparisons --------------------------------------------------------
    bool r3 = (r1 == r1);                 // Eq
    if (!r3) err |= 1u << 0;

    bool r4 = (r1 != r1);                 // Neq  (false, even for NaN-free input)
    if (r4) err |= 1u << 1;

    bool r5 = (r1 < r2);                  // Lt
    bool r6 = (r1 >= r2);                 // Ge
    if (r5 == r6) err |= 1u << 2;         // exactly one must hold

    bool r7 = (r1 <= r1);                 // Le
    if (!r7) err |= 1u << 3;

    bool r8 = (r1 > r1);                  // Gt
    if (r8) err |= 1u << 4;

    // --- Select -------------------------------------------------------------
    float r9 = r5 ? r1 : r2;              // Select
    if (r5 && r9 != r1) err |= 1u << 5;
    if (!r5 && r9 != r2) err |= 1u << 6;

    // Select must also work on integers and on the mask type itself.
    uint32_t r10 = r5 ? 11u : 22u;        // Select (int)
    if (r10 != (r5 ? 11u : 22u)) err |= 1u << 7;

    // --- Rounding -----------------------------------------------------------
    //
    // Note Trunc is a *cast-like* rounding: for floats it stays float. The
    // emitter fuses Trunc with a float->int conversion when the result type is
    // integral, which is why these are checked in float form here.
    float r11 = 2.7f;
    float r12 = -2.7f;

    float r13 = ceilf(r11);     // Ceil
    if (r13 != 3.0f) err |= 1u << 8;

    float r14 = floorf(r11);    // Floor
    if (r14 != 2.0f) err |= 1u << 9;

    float r15 = truncf(r11);    // Trunc
    if (r15 != 2.0f) err |= 1u << 10;

    float r16 = truncf(r12);    // Trunc toward zero, not floor
    if (r16 != -2.0f) err |= 1u << 11;

    // Round is round-to-nearest-EVEN, matching IEEE and the CUDA backend --
    // NOT roundf(), which rounds half away from zero and would disagree.
    float r17 = rintf(2.5f);    // Round -> 2.0 (even)
    if (r17 != 2.0f) err |= 1u << 12;

    float r18 = rintf(3.5f);    // Round -> 4.0 (even)
    if (r18 != 4.0f) err |= 1u << 13;

    out[r0] = (float) err;
}
