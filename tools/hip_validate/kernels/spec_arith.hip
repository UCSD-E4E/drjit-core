// SPECIMEN: the source jitc_hip_render() should emit for arithmetic VarKinds.
//
// Deliberately written as MACHINE OUTPUT, not idiomatic HIP: SSA temporaries,
// one operation per line, explicit types. That is the point -- this file is a
// specification of what the emitter must produce, so when hip_eval.cpp is
// written there is an unambiguous target rather than a prose description.
//
// Self-checking: each op is compared against an independently-computed value
// and a distinct bit is set on mismatch, so out[] is 0 exactly when every
// opcode is correct. Run with --expect-zero.
//
// Covers VarKind: Neg Abs Sqrt Add Sub Mul Div Fma Min Max Mod

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // in[] is sin(i degrees)*0.5+0.5, so within [0,1] -- no division by zero,
    // no sqrt of a negative.
    float r1 = in[r0];
    float r2 = 2.5f;

    // --- Unary -------------------------------------------------------------
    float r3 = -r1;                       // Neg
    if (r3 + r1 != 0.0f) err |= 1u << 0;

    float r4 = fabsf(r3);       // Abs
    if (r4 != r1) err |= 1u << 1;

    float r5 = sqrtf(r2);       // Sqrt
    if (!(r5 * r5 > 2.49f && r5 * r5 < 2.51f)) err |= 1u << 2;

    // --- Binary ------------------------------------------------------------
    //
    // NOTE: exact equality is only legitimate here because both operands are
    // exactly representable in binary. Round-tripping the *input* instead --
    // asserting (r1 + r2) - r2 == r1 -- is NOT valid: float addition is not
    // associative and the low mantissa bits are lost. That mistake was caught
    // by --expect-zero on 788/1024 lanes rather than by inspection, which is a
    // fair advertisement for self-checking specimens.
    float r6 = 0.5f + 2.5f;               // Add (both exact)
    if (r6 != 3.0f) err |= 1u << 3;

    float r7 = r6 - r2;                   // Sub
    if (r7 != 0.5f) err |= 1u << 4;

    // The input-dependent form gets a tolerance, as it must.
    float r6b = r1 + r2;
    float r7b = r6b - r2;
    float d1  = r7b - r1;
    if (!(d1 < 1e-6f && d1 > -1e-6f)) err |= 1u << 12;

    float r8 = r1 * r2;                   // Mul
    if (!(r8 <= r2 && r8 >= 0.0f)) err |= 1u << 5;

    float r9 = r8 / r2;                   // Div
    if (!(r9 - r1 < 1e-5f && r1 - r9 < 1e-5f)) err |= 1u << 6;

    // --- Fma ---------------------------------------------------------------
    //
    // Must be a genuine fused op, not a * b + c: the emitter has to pick the
    // fused form or precision silently differs from the CUDA backend.
    float r10 = fmaf(r1, r2, r2);   // Fma
    float r11 = r1 * r2 + r2;
    if (!(r10 - r11 < 1e-3f && r11 - r10 < 1e-3f)) err |= 1u << 7;

    // --- Min / Max ----------------------------------------------------------
    float r12 = fminf(r1, r2);  // Min
    if (r12 != r1) err |= 1u << 8;

    float r13 = fmaxf(r1, r2);  // Max
    if (r13 != r2) err |= 1u << 9;

    // --- Integer arithmetic, including Mod ---------------------------------
    //
    // Mod has no float form in HIP; integers use %, floats would need fmodf.
    uint32_t r14 = r0 + 7u;               // Add (int)
    uint32_t r15 = r14 % 5u;              // Mod
    if (r15 != (r0 + 7u) % 5u) err |= 1u << 10;

    int32_t r16 = -((int32_t) r0);        // Neg (int)
    if (r16 + (int32_t) r0 != 0) err |= 1u << 11;

    out[r0] = (float) err;
}
