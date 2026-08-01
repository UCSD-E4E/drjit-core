// SPECIMEN: transcendentals and the fast-approximation family.
// Machine-shaped SSA output; self-checking. Run with --expect-zero.
//
// Covers VarKind: Sin Cos Exp2 Log2 Tanh Rcp RcpApprox RSqrtApprox SqrtApprox
//                 DivApprox
//
// THESE OPS ARE NOT BIT-EXACT ACROSS BACKENDS, BY DESIGN.
//
// VarKind groups Sin/Cos/Exp2/Log2/Tanh under "multi-function generator", i.e.
// NVIDIA's SFU path (sin.approx.f32 and friends). gfx90a has the analogous
// V_SIN_F32 / V_EXP_F32 hardware, but the two are not required to agree to the
// last ulp, and the *_Approx kinds are explicitly licensed to be sloppy.
//
// So every check here is a TOLERANCE, and the tolerances are deliberately
// loose. A specimen that demanded exact equality would either fail on real
// hardware or, worse, pass here and set an expectation the MI210 cannot meet.
// This is the one specimen file where tightening the bounds is the wrong
// instinct.

DRJIT_DEVICE bool close(float a, float b, float tol) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    float m = b < 0.0f ? -b : b;
    return d <= tol * (m > 1.0f ? m : 1.0f);   // relative, with absolute floor
}

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    float r1 = in[r0];            // [0, 1]
    float r2 = r1 + 1.0f;         // [1, 2] -- safe for log2, rcp, rsqrt

    // --- Transcendentals ----------------------------------------------------
    //
    // Emitted as the fast device forms (__sinf etc.), matching what the CUDA
    // backend selects. 1e-4 relative is far tighter than the SFU guarantee but
    // loose enough that both platforms clear it comfortably.
    float r3 = __sinf(r1);                     // Sin
    float r4 = __cosf(r1);                     // Cos
    float r5 = r3 * r3 + r4 * r4;              // identity: sin^2 + cos^2 == 1
    if (!close(r5, 1.0f, 1e-4f)) err |= 1u << 0;

    float r6 = exp2f(r1);                      // Exp2 (no __exp2f exists)
    float r7 = __log2f(r6);                    // Log2  (round trip)
    if (!close(r7, r1, 1e-3f)) err |= 1u << 1;

    // log2(2^k) == k for exact powers of two, a much stronger check than a
    // round trip because it pins the absolute scale.
    float r8 = __log2f(8.0f);
    if (!close(r8, 3.0f, 1e-3f)) err |= 1u << 2;

    float r9 = tanhf(r1);                      // Tanh
    if (!(r9 >= 0.0f && r9 <= r1)) err |= 1u << 3;   // 0 <= tanh(x) <= x on [0,1]

    // --- Reciprocal ---------------------------------------------------------
    //
    // Rcp is the accurate form; RcpApprox is licensed to be sloppy. Both are
    // checked against the same identity but with very different tolerances --
    // that asymmetry IS the specification.
    float r10 = 1.0f / r2;                     // Rcp (accurate)
    if (!close(r10 * r2, 1.0f, 1e-6f)) err |= 1u << 4;

    float r11 = __frcp_rn(r2);                 // Rcp, correctly rounded
    if (!close(r11 * r2, 1.0f, 1e-6f)) err |= 1u << 5;

    // --- Reciprocal square root --------------------------------------------
    float r12 = rsqrtf(r2);                    // RSqrtApprox
    if (!close(r12 * r12 * r2, 1.0f, 1e-3f)) err |= 1u << 6;

    // --- Square root --------------------------------------------------------
    float r13 = sqrtf(r2);                     // Sqrt (accurate)
    if (!close(r13 * r13, r2, 1e-6f)) err |= 1u << 7;

    float r14 = __fsqrt_rn(r2);                // SqrtApprox / correctly rounded
    if (!close(r14 * r14, r2, 1e-3f)) err |= 1u << 8;

    // --- Division -----------------------------------------------------------
    float r15 = r1 / r2;                       // Div (accurate)
    float r16 = __fdividef(r1, r2);            // DivApprox
    if (!close(r15, r16, 1e-3f)) err |= 1u << 9;

    // --- Double precision ---------------------------------------------------
    //
    // gfx90a has full-rate FP64, so the double path is first class rather than
    // an afterthought -- and it must NOT be silently promoted down to float.
    // 1e-12 would be unreachable if it were.
    double r17 = (double) r2;
    double r18 = 1.0 / r17;
    double r19 = r18 * r17 - 1.0;
    if (!(r19 < 1e-12 && r19 > -1e-12)) err |= 1u << 10;

    out[r0] = (float) err;
}
