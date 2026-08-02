// SPECIMEN: type conversions. Machine-shaped SSA output; self-checking.
// Run with --expect-zero.
//
// Covers VarKind: Cast (and the Trunc/Cast fusion the emitter performs)
//
// Casts are where a backend most easily disagrees with the others while still
// compiling, because C++ conversion rules are permissive and every wrong
// choice is *some* valid conversion. The three traps asserted here:
//
//   * float -> int TRUNCATES toward zero. It does not round. Emitting a
//     rounding conversion changes results for every negative fractional value.
//   * Narrowing integer conversions WRAP; they must not saturate. GPU ISAs
//     have saturating converts and picking one would silently clamp.
//   * Signed -> unsigned reinterprets the bit pattern rather than clamping
//     at zero.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // --- float -> int truncates toward zero ---------------------------------
    float   r1 = 2.7f;
    float   r2 = -2.7f;

    int32_t r3 = (int32_t) r1;            // Cast f32->i32
    if (r3 != 2) err |= 1u << 0;

    int32_t r4 = (int32_t) r2;            // toward zero, NOT floor(-2.7) == -3
    if (r4 != -2) err |= 1u << 1;

    // --- int -> float --------------------------------------------------------
    int32_t r5 = -5;
    float   r6 = (float) r5;              // Cast i32->f32
    if (r6 != -5.0f) err |= 1u << 2;

    uint32_t r7 = 0x80000000u;            // 2^31, unrepresentable as int32
    float    r8 = (float) r7;             // Cast u32->f32, must not go negative
    if (!(r8 > 2.147e9f && r8 < 2.148e9f)) err |= 1u << 3;

    // --- Narrowing wraps, does not saturate ---------------------------------
    uint32_t r9  = 0x1ffu;
    uint8_t  r10 = (uint8_t) r9;          // Cast u32->u8: 0x1ff -> 0xff (wrap)
    if (r10 != 0xffu) err |= 1u << 4;

    uint32_t r11 = 0x100u;
    uint8_t  r12 = (uint8_t) r11;         // 0x100 -> 0x00, NOT clamped to 0xff
    if (r12 != 0u) err |= 1u << 5;

    int32_t  r13 = 300;
    int8_t   r14 = (int8_t) r13;          // 300 -> 44 (wrap), not 127 (saturate)
    if (r14 != 44) err |= 1u << 6;

    // --- Widening sign-extends signed, zero-extends unsigned ----------------
    int8_t   r15 = -1;
    int32_t  r16 = (int32_t) r15;         // sign extend -> -1
    if (r16 != -1) err |= 1u << 7;

    uint8_t  r17 = 0xffu;
    uint32_t r18 = (uint32_t) r17;        // zero extend -> 255, not -1
    if (r18 != 255u) err |= 1u << 8;

    // --- Signed <-> unsigned reinterprets ------------------------------------
    int32_t  r19 = -1;
    uint32_t r20 = (uint32_t) r19;        // 0xffffffff, not clamped to 0
    if (r20 != 0xffffffffu) err |= 1u << 9;

    // --- 64-bit --------------------------------------------------------------
    //
    // Guards against a 32-bit intermediate sneaking in: the value below is
    // unrepresentable in 32 bits and survives only if the whole chain is wide.
    uint64_t r21 = 0x1234567890abcdefull;
    uint32_t r22 = (uint32_t) r21;        // narrow to low word
    if (r22 != 0x90abcdefu) err |= 1u << 10;

    uint64_t r23 = (uint64_t) r22;        // widen back, high word must be zero
    if (r23 != 0x90abcdefull) err |= 1u << 11;

    // --- float <-> double ----------------------------------------------------
    //
    // f32 -> f64 is exact; f64 -> f32 rounds. 0.1 is not representable in
    // either, so the round trip must NOT be exact -- if it is, the double was
    // silently demoted to float somewhere (the Metal promotion trap).
    double r24 = 0.1;
    float  r25 = (float) r24;             // Cast f64->f32 (lossy)
    double r26 = (double) r25;            // Cast f32->f64 (exact)
    if (r26 == r24) err |= 1u << 12;      // equality here means FP64 was lost

    // ...but the loss must be small, i.e. a genuine rounding rather than junk.
    double r27 = r26 - r24;
    if (!(r27 < 1e-8 && r27 > -1e-8)) err |= 1u << 13;

    // --- Bitcast between two INTEGERS is a plain cast ------------------------
    //
    // Not every Bitcast involves a float. Int64 -> UInt64 and Int32 -> UInt32
    // are pure reinterpretations that C++ already expresses exactly, and
    // routing them through __float_as_uint (or any other float intrinsic)
    // converts the VALUE first: 6 becomes 0x40c00000, the bits of 6.0f. That
    // compiles, runs, and is wrong for every input.
    int64_t  r28 = 6;
    uint64_t r29 = (uint64_t) r28;
    if (r29 != 6ull) err |= 1u << 14;

    int64_t  r30 = -1;
    uint64_t r31 = (uint64_t) r30;
    if (r31 != 0xffffffffffffffffull) err |= 1u << 15;

    int32_t  r32 = -2;
    uint32_t r33 = (uint32_t) r32;
    if (r33 != 0xfffffffeu) err |= 1u << 16;

    out[r0] = (float) err;
}
