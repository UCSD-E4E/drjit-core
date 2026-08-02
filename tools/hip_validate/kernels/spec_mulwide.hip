// SPECIMEN: wide and high multiplication. Machine-shaped SSA output;
// self-checking. Run with --expect-zero.
//
// Covers VarKind: MulHi MulWide
//
// These exist as separate opcodes precisely because the plain `*` operator
// throws away the information they need. `a * b` on two uint32_t is computed
// modulo 2^32, so the high half is gone before any cast can recover it:
//
//     uint64_t wrong = (uint64_t) (a * b);        // truncated, then widened
//     uint64_t right = (uint64_t) a * (uint64_t) b;
//
// Emitting the first form is a plausible mistake that compiles, runs, and is
// correct for every small input -- so it survives casual testing and fails on
// real data. Both forms are computed below and asserted to DIFFER.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // Chosen so the product needs 64 bits: 0xffffffff * 0xffffffff overflows
    // 32 bits by almost the full width.
    uint32_t r1 = 0xffffffffu;
    uint32_t r2 = 0xffffffffu;

    // --- MulWide -------------------------------------------------------------
    //
    // Full 64-bit product of two 32-bit operands. The operands must be widened
    // BEFORE the multiply.
    uint64_t r3 = (uint64_t) r1 * (uint64_t) r2;
    if (r3 != 0xfffffffe00000001ull) err |= 1u << 0;

    // The truncate-then-widen form must NOT agree -- if it does, the widening
    // happened in the wrong place.
    uint64_t r4 = (uint64_t) (r1 * r2);
    if (r4 == r3) err |= 1u << 1;

    // --- MulHi ---------------------------------------------------------------
    //
    // High 32 bits of a 32x32 product. __umulhi is the device intrinsic on
    // both platforms; computing it as (a*b) >> 32 in 32-bit arithmetic yields
    // zero, which is why the opcode exists.
    uint32_t r5 = __umulhi(r1, r2);
    if (r5 != 0xfffffffeu) err |= 1u << 2;
    if (r5 != (uint32_t) (r3 >> 32)) err |= 1u << 3;

    // Signed high multiply is a DIFFERENT instruction, not a cast of the
    // unsigned one: -1 * -1 has high word 0, whereas the unsigned reading of
    // the same bits has high word 0xfffffffe.
    int32_t r6 = -1;
    int32_t r7 = -1;
    int32_t r8 = __mulhi(r6, r7);
    if (r8 != 0) err |= 1u << 4;
    if ((uint32_t) r8 == r5) err |= 1u << 5;   // must differ from unsigned

    // Signed wide multiply, same reasoning.
    int64_t r9 = (int64_t) r6 * (int64_t) r7;
    if (r9 != 1) err |= 1u << 6;

    // --- A case that small inputs would hide ---------------------------------
    //
    // Lane-dependent so it is not constant-folded away at compile time.
    uint32_t r10 = 0x10000u + r0;
    uint32_t r11 = 0x10000u;
    uint64_t r12 = (uint64_t) r10 * (uint64_t) r11;
    uint32_t r13 = __umulhi(r10, r11);
    if (r13 != (uint32_t) (r12 >> 32)) err |= 1u << 7;
    if (r13 != 1u) err |= 1u << 8;   // (0x10000+i) * 0x10000 >> 32 == 1

    // --- 64-bit MulHi --------------------------------------------------------
    //
    // A separate intrinsic, not a wider overload: __umulhi applied to 64-bit
    // operands would silently narrow them to 32 bits, which for the pattern
    // below gives 0xfffffffe instead of 0xfffffffffffffffe.
    uint64_t r14 = 0xffffffffffffffffull;
    uint64_t r15 = __umul64hi(r14, r14);
    if (r15 != 0xfffffffffffffffeull) err |= 1u << 9;

    int64_t r16 = -1, r17 = __mul64hi(r16, r16);
    if (r17 != 0) err |= 1u << 10;
    if ((uint64_t) r17 == r15) err |= 1u << 11;   // must differ from unsigned

    out[r0] = (float) err;
}
