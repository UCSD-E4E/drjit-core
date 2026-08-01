// SPECIMEN: bit-wise ops, shifts and bit counting. Machine-shaped SSA output.
// Self-checking; run with --expect-zero.
//
// Covers VarKind: Not And Or Xor Shl Shr Popc Clz Ctz Brev Bitcast
//
// Two traps the emitter must avoid, both asserted below:
//
//  * Shr on a SIGNED type is arithmetic (sign-extending); on unsigned it is
//    logical. Emitting the operand with the wrong signedness silently changes
//    the result for negative values.
//
//  * Bit-wise ops on floats must go through type_name_hip_bin -- the unsigned
//    integer of the same width -- not the float type. `~` on a float does not
//    compile, and going via a narrower type truncates.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    uint32_t r1 = r0 + 1u;
    uint32_t r2 = 0xf0f0f0f0u;

    // --- Bit-wise -----------------------------------------------------------
    uint32_t r3 = ~r2;                    // Not
    if ((r3 & r2) != 0u) err |= 1u << 0;

    uint32_t r4 = r1 & r2;                // And
    uint32_t r5 = r1 | r2;                // Or
    uint32_t r6 = r1 ^ r2;                // Xor
    if ((r4 | r6) != r5) err |= 1u << 1;  // (a&b) | (a^b) == a|b

    // --- Shifts -------------------------------------------------------------
    uint32_t r7 = r1 << 3;                // Shl
    if (r7 != r1 * 8u) err |= 1u << 2;

    uint32_t r8 = r7 >> 3;                // Shr (logical, unsigned)
    if (r8 != r1) err |= 1u << 3;

    // Arithmetic shift: -16 >> 2 == -4, sign extended. A logical shift here
    // would give a huge positive number.
    int32_t r9  = -16;
    int32_t r10 = r9 >> 2;                // Shr (arithmetic, signed)
    if (r10 != -4) err |= 1u << 4;

    // The same pattern shifted as UNSIGNED must NOT sign-extend.
    uint32_t r11 = ((uint32_t) r9) >> 2;
    if (r11 == (uint32_t) r10) err |= 1u << 5;

    // --- Bit counting -------------------------------------------------------
    uint32_t r12 = (uint32_t) __popc(r2);        // Popc
    if (r12 != 16u) err |= 1u << 6;

    // Clz/Ctz on a known pattern. 0xf0f0f0f0 has no leading zeros and 4
    // trailing zeros.
    uint32_t r13 = (uint32_t) __clz(r2);             // Clz
    if (r13 != 0u) err |= 1u << 7;

    uint32_t r14 = (uint32_t) (__ffs(r2) - 1);               // Ctz (via __ffs)
    if (r14 != 4u) err |= 1u << 8;

    // Clz of 1 is 31 on a 32-bit type -- guards against a 64-bit builtin
    // being selected by accident.
    uint32_t r15 = (uint32_t) __clz(1u);
    if (r15 != 31u) err |= 1u << 9;

    // Brev: bit reversal. 0x1 reversed is 0x80000000.
    uint32_t r16 = __brev(1u);               // Brev
    if (r16 != 0x80000000u) err |= 1u << 10;

    // --- Bitcast ------------------------------------------------------------
    //
    // Float bit manipulation goes through the binary view. Flipping the sign
    // bit of 1.0f must give -1.0f exactly.
    float    r17 = 1.0f;
    uint32_t r18 = __float_as_uint(r17);        // Bitcast f32->u32
    uint32_t r19 = r18 ^ 0x80000000u;                        // Xor on binary view
    float    r20 = __uint_as_float(r19);           // Bitcast u32->f32
    if (r20 != -1.0f) err |= 1u << 11;

    // 64-bit binary view must not truncate: Float64 -> uint64_t.
    double   r21 = 1.0;
    uint64_t r22 = (uint64_t) __double_as_longlong(r21);
    if (r22 != 0x3ff0000000000000ull) err |= 1u << 12;

    out[r0] = (float) err;
}
