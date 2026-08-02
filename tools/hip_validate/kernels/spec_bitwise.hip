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

    // Ctz is clz(brev(x)), NOT __ffs(x) - 1. The two agree everywhere except
    // at zero, where __ffs returns 0 and the subtraction wraps to UINT_MAX,
    // while Dr.Jit's constant folding defines ctz(0) as the bit width. A
    // backend disagreeing with the folder produces results that depend on
    // whether a value happened to be known at trace time.
    uint32_t r14 = (uint32_t) __clz(__brev(r2));             // Ctz
    if (r14 != 4u) err |= 1u << 8;

    uint32_t r14z = (uint32_t) __clz(__brev(0u));            // Ctz(0)
    if (r14z != 32u) err |= 1u << 18;

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

    // --- The shapes the emitter actually produces ---------------------------
    //
    // Above this line each construct is checked in isolation. Dr.Jit emits
    // Not/And/Or/Xor on floats as a single COMPOSED expression, and the
    // composition is where it goes wrong: an `f32` spelling anywhere in the
    // chain either fails to compile (`~` on a float) or silently rounds.

    // Not, on a float. Inverting all bits of 1.0f gives a specific pattern;
    // OR-ing the two binary views back together must set every bit.
    float    r23 = 1.0f;
    float    r24 = __uint_as_float(~(uint32_t) __float_as_uint(r23));
    if ((__float_as_uint(r23) | __float_as_uint(r24)) != 0xffffffffu)
        err |= 1u << 13;

    // And / Or, on floats, through the binary view. Masking off the sign bit
    // of -2.0f is fabsf; setting it on 2.0f is negation. Done in f32 these
    // would not compile at all.
    float r25 = -2.0f, r26 = __uint_as_float(0x7fffffffu);   // sign-bit mask
    float r27 = __uint_as_float((uint32_t) __float_as_uint(r25) &
                                (uint32_t) __float_as_uint(r26));
    if (r27 != 2.0f) err |= 1u << 14;

    float r28 = 2.0f, r29 = __uint_as_float(0x80000000u);
    float r30 = __uint_as_float((uint32_t) __float_as_uint(r28) |
                                (uint32_t) __float_as_uint(r29));
    if (r30 != -2.0f) err |= 1u << 15;

    // Not, on a bool, is `!` -- NOT `~`. `~true` is 0xfe, which converts back
    // to `true`, so a bit-wise spelling here is not merely unidiomatic: it
    // makes the operation a no-op.
    bool r31 = true;
    bool r32 = !r31;
    if (r32) err |= 1u << 16;

    // An all-ones word derived from the binary view, which is what Or against
    // a set mask selects. `~(u8) 0` must survive the round trip to bool.
    bool r33 = (bool) ~(uint8_t) 0;
    if (!r33) err |= 1u << 17;

    // --- 64-bit bit counting -------------------------------------------------
    //
    // Distinct intrinsics, not overloads. The 32-bit spelling applied to a
    // 64-bit value examines only the low half and compiles without complaint:
    // __clz(1ull << 40) is 31, not 23.
    uint64_t r34 = 0xf0f0f0f0f0f0f0f0ull;
    if ((uint32_t) __popcll(r34) != 32u)  err |= 1u << 19;
    if ((uint32_t) __clzll(r34)  != 0u)   err |= 1u << 20;
    if ((uint32_t) __clzll(1ull) != 63u)  err |= 1u << 21;
    if (__brevll(1ull) != 0x8000000000000000ull) err |= 1u << 22;
    if ((uint32_t) __clzll(__brevll(r34)) != 4u)   err |= 1u << 23;  // Ctz64
    if ((uint32_t) __clzll(__brevll(0ull)) != 64u) err |= 1u << 24;  // Ctz64(0)

    out[r0] = (float) err;
}
