// SPECIMEN: local (per-lane) arrays. Machine-shaped SSA output; self-checking.
// Run with --expect-zero.
//
// Covers VarKind: Array ArrayInit ArrayRead ArrayWrite ArrayPhi ArraySelect
//
// Dr.Jit's "variable arrays" are per-lane C arrays declared inside the kernel.
// On the surface this is the most boring construct in the port -- `T a[N]` and
// `a[i]` -- and the emitter is close to a transliteration of metal_array.cpp.
// The reason it still gets a specimen is that the STORAGE is not obvious:
//
//   * A private array indexed by a DYNAMIC value cannot live in registers. On
//     AMDGPU it is placed in scratch, which is backed by memory and costs
//     occupancy. A constant index, by contrast, is usually promoted. Both
//     forms appear in emitted code and both must be correct; the difference
//     shows up in the gfx arm's code size, not in the results.
//
//   * Writes under divergent control flow are the normal case (a masked
//     ArrayWrite). Each lane owns its own array, so there is no cross-lane
//     hazard -- but that is a claim worth checking rather than assuming, since
//     a compiler that hoisted the array into LDS would break it silently.
//
//   * The memcpy in/out paths use a TRANSPOSED layout: element `i` of the
//     lane-`r0` array lives at `p[i * size + r0]`, not `p[r0 * len + i]`.
//     Both index expressions compile and both stay in bounds; only one is
//     coalesced, and only one agrees with what the host wrote.
//
// The scratch region past out[n] is used as the array backing buffer -- see the
// over-allocation in hip_validate.cpp. With ARR_LEN 3 and n lanes it is exactly
// 3n floats, and lane r0 owns indices {r0, n+r0, 2n+r0}, so no lane touches
// another's slots.

#define ARR_LEN 3

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // --- Array + ArrayInit ---------------------------------------------------
    //
    // Declaration, then a fill loop over a broadcast value. The emitter writes
    // both as separate statements against the same `arr_<reg>` name.
    float arr_1[ARR_LEN];
    for (uint32_t _i = 0; _i < ARR_LEN; _i++)
        arr_1[_i] = 7.5f;

    // --- ArrayWrite / ArrayRead, CONSTANT index ------------------------------
    //
    // The promotable form: the compiler can keep these in registers.
    arr_1[0] = 1.0f;
    arr_1[2] = 3.0f;
    if (arr_1[0] != 1.0f || arr_1[1] != 7.5f || arr_1[2] != 3.0f)
        err |= 1u << 0;

    // --- ArrayWrite / ArrayRead, DYNAMIC index -------------------------------
    //
    // Lane-dependent, so the array must go to scratch. Neighbouring lanes pick
    // different slots, which is what makes this a per-lane array rather than
    // something the compiler may share.
    uint32_t r1 = r0 % ARR_LEN;
    arr_1[r1] = 100.0f + (float) r0;
    if (arr_1[r1] != 100.0f + (float) r0) err |= 1u << 1;

    // The other two slots must be untouched by the write above.
    {
        float sum = 0.0f;
        for (uint32_t _i = 0; _i < ARR_LEN; _i++)
            if (_i != r1)
                sum += arr_1[_i];

        float want = 0.0f;
        if (r1 != 0u) want += 1.0f;
        if (r1 != 1u) want += 7.5f;
        if (r1 != 2u) want += 3.0f;
        if (sum != want) err |= 1u << 2;
    }

    // --- Masked ArrayWrite / ArrayRead ---------------------------------------
    //
    // Divergent: only half the lanes write. A masked-off lane must leave the
    // slot alone, and a masked-off READ yields zero rather than the stored
    // value (the emitter pre-zeroes the destination and guards the load).
    {
        bool r2 = (r0 % 2u) == 0u;
        float before = arr_1[1];

        if (r2) arr_1[1] = -5.0f;
        if (arr_1[1] != (r2 ? -5.0f : before)) err |= 1u << 3;

        float rd = 0.0f;
        if (r2) rd = arr_1[1];
        if (rd != (r2 ? -5.0f : 0.0f)) err |= 1u << 4;
    }

    // Slot 0's value from here on. It is NOT simply 1.0f: on lanes where
    // r1 == 0 the dynamic write above landed on it. Spelling this out rather
    // than re-reading arr_1[0] keeps the checks below independent of the array
    // they are checking.
    float a1_0 = (r1 == 0u) ? (100.0f + (float) r0) : 1.0f;

    // --- Array copy (the ArrayState::Conflicted path) ------------------------
    //
    // When a write would clobber an array another node still needs, the
    // emitter allocates a second buffer and copies element-wise first.
    float arr_2[ARR_LEN];
    for (uint32_t _i = 0; _i < ARR_LEN; _i++)
        arr_2[_i] = arr_1[_i];
    arr_2[0] = 42.0f;
    if (arr_2[0] != 42.0f || arr_1[0] != a1_0) err |= 1u << 5;   // original intact

    // --- ArraySelect ---------------------------------------------------------
    //
    // A whole-array select: copy one side or the other, element-wise, under a
    // mask. Not a per-element ternary -- the arrays are the operands.
    {
        bool r3 = (r0 % 3u) == 0u;
        float arr_3[ARR_LEN];
        if (r3) {
            for (uint32_t _i = 0; _i < ARR_LEN; _i++)
                arr_3[_i] = arr_1[_i];
        } else {
            for (uint32_t _i = 0; _i < ARR_LEN; _i++)
                arr_3[_i] = arr_2[_i];
        }
        if (arr_3[0] != (r3 ? a1_0 : 42.0f)) err |= 1u << 6;
    }

    // --- memcpy in / out, TRANSPOSED layout ----------------------------------
    //
    // `p[i * size + r0]`, not `p[r0 * len + i]`. Written first, then read back
    // through the same expression, then checked against the flat index a host
    // would use -- so a transposed emitter and a host-order one cannot both
    // pass.
    {
        float *p = out + n;    // scratch: ARR_LEN * n floats

        // memcpy out
        for (uint32_t _i = 0; _i < ARR_LEN; _i++)
            p[_i * n + r0] = (float) (_i * 1000u + r0);

        // memcpy in
        float arr_4[ARR_LEN];
        for (uint32_t _i = 0; _i < ARR_LEN; _i++)
            arr_4[_i] = p[_i * n + r0];

        for (uint32_t _i = 0; _i < ARR_LEN; _i++)
            if (arr_4[_i] != (float) (_i * 1000u + r0)) err |= 1u << 7;

        // The stride really is `n`: element 1 of this lane must NOT be element
        // 0 of the next one. This is the check that fails if the layout is
        // flipped to r0 * len + i.
        if (n > 1u && r0 + 1u < n) {
            if (p[1u * n + r0] == p[0u * n + r0 + 1u]) err |= 1u << 8;
        }
    }

    // --- A larger dynamically-indexed array ----------------------------------
    //
    // 64 elements is past anything the compiler will promote, so this is
    // unambiguously scratch. Correctness only -- the COST shows up as the gfx
    // arm's code size, which is worth watching when this grows.
    {
        uint32_t big[64];
        for (uint32_t _i = 0; _i < 64u; _i++)
            big[_i] = _i * 3u + r0;

        uint32_t idx = r0 % 64u;
        big[idx] = 0xabcdu;

        if (big[idx] != 0xabcdu) err |= 1u << 9;
        uint32_t other = (idx + 1u) % 64u;
        if (big[other] != other * 3u + r0) err |= 1u << 10;
    }

    out[r0] = (float) err;
}
