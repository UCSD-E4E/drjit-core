// SPECIMEN: the debug-mode bounds check. Machine-shaped SSA output.
// Self-checking; run with --expect-zero.
//
// Covers VarKind: BoundsCheck
//
// Emitted only under JitFlag::Debug, which is exactly when a new backend is
// being brought up -- so "HIP has no BoundsCheck" makes the one mode that
// would explain a wrong answer the one mode that cannot run.
//
// The opcode does two things at once, and both are easy to get subtly wrong:
//
//  * It RETURNS A NARROWED MASK, not a validity flag. The result is the
//    incoming mask with out-of-bounds lanes cleared, and it is what the
//    following gather/scatter is predicated on. Returning the raw comparison
//    would enable lanes the caller had masked off.
//
//  * The comparison must be UNSIGNED. Indices arrive as signed types too, and
//    a negative index compared as signed passes `index < size` and then reads
//    before the buffer. Converting to unsigned first turns it into a huge
//    value that fails the check, which is the intended behaviour.
//
// The out-of-bounds index is also recorded to a one-word buffer that the host
// reads after the launch to produce the diagnostic. Concurrent writers race
// and that is fine -- the host only needs a non-zero word and one offending
// index -- but the write must be predicated on `mask && !in_bounds`, not on
// the comparison alone, or masked-off lanes report positions they never read.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // Stand-in for the recorder buffer. The real one is a device allocation
    // shared by every check in the kernel; per-thread here so the specimen can
    // assert on it without cross-lane interference.
    uint32_t oob_buf = 0u;

    const uint32_t size = 16u;

    // --- In bounds, mask set: passes through unchanged -----------------------
    {
        uint32_t index = r0 % size;
        bool mask = true;

        bool r1 = mask && (index < (uint32_t) size);   // BoundsCheck
        if (mask && !r1)
            oob_buf = index;

        if (!r1) err |= 1u << 0;
        if (oob_buf != 0u) err |= 1u << 1;
    }

    // --- Out of bounds, mask set: cleared, and the index is recorded ---------
    {
        uint32_t index = size + (r0 % 4u);
        bool mask = true;

        bool r2 = mask && (index < (uint32_t) size);
        if (mask && !r2)
            oob_buf = index;

        if (r2) err |= 1u << 2;
        if (oob_buf != index) err |= 1u << 3;
    }

    // --- Out of bounds, mask CLEAR: still false, and NOT recorded ------------
    //
    // The interesting case. A lane the caller already disabled must not
    // contribute a diagnostic, or debug builds report phantom violations for
    // indices that were never used.
    {
        oob_buf = 0u;
        uint32_t index = size + 99u;
        bool mask = false;

        bool r3 = mask && (index < (uint32_t) size);
        if (mask && !r3)
            oob_buf = index;

        if (r3) err |= 1u << 4;
        if (oob_buf != 0u) err |= 1u << 5;
    }

    // --- A NEGATIVE signed index must fail the check -------------------------
    //
    // `(int32_t) -1 < 16` is true; `(uint32_t) -1 < 16` is false. The second is
    // the one that keeps the gather inside the buffer. Written the way the
    // emitter writes it -- an unsigned cast on the bound -- so that the usual
    // arithmetic conversions promote the index to unsigned.
    {
        oob_buf = 0u;
        int32_t index = -1 - (int32_t) (r0 % 4u);
        bool mask = true;

        bool r4 = mask && (index < (int32_t) (uint32_t) size);  // WRONG: signed
        bool r5 = mask && (((uint32_t) index) < (uint32_t) size); // right

        if (!r4) err |= 1u << 6;   // signed compare wrongly admits it
        if (r5)  err |= 1u << 7;   // unsigned compare correctly rejects it
    }

    // --- A 64-bit index compares against a 32-bit bound ----------------------
    //
    // Indices are not always u32. Promotion must widen the BOUND rather than
    // truncate the index, or a large 64-bit index aliases into range.
    {
        uint64_t index = 0x100000000ull + 3ull;   // truncates to 3
        bool mask = true;

        bool r6 = mask && (index < (uint32_t) size);
        if (r6) err |= 1u << 8;
    }

    out[r0] = (float) err;
}
