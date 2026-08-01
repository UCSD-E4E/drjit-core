// SPECIMEN: wave-level operations. Machine-shaped SSA output; self-checking.
// Run with --expect-zero.
//
// Covers the shuffle / ballot / activemask family that appears in the
// scatter-reduce peer-aggregation path and in block reductions.
//
// ############################################################################
// #  THIS IS THE §7.2 FILE. Read the caveat before trusting a green result.  #
// ############################################################################
//
// Every check here is written to hold at BOTH wave widths, and the execution
// arm runs at 32. Passing therefore proves the ops are STRUCTURALLY correct --
// it does NOT prove wave64 semantics. Two things in particular change at 64
// and cannot be exercised on NVIDIA:
//
//   * A ballot needs 64 bits. DRJIT_LANE_MASK_T is uint64_t on both arms
//     precisely so emitted code never assumes 32, but on this arm the upper
//     half is always zero, so a `b32` truncation bug is INVISIBLE here.
//
//   * Shuffle lane arithmetic is modulo the width. An XOR-butterfly that
//     happens to be correct at 32 can address the wrong partner at 64.
//
// §0.2 measured 24 shfl plus vote and activemask in a Cornell box at 1 spp, so
// this is the hot path of every render, not a corner. Everything below belongs
// on the wave64-unverified list until it has run on an MI210.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // Lane index within the wave. Derived from the width parameter rather than
    // a hardcoded mask, so it is correct at 32 and 64.
    uint32_t r1 = r0 % DRJIT_WARP_SIZE;

    // --- The width parameter itself -----------------------------------------
    //
    // §3.3: a single definition, never a literal at the use site. Guard the
    // two legal values so a third (or a zero from a botched #if) is caught.
    if (!(DRJIT_WARP_SIZE == 32u || DRJIT_WARP_SIZE == 64u)) err |= 1u << 0;

    // The lane-mask type must be wide enough to hold one bit per lane. At
    // width 64 a 32-bit mask silently drops half the wave.
    if (sizeof(DRJIT_LANE_MASK_T) * 8u < DRJIT_WARP_SIZE) err |= 1u << 1;

    // --- Broadcast -----------------------------------------------------------
    //
    // Every lane reads lane 0. n is a multiple of the wave size in the harness
    // (1024), so no wave is partially populated.
    float r2 = (float) r0;
    float r3 = DRJIT_SHFL(r2, 0);
    float r4 = (float) (r0 - r1);          // value held by this wave's lane 0
    if (r3 != r4) err |= 1u << 2;

    // --- Butterfly reduction -------------------------------------------------
    //
    // Sums the wave. Written against DRJIT_WARP_SIZE so the partner index is
    // right at either width -- this is exactly the loop that would address the
    // wrong lane if the width were hardcoded to 32.
    float r5 = 1.0f;
    for (uint32_t off = DRJIT_WARP_SIZE / 2u; off > 0u; off >>= 1)
        r5 += DRJIT_SHFL(r5, r1 ^ off);
    // Each lane contributed 1.0, so the total is the wave width.
    if (r5 != (float) DRJIT_WARP_SIZE) err |= 1u << 3;

    // --- Ballot --------------------------------------------------------------
    //
    // All lanes active, so the ballot must have exactly WARP_SIZE bits set.
    // At width 64 this needs the full 64-bit mask; a b32 form gives 32 and
    // trips the check -- but only on hardware that is actually 64 wide.
    DRJIT_LANE_MASK_T r6 = DRJIT_BALLOT(1);
    uint32_t r7 = (uint32_t) (__popcll((unsigned long long) r6));
    if (r7 != DRJIT_WARP_SIZE) err |= 1u << 4;

    // A predicate true on exactly the even lanes gives half the wave.
    DRJIT_LANE_MASK_T r8 = DRJIT_BALLOT((r1 & 1u) == 0u);
    uint32_t r9 = (uint32_t) (__popcll((unsigned long long) r8));
    if (r9 != DRJIT_WARP_SIZE / 2u) err |= 1u << 5;

    // Bit i of the ballot must correspond to lane i.
    DRJIT_LANE_MASK_T r10 = (DRJIT_LANE_MASK_T) 1 << r1;
    if ((r6 & r10) == 0) err |= 1u << 6;

    // --- Activemask ----------------------------------------------------------
    //
    // Taken with no divergence in flight, so it must equal a full ballot.
    DRJIT_LANE_MASK_T r11 = DRJIT_ACTIVEMASK();
    uint32_t r12 = (uint32_t) (__popcll((unsigned long long) r11));
    if (r12 != DRJIT_WARP_SIZE) err |= 1u << 7;

    // --- 64-bit population count ---------------------------------------------
    //
    // __popcll, not __popc: counting a lane mask with the 32-bit form is the
    // single most likely wave64 bug, and it is undetectable on this arm
    // because the upper half is always zero here.
    if (__popcll(0xffffffffffffffffull) != 64) err |= 1u << 8;
    if (__popcll(0x00000000ffffffffull) != 32) err |= 1u << 9;

    out[r0] = (float) err;
}
