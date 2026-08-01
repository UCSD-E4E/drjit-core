// SPECIMEN: memory operations. Machine-shaped SSA output; self-checking.
// Run with --expect-zero.
//
// Covers VarKind: Counter Gather Scatter BoundsCheck ScatterInc ScatterExch
//                 ScatterCAS
//
// Memory ops carry the traps that survive compilation and corrupt data:
//
//   * Gather and Scatter are MASKED. A masked-off lane must not touch memory
//     at all -- not "read and discard". An unmasked gather at an out-of-range
//     index faults or reads a neighbouring allocation; an unmasked scatter
//     writes one. The emitter must guard the access, not the result.
//
//   * Pointers arrive as uint64_t (type_name_hip maps Pointer that way) and
//     are cast at each use site. Casting to the wrong element type silently
//     scales the index.
//
//   * Atomics must be genuinely atomic across the whole grid, not merely
//     per-wavefront.
//
// The kernel writes through a scratch region carved out of out[] itself, so no
// extra buffers are needed beyond the harness convention.

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    // --- Counter -------------------------------------------------------------
    //
    // VarKind::Counter is the lane index. The prologue already materialises it
    // as r0; every other op indexes off it.
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // Pointers are uint64_t in the type table and cast per use site.
    uint64_t r1 = (uint64_t) in;    // Pointer is uint64_t per type_name_hip
    const float *r2 = (const float *) r1;

    // --- Gather --------------------------------------------------------------
    float r3 = r2[r0];                    // Gather, unmasked
    if (r3 != in[r0]) err |= 1u << 0;

    // Masked gather: the guard is on the ACCESS, not on the result. Writing
    // this as `v = ptr[idx]; v = mask ? v : 0;` would read out of bounds on
    // every masked lane -- the exact bug this specimen exists to rule out.
    uint32_t r4 = r0 + n;                 // deliberately out of range
    bool     r5 = (r4 < n);               // false for every lane
    float    r6 = 0.0f;
    if (r5) r6 = r2[r4];                  // Gather, masked -- never executes
    if (r6 != 0.0f) err |= 1u << 1;

    // A mask that IS taken, to prove the guard is not simply always-false.
    bool  r7 = (r0 < n);
    float r8 = 0.0f;
    if (r7) r8 = r2[r0];
    if (r8 != in[r0]) err |= 1u << 2;

    // --- BoundsCheck ---------------------------------------------------------
    //
    // Clamps rather than wraps: a wrapped index still reads valid memory and
    // so produces plausible-but-wrong data instead of an obvious failure.
    uint32_t r9  = r0 < n ? r0 : n - 1u;
    if (r9 >= n) err |= 1u << 3;

    // --- Scatter -------------------------------------------------------------
    //
    // Each lane owns exactly one slot, so this is race-free by construction.
    uint64_t r10 = (uint64_t) out;
    float   *r11 = (float *) r10;
    r11[r0] = 0.0f;                       // Scatter, unmasked

    // Masked scatter must leave the slot untouched. Every lane is masked off
    // here; if the emitter drops the guard it clobbers a neighbour's slot.
    bool r12 = false;
    if (r12) r11[(r0 + 1u) % n] = 12345.0f;

    // NO __syncthreads() HERE. The prologue does an early `return` for lanes
    // past the end, so a barrier below it is UNDEFINED -- the returned threads
    // never arrive and the launch fails. This is a hard constraint on the
    // emitter: any barrier must dominate the bounds-check guard, or the guard
    // must be a predicated region rather than a return. Found the hard way:
    // this kernel compiled on BOTH arms and then failed at launch.
    //
    // No barrier is needed anyway -- each lane owns its own slot.
    if (r11[r0] != 0.0f) err |= 1u << 4;

    // --- Atomics -------------------------------------------------------------
    //
    // ScatterInc / ScatterExch / ScatterCAS. Each lane operates on its own
    // slot, so the VALUE is deterministic; what is being pinned is that the
    // emitted form compiles and returns the documented "old" value on both
    // platforms, since the return-value convention is the easy thing to get
    // backwards.
    // TARGET MUST BE GLOBAL MEMORY. Dr.Jit's scatter-atomics always address a
    // device buffer; a register never appears as a target. An earlier draft
    // took the address of a local, which forces private memory -- it compiled
    // on BOTH arms and then failed at launch. The emitter must always route
    // atomics through the buffer pointer, never a materialised temporary.
    //
    // Each lane uses its own slot, so the value is deterministic and no race
    // is involved; out[r0] is overwritten with `err` at the end.
    uint32_t *r13 = (uint32_t *) r10 + r0;
    *r13 = 0u;

    uint32_t r14 = atomicAdd(r13, 5u);        // ScatterInc: returns OLD value
    if (r14 != 0u || *r13 != 5u) err |= 1u << 5;

    uint32_t r15 = atomicExch(r13, 9u);       // ScatterExch: returns OLD value
    if (r15 != 5u || *r13 != 9u) err |= 1u << 6;

    uint32_t r16 = atomicCAS(r13, 9u, 11u);   // ScatterCAS: returns OLD value
    if (r16 != 9u || *r13 != 11u) err |= 1u << 7;

    // A CAS whose comparand does not match must NOT write.
    uint32_t r17 = atomicCAS(r13, 999u, 22u);
    if (r17 != 11u || *r13 != 11u) err |= 1u << 8;

    // Float atomic add -- gradient accumulation depends on it, so this is
    // load-bearing for the AD path rather than incidental.
    float *r18 = (float *) r10 + r0;
    *r18 = 0.0f;
    float r19 = atomicAdd(r18, 1.5f);
    if (r19 != 0.0f || *r18 != 1.5f) err |= 1u << 9;

    out[r0] = (float) err;
}
