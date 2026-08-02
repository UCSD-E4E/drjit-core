// SPECIMEN: dynamic dispatch (vcalls). Machine-shaped SSA output; self-checking.
// Run with --expect-zero.
//
// Covers VarKind: Call CallSelf CallInput CallOutput CallGetter
//
// This is the one part of the port with no direct template. Metal dispatches
// through `visible_function_table`, CUDA through PTX callables and an indirect
// `call`; HIP C++ has NEITHER. What it does have is a single translation unit
// containing every callable, which opens a door the other two do not: the
// dispatch can be an ordinary `switch` over direct calls.
//
// So this file exists to answer a question before the emitter is written:
// WHICH DISPATCH SHAPE ACTUALLY WORKS ON gfx90a? Both candidates are below and
// both must pass, because the choice between them is then about code size and
// inlining rather than about whether the hardware cooperates.
//
// The traps this pins down, each of which compiles:
//
//   * An indirect call through a __device__ function pointer is legal C++ on
//     both platforms and is NOT obviously legal on AMDGPU, where indirect
//     calls force a real call-ABI frame. If it works at all it must still
//     return the right value -- a table indexed with the wrong stride returns
//     a plausible number from the neighbouring entry.
//
//   * Callables return a STRUCT BY VALUE (Dr.Jit packs all outputs into one).
//     Struct returns cross the ABI boundary differently from scalars, and this
//     is exactly where a wrong calling convention shows up as garbage in the
//     second field only.
//
//   * The call is MASKED. Lanes that do not call still read their output
//     registers afterwards, so those must be defined -- zeroed, not left
//     uninitialized. Reading an undefined register is not a crash, it is a
//     wrong pixel.
//
//   * Divergence is the normal case. Neighbouring lanes take DIFFERENT
//     instances, so a dispatch that only works when a whole wave agrees would
//     pass a careless test and fail on any real scene.

// The by-value return struct. Two fields of different types on purpose: a
// single-field struct can be returned in a register by accident and would hide
// an ABI mismatch.
struct Ret { float a; uint32_t b; };

// ---------------------------------------------------------------------------
//  Callables. In the emitter these are `func_<hash>`; the shape is what
//  matters -- (self, data pointer, live inputs...) -> by-value struct.
// ---------------------------------------------------------------------------

__device__ Ret func_0(uint32_t self, const uint8_t *data, float x) {
    Ret r;
    r.a = x * 2.0f + (float) (data ? data[0] : 0u);
    r.b = self * 10u + 0u;
    return r;
}

__device__ Ret func_1(uint32_t self, const uint8_t *data, float x) {
    Ret r;
    r.a = x * 3.0f + (float) (data ? data[0] : 0u);
    r.b = self * 10u + 1u;
    return r;
}

__device__ Ret func_2(uint32_t self, const uint8_t *data, float x) {
    Ret r;
    r.a = x * 4.0f + (float) (data ? data[0] : 0u);
    r.b = self * 10u + 2u;
    return r;
}

// The reference the checks below compare against.
__device__ Ret ref_call(uint32_t inst, uint32_t self, float x) {
    Ret r;
    float mul = inst == 0u ? 2.0f : (inst == 1u ? 3.0f : 4.0f);
    r.a = x * mul;
    r.b = self * 10u + inst;
    return r;
}

// ---------------------------------------------------------------------------
//  Candidate B: a table of __device__ function pointers.
//
//  Declared at namespace scope because that is how the emitter would have to
//  emit it -- one table per call-site signature, filled with the callables
//  that site can reach. If AMDGPU rejects this, the emitter has no choice but
//  the switch.
// ---------------------------------------------------------------------------

typedef Ret (*CallableFn)(uint32_t, const uint8_t *, float);

__device__ CallableFn call_table[3] = { func_0, func_1, func_2 };

// ---------------------------------------------------------------------------
//  Candidate C: ONE table for the whole kernel, cast per call site. REJECTED.
//
//  This was the decisive experiment. `callable_index` is assigned GLOBALLY
//  across the kernel (the `ctr` loops in cuda_eval.cpp:235 and
//  metal_eval.cpp:375), not per call site, and both of those backends build a
//  single table of that size and reinterpret it with each site's signature --
//  PTX because `.u64 callables[]` is untyped, Metal because
//  visible_function_table is reinterpretable. If HIP could do the same it
//  would inherit the whole scheme unchanged.
//
//  It cannot. A typed C++ array cannot hold callables of differing signatures,
//  and casting them to a common function-pointer type in the initializer:
//
//      typedef void (*AnyFn)();
//      __device__ AnyFn callables[4] = { (AnyFn) func_0, ..., (AnyFn) func_other };
//
//  compiles for gfx90a and is REJECTED BY NVRTC:
//
//      error: dynamic initialization is not supported for a __device__ variable
//
//  A bare function name is a constant address; a cast of one is not, as far as
//  NVRTC is concerned. Note candidate B above -- the same array WITHOUT casts
//  -- compiles on both arms, so this is specifically the cast, not the table.
//
//  That rules out candidate C for the CUDA shim, which is the only place the
//  vcall path can be executed before the MI210 arrives. An emitter shape that
//  cannot run under the shim is not testable, so:
//
//  ==> THE EMITTER USES CANDIDATE A, the switch over the global callable
//      index. It needs no table, no cast, and no indirect-call ABI on AMDGPU
//      at all; every arm is a direct call the compiler may inline. The cost is
//      code size proportional to the instances reachable from each call site.
// ---------------------------------------------------------------------------

DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    uint32_t err = 0u;

    // The instance each lane dispatches to. Deliberately lane-dependent so
    // neighbouring lanes disagree: a dispatch that silently requires wave
    // uniformity passes a uniform test and fails on every real scene.
    uint32_t r1 = r0 % 3u;          // callable index
    uint32_t r2 = r0 % 3u + 1u;     // `self` (instance ID, 1-based; 0 is null)
    float    r3 = in[r0];           // the live input

    // --- Candidate A: switch over direct calls ------------------------------
    //
    // Every callable is in this translation unit, so each arm is a direct,
    // inlinable call. The default arm is not dead code: `self` is data, and an
    // out-of-range instance must produce something defined rather than falling
    // off the end of the function.
    {
        Ret ret;
        ret.a = 0.0f;
        ret.b = 0u;
        switch (r1) {
            case 0u:  ret = func_0(r2, nullptr, r3); break;
            case 1u:  ret = func_1(r2, nullptr, r3); break;
            case 2u:  ret = func_2(r2, nullptr, r3); break;
            default:  break;
        }

        Ret want = ref_call(r1, r2, r3);
        if (ret.a != want.a) err |= 1u << 0;
        if (ret.b != want.b) err |= 1u << 1;
    }

    // --- Candidate B: indirect call through the table -----------------------
    {
        Ret ret = call_table[r1](r2, nullptr, r3);

        Ret want = ref_call(r1, r2, r3);
        if (ret.a != want.a) err |= 1u << 2;
        if (ret.b != want.b) err |= 1u << 3;
    }

    // --- The masked call ----------------------------------------------------
    //
    // Outputs are declared before the guard and zeroed on the else branch.
    // Half the lanes are masked off, so a backend that leaves them undefined
    // fails here rather than at integration time.
    {
        bool  r4 = (r0 % 2u) == 0u;
        float o_a;
        uint32_t o_b;

        if (r4) {
            Ret ret = call_table[r1](r2, nullptr, r3);
            o_a = ret.a;
            o_b = ret.b;
        } else {
            o_a = 0.0f;
            o_b = 0u;
        }

        Ret want = ref_call(r1, r2, r3);
        if (r4) {
            if (o_a != want.a || o_b != want.b) err |= 1u << 4;
        } else {
            if (o_a != 0.0f || o_b != 0u) err |= 1u << 5;
        }
    }

    // --- The offset table ---------------------------------------------------
    //
    // Dr.Jit packs (data_offset << 32) | callable_index into one uint64_t per
    // instance and indexes it by `self`. Reproduced here because the SHIFT is
    // the trap: extracting the index with a 32-bit shift, or the offset
    // without widening, silently reads the wrong half.
    {
        // Element 0 is the null instance; real instances are 1-based.
        uint64_t table[4];
        table[0] = 0ull;
        table[1] = ((uint64_t) 64u << 32) | 0ull;
        table[2] = ((uint64_t) 96u << 32) | 1ull;
        table[3] = ((uint64_t) 128u << 32) | 2ull;

        uint64_t entry = table[r2];
        uint32_t idx   = (uint32_t) (entry & 0xffffffffull);
        uint32_t off   = (uint32_t) (entry >> 32);

        if (idx != r1) err |= 1u << 6;
        if (off != 64u + 32u * r1) err |= 1u << 7;

        // Dispatch through the extracted index, which is the real path.
        Ret ret = call_table[idx](r2, nullptr, r3);
        Ret want = ref_call(r1, r2, r3);
        if (ret.a != want.a || ret.b != want.b) err |= 1u << 8;
    }

    // --- The chosen shape: switch over the GLOBAL callable index ------------
    //
    // As the emitter will produce it. The index comes out of the offset table
    // (not from `self` directly), the arms are the callables this site can
    // reach, and the indices are SPARSE because the index space is kernel-wide
    // -- a switch handles that, an array would need holes.
    {
        uint64_t table[4];
        table[0] = 0ull;
        table[1] = ((uint64_t) 64u  << 32) | 5ull;   // global indices, sparse
        table[2] = ((uint64_t) 96u  << 32) | 9ull;
        table[3] = ((uint64_t) 128u << 32) | 2ull;

        uint32_t idx = (uint32_t) (table[r2] & 0xffffffffull);

        Ret ret;
        ret.a = 0.0f;
        ret.b = 0u;
        switch (idx) {
            case 5u: ret = func_0(r2, nullptr, r3); break;
            case 9u: ret = func_1(r2, nullptr, r3); break;
            case 2u: ret = func_2(r2, nullptr, r3); break;
            default: break;
        }

        Ret want = ref_call(r1, r2, r3);
        if (ret.a != want.a || ret.b != want.b) err |= 1u << 11;
    }

    // --- CallGetter ----------------------------------------------------------
    //
    // A getter reads a field straight out of the instance's call data instead
    // of dispatching. The data pointer is the base plus the offset from the
    // table's high half, so this checks that arithmetic against a real buffer.
    {
        // Instance data laid out in the scratch region past out[n]: one float
        // per instance at the offsets used in the table above.
        uint8_t *base = (uint8_t *) (out + n);
        *(float *) (base + 64u)  = 11.0f;
        *(float *) (base + 96u)  = 22.0f;
        *(float *) (base + 128u) = 33.0f;

        uint64_t table[4];
        table[0] = 0ull;
        table[1] = ((uint64_t) 64u << 32) | 0ull;
        table[2] = ((uint64_t) 96u << 32) | 1ull;
        table[3] = ((uint64_t) 128u << 32) | 2ull;

        uint32_t off = (uint32_t) (table[r2] >> 32);
        float got = *(const float *) (base + off);

        float want = 11.0f + 11.0f * (float) r1;
        if (got != want) err |= 1u << 9;

        // Masked getter: a null instance (self == 0) must yield zero without
        // dereferencing the base at a bogus offset.
        uint32_t self0 = 0u;
        uint64_t e0 = table[self0];
        float got0 = (self0 != 0u) ? *(const float *) (base + (uint32_t) (e0 >> 32))
                                   : 0.0f;
        if (got0 != 0.0f) err |= 1u << 10;
    }

    out[r0] = (float) err;
}
