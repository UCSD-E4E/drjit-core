/*
    src/hip_eval.cpp -- HIP C++ source generation from a Dr.Jit computation graph

    Structural reference is metal_eval.cpp, not cuda_eval.cpp: Metal is the only
    backend here that emits source text, which is the shape we need (§3.1). The
    mapping is written up in tools/hip_validate/BACKEND_NOTES.md, and the source
    each opcode must produce is specified by the self-checking specimens in
    tools/hip_validate/kernels/spec_*.hip.

    Deliberate divergences from the Metal template, each with a test:

      * FP64 is NOT promoted away. Metal jitc_fail()s on Float64 because Apple
        GPUs have no doubles; gfx90a has full-rate FP64 (tests/hip_types.cpp).

      * Wave width comes from HIPDevice::warp_size, never a literal 64 (§3.3).

      * Device intrinsics only -- fabsf, __popc, __uint_as_float and friends.
        Clang's __builtin_* compile under hipcc and are rejected by NVRTC, so
        emitting them would pin generated source to one compiler
        (BACKEND_NOTES §11a).
*/

#include "eval.h"
#include "internal.h"
#include "var.h"
#include "log.h"
#include "strbuf.h"
#include "call.h"      // CallData, GetterData, jitc_call_upload, slot helpers
#include "loop.h"        // LoopData
#include "cond.h"        // CondData, jitc_cond_output_active
#include "op.h"          // ScatterCASDData
#include "trace.h"       // TraceData
#include <cstring>
#include "hip.h"
#include "hip_eval.h"
#include "hip_format.h"
#include "hip_literal.h"
#include "hip_prologue.h"
#include "hip_array.h"
#include "hip_scene.h"

// MUST BE LAST. This redefines `fmt` and `put` as macros, which would otherwise
// mangle StringBuffer's own put() member declarations in strbuf.h. metal_eval.cpp
// orders its includes the same way for the same reason.
#include "hip_codegen.h"

#if defined(DRJIT_ENABLE_HIP)

void jitc_hip_assemble_reset() { }

/// Scratch holding the formatted copy, so the main code buffer keeps the
/// unformatted text. Mirrors metal_reindent_scratch.
static std::string hip_reindent_scratch;

const char *jitc_hip_format(size_t *size_out) {
    hip_reindent_scratch = jitc_hip_reindent(buffer.get(), buffer.size());
    if (size_out)
        *size_out = hip_reindent_scratch.size();
    return hip_reindent_scratch.c_str();
}

// ---------------------------------------------------------------------------
//  Per-opcode rendering
// ---------------------------------------------------------------------------

static void render_unary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    fmt("$t $v = $s($v);\n", v, v, op, a);
}

static void render_binary(Variable *v, const char *op) {
    Variable *a = jitc_var(v->dep[0]);
    Variable *b = jitc_var(v->dep[1]);
    fmt("$t $v = $v $s $v;\n", v, v, a, op, b);
}

static void render_call(Variable *v, const char *fn, uint32_t n_args) {
    Variable *a0 = jitc_var(v->dep[0]);
    fmt("$t $v = $s($v", v, v, fn, a0);
    if (n_args >= 2)
        fmt(", $v", jitc_var(v->dep[1]));
    if (n_args >= 3)
        fmt(", $v", jitc_var(v->dep[2]));
    put(");\n");
}

/// True if `v` is double precision.
///
/// Worth a named helper rather than an inline comparison: nearly every math
/// function below has a distinct `double` spelling (sqrt vs sqrtf), and
/// reaching for the `f` form on a Float64 would silently round-trip through
/// single precision. That is the same class of mistake as Metal's Float64
/// promotion, which this backend exists partly to avoid.
static bool is_f64(const Variable *v) {
    return (VarType) v->type == VarType::Float64;
}

/// Intrinsics that move a value to and from its binary view (type_name_hip_bin).
///
/// PTX applies `and`/`or`/`xor`/`not` to a register's bits whatever the type
/// declares, so the CUDA backend needs no conversion. HIP C++ has no such
/// escape hatch -- `~x` on a float does not compile -- so bit-wise ops on
/// floating point have to be routed through these. Integers need no call: an
/// explicit `($b)` / `($t)` cast is exact and these return nullptr there.
/// Both spellings are checked on gfx90a and NVIDIA by spec_bitwise.hip.
/// Float16 has no such intrinsic pair that is spelled the same on both
/// platforms, so it goes through helpers the preamble defines instead.
static const char *to_bits_fn(const Variable *v) {
    switch ((VarType) v->type) {
        case VarType::Float16: return "drjit_half_to_bits";
        case VarType::Float32: return "__float_as_uint";
        case VarType::Float64: return "__double_as_longlong";
        default:               return nullptr;
    }
}

static const char *from_bits_fn(const Variable *v) {
    switch ((VarType) v->type) {
        case VarType::Float16: return "DRJIT_HALF_FROM_BITS";
        case VarType::Float32: return "__uint_as_float";
        case VarType::Float64: return "__longlong_as_double";
        default:               return nullptr;
    }
}

/// Pick the 32- or 64-bit spelling of a bit-manipulation intrinsic by operand
/// width. These are distinct functions on both platforms, and the narrow one
/// applied to a wide value truncates rather than failing to compile.
static void render_width(Variable *v, const char *n32, const char *n64) {
    render_call(v, type_size[v->type] == 8 ? n64 : n32, 1);
}

/// Bit-wise `op` on two same-typed operands, via the binary view when needed.
static void render_bitwise(Variable *v, const char *op) {
    Variable *a0 = jitc_var(v->dep[0]),
             *a1 = jitc_var(v->dep[1]);

    if (jitc_is_float(v))
        fmt("$t $v = $s(($b) $s($v) $s ($b) $s($v));\n", v, v,
            from_bits_fn(v), v, to_bits_fn(v), a0, op, v, to_bits_fn(v), a1);
    else
        fmt("$t $v = $v $s $v;\n", v, v, a0, op, a1);
}

/// Declare `_a`: the address of element `index` of `ptr`, as a pointer to the
/// binary view of `value`'s type.
///
/// The index is applied in ELEMENT units of the value type and only then
/// reinterpreted. Casting the pointer first and indexing afterwards compiles
/// and scales the index by the wrong element size -- silently, whenever the
/// two widths happen to differ.
static void render_atomic_addr(Variable *value, Variable *ptr, Variable *index) {
    fmt("    $b *_a = ($b *) (($t *) $v + $v);\n", value, value, value, ptr,
        index);
}

/// Emit a scatter-reduce (VarKind::Scatter with a non-Identity ReduceOp).
///
/// HIP/CUDA C++ expose atomics as an OVERLOAD SET with holes in it, not a
/// uniform family. Each hole below is a compile error rather than a wrong
/// answer -- except the first, which is why it is called out:
///
///   * atomicAdd has no `long long` overload, only `unsigned long long`.
///     Two's complement makes the reinterpretation exact for addition.
///   * atomicMin/Max/And/Or have no floating-point overload at all.
///   * atomicMul does not exist for any type.
///
/// Anything without a native form goes through a compare-and-swap loop over
/// the value's binary view. The loop must re-use the old value RETURNED by
/// atomicCAS rather than reloading the address: reloading lets two racing
/// lanes both observe a stale value, and one update is silently lost.
///
/// Both shapes are checked on gfx90a and NVIDIA by spec_memory.hip.
///
/// ReduceMode::Local (warp-level pre-aggregation, which the CUDA and Metal
/// backends implement) is deliberately ignored. Plain atomics are always
/// correct, just slower under heavy contention; adding the aggregation is a
/// performance task and belongs after the backend is correct.
/// `elem_offset` shifts the target by that many ELEMENTS of the value type,
/// for packet scatter-reduce: op.cpp hands the opcode an index already scaled
/// to the packet's first element, and the rest follow contiguously.
static void render_scatter_reduce(ReduceOp op, Variable *ptr, Variable *value,
                                  Variable *index, uint32_t elem_offset = 0) {
    VarType vt = (VarType) value->type;
    uint32_t width = type_size[(int) vt];
    bool is_float = jitc_is_float(value),
         is_int   = !is_float && (vt != VarType::Bool);

    // The index as source text, so the offset can ride along without every
    // address site below growing a second spelling. Zero offset reproduces the
    // previous output exactly, which keeps cached kernels valid.
    char idx[48];
    if (elem_offset)
        snprintf(idx, sizeof(idx), "r%u + %u", index->reg_index, elem_offset);
    else
        snprintf(idx, sizeof(idx), "r%u", index->reg_index);

    if (width != 4 && width != 8)
        jitc_fail("jitc_hip_render(): scatter-reduce on a %u-byte type is not "
                  "supported -- atomicCAS exists only at 32 and 64 bits, so "
                  "there is no correct lowering. (Dr.Jit normally widens these "
                  "before reaching the backend; if this fires, it did not.)",
                  width);
    if ((VarType) value->type == VarType::Float16)
        jitc_fail("jitc_hip_render(): scatter-reduce on Float16 is not "
                  "implemented -- there is no 16-bit atomicCAS, so it needs "
                  "the packed f16x2 treatment the CUDA backend uses.");

    // --- Native atomics ------------------------------------------------------
    const char *fn = nullptr;
    bool via_u64 = false;   // route a signed 64-bit add through the unsigned form

    switch (op) {
        case ReduceOp::Add:
            fn = "atomicAdd";
            via_u64 = (vt == VarType::Int64);
            break;

        case ReduceOp::Min: fn = is_int ? "atomicMin" : nullptr; break;
        case ReduceOp::Max: fn = is_int ? "atomicMax" : nullptr; break;
        case ReduceOp::And: fn = is_int ? "atomicAnd" : nullptr; break;
        case ReduceOp::Or:  fn = is_int ? "atomicOr"  : nullptr; break;

        // No atomicMul anywhere.
        case ReduceOp::Mul: fn = nullptr; break;

        default:
            jitc_fail("jitc_hip_render(): unhandled scatter-reduce operation "
                      "(%u).", (uint32_t) op);
    }

    if (fn) {
        if (via_u64)
            fmt("    $s(($b *) (($t *) $v + $s), ($b) $v);\n",
                fn, value, value, ptr, idx, value, value);
        else
            fmt("    $s(($t *) $v + $s, $v);\n",
                fn, value, ptr, idx, value);
        return;
    }

    // --- Compare-and-swap loop ----------------------------------------------
    //
    // The CAS width follows the VALUE width ($b is the same-width unsigned
    // integer): a 32-bit CAS on a double would spin on half of it forever.
    fmt("    $b *_a = ($b *) (($t *) $v + $s);\n"
        "    $b _o = *_a, _s;\n"
        "    do {\n"
        "        _s = _o;\n",
        value, value, value, ptr, idx,
        value);

    // Recombine through the binary view. from/to_bits are no-ops for integers.
    const char *from = from_bits_fn(value), *to = to_bits_fn(value);
    if (from)
        fmt("        $t _x = $s(_s);\n", value, from);
    else
        fmt("        $t _x = ($t) _s;\n", value, value);

    switch (op) {
        case ReduceOp::Mul:
            fmt("        $t _n = _x * $v;\n", value, value);
            break;
        case ReduceOp::Min:
            if (is_f64(value)) fmt("        $t _n = fmin(_x, $v);\n", value, value);
            else               fmt("        $t _n = fminf(_x, $v);\n", value, value);
            break;
        case ReduceOp::Max:
            if (is_f64(value)) fmt("        $t _n = fmax(_x, $v);\n", value, value);
            else               fmt("        $t _n = fmaxf(_x, $v);\n", value, value);
            break;
        default:
            jitc_fail("jitc_hip_render(): scatter-reduce fell through to the "
                      "CAS loop with an operation that has a native atomic "
                      "(%u) -- the two tables above have drifted apart.",
                      (uint32_t) op);
    }

    if (to)
        fmt("        _o = atomicCAS(_a, _s, ($b) $s(_n));\n", value, to);
    else
        fmt("        _o = atomicCAS(_a, _s, ($b) _n);\n", value);

    put("    } while (_s != _o);\n");
}

// ---------------------------------------------------------------------------
//  Packet memory
// ---------------------------------------------------------------------------
//
// A packet gather/scatter moves `count` CONTIGUOUS elements per lane in one
// operation, so the hardware can coalesce them.
//
// The two halves scale their index differently, and this is a property of the
// IR rather than a backend choice -- op.cpp has already multiplied the SCATTER
// index by `count` before the opcode sees it, and not the gather index:
//
//     PacketGather   byte address = ptr + index * (count * tsize)
//     PacketScatter  byte address = ptr + index * tsize
//
// Both CUDA (cuda_packet.cpp's two `mad.wide` forms) and Metal
// (metal_packet.cpp's two base-pointer expressions) agree. spec_packet.hip
// pins it, because getting it backwards compiles and then reads the wrong
// packet.
//
// Unlike those two backends, HIP needs no wide-vector machinery. Metal
// reinterprets through `uint4`/`as_type<>` and CUDA through `ld.global.v4`
// because MSL and PTX will not merge adjacent accesses for them; HIP compiles
// C++ through LLVM, which does. Measured on gfx90a, the element-wise form
// below and an explicit `float4` form produce the same machine code -- one
// global_load_dwordx4 and one global_store_dwordx4 -- even with the pointer
// arriving as an opaque void* from the params struct, because the `index *
// total_bytes` stride is enough for LLVM to infer 16-byte alignment. Staying
// element-wise is also the only form that is correct for a packet width that
// is not a power of two without a chunk-size computation.

static void render_gather_packet(Variable *v, Variable *ptr, Variable *index,
                                 Variable *mask) {
    uint32_t count       = (uint32_t) v->literal,
             tsize       = type_size[v->type],
             total_bytes = count * tsize;
    bool unmasked = mask->is_literal() && mask->literal == 1;

    fmt("const $t *$v_base = (const $t *) ((const char *) $v + (size_t) $v * $u);\n",
        v, v, v, ptr, index, total_bytes);

    // Every output is DEFINED even on a masked-off lane: they are referenced
    // unconditionally further down the kernel, so a conditional load would
    // propagate an uninitialised register.
    for (uint32_t i = 0; i < count; ++i) {
        if (unmasked)
            fmt("$t $v_out_$u = $v_base[$u];\n", v, v, i, v, i);
        else
            fmt("$t $v_out_$u = $v ? $v_base[$u] : ($t) 0;\n",
                v, v, i, mask, v, i, v);
    }
}

static void render_scatter_packet(Variable *v, Variable *ptr, Variable *index,
                                  Variable *mask) {
    PacketScatterData *psd = (PacketScatterData *) v->data;
    const std::vector<uint32_t> &values = psd->values;
    uint32_t count = (uint32_t) values.size();
    Variable *v0 = jitc_var(values[0]);
    bool unmasked = mask->is_literal() && mask->literal == 1;

    if (!unmasked)
        fmt("if ($v) {\n", mask);
    else
        put("{\n");

    if (psd->op != ReduceOp::Identity) {
        // ReduceMode::Local (warp pre-aggregation) is ignored here for the same
        // reason as in the scalar path: plain atomics are always correct, just
        // slower under contention.
        //
        // Each element gets its OWN block. render_scatter_reduce()'s
        // compare-and-swap form declares `_a`, `_o` and `_s` in the enclosing
        // scope, so emitting several into one block redeclares them -- a
        // compile error for any op without a native atomic (float min/max).
        //
        // This branch is currently UNREACHABLE: op.cpp's `use_packet_op` has no
        // HIP arm, so packet scatter-reduces decompose into scalar ones. It is
        // written and correct so that adding that arm -- which becomes worth
        // doing once ReduceMode::Local lands -- is a one-line change.
        for (uint32_t i = 0; i < count; ++i) {
            put("    {\n");
            render_scatter_reduce(psd->op, ptr, jitc_var(values[i]), index, i);
            put("    }\n");
        }
    } else {
        fmt("    $t *_p = ($t *) $v + $v;\n", v0, v0, ptr, index);
        for (uint32_t i = 0; i < count; ++i)
            fmt("    _p[$u] = $v;\n", i, jitc_var(values[i]));
    }

    put("}\n");
}

// ---------------------------------------------------------------------------
//  Ray tracing (HIP-RT)
// ---------------------------------------------------------------------------

/// Register the HIP-RT device header ahead of the kernel body.
///
/// Deliberately NOT the `intersectFunc` / `filterFunc` definitions, and the
/// reason is worth stating because BACKEND_NOTES §7a originally concluded the
/// opposite.
///
/// HIP-RT declares those two hooks and requires SOMEONE to define them. Who
/// depends on how the traversal library is linked:
///
///   * Hand-linking the device bitcode (`-Xclang -mlink-bitcode-file`, which
///     is what tools/hip_validate does) leaves them undefined, and the link
///     fails with `undefined hidden symbol: intersectFunc(...)`. That is where
///     §7a met them.
///
///   * hiprtBuildTraceKernels() -- the API the backend actually compiles
///     through, on the shim and on real hardware alike -- GENERATES them from
///     its numGeomTypes / numRayTypes / funcNameSets arguments and prepends
///     them to the source. An application definition is then a duplicate, and
///     the build fails with "function has already been defined".
///
/// So emitting them is correct for the harness and wrong for the backend.
/// run_tests.sh prepends them when it hand-links, which is the path that needs
/// them; codegen leaves them to HIP-RT.
///
/// Registration dedups by content, so a kernel with a hundred traces emits one
/// copy, and the globals machinery places it ahead of the kernel body.
static void jitc_hip_emit_trace_preamble(const HIPScene *scene) {
    // A function table means the scene has custom primitives, and HIP-RT
    // dispatches through the hooks to intersect them -- which means telling
    // hiprtBuildTraceKernels() about the geometry and ray types so it can
    // generate the right dispatch. That is not wired up, and proceeding would
    // report a miss for every custom shape: a black object in a render, a very
    // long way from its cause. Fail at the moment the offending scene is used.
    if (scene && scene->func_table)
        jitc_raise("jitc_hip_render(): this scene was configured with a "
                   "hiprtFuncTable, but custom-primitive intersection "
                   "functions are not implemented (PLAN.md Phase 5). Build the "
                   "scene without one, or pass the geometry/ray types through "
                   "to hiprtBuildTraceKernels().");

    size_t off = buffer.size();
    put("#include <hiprt/hiprt_device.h>\n");
    jitc_register_global(buffer.get() + off);
    buffer.rewind_to(off);
}

/// Emit a HIP-RT traversal.
///
/// The structure is Metal's (BACKEND_NOTES §7), and deliberately so: declare
/// all eight outputs at their MISS values, guard the traversal with the mask,
/// and let the hit path overwrite. Masked-off and missed lanes then fall
/// through with correct values and no separate clearing pass.
///
/// Two HIP-RT specifics:
///
///   * `hiprtHit` carries no geometry ID and no user instance ID -- six of the
///     eight outputs. The other two are read from instance-indexed tables
///     supplied by jit_hip_configure_scene(), which is exact rather than a
///     workaround: a HIP-RT instance references exactly one geometry, so both
///     really are properties of the instance (see hip_scene.h).
///
///   * The mask guards the whole traversal rather than selecting on its result.
///     An unmasked traversal against a scene the lane was not meant to touch is
///     not wasted work, it is a fault.
static void render_trace_ray(Variable *v) {
    TraceData *td = (TraceData *) v->data;
    Variable *valid = jitc_var(v->dep[0]);
    HIPScene *scene = (HIPScene *) (uintptr_t) jitc_var(v->dep[1])->literal;
    bool is_unmasked = valid->is_literal() && valid->literal == 1;

    jitc_hip_emit_trace_preamble(scene);

    // Miss values. +inf via its bit pattern rather than a literal, so nothing
    // depends on how the downstream compiler parses `INFINITY`.
    if (td->shadow) {
        fmt("bool $v_out_0 = false;\n", v);
    } else {
        fmt("bool $v_out_0 = false;\n"
            "f32 $v_out_1 = __uint_as_float(0x7f800000u);\n"
            "f32 $v_out_2 = 0.0f;\n"
            "f32 $v_out_3 = 0.0f;\n"
            "u32 $v_out_4 = 0u;\n"
            "u32 $v_out_5 = 0u;\n"
            "u32 $v_out_6 = 0u;\n"
            "u32 $v_out_7 = 0u;\n",
            v, v, v, v, v, v, v, v);
    }

    if (!is_unmasked)
        fmt("if ($v) {\n", valid);
    else
        put("{\n");

    Variable *ox   = jitc_var(td->indices[0]),
             *oy   = jitc_var(td->indices[1]),
             *oz   = jitc_var(td->indices[2]),
             *dx   = jitc_var(td->indices[3]),
             *dy   = jitc_var(td->indices[4]),
             *dz   = jitc_var(td->indices[5]),
             *tmin = jitc_var(td->indices[6]),
             *tmax = jitc_var(td->indices[7]);

    fmt("    hiprtRay _r;\n"
        "    _r.origin    = { $v, $v, $v };\n"
        "    _r.direction = { $v, $v, $v };\n"
        "    _r.minT      = $v;\n"
        "    _r.maxT      = $v;\n",
        ox, oy, oz, dx, dy, dz, tmin, tmax);

    // Shadow rays take the AnyHit traversal, which is what makes the `shadow`
    // flag worth having: it terminates at the first hit rather than sorting.
    const char *traversal = td->shadow ? "hiprtSceneTraversalAnyHit"
                                       : "hiprtSceneTraversalClosest";

    Variable *scene_h = jitc_var(scene->scene_handle);
    if (scene->func_table_handle) {
        fmt("    $s _tr((hiprtScene) $v, _r, hiprtFullRayMask,\n"
            "           hiprtTraversalHintDefault, nullptr,\n"
            "           (hiprtFuncTable) $v);\n",
            traversal, scene_h, jitc_var(scene->func_table_handle));
    } else {
        fmt("    $s _tr((hiprtScene) $v, _r);\n", traversal, scene_h);
    }

    put("    hiprtHit _h = _tr.getNextHit();\n"
        "    if (_h.hasHit()) {\n");

    if (td->shadow) {
        fmt("        $v_out_0 = true;\n", v);
    } else {
        fmt("        $v_out_0 = true;\n"
            "        $v_out_1 = _h.t;\n"
            "        $v_out_2 = _h.uv.x;\n"
            "        $v_out_3 = _h.uv.y;\n"
            "        $v_out_4 = _h.instanceID;\n"
            "        $v_out_5 = _h.primID;\n",
            v, v, v, v, v, v);

        // The two outputs hiprtHit does not carry. Absent a table the answer is
        // 0, which is also what a single-geometry scene would report.
        if (scene->geometry_ids_handle)
            fmt("        $v_out_6 = ((const u32 *) $v)[_h.instanceID];\n",
                v, jitc_var(scene->geometry_ids_handle));
        if (scene->user_instance_ids_handle)
            fmt("        $v_out_7 = ((const u32 *) $v)[_h.instanceID];\n",
                v, jitc_var(scene->user_instance_ids_handle));
    }

    put("    }\n"
        "}\n");
}

/// Emit a math call, choosing the double or float spelling by operand type.
/// `base` is the double name; the float form is `base` + "f".
static void render_math(Variable *v, const char *base, uint32_t n_args = 1) {
    if (is_f64(v)) {
        render_call(v, base, n_args);
    } else {
        char buf[32];
        snprintf(buf, sizeof(buf), "%sf", base);
        render_call(v, buf, n_args);
    }
}

static void jitc_hip_render(Variable *v) {
    switch ((VarKind) v->kind) {
        case VarKind::Nop:
            break;

        case VarKind::Undefined:
            fmt("$t $v = ($t) 0;\n", v, v, v);
            break;

        case VarKind::Literal:
            fmt("$t $v = $s;\n", v, v,
                jitc_hip_literal((VarType) v->type, v->literal).c_str());
            break;

        case VarKind::Counter:
            // The prologue already materialises the global thread index as r0.
            fmt("$t $v = ($t) r0;\n", v, v, v);
            break;

        // --- Unary ----------------------------------------------------------
        case VarKind::Neg:
            // Two's-complement negation on integers; -x on floats.
            fmt("$t $v = -$v;\n", v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::Not: {
            Variable *a = jitc_var(v->dep[0]);
            // `~` on a Bool is NOT logical negation: it yields 0xfe, which
            // converts straight back to `true`, so the operation silently
            // becomes a no-op rather than failing (spec_bitwise.hip bit 16).
            if ((VarType) v->type == VarType::Bool)
                fmt("$t $v = !$v;\n", v, v, a);
            else if (jitc_is_float(v))
                fmt("$t $v = $s(~($b) $s($v));\n", v, v, from_bits_fn(v), v,
                    to_bits_fn(v), a);
            else
                fmt("$t $v = ~$v;\n", v, v, a);
            break;
        }

        // The `f` suffix is not cosmetic: fabsf/sqrtf on a Float64 round-trip
        // through single precision and silently lose the bottom 29 bits.
        case VarKind::Abs:
            if (jitc_is_float(v)) render_math(v, "fabs");
            else                  render_call(v, "abs", 1);
            break;

        case VarKind::Sqrt:
            render_math(v, "sqrt");
            break;

        // --- Binary ---------------------------------------------------------
        case VarKind::Add: render_binary(v, "+"); break;
        case VarKind::Sub: render_binary(v, "-"); break;
        case VarKind::Mul: render_binary(v, "*"); break;
        case VarKind::Div: render_binary(v, "/"); break;
        case VarKind::Mod: render_binary(v, "%"); break;
        // A mismatched operand type means the second argument is a Bool mask
        // rather than a value to combine bit-wise -- Dr.Jit spells `select`
        // that way. The CUDA backend detects the same case and emits `selp`.
        case VarKind::And: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type)
                fmt("$t $v = $v ? $v : ($t) 0;\n", v, v, a1, a0, v);
            else
                render_bitwise(v, "&");
            break;
        }

        case VarKind::Or: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            if (a0->type != a1->type) {
                // A set mask selects an all-ones word: the identity of OR.
                if (jitc_is_float(v))
                    fmt("$t $v = $v ? $s(~($b) 0) : $v;\n", v, v, a1,
                        from_bits_fn(v), v, a0);
                else
                    fmt("$t $v = $v ? ($t) ~($b) 0 : $v;\n", v, v, a1, v, v, a0);
            } else {
                render_bitwise(v, "|");
            }
            break;
        }

        case VarKind::Xor: render_bitwise(v, "^"); break;
        case VarKind::Shl: render_binary(v, "<<"); break;
        case VarKind::Shr: render_binary(v, ">>"); break;

        // --- Comparisons ----------------------------------------------------
        case VarKind::Eq:  render_binary(v, "=="); break;
        case VarKind::Neq: render_binary(v, "!="); break;
        case VarKind::Lt:  render_binary(v, "<");  break;
        case VarKind::Le:  render_binary(v, "<="); break;
        case VarKind::Gt:  render_binary(v, ">");  break;
        case VarKind::Ge:  render_binary(v, ">="); break;

        // --- Ternary --------------------------------------------------------
        case VarKind::Select:
            fmt("$t $v = $v ? $v : $v;\n", v, v, jitc_var(v->dep[0]),
                jitc_var(v->dep[1]), jitc_var(v->dep[2]));
            break;

        case VarKind::Fma:
            // Floats must use the fused form: a * b + c would silently differ
            // in precision from the CUDA backend (spec_arith.hip).
            //
            // Integers must NOT. There is no __mad on either platform, and
            // nothing to fuse: the multiply is exact modulo the type width, so
            // no precision exists for the add to recover.
            if (jitc_is_float(v))
                render_math(v, "fma", 3);
            else
                fmt("$t $v = $v * $v + $v;\n", v, v, jitc_var(v->dep[0]),
                    jitc_var(v->dep[1]), jitc_var(v->dep[2]));
            break;

        case VarKind::Min:
            if (jitc_is_float(v)) render_math(v, "fmin", 2);
            else                  render_call(v, "min", 2);
            break;
        case VarKind::Max:
            if (jitc_is_float(v)) render_math(v, "fmax", 2);
            else                  render_call(v, "max", 2);
            break;

        // --- Rounding -------------------------------------------------------
        //
        // Round is rintf (nearest-EVEN), NOT roundf (half away from zero) --
        // the latter would disagree with the CUDA backend on exactly the .5
        // cases (spec_compare.hip, BACKEND_NOTES §11a).
        case VarKind::Ceil:  render_math(v, "ceil");  break;
        case VarKind::Floor: render_math(v, "floor"); break;
        case VarKind::Round: render_math(v, "rint");  break;
        case VarKind::Trunc: render_math(v, "trunc"); break;

        // --- Transcendentals ------------------------------------------------
        //
        // The fast device forms, matching what the CUDA backend selects from
        // the same VarKinds ("multi-function generator"). Note there is no
        // __exp2f on either platform, unlike its neighbours.
        case VarKind::Sin:  render_call(v, "__sinf", 1);  break;
        case VarKind::Cos:  render_call(v, "__cosf", 1);  break;
        case VarKind::Exp2: render_call(v, "exp2f", 1);   break;
        case VarKind::Log2: render_call(v, "__log2f", 1); break;
        case VarKind::Tanh: render_math(v, "tanh");       break;

        // --- Fast approximations ---------------------------------------------
        case VarKind::Rcp:
            fmt("$t $v = ($t) 1 / $v;\n", v, v, v, jitc_var(v->dep[0]));
            break;
        case VarKind::RcpApprox:   render_call(v, "__frcp_rn", 1);  break;
        case VarKind::RSqrtApprox: render_call(v, "rsqrtf", 1);     break;
        case VarKind::SqrtApprox:  render_call(v, "__fsqrt_rn", 1); break;
        case VarKind::DivApprox:   render_call(v, "__fdividef", 2); break;

        // --- Bit counting -----------------------------------------------------
        //
        // 32- and 64-bit forms are DIFFERENT intrinsics, not overloads: the
        // 32-bit spelling applied to a 64-bit value narrows it silently and
        // still compiles (spec_bitwise.hip).
        case VarKind::Popc: render_width(v, "__popc", "__popcll"); break;
        case VarKind::Clz:  render_width(v, "__clz",  "__clzll");  break;
        case VarKind::Brev: render_width(v, "__brev", "__brevll"); break;

        case VarKind::Ctz: {
            // No __ctz exists on either platform. `__ffs(x) - 1` is the obvious
            // substitute and is wrong at zero: __ffs returns 0, the subtraction
            // wraps, and the result disagrees with Dr.Jit's constant folder,
            // which defines ctz(0) as the bit width. clz(brev(x)) gives that
            // for free -- and is what the CUDA backend emits.
            bool wide = type_size[v->type] == 8;
            fmt("$t $v = $s($s($v));\n", v, v, wide ? "__clzll" : "__clz",
                wide ? "__brevll" : "__brev", jitc_var(v->dep[0]));
            break;
        }

        // --- Wide multiplication ----------------------------------------------
        //
        // These exist as opcodes precisely because `a * b` is computed modulo
        // the operand width, discarding what they need (spec_mulwide.hip).
        case VarKind::MulHi:
            if (type_size[v->type] == 8)
                render_call(v, jitc_is_uint(v) ? "__umul64hi" : "__mul64hi", 2);
            else
                render_call(v, jitc_is_uint(v) ? "__umulhi" : "__mulhi", 2);
            break;

        case VarKind::MulWide:
            // Widen BEFORE multiplying; casting the product would truncate.
            fmt("$t $v = ($t) $v * ($t) $v;\n", v, v, v,
                jitc_var(v->dep[0]), v, jitc_var(v->dep[1]));
            break;

        // --- Conversions -------------------------------------------------------
        case VarKind::Cast:
            fmt("$t $v = ($t) $v;\n", v, v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::Bitcast: {
            Variable *a = jitc_var(v->dep[0]);

            // Reinterpret through the device intrinsics rather than a pointer
            // cast or __builtin_bit_cast -- the latter is Clang-only and NVRTC
            // rejects it (BACKEND_NOTES §11a).
            //
            // Reinterpretation is needed only when EXACTLY ONE side is
            // floating point. Otherwise a plain cast is already exact: between
            // two same-width integers it yields the bit pattern, and a
            // same-type bitcast (which Dr.Jit does emit -- see the size-1
            // gather in tests/mem.cpp) is a copy.
            //
            // Reaching for a float intrinsic in either of those cases converts
            // the VALUE first. `(f32) __float_as_uint(x)` turns 1.0f into
            // 1065353216.0f, and Int64(6) into 0x40c00000 (spec_cast.hip
            // bit 14) -- both compile and run.
            bool a_flt = jitc_is_float(a), v_flt = jitc_is_float(v);
            const char *fn = nullptr;
            if (a_flt && !v_flt)
                fn = to_bits_fn(a);
            else if (!a_flt && v_flt)
                fn = from_bits_fn(v);

            if (fn)
                fmt("$t $v = ($t) $s($v);\n", v, v, v, fn, a);
            else
                fmt("$t $v = ($t) $v;\n", v, v, v, a);
            break;
        }

        // --- Memory -----------------------------------------------------------
        //
        // Dep layouts differ between the two and are NOT guessable, so they are
        // taken from metal_eval.cpp / metal_scatter.cpp rather than inferred:
        //
        //   Gather:  dep[0]=src   dep[1]=index dep[2]=mask
        //   Scatter: dep[0]=ptr   dep[1]=value dep[2]=index dep[3]=mask
        //
        // Note the masked gather is a TERNARY, not a load followed by a select.
        // `?:` short-circuits, so the load genuinely does not execute on masked
        // lanes; the two-statement form would read out of bounds on every one
        // of them (spec_memory.hip).
        // Emitted only under JitFlag::Debug -- which is exactly the mode a
        // backend under construction is run in, so leaving it unhandled makes
        // the one setting that would explain a wrong answer the one setting
        // that cannot run.
        //
        // The result is the incoming MASK with out-of-bounds lanes cleared, not
        // a validity flag: the gather or scatter downstream is predicated on
        // it. The comparison is deliberately written with the bound cast rather
        // than the index, so the usual arithmetic conversions promote a signed
        // index to unsigned (a negative index must fail, and signed it would
        // pass) and widen the bound against a 64-bit index rather than
        // truncating it. Both are asserted by spec_bounds.hip.
        case VarKind::BoundsCheck: {
            Variable *index = jitc_var(v->dep[0]),
                     *mask  = jitc_var(v->dep[1]),
                     *buf   = jitc_var(v->dep[2]);
            uint32_t size = (uint32_t) v->literal;

            // Predicated on `mask && !in_bounds`, not on the comparison alone:
            // a lane the caller already disabled must not report a position it
            // never read. Concurrent writers race, which is fine -- the host
            // wants a non-zero word and one offending index.
            fmt("$t $v = $v && ($v < (u32) $u);\n"
                "if ($v && !$v)\n"
                "    *(u32 *) $v = (u32) $v;\n",
                v, v, mask, index, size,
                mask, v,
                buf, index);
            break;
        }

        case VarKind::Gather: {
            Variable *src   = jitc_var(v->dep[0]);
            Variable *index = jitc_var(v->dep[1]);
            Variable *mask  = jitc_var(v->dep[2]);
            bool unmasked = mask->is_literal() && mask->literal == 1;

            if (unmasked)
                fmt("$t $v = ((const $t *) $v)[$v];\n", v, v, v, src, index);
            else
                fmt("$t $v = ($v) ? ((const $t *) $v)[$v] : ($t) 0;\n",
                    v, v, mask, v, src, index, v);
            break;
        }

        case VarKind::TraceRay:
            render_trace_ray(v);
            break;

        case VarKind::PacketGather:
            render_gather_packet(v, jitc_var(v->dep[0]), jitc_var(v->dep[1]),
                                 jitc_var(v->dep[2]));
            break;

        case VarKind::PacketScatter:
            render_scatter_packet(v, jitc_var(v->dep[0]), jitc_var(v->dep[1]),
                                  jitc_var(v->dep[2]));
            break;

        case VarKind::Scatter: {
            Variable *ptr   = jitc_var(v->dep[0]);
            Variable *value = jitc_var(v->dep[1]);
            Variable *index = jitc_var(v->dep[2]);
            Variable *mask  = jitc_var(v->dep[3]);

            ReduceOp op = (ReduceOp) (uint32_t) v->literal;
            bool unmasked = mask->is_literal() && mask->literal == 1;

            if (op == ReduceOp::Identity) {
                if (unmasked)
                    fmt("(($t *) $v)[$v] = $v;\n", value, ptr, index, value);
                else
                    fmt("if ($v) (($t *) $v)[$v] = $v;\n",
                        mask, value, ptr, index, value);
                break;
            }

            if (!unmasked)
                fmt("if ($v) {\n", mask);
            else
                put("{\n");
            render_scatter_reduce(op, ptr, value, index);
            put("}\n");
            break;
        }

        // --- Atomics with a return value ---------------------------------------
        //
        // All three return the OLD value; getting that backwards is the easy
        // mistake, and it is what spec_memory.hip pins down. Each declares its
        // result BEFORE the mask test so that masked-off lanes still have a
        // defined value -- the variable is referenced unconditionally further
        // down the kernel.
        //
        // The CUDA and Metal backends warp-aggregate ScatterInc (one atomic per
        // group of lanes hitting the same address, then a shuffle). That is a
        // contention optimisation, not a semantic difference; a plain atomic
        // per lane produces the same numbering. Deferred with ReduceMode::Local
        // in render_scatter_reduce().
        case VarKind::ScatterInc: {
            Variable *ptr   = jitc_var(v->dep[0]),
                     *index = jitc_var(v->dep[1]),
                     *mask  = jitc_var(v->dep[2]);
            bool unmasked = mask->is_literal() && mask->literal == 1;

            fmt("$t $v = ($t) 0;\n", v, v, v);
            if (!unmasked)
                fmt("if ($v) {\n", mask);
            else
                put("{\n");
            fmt("    $v = atomicAdd(($t *) $v + $v, ($t) 1);\n", v, v, ptr,
                index, v);
            put("}\n");
            v->consumed = 1;
            break;
        }

        case VarKind::ScatterExch: {
            Variable *ptr   = jitc_var(v->dep[0]),
                     *value = jitc_var(v->dep[1]),
                     *index = jitc_var(v->dep[2]),
                     *mask  = jitc_var(v->dep[3]);
            bool unmasked = mask->is_literal() && mask->literal == 1;

            fmt("$t $v = ($t) 0;\n", value, v, value);
            if (!unmasked)
                fmt("if ($v) {\n", mask);
            else
                put("{\n");

            // Via the binary view: atomicExch has no 64-bit float overload.
            // The reinterpretation is on the VALUE -- reinterpreting the
            // pointer instead compiles and scales the index by the wrong
            // element size (spec_memory.hip bit 18).
            render_atomic_addr(value, ptr, index);
            const char *to = to_bits_fn(value), *from = from_bits_fn(value);
            if (to)
                fmt("    $b _o = atomicExch(_a, ($b) $s($v));\n", value, value,
                    to, value);
            else
                fmt("    $b _o = atomicExch(_a, ($b) $v);\n", value, value, value);
            if (from)
                fmt("    $v = $s(_o);\n", v, from);
            else
                fmt("    $v = ($t) _o;\n", v, value);

            put("}\n");
            v->consumed = 1;
            break;
        }

        case VarKind::ScatterCAS: {
            Variable *ptr     = jitc_var(v->dep[0]),
                     *compare = jitc_var(v->dep[1]),
                     *value   = jitc_var(v->dep[2]),
                     *index   = jitc_var(v->dep[3]);

            // The mask hangs off v->data here rather than a dep slot, all four
            // of which are taken.
            ScatterCASDData *cas_data = (ScatterCASDData *) v->data;
            Variable *mask = jitc_var(cas_data->mask);
            bool unmasked = mask->is_literal() && mask->literal == 1;

            // Two outputs, read back by VarKind::Extract.
            fmt("$t $v_out_0 = ($t) 0;\n"
                "bool $v_out_1 = false;\n",
                value, v, value, v);

            if (!unmasked)
                fmt("if ($v) {\n", mask);
            else
                put("{\n");

            render_atomic_addr(value, ptr, index);
            const char *to = to_bits_fn(value), *from = from_bits_fn(value);
            if (to)
                fmt("    $b _c = ($b) $s($v);\n"
                    "    $b _o = atomicCAS(_a, _c, ($b) $s($v));\n",
                    value, value, to, compare,
                    value, value, to, value);
            else
                fmt("    $b _c = ($b) $v;\n"
                    "    $b _o = atomicCAS(_a, _c, ($b) $v);\n",
                    value, value, compare,
                    value, value, value);

            if (from)
                fmt("    $v_out_0 = $s(_o);\n", v, from);
            else
                fmt("    $v_out_0 = ($t) _o;\n", v, value);

            // "Did it swap" is old == expected. The return value alone does not
            // say; a CAS that failed also returns something.
            fmt("    $v_out_1 = (_o == _c);\n", v);
            put("}\n");
            v->consumed = 1;
            break;
        }

        case VarKind::Extract: {
            Variable *src = jitc_var(v->dep[0]);
            uint32_t sub_index = (uint32_t) v->literal;

            // The multi-output ops this backend emits. TexLookup lands in the
            // default case and says so explicitly rather than silently
            // extracting field 0.
            if ((VarKind) src->kind == VarKind::ScatterCAS ||
                (VarKind) src->kind == VarKind::PacketGather ||
                (VarKind) src->kind == VarKind::TraceRay)
                fmt("$t $v = $v_out_$u;\n", v, v, src, sub_index);
            else
                fmt("$t $v = $v; // extract[$u]\n", v, v, src, sub_index);
            break;
        }

        // --- Control flow -------------------------------------------------------
        //
        // Ports essentially verbatim from metal_eval.cpp: both backends emit
        // C-family source, so `while (true) { ... }` and `if/else` are the same
        // construct. What does NOT port by inspection is the SSA bookkeeping
        // below, which is why it is followed closely rather than reconstructed.
        case VarKind::LoopStart: {
            const LoopData *ld = (LoopData *) v->data;
            // Seed each loop-carried variable from its value outside the loop.
            for (size_t i = 0; i < ld->size; ++i) {
                Variable *inner_in = jitc_var(ld->inner_in[i]),
                         *outer_in = jitc_var(ld->outer_in[i]);
                if (inner_in == outer_in || !inner_in->reg_index ||
                    inner_in->is_array())
                    continue;
                if (outer_in->reg_index)
                    fmt("$t $v = $v;\n", inner_in, inner_in, outer_in);
                else
                    fmt("$t $v = ($t) 0;\n", inner_in, inner_in, inner_in);
            }
            put("while (true) {\n");
            break;
        }

        case VarKind::LoopCond:
            fmt("if (!$v) break;\n", jitc_var(v->dep[1]));
            break;

        case VarKind::LoopEnd: {
            const LoopData *ld = (LoopData *) jitc_var(v->dep[0])->data;
            uint32_t size = (uint32_t) ld->size;

            // THE BACK EDGE. Copying inner_out -> inner_in naively is wrong when
            // the sets alias: an earlier copy can clobber a value a later one
            // still needs (a swap being the minimal example). So stage every
            // aliasing output into a temporary first, then assign.
            //
            // `scratch` is borrowed as a marker and MUST be cleared afterwards,
            // or it corrupts jitc_var_traverse()'s visited tracking in
            // jit_eval() -- a failure that would appear far from here.
            for (uint32_t i = 0; i < size; ++i) {
                jitc_var(ld->inner_in[i])->scratch = 0;
                jitc_var(ld->inner_out[i])->scratch = 0;
            }

            auto carried = [&](uint32_t i, Variable *&in, Variable *&out) {
                in  = jitc_var(ld->inner_in[i]);
                out = jitc_var(ld->inner_out[i]);
                return !(in == out || !in->reg_index || !out->reg_index ||
                         in->is_array());
            };

            Variable *in, *out;
            for (uint32_t i = 0; i < size; ++i)
                if (carried(i, in, out))
                    in->scratch = 1;

            for (uint32_t i = 0; i < size; ++i)
                if (carried(i, in, out) && out->scratch == 1) {
                    fmt("$t $v_tmp = $v;\n", out, out, out);
                    out->scratch = 2;
                }

            for (uint32_t i = 0; i < size; ++i)
                if (carried(i, in, out)) {
                    if (out->scratch == 2)
                        fmt("$v = $v_tmp;\n", in, out);
                    else
                        fmt("$v = $v;\n", in, out);
                }

            for (uint32_t i = 0; i < size; ++i) {
                jitc_var(ld->inner_in[i])->scratch = 0;
                jitc_var(ld->inner_out[i])->scratch = 0;
            }

            put("}\n");
            break;
        }

        case VarKind::LoopPhi:
            // Arrays alias their backing storage; scalars need no declaration
            // here because LoopStart already emitted one.
            if (v->is_array())
                v->reg_index = jitc_var(v->dep[3])->reg_index;
            break;

        case VarKind::LoopOutput: {
            const LoopData *ld = (LoopData *) jitc_var(v->dep[0])->data;
            for (size_t i = 0; i < ld->size; ++i) {
                if (jitc_var(ld->outer_out[i]) != v)
                    continue;
                Variable *inner_in = jitc_var(ld->inner_in[i]);
                if (v->reg_index && inner_in->reg_index)
                    fmt("$t $v = $v;\n", v, v, inner_in);
                break;
            }
            break;
        }

        case VarKind::CondStart: {
            const CondData *cd = (CondData *) v->data;
            // Outputs are declared BEFORE the if, so both arms assign to the
            // same variable and it stays live afterwards.
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vo = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo))
                    fmt("$t $v;\n", vo, vo);
            }
            fmt("if ($v) {\n", jitc_var(v->dep[0]));
            break;
        }

        case VarKind::CondMid: {
            const CondData *cd = (CondData *) jitc_var(v->dep[0])->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vt = jitc_var(cd->indices_t[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo) && vt->reg_index)
                    fmt("$v = $v;\n", vo, vt);
            }
            put("} else {\n");
            break;
        }

        case VarKind::CondEnd: {
            const CondData *cd = (CondData *) jitc_var(v->dep[0])->data;
            for (size_t i = 0; i < cd->indices_out.size(); ++i) {
                Variable *vf = jitc_var(cd->indices_f[i]),
                         *vo = jitc_var(cd->indices_out[i]);
                if (jitc_cond_output_active(vo) && vf->reg_index)
                    fmt("$v = $v;\n", vo, vf);
            }
            put("}\n");
            break;
        }

        case VarKind::CondOutput:
            break;

        // --- Dynamic dispatch ---------------------------------------------------
        case VarKind::Call: {
            Variable *a0 = jitc_var(v->dep[0]),
                     *a1 = jitc_var(v->dep[1]);
            jitc_var_call_assemble((CallData *) v->data, v->reg_index,
                                   a0->reg_index, a1->reg_index);
            break;
        }

        case VarKind::CallGetter: {
            Variable *index = jitc_var(v->dep[0]),
                     *mask  = jitc_var(v->dep[1]);
            jitc_var_call_getter_assemble(v, index, mask);
            break;
        }

        // Both are declared by the dispatch/callable machinery rather than
        // here: CallInput becomes a by-value parameter inside the callable and
        // needs nothing at the call site, and CallOutput is unpacked from the
        // return struct by jitc_var_call_assemble_hip().
        case VarKind::CallInput:
        case VarKind::CallOutput:
            break;

        case VarKind::CallSelf:
            fmt("u32 $v = self;\n", v);
            break;

        // --- Per-lane variable arrays -------------------------------------------
        case VarKind::Array:
            jitc_hip_render_array(v, v->dep[0] ? jitc_var(v->dep[0]) : nullptr);
            break;

        case VarKind::ArrayInit:
            jitc_hip_render_array_init(v, jitc_var(v->dep[0]),
                                       jitc_var(v->dep[1]));
            break;

        case VarKind::ArrayWrite:
            jitc_hip_render_array_write(v, jitc_var(v->dep[0]),
                                        jitc_var(v->dep[1]),
                                        jitc_var(v->dep[2]),
                                        v->dep[3] ? jitc_var(v->dep[3]) : nullptr);
            break;

        case VarKind::ArrayRead:
            jitc_hip_render_array_read(v, jitc_var(v->dep[0]),
                                       jitc_var(v->dep[1]),
                                       v->dep[2] ? jitc_var(v->dep[2]) : nullptr);
            break;

        // No code: the phi aliases its predecessor's storage.
        case VarKind::ArrayPhi:
            v->reg_index = jitc_var(v->dep[0])->reg_index;
            break;

        case VarKind::ArraySelect:
            jitc_hip_render_array_select(v, jitc_var(v->dep[0]),
                                         jitc_var(v->dep[1]),
                                         jitc_var(v->dep[2]));
            break;

        default:
            jitc_fail("jitc_hip_render(): unhandled variable kind \"%s\"! The "
                      "source this opcode must produce is specified in "
                      "tools/hip_validate/kernels/spec_*.hip.",
                      var_kind_name[(uint32_t) v->kind]);
    }
}

// ---------------------------------------------------------------------------
//  Dynamic dispatch (vcalls)
// ---------------------------------------------------------------------------
//
// The dispatch SHAPE is a `switch`, not a function table, and that is a
// measured choice rather than a stylistic one -- see BACKEND_NOTES §9a and
// tools/hip_validate/kernels/spec_call.hip, which tested all three candidates
// on gfx90a and on NVRTC. The single-table-with-casts form that CUDA and Metal
// use is rejected by NVRTC ("dynamic initialization is not supported for a
// __device__ variable"), and NVRTC is the shim's compiler -- so that shape
// could be compiled for the target but never executed anywhere we have.
//
// Two consequences worth stating plainly:
//
//   * The switch keys on `self` (the instance ID), not on the callable index.
//     Instance IDs are known at emission time; the callable index is assigned
//     later, in a pass over globals_map that has not run yet when the dispatch
//     site is written. Instances sharing a callable share a case BODY -- their
//     labels fall through to one call -- so code size still scales with unique
//     callables, not with instances.
//
//   * The offset table's low half (the callable index) is therefore unused by
//     HIP. Only the high half, the instance's data offset, is read.

/// Does this call have any live outputs?
static bool jitc_hip_call_has_out(const CallData *call) {
    for (uint32_t i = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        Variable *v = jitc_var(call->outer_out[i]);
        if (v && v->reg_index && v->param_type != ParamType::Input)
            return true;
    }
    return false;
}

/// Name of the by-value return struct: `Ret_` + the mangled active output
/// types. Keyed by content so two calls with the same output signature share
/// one definition.
static void jitc_hip_put_ret_type(const CallData *call) {
    put("Ret_");
    for (uint32_t i = 0; i < call->n_out; ++i)
        if (call->out_offset[i] != (uint32_t) -1)
            put(type_mangle[jitc_var(call->inner_out[i])->type]);
}

/// Register the return struct definition as a global, so it precedes both the
/// callables and the kernel. Registration dedups by content.
static void jitc_hip_emit_ret_struct(const CallData *call) {
    if (!jitc_hip_call_has_out(call))
        return;
    size_t off = buffer.size();
    put("struct ");
    jitc_hip_put_ret_type(call);
    put(" {\n");
    uint32_t k = 0;
    for (uint32_t i = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        fmt("$t r$u;\n", jitc_var(call->inner_out[i]), k);
        k++;
    }
    put("};");
    jitc_register_global(buffer.get() + off);
    buffer.rewind_to(off);
}

/// The typed parameter list shared by a callable's definition and its forward
/// declaration: a fixed prefix (index, self, data), then the call-data base
/// pointer if this callable contains a nested call, then one by-value
/// parameter per live input.
///
/// Metal also threads a `call_table` handle through every callable. HIP does
/// not need one -- the switch names its targets directly -- so that parameter
/// is absent, and nested dispatch reaches its targets the same way.
static void jitc_hip_callable_signature(const CallData *call, bool with_names) {
    if (with_names)
        put("u32 index, u32 self, const u8 *data");
    else
        put("u32, u32, const u8 *");

    if (call->use_nested) {
        if (with_names)
            put(", const u8 *base");
        else
            put(", const u8 *");
    }

    for (uint32_t i = 0; i < call->n_in; ++i) {
        if (!call->in_active[i])
            continue;
        Variable *vo = jitc_var(call->outer_in[i]);
        if (with_names)
            fmt(", $t a$u", vo, i);
        else
            fmt(", $t", vo);
    }
}

/// Emit one callable body (`func_<hash>`), i.e. one instance's version of the
/// call. Invoked from jitc_assemble_func() once per instance.
void jitc_hip_assemble_func(const CallData *call, uint32_t inst) {
    jitc_hip_emit_ret_struct(call);

    // __device__, not __global__: these are ordinary functions called from the
    // kernel, and `static` would let the compiler drop the ones only reached
    // through a switch arm it cannot see through.
    put("__device__ ");
    if (jitc_hip_call_has_out(call))
        jitc_hip_put_ret_type(call);
    else
        put("void");
    put(" func_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^(");
    jitc_hip_callable_signature(call, /*with_names=*/true);
    put(") {\n");
    fmt("// Call: $s\n", call->name.c_str());

    // Bind this instance's capture slots so jitc_call_slot_rel_offset()
    // resolves them in O(1) below.
    jitc_call_bind_slots(call, inst);

    for (size_t i = 0; i < schedule.size(); ++i) {
        ScheduledVariable &sv = schedule[i];
        Variable *v = jitc_var(sv.index);
        VarType vt = (VarType) v->type;
        VarKind kind = (VarKind) v->kind;

        if (kind == VarKind::Counter) {
            // Inside a callable the kernel's `r0` is out of scope; the thread
            // index arrives as the `index` parameter.
            fmt("$t $v = ($t) index;\n", v, v, v);
        } else if (kind == VarKind::CallInput) {
            // Read the by-value parameter named after this input's index.
            uint32_t in_i = 0;
            for (; in_i < call->n_in; ++in_i)
                if (call->inner_in[in_i] == sv.index)
                    break;
            fmt("$t $v = a$u;\n", v, v, in_i);
        } else if (kind == VarKind::CallSelf) {
            fmt("u32 $v = self;\n", v);
        } else if (v->is_evaluated() ||
                   (vt == VarType::Pointer && kind == VarKind::Literal)) {
            // A captured field, read out of this instance's call-data block.
            //
            // Metal coalesces the packet-loadable prefix into uint4 loads.
            // Deliberately not done here: it is a bandwidth optimisation whose
            // payoff is unmeasured on CDNA2, and getting the word-extraction
            // arithmetic subtly wrong yields plausible garbage rather than a
            // failure. Plain typed loads first, coalescing when there is a
            // profile to justify it.
            uint32_t offset = jitc_call_slot_rel_offset(call, inst, v, sv.index);

            if (vt == VarType::Bool)
                fmt("bool $v = *(const u8 *)(data + $u) != 0;\n", v, offset);
            else if (vt == VarType::Pointer)
                fmt("const u8 *$v = *(const u8 *const *)(data + $u);\n",
                    v, offset);
            else
                fmt("$t $v = *(const $t *)(data + $u);\n", v, v, v, offset);
        } else {
            jitc_hip_render(v);
        }
    }

    // Pack the live outputs into the return struct and return it by value.
    if (jitc_hip_call_has_out(call)) {
        put("    ");
        jitc_hip_put_ret_type(call);
        put(" ret;\n");
        uint32_t k = 0;
        for (uint32_t i = 0; i < call->n_out; ++i) {
            if (call->out_offset[i] == (uint32_t) -1)
                continue;
            const Variable *v =
                jitc_var(call->inner_out[inst * call->n_out + i]);
            fmt("ret.r$u = $v;\n", k, v);
            k++;
        }
        put("return ret;\n");
    }

    put("}\n");
}

/// Getter: read one field straight out of the instance's call data, skipping
/// dispatch entirely. Mirrors the masked Gather, sourced from
/// `base + header_offset`.
void jitc_var_call_getter_assemble_hip(Variable *v, const Variable *index,
                                       const Variable *mask) {
    GetterData *gd = (GetterData *) v->data;
    uint32_t header_offset = gd->header_offset;

    // The kernel binds the call-data base to a register; inside a callable it
    // arrives as the `base` parameter.
    char base[32];
    if (callable_depth == 0)
        snprintf(base, sizeof(base), "r%u", call_buffer.base_reg);
    else
        snprintf(base, sizeof(base), "base");

    bool is_unmasked = mask->is_literal() && mask->literal == 1;
    if (is_unmasked)
        fmt("$t $v = ((const $t *) ($s + $u))[$v];\n",
            v, v, v, base, header_offset, index);
    else
        fmt("$t $v = ($v) ? ((const $t *) ($s + $u))[$v] : ($t) 0;\n",
            v, v, mask, v, base, header_offset, index, v);
}

/// Emit the dispatch site.
void jitc_var_call_assemble_hip(CallData *call, uint32_t call_reg,
                                uint32_t self_reg, uint32_t mask_reg) {
    Variable *mask = jitc_var(jitc_var(call->id)->dep[1]);
    bool is_masked = !mask->is_literal() || mask->literal != 1;
    bool has_out = jitc_hip_call_has_out(call);

    fmt("\n// VCall: $s\n", call->name.c_str());

    // The outputs the kernel actually consumes, paired with their field index
    // in the return struct (`r<k>`, k running over ALL active outputs so it
    // matches jitc_hip_emit_ret_struct).
    std::vector<std::pair<Variable *, uint32_t>> out_regs;
    out_regs.reserve(call->n_out);
    for (uint32_t i = 0, k = 0; i < call->n_out; ++i) {
        if (call->out_offset[i] == (uint32_t) -1)
            continue;
        Variable *v = jitc_var(call->outer_out[i]);
        if (v && v->reg_index && v->param_type != ParamType::Input)
            out_regs.emplace_back(v, k);
        k++;
    }

    // Declared before the guard, uninitialized: the call assigns them, and the
    // `else` branch zeroes them. Lanes that did not call still read these
    // further down the kernel, and an undefined register there is a wrong
    // pixel rather than a crash (spec_call.hip bit 5).
    for (auto [v, field] : out_regs)
        fmt("$t $v;\n", v, v);

    if (is_masked)
        fmt("if (r$u) {\n", mask_reg);
    else
        put("{\n");

    char base[32];
    if (callable_depth == 0)
        snprintf(base, sizeof(base), "r%u", call_buffer.base_reg);
    else
        snprintf(base, sizeof(base), "base");

    bool has_slots = !call->slots.empty();

    // The offset-table entry for this instance: (data_offset << 32) | index.
    // Only the high half is used -- see the note at the top of this section.
    if (has_slots) {
        fmt("u64 _oe_$u = ((const u64 *) $s)[$u + r$u];\n",
            call_reg, base,
            call->offset_base / (uint32_t) sizeof(uint64_t), self_reg);
        fmt("const u8 *_cd_$u = (const u8 *) $s + (u32) (_oe_$u >> 32);\n",
            call_reg, base, call_reg);
    }

    // Inside a callable the kernel-level `r0` is out of scope; the enclosing
    // callable received the thread index as `index`.
    const char *index_name = (callable_depth > 0) ? "index" : "r0";

    auto put_args = [&]() {
        fmt("$s, r$u, ", index_name, self_reg);
        if (has_slots)
            fmt("_cd_$u", call_reg);
        else
            put("(const u8 *) nullptr");
        if (call->use_nested)
            fmt(", $s", base);
        // Live inputs always have a register: in_active mirrors the packing
        // predicate, which excludes inputs without one.
        for (uint32_t i = 0; i < call->n_in; ++i)
            if (call->in_active[i])
                fmt(", r$u", jitc_var(call->outer_in[i])->reg_index);
    };

    if (has_out) {
        put("    ");
        jitc_hip_put_ret_type(call);
        fmt(" ret_$u;\n", call_reg);
    }

    if (call->n_inst == 1) {
        // A single target needs no dispatch at all: call it by name and let
        // the compiler inline it.
        if (has_out)
            fmt("    ret_$u = ", call_reg);
        else
            put("    ");
        put("func_");
        buffer.put_q64_unchecked(call->inst_hash[0].high64);
        buffer.put_q64_unchecked(call->inst_hash[0].low64);
        put("(");
        put_args();
        put(");\n");
    } else {
        // Group instances by callable, so instances sharing an implementation
        // share ONE call body and contribute only a case label each. Without
        // this the emitted code would scale with the instance count, which for
        // a Mitsuba scene is far larger than the number of BSDF types.
        std::vector<std::pair<XXH128_hash_t, std::vector<uint32_t>>> groups;
        for (uint32_t i = 0; i < call->n_inst; ++i) {
            XXH128_hash_t h = call->inst_hash[i];
            auto it = groups.end();
            for (auto g = groups.begin(); g != groups.end(); ++g) {
                if (g->first.low64 == h.low64 && g->first.high64 == h.high64) {
                    it = g;
                    break;
                }
            }
            if (it == groups.end()) {
                groups.emplace_back(h, std::vector<uint32_t>{ call->inst_id[i] });
            } else {
                it->second.push_back(call->inst_id[i]);
            }
        }

        fmt("    switch (r$u) {\n", self_reg);
        for (auto &[hash, ids] : groups) {
            for (uint32_t id : ids)
                fmt("        case $u:\n", id);
            put("            ");
            if (has_out)
                fmt("ret_$u = ", call_reg);
            put("func_");
            buffer.put_q64_unchecked(hash.high64);
            buffer.put_q64_unchecked(hash.low64);
            put("(");
            put_args();
            put(");\n            break;\n");
        }
        // Not dead code: `self` is data. An instance ID outside this call
        // site's set must produce something defined rather than falling
        // through to whatever the next statement happens to be.
        if (has_out) {
            put("        default:\n");
            uint32_t k = 0;
            for (uint32_t i = 0; i < call->n_out; ++i) {
                if (call->out_offset[i] == (uint32_t) -1)
                    continue;
                fmt("            ret_$u.r$u = ($t) 0;\n", call_reg, k,
                    jitc_var(call->inner_out[i]));
                k++;
            }
            put("            break;\n");
        } else {
            put("        default: break;\n");
        }
        put("    }\n");
    }

    for (auto [v, field] : out_regs)
        fmt("$v = ret_$u.r$u;\n", v, call_reg, field);

    if (is_masked && !out_regs.empty()) {
        put("} else {\n");
        for (auto [v, field] : out_regs)
            fmt("$v = ($t) 0;\n", v, v);
    }
    put("}\n\n");
}

// ---------------------------------------------------------------------------
//  Kernel assembly
// ---------------------------------------------------------------------------

void jitc_hip_assemble(ThreadState *ts, ScheduledGroup group,
                       uint32_t /*n_regs*/, uint32_t n_params) {
    // Preamble. Emitted inline rather than #included so the generated source is
    // self-contained -- it goes to hiprtc/NVRTC as a string with no include
    // path, and can be pasted into a file verbatim when debugging.
    //
    // Maps the short type spellings (forced by the 6-character NameTable limit,
    // see hip_eval.h) onto real C++ types.
    put(hip_type_preamble, strlen(hip_type_preamble));

    // put() only takes string literals or (ptr, len); the prologue is built at
    // run time, so pass its length explicitly.
    {
        std::string prologue = jitc_hip_kernel_prologue(n_params);
        put(prologue.c_str(), prologue.size());
    }

    // Bind the call-data base pointer once; every dispatch and getter in this
    // kernel indexes off it. The args index is the parameter index minus one
    // (args[] excludes the leading `size` field -- the `$o` convention).
    if (call_buffer.base_v)
        fmt("const u8 *r$u = (const u8 *) params.args[$u];\n",
            call_buffer.base_reg, call_buffer.base_param_index - 1);

    for (uint32_t gi = group.start; gi != group.end; ++gi) {
        uint32_t index = schedule[gi].index;
        Variable *v = jitc_var(index);
        ParamType ptype = (ParamType) v->param_type;

        if (likely(ptype == ParamType::Input)) {
            if (v->is_literal()) {
                // Pointer literals are LOADED rather than inlined so frozen
                // replay can rebind the address (BACKEND_NOTES §3).
                if ((VarType) v->type == VarType::Pointer)
                    fmt("$t $v = ($t) params.args[$o];\n", v, v, v, v);
                else
                    fmt("$t $v = $s;\n", v, v,
                        jitc_hip_literal((VarType) v->type, v->literal).c_str());
                continue;
            }

            if (!v->is_array()) {
                if (v->size > 1)
                    fmt("$t $v = ((const $t *) params.args[$o])[r0];\n",
                        v, v, v, v);
                else
                    fmt("$t $v = *(const $t *) params.args[$o];\n", v, v, v, v);
            } else {
                // The memcpy helpers reference this named `p<reg>` pointer.
                fmt("const $t *p$v = (const $t *) params.args[$o];\n",
                    v, v, v, v);
                jitc_hip_render_array_memcpy_in(v);
            }
            continue;
        }

        jitc_hip_render(v);

        if (ptype == ParamType::Output) {
            if (!v->is_array()) {
                fmt("(($t *) params.args[$o])[r0] = $v;\n", v, v, v);
            } else {
                fmt("$t *p$v = ($t *) params.args[$o];\n", v, v, v, v);
                jitc_hip_render_array_memcpy_out(v);
            }
        }
    }

    put("}\n");

    // -------------------------------------------------------------------
    //   Callables and other globals
    // -------------------------------------------------------------------
    //
    // Bodies go AFTER the kernel; forward declarations are then moved before
    // it, because C++ requires declare-before-use and the kernel calls them.
    // Metal does the same for the same reason (§9).
    //
    // Unlike Metal, EVERY callable needs a declaration here, not just
    // single-target ones: with switch dispatch the multi-target callables are
    // named at the call site too, rather than being reached anonymously
    // through a function table (§9a).
    if (!globals_map.empty()) {
        // Callable bodies, appended after the kernel.
        for (auto &it : globals_map) {
            if (it.first.type == GlobalType::Global)
                continue;
            put('\n');
            put(globals.get() + it.second.start, it.second.length);
            put('\n');
        }

        // Then the chunk that has to precede everything, built at the end and
        // relocated: struct definitions in full, plus one forward declaration
        // per callable.
        size_t suffix_start = buffer.size();

        // GlobalType::Global entries are full inline definitions -- the vcall
        // return structs among them. They are NOT forward-declarable: a
        // callable's signature names its return type by value, so the
        // definition must come first, not just a declaration.
        for (auto &it : globals_map) {
            if (it.first.type != GlobalType::Global)
                continue;
            put('\n');
            put(globals.get() + it.second.start, it.second.length);
            put('\n');
        }

        // Declarations for EVERY callable, not just single-target ones as
        // Metal does: switch dispatch names its targets, so multi-target
        // callables are referenced by name too (§9a). This also lets callables
        // call each other regardless of emission order.
        for (auto &it : globals_map) {
            if (it.first.type == GlobalType::Global)
                continue;
            const char *sig = globals.get() + it.second.start;
            const char *brace =
                (const char *) memchr(sig, '{', it.second.length);
            if (!brace)
                continue;
            size_t len = (size_t) (brace - sig);
            while (len > 0 && (sig[len - 1] == ' ' || sig[len - 1] == '\n'))
                len--;
            put(sig, len);
            put(";\n");
        }

        // Target: immediately after the type preamble, which every declaration
        // depends on and which nothing depends on.
        if (suffix_start != buffer.size())
            buffer.move_suffix(suffix_start, strlen(hip_type_preamble));
    }

    jitc_call_upload(ts);
}

#endif // DRJIT_ENABLE_HIP
