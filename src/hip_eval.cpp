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
#include "call.h"        // jitc_call_upload
#include "loop.h"        // LoopData
#include "cond.h"        // CondData, jitc_cond_output_active
#include <cstring>
#include "hip.h"
#include "hip_eval.h"
#include "hip_format.h"
#include "hip_literal.h"
#include "hip_prologue.h"

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

/// Pick between the float and integer spelling of an operation.
static void render_fp_or_int(Variable *v, const char *fp_fn, const char *int_fn) {
    render_call(v, jitc_is_float(v) ? fp_fn : int_fn, 2);
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
static void render_scatter_reduce(ReduceOp op, Variable *ptr, Variable *value,
                                  Variable *index) {
    VarType vt = (VarType) value->type;
    uint32_t width = type_size[(int) vt];
    bool is_float = jitc_is_float(value),
         is_int   = !is_float && (vt != VarType::Bool);

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
            fmt("    $s(($b *) (($t *) $v + $v), ($b) $v);\n",
                fn, value, value, ptr, index, value, value);
        else
            fmt("    $s(($t *) $v + $v, $v);\n",
                fn, value, ptr, index, value);
        return;
    }

    // --- Compare-and-swap loop ----------------------------------------------
    //
    // The CAS width follows the VALUE width ($b is the same-width unsigned
    // integer): a 32-bit CAS on a double would spin on half of it forever.
    fmt("    $b *_a = ($b *) (($t *) $v + $v);\n"
        "    $b _o = *_a, _s;\n"
        "    do {\n"
        "        _s = _o;\n",
        value, value, value, ptr, index,
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
            // Only ONE side can be floating point: the two types have the same
            // width but differ, so f32/f64 never face each other. When NEITHER
            // is, there is nothing to reinterpret -- a plain cast between two
            // same-width integers already yields the bit pattern. Reaching for
            // a float intrinsic there converts the value first, turning
            // Int64(6) into 0x40c00000 (spec_cast.hip bit 14).
            const char *fn = jitc_is_float(a) ? to_bits_fn(a) : from_bits_fn(v);
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

        default:
            jitc_fail("jitc_hip_render(): unhandled variable kind \"%s\"! The "
                      "source this opcode must produce is specified in "
                      "tools/hip_validate/kernels/spec_*.hip.",
                      var_kind_name[(uint32_t) v->kind]);
    }
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

            if (v->size > 1)
                fmt("$t $v = ((const $t *) params.args[$o])[r0];\n", v, v, v, v);
            else
                fmt("$t $v = *(const $t *) params.args[$o];\n", v, v, v, v);
            continue;
        }

        jitc_hip_render(v);

        if (ptype == ParamType::Output)
            fmt("(($t *) params.args[$o])[r0] = $v;\n", v, v, v);
    }

    put("}\n");

    jitc_call_upload(ts);
}

#endif // DRJIT_ENABLE_HIP
