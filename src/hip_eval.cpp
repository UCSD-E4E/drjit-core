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

        case VarKind::Not:
            fmt("$t $v = ~$v;\n", v, v, jitc_var(v->dep[0]));
            break;

        case VarKind::Abs:
            render_call(v, jitc_is_float(v) ? "fabsf" : "abs", 1);
            break;

        case VarKind::Sqrt:
            render_call(v, "sqrtf", 1);
            break;

        // --- Binary ---------------------------------------------------------
        case VarKind::Add: render_binary(v, "+"); break;
        case VarKind::Sub: render_binary(v, "-"); break;
        case VarKind::Mul: render_binary(v, "*"); break;
        case VarKind::Div: render_binary(v, "/"); break;
        case VarKind::Mod: render_binary(v, "%"); break;
        case VarKind::And: render_binary(v, "&"); break;
        case VarKind::Or:  render_binary(v, "|"); break;
        case VarKind::Xor: render_binary(v, "^"); break;
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
            // Must be the fused form: a * b + c would silently differ in
            // precision from the CUDA backend (spec_arith.hip).
            render_call(v, jitc_is_float(v) ? "fmaf" : "__mad", 3);
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
        case VarKind::Popc: render_call(v, "__popc", 1); break;
        case VarKind::Clz:  render_call(v, "__clz", 1);  break;
        case VarKind::Brev: render_call(v, "__brev", 1); break;

        case VarKind::Ctz:
            // No __ctz exists on either platform; __ffs is 1-based.
            fmt("$t $v = ($t) (__ffs($v) - 1);\n", v, v, v,
                jitc_var(v->dep[0]));
            break;

        // --- Wide multiplication ----------------------------------------------
        //
        // These exist as opcodes precisely because `a * b` is computed modulo
        // the operand width, discarding what they need (spec_mulwide.hip).
        case VarKind::MulHi:
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
            bool wide = is_f64(v) || is_f64(a);
            // Reinterpret through the device intrinsics rather than a pointer
            // cast or __builtin_bit_cast -- the latter is Clang-only and NVRTC
            // rejects it (BACKEND_NOTES §11a).
            const char *fn;
            if (jitc_is_float(v))
                fn = wide ? "__longlong_as_double" : "__uint_as_float";
            else
                fn = wide ? "__double_as_longlong" : "__float_as_uint";
            render_call(v, fn, 1);
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

            if (op != ReduceOp::Identity)
                jitc_fail("jitc_hip_render(): scatter-reduce is not implemented "
                          "yet. The atomic forms and their return-value "
                          "conventions are specified in "
                          "tools/hip_validate/kernels/spec_memory.hip; note "
                          "they must target GLOBAL memory, never a "
                          "materialised temporary.");

            if (unmasked)
                fmt("(($t *) $v)[$v] = $v;\n", value, ptr, index, value);
            else
                fmt("if ($v) (($t *) $v)[$v] = $v;\n",
                    mask, value, ptr, index, value);
            break;
        }

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
