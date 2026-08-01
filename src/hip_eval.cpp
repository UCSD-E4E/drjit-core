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

        case VarKind::Min: render_fp_or_int(v, "fminf", "min"); break;
        case VarKind::Max: render_fp_or_int(v, "fmaxf", "max"); break;

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
