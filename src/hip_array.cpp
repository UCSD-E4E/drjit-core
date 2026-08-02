/*
    src/hip_array.cpp -- per-lane variable arrays, HIP code generation.

    A close port of metal_array.cpp: both backends emit C-family source, so a
    Dr.Jit variable array is an ordinary local array `arr_<reg>[len]` in both,
    and the interesting logic (when a write needs a fresh buffer, how ArrayPhi
    aliases registers) is Dr.Jit's rather than the language's.

    Two things worth knowing about the emitted code, both checked on gfx90a and
    NVIDIA by tools/hip_validate/kernels/spec_array.hip:

      * A DYNAMIC index forces the array into scratch memory on AMDGPU -- it
        cannot stay in registers. Correctness is unaffected; occupancy is not.
        Constant indices are usually promoted. Both forms are emitted below.

      * memcpy in/out use a TRANSPOSED layout: element `i` of lane `r0` lives
        at `p[i * size + r0]`, not `p[r0 * len + i]`. Both expressions compile
        and stay in bounds; only one is coalesced, and only one agrees with the
        host-side layout.
*/

#include "eval.h"
#include "hip_array.h"
#include "array.h"
#include "log.h"
#include "var.h"

#include "hip_eval.h"

// MUST BE LAST -- redefines `fmt` and `put`. See the note in hip_eval.cpp.
#include "hip_codegen.h"

#if defined(DRJIT_ENABLE_HIP)

void jitc_hip_render_array(Variable *v, Variable *pred) {
    // A non-conflicted predecessor is reused in place: the array is the same
    // storage, so the new node just adopts its register name.
    if (pred && pred->array_state != (uint32_t) ArrayState::Conflicted) {
        v->reg_index = pred->reg_index;
        return;
    }
    fmt("$t arr_$u[$u];\n", v, v->reg_index, (uint32_t) v->array_length);
}

void jitc_hip_render_array_init(Variable *v, Variable *pred, Variable *value) {
    v->reg_index = pred->reg_index;

    fmt("for (u32 _i = 0; _i < $uu; _i++)\n"
        "    arr_$u[_i] = $v;\n",
        (uint32_t) v->array_length, v->reg_index, value);
}

void jitc_hip_render_array_read(Variable *v, Variable *source, Variable *mask,
                                Variable *offset) {
    // A masked read pre-declares its destination and guards the load, rather
    // than loading and selecting: the index may be out of range on lanes the
    // mask excludes.
    if (!mask->is_literal())
        fmt("$t $v = ($t) 0;\n", v, v, v);

    if (offset) {
        if (!mask->is_literal())
            fmt("if ($v) $v = arr_$u[$v];\n", mask, v, source->reg_index, offset);
        else
            fmt("$t $v = arr_$u[$v];\n", v, v, source->reg_index, offset);
    } else {
        if (!mask->is_literal())
            fmt("if ($v) $v = arr_$u[$u];\n", mask, v, source->reg_index,
                (uint32_t) v->literal);
        else
            fmt("$t $v = arr_$u[$u];\n", v, v, source->reg_index,
                (uint32_t) v->literal);
    }
}

void jitc_hip_render_array_write(Variable *v, Variable *target, Variable *value,
                                 Variable *mask, Variable *offset) {
    if (offset && offset->is_array())
        offset = nullptr;

    // A conflicted target is still needed by another node, so the write goes to
    // a fresh buffer seeded with a copy.
    bool copy = target->array_state == (uint32_t) ArrayState::Conflicted;
    uint32_t target_buffer = target->reg_index;

    if (copy) {
        target_buffer = jitc_array_buffer(v)->reg_index;

        fmt("for (u32 _i = 0; _i < $uu; _i++)\n"
            "    arr_$u[_i] = arr_$u[_i];\n",
            (uint32_t) v->array_length, target_buffer, target->reg_index);
    }

    if (offset) {
        if (!mask->is_literal())
            fmt("if ($v) arr_$u[$v] = $v;\n", mask, target_buffer, offset, value);
        else
            fmt("arr_$u[$v] = $v;\n", target_buffer, offset, value);
    } else {
        if (!mask->is_literal())
            fmt("if ($v) arr_$u[$u] = $v;\n", mask, target_buffer,
                (uint32_t) v->literal, value);
        else
            fmt("arr_$u[$u] = $v;\n", target_buffer, (uint32_t) v->literal, value);
    }

    v->reg_index = target_buffer;
}

void jitc_hip_render_array_memcpy_in(const Variable *v) {
    fmt("$t arr_$u[$u];\n", v, v->reg_index, (uint32_t) v->array_length);
    fmt("for (u32 _i = 0; _i < $uu; _i++)\n"
        "    arr_$u[_i] = ((const $t *) p$v)[_i * params.size + r0];\n",
        (uint32_t) v->array_length, v->reg_index, v, v);
}

void jitc_hip_render_array_memcpy_out(const Variable *v) {
    fmt("for (u32 _i = 0; _i < $uu; _i++)\n"
        "    (($t *) p$v)[_i * params.size + r0] = arr_$u[_i];\n",
        (uint32_t) v->array_length, v, v, v->reg_index);
}

void jitc_hip_render_array_select(Variable *v, Variable *mask, Variable *t,
                                  Variable *f) {
    // A whole-array select: the operands are arrays, so this copies one side or
    // the other element-wise. Not a per-element ternary.
    uint32_t reg_index = jitc_array_buffer(v)->reg_index;
    fmt("if ($v) {\n"
        "    for (u32 _i = 0; _i < $uu; _i++)\n"
        "        arr_$u[_i] = arr_$u[_i];\n"
        "} else {\n"
        "    for (u32 _i = 0; _i < $uu; _i++)\n"
        "        arr_$u[_i] = arr_$u[_i];\n"
        "}\n",
        mask,
        (uint32_t) f->array_length, reg_index, t->reg_index,
        (uint32_t) f->array_length, reg_index, f->reg_index);

    v->reg_index = reg_index;
}

#endif // DRJIT_ENABLE_HIP
