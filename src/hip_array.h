/*
    src/hip_array.h -- per-lane variable arrays, HIP code generation.

    Mirrors metal_array.h.
*/

#pragma once

#include "common.h"

struct Variable;

extern void jitc_hip_render_array(Variable *v, Variable *pred);
extern void jitc_hip_render_array_init(Variable *v, Variable *pred, Variable *value);
extern void jitc_hip_render_array_read(Variable *v, Variable *source, Variable *mask, Variable *offset);
extern void jitc_hip_render_array_write(Variable *v, Variable *target, Variable *value, Variable *mask, Variable *offset);
extern void jitc_hip_render_array_select(Variable *v, Variable *mask, Variable *t, Variable *f);
extern void jitc_hip_render_array_memcpy_in(const Variable *v);
extern void jitc_hip_render_array_memcpy_out(const Variable *v);
