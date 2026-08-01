/*
    src/hip_codegen.h -- HIP code generation macros.

    Mirrors metal_eval.h. Kept separate from hip_eval.h (the type tables)
    because these macros redefine `fmt` and `put` for the whole translation
    unit, so only hip_eval.cpp may include this -- pulling it into a header
    that other code includes would hijack their formatting.
*/

#pragma once

#include "common.h"

#define fmt(fmt, ...)                                                          \
    buffer.fmt_hip(count_args(__VA_ARGS__), fmt_strlen(fmt), fmt,              \
                   ##__VA_ARGS__)
#define put(...) buffer.put(__VA_ARGS__)
#define fmt_intrinsic(fmt, ...)                                                \
    do {                                                                       \
        size_t tmpoff = buffer.size();                                         \
        buffer.fmt_hip(count_args(__VA_ARGS__), fmt_strlen(fmt), fmt,          \
                       ##__VA_ARGS__);                                         \
        jitc_register_global(buffer.get() + tmpoff);                           \
        buffer.rewind_to(tmpoff);                                              \
    } while (0)
