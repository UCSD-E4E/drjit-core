/*
    src/hip_eval.h -- HIP C++ type mapping for code generation

    Note this lives in a header as `inline constexpr` rather than alongside the
    other backends' NameTable entries in var.cpp. That is deliberate: it keeps
    the tables reachable from tests/hip_types.cpp without exporting internal
    symbols from a library built with -fvisibility=hidden. A pure lookup table
    is a good thing to be able to assert on directly, and this costs nothing at
    run time.
*/

#pragma once

#include <drjit-core/jit.h>

/// HIP C++ type names, indexed by VarType.
///
/// Two deliberate differences from `type_name_metal` (src/var.cpp):
///
///  1. Float64 maps to `double`, NOT `float`. Metal promotes Float64 ->
///     Float32 because Apple GPUs have no double-precision hardware at all.
///     gfx90a has FULL-RATE FP64 -- one of the MI210's headline capabilities --
///     so inheriting that promotion would quietly halve precision on the thing
///     the card is best at. See tools/hip_validate/BACKEND_NOTES.md §4.
///
///  2. Float16 maps to the prelude macro DRJIT_HALF rather than a concrete
///     type name. `__half` is not a builtin on either platform -- it is declared
///     by a header (hip_fp16.h on AMD) -- so emitting it directly fails to
///     compile. This was caught by tools/hip_validate rather than reasoned
///     about, which is exactly what that harness is for.
///
///  3. Fixed-width `int32_t`-style spellings rather than Metal's `int`/`long`.
///     HIP is ordinary C++, where the width of `long` is platform-defined;
///     emitted code must not depend on host/device agreeing about it.
///
/// "???" marks the promotion-only sentinels (Void, BaseInt, BaseFloat), which
/// never reach code generation.
inline constexpr const char *type_name_hip[(int) VarType::Count] = {
    /* Void     */ "void",
    /* Bool     */ "bool",
    /* BaseInt  */ "???",
    /* Int8     */ "int8_t",
    /* UInt8    */ "uint8_t",
    /* Int16    */ "int16_t",
    /* UInt16   */ "uint16_t",
    /* Int32    */ "int32_t",
    /* UInt32   */ "uint32_t",
    /* Int64    */ "int64_t",
    /* UInt64   */ "uint64_t",
    /* Pointer  */ "uint64_t",
    /* BaseFloat*/ "???",
    /* Float16  */ "DRJIT_HALF",
    /* Float32  */ "float",
    /* Float64  */ "double"
};

/// Binary view of each type, for bitcasts and bit-wise operations.
///
/// Always an *unsigned integer of the same width*, so that e.g. a bitwise NOT
/// on a float does not round-trip through the FPU. Note Float64 -> uint64_t:
/// Metal's equivalent table maps it to a 32-bit `uint` because Float64 cannot
/// occur there, and copying that would silently truncate.
inline constexpr const char *type_name_hip_bin[(int) VarType::Count] = {
    /* Void     */ "???",
    /* Bool     */ "uint8_t",
    /* BaseInt  */ "???",
    /* Int8     */ "uint8_t",
    /* UInt8    */ "uint8_t",
    /* Int16    */ "uint16_t",
    /* UInt16   */ "uint16_t",
    /* Int32    */ "uint32_t",
    /* UInt32   */ "uint32_t",
    /* Int64    */ "uint64_t",
    /* UInt64   */ "uint64_t",
    /* Pointer  */ "uint64_t",
    /* BaseFloat*/ "???",
    /* Float16  */ "uint16_t",
    /* Float32  */ "uint32_t",
    /* Float64  */ "uint64_t"
};
