/*
    src/hip_eval.h -- HIP C++ type mapping for code generation

    ###########################################################################
    #  TYPE NAMES MUST BE 6 CHARACTERS OR FEWER.                              #
    ###########################################################################

    Not a style preference -- a hard constraint of the formatter. `w_type()` in
    strbuf.cpp copies a fixed 8-byte row and reads its LENGTH from byte 7:

        memcpy(&w, row, 8);            // exactly 8 bytes
        memcpy(d, &w, 8);
        return d + (uint8_t) row[7];   // byte 7 is the length tag

    So each entry is at most 7 characters plus that tag, and NameTable rejects
    anything over 6. An over-long name is not a compile error in a release
    build: byte 7 is read as a CHARACTER and the cursor jumps that many bytes
    past the end. "uint32_t" advanced it by 't' == 116 and silently truncated
    the kernel mid-statement.

    Hence the PTX-style short spellings below, which the kernel preamble
    typedefs back to real C++ types (see jitc_hip_assemble). That is also how
    the CUDA backend's table works -- "u32", "f32" -- rather than an accident of
    taste.

    Two deliberate differences from `type_name_metal` (src/var.cpp):

     1. Float64 maps to a REAL 64-bit type. Metal promotes Float64 -> Float32
        because Apple GPUs have no double-precision hardware; gfx90a has
        full-rate FP64, so inheriting that would quietly halve precision on the
        thing the card is best at. See BACKEND_NOTES.md §4.

     2. The binary view of Float64 is 64 bits wide. Metal's table maps it to a
        32-bit "uint" because Float64 cannot occur there; copying that would
        truncate every double bitcast.

    "???" marks the promotion-only sentinels (Void, BaseInt, BaseFloat), which
    never reach code generation.
*/

#pragma once

#include "var.h"
#include "hip_preamble.h"   // hip_type_preamble (kept dependency-free on purpose)

/// HIP type names. Short spellings, typedef'd in the kernel preamble.
inline constexpr NameTable<(size_t) VarType::Count> type_name_hip({
    /* Void    */ "void",
    /* Bool    */ "bool",
    /* BaseInt */ "???",
    /* Int8    */ "i8",
    /* UInt8   */ "u8",
    /* Int16   */ "i16",
    /* UInt16  */ "u16",
    /* Int32   */ "i32",
    /* UInt32  */ "u32",
    /* Int64   */ "i64",
    /* UInt64  */ "u64",
    /* Pointer */ "u64",
    /* BaseFlt */ "???",
    /* Float16 */ "f16",
    /* Float32 */ "f32",
    /* Float64 */ "f64"
});

/// Binary view of each type, for bitcasts and bit-wise operations.
///
/// Always an unsigned integer of the SAME WIDTH, so that e.g. a bitwise NOT on
/// a float does not round-trip through the FPU.
inline constexpr NameTable<(size_t) VarType::Count> type_name_hip_bin({
    /* Void    */ "???",
    /* Bool    */ "u8",
    /* BaseInt */ "???",
    /* Int8    */ "u8",
    /* UInt8   */ "u8",
    /* Int16   */ "u16",
    /* UInt16  */ "u16",
    /* Int32   */ "u32",
    /* UInt32  */ "u32",
    /* Int64   */ "u64",
    /* UInt64  */ "u64",
    /* Pointer */ "u64",
    /* BaseFlt */ "???",
    /* Float16 */ "u16",
    /* Float32 */ "u32",
    /* Float64 */ "u64"
});

// The preamble that maps these short spellings onto real types lives in
// hip_preamble.h, which the include above pulls in. It is kept in a separate,
// dependency-free header so the validation harness can compile it for gfx90a.
