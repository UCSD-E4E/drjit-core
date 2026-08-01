/*
    tests/hip_types.cpp -- HIP type-name mapping.

    Written before src/hip_eval.h exists. Every opcode emitter depends on this
    table, so it is the first thing worth pinning down.

    These are not tautologies restating the table. Each check encodes an
    invariant that a plausible mistake would violate -- above all the FP64 one,
    which is the single place the Metal template is actively wrong for us
    (tools/hip_validate/BACKEND_NOTES.md §4). Copying Metal's table carelessly
    would silently downgrade every double on a card whose headline feature is
    full-rate FP64.
*/

#include "../src/hip_eval.h"
#include <drjit-core/jit.h>
#include <cstdio>
#include <cstring>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-56s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

static void check_str(const char *got, const char *want, const char *what) {
    bool ok = got && strcmp(got, want) == 0;
    if (!ok)
        printf("  %-56s FAIL (got \"%s\", want \"%s\")\n", what,
               got ? got : "(null)", want);
    else
        printf("  %-56s ok\n", what);
    if (!ok)
        failures++;
}

int main(int, char **) {
    printf("hip_types: HIP type-name mapping\n");

    // --- The FP64 divergence: the whole point of this file ------------------
    //
    // gfx90a has full-rate FP64. Metal promotes Float64 -> Float32 because
    // Apple GPUs have no doubles at all; inheriting that here would quietly
    // halve precision on the MI210's strongest capability.
    check_str(type_name_hip[(int) VarType::Float64], "f64",
              "Float64 -> \"f64\" (a REAL 64-bit type, NOT promoted)");
    check(strcmp(type_name_hip[(int) VarType::Float64],
                 type_name_hip[(int) VarType::Float32]) != 0,
          "Float64 and Float32 map to different types");

    // --- Floating point -----------------------------------------------------
    check_str(type_name_hip[(int) VarType::Float32], "f32", "Float32 -> f32");
    check_str(type_name_hip[(int) VarType::Float16], "f16", "Float16 -> f16");

    // --- Integers. HIP is C++, so use the fixed-width types rather than
    //     Metal's short/long spellings, whose sizes are platform-defined.
    check_str(type_name_hip[(int) VarType::Int8],   "i8",  "Int8");
    check_str(type_name_hip[(int) VarType::UInt8],  "u8",  "UInt8");
    check_str(type_name_hip[(int) VarType::Int16],  "i16", "Int16");
    check_str(type_name_hip[(int) VarType::UInt16], "u16", "UInt16");
    check_str(type_name_hip[(int) VarType::Int32],  "i32", "Int32");
    check_str(type_name_hip[(int) VarType::UInt32], "u32", "UInt32");
    check_str(type_name_hip[(int) VarType::Int64],  "i64", "Int64");
    check_str(type_name_hip[(int) VarType::UInt64], "u64", "UInt64");
    check_str(type_name_hip[(int) VarType::Bool],   "bool",     "Bool");

    // A pointer must be an integer wide enough to hold a device address, not
    // a typed pointer -- emitted code casts it per use site.
    check_str(type_name_hip[(int) VarType::Pointer], "u64", "Pointer -> u64");

    // --- Binary view --------------------------------------------------------
    //
    // Used for bitcasts and bit-wise ops. Must be an unsigned integer of the
    // same width, so a bitwise NOT on a float does not go through the FPU.
    check_str(type_name_hip_bin[(int) VarType::Float32], "u32", "bin Float32 -> u32");
    check_str(type_name_hip_bin[(int) VarType::Float64], "u64", "bin Float64 -> u64 (64-bit!)");
    check_str(type_name_hip_bin[(int) VarType::Float16], "u16", "bin Float16 -> u16");
    check_str(type_name_hip_bin[(int) VarType::Bool],    "u8",  "bin Bool -> u8");

    // Regression guard: the binary view must never be narrower than the value
    // it aliases. Metal's table maps Float64 to "uint" (32-bit) because
    // Float64 cannot occur there -- copying that would truncate.
    check(strcmp(type_name_hip_bin[(int) VarType::Float64],
                 type_name_hip_bin[(int) VarType::Float32]) != 0,
          "bin Float64 is wider than bin Float32");

    // --- Completeness -------------------------------------------------------
    //
    // Every type Dr.Jit can actually instantiate must have a mapping. Void,
    // BaseInt and BaseFloat are promotion-only sentinels and never reach
    // codegen, so "???" is correct for those alone.
    const VarType must_map[] = {
        VarType::Bool,   VarType::Int8,    VarType::UInt8,  VarType::Int16,
        VarType::UInt16, VarType::Int32,   VarType::UInt32, VarType::Int64,
        VarType::UInt64, VarType::Pointer, VarType::Float16,
        VarType::Float32, VarType::Float64
    };
    bool complete = true;
    for (VarType vt : must_map)
        if (strcmp(type_name_hip[(int) vt], "???") == 0)
            complete = false;
    check(complete, "every instantiable VarType has a HIP mapping");

    // --- The 6-character limit ----------------------------------------------
    //
    // THE CHECK THAT WOULD HAVE SAVED AN AFTERNOON. w_type() in strbuf.cpp
    // copies a fixed 8-byte row and reads its length from BYTE 7, so a name
    // longer than 6 characters is not a compile error in a release build --
    // byte 7 is read as a character and the output cursor jumps that far past
    // the end. "uint32_t" advanced it by 't' == 116 and silently truncated the
    // generated kernel mid-statement, which surfaced as an NVRTC syntax error
    // about an unterminated function.
    //
    // NameTable throws on over-long entries in debug builds; this asserts it in
    // every build, and for the binary table too.
    {
        bool short_enough = true;
        for (int i = 0; i < (int) VarType::Count; ++i) {
            if (strlen(type_name_hip[i]) > 6)     short_enough = false;
            if (strlen(type_name_hip_bin[i]) > 6) short_enough = false;
        }
        check(short_enough,
              "every type name fits w_type()'s 6-char limit");
    }

    if (failures) {
        printf("hip_types: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_types: all checks passed.\n");
    return 0;
}
