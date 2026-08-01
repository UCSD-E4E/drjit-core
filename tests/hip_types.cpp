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
    check_str(type_name_hip[(int) VarType::Float64], "double",
              "Float64 -> \"double\"  (NOT \"float\", cf. Metal)");
    check(strcmp(type_name_hip[(int) VarType::Float64],
                 type_name_hip[(int) VarType::Float32]) != 0,
          "Float64 and Float32 map to different types");

    // --- Floating point -----------------------------------------------------
    check_str(type_name_hip[(int) VarType::Float32], "float",  "Float32 -> \"float\"");
    // Not a concrete type: __half is header-declared on both platforms, so the
    // emitter must go through the prelude. Proven by tools/hip_validate.
    check_str(type_name_hip[(int) VarType::Float16], "DRJIT_HALF",
              "Float16 -> DRJIT_HALF (prelude macro, not a builtin)");

    // --- Integers. HIP is C++, so use the fixed-width types rather than
    //     Metal's short/long spellings, whose sizes are platform-defined.
    check_str(type_name_hip[(int) VarType::Int8],   "int8_t",   "Int8");
    check_str(type_name_hip[(int) VarType::UInt8],  "uint8_t",  "UInt8");
    check_str(type_name_hip[(int) VarType::Int16],  "int16_t",  "Int16");
    check_str(type_name_hip[(int) VarType::UInt16], "uint16_t", "UInt16");
    check_str(type_name_hip[(int) VarType::Int32],  "int32_t",  "Int32");
    check_str(type_name_hip[(int) VarType::UInt32], "uint32_t", "UInt32");
    check_str(type_name_hip[(int) VarType::Int64],  "int64_t",  "Int64");
    check_str(type_name_hip[(int) VarType::UInt64], "uint64_t", "UInt64");
    check_str(type_name_hip[(int) VarType::Bool],   "bool",     "Bool");

    // A pointer must be an integer wide enough to hold a device address, not
    // a typed pointer -- emitted code casts it per use site.
    check_str(type_name_hip[(int) VarType::Pointer], "uint64_t", "Pointer -> uint64_t");

    // --- Binary view --------------------------------------------------------
    //
    // Used for bitcasts and bit-wise ops. Must be an unsigned integer of the
    // same width, so a bitwise NOT on a float does not go through the FPU.
    check_str(type_name_hip_bin[(int) VarType::Float32], "uint32_t", "bin Float32 -> uint32_t");
    check_str(type_name_hip_bin[(int) VarType::Float64], "uint64_t", "bin Float64 -> uint64_t (64-bit!)");
    check_str(type_name_hip_bin[(int) VarType::Float16], "uint16_t", "bin Float16 -> uint16_t");
    check_str(type_name_hip_bin[(int) VarType::Bool],    "uint8_t",  "bin Bool -> uint8_t");

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

    if (failures) {
        printf("hip_types: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_types: all checks passed.\n");
    return 0;
}
