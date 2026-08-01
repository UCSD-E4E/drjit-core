/*
    tests/hip_literal.cpp -- materialising constants into HIP source.

    Written before jitc_hip_literal() exists.

    Dr.Jit stores every literal as a raw 64-bit pattern. Turning that back into
    source is where a backend most easily loses bits, and the failure is silent:
    a float printed as decimal and re-parsed may not round-trip, and a wide
    constant written without a suffix gets promoted or truncated by ordinary C++
    integer rules. Nothing downstream notices; the render is just slightly wrong.

    So the invariant this file defends is: FLOATS ARE REINTERPRETED, NEVER
    PRINTED, and integer literals carry a suffix wide enough to hold them.

    FP64 has no Metal precedent to copy -- Metal cannot represent Float64 at all
    (BACKEND_NOTES §4) -- so the 64-bit path here is new code, and the place a
    32-bit assumption would hide.
*/

#include "../src/hip_literal.h"
#include <drjit-core/jit.h>
#include <cstdio>
#include <cstring>
#include <string>

static int failures = 0;

static void expect(VarType vt, uint64_t value, const char *want,
                   const char *what) {
    std::string got = jitc_hip_literal(vt, value);
    bool ok = got == want;
    printf("  %-44s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok)
        printf("      got  \"%s\"\n      want \"%s\"\n", got.c_str(), want);
    if (!ok)
        failures++;
}

static void check(bool cond, const char *what) {
    printf("  %-44s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

int main(int, char **) {
    printf("hip_literal: constant materialisation\n");

    // --- Floats are bit-reinterpreted, never printed -------------------------
    //
    // 0x3f800000 is 1.0f. Emitting "1.0f" would happen to work here and fail
    // for values that do not round-trip through decimal, so the form itself is
    // what gets asserted.
    expect(VarType::Float32, 0x3f800000ull, "__uint_as_float(0x3f800000u)",
           "Float32 via __uint_as_float");

    // 0x3ff0000000000000 is 1.0. This path has NO Metal equivalent.
    expect(VarType::Float64, 0x3ff0000000000000ull,
           "__longlong_as_double(0x3ff0000000000000ull)",
           "Float64 via __longlong_as_double");

    // Half goes through the prelude, like the type itself (DRJIT_HALF).
    expect(VarType::Float16, 0x3c00ull, "DRJIT_HALF_FROM_BITS(0x3c00u)",
           "Float16 via prelude macro");

    // A NaN payload must survive verbatim. Printing and re-parsing is exactly
    // where payload bits get normalised away.
    expect(VarType::Float32, 0x7fc00001ull, "__uint_as_float(0x7fc00001u)",
           "Float32 NaN payload preserved");

    // --- Integers carry a wide-enough suffix --------------------------------
    expect(VarType::UInt32, 4294967295ull, "0xffffffffu", "UInt32 max, u suffix");
    expect(VarType::UInt64, 18446744073709551615ull, "0xffffffffffffffffull",
           "UInt64 max, ull suffix (no truncation)");
    expect(VarType::Int32, (uint64_t) (int64_t) -1, "(int32_t) 0xffffffffu",
           "Int32 -1 via unsigned pattern + cast");

    // Pointers are integers here; the emitter casts per use site.
    expect(VarType::Pointer, 0x7f0011223344ull, "0x7f0011223344ull",
           "Pointer as uint64 literal");

    // --- Bool ---------------------------------------------------------------
    expect(VarType::Bool, 0ull, "false", "Bool false");
    expect(VarType::Bool, 1ull, "true",  "Bool true");

    // --- Structural invariants ----------------------------------------------
    //
    // These catch whole classes of mistake rather than single values.

    // No float literal may be emitted as a decimal number: if the result
    // contains '.', someone printed it.
    {
        bool clean = true;
        const uint64_t pats[] = { 0x3f800000ull, 0x7fc00001ull, 0x00000001ull };
        for (uint64_t p : pats)
            if (jitc_hip_literal(VarType::Float32, p).find('.') != std::string::npos)
                clean = false;
        for (uint64_t p : { 0x3ff0000000000000ull, 0x0000000000000001ull })
            if (jitc_hip_literal(VarType::Float64, p).find('.') != std::string::npos)
                clean = false;
        check(clean, "no float literal is printed as decimal");
    }

    // A 64-bit value must never be emitted through a 32-bit form. Guard by
    // checking the high word survives.
    {
        std::string s = jitc_hip_literal(VarType::UInt64, 0xdeadbeef00000000ull);
        check(s.find("deadbeef") != std::string::npos,
              "64-bit high word survives (no 32-bit truncation)");
    }
    {
        std::string s = jitc_hip_literal(VarType::Float64, 0xdeadbeef00000000ull);
        check(s.find("deadbeef") != std::string::npos,
              "Float64 high word survives");
    }

    // Every unsigned/pointer form needs a suffix, or C++ may pick a signed type
    // and change the value.
    {
        bool suffixed =
            jitc_hip_literal(VarType::UInt32,  0x80000000ull).find('u') != std::string::npos &&
            jitc_hip_literal(VarType::UInt64,  0x8000000000000000ull).find("ull") != std::string::npos &&
            jitc_hip_literal(VarType::Pointer, 0x8000000000000000ull).find("ull") != std::string::npos;
        check(suffixed, "unsigned literals carry a suffix");
    }

    if (failures) {
        printf("hip_literal: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_literal: all checks passed.\n");
    return 0;
}
