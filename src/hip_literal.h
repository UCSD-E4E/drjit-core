/*
    src/hip_literal.h -- materialise a Dr.Jit literal into HIP source

    Dr.Jit stores every constant as a raw 64-bit pattern. Reconstituting it in
    source is where a backend most easily loses bits, silently:

      * a float written as decimal may not round-trip, and NaN payloads get
        normalised away;
      * an integer written without a suffix is subject to ordinary C++
        promotion rules, so a large unsigned constant can become negative or
        be truncated.

    Hence the rule this file implements: FLOATS ARE REINTERPRETED FROM THEIR
    BIT PATTERN, NEVER PRINTED, and integers always carry a suffix wide enough
    to hold them.

    Pure and header-only so tests/hip_literal.cpp can exercise it without JIT
    state or a device (BACKEND_NOTES §11).
*/

#pragma once

#include <drjit-core/jit.h>
#include <string>
#include <cstdio>
#include <cstdint>

/// Format `value` as a hexadecimal literal with the given suffix.
inline std::string jitc_hip_hex(uint64_t value, const char *suffix) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%llx%s", (unsigned long long) value, suffix);
    return buf;
}

/// Render the literal `value` of type `vt` as HIP C++ source.
///
/// `value` is the raw bit pattern, as stored in Variable::literal.
inline std::string jitc_hip_literal(VarType vt, uint64_t value) {
    switch (vt) {
        case VarType::Bool:
            return value ? "true" : "false";

        // --- Floating point: reinterpret, never print ----------------------
        //
        // __uint_as_float / __longlong_as_double are device builtins on both
        // HIP and CUDA, so these forms are portable across both validation
        // arms without needing a prelude macro.
        case VarType::Float32:
            return "__uint_as_float(" +
                   jitc_hip_hex(value & 0xffffffffull, "u") + ")";

        // No Metal equivalent exists -- Metal cannot represent Float64 at all,
        // so this path is new code rather than a port. gfx90a has full-rate
        // FP64 and the full 64-bit pattern must survive.
        case VarType::Float64:
            return "__longlong_as_double(" + jitc_hip_hex(value, "ull") + ")";

        // Half has no portable builtin: the type itself is header-declared
        // (see type_name_hip), so the bit-reinterpretation goes through the
        // prelude alongside it.
        case VarType::Float16:
            return "DRJIT_HALF_FROM_BITS(" +
                   jitc_hip_hex(value & 0xffffull, "u") + ")";

        // --- Signed integers -------------------------------------------------
        //
        // Emitted as the unsigned bit pattern plus an explicit cast. Writing a
        // negative decimal would be equivalent for the value but not for the
        // bit pattern of the minimum of each type, where the negation of the
        // literal overflows before the cast applies.
        //
        // The cast names are the SHORT spellings from type_name_hip, not the
        // stdint ones: the generated kernel preamble (hip_preamble.h) declares
        // i8/i16/i32/i64 because neither runtime compiler can reach <stdint.h>.
        case VarType::Int8:
            return "(i8) " + jitc_hip_hex(value & 0xffull, "u");
        case VarType::Int16:
            return "(i16) " + jitc_hip_hex(value & 0xffffull, "u");
        case VarType::Int32:
            return "(i32) " + jitc_hip_hex(value & 0xffffffffull, "u");
        case VarType::Int64:
            return "(i64) " + jitc_hip_hex(value, "ull");

        // --- Unsigned integers ------------------------------------------------
        case VarType::UInt8:
            return jitc_hip_hex(value & 0xffull, "u");
        case VarType::UInt16:
            return jitc_hip_hex(value & 0xffffull, "u");
        case VarType::UInt32:
            return jitc_hip_hex(value & 0xffffffffull, "u");
        case VarType::UInt64:
        case VarType::Pointer:
            return jitc_hip_hex(value, "ull");

        default:
            // Void / BaseInt / BaseFloat are promotion-only sentinels and must
            // never reach code generation.
            return "???";
    }
}
