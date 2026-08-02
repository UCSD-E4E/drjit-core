/*
    src/hip_preamble.h -- text emitted at the top of every generated HIP kernel.

    Split out of hip_eval.h so that it has NO dependency on the JIT internals:
    tools/hip_validate/gen_prologue.cpp includes this file directly and compiles
    the result for real gfx90a. A preamble that only ever existed inside a
    string constant would otherwise be the one part of every kernel that nothing
    checks.

    Two jobs:

     1. Map the short type spellings (see the 6-character constraint documented
        in hip_eval.h) onto real C++ types.

     2. Establish the fp16 contract that jitc_hip_literal() already emits calls
        to. That contract is compiler-dependent, which is why it is written out
        here rather than assumed.
*/

#pragma once

/// Preamble emitted at the top of every generated kernel.
///
/// NVRTC ships no <stdint.h>, so the fixed-width types are declared here rather
/// than included -- mirroring tools/hip_validate/prelude_nvrtc.h.
inline constexpr const char *hip_type_preamble =
    "typedef signed char        i8;\n"
    "typedef unsigned char      u8;\n"
    "typedef short              i16;\n"
    "typedef unsigned short     u16;\n"
    "typedef int                i32;\n"
    "typedef unsigned int       u32;\n"
    "typedef long long          i64;\n"
    "typedef unsigned long long u64;\n"
    "typedef float              f32;\n"
    "typedef double             f64;\n"
    "\n"
    // --- Half precision -----------------------------------------------------
    //
    // gfx90a has native fp16, and on the AMD arm `_Float16` is a builtin Clang
    // type needing no header -- which matters because hiprtc has no dependable
    // include path.
    //
    // NVRTC rejects `_Float16` outright, so the CUDA shim (PLAN.md §3.5) uses
    // CUDA's own `__half`. Its header is reachable only because
    // jitc_hip_compile() passes an explicit -I; when that is unavailable the
    // last branch aliases f16 to `float`, which COMPILES BUT DOES NOT RUN
    // CORRECTLY -- it is 4 bytes wide, so loads from a half buffer land on the
    // wrong element. That branch is a compile-check fallback, nothing more.
    //
    // fp16 numerics on the shim are CUDA's, not gfx90a's: rounding and
    // denormal behaviour still belong on the "unverified on NVIDIA" list
    // alongside wave64 (BACKEND_NOTES §4).
    "#if defined(__HIP__) && !defined(__CUDACC_RTC__)\n"
    "typedef _Float16 f16;\n"
    "#elif defined(DRJIT_SHIM_HAVE_FP16)\n"
    "#include <cuda_fp16.h>\n"
    "typedef __half f16;\n"
    "#else\n"
    // No usable half type. Deliberately a 2-byte struct rather than `float`:
    // the width is what memory layout depends on, so buffers still index
    // correctly, and any ARITHMETIC on it is a compile error naming this line
    // instead of quietly producing wrong numbers.
    "struct f16 { u16 x; };  /* fp16 unavailable -- "
    "set DRJIT_HIP_SHIM_CUDA_INCLUDE */\n"
    "#endif\n"
    "\n"
    // jitc_hip_literal() emits half constants as raw bit patterns rather than
    // decimals, so that NaN payloads and subnormals survive (asserted by
    // tests/hip_literal.cpp). This is the other half of that contract.
    //
    // The union is deliberate: __builtin_bit_cast is Clang-only and NVRTC
    // rejects it, and __ushort_as_half needs a header neither runtime compiler
    // can be relied on to find (BACKEND_NOTES §11a).
    "__device__ inline f16 drjit_half_from_bits(u32 h) {\n"
    "#if defined(__HIP__) && !defined(__CUDACC_RTC__)\n"
    "    union { u16 u; f16 h; } c; c.u = (u16) h; return c.h;\n"
    "#elif defined(DRJIT_SHIM_HAVE_FP16)\n"
    "    return __ushort_as_half((u16) h);\n"
    "#else\n"
    "    f16 r; r.x = (u16) h; return r;\n"
    "#endif\n"
    "}\n"
    "#define DRJIT_HALF_FROM_BITS(bits) drjit_half_from_bits((u32) (bits))\n"
    "\n"
    // The inverse. Bit-wise ops on f16 (Not/And/Or/Xor) route through this:
    // `~h` does not compile for any of the three spellings above, and going via
    // float would round before the bits were even examined.
    "__device__ inline u16 drjit_half_to_bits(f16 h) {\n"
    "#if defined(__HIP__) && !defined(__CUDACC_RTC__)\n"
    "    union { f16 h; u16 u; } c; c.h = h; return c.u;\n"
    "#elif defined(DRJIT_SHIM_HAVE_FP16)\n"
    "    return __half_as_ushort(h);\n"
    "#else\n"
    "    return h.x;\n"
    "#endif\n"
    "}\n"
    "\n";
