/*
    src/hip_prologue.h -- kernel prologue for generated HIP source

    The first text jitc_hip_assemble() emits: the parameter struct, the entry
    point, and the thread-index setup. Pure and header-only so it is testable
    without JIT state or a device (BACKEND_NOTES §11).

    Modelled on the opening of jitc_metal_assemble(), with three divergences
    that all follow from HIP being ordinary C++ on a block-scheduled GPU rather
    than MSL on Metal's dispatch model. Each is asserted in
    tests/hip_prologue.cpp.
*/

#pragma once

#include <string>
#include <cstdint>
#include <cstdio>

/// Number of '^' characters reserved for the kernel-name hash.
///
/// Not arbitrary: eval.cpp finds "drjit_^" with strstr(), rewinds to
/// marker + 6, and writes two q64 values -- 2 x 16 hex digits. Too few and the
/// hash overruns into the signature; too many and stale carets survive into
/// the kernel name. The needle must also appear EXACTLY ONCE in the emitted
/// source, since strstr() takes the first match.
inline constexpr int JITC_HIP_HASH_CARETS = 32;

/// Emit the kernel prologue for a launch taking `n_params` kernel parameters.
///
/// Leaves the kernel body open; the caller emits the variable loop and the
/// closing brace.
inline std::string jitc_hip_kernel_prologue(uint32_t n_params) {
    // args[] excludes the leading `size` field, which is why the `$o` format
    // code indexes it by parameter index MINUS ONE. Clamp to 1 because a
    // zero-length array is ill-formed in C++ -- Metal clamps for the same
    // reason.
    //
    // (Metal additionally widens this to n_params when a visible function
    // table is bound. HIP has no equivalent until callables land in Phase 4;
    // revisit here at that point.)
    uint32_t n_args = n_params > 1 ? n_params - 1 : 1;

    std::string s;

    // Divergence 1: HIP takes the parameter block BY VALUE. Metal binds it as
    // `constant Params&` at buffer(0) because MSL has no by-value struct
    // parameters; hipModuleLaunchKernel accepts the struct directly, so no
    // buffer binding and no address space qualifier are needed.
    char buf[128];
    snprintf(buf, sizeof(buf),
             "struct Params {\n"
             "    uint32_t size;\n"
             "    void *args[%u];\n"
             "};\n\n",
             n_args);
    s += buf;

    // Divergence 2: `extern \"C\"` so the symbol name survives into the code
    // object unmangled and can be looked up by hipModuleGetFunction().
    s += "extern \"C\" __global__ void drjit_";
    s.append((size_t) JITC_HIP_HASH_CARETS, '^');
    s += "(Params params) {\n";

    // Divergence 3: THE BOUNDS CHECK.
    //
    // Metal dispatches thread_position_in_grid over an exactly-sized grid and
    // needs no guard. HIP launches whole blocks, so the final block runs
    // threads past the end of every buffer. Without this, those threads write
    // out of bounds -- corrupting neighbouring allocations rather than
    // faulting, which is the hardest failure mode to trace back here.
    s += "    uint32_t r0 = blockIdx.x * blockDim.x + threadIdx.x;\n"
         "    if (r0 >= params.size)\n"
         "        return;\n";

    return s;
}
