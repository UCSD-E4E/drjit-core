/*
    tests/hip_prologue.cpp -- the kernel prologue: Params struct + entry point.

    Written before src/hip_prologue.h exists.

    This is the first text jitc_hip_assemble() emits, and two of its properties
    are load-bearing in ways that are invisible by inspection:

    1. THE NAME PLACEHOLDER. eval.cpp locates "drjit_^" with strstr(), then does
       rewind_to(marker + 6) and writes two q64 values -- 32 hex characters.
       So the prologue must contain exactly 32 '^' after "drjit_", and the
       needle must occur exactly ONCE, or the hash lands in the wrong place and
       the kernel cache silently keys on garbage.

    2. THE BOUNDS CHECK. This is a genuine divergence from Metal, not an
       oversight. Metal dispatches with thread_position_in_grid over an exact
       grid; HIP launches whole blocks, so the final block runs threads past
       the end of the buffers. Omitting the guard gives out-of-bounds writes
       that corrupt neighbouring allocations rather than crashing -- the worst
       possible failure mode. There is nothing in the Metal source to remind
       us, which is exactly why it is asserted here.
*/

#include "../src/hip_prologue.h"
#include <cstdio>
#include <cstring>
#include <string>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-52s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

/// Count non-overlapping occurrences of `needle`.
static size_t count(const std::string &hay, const char *needle) {
    size_t n = 0, pos = 0, len = strlen(needle);
    while ((pos = hay.find(needle, pos)) != std::string::npos) { n++; pos += len; }
    return n;
}

int main(int, char **) {
    printf("hip_prologue: kernel prologue\n");

    std::string p = jitc_hip_kernel_prologue(4);

    // --- The name placeholder ----------------------------------------------
    size_t at = p.find("drjit_");
    check(at != std::string::npos, "contains the drjit_ kernel-name marker");

    size_t carets = 0;
    if (at != std::string::npos)
        while (at + 6 + carets < p.size() && p[at + 6 + carets] == '^')
            carets++;
    check(carets == 32, "exactly 32 carets (2 x q64 hash written by eval.cpp)");

    // strstr() takes the FIRST match, so a second needle would misplace the
    // hash. Comments are free to contain '^' as long as not as "drjit_^".
    check(count(p, "drjit_^") == 1, "the drjit_^ needle occurs exactly once");

    // --- Entry point --------------------------------------------------------
    check(p.find("extern \"C\" __global__") != std::string::npos,
          "entry point is extern \"C\" __global__");

    // --- Params struct ------------------------------------------------------
    check(p.find("struct Params") != std::string::npos, "declares struct Params");
    check(p.find("size") != std::string::npos, "Params carries the launch size");

    // args[] excludes the leading size field: the `$o` convention indexes
    // params.args[] by parameter index MINUS ONE. With 4 kernel params that is
    // args[3]; a mismatch here overruns or wastes a slot.
    check(p.find("args[3]") != std::string::npos,
          "args[] sized n_params-1 (4 params -> args[3])");

    // Degenerate case: never emit a zero-length array, which is ill-formed.
    for (uint32_t n : { 0u, 1u }) {
        std::string q = jitc_hip_kernel_prologue(n);
        check(q.find("args[0]") == std::string::npos &&
              q.find("args[1]") != std::string::npos,
              n == 0 ? "n_params=0 clamps to args[1]" : "n_params=1 clamps to args[1]");
    }

    // --- The bounds check ---------------------------------------------------
    check(p.find("blockIdx.x") != std::string::npos &&
          p.find("blockDim.x") != std::string::npos &&
          p.find("threadIdx.x") != std::string::npos,
          "computes a global thread index");
    check(p.find("return") != std::string::npos,
          "guards against threads past the end (HIP launches whole blocks)");

    // The guard must compare against the launch size, not a constant.
    check(p.find("params.size") != std::string::npos,
          "bounds check compares against params.size");

    // --- Structural ---------------------------------------------------------
    // The prologue opens the kernel body and leaves it open for the variable
    // loop; braces must therefore be unbalanced by exactly one.
    {
        int depth = 0;
        for (char c : p) { if (c == '{') depth++; else if (c == '}') depth--; }
        check(depth == 1, "leaves exactly one brace open for the body");
    }

    if (failures) {
        printf("hip_prologue: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_prologue: all checks passed.\n");
    return 0;
}
