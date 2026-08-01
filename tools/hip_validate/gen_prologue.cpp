/*
    gen_prologue.cpp -- emit the generated kernel prologue for validation.

    tests/hip_prologue.cpp asserts the prologue's SHAPE (caret count, args[]
    sizing, bounds check). It cannot assert that the result compiles, because
    the emitted text is not valid C++ until the kernel-name placeholder is
    substituted -- '^' is not a legal identifier character.

    So this reproduces what eval.cpp does at the equivalent point: replace the
    caret run with a hash, append a minimal body, and print. run_tests.sh pipes
    the result through hip_validate --no-exec, which compiles it for real
    gfx90a.

    --no-exec because the drjit kernel signature (a Params struct by value) is
    deliberately not the (out, in, n) convention the execution arm launches.
*/

#include "../../src/hip_prologue.h"
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    uint32_t n_params = (argc > 1) ? (uint32_t) atoi(argv[1]) : 4;

    std::string p = jitc_hip_kernel_prologue(n_params);

    size_t at = p.find("drjit_");
    if (at == std::string::npos) {
        fprintf(stderr, "gen_prologue: no kernel-name marker\n");
        return 1;
    }
    p.replace(at + 6, JITC_HIP_HASH_CARETS,
              "0123456789abcdef0123456789abcdef");

    // Touch a parameter so args[] is not optimised away, then close the body
    // the prologue left open.
    p += "    ((float *) params.args[0])[r0] = (float) r0;\n}\n";

    fputs(p.c_str(), stdout);
    return 0;
}
