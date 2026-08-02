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
#include "../../src/hip_preamble.h"
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
    // the prologue left open. The f16 line is not decoration: the type and
    // DRJIT_HALF_FROM_BITS are emitted by jitc_hip_literal() into every kernel
    // that carries a half constant, and nothing else compiles that contract for
    // gfx90a -- where f16 is `_Float16` rather than the shim's `float`.
    p += "    f16 h = DRJIT_HALF_FROM_BITS(0x3c00u);   /* 1.0 */\n"
         // Round-trip through the binary view, which is how Not/And/Or/Xor on
         // a half are emitted. On gfx90a f16 is `_Float16`, where `~h` does not
         // compile at all -- this is the only place that pairing is checked
         // against the real target.
         "    u16 hb = drjit_half_to_bits(h);\n"
         "    f16 h2 = DRJIT_HALF_FROM_BITS((u32) ~(u16) ~hb);\n"
         "    ((float *) params.args[0])[r0] =\n"
         "        (float) r0 + (float) h + (float) h2 + (float) (hb != 0x3c00u);\n}\n";

    fputs(hip_type_preamble, stdout);
    fputs(p.c_str(), stdout);
    return 0;
}
