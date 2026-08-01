/*
    src/hip_eval.cpp -- HIP C++ source generation from a Dr.Jit computation graph

    Phase 0a (PLAN.md §5): stubs only. The backend is wired into the enum, the
    dispatch chain and the build, but produces no code yet -- Phase 2 fills this
    in, following tools/hip_validate/BACKEND_NOTES.md.

    Structural reference is metal_eval.cpp, not cuda_eval.cpp: Metal is the only
    backend here that emits source text, which is the shape we need (§3.1). The
    two deliberate divergences from that template, both recorded in the notes:

      * FP64 is NOT promoted away. Metal jitc_fail()s on Float64 because Apple
        GPUs lack doubles; gfx90a has full-rate FP64 and must emit `double`.

      * Wave width is read from HIPDevice::warp_size rather than hardcoded,
        per §3.3. HIP-RT uses the same arch-varying convention.

    Emitted kernels must stay within the portable subset documented in
    tools/hip_validate/prelude_portable.h so that hip_validate can check the
    same source two ways -- compiled for real gfx90a, and executed on NVIDIA.
*/

#include "hip.h"
#include "hip_eval.h"
#include "hip_format.h"
#include "eval.h"
#include "log.h"
#include "strbuf.h"

#if defined(DRJIT_ENABLE_HIP)

void jitc_hip_assemble(ThreadState * /*ts*/, ScheduledGroup /*group*/,
                       uint32_t /*n_regs*/, uint32_t /*n_params*/) {
    jitc_fail("jitc_hip_assemble(): the HIP backend is not implemented yet "
              "(Phase 0a: registered but inert). This is unreachable while "
              "jitc_hip_init() returns false.");
}

void jitc_hip_assemble_reset() { }

/// Scratch holding the formatted copy, so the main code buffer keeps the
/// unformatted text. Mirrors metal_reindent_scratch.
static std::string hip_reindent_scratch;

const char *jitc_hip_format(size_t *size_out) {
    hip_reindent_scratch = jitc_hip_reindent(buffer.get(), buffer.size());
    if (size_out)
        *size_out = hip_reindent_scratch.size();
    return hip_reindent_scratch.c_str();
}

#endif // DRJIT_ENABLE_HIP
