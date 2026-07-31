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
#include "eval.h"
#include "log.h"

#if defined(DRJIT_ENABLE_HIP)

void jitc_hip_assemble(ThreadState * /*ts*/, ScheduledGroup /*group*/,
                       uint32_t /*n_regs*/, uint32_t /*n_params*/) {
    jitc_fail("jitc_hip_assemble(): the HIP backend is not implemented yet "
              "(Phase 0a: registered but inert). This is unreachable while "
              "jitc_hip_init() returns false.");
}

void jitc_hip_assemble_reset() { }

const char *jitc_hip_format(size_t *size_out) {
    if (size_out)
        *size_out = 0;
    return "";
}

#endif // DRJIT_ENABLE_HIP
