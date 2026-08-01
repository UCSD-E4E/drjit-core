/*
    src/hip_core.cpp -- HIP backend device discovery and lifecycle

    Two modes:

    REAL (default). Enumerates AMD devices through the HIP runtime. Lands in
    Phase 1 as the mechanical cu* -> hip* port of cuda_api.cpp / cuda_core.cpp
    (PLAN.md §4). Not implemented yet.

    SHIM (-DDRJIT_HIP_CUDA_SHIM=1). A DEVELOPMENT SCAFFOLD that backs the HIP
    backend with the CUDA runtime so the whole pipeline -- codegen, compile,
    launch, readback -- can be exercised end to end on an NVIDIA GPU with no
    AMD hardware present. PLAN.md §0.3 calls for exactly this:

        "Do not wait on the HIP runtime layer to start executing generated
         kernels ... pair HIP codegen with the existing CUDA runtime path."

    The shim is honest rather than fake: allocations, launches and copies all
    really happen, on a real device. What it does NOT exercise is the HIP API
    surface itself (hipMalloc, hipModuleLaunchKernel), which is why Phase 1
    still has to be written and run on an MI210.

    Crucially the shim reports warp_size = 32, the true width of the device it
    is running on -- NOT 64. Codegen reads the width from HIPDevice (§3.3), so
    emitted kernels match the hardware they actually execute on, and the wave64
    gap stays visible in data rather than being papered over.
*/

#include "hip.h"
#include "log.h"
#include "internal.h"

#if defined(DRJIT_ENABLE_HIP)

#if defined(DRJIT_HIP_CUDA_SHIM)
extern bool jitc_cuda_init();
#endif

bool jitc_hip_init() {
#if defined(DRJIT_HIP_CUDA_SHIM)
    // Piggy-back on the CUDA backend's device discovery.
    if (state.devices.empty()) {
        if (!jitc_cuda_init()) {
            jitc_log(Info, "jit_hip_init(): CUDA shim requested but no CUDA "
                           "device is available.");
            return false;
        }
        state.backends |= 1u << (uint32_t) JitBackend::CUDA;
    }

    if (state.devices.empty())
        return false;

    HIPDevice dev { };
    dev.id      = 0;
    dev.context = nullptr;

    // The NOMINAL codegen target. Emitted source is still checked against real
    // gfx90a by tools/hip_validate; here it only labels the kernel cache.
    snprintf(dev.arch, sizeof(dev.arch), "gfx90a");

    // The ACTUAL width of the device we will execute on. Reporting 64 here
    // would make codegen emit width-64 wave ops onto warp-32 hardware and
    // silently compute the wrong thing -- the exact failure §3.3's
    // single-definition parameter exists to prevent.
    dev.warp_size = 32;

    dev.memory_total = 0;
    dev.name = nullptr;

    state.hip_devices.push_back(dev);

    jitc_log(Warn,
             "jit_hip_init(): running the CUDA SHIM (DRJIT_HIP_CUDA_SHIM). "
             "HIP codegen is executing on an NVIDIA device at warp_size=%u. "
             "This is a development scaffold: it does not exercise the HIP "
             "runtime API, and wave64 semantics are NOT verified.",
             dev.warp_size);
    return true;
#else
    // Phase 1 will dlopen libamdhip64.so and enumerate devices here, following
    // the DRJIT_DYNAMIC_HIP pattern that cuda_api.cpp uses for libcuda.
    //
    // Returning false leaves the backend unregistered, so jit_has_backend()
    // reports absence and nothing downstream attempts to dispatch to it.
    jitc_log(Info, "jit_hip_init(): HIP backend is not implemented yet "
                   "(Phase 0a: registered but inert). Build with "
                   "-DDRJIT_HIP_CUDA_SHIM=ON to exercise codegen on an "
                   "NVIDIA device.");
    return false;
#endif
}

void jitc_hip_shutdown() {
#if defined(DRJIT_HIP_CUDA_SHIM)
    state.hip_devices.clear();
#endif
}

#endif // DRJIT_ENABLE_HIP
