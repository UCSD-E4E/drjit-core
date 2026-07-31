/*
    src/hip_core.cpp -- HIP backend device discovery and lifecycle

    Phase 0a (PLAN.md §5): the backend is registered and inert. Device
    enumeration lands in Phase 1, which is the mechanical cu* -> hip* port of
    cuda_api.cpp / cuda_core.cpp and is the one phase that cannot be executed
    without an MI210 (§0.3).

    Keeping this file present-but-empty from the start means the enum entry,
    the dispatch arms and the build plumbing are all exercised by the compiler
    now, rather than landing in one large untested change later.
*/

#include "hip.h"
#include "log.h"

#if defined(DRJIT_ENABLE_HIP)

bool jitc_hip_init() {
    // Phase 1 will dlopen libamdhip64.so and enumerate devices here, following
    // the DRJIT_DYNAMIC_HIP pattern that cuda_api.cpp uses for libcuda.
    //
    // Returning false leaves the backend unregistered, so jit_has_backend()
    // reports absence and nothing downstream attempts to dispatch to it.
    jitc_log(Info, "jit_hip_init(): HIP backend is not implemented yet "
                   "(Phase 0a: registered but inert).");
    return false;
}

void jitc_hip_shutdown() {
    // Nothing acquired yet; Phase 1 releases contexts and the device list here.
}

#endif // DRJIT_ENABLE_HIP
