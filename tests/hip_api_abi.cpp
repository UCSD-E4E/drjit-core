/*
    tests/hip_api_abi.cpp -- does src/hip_api.h agree with a REAL ROCm install?

    Phase 1 is described as a mechanical cu* -> hip* port, and mechanical ports
    fail in exactly two ways that a compiler cannot catch:

      1. A misspelled or renamed symbol. dlsym returns null, the backend
         disables itself, and the message says "HIP unavailable" -- which reads
         like a missing driver rather than a typo in our own source.

      2. A wrong CONSTANT. HIP numbers its device attributes completely
         differently from CUDA, so a port that renames the calls but keeps
         CUDA's enumerators queries the wrong attribute and gets a plausible
         number back. There is no error, just a wrong warp size.

    Both are checkable HERE, with no AMD GPU: libamdhip64.so is an ordinary
    shared library that loads and exports its symbols on any machine, and ROCm
    ships the headers the constants come from. Only actually RUNNING the calls
    needs hardware.

    Skips (77) when no ROCm is present, so the suite stays green elsewhere.
*/

#include "../src/hip_api.h"
#include <drjit-core/jit.h>
#include <cstdio>
#include <cstring>

#if defined(DRJIT_ENABLE_HIP) && !defined(_WIN32)
#include <dlfcn.h>

#if defined(DRJIT_HIP_HAVE_HEADERS)
// Supplied by hip_api_abi_rocm.cpp, which is the only TU that sees ROCm's real
// header -- it cannot be included here, since our macros would shadow every
// enumerator we want to compare against and our typedefs collide with its
// types.
extern "C" {
int rocm_attr_cc_major();       int rocm_attr_cc_minor();
int rocm_attr_mp_count();       int rocm_attr_warp_size();
int rocm_attr_max_shmem();      int rocm_attr_pci_bus_id();
int rocm_attr_mem_pools();      int rocm_attr_unified_addr();
int rocm_success();             int rocm_stream_nonblocking();
int rocm_event_disable_timing(); int rocm_host_malloc_default();
int rocm_prop_size();           int rocm_prop_gcn_arch_off();
int rocm_prop_gcn_arch_size();
}
#endif

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-52s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

/// Every symbol jitc_hip_api_init() binds. Kept as data so the test can report
/// ALL missing names in one run rather than stopping at the first.
static const char *required_symbols[] = {
    "hipInit", "hipGetDeviceCount", "hipDeviceGet", "hipDeviceGetAttribute",
    "hipDeviceGetName", "hipDeviceTotalMem", "hipDeviceGetPCIBusId",
    "hipGetErrorString",
    "hipCtxCreate", "hipCtxDestroy", "hipCtxSetCurrent", "hipCtxPushCurrent",
    "hipCtxPopCurrent", "hipCtxSynchronize",
    "hipStreamCreateWithFlags", "hipStreamDestroy", "hipStreamSynchronize",
    "hipStreamWaitEvent",
    "hipEventCreateWithFlags", "hipEventDestroy", "hipEventRecord",
    "hipMalloc", "hipFree", "hipMallocAsync", "hipFreeAsync",
    "hipHostMalloc", "hipHostFree", "hipMemGetInfo",
    "hipMemcpyAsync", "hipMemsetD8Async", "hipMemsetD16Async",
    "hipMemsetD32Async",
    "hipModuleLoadData", "hipModuleUnload", "hipModuleGetFunction",
    "hipModuleLaunchKernel", "hipModuleOccupancyMaxPotentialBlockSize",
    "hipLaunchHostFunc"
};

int main(int, char **) {
    printf("hip_api_abi: src/hip_api.h vs. a real ROCm install\n");

    // $DRJIT_LIBHIP_PATH is what jitc_hip_api_init() itself honours; the dev
    // shell points it at the ROCm in the Nix store.
    const char *path = getenv("DRJIT_LIBHIP_PATH");
    void *h = dlopen(path && *path ? path : "libamdhip64.so", RTLD_LAZY);
    if (!h)
        h = dlopen("libamdhip64.so.7", RTLD_LAZY);

    if (!h) {
        printf("  no libamdhip64 available -- SKIPPED\n"
                "  (set $DRJIT_LIBHIP_PATH to a ROCm libamdhip64.so to run this)\n");
        return 77;
    }

    // --- 1. Every symbol we bind actually exists -----------------------------
    //
    // HIP exports VERSIONED symbols (nm shows hipInit@@hip_4.5); dlsym with the
    // plain name resolves the default version, which is what we rely on.
    int missing = 0;
    for (const char *s : required_symbols) {
        if (!dlsym(h, s)) {
            printf("    MISSING SYMBOL: %s\n", s);
            missing++;
        }
    }
    char msg[128];
    snprintf(msg, sizeof(msg), "all %zu bound symbols exist in libamdhip64",
             sizeof(required_symbols) / sizeof(required_symbols[0]));
    check(missing == 0, msg);

    // --- 2. The constants match ROCm's own header ----------------------------
    //
    // Compiled in only when the ROCm headers are reachable, so the symbol check
    // above still runs on a runtime-only install. The values are compared
    // against the real enumerators rather than restated, so this catches drift
    // when ROCm renumbers.
#if defined(DRJIT_HIP_HAVE_HEADERS)
    check(hipDeviceAttributeComputeCapabilityMajor == rocm_attr_cc_major(),
          "ComputeCapabilityMajor matches ROCm's enum");
    check(hipDeviceAttributeComputeCapabilityMinor == rocm_attr_cc_minor(),
          "ComputeCapabilityMinor matches ROCm's enum");
    check(hipDeviceAttributeMultiprocessorCount == rocm_attr_mp_count(),
          "MultiprocessorCount matches ROCm's enum");
    check(hipDeviceAttributeWarpSize == rocm_attr_warp_size(),
          "WarpSize matches ROCm's enum");
    check(hipDeviceAttributeMaxSharedMemoryPerBlock == rocm_attr_max_shmem(),
          "MaxSharedMemoryPerBlock matches ROCm's enum");
    check(hipDeviceAttributePciBusId == rocm_attr_pci_bus_id(),
          "PciBusId matches ROCm's enum");
    check(hipDeviceAttributeMemoryPoolsSupported == rocm_attr_mem_pools(),
          "MemoryPoolsSupported matches ROCm's enum");
    check(hipDeviceAttributeUnifiedAddressing == rocm_attr_unified_addr(),
          "UnifiedAddressing matches ROCm's enum");
    check(hipSuccess == rocm_success(), "hipSuccess matches");
    check(hipStreamNonBlocking == rocm_stream_nonblocking(),
          "hipStreamNonBlocking matches");
    check(hipEventDisableTiming == rocm_event_disable_timing(),
          "hipEventDisableTiming matches");
    check(hipHostMallocDefault == rocm_host_malloc_default(),
          "hipHostMallocDefault matches");

    // hipDeviceProp_t layout. hip_core.cpp reads gcnArchName out of an opaque
    // byte buffer at a fixed offset rather than transcribing ~100 unstable
    // fields, so these three numbers ARE the contract. A ROCm release that
    // moves the field would otherwise yield a garbage architecture string --
    // and a garbage arch means the wrong kernel-cache key and the wrong
    // codegen target, silently.
    check(DR_HIP_PROP_SIZE == rocm_prop_size(),
          "sizeof(hipDeviceProp_t) matches");
    check(DR_HIP_PROP_GCN_ARCH_OFFSET == rocm_prop_gcn_arch_off(),
          "offsetof(gcnArchName) matches");
    check(DR_HIP_PROP_GCN_ARCH_SIZE == rocm_prop_gcn_arch_size(),
          "sizeof(gcnArchName) matches");

    // The trap this test exists for: these must NOT be CUDA's numbers. If a
    // future edit "simplifies" them back to the CUDA values, every device
    // query silently returns the wrong property.
    check(hipDeviceAttributeComputeCapabilityMajor != 75 &&
          hipDeviceAttributeWarpSize != 10 &&
          hipDeviceAttributeMultiprocessorCount != 16,
          "attribute values are HIP's, not CUDA's");
#else
    printf("  (ROCm headers not present -- constant cross-check skipped)\n");
#endif

    dlclose(h);

    if (failures) {
        printf("hip_api_abi: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_api_abi: hip_api.h agrees with the installed ROCm.\n");
    return 0;
}

#else

int main(int, char **) {
    printf("hip_api_abi: HIP backend disabled -- SKIPPED\n");
    return 77;
}

#endif
