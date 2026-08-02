/*
    src/hip_api.h -- Low-level interface to the HIP runtime API

    Mirrors cuda_api.h: when DRJIT_DYNAMIC_HIP is set (the default) the API is
    declared here as function POINTERS and resolved with dlsym at startup, so
    drjit-core links against no ROCm library and runs on machines without one.

    ------------------------------------------------------------------------
     The enum values below are NOT CUDA's, and that is the point
    ------------------------------------------------------------------------

    Phase 1 is described as a mechanical cu* -> hip* port, and for the FUNCTION
    NAMES it very nearly is. The device-attribute enumerators are the trap: HIP
    numbers them completely differently, and a port that renames the calls while
    keeping CUDA's constants queries the wrong attribute and gets a plausible
    number back.

        attribute                CUDA   HIP
        ComputeCapabilityMajor     75     23
        ComputeCapabilityMinor     76     61
        MultiprocessorCount        16     63
        WarpSize                   10     87
        MaxSharedMemoryPerBlock    97     74
        PciBusId                   33     67

    Every value here was extracted by compiling against ROCm's real
    hip_runtime_api.h rather than read off by counting enumerators, and
    tests/hip_api_abi.cpp re-derives them the same way so drift fails loudly.
*/

#pragma once

#include <drjit-core/jit.h>
#include <cstddef>

/// Resolve the HIP API functions. False if the runtime is unavailable.
extern bool jitc_hip_api_init();

/// Free resources allocated by jitc_hip_api_init()
extern void jitc_hip_api_shutdown();

/// Look up a HIP runtime function by name (nullptr if absent)
extern void *jitc_hip_lookup(const char *name);

#if defined(DRJIT_ENABLE_HIP)

// ---------------------------------------------------------------------------
//  Types. Opaque handles, deliberately not the ROCm typedefs: including
//  <hip/hip_runtime_api.h> here would make libdrjit-core need ROCm headers at
//  BUILD time, which is exactly what the dlopen design avoids.
// ---------------------------------------------------------------------------

using hipError_t     = int;
using hipDevice_t    = int;
using hipDeviceptr_t = void *;

struct hipCtx_st;      using hipCtx_t      = hipCtx_st *;
struct hipStream_st;   using hipStream_t   = hipStream_st *;
struct hipEvent_st;    using hipEvent_t    = hipEvent_st *;
struct hipModule_st;   using hipModule_t   = hipModule_st *;
struct hipFunction_st; using hipFunction_t = hipFunction_st *;
struct hipMemPool_st;  using hipMemPool_t  = hipMemPool_st *;

// ---------------------------------------------------------------------------
//  Constants (see the header comment -- these are HIP's, not CUDA's)
// ---------------------------------------------------------------------------

#define hipSuccess                                   0
#define hipErrorNotReady                           600

#define hipDeviceAttributeComputeCapabilityMajor    23
#define hipDeviceAttributeManagedMemory             24
#define hipDeviceAttributeComputeCapabilityMinor    61
#define hipDeviceAttributeMultiprocessorCount       63
#define hipDeviceAttributePciBusId                  67
#define hipDeviceAttributePciDeviceId               68
#define hipDeviceAttributePciDomainID               69
#define hipDeviceAttributeMaxSharedMemoryPerBlock   74
#define hipDeviceAttributeUnifiedAddressing         85
#define hipDeviceAttributeWarpSize                  87
#define hipDeviceAttributeMemoryPoolsSupported      88
#define hipDeviceAttributeConcurrentManagedAccess    9

#define hipStreamNonBlocking                         1
#define hipEventDisableTiming                        2
#define hipHostMallocDefault                         0
#define hipHostMallocPortable                        1

#if defined(DRJIT_DYNAMIC_HIP)
#  define DR_HIP_SYM(x) extern x;
#else
#  define DR_HIP_SYM(x) x;
#endif

// ---------------------------------------------------------------------------
//  The API surface Phase 1 needs. Every name here is checked against a real
//  libamdhip64.so by tests/hip_api_abi.cpp.
// ---------------------------------------------------------------------------

DR_HIP_SYM(hipError_t (*hipInit)(unsigned int));
DR_HIP_SYM(hipError_t (*hipGetDeviceCount)(int *));
DR_HIP_SYM(hipError_t (*hipDeviceGet)(hipDevice_t *, int));
DR_HIP_SYM(hipError_t (*hipDeviceGetAttribute)(int *, int, hipDevice_t));
DR_HIP_SYM(hipError_t (*hipDeviceGetName)(char *, int, hipDevice_t));
DR_HIP_SYM(hipError_t (*hipDeviceTotalMem)(size_t *, hipDevice_t));
DR_HIP_SYM(hipError_t (*hipDeviceGetPCIBusId)(char *, int, hipDevice_t));
DR_HIP_SYM(const char *(*hipGetErrorString)(hipError_t));

DR_HIP_SYM(hipError_t (*hipCtxCreate)(hipCtx_t *, unsigned int, hipDevice_t));
DR_HIP_SYM(hipError_t (*hipCtxDestroy)(hipCtx_t));
DR_HIP_SYM(hipError_t (*hipCtxSetCurrent)(hipCtx_t));
DR_HIP_SYM(hipError_t (*hipCtxPushCurrent)(hipCtx_t));
DR_HIP_SYM(hipError_t (*hipCtxPopCurrent)(hipCtx_t *));
DR_HIP_SYM(hipError_t (*hipCtxSynchronize)());

DR_HIP_SYM(hipError_t (*hipStreamCreateWithFlags)(hipStream_t *, unsigned int));
DR_HIP_SYM(hipError_t (*hipStreamDestroy)(hipStream_t));
DR_HIP_SYM(hipError_t (*hipStreamSynchronize)(hipStream_t));
DR_HIP_SYM(hipError_t (*hipStreamWaitEvent)(hipStream_t, hipEvent_t, unsigned int));

DR_HIP_SYM(hipError_t (*hipEventCreateWithFlags)(hipEvent_t *, unsigned int));
DR_HIP_SYM(hipError_t (*hipEventDestroy)(hipEvent_t));
DR_HIP_SYM(hipError_t (*hipEventRecord)(hipEvent_t, hipStream_t));

DR_HIP_SYM(hipError_t (*hipMalloc)(void **, size_t));
DR_HIP_SYM(hipError_t (*hipFree)(void *));
DR_HIP_SYM(hipError_t (*hipMallocAsync)(void **, size_t, hipStream_t));
DR_HIP_SYM(hipError_t (*hipFreeAsync)(void *, hipStream_t));
DR_HIP_SYM(hipError_t (*hipHostMalloc)(void **, size_t, unsigned int));
DR_HIP_SYM(hipError_t (*hipHostFree)(void *));
DR_HIP_SYM(hipError_t (*hipMemGetInfo)(size_t *, size_t *));

DR_HIP_SYM(hipError_t (*hipMemcpyAsync)(void *, const void *, size_t, int,
                                        hipStream_t));
DR_HIP_SYM(hipError_t (*hipMemsetD8Async)(hipDeviceptr_t, unsigned char, size_t,
                                          hipStream_t));
DR_HIP_SYM(hipError_t (*hipMemsetD16Async)(hipDeviceptr_t, unsigned short,
                                           size_t, hipStream_t));
DR_HIP_SYM(hipError_t (*hipMemsetD32Async)(hipDeviceptr_t, int, size_t,
                                           hipStream_t));

DR_HIP_SYM(hipError_t (*hipModuleLoadData)(hipModule_t *, const void *));
DR_HIP_SYM(hipError_t (*hipModuleUnload)(hipModule_t));
DR_HIP_SYM(hipError_t (*hipModuleGetFunction)(hipFunction_t *, hipModule_t,
                                              const char *));
DR_HIP_SYM(hipError_t (*hipModuleLaunchKernel)(hipFunction_t, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               unsigned int, unsigned int,
                                               hipStream_t, void **, void **));
DR_HIP_SYM(hipError_t (*hipModuleOccupancyMaxPotentialBlockSize)(
    int *, int *, hipFunction_t, size_t, int));

DR_HIP_SYM(hipError_t (*hipLaunchHostFunc)(hipStream_t, void (*)(void *),
                                           void *));

#undef DR_HIP_SYM

#endif // DRJIT_ENABLE_HIP
