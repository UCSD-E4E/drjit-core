/*
    src/hip_api.cpp -- dlopen/dlsym binding for the HIP runtime API.

    Mirrors cuda_api.cpp. The whole point is that libdrjit-core links against no
    ROCm library: a build with the HIP backend enabled still loads and runs on a
    machine with no ROCm installed, and simply reports the backend as
    unavailable.

    Note HIP exports VERSIONED symbols -- `nm -D` shows `hipInit@@hip_4.5` --
    but dlsym("hipInit") resolves the default version, so the plain names work.
*/

// MUST precede the include: hip_api.h guards this definition, so setting it
// here turns its declaration list into DEFINITIONS. `#pragma once` means a
// second include would do nothing, which is why this cannot be done later --
// cuda_api.cpp is arranged the same way.
#if defined(DRJIT_DYNAMIC_HIP)
#  define DR_HIP_SYM(...) __VA_ARGS__ = nullptr;
#endif

#include "hip_api.h"
#include "log.h"
#include "internal.h"

#if defined(DRJIT_ENABLE_HIP)

#if defined(DRJIT_DYNAMIC_HIP)

#  if !defined(_WIN32)
#    include <dlfcn.h>
#  else
#    include <windows.h>
#  endif

static void *jitc_hip_handle = nullptr;

void *jitc_hip_lookup(const char *name) {
    if (!jitc_hip_handle)
        return nullptr;
#  if !defined(_WIN32)
    return dlsym(jitc_hip_handle, name);
#  else
    return (void *) GetProcAddress((HMODULE) jitc_hip_handle, name);
#  endif
}

bool jitc_hip_api_init() {
    if (jitc_hip_handle)
        return true;

#  if defined(_WIN32)
    const char *hip_fname = "amdhip64_7.dll", *hip_glob = nullptr;
#  else
    // The unversioned name is a development symlink and is frequently absent
    // on runtime-only installs, so the glob covers the versioned SONAMEs the
    // way HIP-RT's own loader does.
    const char *hip_fname = "libamdhip64.so",
               *hip_glob  = "/opt/rocm/lib/libamdhip64.so.*";
#  endif

    jitc_hip_handle = jitc_find_library(hip_fname, hip_glob, "DRJIT_LIBHIP_PATH");
    if (!jitc_hip_handle)
        return false;   // no ROCm on this machine; backend stays unavailable

    const char *symbol = nullptr;

    do {
        #define LOAD(name)                                          \
            symbol = #name;                                         \
            name = decltype(name)(jitc_hip_lookup(symbol));         \
            if (!name)                                              \
                break;                                              \
            symbol = nullptr

        LOAD(hipInit);
        LOAD(hipGetDeviceCount);
        LOAD(hipDeviceGet);
        LOAD(hipDeviceGetAttribute);
        LOAD(hipDeviceGetName);
        LOAD(hipDeviceTotalMem);
        LOAD(hipDeviceGetPCIBusId);
        LOAD(hipGetErrorString);

        LOAD(hipCtxCreate);
        LOAD(hipCtxDestroy);
        LOAD(hipCtxSetCurrent);
        LOAD(hipCtxPushCurrent);
        LOAD(hipCtxPopCurrent);
        LOAD(hipCtxSynchronize);

        LOAD(hipStreamCreateWithFlags);
        LOAD(hipStreamDestroy);
        LOAD(hipStreamSynchronize);
        LOAD(hipStreamWaitEvent);

        LOAD(hipEventCreateWithFlags);
        LOAD(hipEventDestroy);
        LOAD(hipEventRecord);

        LOAD(hipMalloc);
        LOAD(hipFree);
        LOAD(hipMallocAsync);
        LOAD(hipFreeAsync);
        LOAD(hipHostMalloc);
        LOAD(hipHostFree);
        LOAD(hipMemGetInfo);

        LOAD(hipMemcpyAsync);
        LOAD(hipMemsetD8Async);
        LOAD(hipMemsetD16Async);
        LOAD(hipMemsetD32Async);

        LOAD(hipModuleLoadData);
        LOAD(hipModuleUnload);
        LOAD(hipModuleGetFunction);
        LOAD(hipModuleLaunchKernel);
        LOAD(hipModuleOccupancyMaxPotentialBlockSize);

        LOAD(hipLaunchHostFunc);

        #undef LOAD
    } while (false);

    if (symbol) {
        // Naming the symbol matters: a mechanical port's usual failure is one
        // wrong name out of forty, and "HIP initialisation failed" would send
        // the reader looking at the driver instead.
        jitc_log(Warn,
                 "jit_hip_api_init(): could not find symbol \"%s\" in the HIP "
                 "runtime -- disabling the HIP backend.", symbol);
        jitc_hip_api_shutdown();
        return false;
    }

    return true;
}

void jitc_hip_api_shutdown() {
    if (!jitc_hip_handle)
        return;

    // Listed explicitly rather than re-including the header, which `#pragma
    // once` would make a no-op. Same as jitc_cuda_api_shutdown().
    #define Z(x) x = nullptr
    Z(hipInit); Z(hipGetDeviceCount); Z(hipDeviceGet);
    Z(hipDeviceGetAttribute); Z(hipDeviceGetName); Z(hipDeviceTotalMem);
    Z(hipDeviceGetPCIBusId); Z(hipGetErrorString);
    Z(hipCtxCreate); Z(hipCtxDestroy); Z(hipCtxSetCurrent);
    Z(hipCtxPushCurrent); Z(hipCtxPopCurrent); Z(hipCtxSynchronize);
    Z(hipStreamCreateWithFlags); Z(hipStreamDestroy);
    Z(hipStreamSynchronize); Z(hipStreamWaitEvent);
    Z(hipEventCreateWithFlags); Z(hipEventDestroy); Z(hipEventRecord);
    Z(hipMalloc); Z(hipFree); Z(hipMallocAsync); Z(hipFreeAsync);
    Z(hipHostMalloc); Z(hipHostFree); Z(hipMemGetInfo);
    Z(hipMemcpyAsync); Z(hipMemsetD8Async); Z(hipMemsetD16Async);
    Z(hipMemsetD32Async);
    Z(hipModuleLoadData); Z(hipModuleUnload); Z(hipModuleGetFunction);
    Z(hipModuleLaunchKernel); Z(hipModuleOccupancyMaxPotentialBlockSize);
    Z(hipLaunchHostFunc);
    #undef Z

#  if !defined(_WIN32)
    if (jitc_hip_handle != RTLD_NEXT)
        dlclose(jitc_hip_handle);
#  endif
    jitc_hip_handle = nullptr;
}

#else // !DRJIT_DYNAMIC_HIP

bool jitc_hip_api_init() { return true; }
void jitc_hip_api_shutdown() { }
void *jitc_hip_lookup(const char *) { return nullptr; }

#endif

#endif // DRJIT_ENABLE_HIP
