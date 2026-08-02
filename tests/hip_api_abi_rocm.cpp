/*
    tests/hip_api_abi_rocm.cpp -- the ROCm half of the ABI cross-check.

    Deliberately a SEPARATE translation unit. src/hip_api.h and ROCm's
    <hip/hip_runtime_api.h> cannot coexist in one file: ours declares
    `hipError_t` as an int typedef and the constants as macros, so including
    both collides on the type names AND textually shadows every enumerator we
    wanted to compare against.

    So this file sees only the REAL ROCm header and hands the values out
    through functions, which hip_api_abi.cpp compares with its own.

    Compiled only when $ROCM_INCLUDE_PATH points at a ROCm install; without it
    the cross-check is skipped and only the symbol check runs.
*/

#if defined(DRJIT_HIP_HAVE_HEADERS)

#include <hip/hip_runtime_api.h>
#include <cstddef>

extern "C" {

int rocm_attr_cc_major()      { return (int) hipDeviceAttributeComputeCapabilityMajor; }
int rocm_attr_cc_minor()      { return (int) hipDeviceAttributeComputeCapabilityMinor; }
int rocm_attr_mp_count()      { return (int) hipDeviceAttributeMultiprocessorCount; }
int rocm_attr_warp_size()     { return (int) hipDeviceAttributeWarpSize; }
int rocm_attr_max_shmem()     { return (int) hipDeviceAttributeMaxSharedMemoryPerBlock; }
int rocm_attr_pci_bus_id()    { return (int) hipDeviceAttributePciBusId; }
int rocm_attr_mem_pools()     { return (int) hipDeviceAttributeMemoryPoolsSupported; }
int rocm_attr_unified_addr()  { return (int) hipDeviceAttributeUnifiedAddressing; }

int rocm_success()            { return (int) hipSuccess; }

// hipDeviceProp_t layout. hip_core.cpp treats the struct as opaque bytes and
// reads gcnArchName at a fixed offset rather than transcribing ~100 fields, so
// these two numbers are the entire contract -- and getting them wrong yields a
// nonsense architecture string with no error anywhere.
int rocm_prop_size()          { return (int) sizeof(hipDeviceProp_t); }
int rocm_prop_gcn_arch_off()  { return (int) offsetof(hipDeviceProp_t, gcnArchName); }
int rocm_prop_gcn_arch_size() { return (int) sizeof(((hipDeviceProp_t *) 0)->gcnArchName); }
int rocm_stream_nonblocking() { return (int) hipStreamNonBlocking; }
int rocm_event_disable_timing() { return (int) hipEventDisableTiming; }
int rocm_host_malloc_default(){ return (int) hipHostMallocDefault; }

}

#endif
