/*
    src/hip_core.cpp -- HIP backend device discovery and lifecycle

    Two modes:

    REAL (default). Enumerates AMD devices through the HIP runtime, via the
    dlopen'd bindings in hip_api.{h,cpp}. Implemented; what it cannot do on a
    machine without an AMD GPU is get past hipInit(), which then reports
    "no ROCm-capable device is detected" and leaves the backend unavailable.
    The remaining Phase 1 piece is HIPThreadState (alloc/launch/memcpy).

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
#include "hip_api.h"
#include "log.h"
#include "internal.h"
#include "cuda.h"
#include <dlfcn.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <utility>

#if defined(DRJIT_ENABLE_HIP)

#if defined(DRJIT_HIP_CUDA_SHIM)
extern bool jitc_cuda_init();
#endif

#if !defined(DRJIT_HIP_CUDA_SHIM)

/// Report a HIP error and return false.
static bool jitc_hip_fail(const char *what, hipError_t rv) {
    const char *msg = hipGetErrorString ? hipGetErrorString(rv) : nullptr;
    jitc_log(Warn, "jit_hip_init(): %s failed (%d%s%s).", what, rv,
             msg ? ": " : "", msg ? msg : "");
    return false;
}

// ---------------------------------------------------------------------------
//  hipDeviceProp_t, accessed by verified offset rather than by declaration
// ---------------------------------------------------------------------------
//
// The device's ISA string ("gfx90a") is only available through
// hipGetDeviceProperties -- hipDeviceAttributeGcnArchName was deprecated to
// `hipDeviceAttributeUnused5`. That drags in hipDeviceProp_t, a ~100-field
// struct whose layout is NOT stable across ROCm versions.
//
// Two traps here, and the first is silent:
//
//   1. `hipGetDeviceProperties` is a MACRO in ROCm >= 6:
//
//          #define hipGetDeviceProperties hipGetDevicePropertiesR0600
//
//      Source code calling it therefore reaches R0600, but dlsym("hipGet\
//      DeviceProperties") binds the LEGACY R0000 entry point, which fills a
//      differently-shaped struct. Every field would be read at the wrong
//      offset -- the arch string would be garbage, and garbage arch means the
//      wrong kernel-cache key and the wrong codegen target, with no error.
//      So we resolve the VERSIONED name explicitly.
//
//   2. Declaring our own copy of the struct would be ~100 fields of pure
//      transcription risk for the two values we need. Instead we allocate a
//      generously-sized zeroed buffer and read the one field by byte offset.
//
// The two constants live in hip_api.h so tests/hip_api_abi.cpp can check them
// against ROCm's real header -- a layout change then fails loudly instead of
// yielding a nonsense architecture.

static hipError_t (*hipGetDevicePropertiesR0600)(void *, int) = nullptr;

bool jitc_hip_init() {
    if (!jitc_hip_api_init()) {
        jitc_log(Info,
                 "jit_hip_init(): the HIP runtime (libamdhip64) could not be "
                 "loaded -- the HIP backend is unavailable. Set "
                 "DRJIT_LIBHIP_PATH if ROCm is installed somewhere unusual.");
        return false;
    }

    // Resolved here rather than in the main symbol table because the versioned
    // name is what we must bind (see above), and because a ROCm older than 6.0
    // should degrade rather than fail to load the whole backend.
    if (!hipGetDevicePropertiesR0600)
        hipGetDevicePropertiesR0600 =
            decltype(hipGetDevicePropertiesR0600)(
                jitc_hip_lookup("hipGetDevicePropertiesR0600"));

    hipError_t rv = hipInit(0);
    if (rv != hipSuccess)
        return jitc_hip_fail("hipInit", rv);

    int n_devices = 0;
    rv = hipGetDeviceCount(&n_devices);
    if (rv != hipSuccess)
        return jitc_hip_fail("hipGetDeviceCount", rv);

    if (n_devices == 0) {
        jitc_log(Info, "jit_hip_init(): no AMD devices found.");
        return false;
    }

    for (int i = 0; i < n_devices; ++i) {
        hipDevice_t dev_id;
        if ((rv = hipDeviceGet(&dev_id, i)) != hipSuccess)
            return jitc_hip_fail("hipDeviceGet", rv);

        HIPDevice dev { };
        dev.id = (int) dev_id;

        // Wave width is §3.3's single-definition parameter: 64 on CDNA/GCN, 32
        // on RDNA. Read, never assumed -- codegen keys wave ops off this.
        int warp_size = 0;
        if ((rv = hipDeviceGetAttribute(&warp_size,
                                        hipDeviceAttributeWarpSize,
                                        dev_id)) != hipSuccess)
            return jitc_hip_fail("hipDeviceGetAttribute(WarpSize)", rv);
        dev.warp_size = (uint32_t) warp_size;

        size_t mem_total = 0;
        if ((rv = hipDeviceTotalMem(&mem_total, dev_id)) != hipSuccess)
            return jitc_hip_fail("hipDeviceTotalMem", rv);
        dev.memory_total = mem_total;

        // Architecture string. gcnArchName carries feature suffixes
        // ("gfx90a:sramecc+:xnack-"); the ISA name is the part before the
        // first ':', and that is what --offload-arch wants.
        if (hipGetDevicePropertiesR0600) {
            std::vector<char> props(DR_HIP_PROP_SIZE, 0);
            if ((rv = hipGetDevicePropertiesR0600(props.data(), (int) dev_id))
                    != hipSuccess)
                return jitc_hip_fail("hipGetDevicePropertiesR0600", rv);

            char arch[DR_HIP_PROP_GCN_ARCH_SIZE];
            memcpy(arch, props.data() + DR_HIP_PROP_GCN_ARCH_OFFSET,
                   sizeof(arch));
            arch[sizeof(arch) - 1] = '\0';

            if (char *colon = strchr(arch, ':'))
                *colon = '\0';

            snprintf(dev.arch, sizeof(dev.arch), "%s", arch);
        } else {
            jitc_log(Warn,
                     "jit_hip_init(): hipGetDevicePropertiesR0600 is missing "
                     "(ROCm older than 6.0?) -- cannot read the device ISA. "
                     "The HIP backend needs it to select a codegen target.");
            return false;
        }

        if (dev.arch[0] == '\0') {
            jitc_log(Warn, "jit_hip_init(): device %i reported an empty "
                           "architecture string.", i);
            return false;
        }

        char name[256] = { 0 };
        if (hipDeviceGetName(name, sizeof(name) - 1, dev_id) == hipSuccess)
            dev.name = strdup(name);

        // The context is created lazily per thread state, mirroring CUDA.
        dev.context = nullptr;

        jitc_log(Info,
                 " - Found HIP device %i: \"%s\" (%s, wave%u, %.2f GiB)",
                 i, dev.name ? dev.name : "?", dev.arch, dev.warp_size,
                 dev.memory_total / (1024.0 * 1024.0 * 1024.0));

        state.hip_devices.push_back(dev);
    }

    return !state.hip_devices.empty();
}

void jitc_hip_shutdown() {
    for (HIPDevice &d : state.hip_devices)
        free(d.name);
    state.hip_devices.clear();
    jitc_hip_api_shutdown();
}

#endif // !DRJIT_HIP_CUDA_SHIM

#if defined(DRJIT_HIP_CUDA_SHIM)

bool jitc_hip_init() {
    // Piggy-back on the CUDA backend's device discovery.
    if (state.devices.empty() && !jitc_cuda_init()) {
        jitc_log(Info, "jit_hip_init(): CUDA shim requested but no CUDA "
                       "device is available.");
        return false;
    }

    if (state.devices.empty())
        return false;

    // Unconditionally, not just when this call performed the discovery. The
    // flag is what tells the allocator that the CUDA runtime is live, and
    // jit_shutdown(light) clears it while leaving state.devices populated --
    // so on a second jit_init(HIP) the discovery is skipped, the flag stays
    // clear, and jitc_flush_malloc_cache() silently drops every HIP allocation
    // on the floor instead of freeing it. That shows up as an out-of-memory
    // several tests later, nowhere near the cause.
    state.backends |= 1u << (uint32_t) JitBackend::CUDA;

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
}

void jitc_hip_shutdown() {
    state.hip_devices.clear();
}

// ---------------------------------------------------------------------------
//  Shim kernel compilation: HIP source -> NVRTC -> PTX -> CUmodule
// ---------------------------------------------------------------------------
//
// NVRTC is resolved with dlopen rather than linked, matching how cuda_api.cpp
// resolves libcuda -- drjit-core must keep building on machines without a CUDA
// toolkit (§3.4).

typedef void *nvrtcProgram;
typedef int   nvrtcResult;

static nvrtcResult (*nv_CreateProgram)(nvrtcProgram *, const char *, const char *,
                                       int, const char **, const char **);
static nvrtcResult (*nv_CompileProgram)(nvrtcProgram, int, const char **);
static nvrtcResult (*nv_GetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*nv_GetProgramLog)(nvrtcProgram, char *);
static nvrtcResult (*nv_GetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*nv_GetPTX)(nvrtcProgram, char *);
static nvrtcResult (*nv_DestroyProgram)(nvrtcProgram *);

static bool jitc_hip_nvrtc_init() {
    static int state = 0;   // 0 = untried, 1 = ready, -1 = unavailable
    if (state)
        return state > 0;

    void *h = dlopen("libnvrtc.so.12", RTLD_LAZY);
    if (!h) h = dlopen("libnvrtc.alt.so.12", RTLD_LAZY);
    if (!h) h = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!h) {
        jitc_log(Warn, "jit_hip_compile(): the CUDA shim needs libnvrtc to "
                       "compile generated HIP source, but it could not be "
                       "loaded.");
        state = -1;
        return false;
    }

#define BIND(dst, name)                                                        \
    *(void **) &dst = dlsym(h, name);                                          \
    if (!dst) { state = -1; return false; }

    BIND(nv_CreateProgram,     "nvrtcCreateProgram")
    BIND(nv_CompileProgram,    "nvrtcCompileProgram")
    BIND(nv_GetProgramLogSize, "nvrtcGetProgramLogSize")
    BIND(nv_GetProgramLog,     "nvrtcGetProgramLog")
    BIND(nv_GetPTXSize,        "nvrtcGetPTXSize")
    BIND(nv_GetPTX,            "nvrtcGetPTX")
    BIND(nv_DestroyProgram,    "nvrtcDestroyProgram")
#undef BIND

    state = 1;
    return true;
}

std::pair<void *, bool> jitc_hip_compile(const char *source) {
    if (!jitc_hip_nvrtc_init())
        jitc_fail("jit_hip_compile(): NVRTC is unavailable.");

    nvrtcProgram prog;
    if (nv_CreateProgram(&prog, source, "drjit_hip.hip", 0, nullptr, nullptr))
        jitc_fail("jit_hip_compile(): nvrtcCreateProgram() failed.");

    // The real backend passes --offload-arch=<gfx>; only the option spelling
    // differs, not the shape of the call.
    std::vector<const char *> opts{ "--gpu-architecture=compute_86",
                                   "--std=c++17" };

    // Half precision on the shim arm.
    //
    // NVRTC rejects `_Float16` (which is what gfx90a uses) and starts with an
    // EMPTY include search list, so <cuda_fp16.h> is unreachable unless we
    // point at it. Without it the preamble has to fall back to `typedef float
    // f16`, and that is not merely imprecise -- it is four bytes wide, so every
    // load from a half buffer reads the wrong element. Aliasing is fine for a
    // compile check and wrong for execution.
    //
    // The path is supplied by the environment (the `#hip` dev shell sets it)
    // rather than baked in, because there is no way to derive it: the headers
    // live in a different package from the libnvrtc we dlopen.
    std::string inc_opt;
    const char *cuda_inc = getenv("DRJIT_HIP_SHIM_CUDA_INCLUDE");
    if (cuda_inc && *cuda_inc) {
        inc_opt = std::string("-I") + cuda_inc;
        opts.push_back(inc_opt.c_str());
        opts.push_back("-DDRJIT_SHIM_HAVE_FP16=1");
    } else {
        static bool warned = false;
        if (!warned) {
            warned = true;
            jitc_log(Warn,
                     "jit_hip_compile(): DRJIT_HIP_SHIM_CUDA_INCLUDE is unset, "
                     "so <cuda_fp16.h> is unreachable and Float16 falls back to "
                     "a 4-byte float. Kernels touching half precision will read "
                     "the wrong elements. Set it to the directory containing "
                     "cuda_fp16.h.");
        }
    }

    if (nv_CompileProgram(prog, (int) opts.size(), opts.data())) {
        size_t log_size = 0;
        nv_GetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size + 1, 0);
        nv_GetProgramLog(prog, log.data());
        // Print the source alongside the diagnostics. When a code GENERATOR
        // produces something the compiler rejects, the message alone is close
        // to useless -- the interesting artifact is the text itself, and
        // reproducing it otherwise means re-running under PrintIR.
        jitc_log(Warn, "jit_hip_compile(): generated source that failed to "
                       "compile:\n%s", source);
        jitc_fail("jit_hip_compile(): compilation of generated source failed:\n%s",
                  log.data());
    }

    size_t ptx_size = 0;
    nv_GetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nv_GetPTX(prog, ptx.data());
    nv_DestroyProgram(&prog);

    // Hand the PTX to the CUDA path so the module lands in kernel.cuda.mod,
    // which is what CUDAThreadState::launch() reads.
    auto [mod, hit] = jitc_cuda_compile(ptx.data());
    return { (void *) mod, hit };
}
#endif // DRJIT_HIP_CUDA_SHIM

#endif // DRJIT_ENABLE_HIP
