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
#include "malloc.h"
#include "io.h"
#if defined(DRJIT_ENABLE_HIP) && !defined(DRJIT_HIP_CUDA_SHIM)
#  include "resources/kernels_hip.h"
#  include <lz4.h>
#endif
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

        int sm_count = 0, shared_memory_bytes = 0;
        if ((rv = hipDeviceGetAttribute(&sm_count,
                                        hipDeviceAttributeMultiprocessorCount,
                                        dev_id)) != hipSuccess)
            return jitc_hip_fail("hipDeviceGetAttribute(MultiprocessorCount)", rv);
        if ((rv = hipDeviceGetAttribute(&shared_memory_bytes,
                                        hipDeviceAttributeMaxSharedMemoryPerBlock,
                                        dev_id)) != hipSuccess)
            return jitc_hip_fail("hipDeviceGetAttribute(MaxSharedMemoryPerBlock)", rv);
        dev.sm_count = (uint32_t) sm_count;
        dev.shared_memory_bytes = (uint32_t) shared_memory_bytes;

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

// ---------------------------------------------------------------------------
//  Precompiled device library (PLAN.md Phase 3)
// ---------------------------------------------------------------------------
//
// block_reduce, compress, mkperm and friends are not API calls; they launch
// kernels from resources/kernels.cu. CUDA ships those as PTX and JITs each one
// on first use, because compiling all ~650 of them to reach one costs seconds.
// A gfx90a code object is already machine code, so there is nothing to defer:
// the blob is decompressed once, loaded once per device, and functions are
// resolved by name.
//
// The width the blob was BUILT for is checked against the width the device
// REPORTS before anything is launched. resources/common.h fixes WarpSize at
// compile time from the target (__GFX9__ => 64), so a device that disagrees --
// an RDNA part, say -- would run wave64 cross-lane code at width 32 and return
// quietly wrong reductions. §3.3 asks for that to be loud instead.

static char *jitc_hip_kernels_alloc = nullptr;

/// Wavefront width resources/kernels_gfx90a.lz4 was compiled for.
#define DR_HIP_KERNELS_WARP_SIZE 64

static bool jitc_hip_load_kernels(HIPDevice &dev) {
    if (dev.kernel_module)
        return true;

    if (dev.warp_size != DR_HIP_KERNELS_WARP_SIZE) {
        jitc_log(Warn,
                 "jit_hip(): the precompiled device library targets wave%u, but "
                 "device \"%s\" (%s) reports wave%u. Reductions, scans and "
                 "permutations would silently return wrong results, so they "
                 "are unavailable. Rebuild resources/ for this target "
                 "(make -C resources hip HIP_ARCH=%s).",
                 DR_HIP_KERNELS_WARP_SIZE, dev.name ? dev.name : "?", dev.arch,
                 dev.warp_size, dev.arch);
        return false;
    }

    if (!jitc_hip_kernels_alloc) {
        jitc_lz4_init();

        jitc_hip_kernels_alloc =
            (char *) malloc_check(kernels_gfx90a_size_uncompressed);

        int actual = LZ4_decompress_safe(
            kernels_gfx90a, jitc_hip_kernels_alloc,
            (int) kernels_gfx90a_size_compressed,
            (int) kernels_gfx90a_size_uncompressed);

        if ((size_t) actual != kernels_gfx90a_size_uncompressed)
            jitc_fail("jit_hip(): decompression of the builtin kernels failed! "
                      "Expected %zu bytes (a negative value indicates an "
                      "error), got %d.",
                      kernels_gfx90a_size_uncompressed, actual);
    }

    hipModule_t mod = nullptr;
    hipError_t rv = hipModuleLoadData(&mod, jitc_hip_kernels_alloc);
    if (rv != hipSuccess) {
        const char *msg = hipGetErrorString ? hipGetErrorString(rv) : nullptr;
        jitc_log(Warn,
                 "jit_hip(): could not load the precompiled device library on "
                 "device \"%s\" (%s): %d%s%s.", dev.name ? dev.name : "?",
                 dev.arch, (int) rv, msg ? ": " : "", msg ? msg : "");
        return false;
    }

    dev.kernel_module = (void *) mod;
    return true;
}

void *jitc_hip_kernel(int device, const char *name) {
    HIPDevice &dev = state.hip_devices[device];

    if (!jitc_hip_load_kernels(dev))
        return nullptr;

    hipFunction_t func = nullptr;
    hipError_t rv = hipModuleGetFunction(&func, (hipModule_t) dev.kernel_module,
                                         name);
    if (rv != hipSuccess)
        return nullptr;   // caller reports; the name is the useful detail

    return (void *) func;
}

void jitc_hip_shutdown() {
    for (HIPDevice &d : state.hip_devices) {
        if (d.kernel_module)
            hipModuleUnload((hipModule_t) d.kernel_module);
        free(d.name);
    }
    state.hip_devices.clear();
    free(jitc_hip_kernels_alloc);
    jitc_hip_kernels_alloc = nullptr;
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

    // Info, NOT Warn -- a correctness constraint rather than a matter of taste.
    //
    // jit_init_async() runs backend initialization on a BACKGROUND thread that
    // holds state.lock. A Warn-level message reaches drjit's Python log
    // callback, which acquires the GIL -- while the main thread holds the GIL
    // and waits on state.lock. That lock-order inversion deadlocks at
    // interpreter EXIT: the process does all of its work and prints correct
    // results, then never terminates, with both threads spinning. It cost most
    // of a day, because every symptom pointed at the code that ran BEFORE it.
    //
    // Info-level messages do not reach the Python callback at the default log
    // level, which is why every other backend's init logs freely. The loud
    // scaffold warning still exists -- jitc_init_thread_state() emits it on the
    // caller's own thread, which is also the more useful moment: it fires when
    // the backend is actually USED rather than merely linked in.
    jitc_log(Info,
             "jit_hip_init(): running the CUDA SHIM (DRJIT_HIP_CUDA_SHIM) at "
             "warp_size=%u.", dev.warp_size);
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

// ---------------------------------------------------------------------------
//  Ray tracing under the shim
// ---------------------------------------------------------------------------
//
// A generated kernel that traverses cannot go through bare NVRTC: it includes
// <hiprt/hiprt_device.h> and must be LINKED against HIP-RT's traversal
// library. hiprtBuildTraceKernels() does the compile and the link together,
// and on NVIDIA it drives NVRTC underneath -- which is why this works at all
// (BACKEND_NOTES §7b: HIP-RT supports NVIDIA through Orochi; only the stock
// nixpkgs package disables it).
//
// This is shim-only. On real hardware the same call exists and takes the same
// arguments, so the shape being exercised here is the shape the MI210 will use.

#if defined(DRJIT_HIP_SHIM_HIPRT)

#include <hiprt/hiprt.h>

static hiprtContext jitc_hiprt_ctx = nullptr;

/// Create the HIP-RT context on first use, against the current CUDA context.
static bool jitc_hip_shim_rt_init() {
    if (jitc_hiprt_ctx)
        return true;

    // HIP-RT locates its device libraries and precompiled builder kernels
    // relative to $HIPRT_PATH. The dev shell points that at the AMD build (for
    // hip_validate's gfx arm), so it has to be redirected here or context
    // creation fails as hiprtErrorInternal -- with the missing filename
    // visible only after hiprtSetLogLevel(), which is exactly the "fails late
    // and unhelpfully" mode §7b warned about.
    const char *want = DRJIT_HIP_SHIM_HIPRT_PATH;
    const char *have = getenv("HIPRT_PATH");
    if (!have || strcmp(have, want) != 0)
        setenv("HIPRT_PATH", want, 1);

    ThreadState *ts = thread_state(JitBackend::HIP);

    hiprtContextCreationInput ci {};
    ci.ctxt = (hiprtApiCtx) ts->context;
    ci.device = (hiprtApiDevice) state.devices[ts->device].id;
    ci.deviceType = hiprtDeviceNVIDIA;

    // Current for the duration, for the same reason the build path below binds
    // it: HIP-RT touches the context while setting itself up.
    scoped_set_context guard(ts->context);

    hiprtError rv = hiprtCreateContext(HIPRT_API_VERSION, ci, jitc_hiprt_ctx);
    if (rv != hiprtSuccess) {
        jitc_hiprt_ctx = nullptr;
        jitc_log(Warn,
                 "jit_hip_compile(): hiprtCreateContext() failed (%d). Ray "
                 "tracing is unavailable under the shim.", (int) rv);
        return false;
    }

    // HIP-RT reports most build failures as a bare hiprtErrorInternal, and the
    // actual cause -- usually a missing file, named -- only appears in its log
    // (BACKEND_NOTES §7b). Turning that on costs nothing and is the difference
    // between a five-minute fix and an afternoon.
    hiprtSetLogLevel(jitc_hiprt_ctx,
                     hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError);

    jitc_log(Info, "jit_hip_compile(): HIP-RT context created (shim, NVIDIA).");
    return true;
}

/// Compile a traversing kernel through HIP-RT and return its CUmodule.
/// Application-provided custom-primitive intersection source, see
/// jit_hip_set_isect_source(). Owned copies: the caller's buffers need not
/// outlive the call, and every traversing kernel built afterwards uses these.
static std::string jitc_hip_isect_source;
static std::vector<std::string> jitc_hip_isect_names, jitc_hip_filter_names;

void jitc_hip_set_isect_source(const char *source, const char **isect_names,
                               const char **filter_names,
                               uint32_t n_geom_types) {
    jitc_hip_isect_source.clear();
    jitc_hip_isect_names.clear();
    jitc_hip_filter_names.clear();

    if (!source || n_geom_types == 0)
        return;

    jitc_hip_isect_source = source;
    for (uint32_t i = 0; i < n_geom_types; ++i) {
        jitc_hip_isect_names.push_back(isect_names && isect_names[i]
                                           ? isect_names[i] : std::string());
        jitc_hip_filter_names.push_back(filter_names && filter_names[i]
                                            ? filter_names[i] : std::string());
    }

    jitc_log(Info,
             "jit_hip_set_isect_source(): registered %u geometry type%s of "
             "custom-primitive intersection source (%zu bytes).",
             n_geom_types, n_geom_types == 1 ? "" : "s",
             jitc_hip_isect_source.size());
}

uint32_t jitc_hip_isect_geom_types() {
    return (uint32_t) jitc_hip_isect_names.size();
}

static std::pair<void *, bool> jitc_hip_shim_rt_compile(const char *source,
                                                        const char *name) {
    if (!jitc_hip_shim_rt_init())
        jitc_raise("jit_hip_compile(): this kernel performs ray tracing, but "
                   "the HIP-RT context could not be created.");

    // Bind the CUDA context for the duration of the build.
    //
    // hiprtBuildTraceKernels() compiles and then LOADS a module, which is a
    // context operation: CUDA applies it to whatever context is current on this
    // thread. Every other CUDA-touching path in drjit-core takes this guard
    // (jitc_cuda_compile, jitc_cuda_sync_stream, ...); this one did not, and
    // the omission is invisible until something else changes the current
    // context. Then hiprtBuildTraceKernels either returns hiprtErrorInternal or
    // SEGFAULTS INSIDE ITSELF, with no diagnostic, on source that is perfectly
    // valid -- proven by building the identical source standalone, where it
    // succeeds.
    //
    // That is why the symptom looked like "kernels from symbolic loops and
    // vcalls are miscompiled": those merely execute enough surrounding work to
    // leave a different context current.
    ThreadState *ts = thread_state(JitBackend::HIP);
    scoped_set_context guard(ts->context);

    // NVRTC starts with an EMPTY include search list, so both the HIP-RT
    // device headers and <cuda_fp16.h> have to be spelled out.
    std::vector<std::string> opt_storage{
        std::string("-I") + DRJIT_HIP_SHIM_HIPRT_PATH + "/include",
        "--std=c++17"
    };
    const char *cuda_inc = getenv("DRJIT_HIP_SHIM_CUDA_INCLUDE");
    if (cuda_inc && *cuda_inc) {
        opt_storage.push_back(std::string("-I") + cuda_inc);
        opt_storage.push_back("-DDRJIT_SHIM_HAVE_FP16=1");
    }

    std::vector<const char *> opts;
    for (const std::string &s : opt_storage)
        opts.push_back(s.c_str());

    hiprtApiFunction func = nullptr;
    hiprtApiModule mod = nullptr;
    const char *names[1] = { name };

    // Custom-primitive intersection, if the application registered any. The
    // definitions must be compiled WITH the kernel (hiprtBuildTraceKernels
    // generates the dispatch and prepends it to this source), so they are
    // concatenated ahead of the generated body rather than linked.
    std::string combined;
    const char *build_src = source;
    uint32_t n_geom_types = (uint32_t) jitc_hip_isect_names.size();
    std::vector<hiprtFuncNameSet> fn_sets;

    if (n_geom_types) {
        combined.reserve(jitc_hip_isect_source.size() + strlen(source) + 2);
        combined = jitc_hip_isect_source;
        combined += '\n';
        combined += source;
        build_src = combined.c_str();

        fn_sets.resize(n_geom_types);
        for (uint32_t i = 0; i < n_geom_types; ++i) {
            fn_sets[i].intersectFuncName =
                jitc_hip_isect_names[i].empty() ? nullptr
                                                : jitc_hip_isect_names[i].c_str();
            fn_sets[i].filterFuncName =
                jitc_hip_filter_names[i].empty() ? nullptr
                                                 : jitc_hip_filter_names[i].c_str();
        }
    }

    hiprtError rv = hiprtBuildTraceKernels(
        jitc_hiprt_ctx, 1, names, build_src, name,
        /* numHeaders = */ 0, nullptr, nullptr,
        (uint32_t) opts.size(), opts.data(),
        n_geom_types, /* numRayTypes = */ n_geom_types ? 1u : 0u,
        fn_sets.empty() ? nullptr : fn_sets.data(),
        &func, &mod, /* cache = */ false);

    if (rv != hiprtSuccess) {
        jitc_log(Warn, "jit_hip_compile(): generated source that HIP-RT failed "
                       "to build:\n%s", source);
        jitc_fail("jit_hip_compile(): hiprtBuildTraceKernels() failed (%d).",
                  (int) rv);
    }

    // `cache = false` above, so this is never a hit. HIP-RT has its own kernel
    // cache; wiring drjit's on top of it would need the two to agree on a key,
    // and the shim is not where that is worth doing.
    return { (void *) mod, false };
}

#endif // DRJIT_HIP_SHIM_HIPRT

/// Public accessor for the HIP-RT context, mirroring jit_metal_context().
///
/// The application (Mitsuba's src/render/hip/accel.cpp) builds its geometries
/// and scenes with this. It MUST be the same context the traversing kernels are
/// compiled against: hiprtGeometry and hiprtScene handles are context-scoped,
/// so a scene built against a second context traverses garbage rather than
/// failing cleanly.
///
/// Creates the context on demand, so the first caller may equally be the
/// application or the kernel compiler.
void *jitc_hip_rt_context() {
#if defined(DRJIT_HIP_SHIM_HIPRT)
    if (!jitc_hip_shim_rt_init())
        return nullptr;
    return (void *) jitc_hiprt_ctx;
#else
    return nullptr;
#endif
}

std::pair<void *, bool> jitc_hip_compile(const char *source,
                                         const char *kernel_name) {
    // Does this kernel traverse? The emitted HIP-RT include is the marker, and
    // it is emitted exactly when a TraceRay node is present.
    if (strstr(source, "hiprt/hiprt_device.h")) {
#if defined(DRJIT_HIP_SHIM_HIPRT)
        return jitc_hip_shim_rt_compile(source, kernel_name);
#else
        jitc_raise("jit_hip_compile(): this kernel performs ray tracing, which "
                   "under the CUDA shim needs a CUDA-enabled HIP-RT build. "
                   "Configure with -DDRJIT_HIPRT_PATH=... (or set $HIPRT_NV_PATH "
                   "before configuring). The stock HIP-RT package cannot target "
                   "NVIDIA -- see the hipRtNv derivation in flake.nix.");
#endif
    }

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
