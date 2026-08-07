/*
    nvrtc_then_hiprt.cpp -- does compiling with NVRTC first break HIP-RT?

    The §11n.1 puzzle in one line: the exact source drjit-core logs on failure
    builds through hiprtBuildTraceKernels() 60 times out of 60 standalone, and
    faults inside the builder in-process. So the trigger is process STATE, and
    the question is which piece of it.

    The most conspicuous difference between rtbuild and drjit-core: drjit-core
    dlopens libnvrtc.so.12 and compiles its own (non-traced) kernels with it
    before HIP-RT ever runs, while HIP-RT compiles through Orochi/hiprtc. Two
    device compilers, one process. This program reproduces that ordering and
    nothing else:

        1. dlopen libnvrtc, compile a trivial kernel, load it -- exactly the
           shape of jitc_hip_shim_nvrtc_compile().
        2. hiprtBuildTraceKernels() on the failing kernel.

    Run it with and without step 1 (--no-nvrtc) -- the pair is the experiment;
    either result alone means nothing.

    Build (inside `nix develop --impure .#hip`):
      g++ -std=c++17 nvrtc_then_hiprt.cpp \
          -I$HIPRT_NV_PATH/include -I$DRJIT_HIP_SHIM_CUDA_INCLUDE \
          -L$HIPRT_NV_PATH/lib -L/run/opengl-driver/lib \
          -lhiprt64 -lcuda -ldl -Wl,-rpath,$HIPRT_NV_PATH/lib -o nvrtc_then_hiprt
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <dlfcn.h>

#include <cuda.h>
#include <hiprt/hiprt.h>

typedef void *nvrtcProgram;
typedef int nvrtcResult;

static nvrtcResult (*nv_CreateProgram)(nvrtcProgram *, const char *,
                                       const char *, int, const char **,
                                       const char **);
static nvrtcResult (*nv_CompileProgram)(nvrtcProgram, int, const char **);
static nvrtcResult (*nv_GetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*nv_GetPTX)(nvrtcProgram, char *);
static nvrtcResult (*nv_DestroyProgram)(nvrtcProgram *);

static std::string slurp(const char *path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

/// The shape of drjit-core's jitc_hip_shim_nvrtc_compile(): dlopen NVRTC,
/// compile a kernel to PTX, hand the PTX to the driver.
static bool nvrtc_compile_something() {
    void *h = dlopen("libnvrtc.so.12", RTLD_LAZY);
    if (!h) h = dlopen("libnvrtc.alt.so.12", RTLD_LAZY);
    if (!h) h = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!h) { fprintf(stderr, "  nvrtc: dlopen failed (%s)\n", dlerror()); return false; }

    *(void **) &nv_CreateProgram   = dlsym(h, "nvrtcCreateProgram");
    *(void **) &nv_CompileProgram  = dlsym(h, "nvrtcCompileProgram");
    *(void **) &nv_GetPTXSize      = dlsym(h, "nvrtcGetPTXSize");
    *(void **) &nv_GetPTX          = dlsym(h, "nvrtcGetPTX");
    *(void **) &nv_DestroyProgram  = dlsym(h, "nvrtcDestroyProgram");
    if (!nv_CreateProgram || !nv_CompileProgram || !nv_GetPTX) {
        fprintf(stderr, "  nvrtc: missing symbols\n");
        return false;
    }

    const char *src =
        "extern \"C\" __global__ void trivial(float *x, unsigned n) {\n"
        "    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (i < n) x[i] = x[i] * 2.0f + 1.0f;\n"
        "}\n";

    nvrtcProgram prog = nullptr;
    if (nv_CreateProgram(&prog, src, "trivial.cu", 0, nullptr, nullptr) != 0) {
        fprintf(stderr, "  nvrtc: create failed\n");
        return false;
    }
    const char *opts[] = { "--std=c++17" };
    if (nv_CompileProgram(prog, 1, opts) != 0) {
        fprintf(stderr, "  nvrtc: compile failed\n");
        return false;
    }
    size_t sz = 0;
    nv_GetPTXSize(prog, &sz);
    std::string ptx(sz, '\0');
    nv_GetPTX(prog, ptx.data());
    nv_DestroyProgram(&prog);

    CUmodule mod = nullptr;
    CUresult rv = cuModuleLoadData(&mod, ptx.c_str());
    printf("  nvrtc: compiled %zu bytes of PTX, cuModuleLoadData -> %d\n",
           sz, (int) rv);
    return rv == CUDA_SUCCESS;
}

int main(int argc, char **argv) {
    bool do_nvrtc = true;
    const char *path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-nvrtc")) do_nvrtc = false;
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: nvrtc_then_hiprt [--no-nvrtc] <kernel.hip>\n"); return 2; }

    std::string src = slurp(path);
    std::string entry;
    {
        const char *k = "__global__ void ";
        size_t p = src.find(k);
        if (p == std::string::npos) { fprintf(stderr, "no __global__\n"); return 2; }
        p += strlen(k);
        entry = src.substr(p, src.find('(', p) - p);
    }

    CUdevice dev; CUcontext ctx;
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 3; }
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    printf("nvrtc-first: %s\n", do_nvrtc ? "YES" : "no");
    if (do_nvrtc && !nvrtc_compile_something())
        fprintf(stderr, "  (nvrtc step failed; result below is not the experiment)\n");

    const char *want = getenv("HIPRT_NV_PATH");
    if (want) setenv("HIPRT_PATH", want, 1);

    hiprtContextCreationInput ci{};
    ci.ctxt = (hiprtApiCtx) ctx;
    ci.device = (hiprtApiDevice) dev;
    ci.deviceType = hiprtDeviceNVIDIA;

    hiprtContext hctx = nullptr;
    if (hiprtCreateContext(HIPRT_API_VERSION, ci, hctx) != hiprtSuccess) {
        fprintf(stderr, "hiprtCreateContext failed\n"); return 4;
    }

    std::vector<std::string> opt_storage{
        std::string("-I") + (want ? want : "") + "/include", "--std=c++17"
    };
    const char *cuda_inc = getenv("DRJIT_HIP_SHIM_CUDA_INCLUDE");
    if (cuda_inc && *cuda_inc) {
        opt_storage.push_back(std::string("-I") + cuda_inc);
        opt_storage.push_back("-DDRJIT_SHIM_HAVE_FP16=1");
    }
    std::vector<const char *> opts;
    for (auto &s : opt_storage) opts.push_back(s.c_str());

    const char *fn_names[] = { entry.c_str() };
    hiprtApiFunction func = nullptr;
    hiprtApiModule mod = nullptr;

    printf("calling hiprtBuildTraceKernels...\n");
    fflush(stdout);

    hiprtError rv = hiprtBuildTraceKernels(
        hctx, 1, fn_names, src.c_str(), entry.c_str(),
        0, nullptr, nullptr, (uint32_t) opts.size(), opts.data(),
        0, 0, nullptr, &func, &mod, false);

    printf("hiprtBuildTraceKernels -> %d (%s)\n", (int) rv,
           rv == hiprtSuccess ? "SUCCESS" : "FAILURE");
    fflush(stdout);
    return rv == hiprtSuccess ? 0 : 1;
}
