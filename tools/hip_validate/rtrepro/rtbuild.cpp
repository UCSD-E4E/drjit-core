/*
    rtbuild.cpp -- minimal standalone reproducer for the Phase 7 blocker.

    Calls hiprtBuildTraceKernels() on a source file with exactly the arguments
    drjit-core's jitc_hip_shim_rt_compile() uses, so a failure here is HIP-RT's
    and a success here means the problem is in how drjit-core invokes it (or in
    surrounding state), not in the emitted text.

    Build (inside `nix develop --impure .#hip`):
      g++ -std=c++17 rtbuild.cpp -I$HIPRT_NV_PATH/include -I$CUDA_PATH/include \
          -L$HIPRT_NV_PATH/lib -lhiprt64 -lcuda -Wl,-rpath,$HIPRT_NV_PATH/lib \
          -o rtbuild
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <hiprt/hiprt.h>

static std::string slurp(const char *path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: rtbuild <kernel.hip> [entry]\n"); return 2; }
    const char *path = argv[1];

    std::string src = slurp(path);

    // Recover the entry-point name from the source so this needs no bookkeeping.
    std::string entry;
    if (argc > 2) {
        entry = argv[2];
    } else {
        const char *k = "__global__ void ";
        size_t p = src.find(k);
        if (p == std::string::npos) { fprintf(stderr, "no __global__ found\n"); return 2; }
        p += strlen(k);
        size_t q = src.find('(', p);
        entry = src.substr(p, q - p);
    }
    printf("entry = %s\n", entry.c_str());

    // --- CUDA context, as the shim's thread state would have ---
    CUdevice dev; CUcontext ctx;
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 3; }
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    // HIP-RT locates its device libraries relative to $HIPRT_PATH.
    const char *want = getenv("HIPRT_NV_PATH");
    if (want) setenv("HIPRT_PATH", want, 1);

    hiprtContextCreationInput ci{};
    ci.ctxt = (hiprtApiCtx) ctx;
    ci.device = (hiprtApiDevice) dev;
    ci.deviceType = hiprtDeviceNVIDIA;

    hiprtContext hctx = nullptr;
    hiprtError rv = hiprtCreateContext(HIPRT_API_VERSION, ci, hctx);
    if (rv != hiprtSuccess) { fprintf(stderr, "hiprtCreateContext failed (%d)\n", (int) rv); return 4; }
    hiprtSetLogLevel(hctx, hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError);

    // --- Exactly drjit-core's invocation ---
    std::vector<std::string> opt_storage{
        std::string("-I") + (want ? want : "") + "/include",
        "--std=c++17"
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

    printf("calling hiprtBuildTraceKernels (%zu opts)...\n", opts.size());
    fflush(stdout);

    // Argument-for-argument identical to jitc_hip_shim_rt_compile(), including
    // passing the kernel name as moduleName.
    rv = hiprtBuildTraceKernels(
        hctx, 1, fn_names, src.c_str(), entry.c_str(),
        /* numHeaders = */ 0, nullptr, nullptr,
        (uint32_t) opts.size(), opts.data(),
        /* numGeomTypes = */ 0, /* numRayTypes = */ 0,
        /* funcNameSets = */ nullptr, &func, &mod, /* cache = */ false);

    printf("hiprtBuildTraceKernels -> %d (%s)\n", (int) rv,
           rv == hiprtSuccess ? "SUCCESS" : "FAILURE");
    fflush(stdout);
    return rv == hiprtSuccess ? 0 : 1;
}
