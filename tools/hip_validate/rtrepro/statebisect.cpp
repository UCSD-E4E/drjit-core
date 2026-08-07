/*
    statebisect.cpp -- which piece of process state makes hiprtBuildTraceKernels
    fault?

    §11n.1 in one line: the exact source drjit-core logs on failure builds 60/60
    standalone and faults in-process. The source is clean (proven), the
    arguments are clean (proven -- rtbuild passes them identically). So the
    trigger is process STATE, and this program bisects it by adding one
    ingredient at a time before the build:

      --nvrtc     compile a kernel with NVRTC first (drjit-core does this for
                  every non-traced kernel)          [already eliminated: 0/10]
      --geom      build a hiprtGeometry first
      --scene     build a hiprtScene first (implies --geom)
      --alloc N   cuMemAlloc N MiB and keep it      (drjit's allocator holds a
                  lot of device memory by the time a render traces)
      --repeat N  call hiprtBuildTraceKernels N times in this process

    In Mitsuba the order is always: geometry + scene built by build_hip_accel(),
    THEN the first traced kernel compiled. rtbuild reproduces none of that,
    which is the most conspicuous remaining difference.

    Exit status is 0 only if every build succeeded.

    Build (inside `nix develop --impure .#hip`):
      g++ -std=c++17 statebisect.cpp \
          -I$HIPRT_NV_PATH/include -I$DRJIT_HIP_SHIM_CUDA_INCLUDE \
          -L$HIPRT_NV_PATH/lib -L/run/opengl-driver/lib \
          -lhiprt64 -lcuda -ldl -Wl,-rpath,$HIPRT_NV_PATH/lib -o statebisect
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

static std::string slurp(const char *path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

typedef void *nvrtcProgram;
static int (*nv_CreateProgram)(nvrtcProgram *, const char *, const char *, int,
                               const char **, const char **);
static int (*nv_CompileProgram)(nvrtcProgram, int, const char **);
static int (*nv_GetPTXSize)(nvrtcProgram, size_t *);
static int (*nv_GetPTX)(nvrtcProgram, char *);

static void do_nvrtc() {
    void *h = dlopen("libnvrtc.so.12", RTLD_LAZY);
    if (!h) h = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!h) { printf("  [nvrtc] dlopen failed\n"); return; }
    *(void **) &nv_CreateProgram  = dlsym(h, "nvrtcCreateProgram");
    *(void **) &nv_CompileProgram = dlsym(h, "nvrtcCompileProgram");
    *(void **) &nv_GetPTXSize     = dlsym(h, "nvrtcGetPTXSize");
    *(void **) &nv_GetPTX         = dlsym(h, "nvrtcGetPTX");
    const char *src = "extern \"C\" __global__ void t(float *x){ x[0]*=2.f; }\n";
    nvrtcProgram p = nullptr;
    nv_CreateProgram(&p, src, "t.cu", 0, nullptr, nullptr);
    const char *o[] = { "--std=c++17" };
    nv_CompileProgram(p, 1, o);
    size_t sz = 0; nv_GetPTXSize(p, &sz);
    std::string ptx(sz, '\0'); nv_GetPTX(p, ptx.data());
    CUmodule m = nullptr;
    printf("  [nvrtc] %zu bytes PTX, load -> %d\n", sz,
           (int) cuModuleLoadData(&m, ptx.c_str()));
}

/// One triangle, so the BVH builder actually runs.
static hiprtGeometry build_geom(hiprtContext ctx) {
    static const float verts[9] = { 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f };
    static const uint32_t idx[3] = { 0, 1, 2 };

    CUdeviceptr v = 0, i = 0;
    cuMemAlloc(&v, sizeof(verts)); cuMemcpyHtoD(v, verts, sizeof(verts));
    cuMemAlloc(&i, sizeof(idx));   cuMemcpyHtoD(i, idx, sizeof(idx));

    hiprtGeometryBuildInput bi{};
    bi.type = hiprtPrimitiveTypeTriangleMesh;
    bi.primitive.triangleMesh.vertices       = (hiprtDevicePtr) v;
    bi.primitive.triangleMesh.vertexCount    = 3;
    bi.primitive.triangleMesh.vertexStride   = 3 * sizeof(float);
    bi.primitive.triangleMesh.triangleIndices= (hiprtDevicePtr) i;
    bi.primitive.triangleMesh.triangleCount  = 1;
    bi.primitive.triangleMesh.triangleStride = 3 * sizeof(uint32_t);

    hiprtBuildOptions opts{};
    opts.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;

    size_t tmp_sz = 0;
    hiprtGetGeometryBuildTemporaryBufferSize(ctx, bi, opts, tmp_sz);
    CUdeviceptr tmp = 0;
    if (tmp_sz) cuMemAlloc(&tmp, tmp_sz);

    hiprtGeometry g = nullptr;
    hiprtCreateGeometry(ctx, bi, opts, g);
    hiprtBuildGeometry(ctx, hiprtBuildOperationBuild, bi, opts,
                       (hiprtDevicePtr) tmp, nullptr, g);
    cuCtxSynchronize();
    printf("  [geom] built\n");
    return g;
}

static hiprtScene build_scene(hiprtContext ctx, hiprtGeometry g) {
    hiprtInstance inst{};
    inst.type = hiprtInstanceTypeGeometry;
    inst.geometry = g;

    hiprtFrameMatrix fm{};
    for (int r = 0; r < 3; ++r) fm.matrix[r][r] = 1.f;
    hiprtTransformHeader hdr{ 0u, 1u };

    CUdeviceptr di = 0, dh = 0, df = 0;
    cuMemAlloc(&di, sizeof(inst)); cuMemcpyHtoD(di, &inst, sizeof(inst));
    cuMemAlloc(&dh, sizeof(hdr));  cuMemcpyHtoD(dh, &hdr, sizeof(hdr));
    cuMemAlloc(&df, sizeof(fm));   cuMemcpyHtoD(df, &fm, sizeof(fm));

    hiprtSceneBuildInput sbi{};
    sbi.instances                = (hiprtDevicePtr) di;
    sbi.instanceTransformHeaders = (hiprtDevicePtr) dh;
    sbi.instanceFrames           = (hiprtDevicePtr) df;
    sbi.instanceCount            = 1;
    sbi.frameCount               = 1;
    sbi.frameType                = hiprtFrameTypeMatrix;

    hiprtBuildOptions opts{};
    opts.buildFlags = hiprtBuildFlagBitPreferBalancedBuild;

    size_t tmp_sz = 0;
    hiprtGetSceneBuildTemporaryBufferSize(ctx, sbi, opts, tmp_sz);
    CUdeviceptr tmp = 0;
    if (tmp_sz) cuMemAlloc(&tmp, tmp_sz);

    hiprtScene s = nullptr;
    hiprtCreateScene(ctx, sbi, opts, s);
    hiprtBuildScene(ctx, hiprtBuildOperationBuild, sbi, opts,
                    (hiprtDevicePtr) tmp, nullptr, s);
    cuCtxSynchronize();
    printf("  [scene] built\n");
    return s;
}

int main(int argc, char **argv) {
    bool want_nvrtc = false, want_geom = false, want_scene = false;
    size_t alloc_mib = 0;
    int repeat = 1;
    std::vector<const char *> paths;   // built in order, one hiprtContext

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--nvrtc")) want_nvrtc = true;
        else if (!strcmp(argv[i], "--geom")) want_geom = true;
        else if (!strcmp(argv[i], "--scene")) { want_scene = want_geom = true; }
        else if (!strcmp(argv[i], "--alloc")) alloc_mib = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--repeat")) repeat = atoi(argv[++i]);
        else paths.push_back(argv[i]);
    }
    if (paths.empty()) { fprintf(stderr, "usage: statebisect [flags] <kernel.hip> [more.hip ...]\n"); return 2; }

    // Several DISTINCT kernels through ONE hiprtContext, in order. That is what
    // a render does and what rtbuild never did: drjit-core compiled six kernels
    // before the crash, and the sixth had the SAME source and hash as the first
    // -- which built fine. Position in the sequence, not content, is the
    // variable.
    std::vector<std::string> srcs, entries;
    for (const char *pp : paths) {
        std::string src = slurp(pp);
        const char *k = "__global__ void ";
        size_t p = src.find(k);
        if (p == std::string::npos) { fprintf(stderr, "no __global__ in %s\n", pp); return 2; }
        p += strlen(k);
        entries.push_back(src.substr(p, src.find('(', p) - p));
        srcs.push_back(std::move(src));
    }

    CUdevice dev; CUcontext ctx;
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 3; }
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    printf("state: nvrtc=%d geom=%d scene=%d alloc=%zuMiB repeat=%d\n",
           want_nvrtc, want_geom, want_scene, alloc_mib, repeat);

    if (want_nvrtc) do_nvrtc();

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

    hiprtGeometry geom = nullptr;
    if (want_geom)  geom = build_geom(hctx);
    if (want_scene) build_scene(hctx, geom);

    if (alloc_mib) {
        CUdeviceptr p = 0;
        CUresult rv = cuMemAlloc(&p, alloc_mib << 20);
        printf("  [alloc] %zu MiB -> %d\n", alloc_mib, (int) rv);
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

    int failures = 0, n = 0;
    for (int r = 0; r < repeat; ++r) {
        for (size_t k = 0; k < srcs.size(); ++k) {
            const char *fn_names[] = { entries[k].c_str() };
            hiprtApiFunction func = nullptr;
            hiprtApiModule mod = nullptr;
            printf("  [%d] %s ... ", n, paths[k]);
            fflush(stdout);   // flush BEFORE: a segfault inside the call prints nothing after
            hiprtError rv = hiprtBuildTraceKernels(
                hctx, 1, fn_names, srcs[k].c_str(), entries[k].c_str(),
                0, nullptr, nullptr, (uint32_t) opts.size(), opts.data(),
                0, 0, nullptr, &func, &mod, false);
            printf("-> %d%s\n", (int) rv, rv == hiprtSuccess ? "" : "  <-- FAILURE");
            fflush(stdout);
            if (rv != hiprtSuccess) ++failures;
            ++n;
        }
    }
    printf("failures: %d/%d\n", failures, n);
    return failures ? 1 : 0;
}
