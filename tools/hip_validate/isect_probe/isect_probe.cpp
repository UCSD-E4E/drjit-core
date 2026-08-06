/*
    isect_probe.cpp -- settle the custom-primitive function signature HIP-RT
    expects, by building against it rather than by reading documentation.

    HIP-RT declares the DISPATCHER in impl/hiprt_device_impl.h:54

        HIPRT_DEVICE bool intersectFunc(uint32_t geomType, uint32_t rayType,
                                        const hiprtFuncTableHeader &tableHeader,
                                        const hiprtRay &ray, void *payload,
                                        hiprtHit &hit);

    but not the signature it expects of the APPLICATION function named in
    `funcNameSets`, because the code that calls it is generated inside
    hiprtBuildTraceKernels() at build time and appears in no header.

    The generated text is a string literal in libhiprt0300064.so:

        __device__ bool <name>( const hiprtRay& ray, const void* data,
                                void* payload, hiprtHit& hit );
        ...
        HIPRT_DEVICE bool intersectFunc( uint32_t geomType, uint32_t rayType,
            const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray,
            void* payload, hiprtHit& hit )
        {
            const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
            [[maybe_unused]] const void* data =
                tableHeader.funcDataSets[index].intersectFuncData;
            switch ( index )
            {
                case <i>: { return <name>( ray, data, payload, hit ); }
                 default: { return false; }
            }
        }

    This program checks that reading against the shipped library. It builds a
    kernel through the same hiprtBuildTraceKernels() call drjit-core uses, but
    with numGeomTypes / numRayTypes / funcNameSets actually populated.

    A probe is only worth trusting if it can fail, so the directory ships a
    deliberately WRONG signature too. Run both:

        ./isect_probe probe_sphere.hip     mi_isect_sphere   # expect exit 0
        ./isect_probe probe_wrong_sig.hip  mi_isect_wrong    # expect exit 1

    A build that succeeds on both proves nothing -- it means the generated
    dispatcher never referenced our function.

    Build (inside `nix develop --impure .#hip`):
      g++ -std=c++17 isect_probe.cpp \
          -I$HIPRT_NV_PATH/include -I$DRJIT_HIP_SHIM_CUDA_INCLUDE \
          -L$HIPRT_NV_PATH/lib -L/run/opengl-driver/lib \
          -lhiprt64 -lcuda -Wl,-rpath,$HIPRT_NV_PATH/lib -o isect_probe

    `-lcuda` needs /run/opengl-driver/lib on NixOS: the driver stub is not in
    the CUDA package.
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
    if (argc < 2) {
        fprintf(stderr,
                "usage: isect_probe <kernel.hip> [isect_fn] [filter_fn]\n");
        return 2;
    }
    const char *path      = argv[1];
    const char *isect_fn  = argc > 2 ? argv[2] : nullptr;
    const char *filter_fn = argc > 3 ? argv[3] : nullptr;

    std::string src = slurp(path);

    // A source with no kernel of its own gets one appended. That is not a
    // convenience -- it is the shape drjit-core actually builds: registered
    // intersection source followed by a generated traversing kernel, both in
    // one translation unit. It lets this probe compile
    // src/render/hip/intersection_functions.hip from the Mitsuba tree directly,
    // in a second, instead of finding its syntax errors via a failed render.
    std::string entry;
    {
        const char *k = "__global__ void ";
        size_t p = src.find(k);
        if (p == std::string::npos) {
            printf("no __global__ in source; appending the standard "
                   "traversal kernel\n");
            src +=
                "\n"
                "extern \"C\" __global__ void probe_kernel(\n"
                "        hiprtScene scene, hiprtFuncTable table,\n"
                "        const float *rays, float *out) {\n"
                "    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    hiprtRay _r;\n"
                "    _r.origin    = { rays[i*8+0], rays[i*8+1], rays[i*8+2] };\n"
                "    _r.direction = { rays[i*8+3], rays[i*8+4], rays[i*8+5] };\n"
                "    _r.minT      = rays[i*8+6];\n"
                "    _r.maxT      = rays[i*8+7];\n"
                "    hiprtSceneTraversalClosest _tr(scene, _r, hiprtFullRayMask,\n"
                "        hiprtTraversalHintDefault, nullptr, table);\n"
                "    hiprtHit _h = _tr.getNextHit();\n"
                "    out[i] = _h.hasHit() ? _h.t : -1.0f;\n"
                "}\n";
            entry = "probe_kernel";
        } else {
            p += strlen(k);
            size_t q = src.find('(', p);
            entry = src.substr(p, q - p);
        }
    }
    printf("entry     = %s\n", entry.c_str());
    printf("intersect = %s\n", isect_fn  ? isect_fn  : "(none)");
    printf("filter    = %s\n", filter_fn ? filter_fn : "(none)");

    CUdevice dev; CUcontext ctx;
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 3; }
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, 0, dev);

    const char *want = getenv("HIPRT_NV_PATH");
    if (want) setenv("HIPRT_PATH", want, 1);

    hiprtContextCreationInput ci{};
    ci.ctxt       = (hiprtApiCtx) ctx;
    ci.device     = (hiprtApiDevice) dev;
    ci.deviceType = hiprtDeviceNVIDIA;

    hiprtContext hctx = nullptr;
    hiprtError rv = hiprtCreateContext(HIPRT_API_VERSION, ci, hctx);
    if (rv != hiprtSuccess) {
        fprintf(stderr, "hiprtCreateContext failed (%d)\n", (int) rv);
        return 4;
    }
    hiprtSetLogLevel(hctx, hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError);

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

    // One geometry type, one ray type -- the smallest table that still forces
    // HIP-RT to generate a dispatcher that calls our function.
    uint32_t n_geom_types = isect_fn ? 1u : 0u;
    hiprtFuncNameSet fn_set{};
    fn_set.intersectFuncName = isect_fn;
    fn_set.filterFuncName    = filter_fn;

    hiprtApiFunction func = nullptr;
    hiprtApiModule mod = nullptr;

    printf("calling hiprtBuildTraceKernels (numGeomTypes=%u)...\n", n_geom_types);
    fflush(stdout);

    rv = hiprtBuildTraceKernels(
        hctx, 1, fn_names, src.c_str(), entry.c_str(),
        /* numHeaders = */ 0, nullptr, nullptr,
        (uint32_t) opts.size(), opts.data(),
        n_geom_types, /* numRayTypes = */ n_geom_types ? 1u : 0u,
        n_geom_types ? &fn_set : nullptr, &func, &mod, /* cache = */ false);

    printf("hiprtBuildTraceKernels -> %d (%s)\n", (int) rv,
           rv == hiprtSuccess ? "SUCCESS" : "FAILURE");
    fflush(stdout);
    return rv == hiprtSuccess ? 0 : 1;
}
