/*
    hiprt_triangle.cpp -- HIP-RT ray tracing, EXECUTED, on an NVIDIA GPU.

    Builds a real BVH on the device, compiles a traversal kernel through NVRTC,
    launches it, and checks the hits. This is the ray-tracing counterpart to
    hipnv_pipeline.cpp, and the reason spec_trace.hip's "never executed" caveat
    no longer covers correctness.

    ------------------------------------------------------------------------
     Why an NVIDIA arm is a good proxy here, unusually
    ------------------------------------------------------------------------

    gfx90a has NO ray-tracing hardware -- CDNA2 is a compute architecture, and
    HIP-RT puts it on the RTIP 0 path: a software BVH walk in ordinary vector
    ALU (confirmed by disassembly, BACKEND_NOTES §7a: zero image_bvh
    instructions in the linked gfx90a object). That software path is portable
    C++, so NVIDIA runs very nearly the same code the MI210 will. Contrast an
    RDNA card, which would take the hardware branch and prove much less.

    What this still does NOT cover: wave64 (this runs at 32) and the AMD host
    API. Those need the card.

    ------------------------------------------------------------------------
     Requires $HIPRT_NV_PATH -- the CUDA-enabled HIP-RT build
    ------------------------------------------------------------------------

    NOT the stock package, which cannot target NVIDIA at all. See the hipRtNv
    derivation in flake.nix for the three separate things nixpkgs disables.

    ------------------------------------------------------------------------
     The test
    ------------------------------------------------------------------------

    The triangle spans (0,0,0)-(1,0,0)-(0,1,0) in the z=0 plane. Rays fire down
    -z from z=+1. Lane 0 aims at (0.1,0.1) -- inside; lane 1 at (0.9,0.9) --
    outside, past the hypotenuse. So the expected result is HIT at t=1 for lane
    0 and MISS for lane 1.

    Both lanes matter: a backend that always hits, always misses, or ignores
    the ray origin entirely fails at least one of them. Checking only the hit
    would pass on a stub that reports every ray as hitting.
*/
#include <hiprt/hiprt.h>
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#define CU(x) do { CUresult r=(x); if(r){const char*m=nullptr;cuGetErrorString(r,&m); \
    printf("CUDA fail %s: %s\n",#x,m?m:"?"); return 1;} } while(0)
#define RT(x) do { hiprtError e=(x); if(e!=hiprtSuccess){ \
    printf("HIPRT fail %s -> %d\n",#x,(int)e); return 1;} } while(0)

static const char *kernel_src = R"(
#include <hiprt/hiprt_device.h>

extern "C" __global__ void trace(hiprtGeometry geom, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    hiprtRay ray;
    ray.origin    = { i == 0 ? 0.1f : 0.9f, i == 0 ? 0.1f : 0.9f, 1.0f };
    ray.direction = { 0.0f, 0.0f, -1.0f };
    ray.minT      = 0.0f;
    ray.maxT      = 1000.0f;

    hiprtGeomTraversalClosest tr(geom, ray);
    hiprtHit hit = tr.getNextHit();

    out[i * 2 + 0] = hit.hasHit() ? 1.0f : 0.0f;
    out[i * 2 + 1] = hit.hasHit() ? hit.t : -1.0f;
}
)";

int main() {
    CU(cuInit(0));
    CUdevice dev; CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CU(cuCtxCreate(&ctx, 0, dev));

    hiprtContextCreationInput ci {};
    ci.ctxt = (hiprtApiCtx) ctx; ci.device = (hiprtApiDevice) dev;
    ci.deviceType = hiprtDeviceNVIDIA;
    hiprtContext hctx;
    RT(hiprtCreateContext(HIPRT_API_VERSION, ci, hctx));
    hiprtSetLogLevel(hctx, hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError);
    printf("context ok\n");

    // --- geometry ---
    float verts[] = { 0,0,0,  1,0,0,  0,1,0 };
    uint32_t idx[] = { 0,1,2 };
    CUdeviceptr d_v, d_i;
    CU(cuMemAlloc(&d_v, sizeof(verts))); CU(cuMemcpyHtoD(d_v, verts, sizeof(verts)));
    CU(cuMemAlloc(&d_i, sizeof(idx)));   CU(cuMemcpyHtoD(d_i, idx, sizeof(idx)));

    hiprtTriangleMeshPrimitive mesh {};
    mesh.vertexCount = 3; mesh.vertexStride = sizeof(float)*3;
    mesh.vertices = (void *) d_v;
    mesh.triangleCount = 1; mesh.triangleStride = sizeof(uint32_t)*3;
    mesh.triangleIndices = (void *) d_i;

    hiprtGeometryBuildInput bi {};
    bi.type = hiprtPrimitiveTypeTriangleMesh;
    bi.primitive.triangleMesh = mesh;

    hiprtBuildOptions bo {}; bo.buildFlags = hiprtBuildFlagBitPreferFastBuild;
    size_t tmp_size; RT(hiprtGetGeometryBuildTemporaryBufferSize(hctx, bi, bo, tmp_size));
    CUdeviceptr d_tmp = 0;
    if (tmp_size) CU(cuMemAlloc(&d_tmp, tmp_size));

    hiprtGeometry geom;
    RT(hiprtCreateGeometry(hctx, bi, bo, geom));
    RT(hiprtBuildGeometry(hctx, hiprtBuildOperationBuild, bi, bo,
                          (void *) d_tmp, 0, geom));
    printf("BVH built\n");

    // --- kernel ---
    const char *fn[] = { "trace" };
    // NVRTC starts with an empty include search list, so the hiprt device
    // headers the kernel includes are unreachable without this.
    std::string inc = std::string("-I") + getenv("HIPRT_PATH") + "/include";
    const char *opts[] = { inc.c_str() };
    hiprtApiFunction func; hiprtApiModule mod;
    RT(hiprtBuildTraceKernels(hctx, 1, fn, kernel_src, "tri_module",
                              0, nullptr, nullptr, 1, opts,
                              0, 0, nullptr, &func, &mod, false));
    printf("kernel built\n");

    const int n = 2;
    CUdeviceptr d_out; CU(cuMemAlloc(&d_out, sizeof(float)*2*n));
    CU(cuMemsetD8(d_out, 0, sizeof(float)*2*n));

    int nn = n;
    void *args[] = { &geom, &d_out, &nn };
    CU(cuLaunchKernel((CUfunction) func, 1,1,1, n,1,1, 0, nullptr, args, nullptr));
    CU(cuCtxSynchronize());

    float out[2*n];
    CU(cuMemcpyDtoH(out, d_out, sizeof(out)));

    int bad = 0;
    printf("lane 0 (inside):  hit=%.0f t=%.3f   want hit=1 t=1.000\n", out[0], out[1]);
    printf("lane 1 (outside): hit=%.0f t=%.3f   want hit=0 t=-1.000\n", out[2], out[3]);
    if (out[0] != 1.0f || out[1] < 0.999f || out[1] > 1.001f) bad++;
    if (out[2] != 0.0f) bad++;

    printf(bad ? "TRAVERSAL WRONG (%d checks failed)\n" : "TRAVERSAL CORRECT\n", bad);
    return bad ? 1 : 0;
}
