/*
    hipnv_pipeline.cpp -- prove the Phase 1 API surface on HIP-on-CUDA.

    Exercises the ENTIRE Dr.Jit JIT pipeline using nothing but the real HIP
    API, compiled by nvcc and run on an NVIDIA GPU (PLAN.md §3.5):

        hiprtcCreateProgram  ->  hiprtcCompileProgram  ->  hiprtcGetCode
                             ->  hipModuleLoadData     ->  hipModuleGetFunction
                             ->  hipMalloc / hipMemcpyHtoD
                             ->  hipModuleLaunchKernel
                             ->  hipMemcpyDtoH

    That is exactly the sequence jitc_hip_* will perform, so a green run here
    means Phase 1 can be WRITTEN AND TESTED locally rather than blind against
    hardware we do not have. This file exists to be run before hip_api.cpp is
    written, not after.

    Build (from a `nix develop .#hip` shell):

        nvcc -x cu $HIPNV_CFLAGS -arch=sm_86 hipnv_pipeline.cpp \
             -o hipnv_pipeline $HIPNV_LDFLAGS -lnvrtc

    WHAT THIS DOES NOT PROVE. On NVIDIA, HIP is a header-level translation to
    CUDA, so this validates API USAGE -- signatures, argument order, flags,
    error handling, the launch-parameter convention -- and says nothing about
    AMD behaviour. hiprtc lowers to nvrtc and emits PTX rather than a gfx90a
    code object; wave width is 32. Those remain MI210 work.
*/

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-52s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

#define HIP_CHECK(expr, what)                                                  \
    do {                                                                       \
        hipError_t e_ = (expr);                                                \
        check(e_ == hipSuccess, what);                                         \
        if (e_ != hipSuccess) {                                                \
            printf("      %s -> %s\n", #expr, hipGetErrorString(e_));          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// The kernel is written in the portable subset the emitter will produce, and
// takes its parameters the way jitc_hip_assemble() will: a plain pointer list
// rather than a bound buffer.
static const char *kernel_src = R"HIP(
extern "C" __global__ void drjit_test(float *out, const float *in, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = in[i] * 2.0f + 1.0f;
}
)HIP";

int main() {
    printf("hipnv_pipeline: Phase 1 API surface on HIP-on-CUDA\n");

    // --- Device setup --------------------------------------------------------
    HIP_CHECK(hipInit(0), "hipInit");

    int n_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&n_devices), "hipGetDeviceCount");
    check(n_devices > 0, "at least one device present");
    if (n_devices <= 0)
        return 1;

    hipDevice_t dev;
    HIP_CHECK(hipDeviceGet(&dev, 0), "hipDeviceGet");

    char name[256] = { 0 };
    HIP_CHECK(hipDeviceGetName(name, sizeof(name), dev), "hipDeviceGetName");
    printf("      device: %s\n", name);

    // warpSize is the property §3.3 makes codegen read. On this platform it is
    // 32; on gfx90a it will be 64. Reading it rather than assuming is the whole
    // point of HIPDevice::warp_size.
    int warp = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev),
              "hipDeviceGetAttribute(warpSize)");
    printf("      warp_size: %d  (64 on gfx90a -- read, never assumed)\n", warp);
    check(warp == 32 || warp == 64, "warp size is 32 or 64");

    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, dev), "hipCtxCreate");

    // --- Runtime compilation -------------------------------------------------
    hiprtcProgram prog;
    hiprtcResult rr = hiprtcCreateProgram(&prog, kernel_src, "drjit_test.hip",
                                          0, nullptr, nullptr);
    check(rr == HIPRTC_SUCCESS, "hiprtcCreateProgram");
    if (rr != HIPRTC_SUCCESS)
        return 1;

    // On the AMD platform this would be --offload-arch=gfx90a. The option
    // differs; the CALL SHAPE does not, which is what is being pinned here.
    const char *opts[] = { "--gpu-architecture=compute_86" };
    rr = hiprtcCompileProgram(prog, 1, opts);
    if (rr != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size + 1, 0);
        hiprtcGetProgramLog(prog, log.data());
        printf("  hiprtcCompileProgram FAILED:\n%s\n", log.data());
        return 1;
    }
    check(true, "hiprtcCompileProgram");

    size_t code_size = 0;
    check(hiprtcGetCodeSize(prog, &code_size) == HIPRTC_SUCCESS,
          "hiprtcGetCodeSize");
    std::vector<char> code(code_size);
    check(hiprtcGetCode(prog, code.data()) == HIPRTC_SUCCESS, "hiprtcGetCode");
    check(code_size > 0, "compiled code is non-empty");
    printf("      code: %zu bytes\n", code_size);
    hiprtcDestroyProgram(&prog);

    // --- Module load and symbol lookup --------------------------------------
    hipModule_t mod;
    HIP_CHECK(hipModuleLoadData(&mod, code.data()), "hipModuleLoadData");

    hipFunction_t fn;
    HIP_CHECK(hipModuleGetFunction(&fn, mod, "drjit_test"),
              "hipModuleGetFunction");

    // --- Allocation and transfer ---------------------------------------------
    const unsigned N = 1024;
    const size_t bytes = N * sizeof(float);

    std::vector<float> host_in(N), host_out(N, -1.0f);
    for (unsigned i = 0; i < N; ++i)
        host_in[i] = (float) i;

    void *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes),  "hipMalloc (in)");
    HIP_CHECK(hipMalloc(&d_out, bytes), "hipMalloc (out)");
    HIP_CHECK(hipMemcpyHtoD((hipDeviceptr_t) d_in, host_in.data(), bytes),
              "hipMemcpyHtoD");

    // --- Launch --------------------------------------------------------------
    //
    // hipModuleLaunchKernel takes an array of POINTERS TO the arguments, same
    // as cuLaunchKernel. Passing the values directly is the classic mistake and
    // produces garbage rather than an error.
    unsigned n_arg = N;
    void *args[] = { &d_out, &d_in, &n_arg };

    unsigned block = 128, grid = (N + block - 1) / block;
    HIP_CHECK(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1,
                                    /*sharedMemBytes=*/0, /*stream=*/nullptr,
                                    args, nullptr),
              "hipModuleLaunchKernel");
    HIP_CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize");

    // --- Readback and verify -------------------------------------------------
    HIP_CHECK(hipMemcpyDtoH(host_out.data(), (hipDeviceptr_t) d_out, bytes),
              "hipMemcpyDtoH");

    bool correct = true;
    for (unsigned i = 0; i < N; ++i)
        if (host_out[i] != (float) i * 2.0f + 1.0f)
            correct = false;
    check(correct, "kernel computed out = in * 2 + 1 for all lanes");

    HIP_CHECK(hipFree(d_in),  "hipFree (in)");
    HIP_CHECK(hipFree(d_out), "hipFree (out)");
    HIP_CHECK(hipModuleUnload(mod), "hipModuleUnload");
    HIP_CHECK(hipCtxDestroy(ctx), "hipCtxDestroy");

    if (failures) {
        printf("hipnv_pipeline: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hipnv_pipeline: full JIT pipeline works through the real HIP API.\n");
    return 0;
}
