/*
    hipnv_ts_calls.cpp -- do HIPThreadState's calls have the right SHAPE?

    src/hip_ts.cpp is the mechanical cu* -> hip* port of cuda_ts.cpp, and it
    cannot be run here: hipInit() fails without an AMD GPU. What CAN be checked
    is every call's shape -- signature, argument ORDER, flag VALUES, error
    convention -- by issuing the same sequences through HIP-on-CUDA (PLAN.md
    §3.5), where the real HIP headers translate to CUDA and actually execute.

    That covers the port's characteristic failure modes:

      * a swapped argument (hipMemsetD32Async's count is in ELEMENTS, not bytes)
      * a CUDA constant left behind (hipMemcpyDefault is 4, cudaMemcpyDefault
        is also 4 -- but hipStreamNonBlocking is 1 where CU_STREAM_NON_BLOCKING
        is also 1, so the ones that DO differ hide among the ones that do not)
      * a missing stream parameter, which compiles into a synchronous call
      * a wrong direction constant, which fails only for device-to-device

    What it does NOT cover: anything AMD-specific about the runtime. Those live
    with wave64 and fp16 on the needs-an-MI210 list.

    Build/run is driven by run_tests.sh; see hipnv_pipeline.cpp for the
    HIPNV_CFLAGS contract.
*/

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-56s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

#define HIP(x)                                                                 \
    do {                                                                       \
        hipError_t rv_ = (x);                                                  \
        if (rv_ != hipSuccess) {                                               \
            printf("  %s -> %d (%s)\n", #x, (int) rv_, hipGetErrorString(rv_));\
            failures++;                                                        \
        }                                                                      \
    } while (0)

// The constants src/hip_api.h hardcodes, restated here so the two are compared
// through a compiler that has the REAL header.
static void check_constants() {
    check((int) hipSuccess == 0, "hipSuccess == 0");
    check((int) hipMemcpyDefault == 4, "hipMemcpyDefault == 4 (DR_HIP_MEMCPY_DEFAULT)");
    check((int) hipStreamNonBlocking == 1, "hipStreamNonBlocking == 1");
    check((int) hipEventDisableTiming == 2, "hipEventDisableTiming == 2");
    check((int) hipHostMallocDefault == 0, "hipHostMallocDefault == 0");
}

int main() {
    printf("hipnv_ts_calls: HIPThreadState call shapes, via HIP-on-CUDA\n");

    check_constants();

    HIP(hipInit(0));

    // --- Thread-state setup, as jitc_init_thread_state() performs it ---------
    hipDevice_t dev;
    HIP(hipDeviceGet(&dev, 0));

    hipCtx_t ctx;
    HIP(hipCtxCreate(&ctx, 0, dev));

    hipStream_t stream;
    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    hipEvent_t event;
    HIP(hipEventCreateWithFlags(&event, hipEventDisableTiming));

    // Wave width, read rather than assumed (§3.3). 32 here, 64 on CDNA.
    int warp_size = 0;
    HIP(hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, dev));
    check(warp_size == 32 || warp_size == 64,
          "hipDeviceGetAttribute(WarpSize) returns a legal width");

    // --- memset_async, all four element sizes --------------------------------
    //
    // The COUNT argument is in elements, not bytes. Passing a byte count would
    // overrun by 2x/4x and corrupt whatever follows -- checked by verifying the
    // element AFTER the filled range is untouched.
    const size_t n = 256;
    void *buf = nullptr;
    HIP(hipMalloc(&buf, n * sizeof(uint64_t)));
    HIP(hipMemsetD8Async((hipDeviceptr_t) buf, 0xAB, n, stream));
    HIP(hipMemsetD16Async((hipDeviceptr_t) buf, 0xBEEF, n, stream));
    HIP(hipMemsetD32Async((hipDeviceptr_t) buf, 0xDEADBEEF, n, stream));
    HIP(hipStreamSynchronize(stream));

    {
        // Fill 8 u32 elements, then confirm element 8 was NOT written.
        std::vector<uint32_t> host(16, 0x11111111u);
        HIP(hipMemcpyAsync(buf, host.data(), host.size() * 4,
                           hipMemcpyDefault, stream));
        HIP(hipMemsetD32Async((hipDeviceptr_t) buf, 0x22222222, 8, stream));
        HIP(hipStreamSynchronize(stream));
        HIP(hipMemcpyAsync(host.data(), buf, host.size() * 4,
                           hipMemcpyDefault, stream));
        HIP(hipStreamSynchronize(stream));
        check(host[7] == 0x22222222u && host[8] == 0x11111111u,
              "hipMemsetD32Async count is in ELEMENTS, not bytes");
    }

    // --- memcpy_async in both directions, with hipMemcpyDefault --------------
    //
    // The direction constant is the divergence from the CUDA driver API, which
    // infers it. Default must work for host->device, device->host AND
    // device->device; a reflexive hipMemcpyHostToDevice would pass the first
    // two tests here and fail the third.
    {
        std::vector<uint32_t> src(n, 0x5A5A5A5Au), dst(n, 0);
        void *buf2 = nullptr;
        HIP(hipMalloc(&buf2, n * sizeof(uint32_t)));

        HIP(hipMemcpyAsync(buf, src.data(), n * 4, hipMemcpyDefault, stream));
        HIP(hipMemcpyAsync(buf2, buf, n * 4, hipMemcpyDefault, stream));  // D2D
        HIP(hipMemcpyAsync(dst.data(), buf2, n * 4, hipMemcpyDefault, stream));
        HIP(hipStreamSynchronize(stream));

        bool ok = true;
        for (uint32_t v : dst)
            ok &= (v == 0x5A5A5A5Au);
        check(ok, "hipMemcpyDefault round-trips H2D -> D2D -> D2H");
        HIP(hipFree(buf2));
    }

    // --- poke: staged through pinned host memory -----------------------------
    {
        void *pinned = nullptr;
        HIP(hipHostMalloc(&pinned, 4, hipHostMallocDefault));
        uint32_t v = 0xC0FFEEu;
        memcpy(pinned, &v, 4);
        HIP(hipMemcpyAsync(buf, pinned, 4, hipMemcpyDefault, stream));
        HIP(hipStreamSynchronize(stream));

        uint32_t back = 0;
        HIP(hipMemcpyAsync(&back, buf, 4, hipMemcpyDefault, stream));
        HIP(hipStreamSynchronize(stream));
        check(back == 0xC0FFEEu, "poke's staged pinned copy lands");
        HIP(hipHostFree(pinned));
    }

    // --- enqueue_host_func / flush_deferred_free -----------------------------
    //
    // The callback must run when the stream reaches it, which is what makes
    // deferred frees safe. If it never fires, freed allocations are never
    // recycled -- a leak that surfaces as an out-of-memory much later.
    {
        static int fired = 0;
        HIP(hipLaunchHostFunc(stream, [](void *p) { *(int *) p = 1; }, &fired));
        HIP(hipStreamSynchronize(stream));
        check(fired == 1, "hipLaunchHostFunc callback runs on stream progress");
    }

    // --- event record + stream wait ------------------------------------------
    HIP(hipEventRecord(event, stream));
    HIP(hipStreamWaitEvent(stream, event, 0));
    HIP(hipStreamSynchronize(stream));
    check(true, "hipEventRecord / hipStreamWaitEvent accepted");

    // --- async allocation ----------------------------------------------------
    {
        void *p = nullptr;
        HIP(hipMallocAsync(&p, 1024, stream));
        HIP(hipFreeAsync(p, stream));
        HIP(hipStreamSynchronize(stream));
        check(true, "hipMallocAsync / hipFreeAsync accepted");
    }

    HIP(hipFree(buf));
    HIP(hipEventDestroy(event));
    HIP(hipStreamDestroy(stream));
    HIP(hipCtxDestroy(ctx));

    if (failures) {
        printf("hipnv_ts_calls: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hipnv_ts_calls: every HIPThreadState call shape is correct.\n");
    return 0;
}
