/*
    tests/hip_packing.cpp -- every place a JitBackend is squeezed into a bit
    field must have room for HIP.

    HIP is backend 4, so it is the first value that needs a THIRD bit. Two
    separate 2-bit fields have already truncated it, and neither failed loudly:

      * Variable::backend -- HIP became JitBackend::None, and the error came
        out as "the host backend is unavailable" during code generation.

      * AllocInfo -- HIP became None *and* corrupted the adjacent `shared` bit,
        so jitc_free() took the "shared allocation after shutdown" path and
        leaked every buffer. That surfaced as an out-of-memory in a reduction
        test, several suites away, with no leak warning to point at it.

    Both were found by chasing a symptom backwards. This test is cheap, pure,
    needs no device, and turns the next one into an immediate failure -- which
    matters because the next such field will be added by someone who has never
    heard of this file.
*/

#include "../src/malloc.h"
#include <drjit-core/jit.h>
#include <cstdio>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

int main(int, char **) {
    printf("hip_packing: backend values survive every packed representation\n");

    // The enum itself. If HIP ever stops being 4 the fields below are still
    // fine, but the comments above stop making sense.
    check((uint32_t) JitBackend::HIP == 4, "JitBackend::HIP == 4");
    check((uint32_t) JitBackend::Count <= 8,
          "JitBackend::Count fits in 3 bits");

    // --- AllocInfo ----------------------------------------------------------
    //
    // Round-trip EVERY backend, not just HIP: a field that is too narrow
    // corrupts its neighbours, so the interesting failures are in `shared` and
    // `size` rather than in the backend itself.
    for (uint32_t b = 0; b < (uint32_t) JitBackend::Count; ++b) {
        JitBackend backend = (JitBackend) b;

        for (int shared = 0; shared <= 1; ++shared) {
            size_t size = 1ull << 30;   // 1 GiB, well past the field boundary
            int device  = 7;

            AllocInfo info = alloc_info_encode(size, backend, shared != 0, device);
            auto [d_size, d_backend, d_shared, d_device] = alloc_info_decode(info);

            char msg[128];
            snprintf(msg, sizeof(msg),
                     "AllocInfo round-trip: backend=%u shared=%d",
                     b, shared);
            check(d_size == size && d_backend == backend &&
                  d_shared == (shared != 0) && d_device == device, msg);
        }
    }

    // A HIP allocation must NOT decode as shared. This is the exact corruption
    // that leaked: backend 4 shifted into the `shared` bit's position.
    {
        AllocInfo info = alloc_info_encode(64, JitBackend::HIP, /*shared=*/false, 0);
        auto [size, backend, shared, device] = alloc_info_decode(info);
        (void) size; (void) device;
        check(backend == JitBackend::HIP && !shared,
              "a non-shared HIP allocation does not decode as shared");
    }

    // Two backends must never collide in the map key -- the free cache is
    // keyed by this value, so a collision hands CUDA memory to HIP.
    {
        bool distinct = true;
        for (uint32_t i = 0; i < (uint32_t) JitBackend::Count; ++i)
            for (uint32_t j = i + 1; j < (uint32_t) JitBackend::Count; ++j)
                if (alloc_info_encode(4096, (JitBackend) i, false, 0) ==
                    alloc_info_encode(4096, (JitBackend) j, false, 0))
                    distinct = false;
        check(distinct, "distinct backends produce distinct AllocInfo keys");
    }

    if (failures) {
        printf("hip_packing: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_packing: all packed backend fields have room for HIP.\n");
    return 0;
}
