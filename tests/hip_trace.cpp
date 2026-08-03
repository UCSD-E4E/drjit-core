/*
    tests/hip_trace.cpp -- what does the ray-tracing emitter actually produce?

    Phase 5 emits HIP-RT traversal into generated kernels. Two things can go
    wrong that nothing else in the suite would catch:

      1. The emitted source does not compile or LINK for gfx90a. HIP-RT
         *declares* intersectFunc and filterFunc and leaves the definitions to
         the application, so a kernel that traverses without them fails to link
         with `undefined hidden symbol` (BACKEND_NOTES §7a). A backend that
         forgets them produces source that looks perfect and never runs.

      2. The traversal is wired to the wrong thing -- the closest-hit object for
         a shadow ray, the wrong hit field on an output, a missing table lookup
         for the two outputs hiprtHit does not carry.

    This test captures the REAL emitted text -- not a hand-mirrored copy -- and
    asserts (2) directly, then writes it out so run_tests.sh can hand it to
    hip_validate, which compiles and links it for real gfx90a and so covers (1).

    ------------------------------------------------------------------------
     Why this exits from inside a log callback
    ------------------------------------------------------------------------

    Codegen dumps the assembled source through the log callback under
    JitFlag::PrintIR, and does so BEFORE handing it to the compiler. Under the
    CUDA shim that compiler is NVRTC, which has no HIP-RT headers and so fails
    -- via jitc_fail(), which aborts the process rather than throwing. There is
    no point after eval at which this test could run its assertions.

    So the assertions run in the callback, at the only moment the source exists
    and the process is still alive, and it exits there with a meaningful status.
    That is unusual enough to be mistaken for a bug, hence this comment. It is
    not a workaround for a missing feature: the compile genuinely cannot
    succeed until the shim links the HIP-RT device library, which is the next
    piece of Phase 5.
*/

#include <drjit-core/jit.h>
#include <drjit-core/hip.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int failures = 0;
static bool shadow_mode = false;

static void check(bool cond, const char *what) {
    printf("  %-56s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

static bool has(const std::string &s, const char *needle) {
    return s.find(needle) != std::string::npos;
}

static void inspect(const char *ir) {
    std::string s(ir);

    // The hooks HIP-RT requires at link time. Their absence is the failure
    // mode that never shows up as a compile error in this process.
    check(has(s, "#include <hiprt/hiprt_device.h>"),
          "emits the HIP-RT device header");
    check(has(s, "__device__ bool intersectFunc("),
          "defines intersectFunc (HIP-RT declares, we must define)");
    check(has(s, "__device__ bool filterFunc("),
          "defines filterFunc");

    // Traversal kind follows the shadow flag: any-hit terminates at the first
    // intersection, which is the only reason the flag is worth carrying.
    if (shadow_mode) {
        check(has(s, "hiprtSceneTraversalAnyHit"),
              "shadow ray uses the any-hit traversal");
        check(!has(s, "hiprtSceneTraversalClosest"),
              "shadow ray does not also emit closest-hit");
    } else {
        check(has(s, "hiprtSceneTraversalClosest"),
              "closest-hit ray uses the closest-hit traversal");
        check(!has(s, "hiprtSceneTraversalAnyHit"),
              "closest-hit ray does not also emit any-hit");
    }

    if (!shadow_mode) {
        // Miss defaults, so masked and missed lanes need no separate clearing.
        // Only meaningful for a closest-hit ray: a shadow ray has no distance
        // output to default, which is the whole point of the flag.
        check(has(s, "__uint_as_float(0x7f800000u)"),
              "distance defaults to +infinity on miss");

        check(has(s, "_h.t") && has(s, "_h.uv.x") && has(s, "_h.uv.y") &&
                  has(s, "_h.instanceID") && has(s, "_h.primID"),
              "reads the five hit fields hiprtHit carries");

        // The two it does not carry, resolved through instance-indexed tables.
        check(has(s, "[_h.instanceID]"),
              "geometry_id / user_instance_id come from instance-indexed tables");
    } else {
        check(!has(s, "_h.t"),
              "shadow ray computes only the hit flag");
    }

    // Write the artifact for the gfx arm. This is the point of the exercise:
    // shape assertions here, and a real gfx90a compile+link over there.
    const char *path = getenv("DRJIT_HIP_TRACE_OUT");
    if (path) {
        FILE *f = fopen(path, "w");
        if (f) {
            fputs(ir, f);
            fclose(f);
            printf("  wrote emitted source to %s\n", path);
        } else {
            printf("  could not write %s\n", path);
            failures++;
        }
    }

    printf(failures ? "hip_trace: %d check(s) FAILED\n"
                    : "hip_trace: emitted traversal verified\n", failures);
    fflush(stdout);

    // See the header comment: the compile that follows cannot succeed and
    // aborts, so this is the last moment at which a status can be reported.
    _Exit(failures ? 1 : 0);
}

static void log_callback(LogLevel, const char *msg) {
    // Codegen sends the whole assembled kernel through here in one call. Other
    // log traffic is much shorter and never contains a kernel signature.
    if (msg && strstr(msg, "drjit_"))
        inspect(msg);
}

int main(int argc, char **argv) {
    shadow_mode = argc > 1 && strcmp(argv[1], "shadow") == 0;

    printf("hip_trace: %s ray\n", shadow_mode ? "shadow" : "closest-hit");

    jit_init(1u << (uint32_t) JitBackend::HIP);

    if (!jit_has_backend(JitBackend::HIP)) {
        printf("  HIP backend unavailable -- build with -DDRJIT_HIP_CUDA_SHIM=ON.\n"
               "hip_trace: SKIPPED\n");
        return 77;
    }

    constexpr uint32_t N = 64, N_INST = 4;

    // ---- Scene lifetime ---------------------------------------------------
    //
    // Run first, because it is the check most likely to be broken silently.
    // The application drops its scene reference and expects the cleanup
    // callback to fire, which is its signal that the HIP-RT objects can be
    // freed. A reference CYCLE anywhere in the scene's own bookkeeping stops
    // that happening: nothing errors, nothing leaks visibly at exit, the
    // callback simply never runs and the application holds the scene forever.
    //
    // jitc_var_pointer() takes a reference on its `dep` argument, so caching a
    // pointer handle on the scene with the scene as its dep does exactly that.
    {
        static bool freed = false;
        void *dummy = jit_malloc(JitBackend::HIP, 16);
        uint32_t sc = jit_hip_configure_scene(dummy, nullptr, nullptr, nullptr, 1);

        jit_hip_scene_set_cleanup(sc, [](void *) { freed = true; }, nullptr);

        // Issue a trace so the scene creates and caches its pointer handles --
        // an unused scene would not exercise the cycle at all.
        uint32_t z = jit_var_f32(JitBackend::HIP, 0.f);
        uint32_t ray[8] = { z, z, z, z, z, z, z, z };
        uint32_t m = jit_var_bool(JitBackend::HIP, true);
        uint32_t o[8] = { 0 };
        jit_hip_ray_trace(8, ray, m, o, 8, sc, 0);

        for (uint32_t i = 0; i < 8; ++i)
            jit_var_dec_ref(o[i]);
        jit_var_dec_ref(z);
        jit_var_dec_ref(m);
        jit_var_dec_ref(sc);

        check(freed, "dropping the scene reference runs the cleanup callback");
    }

    // Stand-ins for the application's HIP-RT objects. Nothing dereferences them
    // during codegen -- the scene pointer becomes a kernel parameter and the
    // tables become gathers -- and the kernel is never launched.
    void *fake_scene = jit_malloc(JitBackend::HIP, 16);
    void *geom_ids   = jit_malloc(JitBackend::HIP, N_INST * sizeof(uint32_t));
    void *user_ids   = jit_malloc(JitBackend::HIP, N_INST * sizeof(uint32_t));

    uint32_t scene = jit_hip_configure_scene(fake_scene, /* func_table = */ nullptr,
                                             geom_ids, user_ids,
                                             /* geometry_types_mask = */ 1);

    // A ray per lane, fanning out along X so nothing folds to a constant.
    uint32_t idx  = jit_var_counter(JitBackend::HIP, N);
    uint32_t idxf = jit_var_cast(idx, VarType::Float32, 0);
    uint32_t zero = jit_var_f32(JitBackend::HIP, 0.f);
    uint32_t one  = jit_var_f32(JitBackend::HIP, 1.f);
    uint32_t big  = jit_var_f32(JitBackend::HIP, 1e30f);

    uint32_t args[8] = { idxf, zero, zero,   // origin
                         zero, zero, one,    // direction
                         zero, big };        // tmin, tmax
    uint32_t mask = jit_var_bool(JitBackend::HIP, true);

    uint32_t out[8] = { 0 };
    jit_hip_ray_trace(8, args, mask, out, 8, scene, shadow_mode ? 1 : 0);

    check(out[0] != 0, "trace produced a validity output");
    if (!shadow_mode)
        check(out[7] != 0, "trace produced all eight outputs");

    // Force assembly. PrintIR routes the source through log_callback(), which
    // does the real work and exits.
    jit_set_flag(JitFlag::PrintIR, 1);
    jit_set_log_level_callback(LogLevel::Info, log_callback);

    // Consume an output so the trace is not dead code.
    jit_var_eval(out[0]);

    // Only reached if codegen never emitted -- e.g. the trace was optimised
    // away, which would make every assertion above vacuous.
    printf("hip_trace: FAILED -- codegen never emitted a kernel\n");
    return 1;
}
