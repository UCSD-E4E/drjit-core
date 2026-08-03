/*
    tests/hip_trace_exec.cpp -- a TRACED KERNEL, actually executed.

    tests/hip_trace.cpp checks what codegen emits and run_tests.sh links that
    text for gfx90a. Neither runs it. This does: it builds a real HIP-RT scene,
    hands it to jit_hip_configure_scene(), issues a trace through
    jit_hip_ray_trace(), evaluates, and checks the hits.

    That closes the last gap in Phase 5 that hardware was not actually required
    for. What makes it possible is that HIP-RT supports NVIDIA through Orochi
    (BACKEND_NOTES §7b) and jitc_hip_compile() now routes traversing kernels
    through hiprtBuildTraceKernels(), which compiles AND links them -- bare
    NVRTC cannot, because the generated source includes <hiprt/hiprt_device.h>.

    ------------------------------------------------------------------------
     Why NVIDIA is a fair proxy here
    ------------------------------------------------------------------------

    gfx90a has no ray-tracing hardware, so HIP-RT puts it on the RTIP 0
    software path -- portable C++ that NVIDIA runs too. This is very nearly the
    code the MI210 will execute. What it does NOT cover: wave64 (this runs at
    32) and the AMD host API.

    ------------------------------------------------------------------------
     The geometry
    ------------------------------------------------------------------------

    One triangle (0,0,0)-(1,0,0)-(0,1,0) in the z=0 plane, wrapped in a scene
    with a single identity-transformed instance -- a scene rather than a bare
    geometry because the emitter uses hiprtSceneTraversal*, which is what
    Mitsuba needs.

    Rays fire down -z from z=+1, one per lane, at x=y=(i+0.5)/N. Lanes whose
    x+y < 1 are inside the triangle and must hit at t=1; the rest must miss.
    Both outcomes are checked: a backend that always hits, always misses, or
    ignores the ray origin fails at least one of them. The hit DISTANCE is
    checked too, so a stub returning `hasHit()` alone does not pass.
*/

#include <drjit-core/jit.h>
#include <drjit-core/hip.h>
#include <hiprt/hiprt.h>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool cond, const char *what) {
    printf("  %-56s %s\n", what, cond ? "ok" : "FAIL");
    if (!cond)
        failures++;
}

#define RT(x)                                                                  \
    do {                                                                       \
        hiprtError e_ = (x);                                                   \
        if (e_ != hiprtSuccess) {                                              \
            printf("  %s -> %d\n", #x, (int) e_);                              \
            printf("hip_trace_exec: HIP-RT setup FAILED\n");                   \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int, char **) {
    printf("hip_trace_exec: traced kernel, executed\n");

    jit_init(1u << (uint32_t) JitBackend::HIP);

    if (!jit_has_backend(JitBackend::HIP)) {
        printf("  HIP backend unavailable -- build with -DDRJIT_HIP_CUDA_SHIM=ON.\n"
               "hip_trace_exec: SKIPPED\n");
        return 77;
    }

    // The HIP-RT context is built on drjit's own device context, so that the
    // scene and the kernel that traverses it live in the same one.
    hiprtContextCreationInput ci {};
    ci.ctxt = (hiprtApiCtx) jit_cuda_context();
    ci.device = (hiprtApiDevice) jit_cuda_device_raw();
    ci.deviceType = hiprtDeviceNVIDIA;

    hiprtContext hctx = nullptr;
    RT(hiprtCreateContext(HIPRT_API_VERSION, ci, hctx));

    // --- geometry: one triangle -------------------------------------------
    float verts[9]    = { 0,0,0,  1,0,0,  0,1,0 };
    uint32_t tris[3]  = { 0,1,2 };

    void *d_v = jit_malloc(JitBackend::HIP, sizeof(verts));
    void *d_i = jit_malloc(JitBackend::HIP, sizeof(tris));
    jit_memcpy(JitBackend::HIP, d_v, verts, sizeof(verts));
    jit_memcpy(JitBackend::HIP, d_i, tris, sizeof(tris));

    hiprtTriangleMeshPrimitive mesh {};
    mesh.vertexCount = 3;
    mesh.vertexStride = sizeof(float) * 3;
    mesh.vertices = d_v;
    mesh.triangleCount = 1;
    mesh.triangleStride = sizeof(uint32_t) * 3;
    mesh.triangleIndices = d_i;

    hiprtGeometryBuildInput gbi {};
    gbi.type = hiprtPrimitiveTypeTriangleMesh;
    gbi.primitive.triangleMesh = mesh;

    hiprtBuildOptions bo {};
    bo.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t tmp_size = 0;
    RT(hiprtGetGeometryBuildTemporaryBufferSize(hctx, gbi, bo, tmp_size));
    void *d_tmp = tmp_size ? jit_malloc(JitBackend::HIP, tmp_size) : nullptr;

    hiprtGeometry geom = nullptr;
    RT(hiprtCreateGeometry(hctx, gbi, bo, geom));
    RT(hiprtBuildGeometry(hctx, hiprtBuildOperationBuild, gbi, bo, d_tmp, 0, geom));

    // --- scene: one identity instance of that geometry ---------------------
    hiprtInstance inst {};
    inst.type = hiprtInstanceTypeGeometry;
    inst.geometry = geom;

    hiprtFrameSRT frame {};
    frame.rotation = { 0.f, 0.f, 1.f, 0.f };   // axis + angle, zero rotation
    frame.scale = { 1.f, 1.f, 1.f };
    frame.translation = { 0.f, 0.f, 0.f };
    frame.time = 0.f;

    void *d_inst  = jit_malloc(JitBackend::HIP, sizeof(inst));
    void *d_frame = jit_malloc(JitBackend::HIP, sizeof(frame));
    jit_memcpy(JitBackend::HIP, d_inst, &inst, sizeof(inst));
    jit_memcpy(JitBackend::HIP, d_frame, &frame, sizeof(frame));

    hiprtSceneBuildInput sbi {};
    sbi.instances = d_inst;
    sbi.instanceCount = 1;
    sbi.instanceFrames = d_frame;
    sbi.frameCount = 1;
    sbi.frameType = hiprtFrameTypeSRT;

    size_t stmp_size = 0;
    RT(hiprtGetSceneBuildTemporaryBufferSize(hctx, sbi, bo, stmp_size));
    void *d_stmp = stmp_size ? jit_malloc(JitBackend::HIP, stmp_size) : nullptr;

    hiprtScene scene_obj = nullptr;
    RT(hiprtCreateScene(hctx, sbi, bo, scene_obj));
    RT(hiprtBuildScene(hctx, hiprtBuildOperationBuild, sbi, bo, d_stmp, 0, scene_obj));
    printf("  BVH + scene built\n");

    // --- register it with drjit -------------------------------------------
    //
    // A geometry_ids table with a recognisable value, so the instance-indexed
    // lookup can be told apart from the zero default.
    constexpr uint32_t GEOM_ID_MARKER = 0x5a;
    uint32_t geom_ids_host[1] = { GEOM_ID_MARKER };
    void *d_geom_ids = jit_malloc(JitBackend::HIP, sizeof(geom_ids_host));
    jit_memcpy(JitBackend::HIP, d_geom_ids, geom_ids_host, sizeof(geom_ids_host));

    uint32_t scene = jit_hip_configure_scene(scene_obj, nullptr, d_geom_ids,
                                             nullptr, /* triangles */ 1);

    // --- trace one ray per lane -------------------------------------------
    constexpr uint32_t N = 64;

    uint32_t idx   = jit_var_counter(JitBackend::HIP, N);
    uint32_t idxf  = jit_var_cast(idx, VarType::Float32, 0);
    uint32_t half  = jit_var_f32(JitBackend::HIP, 0.5f);
    uint32_t invn  = jit_var_f32(JitBackend::HIP, 1.f / (float) N);

    // x = y = (i + 0.5) / N, so lanes sweep the unit square's diagonal.
    uint32_t add_d[2] = { idxf, half };
    uint32_t shifted  = jit_var_op(JitOp::Add, add_d);
    uint32_t mul_d[2] = { shifted, invn };
    uint32_t coord    = jit_var_op(JitOp::Mul, mul_d);

    uint32_t one   = jit_var_f32(JitBackend::HIP, 1.f);
    uint32_t zero  = jit_var_f32(JitBackend::HIP, 0.f);
    uint32_t neg1  = jit_var_f32(JitBackend::HIP, -1.f);
    uint32_t big   = jit_var_f32(JitBackend::HIP, 1000.f);
    uint32_t mask  = jit_var_bool(JitBackend::HIP, true);

    uint32_t args[8] = { coord, coord, one,    // origin  (x, y, 1)
                         zero,  zero,  neg1,   // direction (0, 0, -1)
                         zero,  big };         // tmin, tmax

    uint32_t out[8] = { 0 };
    jit_hip_ray_trace(8, args, mask, out, 8, scene, /* shadow = */ 0);

    // --- evaluate and check ------------------------------------------------
    jit_var_eval(out[0]);
    jit_var_eval(out[1]);
    jit_var_eval(out[6]);

    void *p_hit = nullptr, *p_t = nullptr, *p_gid = nullptr;
    uint32_t e0 = jit_var_data(out[0], &p_hit),
             e1 = jit_var_data(out[1], &p_t),
             e6 = jit_var_data(out[6], &p_gid);

    uint8_t  hit[N];
    float    t[N];
    uint32_t gid[N];
    jit_memcpy(JitBackend::HIP, hit, p_hit, sizeof(hit));
    jit_memcpy(JitBackend::HIP, t, p_t, sizeof(t));
    jit_memcpy(JitBackend::HIP, gid, p_gid, sizeof(gid));

    uint32_t n_hit = 0, n_miss = 0;
    bool hits_ok = true, t_ok = true, gid_ok = true;

    for (uint32_t i = 0; i < N; ++i) {
        float c = ((float) i + 0.5f) / (float) N;
        bool inside = (c + c) < 1.f;   // x + y < 1

        // Lanes right at the hypotenuse are a coin flip in floating point;
        // skip a narrow band rather than encode a tie-break the BVH does not
        // promise. With N = 64 this excludes at most one lane.
        if (fabsf(c + c - 1.f) < 1e-3f)
            continue;

        if ((hit[i] != 0) != inside)
            hits_ok = false;

        if (inside) {
            n_hit++;
            if (t[i] < 0.999f || t[i] > 1.001f)
                t_ok = false;
            if (gid[i] != GEOM_ID_MARKER)
                gid_ok = false;
        } else {
            n_miss++;
        }
    }

    printf("  %u lanes hit, %u missed\n", n_hit, n_miss);

    check(n_hit > 0 && n_miss > 0,
          "both hits and misses occur (the test discriminates)");
    check(hits_ok, "every lane hits exactly when it is inside the triangle");
    check(t_ok, "hit distance is 1.0 for every hit");
    check(gid_ok, "geometry_id came from the instance-indexed table");

    jit_var_dec_ref(e0); jit_var_dec_ref(e1); jit_var_dec_ref(e6);
    for (uint32_t i = 0; i < 8; ++i)
        jit_var_dec_ref(out[i]);
    jit_var_dec_ref(idx);  jit_var_dec_ref(idxf); jit_var_dec_ref(half);
    jit_var_dec_ref(invn); jit_var_dec_ref(shifted); jit_var_dec_ref(coord);
    jit_var_dec_ref(one);  jit_var_dec_ref(zero); jit_var_dec_ref(neg1);
    jit_var_dec_ref(big);  jit_var_dec_ref(mask); jit_var_dec_ref(scene);

    jit_shutdown(0);

    if (failures) {
        printf("hip_trace_exec: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("hip_trace_exec: traced kernel executes and returns correct hits.\n");
    return 0;
}
