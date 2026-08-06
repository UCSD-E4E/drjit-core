/*
    probe_sphere.hip -- the POSITIVE arm.

    A sphere intersector written against the signature recovered from
    libhiprt0300064.so, plus a kernel shaped like the one drjit-core emits for
    a scene traversal with a function table (hip_eval.cpp::render_trace_ray).

    Expected: hiprtBuildTraceKernels() returns hiprtSuccess.

    The math is a transcription of intersection_sphere() in
    src/render/metal/intersection_functions.metal:86, which is itself a
    transcription of optix/sphere.cuh. It is here so the probe exercises a
    realistic body rather than `return true`, and so the arithmetic gets a
    compile before it gets a render.
*/

#include <hiprt/hiprt_device.h>

// Matches shapedata::SphereData in the Mitsuba tree. This file is the paired
// positive control for probe_wrong_sig.hip -- the two must differ ONLY in the
// signature under test -- so it keeps its own copy rather than sharing one.
struct SphereData {
    float4 center_radius;   // xyz = center, w = radius
};

// Numerically stable quadratic solver; ordered roots x0 <= x1.
// Mirrors optix/math.cuh::solve_quadratic.
__device__ bool mi_solve_quadratic(float a, float b, float c,
                                   float &x0, float &x1) {
    bool linear_case = (a == 0.0f);
    if (linear_case && b == 0.0f)
        return false;

    x0 = x1 = -c / b;
    if (linear_case)
        return true;

    float discrim = fmaf(b, b, -4.0f * a * c);
    if (discrim < 0.0f)
        return false;

    float temp = -0.5f * (b + copysignf(sqrtf(discrim), b));
    float x0p = temp / a;
    float x1p = c / temp;
    x0 = fminf(x0p, x1p);
    x1 = fmaxf(x0p, x1p);
    return true;
}

__device__ float mi_dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*
    THE SIGNATURE UNDER TEST.

    HIP-RT generates, inside hiprtBuildTraceKernels():

        __device__ bool mi_isect_sphere( const hiprtRay& ray, const void* data,
                                         void* payload, hiprtHit& hit );
        ...
            case 0: { return mi_isect_sphere( ray, data, payload, hit ); }

    `hit.primID` and `hit.instanceID` are both already populated on entry
    (hiprt_device_impl.h:1032 and :1043), and `ray` is in object space -- the
    instance transform has been applied by transformRay() further up. Writing
    `hit.t`, `hit.uv` and `hit.normal` and returning true accepts the hit.
*/
__device__ bool mi_isect_sphere(const hiprtRay &ray, const void *data,
                                void *payload, hiprtHit &hit) {
    const SphereData *spheres = (const SphereData *) data;
    SphereData s = spheres[hit.primID];

    float3 center = make_float3(s.center_radius.x, s.center_radius.y,
                                s.center_radius.z);
    float  radius = s.center_radius.w;

    float3 d = ray.direction;
    float3 l = make_float3(ray.origin.x - center.x, ray.origin.y - center.y,
                           ray.origin.z - center.z);

    // Project onto the perpendicular plane first (see optix/sphere.cuh): the
    // quadratic is conditioned far better about that point than about the ray
    // origin when the origin is distant.
    float plane_t = -mi_dot(l, d) / sqrtf(mi_dot(d, d));
    float3 plane_p = make_float3(ray.origin.x + plane_t * d.x,
                                 ray.origin.y + plane_t * d.y,
                                 ray.origin.z + plane_t * d.z);

    float3 pc = make_float3(plane_p.x - center.x, plane_p.y - center.y,
                            plane_p.z - center.z);
    if (plane_t == 0.0f && sqrtf(mi_dot(pc, pc)) > radius)
        return false;

    float A = mi_dot(d, d);
    float B = 2.0f * mi_dot(pc, d);
    float C = mi_dot(pc, pc) - radius * radius;

    float near_t, far_t;
    bool ok = mi_solve_quadratic(A, B, C, near_t, far_t);

    near_t += plane_t;
    far_t  += plane_t;

    bool out_bounds = !(near_t <= ray.maxT && far_t >= 0.0f);
    bool in_bounds  = near_t < ray.minT && far_t > ray.maxT;

    float t = (near_t < 0.0f ? far_t : near_t);

    if (!ok || out_bounds || in_bounds || t < ray.minT || t > ray.maxT)
        return false;

    hit.t = t;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}

// A traversal kernel shaped like drjit-core's, so the probe builds the same
// shape of translation unit the backend does.
extern "C" __global__ void probe_kernel(hiprtScene scene,
                                        hiprtFuncTable table,
                                        const float *rays, float *out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    hiprtRay _r;
    _r.origin    = { rays[i * 8 + 0], rays[i * 8 + 1], rays[i * 8 + 2] };
    _r.direction = { rays[i * 8 + 3], rays[i * 8 + 4], rays[i * 8 + 5] };
    _r.minT      = rays[i * 8 + 6];
    _r.maxT      = rays[i * 8 + 7];

    hiprtSceneTraversalClosest _tr(scene, _r, hiprtFullRayMask,
                                   hiprtTraversalHintDefault, nullptr, table);
    hiprtHit _h = _tr.getNextHit();
    out[i] = _h.hasHit() ? _h.t : -1.0f;
}
