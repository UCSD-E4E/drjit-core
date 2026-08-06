/*
    probe_wrong_sig.hip -- the NEGATIVE arm, and the reason to believe the
    positive one.

    Identical to probe_sphere.hip except that the intersection function takes
    the arguments in the order OptiX and Metal would suggest -- hit first, ray
    last -- instead of the order HIP-RT's generated dispatcher uses. Nothing
    else differs.

    Expected: hiprtBuildTraceKernels() FAILS.

    If this arm succeeds, then the build is not referencing our function at all
    and the positive arm's success means nothing. That is the whole point of
    shipping it.
*/

#include <hiprt/hiprt_device.h>

struct SphereData {
    float4 center_radius;   // xyz = center, w = radius
};

__device__ float mi_dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Wrong on purpose: argument order permuted relative to
// `( ray, data, payload, hit )`.
__device__ bool mi_isect_wrong(hiprtHit &hit, void *payload, const void *data,
                               const hiprtRay &ray) {
    const SphereData *spheres = (const SphereData *) data;
    SphereData s = spheres[hit.primID];
    float3 d = ray.direction;
    if (mi_dot(d, d) <= 0.0f)
        return false;
    hit.t = s.center_radius.w;
    hit.uv = make_float2(0.0f, 0.0f);
    return true;
}

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
