// SPECIMEN: ray tracing via HIP-RT. gfx arm ONLY -- see the caveat below.
//
// Covers VarKind: TraceRay
//
// ############################################################################
// #  THIS KERNEL CANNOT BE EXECUTED HERE, AND THAT IS NOT A HARNESS GAP.     #
// ############################################################################
//
// HIP-RT as packaged ships device libraries for AMD targets only
// (hiprt*_amd_lib_linux.bc, hiprt*_amd.hipfb) and its host library dlopens
// libamdhip64 / libhiprtc with no CUDA driver paths. The API has a
// hiprtDeviceNVIDIA enum value, but this build cannot service it. So unlike
// every other specimen, there is no execution arm: `hiprt` in the source makes
// hip_validate link the HIP-RT device library and compile for gfx90a, and the
// NVIDIA arm is skipped.
//
// What that DOES establish, and it is the bulk of the Phase 0b risk:
//
//   1. HIP-RT's device header compiles for gfx90a with our toolchain.
//   2. The shipped device library really contains a gfx90a slice, and it LINKS
//      -- i.e. the traversal implementation exists for our exact target rather
//      than being a claim on a support matrix.
//   3. The emitted traversal's register and scratch footprint, which the
//      harness prints. That is the only occupancy signal available without the
//      card, and traversal is where it matters: HIP-RT keeps its stack in
//      scratch, so a shape change that doubles it still compiles and still
//      runs, and shows up as halved throughput on the MI210 much later.
//
// What remains UNVERIFIABLE until an MI210 is present: whether traversal
// returns correct hits. Treat this file the way run_tests.sh treats the wave64
// list -- structure confirmed, semantics pending hardware.
//
// ---------------------------------------------------------------------------
//  The output contract
// ---------------------------------------------------------------------------
//
// Metal's jit_metal_ray_trace has 8 outputs (hit, t, u, v, instance_id,
// primitive_id, geometry_id, user_instance_id) and BACKEND_NOTES §7 proposes
// carrying that contract over verbatim. hiprtHit supplies five of them
// directly -- hasHit(), t, uv.x, uv.y, primID, instanceID -- and notably does
// NOT carry a geometry ID or a user instance ID. Those two have to come from
// somewhere else (an application-side table indexed by instanceID, most
// likely). Recorded here because it is a contract difference, not an
// implementation detail, and it is cheaper to discover now than mid-port.

#include <hip/hip_runtime.h>
#include <hiprt/hiprt_device.h>

// HIP-RT DECLARES these two and requires the application to DEFINE them; they
// are the custom-primitive intersection and any-hit filter hooks, the analogue
// of Metal's intersection function table and OptiX's IS/AH programs.
//
// This is a hard requirement on the emitter, not an optional extra: without
// them the link fails with `undefined hidden symbol: intersectFunc(...)`. Every
// generated kernel that traverses must carry at least these stubs, and a scene
// with custom primitives must dispatch here rather than return false.
__device__ bool intersectFunc(uint32_t /*geomType*/, uint32_t /*rayType*/,
                              const hiprtFuncTableHeader & /*table*/,
                              const hiprtRay & /*ray*/, void * /*payload*/,
                              hiprtHit & /*hit*/) {
    return false;
}

__device__ bool filterFunc(uint32_t /*geomType*/, uint32_t /*rayType*/,
                           const hiprtFuncTableHeader & /*table*/,
                           const hiprtRay & /*ray*/, void * /*payload*/,
                           const hiprtHit & /*hit*/) {
    return false;
}

// The harness's (out, in, n) convention plus the scene handle. In generated
// code the handle arrives through params.args[] as an ordinary pointer -- HIP-RT
// scene handles are plain pointers, which is why BACKEND_NOTES §6 expects the
// resource-handle machinery to be far smaller than Metal's.
DRJIT_KERNEL void k(float *out, const float *in, unsigned n) {
    uint32_t r0 = DRJIT_TID;
    if (r0 >= n) return;

    // The scene handle. Null here: this kernel is never launched, and a null
    // handle keeps the traversal object constructible for the compiler while
    // making it obvious that no scene is being described.
    hiprtScene scene = (hiprtScene) nullptr;

    // --- Outputs, pre-set to MISS -------------------------------------------
    //
    // Metal's trick (§7), carried over: declare all outputs with miss values so
    // masked-off and missed lanes fall through with correct results and the hit
    // path only ever overwrites. t = +inf via the bit pattern rather than a
    // literal, so no float parsing is involved.
    bool     o_hit  = false;
    float    o_t    = __uint_as_float(0x7f800000u);
    float    o_u    = 0.0f;
    float    o_v    = 0.0f;
    uint32_t o_inst = 0u;
    uint32_t o_prim = 0u;

    // --- The mask -----------------------------------------------------------
    //
    // Emitted as a guard around the whole traversal, never as a select on the
    // result: an unmasked traversal against a null or wrong scene is not a
    // wasted computation, it is a fault.
    bool mask = (in[r0] >= 0.0f);

    if (mask) {
        hiprtRay ray;
        ray.origin    = { in[r0], 0.0f, -1.0f };
        ray.minT      = 0.0f;
        ray.direction = { 0.0f, 0.0f, 1.0f };
        ray.maxT      = __uint_as_float(0x7f800000u);

        // Closest-hit traversal over a scene (instances). The geometry-level
        // object is hiprtGeomTraversalClosest; shadow rays use the AnyHit
        // variants, which is how §7's `shadow` flag lowers.
        hiprtSceneTraversalClosest tr(scene, ray);
        hiprtHit hit = tr.getNextHit();

        if (hit.hasHit()) {
            o_hit  = true;
            o_t    = hit.t;
            o_u    = hit.uv.x;
            o_v    = hit.uv.y;
            o_inst = hit.instanceID;
            o_prim = hit.primID;
        }
    }

    // Consume every output so none is optimised away -- the register and
    // scratch numbers the harness prints are only meaningful if the traversal
    // and its results actually survive to here.
    out[r0] = (o_hit ? 1.0f : 0.0f) + o_t + o_u + o_v +
              (float) o_inst + (float) o_prim;
}
