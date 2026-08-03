// SPECIMEN: ray tracing via HIP-RT. gfx arm ONLY -- see the caveat below.
//
// Covers VarKind: TraceRay
//
// ############################################################################
// #  THIS KERNEL CANNOT BE EXECUTED HERE, AND THAT IS NOT A HARNESS GAP.     #
// ############################################################################
//
// hip_validate's execution arm is plain NVRTC with no HIP-RT linked, so there
// is nothing here to run a traversal against. `hiprt` in the source makes the
// harness link the HIP-RT device library and compile for gfx90a, and skip the
// NVIDIA arm.
//
// An earlier version of this note claimed traversal was unverifiable off AMD
// hardware because "HIP-RT ships AMD-only device libraries". That was true of
// the nixpkgs PACKAGE and false of HIP-RT, which supports NVIDIA through
// Orochi. flake.nix now builds a CUDA-enabled `hipRtNv`, and
// tools/hip_validate/hiprt_triangle.cpp builds a BVH and traverses it on the
// local NVIDIA GPU with verified hits (BACKEND_NOTES §7b). Traversal
// CORRECTNESS is covered there; this file covers the gfx90a compile and link.
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
// What remains UNVERIFIED until an MI210 is present: whether traversal returns
// correct hits ON gfx90a specifically -- wave64, and AMD's own traversal path.
// Structure confirmed here, correctness confirmed on NVIDIA by hiprt_triangle,
// and gfx90a takes HIP-RT's RTIP 0 SOFTWARE path (it has no RT hardware), which
// is what makes that proxy worth having rather than a curiosity.
//
// ---------------------------------------------------------------------------
//  The output contract
// ---------------------------------------------------------------------------
//
// Metal's jit_metal_ray_trace has 8 outputs (hit, t, u, v, instance_id,
// primitive_id, geometry_id, user_instance_id) and BACKEND_NOTES §7 proposes
// carrying that contract over verbatim. hiprtHit supplies five of them
// directly -- hasHit(), t, uv.x, uv.y, primID, instanceID -- and notably does
// NOT carry a geometry ID or a user instance ID.
//
// RESOLVED in Phase 5, and the contract did not have to change: both come from
// device tables indexed by hit.instanceID, supplied to jit_hip_configure_scene.
// That is exact rather than a workaround -- a HIP-RT instance references
// exactly one geometry, so "which geometry was hit" really is a property of the
// instance. So jit_hip_ray_trace IS jit_metal_ray_trace's signature.
//
// ---------------------------------------------------------------------------
//  This file vs. the emitted traversal
// ---------------------------------------------------------------------------
//
// This is a HAND-WRITTEN traversal, so it says nothing about what codegen
// produces. tests/hip_trace.cpp captures the real emitted source and
// run_tests.sh compiles THAT for gfx90a as well ("trace_closest" /
// "trace_shadow"). Keep both: this one is the cheap place to try a shape before
// teaching the emitter to produce it.

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
