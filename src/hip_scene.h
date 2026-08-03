/*
    src/hip_scene.h -- per-scene state for HIP-RT ray tracing

    The HIP counterpart of MetalScene, and much smaller. Metal's version carries
    a TLAS resource ID reconstructed in-shader, a list of child resources to
    make resident at every launch, a retained MTLLibrary of intersection
    functions, per-entry function names, IFT buffer bindings and a PSO cache.
    None of that has a HIP-RT analogue: a hiprtScene is a plain device pointer
    that travels in the kernel parameter block, and custom-primitive dispatch
    goes through a single hiprtFuncTable rather than a table of named MSL
    functions (BACKEND_NOTES §6).

    What HIP-RT does NOT give us, and Metal does, is two of the eight hit
    outputs. `hiprtHit` carries instanceID, primID, uv, normal and t -- no
    geometry ID and no user instance ID. The two tables below supply them,
    indexed by the hit's instance ID.

    That indexing is exact rather than approximate, which is why the eight-output
    contract survives unchanged: in HIP-RT an instance references exactly ONE
    geometry, so "which geometry was hit" really is a property of the instance.
    Metal needs a per-hit field because its acceleration structures nest several
    geometries inside one instance. The difference is in the scene model, not in
    the fidelity of the answer.
*/

#pragma once

#include "internal.h"

#if defined(DRJIT_ENABLE_HIP)

struct HIPScene {
    /// hiprtScene. A plain device pointer; not owned.
    void *scene = nullptr;

    /// Optional hiprtFuncTable for custom-primitive intersection / any-hit
    /// filtering. Not owned.
    void *func_table = nullptr;

    /// Optional device uint32_t[], indexed by hit.instanceID -> output 6.
    const void *geometry_ids = nullptr;

    /// Optional device uint32_t[], indexed by hit.instanceID -> output 7.
    const void *user_instance_ids = nullptr;

    /// Bit 0 = triangle, 1 = bounding box, 2 = curves, 3 = backface culling.
    /// Same encoding as Metal, so Mitsuba's ShapeIR::Kind maps onto both.
    uint32_t geometry_types_mask = 0;

    /// Pointer variables handed to the emitter, created lazily and cached so
    /// that repeated traces against one scene share a single kernel parameter.
    uint32_t scene_handle = 0;
    uint32_t func_table_handle = 0;
    uint32_t geometry_ids_handle = 0;
    uint32_t user_instance_ids_handle = 0;

    /// Runs after drjit-core releases this object, so the application can free
    /// the HIP-RT objects by dropping its reference rather than directly.
    void (*cleanup)(void *) = nullptr;
    void *cleanup_payload = nullptr;
};

/// Recover the HIPScene behind a scene variable, or fail loudly.
extern HIPScene *jitc_hip_get_scene(uint32_t scene_index);

extern uint32_t jitc_hip_configure_scene(void *scene, void *func_table,
                                         const void *geometry_ids,
                                         const void *user_instance_ids,
                                         uint32_t geometry_types_mask);

extern void jitc_hip_ray_trace(uint32_t n_args, uint32_t *args, uint32_t mask,
                               uint32_t *out, uint32_t n_out, uint32_t scene,
                               int shadow);

extern uint32_t jitc_hip_scene_owner_handle(uint32_t scene_index);

#endif // DRJIT_ENABLE_HIP
