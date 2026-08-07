/*
    src/hip_scene.cpp -- scene registration and TraceRay node construction

    Mirrors the MetalScene half of metal_core.mm. The differences are all
    subtractions: a hiprtScene is a plain device pointer, so there is no
    resource-handle reconstruction, no per-launch residency list, and no
    intersection-function-table object to retain (BACKEND_NOTES §6). What is
    added instead are the two instance-indexed tables that stand in for the hit
    fields HIP-RT does not carry -- see hip_scene.h for why that substitution is
    exact rather than approximate.
*/

#include "hip_scene.h"

#if defined(DRJIT_ENABLE_HIP)

#include "internal.h"
#include "var.h"
#include "log.h"
#include "trace.h"
#include "op.h"

HIPScene *jitc_hip_get_scene(uint32_t scene_index) {
    Variable *v = scene_index ? jitc_var(scene_index) : nullptr;
    if (!v || (VarKind) v->kind != VarKind::Nop ||
        (VarType) v->type != VarType::Void)
        jitc_raise("jit_hip_ray_trace(): r%u is not a HIP scene handle "
                   "(expected the value returned by "
                   "jit_hip_configure_scene()).", scene_index);
    return (HIPScene *) v->literal;
}

uint32_t jitc_hip_configure_scene(void *scene, void *func_table,
                                  const void *geometry_ids,
                                  const void *user_instance_ids,
                                  uint32_t geometry_types_mask) {
    if (!scene)
        jitc_raise("jit_hip_configure_scene(): a built hiprtScene is required.");

    jitc_log(InfoSym,
             "jit_hip_configure_scene(scene=" DRJIT_PTR ", func_table=" DRJIT_PTR
             ", geometry_ids=" DRJIT_PTR ", user_instance_ids=" DRJIT_PTR
             ", geom_mask=%u)",
             (uintptr_t) scene, (uintptr_t) func_table,
             (uintptr_t) geometry_ids, (uintptr_t) user_instance_ids,
             geometry_types_mask);

    // A fresh scene per configuration, as on Metal: a geometry edit registers
    // another one rather than mutating this, so kernels already recorded
    // against the old scene stay valid.
    HIPScene *s = new HIPScene();
    s->scene = scene;
    s->func_table = func_table;
    s->geometry_ids = geometry_ids;
    s->user_instance_ids = user_instance_ids;
    s->geometry_types_mask = geometry_types_mask;

    uint32_t index = jitc_var_new_node_0(JitBackend::HIP, VarKind::Nop,
                                         VarType::Void, 1, 0, (uintptr_t) s);

    jitc_var_set_callback(
        index,
        [](uint32_t, int free, void *ptr) {
            if (!free)
                return;
            HIPScene *sc = (HIPScene *) ptr;
            jitc_log(InfoSym,
                     "jit_hip_configure_scene(): freeing HIPScene (scene="
                     DRJIT_PTR ")", (uintptr_t) sc->scene);

            jitc_var_dec_ref(sc->scene_handle);
            jitc_var_dec_ref(sc->func_table_handle);
            jitc_var_dec_ref(sc->geometry_ids_handle);
            jitc_var_dec_ref(sc->user_instance_ids_handle);

            void (*cleanup)(void *) = sc->cleanup;
            void *payload = sc->cleanup_payload;
            delete sc;

            // Only now may the application release the HIP-RT objects: an
            // unevaluated kernel could still have been holding this scene.
            if (cleanup)
                cleanup(payload);
        },
        s, true);

    return index;
}

uint32_t jitc_hip_scene_owner_handle(uint32_t scene_index) {
    HIPScene *s = jitc_hip_get_scene(scene_index);
    /* An identity token for dr.freeze, not an owning reference: the pointer is
       drjit-core's own bookkeeping object, which outlives any single launch.

       MUST be a UInt64 variable rather than a Pointer one. dr.freeze REJECTS
       pointer-typed inputs outright ("Pointer inputs not supported!",
       src/python/freeze.cpp), so jitc_var_pointer() here defeats the very
       thing this handle exists for -- and does it at the point where a frozen
       function is first called, far from this line. This mirrors
       jitc_metal_scene_owner_handle(), which is the template
       scene_hip.inl follows; the data pointer is the HIPScene owner, the same
       pointer carried by jit_hip_ray_trace's scene parameter, so the recorder
       keys both to one input slot. */
    return jitc_var_mem_map(JitBackend::HIP, VarType::UInt64, (void *) s, 1,
                            /* free = */ 0);
}

/// Cache a pointer variable for one of the scene's device resources.
///
/// Cached rather than rebuilt per trace so that several traces against one
/// scene share a single kernel parameter -- the parameter block is a fixed
/// budget and a path tracer issues many traces against one scene.
///
/// EACH RESOURCE NEEDS A DEP, and it may not be the scene variable.
///
/// The frozen-function recorder resolves a resource parameter by following the
/// pointer variable's dep[3] and checking that the target's `data` is the same
/// address (RecordThreadState::launch, record_ts.cpp):
///
///     index = v->dep[3];  v = jitc_var(index);
///     if (v->data != ptr) jitc_fail("... memory address did not match!");
///
/// So `dep = 0` -- which this used to pass -- makes a traced scene impossible
/// to record, and the failure surfaces as an abort inside a frozen render with
/// no mention of scenes or dependencies.
///
/// The dep may NOT be the scene variable: jitc_var_pointer() takes a reference
/// on its dep, the scene owns these handles, and the cycle would keep the
/// refcount off zero forever -- the cleanup callback would never fire and the
/// application's HIP-RT objects would leak.
///
/// Both constraints are satisfied by a non-owning memory map over the resource
/// itself: its `data` is the pointer by construction, and it references no
/// scene variable, so there is no cycle. `free = 0` because a hiprtScene and
/// the ID tables are the application's memory, not drjit allocations -- their
/// lifetime is what the cleanup callback manages. This is Metal's arrangement
/// (jitc_metal_scene_owner_handle) reached from the other direction.
static uint32_t hip_scene_handle(uint32_t &slot, const void *ptr) {
    if (!ptr)
        return 0;
    if (!slot) {
        uint32_t owner = jitc_var_mem_map(JitBackend::HIP, VarType::UInt64,
                                          (void *) ptr, 1, /* free = */ 0);
        slot = jitc_var_pointer(JitBackend::HIP, ptr, owner, 0);
        // jitc_var_pointer() took its own reference on `owner`; drop ours so
        // the map dies with the pointer variable rather than outliving it.
        jitc_var_dec_ref(owner);
    }
    return slot;
}

void jitc_hip_ray_trace(uint32_t n_args, uint32_t *args, uint32_t mask,
                        uint32_t *out, uint32_t n_out, uint32_t scene,
                        int shadow) {
    if (n_args != 8)
        jitc_raise("jit_hip_ray_trace(): expected 8 ray arguments, got %u.",
                   n_args);
    if (n_out != 8)
        jitc_raise("jit_hip_ray_trace(): expected 8 outputs, got %u.", n_out);
    if (!scene)
        jitc_raise("jit_hip_ray_trace(): a valid scene index (returned by "
                   "jit_hip_configure_scene) is required.");

    HIPScene *s = jitc_hip_get_scene(scene);

    uint32_t size = 0;
    bool symbolic = false;
    for (uint32_t i = 0; i < n_args; ++i) {
        const Variable *vi = jitc_var(args[i]);
        size = std::max(size, vi->size);
        symbolic |= (bool) vi->symbolic;
    }
    {
        const Variable *vm = jitc_var(mask);
        size = std::max(size, vm->size);
        symbolic |= (bool) vm->symbolic;
    }
    if (size == 0)
        size = 1;

    Ref valid = steal(jitc_var_mask_apply(mask, size));

    TraceData *td = new TraceData();
    td->shadow = shadow != 0;
    td->indices.reserve(n_args + 4);
    for (uint32_t i = 0; i < n_args; ++i) {
        td->indices.push_back(args[i]);
        jitc_var_inc_ref(args[i]);
    }

    // The device pointers this trace reads. They live in `indices` purely so
    // that jitc_var_traverse() schedules them and allocates their parameter
    // slots (see eval.cpp's TraceRay case); the emitter recovers WHICH is which
    // from HIPScene rather than from a position here, so absent ones are simply
    // not pushed and there are no placeholder entries to misread.
    uint32_t handles[4] = {
        hip_scene_handle(s->scene_handle, s->scene),
        hip_scene_handle(s->func_table_handle, s->func_table),
        hip_scene_handle(s->geometry_ids_handle, s->geometry_ids),
        hip_scene_handle(s->user_instance_ids_handle, s->user_instance_ids)
    };
    for (uint32_t h : handles) {
        if (h) {
            td->indices.push_back(h);
            jitc_var_inc_ref(h);
        }
    }

    // dep[0] = valid mask, dep[1] = the scene variable. The latter both keeps
    // HIPScene alive for the node's lifetime and is how the emitter finds it.
    Ref trace = steal(jitc_var_new_node_2(
        JitBackend::HIP, VarKind::TraceRay, VarType::Void, size, symbolic,
        valid, jitc_var(valid), scene, jitc_var(scene), (uintptr_t) td));

    jitc_var_set_callback(
        trace,
        [](uint32_t, int free, void *ptr) {
            if (free)
                delete (TraceData *) ptr;
        },
        td, true);

    static const VarType out_types[8] = {
        VarType::Bool,    // valid
        VarType::Float32, // distance
        VarType::Float32, // bary_u
        VarType::Float32, // bary_v
        VarType::UInt32,  // instance_id
        VarType::UInt32,  // primitive_id
        VarType::UInt32,  // geometry_id      (instance-indexed table)
        VarType::UInt32   // user_instance_id (instance-indexed table)
    };

    for (uint32_t i = 0; i < (td->shadow ? 1u : 8u); ++i)
        out[i] = jitc_var_new_node_1(JitBackend::HIP, VarKind::Extract,
                                     out_types[i], size, symbolic, trace,
                                     jitc_var(trace), (uint64_t) i);
}

#endif // DRJIT_ENABLE_HIP
