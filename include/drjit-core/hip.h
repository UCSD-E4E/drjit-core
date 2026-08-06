/*
    drjit-core/hip.h -- HIP (AMD ROCm) ray-tracing interface

    Deliberately a near-copy of drjit-core/metal.h rather than of optix.h.
    PLAN.md §3.2 explains why: OptiX's pipeline + SBT + callable-program model
    restructures the whole kernel and accounts for most of drjit-core's 1,922
    OptiX lines and Mitsuba's 818-line scene_optix.inl. Metal's inline
    intersector is the shape HIP-RT has, so Layer C can share the Metal path
    almost verbatim.

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#pragma once

#include <drjit-core/jit.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * \brief Register application device source providing custom-primitive
 * intersection functions.
 *
 * HIP-RT dispatches custom (AABB / implicit) primitives through an
 * ``intersectFunc`` and an optional ``filterFunc``, one pair per *geometry
 * type*. Unlike Metal, those are not a separate compiled library that the
 * driver links at trace time: ``hiprtBuildTraceKernels()`` GENERATES the
 * dispatch and prepends it to the source it is given, so the definitions have
 * to be compiled **together with each generated kernel**. That is why this is a
 * source-registration call rather than a handle: Dr.Jit concatenates \c source
 * ahead of every traversing kernel it builds.
 *
 * Call once, before any scene with custom geometry is configured. Registering
 * different source after kernels have been built is not supported — the already
 * compiled kernels keep the old definitions.
 *
 * \param source
 *     HIP C++ device source defining every function named in \c isect_names and
 *     \c filter_names. May be \c NULL to clear a previous registration.
 *
 *     The signatures are fixed by the dispatcher HIP-RT generates, and appear
 *     in none of its headers because the calling code does not exist until
 *     build time:
 *
 *     \code
 *     __device__ bool name(const hiprtRay &ray, const void *data,
 *                          void *payload, hiprtHit &hit);        // intersect
 *     __device__ bool name(const hiprtRay &ray, const void *data,
 *                          void *payload, const hiprtHit &hit);  // filter
 *     \endcode
 *
 *     \c data is the matching ``hiprtFuncDataSet`` entry, so it is per geometry
 *     type rather than per geometry. On entry \c ray is in OBJECT space and
 *     both \c hit.primID and \c hit.instanceID are already populated; writing
 *     \c hit.t / \c hit.uv and returning \c true accepts the hit.
 *
 *     A mismatched signature is not a compile error. HIP-RT forward-declares
 *     each name it is given, so a wrong definition is simply a different
 *     overload and the failure arrives later as an unresolved symbol during the
 *     device link. See ``tools/hip_validate/isect_probe/``, which establishes
 *     all of the above by building it.
 *
 * \param isect_names
 *     Array of \c n_geom_types intersection-function names, indexed by geometry
 *     type. The geometry type of a primitive is set through
 *     ``hiprtGeometryBuildInput::geomType`` when its geometry is built, so the
 *     ordering here is the application's own and must match what it builds.
 *     An entry may be \c NULL for a geometry type that needs no intersector.
 *
 * \param filter_names
 *     Array of \c n_geom_types any-hit filter names, or \c NULL if none are
 *     needed. Individual entries may be \c NULL.
 *
 * \param n_geom_types
 *     Number of geometry types. Zero clears the registration.
 */
extern JIT_EXPORT void jit_hip_set_isect_source(const char *source,
                                                const char *const *isect_names,
                                                const char *const *filter_names,
                                                uint32_t n_geom_types);

/**
 * \brief Return the ``hiprtContext`` Dr.Jit compiles traversing kernels against
 *
 * The analogue of \c jit_metal_context(). An application that builds its own
 * geometries and scenes (Mitsuba's ``src/render/hip/accel.cpp``) must use this
 * context rather than creating one: HIP-RT's geometry and scene handles are
 * context-scoped, so a scene built against a second context traverses garbage
 * instead of failing cleanly.
 *
 * The context is created on demand, so it does not matter whether the
 * application or the kernel compiler asks first.
 *
 * \return The ``hiprtContext``, or \c NULL if HIP-RT is unavailable in this
 * build (see \ref jit_hip_compile's diagnostic).
 */
extern JIT_EXPORT void *jit_hip_rt_context();

/**
 * \brief Inform Dr.Jit about a per-scene HIP-RT configuration.
 *
 * Registers a built ``hiprtScene`` and returns a JIT variable index that owns
 * the lifetime of drjit-core's per-scene state. That index is passed as the
 * trailing argument to \ref jit_hip_ray_trace to select which scene a trace
 * runs against. When its reference count reaches zero, drjit-core drops its
 * per-scene bookkeeping and invokes the cleanup callback registered with
 * \ref jit_hip_scene_set_cleanup, so the application can release the HIP-RT
 * objects by dropping a reference rather than freeing them directly.
 *
 * This is much smaller than its Metal counterpart, and the reason is worth
 * stating: a ``hiprtScene`` is a plain device pointer. Metal needs a resource
 * handle mechanism because its TLAS is an ``id<MTLAccelerationStructure>``
 * reconstructed in-shader from a ``gpuResourceID`` and made resident with
 * useResource() at every launch; HIP-RT's handle travels in the kernel
 * parameter block like any other pointer.
 *
 * \param scene
 *     The built ``hiprtScene``.
 *
 * \param func_table
 *     Optional ``hiprtFuncTable`` supplying custom-primitive intersection and
 *     any-hit filter functions. May be \c NULL, in which case generated kernels
 *     still emit the stub ``intersectFunc`` / ``filterFunc`` definitions HIP-RT
 *     requires at link time (see the note in \ref jit_hip_ray_trace).
 *
 * \param geometry_ids
 *     Optional DEVICE array of ``uint32_t``, indexed by the hit's instance ID,
 *     supplying output 6 (``geometry_id``). May be \c NULL, in which case that
 *     output is always zero.
 *
 *     This exists because ``hiprtHit`` carries no geometry ID. That is not an
 *     omission to work around but a consequence of HIP-RT's scene model: an
 *     instance references exactly one geometry, so "which geometry was hit" is
 *     a property of the instance and an instance-indexed table states it
 *     exactly. Metal, whose acceleration structures nest geometries inside an
 *     instance, needs the per-hit field that HIP-RT does not.
 *
 * \param user_instance_ids
 *     Optional DEVICE array of ``uint32_t``, indexed by the hit's instance ID,
 *     supplying output 7 (``user_instance_id``). May be \c NULL, in which case
 *     that output is always zero. Metal sets this field when building the
 *     instance descriptor; HIP-RT has no equivalent, so it is a table here.
 *
 * \param geometry_types_mask
 *     Bit 0: triangle geometry present.
 *     Bit 1: bounding-box (custom-intersection) geometry present.
 *     Bit 2: curve geometry present.
 *     Bit 3: triangle backface culling required for at least one instance.
 *
 *     The same encoding Metal uses, so Mitsuba's ``ShapeIR::Kind`` maps onto
 *     both unchanged.
 */
extern JIT_EXPORT uint32_t jit_hip_configure_scene(void *scene,
                                                   void *func_table,
                                                   const void *geometry_ids,
                                                   const void *user_instance_ids,
                                                   uint32_t geometry_types_mask);

/**
 * \brief Perform an inline ray intersection in a HIP compute kernel
 *
 * Creates a ``VarKind::TraceRay`` IR node that emits a HIP-RT
 * ``hiprtSceneTraversalClosest`` (or ``...AnyHit`` for shadow rays) against
 * the scene identified by \c scene. For lanes that are masked off or that miss,
 * the distance output is +infinity, the validity flag \c false, and every other
 * output zero, so callers need not clear them separately.
 *
 * The signature is identical to \c jit_metal_ray_trace, deliberately: the
 * eight-output hit record is what Layer C is written against.
 *
 * A HIP-RT specific obligation the emitter discharges on the caller's behalf:
 * HIP-RT *declares* ``intersectFunc`` and ``filterFunc`` and leaves the
 * definitions to the application, so a kernel that traverses without them fails
 * to link with ``undefined hidden symbol``. Every generated kernel containing a
 * trace therefore carries at least stub definitions.
 *
 * \param n_args
 *     Number of ray input arguments. Must be 8.
 *
 * \param args
 *     Array of 8 JIT variable indices:
 *       [0] ox     (Float32) — ray origin X
 *       [1] oy     (Float32) — ray origin Y
 *       [2] oz     (Float32) — ray origin Z
 *       [3] dx     (Float32) — ray direction X
 *       [4] dy     (Float32) — ray direction Y
 *       [5] dz     (Float32) — ray direction Z
 *       [6] tmin   (Float32) — minimum ray distance
 *       [7] tmax   (Float32) — maximum ray distance
 *
 * \param mask
 *     JIT variable index of the active lane mask (Bool).
 *
 * \param out
 *     Array of 8 JIT variable indices (output, written by this function):
 *       [0] valid        (Bool)    — true if a hit was found
 *       [1] distance     (Float32) — distance to the closest hit
 *       [2] bary_u       (Float32) — barycentric U coordinate
 *       [3] bary_v       (Float32) — barycentric V coordinate
 *       [4] instance_id  (UInt32)  — instance index in the scene
 *       [5] primitive_id (UInt32)  — primitive index within the geometry
 *       [6] geometry_id  (UInt32)  — from the geometry_ids table, else 0
 *       [7] user_instance_id (UInt32) — from the user_instance_ids table, else 0
 *
 * \param n_out
 *     Number of output variables. Must be 8.
 *
 * \param scene
 *     JIT variable index returned by \ref jit_hip_configure_scene.
 *
 * \param shadow
 *     If nonzero, performs a shadow-ray test using the any-hit traversal. Only
 *     output 0 (the hit flag) is computed; outputs 1-7 are left untouched.
 */
extern JIT_EXPORT void jit_hip_ray_trace(uint32_t n_args, uint32_t *args,
                                         uint32_t mask, uint32_t *out,
                                         uint32_t n_out, uint32_t scene,
                                         int shadow);

/**
 * \brief Register a cleanup callback that runs when the scene variable dies
 *
 * The application's HIP-RT objects (scene, geometries, buffers) must outlive
 * the scene variable, which can outlast the application's own use of the scene
 * because unevaluated kernels and frozen-function recordings reference it
 * through their TraceRay nodes. \c callback runs once, right after drjit-core
 * releases its per-scene state.
 */
extern JIT_EXPORT void jit_hip_scene_set_cleanup(uint32_t scene_index,
                                                 void (*callback)(void *),
                                                 void *payload);

/**
 * \brief Create a handle exposing a scene as a frozen-function input
 *
 * Returns a ``UInt64`` variable whose data pointer is drjit-core's internal
 * per-scene bookkeeping object. It is used as an identity token so that
 * ``dr.freeze`` can capture the scene and rebind it across launches. The handle
 * does not own the scene.
 */
extern JIT_EXPORT uint32_t jit_hip_scene_owner_handle(uint32_t scene_index);

#if defined(__cplusplus)
}
#endif
