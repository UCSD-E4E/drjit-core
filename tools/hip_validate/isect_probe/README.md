# `isect_probe` — what signature does HIP-RT expect of a custom intersector?

HIP-RT declares the **dispatcher** in `impl/hiprt_device_impl.h:54`:

```c
HIPRT_DEVICE bool intersectFunc(uint32_t geomType, uint32_t rayType,
                                const hiprtFuncTableHeader &tableHeader,
                                const hiprtRay &ray, void *payload,
                                hiprtHit &hit);
```

but the signature it expects of **your** function — the one named in
`hiprtFuncNameSet::intersectFuncName` — appears in no header, because the code
that calls it is generated inside `hiprtBuildTraceKernels()` at build time.
Writing ~500 lines of device math against a guessed prototype is a bad trade,
so this directory settles it.

## The answer

```c
__device__ bool <name>(const hiprtRay &ray, const void *data,
                       void *payload, hiprtHit &hit);          // intersection
__device__ bool <name>(const hiprtRay &ray, const void *data,
                       void *payload, const hiprtHit &hit);    // filter
```

`data` is `funcDataSets[index].intersectFuncData` (resp. `filterFuncData`),
where `index = numGeomTypes * rayType + geomType` — so it is per
(geometry type, ray type), **not** per geometry.

On entry:

| field           | state                                                      |
|-----------------|------------------------------------------------------------|
| `ray`           | **object space** — `transformRay()` has already run         |
| `hit.primID`    | set, from the custom node (`hiprt_device_impl.h:1043`)      |
| `hit.instanceID`| set, at the top of `testLeafNode` (`:1032`)                 |
| `hit.t/uv/normal` | yours to write; returning `true` accepts the hit          |

`instanceID` being live is what makes the Metal data-lookup port possible:
Metal keys per-primitive data on `(instance_id, geometry_id)`, and HIP-RT
supplies `instanceID` — the geometry ID is redundant here because a HIP-RT
instance references exactly one geometry (see `src/hip_scene.h`).

## Where the answer came from

Two independent sources agreeing:

1. **The shipped library.** `libhiprt0300064.so` carries the codegen templates
   as string literals. `strings` recovers the forward declaration
   (`( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );`),
   the dispatcher body, and the call fragment
   (`( ray, data, payload, hit ); }`).
2. **A build.** The two `.hip` files here, run through the real
   `hiprtBuildTraceKernels()`.

## Running it

```sh
g++ -std=c++17 isect_probe.cpp \
    -I$HIPRT_NV_PATH/include -I$DRJIT_HIP_SHIM_CUDA_INCLUDE \
    -L$HIPRT_NV_PATH/lib -L/run/opengl-driver/lib \
    -lhiprt64 -lcuda -Wl,-rpath,$HIPRT_NV_PATH/lib -o isect_probe

./isect_probe probe_sphere.hip    mi_isect_sphere   # expect exit 0
./isect_probe probe_wrong_sig.hip mi_isect_wrong    # expect exit 1
```

Observed, HIP-RT 3.0.3 on the CUDA shim (RTX 3060):

```
probe_sphere.hip     -> 0 (SUCCESS)
probe_wrong_sig.hip  -> 2 (FAILURE)
    Orochi error: 'a PTX JIT compilation failed' [218] on line 324 in
    'hiprt/impl/Compiler.cpp'
```

**Run both.** The negative arm is not decoration: if a wrong signature also
built, it would mean the generated dispatcher never referenced our function and
the positive result proved nothing. Note that it fails at PTX *link*, not at
compile — HIP-RT forward-declares the name it was given, so a mismatched
definition is simply a different overload and the declared one goes undefined.

## A side result worth recording

This build path — `numGeomTypes=1` with a populated `funcNameSets` — succeeds
under the CUDA shim. The §11n.1 failure is not triggered merely by asking for a
function table.
