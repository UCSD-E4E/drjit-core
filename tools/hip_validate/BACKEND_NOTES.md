# hip_eval.cpp blueprint — notes from reading the Metal backend

Output of the §10 task "read `metal_eval.cpp`, `metal.h` and `scene_metal.inl`
end to end before writing any HIP code". These are the structural decisions the
Metal backend already made, and what each one maps to for HIP.

Metal is the reference throughout, **not** CUDA/OptiX — it is the only backend in
this tree that emits *source text* and integrates ray tracing inline, which is
exactly the shape `hip_eval.cpp` needs (PLAN.md §3.1, §3.2).

---

## 1. Entry point and skeleton

`jitc_metal_assemble(ThreadState*, ScheduledGroup, n_regs, n_params)` is called
from the dispatch chain in `eval.cpp:589-599`. `jitc_hip_assemble` slots in
alongside, guarded by `DRJIT_ENABLE_HIP`. That plus the enum entry is most of
the upstream seam (§9).

Emission order, from `jitc_metal_assemble`:

1. Preamble — `#include <metal_stdlib>`, `using namespace metal;`
2. `struct Params { uint size; device void *args[N]; }`
3. Kernel signature with a **literal `^^^^...` placeholder** for the name
4. Call-data base pointer, if the kernel has calls
5. The variable loop (§3 below)
6. Closing brace, then scene configuration comment
7. Callable bodies, then forward declarations moved *before* the kernel

The `^^^^` run is not decoration: `eval.cpp:607-625` replaces it in-place with
the kernel hash after assembly. Emit the same placeholder width.

**HIP mapping.** Preamble becomes the prelude contract in `prelude_hip.h`. The
kernel becomes `extern "C" __global__ void drjit_^^^^(Params params)`, and
`thread_position_in_grid` becomes `blockIdx.x * blockDim.x + threadIdx.x`. HIP
takes struct kernel parameters by value, so `Params` can pass directly rather
than through a buffer binding.

---

## 2. The `fmt` mini-language

Shared with `cuda_eval.cpp`; documented at the top of both files.

| Code | Meaning |
|---|---|
| `$u` | uint32 |
| `$s` | C string |
| `$t` | variable type |
| `$b` | variable type, binary/bit-pattern form |
| `$v` | variable name (`r1234`) |
| `$l` | literal value |
| `$o` | index into `params.args[]` |

Reuse verbatim. The `$o` convention is off-by-one against the parameter index
because of the leading `size` field — see the comment at `metal_eval.cpp:278`.

---

## 3. The variable loop

For each `schedule[gi]` in the group:

- `ParamType::Input` + literal → materialise inline (`as_type<float>` for f32,
  `as_type<half>` for f16, plain cast otherwise).
  **Pointer literals are deliberately loaded from `params.args[]`, not inlined**,
  so frozen-function replay can rebind the address. Preserve this or `@dr.freeze`
  breaks (§Phase 6).
- `ParamType::Input`, non-literal → `((device const T*) params.args[$o])[r0]`
  when `size > 1`, else a scalar dereference.
- Otherwise → `jitc_metal_render(v)`, then if `ParamType::Output`, store to
  `((device T*) params.args[$o])[r0]`.

Arrays take a separate path via `render_array_memcpy_in/out` against a named
`p<reg>` pointer.

---

## 4. FP64 — a real divergence

```c
if ((VarType) v->type == VarType::Float64)
    jitc_fail("... the program should not contain Float64 variables.");
```

Apple GPUs have no FP64, so Dr.Jit promotes `Float64`→`Float32` at variable
creation and codegen treats arrival as a bug.

**Do not copy this.** `gfx90a` has *full-rate* FP64 — one of the MI210's headline
capabilities. HIP emits `double` natively and must not inherit the promotion.
This is the one place the Metal template is actively wrong for us.

---

## 5. Opcode rendering

Three helpers cover most of the surface:

```c
render_unary (v, op)          ->  T v = op(a);
render_binary(v, op)          ->  T v = a op b;
render_call  (v, fn, n_args)  ->  T v = fn(a0, a1, a2);
render_round (v, fn)          ->  rounding, optionally fused with a float->int cast
```

then a `switch ((VarKind) v->kind)`. §0.2 measured **54 distinct opcodes** in the
simplest possible Mitsuba scene — that is the floor for coverage, and a useful
ordering hint: the measured frequency was `mov` 390, `mul` 145, `ld` 129,
`fma` 104, `add` 89, `st` 62, `setp` 61, `selp` 59. Bring those up first.

---

## 6. Resource handles — simpler for HIP

`jitc_metal_render_resource_handle` reconstructs a typed reference from an opaque
`gpuResourceID` for textures, samplers, acceleration structures and intersection
function tables, because MSL resource types are opaque and cannot be cast from an
integer.

**HIP is materially simpler here.** `hipTextureObject_t` is an ordinary 64-bit
handle and HIP-RT scene handles are plain pointers, so most of this collapses to
a normal pointer load. Expect `jitc_hip_render_resource_handle` to be a fraction
of Metal's size.

---

## 7. Ray tracing — the direct template

`VarKind::TraceRay` at `metal_eval.cpp:917-1031`. Shape:

1. Declare 8 outputs, pre-set to **miss** values — `hit=false`,
   `t=+inf` (`as_type<float>(0x7f800000u)`), rest zero. Masked and missed lanes
   then simply fall through with correct values.
2. `if ($v) {` when the mask is not the literal `1`, else a bare block.
3. Read the 8 ray parameters from `td->indices[0..7]`
   (origin xyz, direction xyz, tmin, tmax).
4. Configure the intersector from `scene->geometry_types_mask`.
5. Build the ray, call `intersect`.
6. On hit, overwrite the outputs.

Only outputs 1–7 are computed when `td->shadow` is unset; shadow rays write the
hit flag alone.

`geometry_types_mask` bits: `0` triangle, `1` bounding_box, `2` curves,
`3` backface-culled triangles. These map 1:1 onto Mitsuba's `ShapeIR::Kind`
(`include/mitsuba/render/scene_ir.h`), which is why the Mitsuba-side port is
cheap (§2 Layer C).

**HIP mapping.** `raytracing::intersector<...>` becomes a `hiprtGeomTraversal*` /
`hiprtSceneTraversal*` object, `_hit` becomes `hiprtHit`, and the IFT becomes a
`hiprtFuncTable`. The 8-output contract, the miss-value trick and the mask
handling all carry over unchanged — that contract is `jit_metal_ray_trace`'s
signature and should become `jit_hip_ray_trace` verbatim.

Contrast with OptiX, whose `_optix_hitobject_traverse` call takes ~50 arguments
(§0.2). Mirroring Metal here avoids roughly an order of magnitude of interface.

### 7a. Phase 0b spike results — measured, not assumed

`spec_trace.hip` compiles and **links** a real HIP-RT scene traversal for
gfx90a. Five findings, three of which change the plan.

**1. The device library is a compressed offload bundle, not bitcode.**
`hiprt*_amd_lib_linux.bc` starts with the magic `CCOB`. Handing it to clang
fails with *"file doesn't start with bitcode header"*. It must be unbundled per
target with `clang-offload-bundler --type=bc --unbundle`, which is also the
only way to confirm a given arch is present. `hip_validate` now does this
automatically for any kernel mentioning `hiprt`.

**2. `gfx90a` really is in the bundle, and the traversal links.** This upgrades
§0.2's "HIP-RT supports gfx90a" from a support-matrix claim to a linked code
object. The build line that works:

```
hipcc --offload-arch=gfx90a --genco --rocm-device-lib-path=$HIP_DEVICE_LIB_PATH \
      -I$HIPRT_PATH/include \
      -Xclang -mlink-bitcode-file -Xclang <unbundled-gfx90a>.bc  kernel.hip
```

**3. Somebody must define `intersectFunc` and `filterFunc` — but not always
us.** HIP-RT declares them and leaves the definitions to be supplied: they are
the custom-primitive intersection and any-hit filter hooks, analogous to
Metal's intersection function table and OptiX's IS/AH programs. Hand-link the
device bitcode without them and the link fails with `undefined hidden symbol:
intersectFunc(...)`.

**CORRECTED in Phase 5 — the original conclusion here ("the emitter must emit
at least stubs") was wrong, and wrong in the expensive direction.** Which side
owns the definitions depends on how the traversal library is linked:

| link route | who defines the hooks |
|---|---|
| hand-linked bitcode (`-Xclang -mlink-bitcode-file`) — what `hip_validate` does | the application |
| `hiprtBuildTraceKernels()` — what the BACKEND uses, shim and hardware alike | HIP-RT, generated and prepended |

Emitting stubs is therefore correct for this harness and *breaks* the backend:
`hiprtBuildTraceKernels()` fails with `function "intersectFunc" has already
been defined`, reported only as a bare `hiprtErrorInternal` from the API. So
codegen emits the `#include` and nothing else, and `run_tests.sh` prepends the
definitions itself before its hand-linked gfx90a compile.

The finding generalises: **§7a's observations were made through the harness's
link path, which is not the backend's.** Anything else recorded here about
linking should be re-checked against `hiprtBuildTraceKernels()` before being
built on.

**4. `hiprtHit` does not carry a geometry ID or a user instance ID.** It has
`hasHit()`, `t`, `uv.x`, `uv.y`, `primID`, `instanceID` (plus a geometric
normal Metal does not give) — six of Metal's eight outputs. `geometry_id` and
`user_instance_id` have to be reconstructed application-side.

**Resolved in Phase 5, and the signature did NOT need qualifying after all.**
Both missing outputs are supplied by device tables indexed by the hit's
instance ID, passed to `jit_hip_configure_scene()`. That is exact rather than a
workaround: **a HIP-RT instance references exactly one geometry**, so "which
geometry was hit" genuinely is a property of the instance. Metal needs a
per-hit field because its acceleration structures nest several geometries
inside one instance; the difference is in the scene model, not in the fidelity
of the answer. `jit_hip_ray_trace` is therefore byte-for-byte
`jit_metal_ray_trace`'s signature, which is what makes Layer C cheap.

**5. Traversal costs 64 VGPR / 38 AGPR / 54 SGPR / 800 B scratch** on gfx90a
for a minimal closest-hit scene traversal. HIP-RT keeps its stack in scratch,
so this is the number to watch: a shape change that doubles it still compiles,
still runs identically on the shim (different register file entirely), and
surfaces as halved MI210 throughput much later. `hip_validate` now prints
register and scratch usage for **every** kernel for this reason.

### 7b. Traversal IS executed, on NVIDIA -- and it is a good proxy

The first draft of §7a said hit correctness was unverifiable without an MI210,
because "the packaged HIP-RT is AMD-only". That was true of the PACKAGE and
false of HIP-RT: upstream supports NVIDIA through Orochi, and nixpkgs disables
it in three independent places (see the `hipRtNv` derivation in flake.nix).

With those undone, `hiprt_triangle.cpp` builds a real BVH on the device,
compiles a traversal kernel through NVRTC, launches it, and checks the results:

```
lane 0 (inside):  hit=1 t=1.000   want hit=1 t=1.000
lane 1 (outside): hit=0 t=-1.000  want hit=0 t=-1.000
TRAVERSAL CORRECT
```

**This is an unusually good cross-vendor proxy, for the reason in §7a:** gfx90a
has no ray-tracing hardware, so it takes HIP-RT's RTIP 0 SOFTWARE path -- the
same portable C++ that compiles for NVIDIA. An RDNA card would take the
hardware branch and prove much less. What it still does not cover: wave64 (this
runs at 32) and the AMD host API.

Three NVIDIA artifacts are required and missing any one fails late and
unhelpfully -- `hiprtErrorInternal`, with the actual filename visible only
after `hiprtSetLogLevel()`:

| file | role |
|---|---|
| `hiprt*_nv_lib.fatbin` | traversal library linked into user kernels |
| `hiprt*_nv.fatbin` | precompiled BVH-BUILDER kernels |
| `oro_compiled_kernels.fatbin` | Orochi parallel primitives |

Also: `hiprtBuildTraceKernels()` takes compiler options, and NVRTC starts with
an empty include search list -- pass `-I/include` or the kernel
cannot find `hiprt_types.h`.

**Still needs an MI210:** wave64 semantics, fp16 numerics, and the AMD host API.

---

## 8. Scene discovery happens *during* codegen

`metal_kernel_scenes` is appended to by `metal_register_kernel_scene()` as
`TraceRay` nodes are rendered — there is no separate pre-walk. The list is then
consumed at compile time to link intersection functions into the pipeline.

`jitc_metal_render_scene_configuration()` emits the scene's properties as a
**comment**. That is not documentation: intersection functions and geometry types
affect pipeline linking but are otherwise invisible in the MSL, so folding them
into the source text ensures incompatible configurations hash to different
kernels. **HIP needs the same trick** or the disk cache will serve a kernel built
against the wrong function table.

---

## 9. Callables and ordering

Callable bodies are emitted *after* the kernel, then forward declarations (plus
`GlobalType::Global` bodies in full) are moved before it with
`buffer.move_suffix`. Only single-target callables need declarations —
multi-target ones are reached through the function table and never named.

HIP C++ has the same declare-before-use requirement, so this machinery ports
directly.

### 9a. Dispatch shape — HIP uses a `switch`, not a function table

The one place with no usable template. `callable_index` is assigned **globally
across the kernel** (`cuda_eval.cpp:235`, `metal_eval.cpp:375`), and both of
those backends build a single table of that size and reinterpret it with each
call site's signature — PTX because `.u64 callables[]` is untyped, Metal
because `visible_function_table` is reinterpretable.

HIP C++ has neither escape hatch. `spec_call.hip` tested all three candidates
on both arms:

| shape | gfx90a | NVRTC (shim) |
|---|---|---|
| A. `switch` over the global index, direct calls | ✅ | ✅ |
| B. typed `__device__` fn-pointer table, no casts | ✅ | ✅ |
| C. one table cast to a common fn-pointer type | ✅ | ❌ |

C is what would have let HIP inherit the existing scheme unchanged, and it is
the one that fails:

```
error: dynamic initialization is not supported for a __device__ variable
```

A bare function name is a constant address; a *cast* of one is not, as far as
NVRTC is concerned. B compiles because it has no casts — but B only works when
every entry shares a signature, which across a whole kernel they do not.

Since the shim is the only place the vcall path can be executed before the
MI210 arrives, a shape that cannot run there is not testable. **So the emitter
uses A.** It needs no table, no cast, and no indirect-call ABI on AMDGPU at
all; every arm is a direct call the compiler may inline. The cost is code size
proportional to the instances reachable from each call site — worth measuring
on a real Mitsuba scene (§5 Phase 2 "codegen tuning"), not worth pre-empting.

### 9b. What the switch actually keys on

Not the callable index — **the instance ID (`self`)**. The callable index is
assigned in a pass over `globals_map` that has not run when the dispatch site
is written, so its value is not available as a `case` label. Instance IDs are,
via `call->inst_id[]`.

Instances sharing a callable are grouped so their labels fall through to **one**
call body:

```c
switch (r7) {
    case 1:
    case 4:
        ret_9 = func_<hashA>(r0, r7, _cd_9, r3); break;
    case 2:
        ret_9 = func_<hashB>(r0, r7, _cd_9, r3); break;
    default:
        ret_9.r0 = (f32) 0; break;
}
```

So code size scales with *unique callables*, not with instances — which matters
because a Mitsuba scene has far more shapes than BSDF types. The `default` arm
is not dead code: `self` is data, and an instance ID outside this site's set
must produce something defined.

A consequence worth knowing: **HIP never reads the offset table's low half.**
`jitc_call_upload()` still packs `(data_offset << 32) | callable_index` for
every backend; HIP uses only the high half, for the instance's data pointer.

### 9c. Emission order

Three chunks, in this order in the final source:

1. type preamble
2. **struct definitions in full** (`GlobalType::Global` — the vcall return
   structs live here) **+ one forward declaration per callable**
3. kernel, then callable bodies

Chunk 2 is built last and relocated with `buffer.move_suffix()`. The return
structs must be *definitions*, not declarations: a callable's signature names
its return type by value.

Unlike Metal, **every** callable needs a declaration, not just single-target
ones. Metal can skip multi-target callables because they are reached
anonymously through the function table; switch dispatch names them.

---

## 10. Formatting pass — copy wholesale

`jitc_metal_format()` (~80 lines) generates *unformatted* source during codegen —
indentation tracking is costly — and reindents only when `JitFlag.PrintIR` or a
high log level is set.

It relies on three properties of the emitted text: braces only ever denote
control flow, comments are `//` only, and no statement is split across a blank or
comment line. Emitted HIP satisfies all three, so this is directly reusable and
worth taking as-is. Violating the properties degrades indentation but never
semantics.

Given §0.2 measured a 2039-line kernel for the *simplest* scene, readable dumps
matter — this is the §3.1 debuggability argument in practice.

---

## 11. Working discipline — test first

Every change from here is test-driven. Concretely:

**Write the assertion before the implementation.** `tests/hip_codegen.cpp` was
written before any codegen exists and currently exits 1. It is registered with
`WILL_FAIL TRUE` so `ctest` stays green while Phase 2 is incomplete — and turns
**red the moment codegen starts working**, which is the signal to delete that
property. Do not weaken the test to make it pass.

**Two loops, different speeds.**

| Loop | Command | Needs | Use for |
|---|---|---|---|
| Fast | `tools/hip_validate/run_tests.sh ./build-hip/hip_validate` | nothing (no AMD GPU) | every opcode, every edit |
| Acceptance | `ctest -R hip_` from `build-*/tests` | MI210 for the codegen test | declaring a phase done |

The fast loop compiles each kernel for **real gfx90a** and **executes** it on the
local NVIDIA GPU. Per opcode brought up in `hip_eval.cpp`:

1. Add a kernel under `tools/hip_validate/kernels/` exercising it, plus an
   assertion in `tests/hip_codegen.cpp`.
2. Run the fast loop — watch it fail.
3. Implement the opcode.
4. Run again — green.

**Negative tests are load-bearing.** `kernels/broken_syntax.hip` must FAIL. The
runner treats a `broken_*` kernel that passes as an error, because a harness
that has silently stopped detecting errors makes every other result worthless.
Keep at least one negative case as coverage grows.

**Phase 0a was not done this way** — the wiring was written first and verified
with a throwaway program. `tests/hip_wiring.cpp` retrofits those invariants so
they are at least permanent and repeatable. Do not repeat that pattern.

### Known limit: opcode emission cannot be TDD'd until Phase 1

`jitc_hip_assemble()` takes a `ThreadState` and a `ScheduledGroup`, and the only
way to reach it is through `jitc_eval()` — which first needs `jit_malloc` for
the backend, a live ThreadState, and scheduling. In other words **reaching
codegen requires most of Phase 1**, which requires an MI210.

A "codegen-only" mode that registers the backend without a device was
considered and rejected: it is not a small hook, it would make
`jit_has_backend(HIP)` lie, and every allocation path would need a fake.

So the ladder is:

- **Now, fully testable:** anything pure — the type tables
  (`tests/hip_types.cpp`), the reindent pass (`tests/hip_format.cpp`). Build
  these as pure functions in headers *specifically so* they stay testable
  without JIT state or exported symbols.
- **Now, testable as specification:** hand-write the source an opcode *should*
  produce as a kernel under `kernels/` and validate it with `hip_validate`.
  That proves the target shape compiles for gfx90a and computes the right
  numbers, so when the emitter is written there is an unambiguous target.
- **After Phase 1:** the emitter itself, end-to-end, via `tests/hip_codegen.cpp`.

Prefer factoring emitter logic into pure helpers wherever it does not distort
the design — every piece moved into that first category is a piece that gets
tested months before the hardware arrives.

## 11a. Emit CUDA/HIP intrinsics, never Clang `__builtin_*`

Discovered by the opcode specimens, and binding on `hip_eval.cpp`.

`hipcc` is Clang-based, so `__builtin_fabsf`, `__builtin_popcount`,
`__builtin_clz`, `__builtin_bitreverse32`, `__builtin_bit_cast` and friends all
compile cleanly for `gfx90a`. **NVRTC rejects every one of them.** Emitting
builtins would therefore pass the gfx arm, fail the execution arm, and — more
importantly — pin generated source to one compiler for no benefit.

Use the device-intrinsic spellings that exist on both:

| Operation | Emit | Not |
|---|---|---|
| Abs / Sqrt / Fma | `fabsf` `sqrtf` `fmaf` | `__builtin_*` |
| Min / Max | `fminf` `fmaxf` | `__builtin_fmin*` |
| Ceil / Floor / Trunc | `ceilf` `floorf` `truncf` | `__builtin_*` |
| Round | `rintf` (nearest-**even**) | `roundf` (half away from zero) |
| Popc / Clz / Brev | `__popc` `__clz` `__brev` | `__builtin_popcount` … |
| Ctz | `__ffs(x) - 1` | `__builtin_ctz` (no CUDA analog) |
| Bitcast f32↔u32 | `__float_as_uint` / `__uint_as_float` | `__builtin_bit_cast` |
| Bitcast f64↔u64 | `__double_as_longlong` / `__longlong_as_double` | `__builtin_bit_cast` |

`rintf` versus `roundf` is worth singling out: `VarKind::Round` is
round-to-nearest-even, matching IEEE and the CUDA backend. `roundf` rounds half
away from zero and would silently disagree on exactly the .5 cases.

## 11b. Opcode specimens

`kernels/spec_*.hip` hand-write the source the emitter *should* produce for each
`VarKind`, deliberately shaped like machine output — SSA temporaries, one op per
line, explicit types — rather than idiomatic HIP. They are a specification: when
`hip_eval.cpp` is written there is an unambiguous target instead of prose.

They are **self-checking**. Each op is compared against an independently
computed value and sets a distinct bit on mismatch, so `out[]` is zero exactly
when every opcode is right; `run_tests.sh` runs them with `--expect-zero`. That
turns "it compiled" into "it computed the right thing" without golden-value
files, which would have to be generated by running the very code they validate.

The mechanism immediately earned itself: `spec_arith` failed on 788/1024 lanes
with error code 24, which was a bug in the *specimen* — it asserted
`(a + b) - b == a` exactly. Float addition is not associative; exact equality is
only legitimate for exactly-representable operands. Inspection would not have
caught that.

## 11c. Further constraints found by the specimens

Each of these compiled cleanly and still would have been wrong.

**`__exp2f` does not exist.** The fast-intrinsic naming is *not* uniform:
`__sinf`, `__cosf`, `__log2f`, `__fdividef`, `__frcp_rn`, `__fsqrt_rn` all
exist on both platforms, but there is no `__exp2f` on either. Emit `exp2f` for
`VarKind::Exp2`. Do not assume a `__`-prefixed form exists just because its
neighbours do.

**No barrier below the bounds-check `return`.** The prologue guards with an
early `return` for lanes past the end (§ prologue). Any `__syncthreads()` after
that point is undefined — the returned threads never arrive — and the launch
fails at runtime rather than at compile time. If codegen ever needs a barrier
(block reductions, shared-memory staging), either the barrier must dominate the
guard or the guard must become a predicated region instead of a return.

**Atomics must target global memory.** Dr.Jit's `ScatterInc` / `ScatterExch` /
`ScatterCAS` always address a device buffer. Emitting an atomic against a
materialised temporary takes the address of a local, forces private memory, and
— this is the dangerous part — **compiles on both arms and then fails at
launch**. Always route atomics through the buffer pointer.

The last two share a signature worth internalising: *compiles everywhere, dies
at launch*. Neither arm's compiler can catch them, so the execution arm is the
only thing standing between these and a debugging session on the MI210.

**Approximations are not bit-exact and must not be pinned as if they were.**
`Sin`/`Cos`/`Exp2`/`Log2`/`Tanh` are the multi-function-generator ops — NVIDIA's
SFU path, gfx90a's `V_SIN_F32`/`V_EXP_F32` — and the `*Approx` kinds are
explicitly licensed to be sloppy. `kernels/spec_math.hip` checks them against
identities with deliberately loose tolerances. Tightening those is the wrong
instinct: it would either fail on real hardware or set an expectation the MI210
cannot meet. Accurate and approximate forms are checked at *different*
tolerances on purpose — that asymmetry is the specification.

## 11d. The next concrete blocker: `fmt_hip`

`fmt` is not one function. `src/strbuf.cpp` carries a SEPARATE implementation per
backend — `fmt_llvm`, `fmt_cuda`, `fmt_metal` — each with its type table baked
into the `$t` / `$b` cases:

```c
case 't': cur = w_type(cur, type_name_llvm [v->type]); break;   // fmt_llvm
case 't': cur = w_type(cur, type_name_ptx  [v->type]); break;   // fmt_cuda
case 't': cur = w_type(cur, type_name_metal[v->type]); break;   // fmt_metal
```

So `jitc_hip_render()` cannot be written until `fmt_hip` exists, wired to
`type_name_hip` / `type_name_hip_bin` (already implemented and tested in
`src/hip_eval.h`). The work is mechanical — copy `fmt_metal`, swap the two
tables, declare it in `strbuf.h` — but it is a genuine upstream-seam addition of
roughly 150 lines, so it belongs in the §9 patch inventory rather than in a
`hip_*` file we own.

**Do this before attempting the emitter**, and note it is the one piece so far
that could NOT be kept out of upstream files. Everything else has stayed in new
`hip_*` sources or behind a one-line guard.

## 11e. Control flow — VERIFIED (this section is kept for the reasoning)

**Resolved.** HIP is registered with the `TEST_*` macros and `test_loop` passes
9/9. What follows is why the bespoke test was the wrong instinct, because the
same temptation recurs for every subsystem still missing.

`jitc_hip_render()` implements `LoopStart` / `LoopCond` / `LoopEnd` / `LoopPhi` /
`LoopOutput` and `CondStart` / `CondMid` / `CondEnd`, ported closely from
`metal_eval.cpp`.

The subtle part is `LoopEnd`'s back edge. Copying `inner_out -> inner_in`
naively is wrong when the two sets alias -- an earlier copy clobbers a value a
later one still needs, a swap being the minimal example -- so every aliasing
output is staged into a temporary first. It also borrows `Variable::scratch` as
a marker and MUST clear it, or it corrupts `jitc_var_traverse()`'s visited
tracking in `jit_eval()`, which would fail somewhere else entirely.

A hand-rolled symbolic-loop test was attempted and removed. `jit_var_loop_end()`
needs a `jit_record_begin()` checkpoint and may return 0, meaning the body has
to be recorded a SECOND time after Dr.Jit simplifies the loop state. A test that
gets that protocol subtly wrong fails for reasons unrelated to codegen, which is
worse than no test: it points at the wrong suspect.

**The right coverage was to register HIP with the `TEST_*` macros in
`tests/test.h`**, so the existing suites run against this backend. They already
encode the protocol correctly and cover far more than a bespoke test would.

## 11f. What the registered suites found

Doing that turned up nine emitter bugs and six wiring bugs in an afternoon,
none of which the specimens could have caught on their own — the specimens
check that a *construct* works, the suites check that the emitter *chooses* it.
Two are worth carrying forward as patterns rather than incidents:

* **Two-bit backend fields.** HIP is backend 4, the first value needing a third
  bit. `Variable::backend` and `AllocInfo` both truncated it, and neither
  failed at the truncation — one surfaced as "the host backend is unavailable"
  during codegen, the other as an out-of-memory two suites away. Any new packed
  field must be added to `tests/hip_packing.cpp`.

* **The shim is a CUDA ThreadState.** Every site that allocates, launches, or
  **synchronises** has to treat a HIP backend as CUDA-backed; that is what
  `jitc_is_cuda_backed()` is for. A missing sync site does not fail loudly, it
  reads pinned memory before the copy into it lands — which looks exactly like
  a wrong reduction result.

### Current state per suite (CUDA shim)

| suite | HIP |
|---|---|
| `test_basics` | 7/7 |
| `test_loop` | 9/9 |
| `test_mem` | 17/17 |
| `test_reductions` | 14/14 |
| `test_vcall` | 14/14 |
| `test_record` | 9/9 |
| `test_array` | 15/15 |

**Every suite in the tree now runs against HIP.** The skip mechanism in
`tests/test.cpp` is retained with an empty table: the next subsystem ported
will want it, and an empty table states the position more clearly than a
deleted one.

`test_graphviz` is excluded, for the reason below.

### `test_graphviz` fails, and it is not ours

It fails identically at the upstream merge-base (`7a9ab1f`) with HIP absent:
the hardcoded reference string in `tests/graphviz.cpp` has drifted from the
generator (font names, `dpi`, backend-name capitalisation, and variable
ordering). Do not chase it as a regression; it needs an upstream fix or a
regenerated reference.

## 11g. The device library — what the port actually changed

Phase 3 needed `resources/*.cuh` (the CUDA-only half that `block_reduce`,
`compress`, `mkperm` and friends live in) to compile for `gfx90a`. Ported in
place rather than forked, so the two targets cannot drift. Findings worth
carrying:

- **ROCm 7.2 has `__ballot_sync` / `__shfl_*_sync` / `__any_sync` /
  `__match_any_sync` / `__syncwarp`**, and `amd_warp_sync_functions.h`
  `static_assert`s that the mask is 64-bit. That turns "you forgot a
  `0xFFFFFFFF`" from a silent wrong answer into a compile error — the single
  most useful safety property in this whole area. Do not defeat it by casting.
- **`warpSize` is not a constant expression on HIP.** It cannot size a template
  or a `constexpr`. There is also no wavefront-width macro in 7.2 —
  `__AMDGCN_WAVEFRONT_SIZE__` is gone. `resources/common.h` derives the width
  from `__GFX9__` (every GFX9 part is wave64-only) and the host refuses to load
  the blob if the device disagrees.
- **`.cg` has no AMD spelling.** The decoupled-lookback scans publish a
  (value, status) word that successors spin on; a read served from the
  CU-private, non-coherent L1 either spins forever or pairs a "done" status
  with a stale value. The equivalent is `__hip_atomic_load/store` at
  `__HIP_MEMORY_SCOPE_AGENT`. This is semantics, not syntax.
- **`umax` / `ullmax` / `llmax` and friends do not exist on HIP.** Clang finds
  the signed 32-bit forms through `<algorithm>`; the rest need shims.
- **`gemm.cuh` uses `__grid_constant__`**, which has no HIP spelling. Excluded
  on HIP — `batched_gemm` routes to rocBLAS, which nothing we target reaches.
- **`--genco` emits an offload bundle** (`CCOB` once compressed), same as
  HIP-RT's bitcode in §7a. The Makefile unbundles it; `pack_hip` embeds the raw
  code object, which every HIP version accepts.

Two idioms were replaced wholesale because they encode the width in a literal
and compile happily at the wrong one: `31 - __clz(ballot)` became
`highest_lane_()`, and `peers << (32 - lane)` became
`popc_(peers & lanemask_lt_(lane))` — the latter was also UB at lane 0.

### Proving the port did not break anything

Two independent checks, because neither alone is enough:

1. **nvcc PTX, entry by entry.** 638 of 646 entries are byte-identical to the
   pre-port build; the 8 that differ are exactly the intended ones (7 mkperm
   plus `compress_large`). This bounds the blast radius without running a thing.
2. **`devlib_check.sh`.** The suite cannot otherwise see the port at all —
   drjit-core embeds the *committed* `kernels_75.lz4`, which predates it. The
   script regenerates that blob from the ported sources, rebuilds, and runs
   drjit-core's own tests against it: 8/8. That is real execution of the eight
   changed kernels on real hardware.

Both run at width 32. Neither says anything about wave64, which is the point of
carrying the width as a parameter rather than fixing it.

## 11h. Phase 4 — what turned out not to be work

The remaining opcode gap after Phase 3 was `BoundsCheck`, `PacketGather` /
`PacketScatter`, the texture trio, and the OptiX-only kinds. Two of those four
dissolved on inspection.

**Textures need nothing.** `drjit/texture.h:43` gates every hardware path on

```cpp
static constexpr bool HasGPUTexture =
    (IsHalf || IsSingle || IsUInt8) && (IsCUDA || IsMetal);
```

A HIP array is neither, so every `if constexpr (HasGPUTexture)` branch compiles
out and `dr::Texture` takes its software path automatically. `TexLookup`,
`TexFetchBilerp` and `TexWrite` are therefore never emitted for this backend,
and Phases 6-7 are not blocked on them. PLAN §5 called textures "more
deferrable than they look"; they are in fact free.

The trap for later: when Layer B adds `is_hip_v`, do **not** add `IsHIP` to
that disjunction. Doing so switches on a hardware path (`jit_hip_tex_*`) that
does not exist. Measure the software path's cost on the MI210 first — and note
that on CUDA the hardware units resolve sub-texel position with only 8
fractional bits, so the software path is *more* accurate, not less.

**Scatter aggregation is not a correctness gap.** §3.3 flagged
`cuda_scatter.cpp`'s peer aggregation -- `activemask.b32`, 31-clamped
`shfl.sync.bfly.b32`, `vote.sync.ballot.b32` -- as needing redesign for a
64-wide ballot. It does, but only if we want it: the HIP backend issues plain
per-lane atomics, which are always correct and merely slower under contention.
There is no wave64 ballot to redesign because there is no ballot. That leaves
the redesign a performance task for after the backend is correct, which is
where §11's discipline says it belongs.

**Packet memory needs no wide-vector machinery.** Metal reinterprets through
`uint4` / `as_type<>` and CUDA through `ld.global.v4`, because MSL and PTX will
not merge adjacent accesses for them. HIP compiles C++ through LLVM, which
does. Measured on gfx90a: an element-wise packet body and an explicit `float4`
body produce the same machine code -- one `global_load_dwordx4`, one
`global_store_dwordx4` -- even when the pointer arrives as an opaque `void *`
from the params struct, because the `index * total_bytes` stride is enough for
LLVM to infer 16-byte alignment. Element-wise it is.

The one thing packet memory *does* demand care with is the index scaling, and
it is asymmetric: `op.cpp` pre-multiplies the SCATTER index by the packet width
and not the gather index, so

```
PacketGather   byte address = ptr + index * (count * tsize)
PacketScatter  byte address = ptr + index * tsize
```

CUDA and Metal agree on this, so it is a contract of the IR rather than a
backend convention. Getting it backwards compiles and reads the wrong packet.
`spec_packet.hip` pins it, and `tests/hip_codegen.cpp` checks both halves
against independently computed expectations rather than against each other --
a round trip through one opcode would hide a shared error.

**Verify that a new opcode is actually reached.** This is the sharpest lesson
of Phase 4, and it cost two wrong conclusions before it stuck.

`jit_var_gather_packet()` silently falls back to N scalar gathers under half a
dozen conditions (`op.cpp:2169`) -- `JitFlag::PacketOps` off, an unaligned or
literal source, a symbolic source. A test written against the public API can
therefore pass without ever entering the new code. Both packet renderers were
confirmed by temporarily replacing their bodies with `jitc_fail()` and watching
the test die.

The same technique then overturned a fix. `render_scatter_packet()`'s reduce
path emitted every element into one block, which redeclares the locals that
`render_scatter_reduce()`'s compare-and-swap form introduces -- a real defect.
A test was written for it, using `ReduceOp::Min` on `Float32` because that is
the combination with no native atomic. It passed *with the defect reinstated*.
The tripwire explained why: `op.cpp`'s `use_packet_op` selection has arms for
LLVM, CUDA and Metal and **none for HIP**, so a non-Identity packet scatter
decomposes into scalar ones and never reaches the branch at all.

Three things to take from that:

- **A per-backend if/else chain in shared code is where HIP goes missing.** §9
  warns about this in general; `use_packet_op` is a live instance, and it fails
  open (correct, slower) rather than loudly. The omission is now commented at
  the site rather than left to be rediscovered.
- **Name a test after what it exercises, not what you meant it to.** The check
  above is worth keeping -- the decomposed path really is what HIP runs, and it
  really is correct -- but calling it "CAS path" was a lie that would have
  outlived the memory of writing it.
- **`ReduceOp::Mul` never reaches any backend.** `jitc_can_scatter_reduce()`
  rejects it outright ("no multiplication reduction atomics"), so the `Mul` arm
  in `render_scatter_reduce()` is dead code. Do not reach for it as a test
  vector.

**`jitc_can_scatter_reduce()` had no HIP arm either**, which was worse than the
packet one: the generic answers permit `Float16` atomics, while
`jitc_hip_render()` fails on them (there is no 16-bit `atomicCAS`, hence no
correct lowering). Capability tables that over-promise turn a graceful fallback
into an abort halfway through codegen. It now says no, and the packed `f16x2`
treatment stays on the Phase 4 remainder list.

## 11i. Phase 5 — validating an emitter you cannot execute

Ray tracing had a validation problem the other phases did not. The generated
kernel `#include`s `<hiprt/hiprt_device.h>`, which the CUDA shim's NVRTC cannot
find, so `jit_var_eval()` on a traced graph aborts before anything runs. The
usual end-to-end check is unavailable.

The resolution is to validate the **emitted text** rather than the emitter's
effects, and to do it against the real target:

1. `tests/hip_trace.cpp` sets `JitFlag::PrintIR` and installs a log callback.
   Codegen routes the assembled source through it *before* handing it to the
   compiler, so the callback sees the genuine artifact.
2. The callback asserts the structural properties — the HIP-RT include, both
   hook definitions, `...TraversalAnyHit` for shadow versus `...Closest`
   otherwise, the five hit fields, the two table lookups — writes the source
   out, and **exits from inside the callback**, because the compile that
   follows aborts the process via `jitc_fail()` rather than throwing.
3. `run_tests.sh` compiles that file for real `gfx90a` with the HIP-RT device
   library linked.

Step 3 is the one that matters. HIP-RT declares `intersectFunc` / `filterFunc`
and leaves the definitions to the application (§7a finding 3), so a backend
that omits them emits source that compiles cleanly on every platform anyone
here has and fails at **link** time on hardware nobody here has. Nothing short
of a real gfx90a link catches that.

Measured footprint of the emitted closest-hit traversal: 64 VGPR / 38 AGPR /
54 SGPR / 784 B scratch — within noise of the 800 B §7a measured for a
hand-written minimal traversal, which is the check that the traversal really
was linked in rather than optimised away.

Exiting from a log callback is strange enough to read as a bug later, so it is
commented at length in the test.

### And then the shim learned to run them

`jitc_hip_compile()` now routes any kernel whose source contains the HIP-RT
include through `hiprtBuildTraceKernels()` instead of bare NVRTC. That call
compiles *and links* the traversal library, and on NVIDIA it drives NVRTC
underneath — so `tests/hip_trace_exec.cpp` builds a real BVH, traces through
`jit_hip_ray_trace()`, evaluates, and checks the hits: 32 lanes inside a
triangle hit at t = 1.0, 32 outside miss, and `geometry_id` arrives from the
instance-indexed table.

Phase 5 is therefore emitted, linked for gfx90a, **and executed**. What is
still owed to the MI210 is wave64 and the AMD host API — not traversal
correctness, because gfx90a has no RT hardware and takes the same software path
NVIDIA just ran (§7b).

Two things this cost, both worth remembering:

- **The stub correction above.** The end-to-end test is what surfaced it; the
  emitted-source check could not, because it validates against the harness's
  link path, which wants the opposite.
- **`HIPRT_PATH` is read at context creation** and the dev shell points it at
  the AMD build for `hip_validate`'s benefit. The shim overrides it to the
  CUDA-enabled build before calling `hiprtCreateContext()`, or context creation
  fails as `hiprtErrorInternal` with the missing filename visible only after
  `hiprtSetLogLevel()` — which the shim now enables unconditionally, because it
  costs nothing and is the difference between a five-minute fix and an
  afternoon.

### `jitc_var_pointer()` takes a reference on its `dep`

Worth its own heading, because the failure is silent in every direction.

`HIPScene` caches the pointer variables it hands the emitter, so that several
traces against one scene share a kernel parameter. The obvious `dep` for those
handles is the scene variable — and it closes a cycle: the scene owns the
handle, the handle references the scene, the reference count never reaches
zero, the cleanup callback never fires, and the application waits forever for
permission to free its BVH. Nothing errors. Nothing shows up as a leak at
`jit_shutdown()`, because from drjit's point of view the variable is still
legitimately alive.

The right `dep` here is **0**. That argument exists to keep a *drjit
allocation* alive while a pointer to it is in flight; a `hiprtScene` and the
two ID tables are the application's memory, and their lifetime is exactly what
the cleanup callback is for.

`tests/hip_trace.cpp` checks this directly — configure a scene, issue a trace so
the handles are actually created, drop the reference, assert the callback ran.
Verified by reinstating the cycle and watching it fail. Any future per-scene
resource cached this way needs the same treatment.

## 11j. Layer B — every failure was silent

Phase 6 (`drjit.hip` / `drjit.hip.ad`) is mechanical work with one lesson worth
carrying into Layer C: **not one of the bugs announced itself.** Each produced a
plausible-looking system that was simply wrong somewhere else.

`drjit.hip` came out empty — `module 'drjit.hip' has no attribute 'Float'` —
and that single symptom had **two independent causes, either of which would
have produced it alone**:

1. `detail::backend<T>` in `array_traits.h` specialises for CUDA, LLVM and
   Metal. With no HIP arm the primary template applies and
   `backend_v<HIPArray<float>>` is `JitBackend::None`, so `drjit::bind()` files
   every HIP type under `drjit.scalar`.
2. `ArrayMeta::backend` was a **2-bit** field. `JitBackend::HIP` is 4, which
   truncates to 0 — the same wrong answer by a different route.

Fixing one alone would have left the symptom unchanged and sent the search in
the wrong direction. This is the **third** packed backend field caught one bit
too narrow (`Variable::backend`, `AllocInfo`, now `ArrayMeta::backend`), so
that one now carries `static_assert((uint32_t) JitBackend::Count <= 8, ...)`
beside it.

### Widening ArrayMeta was not free, and the fallout was worth having

`ArrayMeta`'s bitfields filled a `uint32_t` exactly. Neither neighbour could
give up a bit — `tsize_rel` reaches exactly 64 for
`Array<Array<Array<JitArray<float>,4>,4>,4>`, and `talign` reaches 64 for
AVX-512-aligned types. (Both bounds were found by narrowing them and reading
the `static_assert` that fired, which is the cheap way to answer this.)

So the struct is 12 bytes now, and the two places that assumed 8 —
`operator==`'s whole-struct `memcmp` and `meta_get_type()`'s `uint64_t` cache
key — had to change. They now share a `meta_identity()` helper, which is
**more** correct than what it replaced: with 33 bits of bitfields the second
word is nearly all padding, and padding is indeterminate, so memcmp'ing the
whole struct would have been a latent bug regardless of HIP.

### Thirteen per-backend chains in shipped code, plus six in the tests

Sites in shared code that enumerated backends and omitted HIP, in this phase
alone: `detail::backend`, `interop.py::_migrate_backend` (which sent anything
not CUDA or Metal to **LLVM**), `math.cpp::is_gpu`, `dlpack.cpp` and
`init.cpp`'s "is this device memory" tests, `meta.cpp`'s module index and
`__meta__` string, `freeze.cpp`'s diagnostic names, `main.cpp`'s `JitBackend`
enum binding and `array_submodules` table, `base.cpp`'s backend test,
`quat.cpp`'s export list, `coop_vec.cpp`'s capability table, and the three
layers of `Resampler` described below.

The dlpack one is the instructive failure: HIP fell through to *host*, so a
device pointer would have been handed to a consumer labelled CPU — a segfault
in somebody else's process. The device tests now go through
`is_device_backend()` in `src/python/common.h`, so there is one place to add to
rather than five.

**For Layer C, grep before building.** `JitBackend::CUDA`, `is_cuda_v`,
`MI_ENABLE_CUDA` and `backend == ` are the patterns; the ones that read as
"this is a GPU" rather than "this is CUDA specifically" are the ones that will
be wrong.

### One feature, four layers, and the one that hides

`Resampler` was missing HIP in **four** places at once, and they fail
differently:

| Layer | File | Symptom if only this one is missing |
|---|---|---|
| Python binding | `src/python/resample.cpp` | `TypeError` on a HIP argument |
| Explicit instantiation | `src/extra/resample.cpp` | link error, named symbol |
| **CMake define** | `src/extra/CMakeLists.txt` | link error, and the source looks correct |
| Type export | `src/python/quat.cpp` | attribute missing from the module |

The CMake one is the trap. `src/extra/CMakeLists.txt` re-derives
`-DDRJIT_ENABLE_HIP` per target, and without it the instantiations sit inside a
`#if` that is false — so the code is right, reads as right, and is compiled out.
The error is an `undefined symbol` for a template you can see being
instantiated. **When a link error names a symbol whose definition is plainly
there, check whether that target got the define.**

### Capability tables must default to "unsupported"

`jitc_coop_vec_supported()` asked only about CUDA and returned `true` otherwise.
HIP implements no `CoopVec*` opcode at all, so the "supported" path ran and hit
`jitc_fail()` — which **aborts**. A whole pytest process died on one unsupported
feature instead of reporting a skip. Same shape as `jitc_can_scatter_reduce()`'s
missing HIP arm in Phase 4.

Two rules fall out:

- **A capability table whose fallthrough is "yes" is a trap for every backend
  added after it was written.** Ask the negative question first.
- **Don't reach `jitc_fail()` for a capability question.** `jitc_fail` aborts;
  `jitc_raise` throws and becomes a catchable Python exception. Unsupported is
  not a bug in the caller.

The corollary is that the *test* suite must ask the library rather than
enumerate backends. `test_coop_vec.py` and `test_nn.py` each carried their own
`skip_if_coopvec_not_supported()` re-deriving the CUDA and LLVM version gates by
hand, so HIP was "supported" in both. Both now call
`dr.detail.coop_vec_supported()`, and `jitc_coop_vec_supported()` answers for
LLVM (>= 17.0) too so there is exactly one authority. The raise message no
longer hardcodes "CUDA/OptiX" either — it named a CUDA driver requirement for a
failure in which CUDA was never involved.

Other test-side backend enumerations found the same way: `test_traits.py`'s
`is_jit` list (272 failures from one missing string) and three sites in
`test_init.py` asserting that a backend aliases host `ndarray` memory rather
than copying — true only of LLVM. Those now name LLVM positively, so the next
device backend takes the safe branch by default.

### `@dr.freeze` needed no work — check parity, not absolutes

A standalone test asserting that a frozen function traces once and then replays
reported four traces on HIP. It reports four on CUDA and LLVM too. The
expectation was wrong, not the backend, and an absolute assertion would have
sent someone hunting a HIP bug that does not exist. Compare against a reference
backend, and let upstream's own `test_freeze.py` be the real check.

## 11k. Never log above Info during backend init

This one cost most of a day, so it gets its own section.

**Symptom.** Every test passed. Results were correct and printed. The process
then never exited — it sat at 100% CPU after the last line of output, and had to
be killed.

**Cause.** `jit_init_async()` runs backend initialization on a **background
thread that holds `state.lock`**. The HIP shim logged its scaffold warning at
`Warn` level. Warn-level messages reach drjit's Python log callback, which
acquires the **GIL**. Meanwhile the main thread holds the GIL and is waiting on
`state.lock`. Classic lock-order inversion; it only becomes fatal at interpreter
shutdown, when `Py_Finalize` joins the thread that can never make progress.

**Why it was hard.** The deadlock happens *after* all useful work, so every
symptom points at code that ran before it and completed fine. I misread it as an
import hang more than once. What settled it was a backtrace showing
`atexit_callfuncs` → `_Py_Finalize` — i.e. the process was already past the end
of the program.

**Getting that backtrace on this machine:** `ptrace_scope=1` means you cannot
attach gdb to a running process; you must *launch* under gdb.

```
gdb -q --args python3 -m pytest ...     # attach won't work; launch will
thread apply all bt
```

**Rules.**

- Backend `*_init()` runs on a lock-holding background thread. Log at `Info`,
  never `Warn` or above. Info does not reach the Python callback at the default
  log level, which is why every other backend logs freely there.
- If you need a message the user will actually see, emit it from
  `jitc_init_thread_state()` instead — that runs on the caller's own thread, and
  fires when the backend is *used* rather than merely linked in, which is the
  more useful moment anyway.
- **A process that produces correct output and then hangs is a shutdown bug, not
  a startup bug.** Check `Py_Finalize` before re-reading the code that worked.

## 11m. A skipped test is not a passing test

The suite reported "green" while **497 tests never ran** — the seven C++
extension suites (`test_call_ext`, `test_memop_ext`, `test_local_ext`,
`test_while_loop_ext`, `test_if_stmt_ext`, `test_custom_type_ext`,
`test_py_cpp_consistency_ext`). They print as `s`, which reads identically to
"not applicable on this backend".

Three independent reasons, each sufficient, stacked on top of each other:

1. `DRJIT_ENABLE_TESTS` defaults to **OFF**, so the extension modules did not
   exist and `pytest.importorskip` skipped everything.
2. `tests/CMakeLists.txt` passed `-DDRJIT_ENABLE_LLVM/CUDA/METAL` to each
   module but **not** `-DDRJIT_ENABLE_HIP` — the same missing-define as
   `src/extra/CMakeLists.txt`. The HIP submodule would have been compiled out
   even with the tests built.
3. All seven `get_pkg()` helpers mapped backend → submodule with an if/elif
   chain ending at Metal, so HIP fell off the end and `get_pkg()` returned
   **None**. Every test would then have failed on `None.A`, pointing at the
   test body rather than at the missing arm.

Fixed at all four levels — four, because after replacing the seven `get_pkg()`
chains there was an **eighth** copy inside `test_freeze.py`, which is not an
`_ext` file and so was not on the list. It surfaced only after the other three
fixes let those tests run at all, as `'NoneType' object has no attribute 'A'`
across 40 frozen-vcall tests. The chains are gone now: `conftest.py` has one
`get_backend_submodule()` that looks the submodule up by the backend's own name.

When you fix a duplicated pattern, grep for the *pattern* (`def get_pkg`), not
for the files you expect to contain it.

**These suites matter for Layer C** — `call_ext` is the C++ side of vcall
dispatch, which §9a-c calls the risky part of the Mitsuba port. Unlocking them
also unlocked ~120 tests inside `test_freeze.py` that use `call_ext`, including
the frozen-vcall set (`test23/24/25`, `test59/60`, `test86/87`, `test95`). All
of it passes on HIP under the shim — **the first direct evidence that HIP vcall
dispatch works end to end**, which is the single most useful result to carry
into Phase 7.

Only two things needed fixing once these ran, both test-side assumptions rather
than backend bugs: `cleanup()` stripped `.llvm.`/`.cuda.`/
`.metal.` from reprs but not `.hip.`, and `test14_array_call_self` assumed any
backend that is not LLVM or Metal emits PTX. HIP emits source text, so `self`
appears as `u32 rN = self;` (hip_eval.cpp) rather than in a PTX parameter list;
that assertion now keys on what the backend *emits* and defaults to the source
form.

**Run the sweep from `build-hip/tests`, not the source tree** — that is where
the `.so` files live, and it is the difference between 497 tests running and
497 tests silently skipping. Note that CMake copies the `.py` files there with
`configure_file(COPYONLY)`, so **editing `tests/*.py` requires re-running cmake**
before the change reaches the copy you are executing.

## 11l. De-duplicating a per-backend chain can reverse a load-bearing order

The one bug in this port that was **caused by removing** a per-backend chain
rather than by a chain missing an arm. Worth reading before doing the same
refactor in Layer C.

`jitc_freeze_start/stop/abort` each carried the same CUDA/Metal/LLVM if/else
chain for swapping thread-state slots, and HIP fell into the trailing `else`,
which assigned the recorder to the **LLVM** slot. The fix — enumerate the slots
once in `for_each_thread_state_slot()` and drive all three sites from it — is
right, and it is what `swap_in_record_ts` / `swap_out_record_ts` do now.

But the original chain restored *the recording backend's own slot first* and
disabled the others afterwards. That ordering was not commented and looked
incidental. It is not:

- `unset_disabled_thread_state()` **rethrows** the exception a disabled slot
  recorded — the normal outcome, since catching a frozen function that touched
  the wrong backend is precisely what disabling is for.
- A single loop over the slots visits LLVM before CUDA. So when freezing
  against backend CUDA, the LLVM slot throws *before* the CUDA slot is restored.
- The CUDA slot is then still pointing at the `RecordThreadState`, which
  `RecordThreadStateGuard` deletes on the way out.
- Result: a **dangling, non-null** thread state. Nothing fails yet. The next
  `thread_state(CUDA)` hands it back and `dynamic_cast` faults reading a freed
  vptr — in a later test, in a function that has nothing to do with the bug.

`swap_out_record_ts()` now runs two passes: restore this backend's slot, then
re-enable the rest, holding the first exception so that a throw cannot skip the
remaining slots. (The old chain leaked the Metal slot for the same reason; the
two-pass form fixes that too.)

**The general shape:** when you replace N copies of a chain with one loop, the
copies may not have been identical. Diff them against each other, not just
against your replacement — an ordering that appears in every copy is a
specification, not a coincidence.

### Two corollaries about how this was found

**A crash masks everything after it.** test72 aborted `test_freeze.py` at 94%,
hiding two further crashes: `jit_coop_vec_pack_matrices` had no capability guard
(`nn.pack()` reaches it without ever calling `jitc_coop_vec_pack`, so guarding
only the latter left `nn.pack` aborting), and `test_freeze.py` turned out to
hold a **third** hand-rolled `skip_if_coopvec_not_supported`. Fixing one crash
is not progress until you re-run and see what it was covering.

**"HIP off" is not "our changes off."** The crash reproduced in a build with
`DRJIT_ENABLE_HIP=OFF`, which read as proof it was upstream's. It was not: that
build still contained every Phase 1–5 drjit-core commit, including the refactor
above. Only a build at the true merge-base (drjit `9a7db92b` + drjit-core
`7a9ab1fa`) answered the question, and it passed 744 tests clean. A control has
to differ in exactly the variable under test — and this one nearly got written
up as somebody else's bug.

(Aside, if you ever build that control: upstream at `9a7db92b` does **not**
compile with `DRJIT_ENABLE_CUDA=OFF` — `optix_api.h` names `CUcontext`,
`CUstream` and `CUdeviceptr` unguarded. Configure the control with CUDA on.)

## 11n. Layer C — what HIP-RT's scene model forces, and what it does not

Phase 7 (`hip_ad_rgb`) took ~700 lines across Mitsuba and one 20-line accessor
in drjit-core. It renders a mesh scene matching `llvm_ad_rgb` to **1.19e-07**.
That number is the correctness argument, not a smoke test: LLVM traces through
Embree and certainly sees the geometry, so HIP-RT agreeing to single-precision
epsilon means it is intersecting rather than quietly missing. A backend that
reports "no hit" for everything also produces a stable image.

### One structural difference, and it is in the scene model

`hiprtHit` is `{instanceID, primID, uv, normal, t}` — verified in
`hiprt_types.h`, not assumed. **There is no geometry ID**, because a
`hiprtInstance` references exactly one geometry, so "which geometry" is a
property of the instance. Metal and OptiX both nest geometries inside an
instance and report the pair per hit.

So a `BlasEntry` holding N same-kind geometries cannot become one instance.
`build_hip_accel()` expands each (instance, geometry) pair into its own HIP-RT
instance and returns the geometry index through the instance-indexed table
`jit_hip_configure_scene()` already accepted. By the time the values reach
Mitsuba they are exactly Metal's — which is why `scene_hip.inl` reuses
`scene_metal.inl`'s recovery logic verbatim rather than reimplementing it.

Two smaller obligations that fail silently if missed:

- **Borrow the context, never create one.** `hiprtGeometry` / `hiprtScene`
  handles are context-scoped. A scene built against a second context does not
  error — it traverses garbage. `jit_hip_rt_context()` exists for this, as
  `jit_metal_context()` does for Metal.
- **Allocate through `jit_malloc(JitBackend::HIP, ...)`, not `hipMalloc`.**
  Under the shim "HIP" is CUDA-backed, so `hipMalloc` would work on the MI210
  and fail on the development machine.

### `to_world` needs a transpose that nothing will catch

SceneIR stores the affine column-major as four columns of three floats —
element (row, col) is `to_world[col * 3 + row]` (confirmed against
`metal_accel.mm`, not from the comment alone). `hiprtFrameMatrix` is
`matrix[row][col]`. Get this backwards and geometry lands somewhere plausible
but wrong, which no assertion catches and a render only hints at. It is spelled
out as a nested loop rather than memcpy'd for that reason.

### The build wiring, where four of six bugs lived

- Mitsuba never forwarded `DRJIT_ENABLE_HIP`, so it compiled `MI_ENABLE_HIP`
  code against a Dr.Jit with no HIP backend.
- It read drjit-core's `DRJIT_HIPRT_PATH`, which is only set inside the
  **CUDA-shim branch** — so the check would have worked on this laptop and
  failed on the MI210. **The shim's usual risk is flattering the developer;
  this was the reverse, and it is the shape to watch for.**
- `PRIVATE` link options on an **OBJECT** library never reach the consuming
  link (`cannot find -lhiprt64`). Use an absolute `.so` path and `PUBLIC`.
- `drjit_v.cpp`'s backend-name chain stopped at Metal, so a HIP variant would
  have imported `drjit.scalar`.

### The color tables: caught by luck

`get_color_space_tables<Float>()` had no HIP arm, so it fell through to the
**scalar** tables. This produced a compile error only because the types differ
(`gather_(DynamicArray<float>&, ...)` with a HIP index). The identical omission
in a non-template context is a host pointer handed to a device kernel, silently.
It is now the worked example in `mitsuba::is_gpu_v`'s documentation of where
*not* to use the trait.

### `is_cuda_v` is four questions, not one

The plan estimated "~20 sites to generalize". There are 31, and they ask
different things — treating them as one class is how a resource selection gets
collapsed onto a capability trait:

| Count | Question | Fix |
|---|---|---|
| 15 | "CPU path?" / "host-addressable memory?" | `mitsuba::is_gpu_v` |
| 4 | genuinely OptiX-specific | unchanged, already inside `MI_ENABLE_CUDA` |
| 2 | per-backend **resource** (color tables, module name) | a real HIP arm |
| ~2 | `sphere`/`cylinder` precision policy | **left alone** |

The last row is worth the ink: they test `is_cuda_v<FloatP>` where `FloatP` is a
**packet** type, so the predicate is always false, and GPU variants never reach
packet methods anyway (`shape.h` throws). Dead code. Changing it would only
perturb numerics for no benefit — *not every match for your grep is a bug*.

### 11n.1 The open bug: hiprtBuildTraceKernels on loop/vcall kernels

Every remaining crash in the Mitsuba suite on HIP has **one** root cause, and it
is worth stating precisely because the surface symptoms look unrelated:
`test_ad::test01_bsdf_reflectance_backward`, `test_aov::test06_..._ad_backward`,
`test_ad_integrators::test01_rendering_primal`, `test_freeze`, and
`test_mesh::test14` all die inside `jitc_hip_compile()`.

What is established:

- The failure is `hiprtBuildTraceKernels()` returning **`hiprtErrorInternal`
  (2)**, or segfaulting *inside itself* — the faulthandler trace names
  `libhiprt0300064.so` frames beneath `hiprtBuildTraceKernels`. So HIP-RT's own
  builder is the thing failing, not our emitted text being rejected by a
  compiler that then reports a diagnostic. **No compiler diagnostic is ever
  printed**, which is §7b's "fails late and unhelpfully" exactly.
- It is **not** cumulative resource exhaustion. A loop building and destroying
  12 scenes, tracing each, is clean. Eight *distinct* traced kernels in one
  process are clean.
- It **is** specific to kernels that contain a trace **and** come from
  `ad_loop()` (a symbolic loop) or `jit_var_call_reduce()` (a vcall). Those are
  exactly the kernels a real integrator generates, which is why a direct
  `ray_intersect_preliminary` renders perfectly (1.19e-07 vs LLVM) while
  `mi.render()` through the path integrator does not.
- `test_mesh::test14` needs `test13` to have run first; run alone it passes. So
  some cross-test state changes which kernel is generated, not whether the bug
  exists.

**The emitted source is VALID — the emitter is no longer a suspect.** The exact
kernel drjit-core logs on failure was extracted and rebuilt standalone through
`hiprtBuildTraceKernels()` with argument-for-argument identical parameters
(`scratchpad/rtbuild.cpp`): **it succeeds**. Same text, same options, same
`numGeomTypes`/`funcNameSets`, fresh process. So this is not a codegen bug, and
"kernels from symbolic loops and vcalls are miscompiled" — which is what the
symptom looked like — is the wrong description. The cause is **process state**.

Eliminated so far, each by direct experiment rather than reasoning:

| Hypothesis | Test | Result |
|---|---|---|
| Invalid emitted HIP | rebuild the logged source standalone | **builds fine** |
| Cumulative scene churn | 12 build/destroy cycles + trace | clean |
| Many distinct kernels | 8 distinct traced kernels | clean |
| Scenes interleaved with distinct kernels | 8 of each, alternating | clean |
| Missing CUDA context binding | added `scoped_set_context` (a real bug, committed) | **still crashes** |
| Device memory exhaustion | `nvidia-smi` trace during the failing run | **5.6 GB of 6 GB free at crash** |

**Working assessment: a probable shim artifact, not yet proven.** What remains
is HIP-RT-on-CUDA through Orochi — the least-travelled path in the stack, and
the one AMD does not test. Against that, `hiprtBuildTraceKernels()` *is* the API
real hardware compiles through (§7a), so this cannot be dismissed. It is
deliberately parked rather than closed: the honest position is that we do not
know, and finding out costs less on an MI210 than it does here.

**If you pick this up, the next probes are:** build the same kernel twice
against one `hiprtContext` in `rtbuild.cpp` (tests repeat-build in one context,
which the in-process case does and the standalone case does not); and bisect
`test13` to find which of its operations arms the failure, since `test14` alone
passes.

## 12. Suggested implementation order

1. Skeleton + `Params` + the variable loop, arithmetic opcodes only. Validate
   with `hip_validate` (both arms) from the first commit.
2. Compare/select, then memory (gather/scatter).
3. Control flow — loops and `if`. Cross-check against `VarKind::CondStart`
   handling near `metal_eval.cpp:850-914`.
4. Calls and callables, including the forward-declaration move.
5. `TraceRay` against HIP-RT.
6. Textures, arrays, packet memory.

Steps 1–4 are fully verifiable on the local NVIDIA box. Step 5 needs the MI210.
Anything touching wave-width semantics goes on the wave64-unverified list
(§0.3, §7.2) regardless of which step introduces it.
