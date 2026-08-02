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

**3. Every traversing kernel MUST define `intersectFunc` and `filterFunc`.**
HIP-RT declares them and leaves the definitions to the application — the custom
-primitive intersection and any-hit filter hooks, analogous to Metal's
intersection function table and OptiX's IS/AH programs. Omit them and the link
fails with `undefined hidden symbol: intersectFunc(...)`. **The emitter must
emit at least stubs into every kernel that traverses**, and dispatch through
them for scenes with custom primitives. Nothing in §7 anticipated this.

**4. `hiprtHit` does not carry a geometry ID or a user instance ID.** It has
`hasHit()`, `t`, `uv.x`, `uv.y`, `primID`, `instanceID` — six of Metal's eight
outputs. `geometry_id` and `user_instance_id` have to be reconstructed
application-side (an indexed table, most likely). This is a **contract
difference**, so §7's "adopt `jit_metal_ray_trace`'s signature verbatim" needs
qualifying before the interface is fixed.

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
