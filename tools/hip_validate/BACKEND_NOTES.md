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
