# hip_validate — dual-path validation for emitted HIP kernels

Implements the harness described in **PLAN.md §0.3**. It exists so the HIP
backend's codegen can be developed on an ordinary NVIDIA workstation, with no
MI210 and no AMD hardware of any kind.

## Why two arms

One emitted kernel, checked two independent ways:

```
              hip_eval.cpp emits HIP C++ source
                            |
        +-------------------+-------------------+
        |                                       |
hipcc --offload-arch=gfx90a          NVRTC -> cuModuleLoadData
--genco   (compile only)             -> cuLaunchKernel  (executes)
        |                                       |
  Valid for the REAL target?          Are the NUMBERS right?
  Address spaces, wave64              Runs on the local NVIDIA GPU
  intrinsics, ISA validity.           and checks the output.
```

Codegen fails in two ways — *invalid for the target* and *valid but wrong* —
and neither arm catches both. The gfx arm never runs the code; the exec arm
never sees AMD. Run both.

## The kernel contract

**Emitted kernels contain no platform includes.** The harness prepends
`prelude_hip.h` or `prelude_nvrtc.h` depending on the arm. That is precisely
what makes a single source checkable twice. See `prelude_portable.h` for the
macro surface an emitter may use; it is documentation and will `#error` if
included.

Signature:

```c
DRJIT_KERNEL void k(float *out, const float *in, unsigned n)
```

Input is generated deterministically, so results are reproducible across arms
and runs.

If a construct cannot be expressed through the prelude macros, that is a signal
it needs an abstraction — not an `#ifdef` at the emission site. Keeping platform
divergence confined to the preludes is the smallest concrete instance of the
emit abstraction PLAN.md §3.1 requires.

## Build and run

Standalone by design — it does **not** build drjit-core, and adds nothing to the
upstream seam we carry across merges (§9). Builds in about a second.

```sh
nix develop /path/to/mitsuba3#hip      # provides hipcc, device libs, NVRTC, bundler
cmake -S tools/hip_validate -B build-hip -G Ninja
cmake --build build-hip

./build-hip/hip_validate tools/hip_validate/kernels/smoke_arith.hip
```

Neither libnvrtc nor libcuda is linked — both are `dlopen`ed, mirroring how
drjit-core resolves libcuda in `src/cuda_api.cpp`. The tool therefore builds
anywhere and degrades to whichever arm the host can actually run, which is what
makes it still useful on the MI210 host later, where there is no NVIDIA GPU.

Options: `--entry NAME`, `--arch ARCH`, `--n N`, `--ref FILE`, `--tol T`,
`--no-gfx`, `--no-exec`, `--dump-ptx`, `--dump-isa`, `-v`.

Exit status is 0 only if every arm that ran passed.

## The wave64 caveat

The execution arm runs on NVIDIA hardware at `DRJIT_WARP_SIZE == 32`. Kernels
using wave operations are therefore checked only for *structure* — **width-64
semantics are not verified**. Ballots do not fit a `b32` and shuffle lane masks
differ.

This is not a corner case: §0.2 measured 24 `shfl` plus `vote` and `activemask`
in a Cornell box at 1 spp, so wave ops are in the hot path of every render. The
harness prints a reminder when a kernel uses them. Keep those kernels on the
wave64-unverified list until MI210 access (PLAN.md §0.3, §7.2 — now the
project's highest risk since §7.3 closed).

## Sample kernels

| File | Purpose |
|---|---|
| `kernels/smoke_arith.hip` | Arithmetic only. Fully verified by both arms. |
| `kernels/smoke_warp.hip` | Wave butterfly reduction. Structure verified; width-64 semantics are not. |
| `kernels/broken_syntax.hip` | Deliberately invalid — tests the tester. Must fail both arms. |

## Notes

Two things cost time to discover; both are handled but worth knowing.

**Offload bundles are compressed**, so a code object's target string cannot be
found by grepping. Confirming the bundle really carries `gfx90a` requires
`clang-offload-bundler --list` (`$HIP_BUNDLER`), which in turn shells out to
`llvm-objcopy` and needs it on `PATH`. Without `$HIP_BUNDLER` the harness still
runs; hipcc rejects unknown targets outright, so a clean exit is already
meaningful, just weaker.

**hipcc needs `--rocm-device-lib-path`** on NixOS or it cannot find the device
bitcode. The `#hip` devShell sets this via `$HIP_DEVICE_LIB_PATH`.
