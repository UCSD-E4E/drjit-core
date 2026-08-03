#!/usr/bin/env bash
# Run hip_validate over every kernel in kernels/ and report.
#
# The fast inner loop for Phase 2 codegen (PLAN.md §0.3). Needs no AMD
# hardware: each kernel is compiled for real gfx90a AND executed on the local
# NVIDIA GPU.
#
#   nix develop /path/to/mitsuba3#hip
#   cmake -S tools/hip_validate -B build-hip -G Ninja && cmake --build build-hip
#   tools/hip_validate/run_tests.sh ./build-hip/hip_validate
#
# broken_syntax.hip is a NEGATIVE test: it must fail. If it ever passes, the
# harness has stopped detecting errors and every other result is worthless.

set -u
BIN="${1:-./build-hip/hip_validate}"
DIR="$(cd "$(dirname "$0")" && pwd)/kernels"

[ -x "$BIN" ] || { echo "usage: $0 <path-to-hip_validate>"; exit 2; }

pass=0; fail=0

for k in "$DIR"/*.hip; do
  name="$(basename "$k" .hip)"
  expect_fail=0
  case "$name" in broken_*) expect_fail=1 ;; esac

  # spec_* kernels are self-checking: they write 0 on success and a bit-per-op
  # error code on mismatch, so --expect-zero turns "it ran" into "it is right".
  extra=""
  case "$name" in spec_*) extra="--expect-zero" ;; esac

  # HIP-RT kernels are gfx-only. The packaged HIP-RT has device libraries for
  # AMD targets alone and its host library dlopens libamdhip64 with no CUDA
  # paths, so there is nothing for the execution arm to run -- this is a
  # property of the library, not a gap in the harness. Compiling and LINKING
  # for gfx90a is still the bulk of the risk (see spec_trace.hip).
  if grep -q "hiprt" "$k" 2>/dev/null; then
    extra="--no-exec"
  fi

  if "$BIN" $extra "$k" >/dev/null 2>&1; then rc=0; else rc=1; fi

  if [ "$rc" -eq "$expect_fail" ]; then
    printf '  %-20s PASS%s\n' "$name" \
      "$([ "$expect_fail" -eq 1 ] && echo '  (negative test failed as required)')"
    pass=$((pass+1))
  else
    printf '  %-20s FAIL%s\n' "$name" \
      "$([ "$expect_fail" -eq 1 ] && echo '  (negative test PASSED -- harness is not detecting errors!)')"
    fail=$((fail+1))
  fi
done

# The generated kernel prologue. Not a file in kernels/ because it has to be
# produced first -- the emitted text is not valid C++ until the kernel-name
# placeholder is substituted (see gen_prologue.cpp).
GEN="$(dirname "$BIN")/gen_prologue"
if [ -x "$GEN" ]; then
  "$GEN" 4 > /tmp/hip_validate_prologue.hip 2>/dev/null
  if "$BIN" --no-exec /tmp/hip_validate_prologue.hip >/dev/null 2>&1; then
    printf "  %-20s PASS  (generated, gfx arm only)\\n" "prologue"
    pass=$((pass+1))
  else
    printf "  %-20s FAIL  (generated prologue does not compile)\\n" "prologue"
    fail=$((fail+1))
  fi
fi

# --- Phase 1 API surface (HIP-on-CUDA, PLAN.md §3.5) ------------------------
#
# Built separately from the kernels: it uses the REAL HIP API and so needs the
# nvcc toolchain rather than hip_validate. Proves the whole JIT pipeline
# (hiprtc -> hipModuleLoadData -> hipModuleLaunchKernel) before hip_api.cpp is
# written against it.
if [ -n "${HIPNV_CFLAGS:-}" ] && command -v nvcc >/dev/null 2>&1; then
  SRC="$(dirname "$0")/hipnv_pipeline.cpp"
  if nvcc -x cu $HIPNV_CFLAGS -arch=sm_86 "$SRC" -o /tmp/hipnv_pipeline $HIPNV_LDFLAGS >/dev/null 2>&1 &&
     LD_LIBRARY_PATH="${CUDART_LIB}:${NVRTC_LIB}:$LD_LIBRARY_PATH" /tmp/hipnv_pipeline >/dev/null 2>&1; then
    printf "  %-20s PASS  (real HIP API on NVIDIA)\n" "hipnv_pipeline"
    pass=$((pass+1))
  else
    printf "  %-20s FAIL  (Phase 1 API surface broken)\n" "hipnv_pipeline"
    fail=$((fail+1))
  fi

  # src/hip_ts.cpp cannot run here (hipInit fails without an AMD GPU), but the
  # SHAPE of every call it makes can be: same sequences, real HIP headers,
  # executed through HIP-on-CUDA. Catches swapped arguments, wrong flag values
  # and element-vs-byte counts -- the mechanical port's failure modes.
  SRC="$(dirname "$0")/hipnv_ts_calls.cpp"
  if nvcc -x cu $HIPNV_CFLAGS -arch=sm_86 "$SRC" -o /tmp/hipnv_ts_calls $HIPNV_LDFLAGS >/dev/null 2>&1 &&
     LD_LIBRARY_PATH="${CUDART_LIB}:${NVRTC_LIB}:$LD_LIBRARY_PATH" /tmp/hipnv_ts_calls >/dev/null 2>&1; then
    printf "  %-20s PASS  (HIPThreadState call shapes)\n" "hipnv_ts_calls"
    pass=$((pass+1))
  else
    printf "  %-20s FAIL  (HIPThreadState call shape wrong)\n" "hipnv_ts_calls"
    fail=$((fail+1))
  fi
fi

# --- The EMITTED traversal, compiled for gfx90a (PLAN.md Phase 5) -----------
#
# spec_trace.hip checks a hand-written traversal. This checks the one codegen
# actually produces: tests/hip_trace writes the assembled kernel out, and it is
# compiled and LINKED here for real gfx90a.
#
# The intersectFunc/filterFunc definitions are prepended HERE rather than
# emitted by codegen, and which side owns them depends on how HIP-RT is linked:
#
#   * This harness hand-links the device bitcode (-mlink-bitcode-file), which
#     declares the hooks and leaves them undefined -- so the application must
#     supply them or the link fails with `undefined hidden symbol`.
#   * The backend compiles through hiprtBuildTraceKernels(), which GENERATES
#     them and prepends them itself -- so an emitted definition is a duplicate
#     and the build fails with "function has already been defined".
#
# Codegen targets the second, so the first is compensated for here. Getting
# this backwards is invisible until whichever path you did not test is run.
TRACE_BIN="${DRJIT_HIP_TRACE_BIN:-}"
if [ -z "$TRACE_BIN" ]; then
    for c in "$(dirname "$0")/../../build-shim/tests/hip_trace" \
             "$(dirname "$0")/../../build-wiring/tests/hip_trace"; do
        [ -x "$c" ] && { TRACE_BIN="$c"; break; }
    done
fi

if [ -n "$TRACE_BIN" ] && [ -x "$TRACE_BIN" ]; then
    for mode in closest shadow; do
        arg=""; [ "$mode" = shadow ] && arg=shadow
        raw="/tmp/hip_trace_emitted_$mode.hip"
        src="/tmp/hip_trace_linkable_$mode.hip"
        rm -f "$raw" "$src"
        # Not piped: the generator's exit status is the shape-check result.
        if DRJIT_HIP_TRACE_OUT="$raw" "$TRACE_BIN" $arg >/dev/null 2>&1 &&
           [ -s "$raw" ] &&
           { printf '#include <hiprt/hiprt_device.h>\n%s\n%s\n' \
               '__device__ bool intersectFunc(unsigned, unsigned, const hiprtFuncTableHeader &, const hiprtRay &, void *, hiprtHit &) { return false; }' \
               '__device__ bool filterFunc(unsigned, unsigned, const hiprtFuncTableHeader &, const hiprtRay &, void *, const hiprtHit &) { return false; }' \
               > "$src"; cat "$raw" >> "$src"; } &&
           "$BIN" --no-exec "$src" >/dev/null 2>&1; then
            printf "  %-20s PASS  (emitted traversal links for gfx90a)\n" \
                   "trace_$mode"
            pass=$((pass+1))
        else
            printf "  %-20s FAIL  (emitted traversal does not build)\n" \
                   "trace_$mode"
            fail=$((fail+1))
        fi
    done
fi

# --- HIP-RT traversal, actually executed (BACKEND_NOTES §7a) ----------------
#
# Needs $HIPRT_NV_PATH: the CUDA-enabled HIP-RT build, which the stock nixpkgs
# package is not. gfx90a takes HIP-RT's RTIP 0 SOFTWARE traversal path, so
# NVIDIA runs very nearly the same code the MI210 will -- which is what makes
# this worth having rather than a curiosity.
if [ -n "${HIPRT_NV_PATH:-}" ] && [ -n "${CUDA_INCLUDE:-}" ]; then
  SRC="$(dirname "$0")/hiprt_triangle.cpp"
  BIN=/tmp/hiprt_triangle
  if g++ -std=c++17 -I"$HIPRT_NV_PATH/include" -I"$CUDA_INCLUDE" "$SRC" \
        -L"$HIPRT_NV_PATH/lib" -lhiprt64 -L/run/opengl-driver/lib -lcuda \
        -o "$BIN" >/dev/null 2>&1 &&
     HIPRT_PATH="$HIPRT_NV_PATH" \
     LD_LIBRARY_PATH="$HIPRT_NV_PATH/lib:/run/opengl-driver/lib:${NVRTC_LIB:-}:$LD_LIBRARY_PATH" \
        "$BIN" >/dev/null 2>&1; then
    printf "  %-20s PASS  (BVH built + traversed on NVIDIA, hits verified)\n" "hiprt_triangle"
    pass=$((pass+1))
  else
    printf "  %-20s FAIL  (HIP-RT traversal broken)\n" "hiprt_triangle"
    fail=$((fail+1))
  fi
fi

# --- The wave64/fp16 unverified list (PLAN.md §0.3, §7.2) -------------------
#
# The execution arm runs at warp 32 and aliases fp16 to float, so kernels that
# touch either had their STRUCTURE checked but not their semantics. §7.2 asks
# for a running list rather than a discovery at integration time, so the runner
# prints one on every run instead of leaving it in prose.
unver=""
for k in "$DIR"/*.hip; do
  if grep -qE "DRJIT_SHFL|DRJIT_BALLOT|DRJIT_ACTIVEMASK|DRJIT_WARP_SIZE|DRJIT_HALF" "$k" 2>/dev/null; then
    unver="$unver $(basename "$k" .hip)"
  fi
done
if [ -n "$unver" ]; then
  echo
  echo "  UNVERIFIED ON NVIDIA (re-run these first on the MI210):"
  for u in $unver; do echo "    - $u"; done
  echo "    reason: exec arm is warp-32 and aliases DRJIT_HALF to float"
fi

# HIP-RT kernels never ran at all -- a stronger caveat than the wave64 list,
# and worth stating separately so the two are not conflated.
rt=""
for k in "$DIR"/*.hip; do
  if grep -q "hiprt" "$k" 2>/dev/null; then
    rt="$rt $(basename "$k" .hip)"
  fi
done
if [ -n "$rt" ]; then
  echo
  echo "  COMPILED AND LINKED FOR gfx90a, NOT EXECUTED:"
  for u in $rt; do echo "    - $u"; done
  echo "    reason: hip_validate's exec arm is plain NVRTC with no HIP-RT."
  echo "    Traversal CORRECTNESS is covered separately by hiprt_triangle,"
  echo "    which builds a BVH and traces on NVIDIA via \$HIPRT_NV_PATH."
fi
echo

echo "  ---- $pass passed, $fail failed"
[ "$fail" -eq 0 ]
