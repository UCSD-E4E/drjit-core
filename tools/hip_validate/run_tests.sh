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
  echo
fi

echo "  ---- $pass passed, $fail failed"
[ "$fail" -eq 0 ]
