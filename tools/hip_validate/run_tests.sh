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

  if "$BIN" "$k" >/dev/null 2>&1; then rc=0; else rc=1; fi

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

echo "  ---- $pass passed, $fail failed"
[ "$fail" -eq 0 ]
