#!/usr/bin/env bash
#
# devlib_check.sh -- run drjit-core's OWN test suite against a device library
# rebuilt from the ported resources/ sources.
#
# Phase 3 rewrote resources/common.h and five .cuh files so they compile for
# both nvcc and hipcc (PLAN.md §3.3). That the gfx90a code object builds says
# nothing about whether the kernels still COMPUTE correctly, and neither does
# the normal test suite: drjit-core embeds the committed kernels_75.lz4, which
# predates the port, so every existing test exercises the OLD device library.
#
# This closes that gap without any hardware we do not have. It regenerates the
# CUDA blob from the ported sources, builds drjit-core against it, and runs the
# full suite. Reductions, scans, compress and the vcall bucketing that drives
# mkperm then all run through the ported kernels, on a real GPU.
#
# What it establishes and what it does not:
#
#   IT DOES  prove the ported sources are algorithmically intact -- including
#            the eight kernels whose PTX actually changed (seven mkperm plus
#            compress_large), which is where the wave-width rewrite bites.
#   IT DOES NOT prove anything about width 64. NVIDIA is warp-32, so this
#            exercises the width-parametric code at the width it was already
#            correct for. Wave64 semantics stay on the needs-an-MI210 list.
#
# The committed kernels_75.lz4 / kernels.h / kernels_dict.lz4 are swapped out
# for the duration and restored from git on exit, including on failure. The
# script refuses to start if they are not clean, so an interrupted run can
# never be mistaken for a source edit.
#
#   nix develop --impure /path/to/mitsuba3#hip
#   tools/hip_validate/devlib_check.sh

set -euo pipefail

CORE="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$CORE/resources"
BUILD="${DEVLIB_CHECK_BUILD:-$CORE/build-devlib}"
GENERATED=(resources/kernels_75.lz4 resources/kernels.h resources/kernels_dict.lz4)

cd "$CORE"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "devlib_check: nvcc not found -- run inside 'nix develop .#hip'." >&2
    exit 2
fi
if [ -z "${CUDA_INCLUDE:-}" ] || [ -z "${CUDA_CCCL_INCLUDE:-}" ]; then
    echo "devlib_check: \$CUDA_INCLUDE / \$CUDA_CCCL_INCLUDE unset -- run inside 'nix develop .#hip'." >&2
    exit 2
fi

if ! git diff --quiet -- "${GENERATED[@]}"; then
    echo "devlib_check: ${GENERATED[*]} are already modified." >&2
    echo "  Refusing to run: the restore step would discard those edits." >&2
    exit 2
fi

WORK="$(mktemp -d)"
restore() {
    git checkout -- "${GENERATED[@]}" 2>/dev/null || true
    rm -rf "$WORK"
}
trap restore EXIT

echo "==> Compiling resources/kernels.cu with nvcc (compute_75)"
cp "$RES"/*.cu "$RES"/*.cuh "$RES"/common.h "$RES"/kernels_dict "$RES"/pack.c "$WORK/"
mkdir -p "$WORK/../ext"

# pack.c is built against drjit-core's vendored lz4, reached by relative path.
nvcc -I"$CUDA_INCLUDE" -I"$CUDA_CCCL_INCLUDE" -m64 --ptx --expt-relaxed-constexpr \
     -std=c++17 -Xcudafe --diag_suppress=550 -Xcudafe --diag_suppress=177 \
     -Xcudafe --diag_suppress=2912 -Wno-deprecated-gpu-targets \
     -gencode arch=compute_75,code=compute_75 \
     "$WORK/kernels.cu" -o "$WORK/kernels_75.ptx"

entries=$(grep -c '^\.visible \.entry' "$WORK/kernels_75.ptx")
echo "    $entries kernels"

echo "==> Packing"
cc "$WORK/pack.c" "$CORE/ext/lz4/lz4hc.c" "$CORE/ext/lz4/lz4.c" \
   "$CORE/ext/lz4/xxhash.c" -o "$WORK/pack" -I "$CORE/ext/lz4"
(cd "$WORK" && ./pack)

for f in kernels_75.lz4 kernels.h kernels_dict.lz4; do
    cp "$WORK/$f" "$RES/$f"
done

echo "==> Building drjit-core against the rebuilt device library"
if [ ! -d "$BUILD" ]; then
    cmake -S "$CORE" -B "$BUILD" -G Ninja \
          -DCMAKE_BUILD_TYPE=Release -DDRJIT_CORE_ENABLE_TESTS=ON >/dev/null
fi
# Not piped: a pipeline reports the exit status of its LAST command, which has
# silently turned failed builds into passes here before.
cmake --build "$BUILD" >"$WORK/build.log" 2>&1 || {
    echo "devlib_check: build FAILED"; tail -30 "$WORK/build.log"; exit 1
}

echo "==> Running the test suite against it"
rc=0
(cd "$BUILD/tests" && ctest -E graphviz) || rc=$?

if [ "$rc" -eq 0 ]; then
    echo
    echo "  devlib_check: PASS -- the ported device library is correct at width 32."
    echo "  Wave64 semantics remain unverified; that needs the MI210."
else
    echo
    echo "  devlib_check: FAIL -- the port changed kernel behaviour."
fi
exit "$rc"
