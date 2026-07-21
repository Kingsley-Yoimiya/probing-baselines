#!/usr/bin/env bash
# Build the XPUTimer MetaX detection hook with plain g++ (no Bazel).
# Run this INSIDE a muxi worker pod (needs /opt/maca headers + g++).
set -euo pipefail

MACA="${MACA_PATH:-/opt/maca}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/libxpu_timer_metax.so}"

# We deliberately do NOT include the MACA headers by default: the hook declares
# the minimal ABI itself so g++ can build it without pulling the __device__/
# clang-maca intrinsics that only the maca clang understands. The runtime
# symbols (mcLaunchKernel, mcEvent*, mccl*) are resolved at load time from the
# process image (torch already links libmcruntime/libmccl), and the originals
# via dlsym(RTLD_NEXT,...).
CXX="${CXX:-g++}"
echo "[build] CXX=$CXX MACA=$MACA -> $OUT"

$CXX -std=c++17 -O2 -fPIC -shared \
  -o "$OUT" \
  "$HERE/xpu_timer_metax_hook.cc" \
  -ldl -lpthread

echo "[build] done: $OUT"
ls -la "$OUT"
echo "[build] exported interposers:"
nm -D "$OUT" | grep -E " T (mcLaunchKernel|mccl)" || true
