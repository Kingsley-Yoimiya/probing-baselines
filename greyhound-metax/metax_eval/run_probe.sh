#!/usr/bin/env bash
# Launch the MetaX workload under the Greyhound MCCL probe (LD_PRELOAD).
#
# Env knobs:
#   NPROC      number of GPUs/ranks (default 4)
#   ITERS      iterations (default 400)
#   SIZE       elements per collective, float32 (default 8388608 = 32MB)
#   SLOW_AFTER / SLOW_RANK / SLOW_MS   fail-slow injection (default off)
#   PROBE_DIR  where libmcclprobe.so lives (default script's ../detector)
#   LOGDIR     run output dir (default /tmp/greyhound-metax/run)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="${PROBE_DIR:-$HERE/../detector}"
LOGDIR="${LOGDIR:-/tmp/greyhound-metax/run}"
mkdir -p "$LOGDIR"

NPROC="${NPROC:-4}"
ITERS="${ITERS:-400}"
SIZE="${SIZE:-8388608}"
SLOW_AFTER="${SLOW_AFTER:--1}"
SLOW_RANK="${SLOW_RANK:--1}"
SLOW_MS="${SLOW_MS:-0}"

PROBE_SO="$PROBE_DIR/libmcclprobe.so"
if [[ ! -f "$PROBE_SO" ]]; then
    echo "ERROR: $PROBE_SO not found; build it with 'make -f Makefile.metax' first" >&2
    exit 1
fi

# The probe needs these env vars (see detector/utils.hpp get_*_path()).
export NCCLPROBE_LOG_PATH="$LOGDIR"
export LOCAL_CONTROLLER_LOG_PATH="$LOGDIR/local_controller.log"
export GLOBAL_CONTROLLER_LOG_PATH="$LOGDIR/global_controller.log"
# Point the probe at MetaX's collective library for dlsym forwarding.
export MCCL_PATH="${MCCL_PATH:-/opt/maca/lib/libmccl.so}"
export NCCL_PATH="$MCCL_PATH"

# Clean any stale shm from a previous run.
rm -f /dev/shm/ncclRecord /dev/shm/recordLock 2>/dev/null || true

echo "[run] probe=$PROBE_SO nproc=$NPROC iters=$ITERS size=$SIZE" \
     "slow_after=$SLOW_AFTER slow_rank=$SLOW_RANK slow_ms=$SLOW_MS logdir=$LOGDIR"

# torchrun single node; RANK/LOCAL_RANK/WORLD_SIZE set per process.
LD_PRELOAD="$PROBE_SO" \
torchrun --nproc_per_node="$NPROC" --nnodes=1 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29555}" \
    "$HERE/metax_workload.py" \
    --iters "$ITERS" --size "$SIZE" \
    --slow-after "$SLOW_AFTER" --slow-rank "$SLOW_RANK" --slow-ms "$SLOW_MS" \
    --hold-sec "${HOLD_SEC:-0}" \
    2>&1 | tee "$LOGDIR/workload.log"

echo "[run] workload finished; shm has been released" \
     "(set HOLD_SEC>0 and run detect_runner.py during the hold window)"
