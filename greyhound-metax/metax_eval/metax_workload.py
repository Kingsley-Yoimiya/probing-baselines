#!/usr/bin/env python3
"""Minimal MetaX (MCCL) training-like workload for Greyhound detection-core eval.

Each process (one per GPU) runs a fixed pattern of collective operations per
"iteration", repeated for many iterations, so the probe's shared-memory ring
buffer fills with a periodic call sequence -- exactly what the detector's ACF
period-finding + change-point iteration-time estimation expect.

The pattern per iteration (mirrors a DP+TP-ish training step):
    AllReduce (gradient sync, large)
    AllGather (activation / param gather)
    ReduceScatter (grad reduce-scatter)
    AllReduce (large)

Fail-slow injection: with --slow-after N --slow-rank R --slow-ms M, rank R
sleeps M ms each iteration after iteration N, simulating a straggler.  The
detector should see rank R's per-iteration time step up at iteration N.

Launch with torchrun, e.g.:
    torchrun --nproc_per_node=4 metax_workload.py --iters 400 --size 8388608
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--size", type=int, default=4 * 1024 * 1024,
                    help="element count per collective (float32)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--slow-after", type=int, default=-1,
                    help="iteration index after which --slow-rank becomes slow")
    ap.add_argument("--slow-rank", type=int, default=-1)
    ap.add_argument("--slow-ms", type=float, default=0.0)
    ap.add_argument("--iter-sleep-ms", type=float, default=2.0,
                    help="baseline per-iter host sleep to keep a stable cadence")
    ap.add_argument("--hold-sec", type=float, default=0.0,
                    help="keep the process (and shm buffer) alive this long after "
                         "the last iteration, so detect_runner can snapshot shm")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # MetaX torch maps 'nccl' -> mccl

    dev = torch.device(f"cuda:{local_rank}")
    n = args.size
    x = torch.ones(n, dtype=torch.float32, device=dev)
    gathered = torch.empty(n * world_size, dtype=torch.float32, device=dev)
    rs_out = torch.empty(n, dtype=torch.float32, device=dev)
    rs_in = torch.ones(n * world_size, dtype=torch.float32, device=dev)

    if rank == 0:
        print(f"[workload] world_size={world_size} size={n} iters={args.iters} "
              f"slow_after={args.slow_after} slow_rank={args.slow_rank} "
              f"slow_ms={args.slow_ms}", flush=True)

    def one_iter():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        dist.all_gather_into_tensor(gathered, x)
        dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Warmup (comm creation + steady state); probe records these too but the
    # detector's period finder tolerates a lead-in.
    for _ in range(args.warmup):
        one_iter()
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.time()
    for it in range(args.iters):
        one_iter()
        # Inject a straggler on one rank after a certain iteration.
        if (args.slow_after >= 0 and it >= args.slow_after
                and rank == args.slow_rank and args.slow_ms > 0):
            torch.cuda.synchronize()
            time.sleep(args.slow_ms / 1000.0)
        if args.iter_sleep_ms > 0:
            time.sleep(args.iter_sleep_ms / 1000.0)
        if rank == 0 and (it + 1) % 50 == 0:
            torch.cuda.synchronize()
            print(f"[workload] iter {it + 1}/{args.iters} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"[workload] DONE {args.iters} iters in {time.time() - t0:.1f}s", flush=True)

    # Hold the shm buffer alive so the detector can snapshot it before the
    # probe's NcclRecordStorage destructor unlinks /dev/shm/ncclRecord.
    if args.hold_sec > 0:
        if rank == 0:
            print(f"[workload] holding {args.hold_sec}s for shm snapshot "
                  f"(run detect_runner.py now)", flush=True)
        time.sleep(args.hold_sec)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
