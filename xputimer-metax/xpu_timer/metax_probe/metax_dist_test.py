"""
metax_dist_test: 2+ rank torch.distributed workload on MetaX C550 to exercise
the XPUTimer MetaX hook's collective interception (mcclAllReduce / mcclAllGather
/ mcclReduceScatter / mcclBroadcast) and, optionally, a stuck-collective fail-slow
injection.

Launch with torchrun, e.g. 2 ranks on 2 GPUs of one node:

  LD_PRELOAD=./libxpu_timer_metax.so XPU_TIMER_DUMP_DIR=/tmp/xpu_dist \
  torchrun --nproc_per_node=2 metax_dist_test.py --iters 100

Fail-slow injection (rank 1 skips one allreduce -> rank 0's matching allreduce
hangs waiting; the hook flags HANG on rank 0):

  LD_PRELOAD=./libxpu_timer_metax.so XPU_TIMER_HANG_TIMEOUT_MS=1000 \
  XPU_TIMER_DUMP_DIR=/tmp/xpu_dist \
  torchrun --nproc_per_node=2 metax_dist_test.py --iters 40 --desync-rank 1 --desync-at 20
"""
import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--size", type=int, default=1 << 20)  # elements
    ap.add_argument("--desync-rank", type=int, default=-1,
                    help="this rank will skip a collective, stalling the group")
    ap.add_argument("--desync-at", type=int, default=-1,
                    help="iteration at which the desync rank bails out")
    ap.add_argument("--exercise-alltoall", action="store_true",
                    help="also run all_to_all_single to cover MoE communication")
    ap.add_argument("--exercise-alltoallv", action="store_true",
                    help="also pass split arrays to cover mcclAllToAllv")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # -> mccl on MetaX
    dev = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"[dist] world={world} device={torch.cuda.get_device_name(0)} "
              f"backend=nccl(->mccl)", flush=True)

    x = torch.randn(args.size, device=dev, dtype=torch.float32)
    g = torch.empty(args.size * world, device=dev, dtype=torch.float32)
    a2a_in = torch.arange(args.size, device=dev, dtype=torch.float32)
    a2a_out = torch.empty_like(a2a_in)

    for i in range(args.iters):
        # If we're the designated desync rank, bail out of the collective at the
        # chosen iteration -> every other rank's matching allreduce blocks -> the
        # XPUTimer hook flags that outstanding coll as HANG (fail-slow).
        if rank == args.desync_rank and i == args.desync_at:
            print(f"[dist][rank{rank}] DESYNC: skipping collectives from iter {i} "
                  f"(simulating a straggler / dead rank)", flush=True)
            time.sleep(30)  # sit out; peers will hang on their collective
            break

        dist.all_reduce(x)                      # mcclAllReduce
        dist.all_gather_into_tensor(g, x)       # mcclAllGather
        if args.exercise_alltoall:
            dist.all_to_all_single(a2a_out, a2a_in)  # mcclAllToAll
        if args.exercise_alltoallv:
            splits = [args.size // world] * world
            dist.all_to_all_single(
                a2a_out,
                a2a_in,
                output_split_sizes=splits,
                input_split_sizes=splits,
            )  # mcclAllToAllv
        if i % 20 == 0:
            torch.cuda.synchronize()
            if rank == 0:
                print(f"[dist] iter {i} ok", flush=True)

    torch.cuda.synchronize()
    time.sleep(2.0)  # let poller drain/dump
    if rank == 0:
        print("[dist] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
