"""
metax_probe self-test: exercise the XPUTimer MetaX hook on real C550 GPUs.

Runs a small torch workload (matmul kernels + collectives if torch.distributed
is initialized, else single-GPU kernels + a manual slow op) so the LD_PRELOAD
hook (libxpu_timer_metax.so) can time mcLaunchKernel / mccl* and emit metrics.

Usage (single GPU, kernel timing only):
    LD_PRELOAD=./libxpu_timer_metax.so \
    XPU_TIMER_DUMP_DIR=/tmp/xpu_timer_metax \
    python3 metax_selftest.py --iters 200

Usage (fail-slow injection: sleep the GPU to trip the HANG detector):
    XPU_TIMER_HANG_TIMEOUT_MS=500 LD_PRELOAD=./libxpu_timer_metax.so \
    python3 metax_selftest.py --iters 50 --inject-hang
"""
import argparse
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--inject-hang", action="store_true",
                    help="issue one very long-running kernel chain to trip HANG")
    ap.add_argument("--hang-chain", type=int, default=2000,
                    help="number of elementwise chain steps for hang injection")
    ap.add_argument("--hang-sleep", type=float, default=3.0,
                    help="seconds to let the poller observe the backlog")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "torch.cuda (MACA) not available"
    dev = torch.device("cuda:0")
    print(f"[selftest] device={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__}")

    n = args.size
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)

    # warmup
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.time()
    for i in range(args.iters):
        c = a @ b            # matmul -> many mcLaunchKernel
        c = torch.relu(c)    # elementwise kernel
        c = c + a            # elementwise kernel
        if i % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[selftest] {args.iters} iters normal workload in {dt*1000:.1f}ms "
          f"({dt/args.iters*1e6:.1f}us/iter)")

    if args.inject_hang:
        # NOTE: matmul (a@b) goes through mcblas GEMM which this hook stubs, so it
        # would not be observed. Elementwise ops DO go through mcLaunchKernel.
        # We enqueue a long chain of dependent elementwise kernels WITHOUT syncing,
        # so the GPU queue backs up: the tail kernels' stop-events stay unready far
        # longer than the hang timeout -> the poller flags them HANG. This is the
        # single-process analogue of XPUTimer's stuck-collective fail-slow signal.
        print("[selftest] injecting long unsynchronized elementwise chain to "
              "trip HANG detector...")
        big = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
        x = big
        for _ in range(args.hang_chain):
            x = torch.relu(x)        # mcLaunchKernel
            x = x + big              # mcLaunchKernel
            x = torch.sin(x)         # mcLaunchKernel
        # do NOT synchronize yet; let the background poller observe the backlog
        time.sleep(args.hang_sleep)
        torch.cuda.synchronize()
        print("[selftest] hang-injection chain finished")

    # give the background poller a moment to drain + dump
    time.sleep(1.5)
    print("[selftest] done")


if __name__ == "__main__":
    main()
