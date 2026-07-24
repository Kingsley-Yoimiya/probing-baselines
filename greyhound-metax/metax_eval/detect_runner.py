#!/usr/bin/env python3
"""Greyhound detection-core runner for MetaX.

Reads the shared-memory ring buffer ("ncclRecord") that libmcclprobe.so fills,
then runs Greyhound's detection core on it:
  * find_period       -- ACF autocorrelation to recover the per-iteration
                         collective call period,
  * find_performance_drop -- Rbeast Bayesian change-point detection on the
                         per-iteration time series + degradation validation.

This is the "detect_failslow" path from control_plane/local_analyzer.py, run
standalone (no redis / controller needed).  It groups records per global_rank,
estimates each rank's iteration time, and reports change points -- and, across
ranks, which rank is the straggler (group comparison).

Usage:
    python3 detect_runner.py --config <config.json> [--snapshot out.npy]
    python3 detect_runner.py --from-snapshot dump.npy   # offline re-analysis
"""
import argparse
import json
import logging
import os
import sys

import numpy as np

# Make control_plane importable (it lives under detector/).
HERE = os.path.dirname(os.path.abspath(__file__))
DETECTOR = os.path.join(HERE, "..", "detector")
sys.path.insert(0, DETECTOR)

OPS = ['Send', 'Recv', 'Bcast', 'Broadcast', 'AllGather', 'ReduceScatter', 'AllReduce']
ATTRS = ['comm_addr', 'call_number', 'count', 'buff1', 'buff2',
         'datatype', 'pid', 'call_time', 'device', 'global_rank',
         'aux', 'duration', 'num_devices', 'event_id']
SIZEOF_INT64 = 8


def read_shm_records(config):
    """Return an (N, 14) int64 array of records currently in the shm buffer."""
    from multiprocessing import shared_memory, resource_tracker

    # Same resource_tracker monkey-patch as local_analyzer to avoid unlinking
    # a buffer we don't own.
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.register = fix_register
    resource_tracker.unregister = fix_unregister
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]

    num_fields = config['NUM_FIELDS']
    buffer_size = config['BUFFER_SIZE']
    meta = config['METADATA_FIELDS']
    shm_size = (num_fields * buffer_size + meta) * SIZEOF_INT64

    shm = shared_memory.SharedMemory("ncclRecord", create=False, size=shm_size)
    data = np.frombuffer(shm.buf, np.int64)
    try:
        # A writer increments metadata[6] after every complete record. Retry if
        # it changes while copying so a live snapshot cannot mix two ring states.
        for _ in range(10):
            event_before = int(data[6])
            n_fields = int(data[0])
            max_records = int(data[1])
            num_records = int(data[2])
            head = int(data[3])
            buf = data[meta:]

            recs = np.empty((num_records, n_fields), dtype=np.int64)
            for i in range(num_records):
                start = ((head + i) % max_records) * n_fields
                recs[i] = buf[start: start + n_fields]

            event_after = int(data[6])
            del buf
            if event_before == event_after:
                return recs
        raise RuntimeError("shared-memory ring changed during 10 snapshot attempts")
    finally:
        del data
        shm.close()


def analyze(records, plot=False):
    """Run the detection core on an (N, 14) record array. Returns per-rank info."""
    import pandas as pd
    # Import slow_detection.py directly (as a top-level module) to avoid
    # triggering control_plane/__init__.py, which pulls in the mitigation stack
    # (global_controller -> mitigation_plan -> cvxpy) that the detection core
    # does not need.
    import importlib.util
    sd_path = os.path.join(DETECTOR, "control_plane", "slow_detection.py")
    spec = importlib.util.spec_from_file_location("gh_slow_detection", sd_path)
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    find_period = sd.find_period
    find_performance_drop = sd.find_performance_drop

    df = pd.DataFrame(records, columns=ATTRS)
    n_total = len(df)
    ranks = sorted(df['global_rank'].unique().tolist())
    print(f"[detect] {n_total} records, ranks={ranks}")

    estimated_iter_time = {}
    change_points = {}
    for gr, g in df.groupby('global_rank'):
        g = g.sort_values(by='event_id')
        call_id = g['call_number'].to_numpy()
        call_time = g['call_time'].to_numpy()
        op_hist = {OPS[k]: int(v) for k, v in
                   zip(*np.unique(g['call_number'].to_numpy(), return_counts=True))
                   if 0 <= k < len(OPS)}
        start, period = find_period(call_id, nlags=200, significance_level=0.95)
        print(f"[detect] rank {gr}: n={len(g)} ops={op_hist} "
              f"period_start={start} period={period}")
        if period is None or period <= 0:
            print(f"[detect] rank {gr}: no period found (need more/steadier data)")
            continue
        cp_df, last_avg = find_performance_drop(call_id, call_time, period, start)
        # last_avg is in microseconds (call_time is us).
        estimated_iter_time[gr] = last_avg / 1e6
        change_points[gr] = cp_df
        cps = cp_df['values'].tolist() if cp_df is not None and len(cp_df) else []
        print(f"[detect] rank {gr}: est_iter_time={last_avg/1e6:.4f}s "
              f"validated_change_points(us)={cps}")

    # Group comparison: which rank is the straggler.
    if estimated_iter_time:
        med = float(np.median(list(estimated_iter_time.values())))
        print(f"[detect] median iter time across ranks = {med:.4f}s")
        for gr, t in sorted(estimated_iter_time.items()):
            flag = ""
            if med > 0 and t >= 1.2 * med:
                flag = "  <-- SLOW (>=1.2x median)"
            print(f"[detect]   rank {gr}: {t:.4f}s{flag}")
    return estimated_iter_time, change_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(DETECTOR, "control_plane", "config.json"))
    ap.add_argument("--snapshot", default=None, help="save raw records to this .npy")
    ap.add_argument("--from-snapshot", default=None, help="analyze a saved .npy instead of shm")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.from_snapshot:
        records = np.load(args.from_snapshot)
        print(f"[detect] loaded {records.shape} from {args.from_snapshot}")
    else:
        with open(args.config) as f:
            config = json.load(f)
        records = read_shm_records(config)
        print(f"[detect] read {records.shape} records from shm 'ncclRecord'")
        if args.snapshot:
            np.save(args.snapshot, records)
            print(f"[detect] saved snapshot to {args.snapshot}")

    analyze(records)


if __name__ == "__main__":
    main()
