"""Whether the memory system is what stops the throughput curve rising.

Two measurements, both on the same node and both without privileged performance counters:

  saturation  a stream of independent processes, each moving arrays far larger than any cache,
              run alone. Their summed rate is what the node's memory system delivers, and where
              that sum stops rising with the process count is where the memory system is full.
  co-running  the same stream processes run beside the training load. If training has the memory
              system full, the stream processes get a small fraction of what they get on an idle
              node; if training is limited by something else, they get most of it. Running the
              comparison at a small and a large copies-per-worker setting says whether it is the
              growing working set that fills the memory system.

The stream kernel is two array passes, `c = b * s` then `c = c + a`, over float64 arrays sized
well beyond the last-level cache. Bytes counted per pass are the array bytes the two calls read
and write, which excludes the read a write line costs the memory system, so every rate here is a
lower bound on the traffic actually carried.

Usage:
  python memory_pressure.py --saturation 1 8 28 56 112 224
  python memory_pressure.py --corun --train-procs 216 --probe-procs 8 --copies 1 128
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
BASE = RUN.parent.parent
sys.path.insert(0, str(BASE / "benchmarks"))
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
sys.path.insert(0, str(RUN / "code"))

MB_PER_ARRAY = 64             # three of these per process, far past any cache on this machine
BYTES_PER_ELEMENT = 8         # float64
PASSES_PER_ITERATION = 6      # array-sized reads and writes in `c = b*s` then `c = c + a`


def note(line):
    """Append one line to the progress file and to standard output, flushed immediately."""
    from bench_copies_per_worker import note as bench_note
    bench_note(line)


def stream_rate(seconds, delay=0.0):
    """Bytes per second one process moves through arrays too large to cache, after a delay.

    The delay lets a co-running training load reach its steady state before the stream starts
    measuring, so the reading is of the loaded machine and not of its opening seconds.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    n = MB_PER_ARRAY * 1024 * 1024 // BYTES_PER_ELEMENT
    a = np.ones(n, dtype=np.float64)
    b = np.full(n, 2.0, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    if delay:
        time.sleep(delay)
    passes, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        np.multiply(b, 3.0, out=c)
        np.add(c, a, out=c)
        passes += 1
    elapsed = time.perf_counter() - t0
    moved = passes * PASSES_PER_ITERATION * n * BYTES_PER_ELEMENT
    return {"bytes_per_sec": moved / elapsed, "passes": passes, "elapsed": elapsed,
            "checksum": float(c[0])}


def role_worker(args):
    """One process of a mixed run: either a stream process or a training worker."""
    role, payload = args
    if role == "stream":
        seconds, delay = payload
        return {"role": "stream", **stream_rate(seconds, delay)}
    from bench_copies_per_worker import timed_worker
    return {"role": "train", **timed_worker(payload)}


def saturation(proc_counts, seconds):
    """The node's memory rate against the number of processes asking for it, measured alone."""
    rows = []
    for p in proc_counts:
        with mp.get_context("spawn").Pool(p) as pool:
            out = pool.map(role_worker, [("stream", (seconds, 0.0))] * p)
        total = sum(r["bytes_per_sec"] for r in out)
        rows.append({"processes": p, "total_gb_per_sec": total / 1e9,
                     "gb_per_sec_per_process": total / 1e9 / p})
        note(f"[stream] processes={p} total={total / 1e9:,.1f}GB/s "
             f"per-process={total / 1e9 / p:,.2f}GB/s")
        (RUN / "data" / "memory_saturation.json").write_text(json.dumps(rows, indent=1))
    return rows


def corun(train_procs, probe_procs, copies, style, iters, seconds, delay):
    """Stream processes measured while training runs beside them, at one copies-per-worker setting.

    before: 216 training workers at 128 copies each and 8 stream processes, all started together
    after:  the 8 stream processes report the rate they reached while the machine was loaded,
            comparable with the same 8 processes run on the idle node
    """
    tasks = ([("stream", (seconds, delay))] * probe_procs
             + [("train", (copies, style, iters, 1))] * train_procs)
    with mp.get_context("spawn").Pool(train_procs + probe_procs) as pool:
        out = pool.map(role_worker, tasks)
    from bench_copies_per_worker import iteration_times, median
    stream = [r for r in out if r["role"] == "stream"]
    train = [r for r in out if r["role"] == "train"]
    total = sum(r["bytes_per_sec"] for r in stream)
    # the training workers report their iteration end stamps, not a rate; the median of their own
    # medians is the seconds per iteration the load was running at while the stream measured
    # before: a worker row carrying {"start": 1.0, "ends": [3.0, 5.2]}
    # after:  that worker's iteration times [2.0, 2.2], median 2.2
    row = {"train_procs": train_procs, "probe_procs": probe_procs, "copies": copies,
           "style": style, "stream_total_gb_per_sec": total / 1e9,
           "stream_gb_per_sec_per_process": total / 1e9 / probe_procs,
           "train_sec_per_iteration_median": median([median(iteration_times(r)) for r in train]),
           "train_peak_rss_mb_max": max(r["peak_rss_mb"] for r in train)}
    note(f"[corun] copies={copies} style={style} train_procs={train_procs} "
         f"probe_procs={probe_procs} stream={row['stream_gb_per_sec_per_process']:,.2f}GB/s "
         f"per process, train sec/iter={row['train_sec_per_iteration_median']:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saturation", type=int, nargs="+", default=[])
    ap.add_argument("--corun", action="store_true")
    ap.add_argument("--train-procs", type=int, default=216)
    ap.add_argument("--probe-procs", type=int, default=8)
    ap.add_argument("--copies", type=int, nargs="+", default=[1, 128])
    ap.add_argument("--style", default="full_batch")
    ap.add_argument("--iters", type=int, nargs="+", default=[])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--delay", type=float, default=25.0)
    args = ap.parse_args()
    note(f"memory pressure on {platform.node()}, {os.cpu_count()} logical processors")

    if args.saturation:
        saturation(args.saturation, args.seconds)
    if args.corun:
        rows = []
        # the idle baseline first: the same stream processes with nothing else on the machine
        with mp.get_context("spawn").Pool(args.probe_procs) as pool:
            out = pool.map(role_worker, [("stream", (args.seconds, 0.0))] * args.probe_procs)
        idle = sum(r["bytes_per_sec"] for r in out) / 1e9
        note(f"[corun] idle baseline {idle / args.probe_procs:,.2f}GB/s per process "
             f"over {args.probe_procs} processes")
        rows.append({"copies": 0, "style": "idle", "probe_procs": args.probe_procs,
                     "stream_total_gb_per_sec": idle,
                     "stream_gb_per_sec_per_process": idle / args.probe_procs})
        (RUN / "data" / "memory_corun.json").write_text(json.dumps(rows, indent=1))
        iters = args.iters or [max(3, int(60 / max(0.4, c * 0.05))) for c in args.copies]
        for copies, it in zip(args.copies, iters):
            rows.append(corun(args.train_procs, args.probe_procs, copies, args.style, it,
                              args.seconds, args.delay))
            (RUN / "data" / "memory_corun.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
