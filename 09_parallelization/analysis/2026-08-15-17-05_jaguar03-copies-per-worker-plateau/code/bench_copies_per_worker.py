"""Total throughput of independent worker processes as the copies per worker grow.

The processor sweep in the report stops at sixteen copies per worker and total throughput is
still rising there, so the machine's ceiling has not been shown. This script continues the same
measurement up the copies-per-worker ladder, and repairs four things the original benchmark got
wrong or did not do at all:

  a common window   the original computes the machine's rate as the SUM of each process's own
                    rate. Over a handful of iterations hundreds of processes are still starting
                    at staggered moments, so that sum describes a load that never existed on the
                    machine at one time. Here every worker waits at a barrier after its warm-up,
                    and the aggregate is the work all workers did inside the wall-clock window in
                    which every one of them was running. The sum-of-rates figure is kept beside it
                    so the difference is visible.
  sustained         the original times five iterations, about two seconds of work, which reads the
                    seconds during which a server processor runs above its sustained clock. Here
                    the warm-up runs about twenty seconds and the timed window about ninety.
  memory            peak resident memory of every worker from the kernel's own accounting, the sum
                    across workers, and the node's memory sampled while the point runs. A point
                    whose predicted memory would not fit is refused rather than run.
  method on record  every result file says how many iterations were timed, how long the warm-up
                    was, and how the aggregate was computed, so a later reader can tell one
                    methodology from another.

Every point is written to its own file the moment it finishes, so a job killed part way keeps
every point below the one in flight.

Usage:
  python bench_copies_per_worker.py --probe-memory 1 16 64 128 256
  python bench_copies_per_worker.py --copies 1 4 16 32 --procs-list 112 224 \
      --styles full_batch epoch_minibatch --tag _plateau_j3
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
BASE = RUN.parent.parent
sys.path.insert(0, str(BASE / "benchmarks"))
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = BASE / "benchmarks" / "results"
PROGRESS = RUN / "logs" / "progress.log"

STEPS_PER_COPY = 512          # num_steps x n_envs, the rows one copy produces per iteration
TARGET_TIMED_SECONDS = 90.0   # how much continuous work one point should time
TARGET_WARMUP_SECONDS = 20.0  # how long the load runs before the timed window opens
MAX_TIMED_SECONDS = 420.0     # ceiling, so a slow point cannot run away with the job
MEMORY_CEILING_FRACTION = 0.7  # of the node's memory, for the refusal guard
RESERVE_GB = 150.0            # left free whatever the fraction allows, for the prediction error
BARRIER_TIMEOUT = 3600.0      # a worker that never reaches the barrier fails the point loudly

BARRIER = None                # set in every pool worker by pool_init


def note(line):
    """Append one line to the progress file and to standard output, flushed immediately."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(f"{stamp} {line}\n")
        f.flush()
    print(f"{stamp} {line}", flush=True)


def iters_for(est_sec, target=TARGET_TIMED_SECONDS, cap=MAX_TIMED_SECONDS, lo=5, hi=150):
    """Iteration count that times about `target` seconds of work, inside `cap` and the bounds.

    before: est_sec = 0.4 seconds per iteration -> 150 iterations (the upper bound bites first)
    after:  est_sec = 7.0 seconds per iteration -> 13 iterations (90 / 7, rounded)
    """
    n = max(lo, min(hi, int(round(target / max(est_sec, 1e-6)))))
    while n > lo and n * est_sec > cap:
        n -= 1
    return n


def warmup_for(est_sec, target=TARGET_WARMUP_SECONDS, lo=1, hi=30):
    """Warm-up iterations, so the load has been running about `target` seconds before timing.

    The warm-up is what takes the processor off its opening clock boost and gets every worker's
    memory touched, and it happens before the barrier, so the timed window starts with every
    worker already at its steady state.
    before: est_sec = 0.4 -> 50 rounded down to the bound, 30 iterations
    after:  est_sec = 7.0 -> 3 iterations
    """
    return max(lo, min(hi, int(round(target / max(est_sec, 1e-6)))))


def per_worker_memory_model(probe_rows):
    """Peak memory of one worker as a straight line in the copies it holds: (base_mb, mb_per_copy).

    Fitted through the smallest and largest probe points rather than by least squares, because
    two points are enough for a line and the guard only needs an upper estimate.
    before: probe_rows = [{"copies": 1, "peak_rss_mb": 402.0}, {"copies": 512,
            "peak_rss_mb": 2438.0}]
    after:  base 398.0 MB, 3.983 MB per copy
    """
    rows = sorted(probe_rows, key=lambda r: r["copies"])
    lo, hi = rows[0], rows[-1]
    if hi["copies"] == lo["copies"]:
        return lo["peak_rss_mb"], 0.0
    slope = (hi["peak_rss_mb"] - lo["peak_rss_mb"]) / (hi["copies"] - lo["copies"])
    return lo["peak_rss_mb"] - slope * lo["copies"], slope


def predicted_point_gb(model, copies, procs):
    """The whole node's predicted memory for one point, in gigabytes, from the per-worker line."""
    base, slope = model
    return procs * (base + slope * copies) / 1024.0


def predicted_node_gb(node_rows, copies):
    """The node's memory at a copy count, from a line through the two largest measured points.

    Summing every worker's peak counts the shared library pages once per worker, so it runs well
    above the truth; the node's own reading counts them once. Once two points exist this is the
    better prediction, and it is the one the guard uses.
    before: node_rows = [{"copies": 64, "node_peak_used_gb": 190.0},
            {"copies": 128, "node_peak_used_gb": 300.0}], copies = 256
    after:  300 + (300-190)/(128-64) * (256-128) = 520.0 GB
    """
    rows = sorted(node_rows, key=lambda r: r["copies"])[-2:]
    (c0, g0), (c1, g1) = [(r["copies"], r["node_peak_used_gb"]) for r in rows]
    if c1 == c0:
        return g1
    return g1 + (g1 - g0) / (c1 - c0) * (copies - c1)


def meminfo_gb():
    """The node's total and available memory in gigabytes, read from the kernel."""
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        fields[key] = float(rest.strip().split()[0]) / (1024.0 * 1024.0)
    return fields["MemTotal"], fields["MemAvailable"]


class NodeMemorySampler:
    """Records the node's peak memory use while a point runs, sampled from /proc/meminfo."""

    def __init__(self, period=0.5):
        self.period = period
        self.total_gb, start_avail = meminfo_gb()
        self.min_available_gb = start_avail
        self.baseline_used_gb = self.total_gb - start_avail
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        """Keep the smallest available-memory reading seen during the point."""
        while not self._stop.wait(self.period):
            _, avail = meminfo_gb()
            self.min_available_gb = min(self.min_available_gb, avail)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2 * self.period)
        return False

    @property
    def peak_used_gb(self):
        """The node's memory in use at the point's peak."""
        return self.total_gb - self.min_available_gb


def median(values):
    """Middle value of a list, taking the upper of the two middles for an even count."""
    s = sorted(values)
    return s[len(s) // 2]


def pool_init(barrier):
    """Give every pool worker the barrier that opens the timed window for all of them at once."""
    global BARRIER
    BARRIER = barrier


def timed_worker(args):
    """One worker: build, warm up, wait for every other worker, then time its own iterations."""
    copies, style, iters, warmup = args
    import bench_train_cpu as bench
    import torch
    torch.set_num_threads(1)
    from torch_ppo_rnd import PPORND
    build_t0 = time.perf_counter()
    trainer = PPORND(bench.cpu_config(copies, style, False), device="cpu")
    trainer.prime_obs_rms()
    build_seconds = time.perf_counter() - build_t0
    update = (trainer.update_full_batch if style == "full_batch"
              else trainer.update_epoch_minibatch)
    for _ in range(warmup):
        update(trainer.rollout())
    # the barrier is what makes the aggregate honest: no worker starts timing until every worker
    # has finished warming up, so the timed windows of all of them overlap
    if BARRIER is not None:
        BARRIER.wait(timeout=BARRIER_TIMEOUT)
    start = time.time()
    ends = []
    for _ in range(iters):
        update(trainer.rollout())
        ends.append(time.time())
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {"start": start, "ends": ends, "peak_rss_mb": ru.ru_maxrss / 1024.0,
            "minor_faults": ru.ru_minflt, "major_faults": ru.ru_majflt,
            "user_seconds": ru.ru_utime, "system_seconds": ru.ru_stime,
            "build_seconds": build_seconds}


def iteration_times(row):
    """One worker's per-iteration durations, from its start and its iteration end stamps."""
    edges = [row["start"]] + row["ends"]
    return [b - a for a, b in zip(edges, edges[1:])]


def common_window(worker_rows, copies):
    """The aggregate over the wall-clock window in which every worker was running.

    The window opens when the last worker leaves the barrier and closes when the first worker
    finishes its last iteration. Only iterations that lie wholly inside it are counted, so the
    figure is work the machine really did while carrying the full load.
    before: two workers, the first timing iterations of 1.0s from t=0, the second from t=0.2s
    after:  window [0.2, 2.0]: worker one contributes its second iteration, worker two its first,
            and the aggregate is those two iterations' steps divided by 1.8 seconds
    """
    start = max(r["start"] for r in worker_rows)
    end = min(r["ends"][-1] for r in worker_rows)
    counted = 0
    for r in worker_rows:
        edges = [r["start"]] + r["ends"]
        counted += sum(1 for a, b in zip(edges, edges[1:]) if a >= start and b <= end)
    seconds = end - start
    return {"window_seconds": seconds, "iterations_in_window": counted,
            "env_steps_per_sec": counted * copies * STEPS_PER_COPY / seconds if seconds > 0 else 0}


def aggregate(worker_rows, copies, procs):
    """Rates for one point, both ways: the common window, and the sum of the workers' own rates."""
    total_copies = copies * procs
    per_worker = [iteration_times(r) for r in worker_rows]
    win = common_window(worker_rows, copies)
    point = {"workers": procs, "n_copies": copies, "total_copies": total_copies,
             "sec_per_iteration": median([median(t) for t in per_worker]),
             "env_steps_per_sec": win["env_steps_per_sec"],
             "env_steps_per_sec_per_copy": win["env_steps_per_sec"] / total_copies,
             "window_seconds": win["window_seconds"],
             "iterations_in_window": win["iterations_in_window"]}
    # the two comparison figures: the old aggregate on the same data, and the old aggregate on
    # only the first three iterations, which is what a five-iteration measurement mostly reads
    for field, times in [("sum_of_worker_rates", per_worker),
                         ("sum_of_worker_rates_opening_three", [t[:3] for t in per_worker])]:
        total = sum(STEPS_PER_COPY * copies / median(t) for t in times)
        point[f"env_steps_per_sec_{field}"] = total
        point[f"env_steps_per_sec_per_copy_{field}"] = total / total_copies
    secs = sorted(median(t) for t in per_worker)
    point["sec_per_iteration_fastest_worker"] = secs[0]
    point["sec_per_iteration_slowest_worker"] = secs[-1]
    return point


def run_point(procs, copies, style, iters, warmup):
    """One measurement: `procs` worker processes, each training `copies` copies of the trainer."""
    note(f"[start] procs={procs} copies={copies} style={style} iters={iters} warmup={warmup}")
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(procs)
    with NodeMemorySampler() as sampler:
        with ctx.Pool(procs, initializer=pool_init, initargs=(barrier,)) as pool:
            rows = pool.map(timed_worker, [(copies, style, iters, warmup)] * procs)
    wall = time.perf_counter() - t0
    point = aggregate(rows, copies, procs)
    # memory is reported three ways: the worst single worker, the sum across workers (which counts
    # shared library pages once per worker, so it is an overestimate), and what the node itself
    # reported at its peak (which counts them once)
    peaks = sorted(r["peak_rss_mb"] for r in rows)
    point.update({
        "iterations_timed": iters, "warmup_iterations": warmup, "wall_seconds": wall,
        "aggregate": "common wall-clock window across all workers, barrier-synchronised",
        "peak_rss_mb_max_worker": peaks[-1], "peak_rss_mb_median_worker": peaks[len(peaks) // 2],
        "sum_peak_rss_gb": sum(peaks) / 1024.0,
        "node_peak_used_gb": sampler.peak_used_gb,
        "node_baseline_used_gb": sampler.baseline_used_gb,
        "node_total_gb": sampler.total_gb,
        "minor_faults_median_worker": sorted(r["minor_faults"] for r in rows)[len(rows) // 2],
        "major_faults_max_worker": max(r["major_faults"] for r in rows),
        "build_seconds_median_worker": sorted(r["build_seconds"] for r in rows)[len(rows) // 2],
        "cpu_seconds_median_worker": sorted(r["user_seconds"] + r["system_seconds"]
                                            for r in rows)[len(rows) // 2]})
    note(f"[done ] procs={procs} copies={copies} style={style} "
         f"total_copies={point['total_copies']} sec/iter={point['sec_per_iteration']:.3f} "
         f"Msteps/s={point['env_steps_per_sec'] / 1e6:.4f} "
         f"per-copy={point['env_steps_per_sec_per_copy']:,.0f} "
         f"sum_of_rates_Msteps/s={point['env_steps_per_sec_sum_of_worker_rates'] / 1e6:.4f} "
         f"window={point['window_seconds']:.0f}s "
         f"peak_rss_worker={point['peak_rss_mb_max_worker']:,.0f}MB "
         f"node_peak={point['node_peak_used_gb']:,.1f}GB wall={wall:.0f}s")
    return point


def write_point(point, style, tag):
    """Write one finished point to its own file, in the schema the report's loader reads."""
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_trainbench_cpu_processes_{tag}.json")
    out.write_text(json.dumps({
        "host": platform.node(), "logical_processors": os.cpu_count(), "mode": "processes",
        "style": style, "compiled": False, "threads": 1,
        "copies_per_proc": point["n_copies"], "steps_per_copy_per_iteration": STEPS_PER_COPY,
        # the methodology, on the record, so a later reader can tell this file from a
        # five-iteration one without knowing which script wrote it
        "iters": point["iterations_timed"], "warmup": point["warmup_iterations"],
        "aggregate": point["aggregate"],
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": [point]}, indent=1))
    note(f"[wrote] {out}")
    return out


def probe_memory(copies_list, style):
    """Peak memory of ONE worker at each copy count, so the guard has a model before the ladder."""
    rows = []
    for c in copies_list:
        with mp.get_context("spawn").Pool(1) as pool:
            r, = pool.map(timed_worker, [(c, style, 1, 0)])
        rows.append({"copies": c, "peak_rss_mb": r["peak_rss_mb"],
                     "build_seconds": r["build_seconds"], "minor_faults": r["minor_faults"]})
        note(f"[probe] style={style} copies={c} peak_rss={r['peak_rss_mb']:,.0f}MB "
             f"build={r['build_seconds']:.1f}s")
        (RUN / "data" / f"memory_probe_{style}.json").write_text(json.dumps(rows, indent=1))
    return rows


def series_state(style, procs, tag):
    """The saved state of one (worker count, update convention) series: its points so far."""
    path = RUN / "data" / f"ladder_{style}_p{procs}{tag}.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    return path, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-memory", type=int, nargs="+", default=[])
    ap.add_argument("--copies", type=int, nargs="+", default=[])
    ap.add_argument("--procs-list", type=int, nargs="+", default=[224])
    ap.add_argument("--styles", nargs="+", default=["full_batch"])
    ap.add_argument("--style", default="full_batch")            # for --probe-memory
    ap.add_argument("--tag", default="")
    ap.add_argument("--est-first", type=float, default=0.5)
    ap.add_argument("--write-results", type=int, default=1)
    args = ap.parse_args()

    total_gb, avail_gb = meminfo_gb()
    note(f"host {platform.node()}, {os.cpu_count()} logical processors, "
         f"{total_gb:,.0f}GB memory, {avail_gb:,.0f}GB available")

    if args.probe_memory:
        probe_memory(args.probe_memory, args.style)
        return

    # one independent series per (worker count, update convention): its own time estimate, its own
    # memory line, its own file. The copy count is the OUTER loop so the cheap rungs of every
    # series are measured before the expensive rungs of any of them.
    series = {}
    for procs in args.procs_list:
        for style in args.styles:
            probe_path = RUN / "data" / f"memory_probe_{style}.json"
            probe_rows = json.loads(probe_path.read_text()) if probe_path.exists() else []
            path, rows = series_state(style, procs, args.tag)
            series[(procs, style)] = {
                "path": path, "rows": rows, "est": args.est_first, "stopped": False,
                "probe": probe_rows,
                "node": [{"copies": p["n_copies"], "node_peak_used_gb": p["node_peak_used_gb"]}
                         for p in rows]}

    for copies in args.copies:
        for procs in args.procs_list:
            for style in args.styles:
                s = series[(procs, style)]
                if s["stopped"]:
                    continue
                # the guard: predict this point's memory before running it, and refuse it rather
                # than let the node run out, which would lose every point still to come
                want, how = None, ""
                if len(s["node"]) >= 2:
                    want, how = predicted_node_gb(s["node"], copies), "from the node's readings"
                elif len(s["probe"]) >= 2:
                    want = predicted_point_gb(per_worker_memory_model(s["probe"]), copies, procs)
                    how = "from the per-worker line"
                if want is not None:
                    total_gb, avail_gb = meminfo_gb()
                    reserve = min(RESERVE_GB, 0.15 * total_gb)
                    ceiling = min(MEMORY_CEILING_FRACTION * total_gb, avail_gb - reserve)
                    if want > ceiling:
                        note(f"[refuse] procs={procs} copies={copies} style={style}: predicted "
                             f"{want:,.0f}GB {how} against a ceiling of {ceiling:,.0f}GB; this "
                             f"series stops here")
                        s["stopped"] = True
                        continue
                    note(f"[guard] procs={procs} copies={copies} style={style} predicted "
                         f"{want:,.0f}GB {how}, ceiling {ceiling:,.0f}GB")
                point = run_point(procs, copies, style, iters_for(s["est"]),
                                  warmup_for(s["est"]))
                # every point is on disk before the next one starts, in both places it belongs
                s["rows"].append(point)
                s["path"].write_text(json.dumps(s["rows"], indent=1))
                if args.write_results:
                    write_point(point, style, f"{args.tag}_c{copies}_p{procs}")
                s["est"] = point["sec_per_iteration"]
                s["probe"].append({"copies": copies,
                                   "peak_rss_mb": point["peak_rss_mb_max_worker"]})
                s["node"].append({"copies": copies,
                                  "node_peak_used_gb": point["node_peak_used_gb"]})


if __name__ == "__main__":
    main()
