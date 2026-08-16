"""Total throughput of independent worker processes as the copies per worker grow.

The processor sweep in the report stops at sixteen copies per worker and total throughput is
still rising there, so the machine's ceiling has not been shown. This script continues the same
measurement up the copies-per-worker ladder at a fixed worker count, and adds three things the
original benchmark does not have:

  memory        peak resident memory of every worker, read from the kernel's own accounting, plus
                the node's memory use sampled while the point runs.
  a guard       each point's memory is predicted from the points below it, and a point that would
                not fit is refused rather than run, because a node that runs out of memory loses
                every later point of the job.
  sustained     the iteration count is chosen so each point times about a minute of continuous
                work. A five-iteration measurement reads the opening seconds of a load, when a
                server processor is still above its sustained clock.

Every point is written to its own file the moment it finishes, so a job killed part way keeps
every point below the one in flight.

Usage:
  python bench_copies_per_worker.py --probe-memory 1 16 64 128 256
  python bench_copies_per_worker.py --ladder 1 4 16 32 64 128 --procs 224 --style full_batch
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
TARGET_TIMED_SECONDS = 60.0   # how much continuous work one point should time
MAX_TIMED_SECONDS = 300.0     # ceiling, so a slow point cannot run away with the job
MEMORY_CEILING_FRACTION = 0.7  # of the node's memory, for the refusal guard
RESERVE_GB = 150.0            # left free whatever the fraction allows, for the prediction error


def note(line):
    """Append one line to the progress file and to standard output, flushed immediately."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(f"{stamp} {line}\n")
        f.flush()
    print(f"{stamp} {line}", flush=True)


def iters_for(est_sec, target=TARGET_TIMED_SECONDS, cap=MAX_TIMED_SECONDS, lo=3, hi=150):
    """Iteration count that times about `target` seconds of work, inside `cap` and the bounds.

    before: est_sec = 0.4 seconds per iteration -> 150 iterations (the upper bound bites first)
    after:  est_sec = 7.0 seconds per iteration -> 9 iterations (60 / 7, rounded up)
    """
    n = max(lo, min(hi, int(round(target / max(est_sec, 1e-6)))))
    while n > lo and n * est_sec > cap:
        n -= 1
    return n


def per_worker_memory_model(probe_rows):
    """Peak memory of one worker as a straight line in the copies it holds: (base_mb, mb_per_copy).

    Fitted through the smallest and largest probe points rather than by least squares, because
    two points are enough for a line and the guard only needs an upper estimate.
    before: probe_rows = [{"copies": 1, "peak_rss_mb": 420.0}, {"copies": 256,
            "peak_rss_mb": 2100.0}]
    after:  base 413.4 MB, 6.588 MB per copy
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
    above the truth; the node's own reading counts them once. Once two ladder points exist this
    is the better prediction, and it is the one the guard uses.
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
        """Node memory in use at the point's peak, above what was already in use before it."""
        return self.total_gb - self.min_available_gb


def median(values):
    """Middle value of a list, taking the upper of the two middles for an even count."""
    s = sorted(values)
    return s[len(s) // 2]


def burst_and_settled(times):
    """Three readings of the same iteration times: the opening ones, all of them, the closing ones.

    A server processor runs above its sustained clock for the first seconds of a load, so a
    three-iteration measurement reads the opening burst and a minute-long one reads the settled
    rate. Reporting both from one run makes the two directly comparable at no extra cost.
    before: times = [0.40, 0.41, 0.42, 0.50, 0.51, 0.52]
    after:  opening 0.41 (first three), all 0.50, closing 0.52 (last third)
    """
    tail = times[len(times) - max(1, len(times) // 3):]
    return {"sec_per_iteration": median(times),
            "sec_per_iteration_opening_three": median(times[:3]),
            "sec_per_iteration_last_third": median(tail),
            "sec_fastest": min(times), "sec_slowest": max(times)}


def timed_worker(args):
    """One worker process: time its own iterations, then report its own peak memory and faults."""
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
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        update(trainer.rollout())
        times.append(time.perf_counter() - t0)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    row = burst_and_settled(times)
    row.update({"peak_rss_mb": ru.ru_maxrss / 1024.0, "minor_faults": ru.ru_minflt,
                "major_faults": ru.ru_majflt, "user_seconds": ru.ru_utime,
                "system_seconds": ru.ru_stime, "build_seconds": build_seconds})
    return row


def probe_worker(args):
    """One process built at a given copy count, run for one iteration, reporting peak memory."""
    copies, style = args
    return timed_worker((copies, style, 1, 0))


def aggregate(worker_rows, copies, procs):
    """Rates for one point: every worker runs concurrently, so the machine's rate is the sum.

    before: worker_rows = two workers at 0.50 and 0.60 seconds per iteration, copies = 16
    after:  total = 512*16/0.50 + 512*16/0.60 = 30,037 steps per second over 32 copies
    """
    total_copies = copies * procs
    point = {"workers": procs, "n_copies": copies, "total_copies": total_copies}
    # the same three arithmetic steps for the settled reading and for the opening-burst reading,
    # so the two protocols can be compared without re-running anything
    for field, suffix in [("sec_per_iteration", ""),
                          ("sec_per_iteration_opening_three", "_opening_three")]:
        total = sum(STEPS_PER_COPY * copies / r[field] for r in worker_rows)
        point[f"sec_per_iteration{suffix}"] = median([r[field] for r in worker_rows])
        point[f"env_steps_per_sec{suffix}"] = total
        point[f"env_steps_per_sec_per_copy{suffix}"] = total / total_copies
    secs = sorted(r["sec_per_iteration"] for r in worker_rows)
    point["sec_per_iteration_fastest_worker"] = secs[0]
    point["sec_per_iteration_slowest_worker"] = secs[-1]
    return point


def run_point(procs, copies, style, iters, warmup):
    """One measurement: `procs` worker processes, each training `copies` copies of the trainer."""
    note(f"[start] procs={procs} copies={copies} style={style} iters={iters} warmup={warmup}")
    t0 = time.perf_counter()
    with NodeMemorySampler() as sampler:
        with mp.get_context("spawn").Pool(procs) as pool:
            rows = pool.map(timed_worker, [(copies, style, iters, warmup)] * procs)
    wall = time.perf_counter() - t0
    point = aggregate(rows, copies, procs)
    # memory is reported three ways: the worst single worker, the sum over workers (which counts
    # shared library pages once per worker, so it is an overestimate), and what the node itself
    # reported at its peak (which counts them once)
    peaks = sorted(r["peak_rss_mb"] for r in rows)
    point.update({
        "iterations_timed": iters, "warmup_iterations": warmup, "wall_seconds": wall,
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
         f"opening_three_Msteps/s={point['env_steps_per_sec_opening_three'] / 1e6:.4f} "
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
            r, = pool.map(probe_worker, [(c, style)])
        rows.append({"copies": c, "peak_rss_mb": r["peak_rss_mb"],
                     "sec_per_iteration_alone": r["sec_per_iteration"],
                     "build_seconds": r["build_seconds"], "minor_faults": r["minor_faults"]})
        note(f"[probe] style={style} copies={c} peak_rss={r['peak_rss_mb']:,.0f}MB "
             f"sec/iter_alone={r['sec_per_iteration']:.3f} build={r['build_seconds']:.1f}s")
        (RUN / "data" / f"memory_probe_{style}.json").write_text(json.dumps(rows, indent=1))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-memory", type=int, nargs="+", default=[])
    ap.add_argument("--ladder", type=int, nargs="+", default=[])
    ap.add_argument("--procs", type=int, default=224)
    ap.add_argument("--style", default="full_batch")
    ap.add_argument("--tag", default="")
    ap.add_argument("--est-first", type=float, default=0.4)
    # rungs at or below this copy count repeat settings the report already has a file for; they
    # are kept in this run folder only, so the report's existing rows are not overwritten by a
    # measurement taken under a different timing window
    ap.add_argument("--results-min-copies", type=int, default=32)
    args = ap.parse_args()

    total_gb, avail_gb = meminfo_gb()
    note(f"host {platform.node()}, {os.cpu_count()} logical processors, "
         f"{total_gb:,.0f}GB memory, {avail_gb:,.0f}GB available")

    if args.probe_memory:
        probe_memory(args.probe_memory, args.style)
        return

    # the guard needs a per-worker memory line; the probe file written earlier in the job supplies
    # it, and the ladder's own points refine it as they finish
    probe_path = RUN / "data" / f"memory_probe_{args.style}.json"
    probe_rows = json.loads(probe_path.read_text()) if probe_path.exists() else []
    model = per_worker_memory_model(probe_rows) if len(probe_rows) >= 2 else None
    if model:
        note(f"[model] per-worker memory = {model[0]:,.0f}MB + {model[1]:.3f}MB per copy")

    est = args.est_first
    ladder_path = RUN / "data" / f"ladder_{args.style}_p{args.procs}{args.tag}.json"
    ladder = json.loads(ladder_path.read_text()) if ladder_path.exists() else []
    node_rows = [{"copies": p["n_copies"], "node_peak_used_gb": p["node_peak_used_gb"]}
                 for p in ladder]
    for copies in args.ladder:
        # the guard: predict this point's memory before running it, and refuse it rather than
        # let the node run out, which would lose every point still to come
        want = None
        if len(node_rows) >= 2:
            want, how = predicted_node_gb(node_rows, copies), "from the node's own readings"
        elif model:
            want, how = predicted_point_gb(model, copies, args.procs), "from the per-worker line"
        if want is not None:
            total_gb, avail_gb = meminfo_gb()
            # the reserve is capped at a share of the machine so the same guard works on a small
            # machine, where a flat 150 GB would be larger than the machine itself
            reserve = min(RESERVE_GB, 0.15 * total_gb)
            ceiling = min(MEMORY_CEILING_FRACTION * total_gb, avail_gb - reserve)
            if want > ceiling:
                note(f"[refuse] procs={args.procs} copies={copies} style={args.style}: predicted "
                     f"{want:,.0f}GB {how} against a ceiling of {ceiling:,.0f}GB "
                     f"({avail_gb:,.0f}GB available); the ladder stops here")
                break
            note(f"[guard] procs={args.procs} copies={copies} predicted {want:,.0f}GB {how}, "
                 f"ceiling {ceiling:,.0f}GB, {avail_gb:,.0f}GB available")
        point = run_point(args.procs, copies, args.style, iters_for(est), warmup=1)
        # every point is on disk before the next one starts, in both places it belongs
        ladder.append(point)
        ladder_path.write_text(json.dumps(ladder, indent=1))
        if copies >= args.results_min_copies:
            write_point(point, args.style, f"{args.tag}_c{copies}_p{args.procs}")
        else:
            note(f"[kept in run folder] copies={copies} repeats a setting the report already "
                 f"has a file for, so it is not written to the shared results directory")
        # the next rung holds twice the copies; the measured time is the better starting estimate
        est = point["sec_per_iteration"]
        # refine both memory lines with the point just measured, which is the real thing rather
        # than a single-process extrapolation
        probe_rows.append({"copies": copies, "peak_rss_mb": point["peak_rss_mb_max_worker"]})
        model = per_worker_memory_model(probe_rows)
        node_rows.append({"copies": copies, "node_peak_used_gb": point["node_peak_used_gb"]})


if __name__ == "__main__":
    main()
