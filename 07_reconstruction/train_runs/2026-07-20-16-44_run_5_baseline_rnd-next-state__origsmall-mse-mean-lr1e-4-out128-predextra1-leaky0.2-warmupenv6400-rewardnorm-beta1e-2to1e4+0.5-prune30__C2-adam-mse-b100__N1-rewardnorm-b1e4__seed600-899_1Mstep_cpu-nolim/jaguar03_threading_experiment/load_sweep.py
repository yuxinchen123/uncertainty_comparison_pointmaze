#!/usr/bin/env python
"""Load-vs-frequency sweep on jaguar03: at how many busy cores does the node start throttling?

Driver: for each core count N in a list, launch N single-thread CPU-burn workers pinned to distinct
physical cores (cores 0..N-1), let them run STEADY_SEC, then read (a) the median achieved clock over
the active cores from /proc/cpuinfo and (b) each worker's matmul rate (iterations/sec, proportional to
the delivered frequency). Kill, move to the next N. Print a table of N vs MHz vs per-core rate so the
throttle knee is visible.

Worker: pinned to one core, repeats a fixed 256x256 matmul batch and appends (wall, iters) to its file.

Run on jaguar03 (whole node): srun --exclusive --nodelist=jaguar03 ... python load_sweep.py
"""
import glob
import os
import re
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(HERE, "load_sweep_work")
STEADY_SEC = int(os.environ.get("STEADY_SEC", "40"))  # run each level this long before measuring; raise
#                                                       it (e.g. 1600) to hold ONE level under SUSTAINED
#                                                       load and see whether the throttle appears over time
BATCH = 40                            # matmuls per reported iteration
PACKING = int(os.environ.get("PACKING", "1"))         # 1 = one worker per physical core; 2 = two per core
# level list is the number of WORKER THREADS at each step (env override e.g. LEVELS=32,64,96,128,...)
_lv = os.environ.get("LEVELS", "")
LEVELS = [int(x) for x in _lv.split(",")] if _lv else [1, 4, 8, 16, 24, 32, 48, 64, 80, 96, 108]


def cpu_for(idx):
    """Hardware-thread id for worker `idx`. PACKING=1: worker i -> core i (cpu i), one per core.
    PACKING=2: workers 2k,2k+1 -> the two siblings of core k (cpu k and cpu k+112), two per core."""
    if PACKING == 1:
        return idx
    core = idx // 2
    return core if idx % 2 == 0 else core + 112


def worker(core, outpath):
    """Pin to `core`, repeat a MEMORY+COMPUTE batch forever, append (wall, iters) every ~2s. The batch
    mixes a random gather over a 256 MB DRAM buffer (mimics SAC replay-buffer sampling, which loads the
    memory controllers / package power) with a small matmul (compute) -- this is the power profile that
    triggered the training throttle, unlike a cache-resident matmul-only burn."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    import numpy as np
    try:
        os.sched_setaffinity(0, {core})
    except OSError:
        pass
    rng = np.random.default_rng(core)
    buf = rng.standard_normal(64_000_000, dtype=np.float32)     # 256 MB, exceeds L3 -> real DRAM traffic
    a = rng.standard_normal((256, 256), dtype=np.float32)
    b = rng.standard_normal((256, 256), dtype=np.float32)
    iters = 0
    t0 = time.time()
    last = t0
    with open(outpath, "w") as fh:
        while True:
            for _ in range(BATCH):
                idx = rng.integers(0, buf.size, size=50_000)   # random gather -> memory-bound access
                _ = float(buf[idx].sum())
                a = (a @ b) * np.float32(1e-4) + b             # compute
            iters += BATCH
            now = time.time()
            if now - last >= 2.0:
                fh.write(f"{now:.3f} {iters}\n")
                fh.flush()
                last = now


def median_active_mhz(active_cpus):
    """Median live MHz over the given active logical CPUs (note: acpi-cpufreq misreports the true
    clock on this node, so this is only a rough signal; the per-worker matmul rate is authoritative)."""
    mhz = {}
    cur = None
    for line in open("/proc/cpuinfo"):
        if line.startswith("processor"):
            cur = int(line.split(":")[1])
        elif line.startswith("cpu MHz") and cur is not None:
            mhz[cur] = float(line.split(":")[1])
    active = [mhz[c] for c in active_cpus if c in mhz]
    return round(statistics.median(active), 1) if active else -1.0


def worker_rate(outpath):
    """Iterations/sec of one worker over its last ~15s of samples (steady-state)."""
    try:
        rows = [line.split() for line in open(outpath) if line.strip()]
    except OSError:
        return None
    pts = [(float(t), int(i)) for t, i in rows]
    if len(pts) < 2:
        return None
    tail = [p for p in pts if p[0] >= pts[-1][0] - 15]
    if len(tail) < 2:
        tail = pts[-2:]
    dt = tail[-1][0] - tail[0][0]
    di = tail[-1][1] - tail[0][1]
    return di / dt if dt > 0 else None


def run_level(n):
    """Launch n pinned workers, wait STEADY_SEC, measure MHz + per-worker rate, kill. Return a dict."""
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.txt")):
        os.remove(f)
    procs = []
    for c in range(n):
        outp = os.path.join(OUT, f"w{c}.txt")
        p = subprocess.Popen([PY, __file__, "--worker", str(cpu_for(c)), outp], start_new_session=True)
        procs.append(p)
    # sample MHz a few times over the steady window, then read rates
    active_cpus = [cpu_for(c) for c in range(n)]
    time.sleep(STEADY_SEC - 8)
    mhz_samples = [median_active_mhz(active_cpus) for _ in range(4) if not time.sleep(2)]
    rates = [r for c in range(n) if (r := worker_rate(os.path.join(OUT, f"w{c}.txt"))) is not None]
    for p in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(3)
    mhz = statistics.median(mhz_samples) if mhz_samples else -1
    per_core = statistics.median(rates) if rates else 0.0
    return dict(n=n, mhz=mhz, per_core=per_core, total=per_core * n, nrates=len(rates))


def main():
    """Sweep the core-count levels and print the load-vs-frequency table."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), sys.argv[3])
        return
    print(f"node={os.uname().nodename}  steady={STEADY_SEC}s  batch={BATCH} matmuls/iter  PACKING={PACKING}/core", flush=True)
    print(f"{'thr':>5} {'med_MHz':>8} {'per-thr_iters/s':>16} {'total_iters/s':>13} {'slowdown_vs_first':>17}", flush=True)
    base = None
    for n in LEVELS:
        r = run_level(n)
        if base is None:
            base = r["per_core"]
        slow = base / r["per_core"] if r["per_core"] else float("nan")
        print(f"{r['n']:5} {r['mhz']:8.0f} {r['per_core']:16.1f} {r['total']:13.1f} {slow:16.2f}x", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
