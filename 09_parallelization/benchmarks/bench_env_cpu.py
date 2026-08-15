"""Environment throughput on ordinary processor cores, for comparison with the graphics processor.

Two ways of using many cores, measured separately because they behave differently:

  threads    one process, the array library given N threads. Each environment step is one set of
             array operations spread across those threads.
  processes  N independent processes, each with one thread and its own share of the environments.
             This is how a work queue would use the machine, and it avoids the synchronisation
             the threaded form pays at every operation.

Reported for each point: aggregate environment steps per second, and steps per second per
environment (the rate at which one individual environment advances).

Usage:
  python bench_env_cpu.py --mode threads   --threads 1 2 4 8 16 32 64 80 160 --n-envs 10000
  python bench_env_cpu.py --mode processes --procs 1 2 4 8 16 32 64 80 160 --n-envs 10000
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def run_batch(n_envs, n_threads, seconds, seed=0):
    """Step n_envs environments for a fixed wall-clock budget; return steps completed."""
    import torch
    torch.set_num_threads(n_threads)
    sys.path.insert(0, str(BASE / "pointmaze" / "common"))
    sys.path.insert(0, str(BASE / "pointmaze" / "torch_env"))
    from pm_common import EnvConfig
    from torch_pointmaze import TorchPointMaze

    env = TorchPointMaze(EnvConfig(), 1, n_envs, device="cpu", base_seed=seed)
    env.reset()
    gen = torch.Generator().manual_seed(seed)
    act = torch.rand(1, n_envs, 2, generator=gen) * 2 - 1

    for _ in range(3):                       # warm up allocation and any lazy initialisation
        env.step(act)
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            env.step(act)
        steps += 10
    elapsed = time.perf_counter() - t0
    return steps, elapsed


def worker(args):
    """One process of the process-parallel mode."""
    n_envs, seconds, seed = args
    steps, elapsed = run_batch(n_envs, 1, seconds, seed)
    return steps * n_envs / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["threads", "processes"], required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 80, 160])
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 80, 160])
    ap.add_argument("--n-envs", type=int, default=10000,
                    help="environments per worker (processes mode) or in total (threads mode)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    host = platform.node()
    rows = []
    print(f"host {host}, {os.cpu_count()} logical processors, mode {args.mode}, "
          f"{args.n_envs} environments per worker")
    print(f"{'workers':>8} {'total envs':>11} {'M steps/s':>11} {'steps/s per env':>16}")

    if args.mode == "threads":
        for n in args.threads:
            steps, elapsed = run_batch(args.n_envs, n, args.seconds)
            total = steps * args.n_envs / elapsed
            rows.append({"workers": n, "n_envs_per_worker": args.n_envs,
                         "total_envs": args.n_envs, "env_steps_per_sec": total,
                         "env_steps_per_sec_per_env": total / args.n_envs})
            print(f"{n:>8} {args.n_envs:>11,} {total/1e6:>11.3f} {total/args.n_envs:>16,.0f}")
    else:
        for p in args.procs:
            with mp.get_context("spawn").Pool(p) as pool:
                rates = pool.map(worker, [(args.n_envs, args.seconds, i) for i in range(p)])
            total = sum(rates)
            envs = args.n_envs * p
            rows.append({"workers": p, "n_envs_per_worker": args.n_envs, "total_envs": envs,
                         "env_steps_per_sec": total, "env_steps_per_sec_per_env": total / envs})
            print(f"{p:>8} {envs:>11,} {total/1e6:>11.3f} {total/envs:>16,.0f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_envbench_cpu_{args.mode}{args.tag}.json"
    out.write_text(json.dumps({
        "host": host, "logical_processors": os.cpu_count(), "mode": args.mode,
        "n_envs_per_worker": args.n_envs, "seconds_per_point": args.seconds,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
