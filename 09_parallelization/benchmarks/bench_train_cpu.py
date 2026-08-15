"""End-to-end training throughput on ordinary processor cores, for comparison with the H100.

The graphics-processor configuration cannot be used here: recorded operation sequences, the
reduced-precision matrix mode and the fused optimiser are all device features. So the processor
runs the same algorithm in its plain form, and optionally with the compiler enabled, which is the
fairest "best effort on a processor" setting available.

Two ways of using the machine, matching the environment benchmark:

  threads    one process trains all the copies, with the array library given N threads.
  processes  N independent processes, one thread each, each training its own share of the copies.
             This is how the project's earlier processor sweeps actually ran, and it is the
             configuration a work queue would use.

Reported per point: seconds per training iteration, aggregate environment steps per second, and
environment steps per second per copy.

Usage:
  python bench_train_cpu.py --mode threads   --copies 1 2 4 8 --threads 8
  python bench_train_cpu.py --mode processes --procs 1 8 32 80 --copies-per-proc 1
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
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"


def cpu_config(n_copies, style, compile_it):
    """The trainer configuration for a processor: every device-only feature switched off."""
    from torch_ppo_rnd import PPOConfig
    # priming the observation statistics is not part of the timed region, so one pass is
    # enough here; the graphics-processor runs use the full ten
    return PPOConfig(n_copies=n_copies, update_style=style, rollout_mode="eager",
                     capture_update=False, one_graph=False, fused_adam=False, tf32=False,
                     compile_post=compile_it, obs_norm_init_iters=1)


def time_training(n_copies, style, n_threads, iters, warmup, compile_it, seed=0):
    """Median seconds per training iteration for one configuration on the processor."""
    import torch
    torch.set_num_threads(n_threads)
    from torch_ppo_rnd import PPORND
    trainer = PPORND(cpu_config(n_copies, style, compile_it), device="cpu")
    trainer.prime_obs_rms()
    update = (trainer.update_full_batch if style == "full_batch"
              else trainer.update_epoch_minibatch)
    for _ in range(warmup):
        update(trainer.rollout())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        update(trainer.rollout())
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def worker(args):
    """One process of the process-parallel mode; returns its own seconds per iteration."""
    n_copies, style, iters, warmup, compile_it, seed = args
    return time_training(n_copies, style, 1, iters, warmup, compile_it, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["threads", "processes"], required=True)
    ap.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 8, 32, 80, 160])
    ap.add_argument("--copies-per-proc", type=int, default=1)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from torch_ppo_rnd import PPOConfig
    steps_per_copy = PPOConfig().num_steps * PPOConfig().n_envs      # 512 by default
    rows = []
    print(f"host {platform.node()}, {os.cpu_count()} logical processors, mode {args.mode}, "
          f"style {args.style}, compiled {args.compile}")
    print(f"{'workers':>8} {'copies':>7} {'sec/iter':>10} {'M steps/s':>11} {'steps/s per copy':>17}")

    if args.mode == "threads":
        for c in args.copies:
            sec = time_training(c, args.style, args.threads, args.iters, args.warmup, args.compile)
            total = steps_per_copy * c / sec
            rows.append({"workers": args.threads, "n_copies": c, "total_copies": c,
                         "sec_per_iteration": sec, "env_steps_per_sec": total,
                         "env_steps_per_sec_per_copy": total / c})
            print(f"{args.threads:>8} {c:>7} {sec:>10.3f} {total/1e6:>11.4f} {total/c:>17,.0f}")
    else:
        for p in args.procs:
            with mp.get_context("spawn").Pool(p) as pool:
                secs = pool.map(worker, [(args.copies_per_proc, args.style, args.iters,
                                          args.warmup, args.compile, i) for i in range(p)])
            # every process runs its own copies concurrently, so the machine's rate is the sum
            total = sum(steps_per_copy * args.copies_per_proc / s for s in secs)
            copies = args.copies_per_proc * p
            rows.append({"workers": p, "n_copies": args.copies_per_proc, "total_copies": copies,
                         "sec_per_iteration": sorted(secs)[len(secs) // 2],
                         "env_steps_per_sec": total,
                         "env_steps_per_sec_per_copy": total / copies})
            print(f"{p:>8} {copies:>7} {sorted(secs)[len(secs)//2]:>10.3f} {total/1e6:>11.4f} "
                  f"{total/copies:>17,.0f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_trainbench_cpu_{args.mode}"
                     f"{args.tag}.json")
    out.write_text(json.dumps({
        "host": platform.node(), "logical_processors": os.cpu_count(), "mode": args.mode,
        "style": args.style, "compiled": args.compile, "threads": args.threads,
        "copies_per_proc": args.copies_per_proc, "steps_per_copy_per_iteration": steps_per_copy,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
