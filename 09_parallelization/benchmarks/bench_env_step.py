"""Benchmark env.step throughput on the GPU. Writes one JSON per run to results/.

Usage (on serval05, under the H100 lock):
  python bench_env_step.py --impl torch --mode eager --n-envs 1000 100000 1000000
Modes: eager | compile (torch.compile fullgraph) | compile-oh (mode=reduce-overhead).
Timing: CUDA events, warmup excluded, median of --repeats measurement blocks; the step
count per block adapts so each block runs a meaningful workload at every batch size.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch._dynamo.config as dynconf

# the grid benches many static shapes in one process; each is its own compile entry
dynconf.cache_size_limit = 64
if hasattr(dynconf, "recompile_limit"):
    dynconf.recompile_limit = 64

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "pointmaze" / "common"))
sys.path.insert(0, str(BASE / "pointmaze" / "torch_env"))

RESULTS = Path(__file__).resolve().parent / "results"


def bench_torch(n_copies, n_envs, mode, repeats):
    """Measure full step() (dynamics + reward + auto-reset) for one batch size.

    Returns dict with steps/sec and ns per env-step (median over measurement blocks).
    """
    from pm_common import EnvConfig
    from torch_pointmaze import TorchPointMaze

    env = TorchPointMaze(EnvConfig(), n_copies, n_envs, device="cuda", dtype=torch.float32)
    env.reset()
    step = env.step
    if mode == "compile-bigfuse":
        # let inductor keep the whole elementwise step in as few kernels as possible
        import torch._inductor.config as icfg
        icfg.realize_opcount_threshold = 100000
        icfg.realize_reads_threshold = 100000
        icfg.realize_acc_reads_threshold = 100000
        step = torch.compile(env.step, fullgraph=True, dynamic=False)
    elif mode == "compile":
        step = torch.compile(env.step, fullgraph=True, dynamic=False)
    elif mode == "compile-oh":
        step = torch.compile(env.step, fullgraph=True, dynamic=False, mode="reduce-overhead")

    total = n_copies * n_envs
    # rotating action buffers so the workload is not a single constant push
    acts = [torch.rand(n_copies, n_envs, 2, device="cuda") * 2 - 1 for _ in range(8)]
    k = max(20, min(2000, int(2e8 / max(total, 1))))

    for i in range(max(10, k // 4)):                      # warmup (includes compilation)
        step(acts[i % 8])
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(k):
            step(acts[i % 8])
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)    # seconds
    t = sorted(times)[len(times) // 2]
    return {
        "n_copies": n_copies, "n_envs": n_envs, "total_envs": total, "mode": mode,
        "steps_per_block": k, "block_seconds_median": t,
        "env_steps_per_sec": total * k / t,
        "us_per_batch_step": t / k * 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="torch")
    ap.add_argument("--mode", default="eager")
    ap.add_argument("--n-envs", type=int, nargs="+", required=True)
    ap.add_argument("--n-copies", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    git = subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for n in args.n_envs:
        r = bench_torch(args.n_copies, n, args.mode, args.repeats)
        rows.append(r)
        print(f"{args.impl}/{args.mode} C={args.n_copies} N={n:>8d}: "
              f"{r['env_steps_per_sec']:.3e} env-steps/s  ({r['us_per_batch_step']:.1f} us/batch-step)")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_envbench_{args.impl}_{args.mode}{args.tag}.json"
    out.write_text(json.dumps({
        "impl": args.impl, "mode": args.mode, "git": git,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "rows": rows,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
