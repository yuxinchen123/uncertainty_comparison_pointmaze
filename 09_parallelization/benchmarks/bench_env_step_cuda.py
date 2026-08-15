"""Benchmark the fused-CUDA env step throughput. Same protocol and JSON shape as
bench_env_step.py (CUDA events, warmup excluded, median of repeats), impl "cuda".

Usage (on serval05, under the H100 lock):
  python bench_env_step_cuda.py --n-envs 1000 10000 100000 1000000 4000000
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "pointmaze" / "common"))
sys.path.insert(0, str(BASE / "pointmaze" / "cuda_env"))

RESULTS = Path(__file__).resolve().parent / "results"


def bench(n_copies, n_envs, repeats, block):
    """Measure the full fused step() for one batch size."""
    from pm_common import EnvConfig
    from cuda_pointmaze import CudaPointMaze

    env = CudaPointMaze(EnvConfig(), n_copies, n_envs, device="cuda", dtype=torch.float32)
    env.block = block
    env.reset()
    total = n_copies * n_envs
    acts = [torch.rand(n_copies, n_envs, 2, device="cuda") * 2 - 1 for _ in range(8)]
    k = max(20, min(2000, int(2e8 / max(total, 1))))

    for i in range(max(10, k // 4)):
        env.step(acts[i % 8])
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(k):
            env.step(acts[i % 8])
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    t = sorted(times)[len(times) // 2]
    return {
        "n_copies": n_copies, "n_envs": n_envs, "total_envs": total, "mode": "fused",
        "block": block, "steps_per_block": k, "block_seconds_median": t,
        "env_steps_per_sec": total * k / t,
        "us_per_batch_step": t / k * 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, nargs="+", required=True)
    ap.add_argument("--n-copies", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    git = subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for n in args.n_envs:
        r = bench(args.n_copies, n, args.repeats, args.block)
        rows.append(r)
        print(f"cuda/fused C={args.n_copies} N={n:>8d}: "
              f"{r['env_steps_per_sec']:.3e} env-steps/s  ({r['us_per_batch_step']:.1f} us/batch-step)")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_envbench_cuda_fused{args.tag}.json"
    out.write_text(json.dumps({
        "impl": "cuda", "mode": "fused", "git": git,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "rows": rows,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
