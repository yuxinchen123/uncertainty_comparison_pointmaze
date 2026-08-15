"""Kernel-level profile of one training iteration at a chosen copy count.

The phase profile says which of the three stages costs the most; this says which individual
device programs inside them do. It runs the iteration's three bodies without graph capture
(the same kernels the captured graph replays, just launched one at a time so the profiler can
name them) and reports device time per kernel, grouped and sorted.

Usage (through the H100 lock wrapper):
  python profile_kernels.py --n-copies 4096 --style epoch_minibatch
"""
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"


def kernel_table(fn, reps):
    """Device time per kernel name over `reps` calls of fn, as {name: microseconds}."""
    # warm up so compilation and allocator growth are not inside the profiled window
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    per_kernel = defaultdict(float)
    for e in prof.key_averages():
        # device_time_total is microseconds summed over every call of this kernel
        t = getattr(e, "device_time_total", 0.0) or 0.0
        if t > 0 and e.key not in ("cudaDeviceSynchronize",):
            per_kernel[e.key] += t / reps
    return dict(per_kernel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=4096)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from torch_ppo_rnd import PPORND, production_config

    C = args.n_copies
    # one_graph=False so the three bodies can be called separately and profiled by kernel
    t = PPORND(production_config(C, style=args.style, one_graph=False, capture_update=False),
               device="cuda")
    t.prime_obs_rms()
    t._perm = torch.arange(t.cfg.num_steps * t.cfg.n_envs, device=t.device).expand(
        t.cfg.update_epochs, C, t.cfg.num_steps * t.cfg.n_envs).contiguous()
    t._loss_out = torch.zeros((), device=t.device)
    t._rollout_body()
    t._post_body()

    stages = {
        "rollout": lambda: t._rollout_body(),
        "post": lambda: t._post_body(),
        "update": lambda: t._update_body_captured(),
    }
    out = {}
    for name, fn in stages.items():
        tbl = kernel_table(fn, args.reps)
        total = sum(tbl.values())
        out[name] = {"total_us": total, "kernels": tbl}
        print(f"\n== {name}: {total/1000:.2f} ms of device time, {C} copies, {args.style} ==")
        for k, v in sorted(tbl.items(), key=lambda kv: -kv[1])[:18]:
            print(f"  {v/1000:8.3f} ms  {v/total*100:5.1f}%  {k[:96]}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_profile_kernels_C{C}{args.tag}.json"
    p.write_text(json.dumps({
        "n_copies": C, "style": args.style, "torch": torch.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "stages": out}, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
