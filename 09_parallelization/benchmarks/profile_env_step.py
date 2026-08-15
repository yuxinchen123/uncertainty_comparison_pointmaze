"""Profile the compiled env step at one batch size: kernel list with times.

Usage: python profile_env_step.py --n-envs 1000000
Prints the CUDA kernel summary so the optimization loop knows what remains in the step.
"""
import argparse
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "pointmaze" / "common"))
sys.path.insert(0, str(BASE / "pointmaze" / "torch_env"))
from pm_common import EnvConfig  # noqa: E402
from torch_pointmaze import TorchPointMaze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=1000000)
    ap.add_argument("--n-copies", type=int, default=1)
    args = ap.parse_args()

    env = TorchPointMaze(EnvConfig(), args.n_copies, args.n_envs, device="cuda",
                         dtype=torch.float32)
    env.reset()
    step = torch.compile(env.step, fullgraph=True, dynamic=False)
    act = torch.rand(args.n_copies, args.n_envs, 2, device="cuda") * 2 - 1

    for _ in range(30):
        step(act)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            step(act)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15,
                                    max_name_column_width=80))


if __name__ == "__main__":
    main()
