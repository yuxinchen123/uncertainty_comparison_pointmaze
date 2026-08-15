"""Compare the two forms of the two-nearest-wall selection directly, state for state.

The gates (fixtures, the A/B against the torch env, reset identity) all compare the CUDA env
against something else. This compares the two CUDA builds against each other on a long rollout
that exercises contacts and auto-resets, which is the sharpest test of the claim that ranking
by counting selects the same two candidates as the running tournament.

  PM_TOURNAMENT=count  python test_tournament_forms_equal.py dump
  PM_TOURNAMENT=select python test_tournament_forms_equal.py dump
  python test_tournament_forms_equal.py compare
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
from pm_common import EnvConfig  # noqa: E402

OUT = HERE / ".tournament_tmp"
C, N, STEPS = 4, 64, 200


def dump():
    """Roll out with a fixed action sequence and save the whole state trajectory."""
    from cuda_pointmaze import CudaPointMaze
    form = os.environ.get("PM_TOURNAMENT", "count")
    # a short episode cap so auto-resets fire inside the rollout
    env = CudaPointMaze(EnvConfig(max_episode_steps=37), C, N, device="cuda", base_seed=5)
    env.reset()
    g = torch.Generator(device="cuda").manual_seed(99)
    states, rewards = [], []
    for _ in range(STEPS):
        act = torch.rand(C, N, 2, generator=g, device="cuda") * 2 - 1
        _, r, _, _, final = env.step(act)
        states.append(final.cpu().numpy().copy())
        rewards.append(r.cpu().numpy().copy())
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"{form}.npz", states=np.array(states), rewards=np.array(rewards),
             reset_count=env.reset_count.cpu().numpy())
    print(f"{form}: dumped {STEPS} steps, {int(env.reset_count.sum())} auto-resets total")


def compare():
    """Every step of both trajectories must match bitwise."""
    a = np.load(OUT / "count.npz")
    b = np.load(OUT / "select.npz")
    for k in ["states", "rewards", "reset_count"]:
        same = (a[k] == b[k]).all()
        worst = np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()
        print(f"{k}: {'BIT-IDENTICAL' if same else 'MISMATCH'} (worst |d| {worst:.3e})")
        assert same, k
    print("tournament forms: IDENTICAL")


if __name__ == "__main__":
    {"dump": dump, "compare": compare}[sys.argv[1]]()
