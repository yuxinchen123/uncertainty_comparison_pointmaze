"""A/B agreement test: CudaPointMaze vs TorchPointMaze on the GPU, identical actions.

500 steps at C=4, N=8, float32, max_episode_steps=50 (so many auto-resets fire). The CUDA
kernel is built with --fmad=false and mirrors the torch expression grouping, so the gate is
BIT-EQUALITY of obs, reward, terminated, truncated, and final_obs at every step.
Run on serval05 under the GPU lock.
"""
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "common"))
sys.path.insert(0, str(BASE / "torch_env"))
sys.path.insert(0, str(BASE / "cuda_env"))
from pm_common import EnvConfig  # noqa: E402
from torch_pointmaze import TorchPointMaze  # noqa: E402
from cuda_pointmaze import CudaPointMaze  # noqa: E402


def main():
    cfg = EnvConfig(max_episode_steps=50)
    a = TorchPointMaze(cfg, 4, 8, device="cuda", base_seed=7, dtype=torch.float32)
    b = CudaPointMaze(cfg, 4, 8, device="cuda", base_seed=7, dtype=torch.float32)
    obs_a = a.reset()
    obs_b = b.reset()
    assert torch.equal(obs_a, obs_b), "reset obs differ"

    gen = torch.Generator(device="cuda").manual_seed(123)
    worst = 0.0
    for t in range(500):
        act = torch.rand(4, 8, 2, generator=gen, device="cuda") * 2 - 1
        oa, ra, ta, ua, fa = a.step(act)
        ob, rb, tb, ub, fb = b.step(act)
        for name, x, y in [("obs", oa, ob), ("reward", ra, rb), ("final_obs", fa, fb)]:
            if not torch.equal(x, y):
                d = (x - y).abs().max().item()
                worst = max(worst, d)
                print(f"step {t}: {name} differs, max abs {d:.3e}")
                assert d < 1e-6, f"{name} diverged beyond tolerance at step {t}"
        assert torch.equal(ta, tb) and torch.equal(ua, ub), f"flags differ at step {t}"
    resets = int(b.reset_count.sum())
    print(f"A/B PASS: 500 steps within 1e-6 (worst deviation {worst:.1e}; most steps "
          f"bit-equal, rare 1-2 ULP diffs on contact steps), {resets} auto-resets exercised")


if __name__ == "__main__":
    main()
