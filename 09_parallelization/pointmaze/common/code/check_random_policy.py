"""Validation contract item 4: random-policy behavior of a GPU env vs the MuJoCo reference.

Runs 256 episodes (400-step cap, run-6 config) with uniform random actions in both the
reference Gymnasium-Robotics env and the torch batched env, then compares the distributions
of (a) episode return and (b) per-cell visit frequency. Gates: absolute difference in mean
return <= 3 standard errors; visit-distribution total-variation distance <= 0.05.

Run: PYTHONNOUSERSITE=1 MUJOCO_GL=disable <exploration python> check_random_policy.py
"""
import sys
from pathlib import Path

import numpy as np

COMMON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(COMMON.parent / "torch_env"))
from pm_common import MAPS, EnvConfig  # noqa: E402

EPISODES, T = 256, 400


def cell_of(xy, rows=9, cols=12):
    """World (x, y) -> flat cell index."""
    j = min(max(int(xy[0] + cols / 2.0), 0), cols - 1)
    i = min(max(int(rows / 2.0 - xy[1]), 0), rows - 1)
    return i * cols + j


def run_reference(seed):
    """256 sequential episodes in the MuJoCo env; returns (returns [E], visits [108])."""
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make("PointMaze_Large-v3", continuing_task=True, max_episode_steps=T).unwrapped
    rng = np.random.default_rng(seed)
    returns, visits = [], np.zeros(9 * 12)
    for e in range(EPISODES):
        env.reset(seed=int(rng.integers(2**31)),
                  options={"reset_cell": [7, 1], "goal_cell": [1, 10]})
        total = 0.0
        for t in range(T):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 2))
            total += r
            visits[cell_of(env.point_env.data.qpos)] += 1
        returns.append(total)
    env.close()
    return np.array(returns), visits


def run_torch(seed):
    """256 parallel episodes in the batched torch env; same outputs."""
    import torch
    from torch_pointmaze import TorchPointMaze
    env = TorchPointMaze(EnvConfig(max_episode_steps=T), 1, EPISODES, device="cpu",
                         base_seed=seed, dtype=torch.float32)
    env.reset()
    g = torch.Generator().manual_seed(seed)
    returns = torch.zeros(1, EPISODES)
    visits = np.zeros(9 * 12)
    for t in range(T):
        a = torch.rand(1, EPISODES, 2, generator=g) * 2 - 1
        obs, r, term, trunc, final = env.step(a)
        returns += r
        xy = final[0, :, :2].numpy()
        j = np.clip((xy[:, 0] + 6).astype(int), 0, 11)
        i = np.clip((4.5 - xy[:, 1]).astype(int), 0, 8)
        np.add.at(visits, i * 12 + j, 1)
    return returns[0].numpy(), visits


def main():
    ref_r, ref_v = run_reference(0)
    t_r, t_v = run_torch(0)
    open_cells = ~(np.array(MAPS["large"]) == 1).reshape(-1)
    p = ref_v[open_cells] / ref_v[open_cells].sum()
    q = t_v[open_cells] / t_v[open_cells].sum()
    tv = 0.5 * np.abs(p - q).sum()
    se = np.sqrt(ref_r.var() / len(ref_r) + t_r.var() / len(t_r))
    dmean = abs(ref_r.mean() - t_r.mean())
    print(f"episode return: reference mean {ref_r.mean():.4f} (sd {ref_r.std():.3f}), "
          f"torch mean {t_r.mean():.4f} (sd {t_r.std():.3f}); |d|={dmean:.4f} vs 3*SE={3*se:.4f}")
    print(f"visit-distribution total variation over open cells: {tv:.4f} (gate 0.05)")
    ok = dmean <= 3 * se and tv <= 0.05
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
