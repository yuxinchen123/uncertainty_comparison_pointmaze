"""Unit tests for the env-layer logic of TorchPointMaze (reward, episode ends, auto-reset,
keyed RNG). Dynamics fidelity is covered separately by check_against_fixtures.py.

Run: PYTHONNOUSERSITE=1 <python> -m pytest test_torch_env.py -q   (CPU, no GPU needed)
"""
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "common"))
sys.path.insert(0, str(BASE / "torch_env"))
from pm_common import EnvConfig, cell_center  # noqa: E402
from torch_pointmaze import TorchPointMaze  # noqa: E402


def make(cfg=None, C=2, N=8, seed=0):
    """Small CPU env for logic tests."""
    return TorchPointMaze(cfg or EnvConfig(), C, N, device="cpu", base_seed=seed,
                          dtype=torch.float64)


def test_reset_ranges_and_shapes():
    """Golden path: reset obs is [C,N,4], positions within start center +-noise, vel 0."""
    env = make()
    obs = env.reset()
    assert obs.shape == (2, 8, 4)
    sx, sy = cell_center((7, 1), env.rows, env.cols)
    assert (obs[..., 0] - sx).abs().max() <= 0.25
    assert (obs[..., 1] - sy).abs().max() <= 0.25
    assert obs[..., 2:].abs().max() == 0
    # goal positions likewise within goal center +- noise
    gx, gy = cell_center((1, 10), env.rows, env.cols)
    assert (env.goal[..., 0] - gx).abs().max() <= 0.25
    assert (env.goal[..., 1] - gy).abs().max() <= 0.25


def test_keyed_rng_reproducible_and_distinct():
    """Same base seed -> bit-identical resets; different seed / copy / env / generation differ."""
    a, b = make(seed=1), make(seed=1)
    assert torch.equal(a.reset(), b.reset())
    c = make(seed=2)
    assert not torch.equal(a.reset(), c.reset())
    obs = a.reset()
    # per-copy and per-env draws differ (noise 0.25 makes collisions vanishingly unlikely)
    assert not torch.equal(obs[0], obs[1])
    assert not torch.equal(obs[0, 0], obs[0, 1])
    # generation keying: after one full episode the respawn differs from generation 0
    env = make(C=1, N=1)
    first = env.reset()
    for _ in range(env.cfg.max_episode_steps):
        obs2, _, _, _, _ = env.step(torch.zeros(1, 1, 2, dtype=torch.float64))
    assert not torch.equal(obs2[..., :2], first[..., :2])


def test_reward_at_goal_and_shift():
    """Reward is 1 within 0.45 of the goal, 0 outside; reward_shift adds a constant."""
    env = make(EnvConfig(reward_shift=-1.0), C=1, N=2)
    env.reset()
    # env 0 placed just inside the goal radius, env 1 far away; zero velocity, zero action
    env.pos[0, 0] = env.goal[0, 0] + torch.tensor([0.3, 0.0])
    env.pos[0, 1] = torch.tensor(cell_center((7, 1), env.rows, env.cols))
    _, reward, _, _, _ = env.step(torch.zeros(1, 2, 2, dtype=torch.float64))
    assert reward[0, 0].item() == 0.0   # 1 (goal) + shift (-1)
    assert reward[0, 1].item() == -1.0  # 0 + shift


def test_truncation_and_auto_reset():
    """Continuing task: truncated exactly at the step cap, then state respawns and count resets."""
    env = make(EnvConfig(max_episode_steps=5), C=1, N=3)
    env.reset()
    zero = torch.zeros(1, 3, 2, dtype=torch.float64)
    for t in range(4):
        _, _, terminated, truncated, _ = env.step(zero)
        assert not truncated.any() and not terminated.any()
    obs, _, terminated, truncated, final_obs = env.step(zero)
    assert truncated.all() and not terminated.any()
    assert (env.step_count == 0).all()
    # obs is the respawned state (vel 0 at start cell); final_obs is the pre-reset state
    assert obs[..., 2:].abs().max() == 0
    assert not torch.equal(obs, final_obs)


def test_termination_non_continuing():
    """continuing_task=False: reaching the goal terminates (not truncates) and auto-resets."""
    env = make(EnvConfig(continuing_task=False), C=1, N=1)
    env.reset()
    env.pos[0, 0] = env.goal[0, 0].clone()
    obs, reward, terminated, truncated, _ = env.step(torch.zeros(1, 1, 2, dtype=torch.float64))
    assert terminated.all() and not truncated.any()
    assert reward[0, 0].item() == 1.0
    assert (env.step_count == 0).all()


def test_final_obs_equals_obs_when_not_done():
    """Mid-episode, final_obs and obs are the same tensor values."""
    env = make(C=1, N=4)
    env.reset()
    obs, _, _, _, final_obs = env.step(torch.zeros(1, 4, 2, dtype=torch.float64))
    assert torch.equal(obs, final_obs)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
