#!/usr/bin/env python3
"""
Quick throughput check for the fully-JAX run-4 trainer: sbx.SAC + the JAX RND (jax_rnd.JaxRND) at step time
on PointMaze_Large (rnd_state). Compares to the torch-RND number (×1.66) to see how much folding RND into
JAX recovers toward the pure-SAC ×2.0 on PointMaze. fps only (correctness of the RND is tested separately).

Usage: <jax-venv py> run4_bench.py --steps 8000 --warmup 1500
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import gymnasium as gym

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jax_rnd import JaxRND  # noqa: E402


class _JaxStepRND(gym.Wrapper):
    """Step-time RND using the fully-JAX JaxRND on the visited state (feature='rnd_state')."""

    def __init__(self, env, jax_rnd, beta):
        super().__init__(env)
        self.rnd = jax_rnd
        self.beta = beta

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        x = np.asarray(obs, np.float32)[None]  # rnd_state feature = the observation
        intr = float(self.rnd.compute(x)[0])
        self.rnd.update(x)
        return obs, r + self.beta * intr, term, trunc, info


def build_env(seed, beta):
    """Exact run-2 PointMaze flat env + the JAX-RND step-time wrapper (rnd_state)."""
    import random
    import torch
    import gymnasium_robotics
    import train as T
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gym.register_envs(gymnasium_robotics)
    cfg = T.Config(env_name="PointMaze_Large-v3", a_seed=seed, goal_position="top_right", env_max_episode=400,
                   discount_factor=0.999, apply_termination_wrapper=False, beta=beta, algorithm="rnd_state")
    base = T.make_base_env(cfg)
    goal, start = T.select_cells(cfg, base, seed)
    env, _, _ = T.build_env_stack(cfg, base, start, goal)
    obs_dim = int(np.prod(env.observation_space.shape))
    return _JaxStepRND(env, JaxRND(input_dim=obs_dim, seed=seed), beta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=1500)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from sbx import SAC
    env = build_env(args.seed, beta=100.0)
    model = SAC("MlpPolicy", env, learning_starts=1000, batch_size=256, train_freq=1, gradient_steps=1,
                buffer_size=200_000, learning_rate=3e-4, gamma=0.999, tau=0.005, verbose=0, seed=args.seed)
    model.learn(total_timesteps=args.warmup)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    fps = round(args.steps / (time.time() - t0), 2)
    print(f"[run4_bench] sbx + JAX-RND on PointMaze rnd_state: fps={fps} "
          f"(torch-RND was 54.5; pure-SAC 72.7; SB3 32.8)")


if __name__ == "__main__":
    main()
