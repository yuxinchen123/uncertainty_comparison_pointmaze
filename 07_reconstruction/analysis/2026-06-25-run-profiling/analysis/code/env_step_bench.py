#!/usr/bin/env python3
"""
Measure the PointMaze env-step cost in isolation (random actions, no SAC), to BOUND how much a faster
(C/C++) env could help. The physics is already C (MuJoCo); this measures the full gymnasium env.step()
including the Python wrapper stack the project adds. Compare env-step ms to the ~50 ms full per-step time:
that ratio is the absolute ceiling on any env-only speedup.

Usage: python env_step_bench.py [--steps 20000]
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    args = p.parse_args()

    import gymnasium as gym
    import gymnasium_robotics
    import train as T

    random.seed(0); np.random.seed(0)
    gym.register_envs(gymnasium_robotics)
    cfg = T.Config(env_name="PointMaze_Large-v3", a_seed=0, goal_position="top_right",
                   env_max_episode=400, discount_factor=0.999, apply_termination_wrapper=False)
    base = T.make_base_env(cfg)
    goal, start = T.select_cells(cfg, base, 0)
    env, _, _ = T.build_env_stack(cfg, base, start, goal)  # the exact flat env the runs train on

    obs, _ = env.reset(seed=0)
    # time the raw env loop: sample a random action, step, reset on episode end
    t0 = time.time()
    for _ in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    dt = time.time() - t0
    ms = dt / args.steps * 1000.0
    print(f"[env_step_bench] PointMaze_Large env.step(): {args.steps} steps in {dt:.2f}s "
          f"= {ms:.3f} ms/step = {args.steps/dt:.0f} env-steps/s")
    print(f"[env_step_bench] at ~50 ms/full-SAC-step, the env is ~{ms/50*100:.1f}% of per-step time "
          f"=> a perfect (free) env would give at most ~{100/(1-ms/50)-100 if ms < 50 else 0:.1f}% e2e speedup")


if __name__ == "__main__":
    main()
