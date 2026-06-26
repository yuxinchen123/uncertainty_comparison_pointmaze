#!/usr/bin/env python3
"""
Per-step phase breakdown for train run 2 (SB3 + RND, rnd_state), to build the 100K-step time table.

Times the phases that make up one training step on a 256-batch / single env step, so each part's share of
the per-step time can be reported. Combines with the measured fps to extrapolate to 100K steps.
Phases: env step (raw), RND compute+update (256-batch), polyak target update (256 critic params),
and the SAC critic+actor+optimizer remainder.

Usage: python breakdown.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, "/p/rlprojects/RND/07_reconstruction")
torch.set_num_threads(2)

import profile_train as PT  # noqa: E402
import train as T  # noqa: E402
import stable_baselines3.common.utils as U  # noqa: E402


def _time(fn, n):
    for _ in range(20):
        fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000.0  # ms/call


def main():
    cfg = T.Config(env_name="PointMaze_Large-v3", a_seed=0, total_timesteps=2000, eval_freq=10**9,
                   n_eval_episodes=1, n_eval_episodes_final=1, device="cpu", beta=100.0, algorithm="rnd_state",
                   discount_factor=0.999, env_max_episode=400, goal_position="top_right", rnd_obs_norm=True,
                   rnd_distance="mse", rnd_output_dim=128, n_predictors=1, apply_termination_wrapper=False,
                   use_wandb=False)
    model = PT.build_model(cfg)
    obs_dim = int(np.prod(model.observation_space.shape))
    act_dim = int(np.prod(model.action_space.shape))
    rnd = model.replay_buffer.intrinsic_reward_model

    # RND compute+update on a realistic 256-batch (same as run-2's per-gradient-step RND cost)
    batch = {"observations": torch.randn(256, obs_dim), "next_observations": torch.randn(256, obs_dim),
             "actions": torch.randn(256, act_dim)}

    def rnd_step():
        rnd.compute(batch); rnd.update(batch)
    rnd_ms = _time(rnd_step, 1000)

    # polyak target update on the 12 critic params (zip_strict, as SB3 calls it)
    shapes = [(256, obs_dim), (256,), (256, 256), (256,), (1, 256), (1,)] * 2
    params = [torch.randn(s) for s in shapes]
    tgt = [torch.randn(s) for s in shapes]
    polyak_ms = _time(lambda: U.polyak_update(params, tgt, 0.005), 2000)

    # raw env step
    env = model.get_env()
    env.reset()

    def env_step():
        env.step(np.array([env.action_space.sample()]))
    env_ms = _time(env_step, 2000)

    print(f"[breakdown] rnd_state per-step phases (ms): RND compute+update={rnd_ms:.3f} | "
          f"polyak={polyak_ms:.3f} | env.step(vec)={env_ms:.3f}")
    # write for the table
    out = {"rnd_compute_update_ms": round(rnd_ms, 3), "polyak_ms": round(polyak_ms, 3), "env_step_ms": round(env_ms, 3)}
    import json
    json.dump(out, open(os.path.join(CODE, "..", "intermediate_results_and_progress", "breakdown_phases.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
