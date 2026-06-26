#!/usr/bin/env python3
"""
JAX vs torch end-to-end A/B on the EXACT train-run-2 task (PointMaze_Large-v3 + RND), short runs for stats.

Backends: --backend sb3 (Stable-Baselines3, torch) | sbx (SB3+JAX). Both train SAC on the project's exact
run-2 env stack (reused from train.py.build_env_stack), with the SAME hyperparameters. The intrinsic bonus,
when --rnd is on, is the project's torch RND (rnd.py) applied at STEP time via a thin wrapper (a standard
RND variant; run-2 applies it at sample time, but for train_freq=1 the per-step RND cost is comparable).
Keeping the RND in torch for both backends isolates the JAX speedup to the SAC update (a realistic partial
port). Reports steady-state fps (after a warmup that includes JAX JIT compile).

Usage (same machine for a clean ratio, or one per node for stats):
  <exploration py> sac_pointmaze_bench.py --backend sb3 --rnd --steps 40000 --warmup 3000 --out sb3.json
  <jax-venv py>    sac_pointmaze_bench.py --backend sbx --rnd --steps 40000 --warmup 3000 --out sbx.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import gymnasium as gym

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/src")  # rnd_exploration package lives under src/ (not pip-installed in the jax venv)


def build_pointmaze(seed, algorithm, beta, with_rnd):
    """Build the exact run-2 PointMaze flat env (extrinsic reward) + optional step-time torch-RND wrapper."""
    import random
    import gymnasium as gym
    import gymnasium_robotics
    import torch
    import train as T

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gym.register_envs(gymnasium_robotics)
    cfg = T.Config(env_name="PointMaze_Large-v3", a_seed=seed, goal_position="top_right",
                   env_max_episode=400, discount_factor=0.999, apply_termination_wrapper=False,
                   beta=beta, algorithm=algorithm, rnd_obs_norm=True, rnd_distance="mse",
                   rnd_output_dim=128, n_predictors=1, device="cpu")
    base = T.make_base_env(cfg)
    goal, start = T.select_cells(cfg, base, seed)
    env, posw, posvelw = T.build_env_stack(cfg, base, start, goal)  # flat env, extrinsic reward
    if not with_rnd or not T.REGISTRY[algorithm].builds_model:
        return env
    # build the project's RND model and wrap the env to add beta*intrinsic at step time
    obs_dim = int(np.prod(env.observation_space.shape)); act_dim = int(np.prod(env.action_space.shape))
    ctx = T.EnvContext(obs_shape=(obs_dim,), action_dim=act_dim,
                       observation_space=env.observation_space, action_space=env.action_space,
                       position_wrapper=posw, position_velocity_wrapper=posvelw)
    rnd = T.build_intrinsic_model(algorithm, cfg, ctx)
    return _StepRND(env, rnd, beta)


class _StepRND(gym.Wrapper):
    """Gymnasium wrapper: on each step, add beta * RND-intrinsic(visited state) and train the RND predictor once.

    A standard step-time RND (as in CleanRL ppo_rnd). before: env returns extrinsic r; after: r + beta*intrinsic,
    with the predictor updated on the visited state each step (the torch RND cost is the same for both backends)."""

    def __init__(self, env, rnd, beta):
        super().__init__(env)
        self.rnd = rnd
        self.beta = beta

    def step(self, action):
        import torch
        obs, r, term, trunc, info = self.env.step(action)
        # build a 1-row samples dict and add the intrinsic reward, then train the predictor on this state
        ot = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        at = torch.as_tensor(np.asarray(action), dtype=torch.float32).unsqueeze(0)
        samples = {"observations": ot, "next_observations": ot, "actions": at}
        intr = float(self.rnd.compute(samples).reshape(-1)[0])
        self.rnd.update(samples)
        return obs, r + self.beta * intr, term, trunc, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["sb3", "sbx"], required=True)
    p.add_argument("--rnd", action="store_true", help="add step-time RND intrinsic (else extrinsic-only SAC)")
    p.add_argument("--algorithm", default="rnd_state")
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=40000, help="timed steady-state steps")
    p.add_argument("--warmup", type=int, default=3000, help="first learn() = JIT warmup + buffer fill")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    env = build_pointmaze(args.seed, args.algorithm, args.beta, args.rnd)

    common = dict(learning_starts=1000, batch_size=256, train_freq=1, gradient_steps=1,
                  buffer_size=200_000, learning_rate=3e-4, gamma=0.999, tau=0.005, verbose=0, seed=args.seed)
    if args.backend == "sbx":
        from sbx import SAC
    else:
        from stable_baselines3 import SAC
    model = SAC("MlpPolicy", env, **common)

    tw = time.time()
    model.learn(total_timesteps=args.warmup)          # warmup (JAX JIT compiles here for sbx)
    warmup_s = time.time() - tw
    t0 = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)  # timed steady state
    dt = time.time() - t0
    fps = round(args.steps / dt, 2)

    out = {"backend": args.backend, "rnd": args.rnd, "algorithm": args.algorithm,
           "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename), "seed": args.seed,
           "steps": args.steps, "warmup_s": round(warmup_s, 2), "wall_s": round(dt, 2), "fps": fps}
    print(f"[sac_pointmaze_bench] backend={args.backend} rnd={args.rnd} algo={args.algorithm} "
          f"fps={fps} steady_wall={dt:.1f}s warmup={warmup_s:.1f}s")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
