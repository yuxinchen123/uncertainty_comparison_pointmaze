#!/usr/bin/env python3
"""
Profile the SAC + intrinsic-bonus training loop (the hot path of train.py).

Builds the exact training stack train.py.run() builds (env stack, intrinsic model, SAC with the
VectorIntrinsicReplayBuffer), warms up past SAC's learning_starts, then cProfiles model.learn() over a
short step budget. Eval / callbacks are deliberately excluded so the profile is the pure train loop.

Outputs one JSON: {node, algorithm, beta, steps, warmup, threads, wall_s, fps, top_tottime, top_cumtime}.
top_* are lists of {func, tottime|cumtime, ncalls}. Run on a node via the slurm launcher.

Usage:
  python profile_train.py --algorithm rnd_state --beta 100 --steps 30000 --warmup 2000 \
      --threads 2 --seed 0 --out <path.json>
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)


def build_model(cfg):
    """Replicate train.py.run() setup up to the SAC model (no eval env, no callbacks)."""
    # import here so torch thread settings (set in main before this) take effect at import-influenced ops
    import random
    import numpy as np
    import torch
    import gymnasium as gym
    import gymnasium_robotics
    import train as T

    # seed every RNG together, exactly as run()
    seed = cfg.a_seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gym.register_envs(gymnasium_robotics)

    # base env -> cells -> train env stack -> spaces (mirrors run() lines 347-364)
    base_env = T.make_base_env(cfg)
    goal_cell, start_cell = T.select_cells(cfg, base_env, seed)
    train_flat, train_position, train_position_velocity = T.build_env_stack(cfg, base_env, start_cell, goal_cell)
    full_obs_dim = int(np.prod(train_flat.observation_space.shape))
    obs_shape = (full_obs_dim,)
    action_dim = int(np.prod(train_flat.action_space.shape))

    # intrinsic model from the registry (only when beta>0), then SAC with the intrinsic replay buffer
    intrinsic_model = None
    if cfg.beta > 0:
        ctx = T.EnvContext(
            obs_shape=obs_shape, action_dim=action_dim,
            observation_space=train_flat.observation_space, action_space=train_flat.action_space,
            position_wrapper=train_position, position_velocity_wrapper=train_position_velocity,
        )
        intrinsic_model = T.build_intrinsic_model(cfg.algorithm, cfg, ctx)
    train_vec = T.wrap_for_rollout(train_flat, cfg, intrinsic_model, seed)
    model = T.build_sac(cfg, train_vec, intrinsic_model, seed)
    return model


def top_functions(stats: pstats.Stats, key: str, n: int = 25) -> list:
    """Extract the top-n functions from a pstats.Stats by 'tottime' or 'cumtime'.

    before: stats.stats = {(file,line,func): (cc, nc, tt, ct, callers)}
    after:  [{"func": "file:line(func)", "tottime"/"cumtime": float, "ncalls": int}, ...] sorted desc.
    """
    # pull (func-id, ncalls, tottime, cumtime) rows and sort by the requested time column
    rows = []
    for (fname, lineno, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        short = f"{os.path.basename(fname)}:{lineno}({func})"
        rows.append((short, tt, ct, nc))
    idx = 1 if key == "tottime" else 2
    rows.sort(key=lambda r: r[idx], reverse=True)
    return [{"func": r[0], key: round(r[idx], 4), "ncalls": r[3]} for r in rows[:n]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", default="rnd_state")
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=30000, help="profiled training steps (cProfile, for hotspots)")
    p.add_argument("--bench_steps", type=int, default=20000, help="non-profiled steps timed for the real fps")
    p.add_argument("--warmup", type=int, default=2000, help="steps to run (unprofiled) to fill the buffer first")
    p.add_argument("--threads", type=int, default=2, help="torch.set_num_threads")
    p.add_argument("--profile", action="store_true", help="also run the cProfile hotspot pass (slower)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu (default) or cuda for GPU bold-runs")
    p.add_argument("--condition", default="baseline", help="label for this run's optimization condition")
    # performance switches (passed through to train.Config; default off = baseline)
    p.add_argument("--opt_torch_reward", action="store_true")
    p.add_argument("--opt_polyak_foreach", action="store_true")
    p.add_argument("--sac_train_freq", type=int, default=1)
    p.add_argument("--sac_gradient_steps", type=int, default=1)
    p.add_argument("--interop_threads", type=int, default=0, help="torch.set_num_interop_threads (0=leave default)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    import train as T
    # pin torch CPU threads BEFORE building the model so intra-op parallelism is controlled + reproducible
    torch.set_num_threads(args.threads)
    if args.interop_threads > 0:
        torch.set_num_interop_threads(args.interop_threads)

    # short, eval-free config matching the run-2 fixed knobs (PointMaze_Large, top_right, 1e6-style settings),
    # plus the performance switches under test
    cfg = T.Config(
        env_name="PointMaze_Large-v3", a_seed=args.seed, total_timesteps=args.steps,
        eval_freq=10_000_000, n_eval_episodes=1, n_eval_episodes_final=1, device=args.device,
        beta=args.beta, algorithm=args.algorithm, discount_factor=0.999, env_max_episode=400,
        goal_position="top_right", rnd_obs_norm=True, rnd_distance="mse", rnd_output_dim=128,
        n_predictors=1, apply_termination_wrapper=False, use_wandb=False,
        opt_torch_reward=args.opt_torch_reward, opt_polyak_foreach=args.opt_polyak_foreach,
        sac_train_freq=args.sac_train_freq, sac_gradient_steps=args.sac_gradient_steps,
    )
    if not T.REGISTRY[cfg.algorithm].builds_model:
        cfg.beta = 0

    model = build_model(cfg)

    # warmup: run some steps unprofiled so the replay buffer is past learning_starts (intrinsic path is hot)
    if args.warmup > 0:
        model.learn(total_timesteps=args.warmup, reset_num_timesteps=True)

    # clean fps: time bench_steps with NO cProfile (cProfile distorts throughput) -> the real steps/sec
    t0 = time.time()
    model.learn(total_timesteps=args.bench_steps, reset_num_timesteps=False)
    bench_wall = time.time() - t0
    fps_bench = round(args.bench_steps / bench_wall, 2)

    out = {
        "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "condition": args.condition, "algorithm": args.algorithm, "beta": cfg.beta, "device": args.device,
        "warmup": args.warmup, "threads": args.threads, "interop_threads": args.interop_threads, "seed": args.seed,
        "opt_torch_reward": args.opt_torch_reward, "sac_train_freq": args.sac_train_freq,
        "sac_gradient_steps": args.sac_gradient_steps,
        "bench_steps": args.bench_steps, "bench_wall_s": round(bench_wall, 3), "fps": fps_bench,
    }
    # optional cProfile pass for the function-level hotspot breakdown (separate, slower segment)
    if args.profile:
        pr = cProfile.Profile()
        t0 = time.time()
        pr.enable()
        model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
        pr.disable()
        stats = pstats.Stats(pr, stream=io.StringIO())
        out.update({
            "profiled_steps": args.steps, "profiled_wall_s": round(time.time() - t0, 3),
            "top_tottime": top_functions(stats, "tottime", 25),
            "top_cumtime": top_functions(stats, "cumtime", 25),
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[profile_train] node={out['node']} {args.algorithm} fps={out['fps']} bench_wall={out['bench_wall_s']}s -> {args.out}")


if __name__ == "__main__":
    main()
