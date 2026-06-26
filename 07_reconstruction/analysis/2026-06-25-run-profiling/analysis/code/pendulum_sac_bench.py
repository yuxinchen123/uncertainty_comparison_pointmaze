#!/usr/bin/env python3
"""
Benchmark SAC throughput on Pendulum-v1 with two backends, identical hyperparameters, to measure the JAX
speedup: --backend sb3 (Stable-Baselines3, torch) vs --backend sbx (SB3+JAX). The first learn() absorbs
JIT compilation (timed separately as warmup); the second learn() is the steady-state fps. Run both on the
SAME machine for a clean ratio.

Usage:
  <exploration py> pendulum_sac_bench.py --backend sb3  --steps 20000 --out sb3.json
  <jax-venv py>    pendulum_sac_bench.py --backend sbx  --steps 20000 --out sbx.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["sb3", "sbx"], required=True)
    p.add_argument("--steps", type=int, default=20000, help="timed steps (steady state)")
    p.add_argument("--warmup", type=int, default=3000, help="first learn() = JIT warmup + buffer fill")
    p.add_argument("--out", default="")
    args = p.parse_args()

    # same SAC hyperparameters for both backends so the only difference is torch vs JAX
    common = dict(learning_starts=1000, batch_size=256, train_freq=1, gradient_steps=1,
                  buffer_size=200_000, learning_rate=3e-4, gamma=0.99, tau=0.005, verbose=0, seed=0)
    if args.backend == "sbx":
        from sbx import SAC
    else:
        from stable_baselines3 import SAC

    model = SAC("MlpPolicy", "Pendulum-v1", **common)
    # warmup learn() (includes JAX JIT compile for sbx) -- timed separately
    tw = time.time()
    model.learn(total_timesteps=args.warmup)
    warmup_s = time.time() - tw
    # steady-state timed segment
    t0 = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    dt = time.time() - t0
    fps = round(args.steps / dt, 2)

    out = {"backend": args.backend, "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
           "steps": args.steps, "warmup_s": round(warmup_s, 2), "wall_s": round(dt, 2), "fps": fps}
    print(f"[pendulum_sac_bench] backend={args.backend} fps={fps} steady_wall={dt:.2f}s warmup={warmup_s:.2f}s")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
