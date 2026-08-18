"""Benchmark the whole fused training iteration on AntMaze — environment, agent and bonus in
one compiled program — on the graphics card. Writes one JSON into results/ beside this file
and prints one row per (map, bonus, copies) setting.

The measured unit is one training ITERATION: a rollout of `num_steps` env steps (each
frame_skip physics steps) plus statistics, advantages and the update. Total env steps per
second is copies x n_envs x num_steps / seconds-per-iteration — the number that compares
directly with the PointMaze ledgers.

Usage (serval05, while this session holds the H100 lock):
  PYTHONNOUSERSITE=1 python bench_antmaze_train.py --maps large --bonuses rnd_next_state none \
      --copies 512 1024 2048 --out results/<name>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

RESULTS = Path(__file__).resolve().parent / "results"


def peak_device_mb():
    """Peak device memory the JAX allocator has held in this process, in mebibytes."""
    import jax
    stats = jax.local_devices()[0].memory_stats()
    if stats is None or "peak_bytes_in_use" not in stats:
        raise RuntimeError("this jax build reports no device memory statistics")
    return stats["peak_bytes_in_use"] / 2 ** 20


def bench(map_name, bonus, n_copies, num_steps, n_envs, update_style, rounds=11, iters=10):
    """Median seconds per training iteration over `rounds` rounds of `iters` iterations."""
    import jax
    import jax.numpy as jnp
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.envs.antmaze.am_common import preset
    from exploration_platform.training.runner import Runner

    cfg = PPOConfig(n_copies=n_copies, n_envs=n_envs, num_steps=num_steps,
                    update_style=update_style)
    tr = Runner(cfg, bonus=bonus, env_cfg=preset(map_name))
    state = tr.prime(tr.init_state(run_seed=1))
    lr = jnp.asarray(cfg.learning_rate, jnp.float32)

    t0 = time.perf_counter()
    state, m = tr.iterate(state, lr)
    jax.block_until_ready(m["loss"])
    compile_s = time.perf_counter() - t0

    times = []
    for r in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            state, m = tr.iterate(state, lr)
        jax.block_until_ready(m["loss"])
        times.append((time.perf_counter() - t0) / iters)
    assert bool(np.isfinite(float(m["loss"]))), "loss went non-finite during the bench"

    sec = float(np.median(times))
    env_steps = n_copies * n_envs * num_steps
    return {"map": map_name, "bonus": bonus, "copies": n_copies, "n_envs": n_envs,
            "num_steps": num_steps, "update_style": update_style,
            "seconds_per_iteration": sec,
            "total_env_steps_per_second": env_steps / sec,
            "env_steps_per_second_per_copy": n_envs * num_steps / sec,
            "hours_per_million_steps_per_copy": 1e6 / 3600.0 / (n_envs * num_steps / sec),
            "compile_seconds": compile_s,
            "peak_device_mb": peak_device_mb(),
            "times_all": times}


def main():
    """Walk (map, bonus, copies) in ascending copy order, print rows, write the JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["large"])
    ap.add_argument("--bonuses", nargs="+", default=["rnd_next_state", "none"])
    ap.add_argument("--copies", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--num-steps", type=int, default=128)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--style", default="full_batch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    rows = []
    for map_name in args.maps:
        for bonus in args.bonuses:
            for c in sorted(args.copies):
                r = bench(map_name, bonus, c, args.num_steps, args.n_envs, args.style)
                rows.append(r)
                print(f"{map_name:7s} {bonus:15s} C={c:5d}: "
                      f"{r['seconds_per_iteration']:8.4f} s/iter  "
                      f"total {r['total_env_steps_per_second']/1e6:7.3f} M/s  "
                      f"per-copy {r['env_steps_per_second_per_copy']:7.1f} /s  "
                      f"{r['hours_per_million_steps_per_copy']:7.2f} h/M-steps-copy  "
                      f"peak {r['peak_device_mb']/1024:6.2f} GiB  "
                      f"compile {r['compile_seconds']:.0f}s", flush=True)
    out = Path(args.out) if args.out else RESULTS / f"antmaze_train_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(jax.devices()[0]), "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
