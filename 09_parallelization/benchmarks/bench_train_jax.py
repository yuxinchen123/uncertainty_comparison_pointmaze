"""Benchmark the JAX PPO+RND training-iteration throughput on GPU. Writes one JSON to
results/ (impl "jax_ppo"). Also prints the first five individual iteration times so a
per-iteration recompile would be visible as a persistently long first-of-each shape.

Usage (serval05, under the H100 lock):
  python bench_train_jax.py --n-copies 8 32 128 --styles full_batch epoch_minibatch
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))

RESULTS = Path(__file__).resolve().parent / "results"


def bench(n_copies, style, repeats=5, num_steps=128, n_envs=4, timing="pipelined"):
    """Median seconds per training iteration (rollout + update) for one config."""
    import jax
    import jax.numpy as jnp
    from jax_ppo_rnd import PPOConfig, JaxPPORND

    cfg = PPOConfig(n_copies=n_copies, update_style=style, num_steps=num_steps, n_envs=n_envs)
    tr = JaxPPORND(cfg)
    state = tr.init_state()
    key = jax.random.PRNGKey(1)
    state = tr.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    lr = jnp.asarray(cfg.learning_rate, jnp.float32)

    # warmup (includes the one compilation) + individual early timings
    early = []
    for it in range(1, 6):
        t0 = time.perf_counter()
        state, m = tr._iterate(state, jax.random.fold_in(key, it), lr)
        jax.block_until_ready(m["loss"])
        early.append(time.perf_counter() - t0)

    k = max(10, min(200, int(20.0 / max(early[-1], 1e-4))))
    times = []
    it0 = 10
    for rep in range(repeats):
        t0 = time.perf_counter()
        for it in range(it0, it0 + k):
            state, m = tr._iterate(state, jax.random.fold_in(key, it), lr)
            # "sync" waits for each iteration, so the number includes any host-side gap;
            # "pipelined" waits once per block, letting the host run ahead of the device
            if timing == "sync":
                jax.block_until_ready(m["loss"])
        jax.block_until_ready(m["loss"])
        times.append((time.perf_counter() - t0) / k)
        it0 += k
    med = sorted(times)[len(times) // 2]
    env_steps = cfg.num_steps * cfg.n_copies * cfg.n_envs
    return {
        "n_copies": n_copies, "n_envs": cfg.n_envs, "num_steps": cfg.num_steps,
        "style": style, "timing": timing, "iters_timed_per_block": k,
        "early_iteration_seconds": [round(t, 4) for t in early],
        "sec_per_iteration_median": med,
        "iterations_per_sec": 1.0 / med,
        "env_steps_per_sec": env_steps / med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--styles", nargs="+", default=["full_batch", "epoch_minibatch"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--timing", default="pipelined", choices=["pipelined", "sync"])
    ap.add_argument("--num-steps", type=int, default=128)
    ap.add_argument("--n-envs", type=int, default=4)
    args = ap.parse_args()

    import jax
    git = subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for style in args.styles:
        for c in args.n_copies:
            r = bench(c, style, num_steps=args.num_steps, n_envs=args.n_envs, timing=args.timing)
            rows.append(r)
            print(f"jax_ppo/{style} C={c:>4d}: {r['iterations_per_sec']:.2f} iter/s  "
                  f"{r['env_steps_per_sec']:.3e} env-steps/s  early={r['early_iteration_seconds']}")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_trainbench_jax_ppo{args.tag}.json"
    out.write_text(json.dumps({
        "impl": "jax_ppo", "git": git, "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()], "rows": rows,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
