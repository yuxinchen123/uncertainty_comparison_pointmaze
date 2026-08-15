"""Benchmark the JAX env step throughput on GPU. Same protocol as bench_env_step.py:
CUDA-side timing via block_until_ready around K-step blocks, warmup excluded, median of
repeats. Modes: jit (per-step jit, donated state) | scan (jax.lax.scan over K steps inside
one jit — the upper bound where step-loop overhead vanishes).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "pointmaze" / "common"))
sys.path.insert(0, str(BASE / "pointmaze" / "jax_env"))

RESULTS = Path(__file__).resolve().parent / "results"


def bench(n_copies, n_envs, mode, repeats, unroll=1):
    """Measure full step (dynamics + reward + auto-reset) for one batch size."""
    import jax
    import jax.numpy as jnp
    from pm_common import EnvConfig
    from jax_pointmaze import JaxPointMaze

    env = JaxPointMaze(EnvConfig(), n_copies, n_envs)
    state = env.reset()
    total = n_copies * n_envs
    k = max(20, min(2000, int(2e8 / max(total, 1))))
    acts = [jnp.asarray(np.random.default_rng(i).uniform(-1, 1, (n_copies, n_envs, 2)),
                        jnp.float32) for i in range(8)]

    if mode == "jit":
        step = jax.jit(env.step, donate_argnums=0)

        def run_block(state):
            for i in range(k):
                state, obs, reward, term, trunc, final = step(state, acts[i % 8])
            jax.block_until_ready(state)
            return state
    else:  # scan: whole K-step block is one compiled program
        acts_stack = jnp.stack([acts[i % 8] for i in range(k)])

        def body(state, act):
            state, obs, reward, term, trunc, final = env.step(state, act)
            return state, reward.sum()

        scan_fn = jax.jit(lambda s: jax.lax.scan(body, s, acts_stack, unroll=unroll),
                          donate_argnums=0)

        def run_block(state):
            state, rs = scan_fn(state)
            jax.block_until_ready(state)
            return state

    state = run_block(state)                      # warmup + compile
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        state = run_block(state)
        times.append(time.perf_counter() - t0)
    t = sorted(times)[len(times) // 2]
    return {
        "n_copies": n_copies, "n_envs": n_envs, "total_envs": total, "mode": mode,
        "unroll": unroll,
        "steps_per_block": k, "block_seconds_median": t,
        "env_steps_per_sec": total * k / t,
        "us_per_batch_step": t / k * 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="jit", choices=["jit", "scan"])
    ap.add_argument("--n-envs", type=int, nargs="+", required=True)
    ap.add_argument("--n-copies", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--unroll", type=int, default=1)
    args = ap.parse_args()

    import jax
    git = subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for n in args.n_envs:
        r = bench(args.n_copies, n, args.mode, args.repeats, args.unroll)
        rows.append(r)
        print(f"jax/{args.mode} C={args.n_copies} N={n:>8d}: "
              f"{r['env_steps_per_sec']:.3e} env-steps/s  ({r['us_per_batch_step']:.1f} us/batch-step)")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_envbench_jax_{args.mode}{args.tag}.json"
    out.write_text(json.dumps({
        "impl": "jax", "mode": args.mode, "git": git, "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()], "rows": rows,
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
