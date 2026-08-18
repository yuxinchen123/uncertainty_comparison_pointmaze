"""Benchmark the Montezuma environment step on the graphics card. Writes one JSON beside
this file into results/ and prints one table row per (map, copies) setting.

The measured unit is one ENVIRONMENT step — frame_skip (4) game frames plus the observation
stack — under random discrete actions, jitted alone with a donated state, so the number is the
environment's and contains no agent arithmetic. Copy counts are walked in ascending order so the peak-memory column is
per-row (the JAX high-water mark never resets within a process).

Usage (serval05, while this session holds the H100 lock):
  PYTHONNOUSERSITE=1 python bench_montezuma_env.py --copies 512 1024 2048 4096 \
      --out results/<name>.json
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


def bench(n_copies, rounds=11, steps_per_round=10):
    """Median seconds per env step over `rounds` rounds of `steps_per_round` steps."""
    import jax
    import jax.numpy as jnp
    from exploration_platform.envs.atari_montezuma.jax_montezuma import (JaxMontezuma,
                                                                         MontezumaConfig)

    env = JaxMontezuma(MontezumaConfig(), n_copies, 1)

    def block(state, key):
        """steps_per_round env steps of keyed random actions, one compiled program."""
        def body(carry, k):
            st = carry
            act = jax.random.randint(jax.random.fold_in(key, k), (n_copies, 1), 0,
                                     env.n_actions, jnp.int32)
            st, obs, rew, term, trunc, final = env.step(st, act)
            return st, rew.sum()
        state, rews = jax.lax.scan(body, state, jnp.arange(steps_per_round))
        return state, rews

    run = jax.jit(block, donate_argnums=(0,))
    state = env.reset()
    key = jax.random.PRNGKey(0)

    t0 = time.perf_counter()
    state, _ = run(state, key)
    jax.block_until_ready(state.stack)
    compile_s = time.perf_counter() - t0

    times = []
    for r in range(rounds):
        t0 = time.perf_counter()
        state, _ = run(state, jax.random.fold_in(key, 1000 + r))
        jax.block_until_ready(state.stack)
        times.append((time.perf_counter() - t0) / steps_per_round)
    assert bool(jnp.isfinite(state.stack).all()), "state went non-finite during the bench"

    sec = float(np.median(times))
    return {"map": "montezuma", "copies": n_copies, "n_envs": 1,
            "seconds_per_env_step": sec,
            "total_env_steps_per_second": n_copies / sec,
            "env_steps_per_second_per_copy": 1.0 / sec,
            "hours_per_million_steps_per_copy": 1e6 / 3600.0 * sec,
            "compile_seconds": compile_s,
            "peak_device_mb": peak_device_mb(),
            "times_all": times}


def main():
    """Walk (map, copies) in ascending copy order, print rows, write the JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    rows = []
    if True:
        for c in sorted(args.copies):
            r = bench(c)
            rows.append(r)
            print(f"montezuma C={c:5d}: {r['seconds_per_env_step']*1e3:8.3f} ms/env-step  "
                  f"total {r['total_env_steps_per_second']/1e6:7.3f} M/s  "
                  f"per-copy {r['env_steps_per_second_per_copy']:7.1f} /s  "
                  f"{r['hours_per_million_steps_per_copy']:6.2f} h/M-steps-copy  "
                  f"peak {r['peak_device_mb']/1024:6.2f} GiB  compile {r['compile_seconds']:.0f}s",
                  flush=True)
    out = Path(args.out) if args.out else RESULTS / f"montezuma_env_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(jax.devices()[0]), "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
