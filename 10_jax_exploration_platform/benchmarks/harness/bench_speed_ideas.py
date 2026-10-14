"""Measure one bounded speed idea without changing the training algorithm."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Each candidate changes one execution choice relative to the corrected baseline.
IDEAS = {
    "baseline": {}, "01_flat_parameters": {"flat_params": True},
    "02_rollout_unroll_1": {"scan_unroll": 1},
    "03_rollout_unroll_4": {"scan_unroll": 4},
    "04_rollout_unroll_8": {"scan_unroll": 8},
    "05_rollout_unroll_32": {"scan_unroll": 32},
    "06_rollout_unroll_64": {"scan_unroll": 64},
    "07_copies_128": {"n_copies": 128},
    "08_copies_256": {"n_copies": 256},
    "09_dispatch_8": {"block_length": 8},
    "10_dispatch_32": {"block_length": 32},
}


def main():
    """Compile, warm up, time three blocks, and record per-copy and aggregate speeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", choices=IDEAS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.training.runner import Runner
    from exploration_platform.training.blocks import iteration_block

    # Warm-up and coverage match the requested scientific run; all seeds stay fixed.
    changes = dict(IDEAS[args.idea])
    length = changes.pop("block_length", 1)
    cfg = replace(PPOConfig(n_copies=512, update_style="full_batch",
                            track_coverage=True), **changes)
    start = time.perf_counter()
    runner = Runner(cfg)
    state = runner.prime(runner.init_state(0))
    rate = jnp.asarray(1e-3, jnp.float32)
    run = (iteration_block(runner, length) if length > 1 else runner.iterate)
    rates = jnp.full((length,), rate) if length > 1 else rate
    state, metrics = run(state, rates)
    jax.block_until_ready(state)
    compile_seconds = time.perf_counter() - start

    # Three equal-duration measurements avoid counting compilation as steady training.
    times = []
    for _ in range(3):
        count = 0
        begin = time.perf_counter()
        while time.perf_counter() - begin < 8:
            for _ in range(max(1, 32 // length)):
                state, metrics = run(state, rates)
                count += length
            jax.block_until_ready(state)
        times.append((time.perf_counter() - begin) / count)
    seconds = sorted(times)[1]
    stats = jax.devices()[0].memory_stats()
    result = {"idea": args.idea, "changes": changes, "block_length": length,
              "copies": cfg.n_copies, "build_prime_compile_seconds": compile_seconds,
              "seconds_per_iteration": seconds, "timing_repeats": times,
              "steps_per_second_per_copy": 512 / seconds,
              "steps_per_second_total": 512 * cfg.n_copies / seconds,
              "seconds_for_100_million_per_copy": 1e8 * seconds / 512,
              "peak_device_memory_mb": stats["peak_bytes_in_use"] / 2**20,
              "device": jax.devices()[0].device_kind, "jax": jax.__version__,
              "wall_seconds": time.perf_counter() - start}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
