"""Measure one jaguar03 speed idea (2026-10-19): time a 128-copy-per-setting RND program and save
the parameters after its first ten iterations, so the idea's arithmetic can be compared with the
baseline's.

The serval05 study (`bench_speed_ideas.py`) varied the program itself (unroll, flat parameters,
dispatch grouping, copy count). These ideas vary how the work is laid out on an RTX A4500 card and
how XLA runs it: several settings in one program, several programs on one card, XLA flags, a
compilation cache, the processor count, and the cost of the study's own measurements. Ideas that
act outside the program (environment variables, processes per card, processors) are applied by the
driver that starts this script; this script only builds the program it is told to build.

Usage: bench_jaguar03_ideas.py --settings 1 --output result.json [--diagnostics]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# the intrinsic weights of the packed settings, all at learning rate 1e-3; setting 0 is the
# baseline's own (1e-3, 100), so its copies can be compared with the one-setting program
WEIGHTS = (100.0, 1.0, 10.0, 1000.0, 3.0, 30.0, 300.0, 3000.0)


def main():
    """Build, run ten checked iterations, then time three eight-second blocks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=int, default=1, help="settings packed in the program")
    parser.add_argument("--copies-per-setting", type=int, default=128)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.training.runner import Runner

    # the program: k settings of 128 copies each, coverage on as in every study run
    settings = tuple((1e-3, w) for w in WEIGHTS[:args.settings])
    cfg = PPOConfig(n_copies=args.settings * args.copies_per_setting, update_style="full_batch",
                    parameter_settings=settings, copies_per_group=args.copies_per_setting,
                    track_coverage=True, diagnostics=args.diagnostics)
    start = time.perf_counter()
    runner = Runner(cfg, bonus="rnd_next_state")
    state = runner.prime(runner.init_state(0))
    jax.block_until_ready(state.obs)
    build_seconds = time.perf_counter() - start

    # ten iterations from the start; the first pays for compiling. Setting 0's parameters after
    # them are saved for the agreement check against the baseline.
    rate = jnp.asarray(1.0, jnp.float32)
    first = time.perf_counter()
    state, metrics = runner.iterate(state, rate)
    jax.block_until_ready(metrics["loss"])
    compile_seconds = time.perf_counter() - first
    for _ in range(9):
        state, metrics = runner.iterate(state, rate)
    jax.block_until_ready(state.agent_params)
    n0 = args.copies_per_setting
    leaves = jax.tree_util.tree_flatten_with_path(
        {"agent": state.agent_params, "bonus": state.bonus_params})[0]
    np.savez(args.output.with_suffix(".npz"),
             **{jax.tree_util.keystr(path): np.asarray(leaf)[:n0] for path, leaf in leaves})

    # three equal-duration timing blocks; the middle one is the result
    times = []
    for _ in range(3):
        count = 0
        begin = time.perf_counter()
        while time.perf_counter() - begin < 8:
            for _ in range(32):
                state, metrics = runner.iterate(state, rate)
            count += 32
            jax.block_until_ready(state.agent_params)
        times.append((time.perf_counter() - begin) / count)
    seconds = sorted(times)[1]
    steps_per_copy = cfg.num_steps * cfg.n_envs
    stats = jax.devices()[0].memory_stats()
    result = {"settings": args.settings, "copies": cfg.n_copies,
              "diagnostics": args.diagnostics,
              "build_seconds": build_seconds, "compile_seconds": compile_seconds,
              "seconds_per_iteration": seconds, "timing_repeats": times,
              "steps_per_second_per_copy": steps_per_copy / seconds,
              "steps_per_second_total": steps_per_copy * cfg.n_copies / seconds,
              "hours_per_million_steps_per_copy": 1e6 / (3600 * steps_per_copy / seconds),
              "peak_device_memory_mb": stats["peak_bytes_in_use"] / 2 ** 20,
              "device": jax.devices()[0].device_kind, "jax": jax.__version__,
              "wall_seconds": time.perf_counter() - start}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
