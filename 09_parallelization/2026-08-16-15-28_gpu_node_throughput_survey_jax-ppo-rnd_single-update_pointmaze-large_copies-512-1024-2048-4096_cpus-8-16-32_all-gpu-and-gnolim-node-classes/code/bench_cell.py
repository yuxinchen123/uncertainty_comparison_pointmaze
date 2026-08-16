"""One measurement cell: the throughput of the JAX PPO+RND trainer at ONE copy count, on
whatever graphics card this process was given.

Run as its own process (one per copy count) so a card that runs out of memory at 4,096 copies
loses only that cell — the parent, `run_job.py`, records the failure and keeps the smaller
counts it already measured.

The measured quantity is one full training iteration of `JaxPPORND._iterate`: the 128-step
rollout, the running statistics, the two advantage streams, and ONE update over the whole batch
(the single-update style the trainer calls `full_batch`). Compilation and the first iterations
are excluded; the loop that is timed is the loop the real trainer runs.
"""
import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]          # 09_parallelization/
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))

import numpy as np                                   # noqa: E402
import jax                                           # noqa: E402
from jax_ppo_rnd import JaxPPORND, PPOConfig         # noqa: E402

# how a round of timing is sized and when the measurement is called stable
TARGET_ROUND_SECONDS = 2.0    # iterations per round are chosen to fill roughly this much time
WARMUP_ITERATIONS = 5         # run and discard: compilation, autotuning, clock ramp-up
MIN_ROUNDS = 6                # never conclude from fewer rounds than this
MAX_ROUNDS = 24               # nor spend more than this many, however unsettled it looks
STABLE_SPREAD = 0.02          # stop early once the middle half of the rounds sits within 2%


def spread_of(round_seconds):
    """Relative width of the middle half of the per-iteration times — the stability measure.

    before: round_seconds = [0.402, 0.399, 0.405, 0.401, 0.400, 0.398]
    after:  quartiles 0.3993 and 0.4023, median 0.4005 -> 0.0075, i.e. the middle half of the
            rounds sits within 0.75% of the median, so the measurement is called settled.
    """
    q1, q3 = np.percentile(round_seconds, [25, 75])
    return float((q3 - q1) / np.median(round_seconds))


def peak_device_memory_mb():
    """Peak bytes the card's allocator has had in use, in MB — None if it does not report."""
    stats = jax.local_devices()[0].memory_stats()
    return None if stats is None else stats.get("peak_bytes_in_use", 0) / 1e6


def measure(n_copies, deadline, style):
    """Time one full training iteration at `n_copies` copies until the rate settles.

    Returns the per-iteration seconds of every round plus the setup times, so the report can
    show both the number and how firm it is.
    """
    # build the trainer and its starting state; the copy count is the only knob that moves
    cfg = PPOConfig(n_copies=n_copies, update_style=style)
    t_build = time.perf_counter()
    trainer = JaxPPORND(cfg)
    state = trainer.init_state()
    key = jax.random.PRNGKey(0)
    build_seconds = time.perf_counter() - t_build

    # first iteration: compilation of the whole program, which is not part of the rate
    t_compile = time.perf_counter()
    state, metrics = trainer._iterate(state, jax.random.fold_in(key, 1),
                                      trainer.lr_argument(1, 10 ** 6))
    jax.block_until_ready(state.params)
    compile_seconds = time.perf_counter() - t_compile

    # warm-up iterations, discarded: allocator growth, autotuning, the card reaching its clock
    t_warm = time.perf_counter()
    for it in range(2, 2 + WARMUP_ITERATIONS):
        state, metrics = trainer._iterate(state, jax.random.fold_in(key, it),
                                          trainer.lr_argument(it, 10 ** 6))
    jax.block_until_ready(state.params)
    warm_seconds = time.perf_counter() - t_warm
    per_iteration_estimate = warm_seconds / WARMUP_ITERATIONS

    # a round holds enough iterations to fill about two seconds, so one round's timing is not
    # dominated by the single synchronisation at its end
    # before: per-iteration estimate 0.04 s -> after: 50 iterations per round
    iters_per_round = max(1, min(200, round(TARGET_ROUND_SECONDS / per_iteration_estimate)))

    # rounds until the middle half of them agrees to within STABLE_SPREAD, at least MIN_ROUNDS
    round_seconds, it = [], 2 + WARMUP_ITERATIONS
    while len(round_seconds) < MAX_ROUNDS:
        t0 = time.perf_counter()
        for _ in range(iters_per_round):
            state, metrics = trainer._iterate(state, jax.random.fold_in(key, it),
                                              trainer.lr_argument(it, 10 ** 6))
            it += 1
        jax.block_until_ready(state.params)
        round_seconds.append((time.perf_counter() - t0) / iters_per_round)
        if len(round_seconds) >= MIN_ROUNDS and spread_of(round_seconds) <= STABLE_SPREAD:
            break
        if time.perf_counter() > deadline and len(round_seconds) >= MIN_ROUNDS:
            break

    return {
        "seconds_per_iteration_round": round_seconds,
        "iterations_per_round": iters_per_round,
        "rounds": len(round_seconds),
        "seconds_per_iteration": float(statistics.median(round_seconds)),
        "seconds_per_iteration_min": float(min(round_seconds)),
        "seconds_per_iteration_max": float(max(round_seconds)),
        "relative_spread_middle_half": spread_of(round_seconds),
        "settled": spread_of(round_seconds) <= STABLE_SPREAD,
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "warmup_seconds": warm_seconds,
        "iterations_timed": len(round_seconds) * iters_per_round,
        "peak_device_memory_mb": peak_device_memory_mb(),
        "final_reward_per_copy_mean": float(np.asarray(metrics["reward_ext_sum"]).mean()),
    }


def main():
    """Measure one copy count and write its own result file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, required=True)
    ap.add_argument("--style", default="full_batch",
                    help="full_batch = the single-update style; epoch_minibatch = the other one")
    ap.add_argument("--budget-seconds", type=float, default=300.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = jax.local_devices()[0]
    assert device.platform == "gpu", f"this cell must run on a graphics card, got {device}"
    result = measure(args.n_copies, time.perf_counter() + args.budget_seconds, args.style)

    # the throughput quantities both rates the throughput rule asks for, from the one timing
    cfg = PPOConfig(n_copies=args.n_copies)
    steps_per_iteration_per_copy = cfg.num_steps * cfg.n_envs
    s = result["seconds_per_iteration"]
    result.update({
        "status": "measured",
        "n_copies": args.n_copies,
        "n_envs": cfg.n_envs,
        "num_steps": cfg.num_steps,
        "update_style": args.style,
        "environment_steps_per_iteration": steps_per_iteration_per_copy * args.n_copies,
        "total_steps_per_second": steps_per_iteration_per_copy * args.n_copies / s,
        "steps_per_second_per_copy": steps_per_iteration_per_copy / s,
        "hours_per_million_steps_per_copy": 1e6 / (3600 * steps_per_iteration_per_copy / s),
        "device_kind": device.device_kind,
        "device_compute_capability": getattr(device, "compute_capability", None),
        "hostname": socket.gethostname(),
        "visible_cpus": len(os.sched_getaffinity(0)),
        "jax_version": jax.__version__,
    })
    args.out.write_text(json.dumps(result, indent=1))
    print(f"{args.n_copies} copies: {s * 1000:.1f} ms per iteration, "
          f"{result['total_steps_per_second'] / 1e6:.2f} million steps per second, "
          f"spread {result['relative_spread_middle_half'] * 100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
