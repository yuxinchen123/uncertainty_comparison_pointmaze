"""Is PPO with no bonus actually faster than PPO with random network distillation?

The compiled program for the "none" arm provably contains none of the bonus's arithmetic
(`tests/bonuses/test_none_has_no_bonus_arithmetic.py`). This asks the other half of the question:
does removing it show up in the time an iteration takes.

Paired: the two arms alternate round by round in one process on one card, so a passing cloud, a
clock change or another job starting mid-measurement hits both arms alike. Timing is sync — every
iteration is waited for — so the number is the device's, not the host's queue depth.

Run on the graphics card, through the serval05 lock (one graphics-card command at a time):
  bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh \
    "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python <this file>"
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
PACIFIC = ZoneInfo("America/Los_Angeles")


def prepared(bonus: str, cfg: PPOConfig):
    """A runner and a primed state, with the program already compiled and warmed up."""
    runner = Runner(cfg, bonus=bonus)
    state = runner.prime(runner.init_state(run_seed=1))
    lr = jnp.asarray(cfg.learning_rate, jnp.float32)
    for _ in range(3):
        state, metrics = runner.iterate(state, lr)
    jax.block_until_ready(metrics["loss"])
    return runner, state, lr


def one_round(runner, state, lr, iterations: int):
    """Seconds per iteration over one block, waiting for the device on every iteration."""
    start = time.perf_counter()
    for _ in range(iterations):
        state, metrics = runner.iterate(state, lr)
        jax.block_until_ready(metrics["loss"])
    return (time.perf_counter() - start) / iterations, state


def throughput_row(name: str, seconds: float, cfg: PPOConfig) -> dict:
    """One row of the throughput table: the aggregate rate AND the rate one copy gets."""
    env_steps = cfg.num_steps * cfg.n_copies * cfg.n_envs
    total = env_steps / seconds
    per_copy = total / cfg.n_copies
    return {"bonus": name, "copies": cfg.n_copies,
            "seconds_per_iteration": seconds,
            "total_env_steps_per_second": total,
            "env_steps_per_second_per_copy": per_copy,
            "hours_per_million_steps_per_copy": 1e6 / (3600 * per_copy)}


def main():
    """Alternate the two arms round by round, then print and store the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--iterations-per-round", type=int, default=10)
    args = parser.parse_args()

    cfg = PPOConfig(n_copies=args.copies, n_envs=4, num_steps=128, update_style="full_batch",
                    prime_iterations=10)
    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"{cfg.n_copies} copies x {cfg.n_envs} environments x {cfg.num_steps} steps, "
          f"update style {cfg.update_style}, {args.rounds} paired rounds of "
          f"{args.iterations_per_round} iterations")

    arms = {name: prepared(name, cfg) for name in ("rnd_next_state", "none")}
    rounds = {name: [] for name in arms}
    wins = 0
    for r in range(args.rounds):
        # alternate within the round too, so a slow first-of-round never lands on the same arm
        order = ("rnd_next_state", "none") if r % 2 == 0 else ("none", "rnd_next_state")
        for name in order:
            runner, state, lr = arms[name]
            seconds, state = one_round(runner, state, lr, args.iterations_per_round)
            arms[name] = (runner, state, lr)
            rounds[name].append(seconds)
        wins += int(rounds["none"][-1] < rounds["rnd_next_state"][-1])
        print(f"  round {r + 1:2d}: distillation {rounds['rnd_next_state'][-1] * 1e3:7.2f} ms, "
              f"none {rounds['none'][-1] * 1e3:7.2f} ms")

    medians = {name: float(np.median(times)) for name, times in rounds.items()}
    table = [throughput_row(name, medians[name], cfg) for name in ("rnd_next_state", "none")]
    speedup = medians["rnd_next_state"] / medians["none"]

    print("\n| bonus | copies | seconds per iteration | total env steps per second | "
          "env steps per second per copy | hours per million steps per copy |")
    print("|---|---|---|---|---|---|")
    for row in table:
        print(f"| {row['bonus']} | {row['copies']} | {row['seconds_per_iteration']:.4f} | "
              f"{row['total_env_steps_per_second'] / 1e6:.2f} M | "
              f"{row['env_steps_per_second_per_copy'] / 1e3:.1f} k | "
              f"{row['hours_per_million_steps_per_copy']:.4f} |")
    print(f"\nno bonus is {speedup:.3f}x the throughput of random network distillation, and won "
          f"{wins} of {args.rounds} paired rounds")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d-%H-%M")
    out = RESULTS / f"{stamp}_none_vs_rnd_copies-{cfg.n_copies}.json"
    out.write_text(json.dumps({
        "measured_at": datetime.now().astimezone().isoformat(),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "copies": cfg.n_copies, "envs_per_copy": cfg.n_envs, "rollout_steps": cfg.num_steps,
        "update_style": cfg.update_style, "rounds": args.rounds,
        "iterations_per_round": args.iterations_per_round,
        "seconds_per_iteration_per_round": rounds, "medians": medians,
        "throughput": table, "speedup_none_over_rnd": speedup,
        "paired_rounds_won_by_none": wins}, indent=1))
    print(f"wrote {out}")
    assert wins > args.rounds // 2, "no bonus did not win most of the paired rounds"


if __name__ == "__main__":
    main()
