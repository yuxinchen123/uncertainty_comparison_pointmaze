"""Does a bonus reach random network distillation's training throughput at the same copy count?

That is the gate every new bonus family has to pass before it joins a science run: the total
environment steps per second of PPO with the new bonus must be at least PPO with random network
distillation's, at the same number of copies, on the same card, measured in the same session.

How it is measured, and why each part is there:

- **The shipped shape.** The copy count comes from the sweep the run will actually launch — the
  cross product of learning rates and intrinsic weights, so many copies per cell — rather than a
  round number, because the shape is what the compiler sees.
- **Paired, round by round.** The arms alternate inside one process on one card, so a passing
  cloud, another job starting, or a clock step hits both arms alike. The order flips every round,
  so a slow first-of-round never lands on the same arm twice running.
- **Sync timing.** Every iteration is waited for, so the number is the device's and not the depth
  of the host's queue.
- **The sign test.** Alongside the medians, how many of the paired rounds each arm won. A
  difference under a couple of percent is only believable if it also wins nearly every round.
- **Proof it ran.** After the rounds, the arm's own state is read back and checked: a visit-count
  arm must hold counts that grew with the iterations and bonuses inside (0, 1].

Peak device memory is a process-wide high-water mark, so it cannot be split between two arms
sharing a process. `--solo` runs one arm alone for that column.

Run on the graphics card, through the serval05 lock (one graphics-card command at a time):
  bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh \
    "PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
     /p/rlprojects/RND/.venvs/platform_jax/bin/python <this file> --arms rnd_next_state none"
"""
import argparse
import json
import subprocess
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
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.sweep import sweep_config  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
# the first batch's sweep: three learning rates, and the intrinsic weight over ten decades
LEARNING_RATES = (1e-3, 1e-4, 1e-5)
BETAS = tuple(10.0 ** e for e in range(-5, 6))


def build_config(copies_per_group: int, sweep_betas: bool, **overrides):
    """The configuration of one arm, in the shape the science run will launch.

    before: copies_per_group 256, sweep_betas True
    after:  3 learning rates x 11 intrinsic weights x 256 copies = 8,448 copies, one update per
            batch, 128 rollout steps of 4 environments per copy
    """
    return sweep_config(learning_rates=LEARNING_RATES,
                        betas=BETAS if sweep_betas else (),
                        copies_per_group=copies_per_group, style="full_batch",
                        num_steps=128, n_envs=4, prime_iterations=10, **overrides)


def peak_device_mib():
    """Peak device memory this process's allocator has held, in mebibytes."""
    stats = jax.local_devices()[0].memory_stats()
    if stats is None or "peak_bytes_in_use" not in stats:
        raise RuntimeError("this jax build reports no device memory statistics, so the peak "
                           "memory column of the gate table cannot be filled")
    return stats["peak_bytes_in_use"] / 2 ** 20


def prepared(bonus: str, cfg):
    """A runner and a warmed-up state: compiled, primed, and three iterations past the start."""
    t0 = time.perf_counter()
    runner = Runner(cfg, bonus=bonus)
    state = runner.prime(runner.init_state(run_seed=1))
    lr = runner.lr_argument(1, 1)
    for _ in range(3):
        state, metrics = runner.iterate(state, lr)
    jax.block_until_ready(metrics["loss"])
    return {"bonus": bonus, "runner": runner, "state": state, "lr": lr,
            "compile_and_warmup_seconds": time.perf_counter() - t0, "iterations_run": 3}


def one_round(arm, iterations: int):
    """Seconds per iteration over one block, waiting for the device on every iteration."""
    runner, state, lr = arm["runner"], arm["state"], arm["lr"]
    start = time.perf_counter()
    for _ in range(iterations):
        state, metrics = runner.iterate(state, lr)
        jax.block_until_ready(metrics["loss"])
    seconds = (time.perf_counter() - start) / iterations
    arm["state"] = state
    arm["iterations_run"] += iterations
    return seconds


def evidence(arm, cfg):
    """Read the arm's own state back and check the bonus under test actually did its work.

    A compiler is free to delete arithmetic whose result nothing consumes, and this project has
    twice measured a variant that had been optimised away. So the visit-count table is read and
    compared against how many rows were counted, and the intrinsic reward is checked to lie where
    the bonus's definition puts it.
    """
    state = arm["state"]
    rows_counted = arm["iterations_run"] * cfg.num_steps * cfg.n_envs
    out = {"iterations_run": arm["iterations_run"], "rows_per_copy": rows_counted}
    if "counts" in state.bonus_state:
        counts = np.asarray(state.bonus_state["counts"])
        per_copy = counts.sum(axis=1)
        out["count_table_total_per_copy_min"] = int(per_copy.min())
        out["count_table_total_per_copy_max"] = int(per_copy.max())
        out["count_table_nonzero_entries_per_copy_mean"] = float((counts > 0).sum(axis=1).mean())
        # every open-cell row of every rollout added exactly one, so the table's total can only
        # fall short of the rows counted by the rows that landed in a wall cell
        if not (0 < per_copy.max() <= rows_counted):
            raise RuntimeError(
                f"the visit-count table holds {per_copy.max()} counts per copy after "
                f"{rows_counted} counted rows, which cannot happen if the scatter ran once per "
                "iteration on the rollout it was given")
    _, metrics = arm["runner"].iterate(state, arm["lr"])
    rint = np.asarray(metrics["rint_mean"])
    out["intrinsic_reward_mean_over_copies"] = float(rint.mean())
    out["intrinsic_reward_min"], out["intrinsic_reward_max"] = float(rint.min()), float(rint.max())
    return out


def throughput_row(name: str, seconds: float, cfg) -> dict:
    """One row of the throughput table: the aggregate rate AND the rate one copy gets."""
    env_steps = cfg.num_steps * cfg.n_copies * cfg.n_envs
    total = env_steps / seconds
    per_copy = total / cfg.n_copies
    return {"bonus": name, "copies": cfg.n_copies,
            "seconds_per_iteration": seconds,
            "total_env_steps_per_second": total,
            "env_steps_per_second_per_copy": per_copy,
            "hours_per_million_steps_per_copy": 1e6 / (3600 * per_copy)}


def print_table(rows):
    """The throughput table in the project's fixed column set."""
    print("\n| bonus | copies | seconds per iteration | total env steps per second | "
          "env steps per second per copy | hours per million steps per copy |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| {row['bonus']} | {row['copies']} | {row['seconds_per_iteration']:.5f} | "
              f"{row['total_env_steps_per_second'] / 1e6:.2f} M | "
              f"{row['env_steps_per_second_per_copy'] / 1e3:.2f} k | "
              f"{row['hours_per_million_steps_per_copy']:.4f} |")


def main():
    """Run the arms against each other (or one arm alone), print the table, store the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["rnd_next_state", "gt_position_velocity_sqrt"],
                        help="bonus presets to compare; the first is the bar")
    parser.add_argument("--copies-per-group", type=int, default=256)
    parser.add_argument("--no-beta-sweep", action="store_true",
                        help="sweep the learning rate only, as the no-bonus arm's run does")
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--iterations-per-round", type=int, default=10)
    parser.add_argument("--solo", action="store_true",
                        help="one arm alone, so the peak device memory belongs to it")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg = build_config(args.copies_per_group, not args.no_beta_sweep)
    if args.solo and len(args.arms) != 1:
        raise ValueError("--solo measures the memory of ONE arm; pass exactly one --arms entry")

    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"{cfg.n_copies} copies x {cfg.n_envs} environments x {cfg.num_steps} steps, "
          f"update style {cfg.update_style}, learning rates {LEARNING_RATES}, "
          f"intrinsic weights {'swept over ten decades' if cfg.betas else 'fixed'}")
    print(f"{args.rounds} paired rounds of {args.iterations_per_round} iterations, "
          f"arms {args.arms}")

    memory_before = peak_device_mib()
    arms, prepare_order = {}, list(args.arms)
    for name in prepare_order:
        arms[name] = prepared(name, cfg)
        print(f"  {name}: compiled and warmed up in "
              f"{arms[name]['compile_and_warmup_seconds']:.1f} s, "
              f"peak device memory now {peak_device_mib():.0f} MiB")

    rounds = {name: [] for name in arms}
    for r in range(args.rounds):
        # flip the order every round, so a slow first-of-round never lands on the same arm twice
        order = prepare_order if r % 2 == 0 else prepare_order[::-1]
        for name in order:
            rounds[name].append(one_round(arms[name], args.iterations_per_round))
        print(f"  round {r + 1:2d}: " + "  ".join(
            f"{name} {rounds[name][-1] * 1e3:8.2f} ms" for name in prepare_order))

    medians = {name: float(np.median(times)) for name, times in rounds.items()}
    table = [throughput_row(name, medians[name], cfg) for name in prepare_order]
    print_table(table)

    # the sign test: how often each arm beat the bar, round by round. A candidate within a couple
    # of percent of the bar is only believable if it also wins most of the rounds.
    bar = prepare_order[0]
    verdicts = {}
    for name in prepare_order[1:]:
        wins = sum(int(c < b) for c, b in zip(rounds[name], rounds[bar]))
        ratio = medians[bar] / medians[name]
        verdicts[name] = {"throughput_ratio_against_bar": ratio,
                          "paired_rounds_faster_than_bar": wins, "rounds": args.rounds,
                          "passes_gate": ratio >= 1.0}
        print(f"\n{name} runs at {ratio:.4f}x the throughput of {bar} and was faster in "
              f"{wins} of {args.rounds} paired rounds — "
              f"{'PASSES' if ratio >= 1.0 else 'BELOW'} the gate")

    checks = {name: evidence(arms[name], cfg) for name in prepare_order}
    for name, ev in checks.items():
        print(f"  {name} ran: {json.dumps(ev)}")

    stamp = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d-%H-%M")
    out_dir = Path(args.out_dir) if args.out_dir else (
        BASE / "benchmark_runs" / f"{stamp}_bonus-throughput-gate{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = "solo" if args.solo else "paired"
    out = out_dir / (f"{stamp}_{kind}_copies-{cfg.n_copies}_"
                     f"{'_vs_'.join(prepare_order)}.json")
    out.write_text(json.dumps({
        "measured_at": datetime.now().astimezone().isoformat(),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "kind": kind, "arms": prepare_order,
        "copies": cfg.n_copies, "copies_per_group": args.copies_per_group,
        "learning_rates": list(LEARNING_RATES),
        "betas": list(cfg.betas) if cfg.betas else None,
        "envs_per_copy": cfg.n_envs, "rollout_steps": cfg.num_steps,
        "update_style": cfg.update_style, "rounds": args.rounds,
        "iterations_per_round": args.iterations_per_round,
        "seconds_per_iteration_per_round": rounds, "medians": medians,
        "throughput": table, "gate": verdicts, "ran_check": checks,
        "compile_and_warmup_seconds": {n: arms[n]["compile_and_warmup_seconds"] for n in arms},
        "peak_device_mib_before_any_arm": memory_before,
        "peak_device_mib": peak_device_mib()}, indent=1))
    print(f"\npeak device memory {peak_device_mib():.0f} MiB ({kind} process)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
