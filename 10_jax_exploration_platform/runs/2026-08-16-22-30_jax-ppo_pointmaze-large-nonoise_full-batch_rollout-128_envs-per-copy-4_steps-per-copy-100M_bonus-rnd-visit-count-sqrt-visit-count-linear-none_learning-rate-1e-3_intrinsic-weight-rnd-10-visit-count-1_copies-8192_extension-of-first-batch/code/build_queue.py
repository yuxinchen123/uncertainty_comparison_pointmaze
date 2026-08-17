"""Write this extension's four work units into `queue/pending/`, and the run's launch-time files.

One run is one sweep. This run is an EXTENSION of train run 1.1
(`runs/2026-08-16-20-39_..._first-batch`): it takes the single best configuration each of that
run's four algorithm arms found and re-runs it at 8,192 copies for about 100 million environment
steps per copy — 32 times the copies of one cell and 10 times the step budget. Each arm is one work
unit holding one (learning rate, intrinsic weight) cell, because each arm is a different compiled
program and their memory and their speed differ by a factor of three.

The launch-time files are written here rather than by `scripts/run_training.py`, which writes its
own only when `manifest.yaml` is missing. Writing them here means the run's manifest can name the
parent run and the configurations being extended, which the generic writer cannot know.

Run once, before any submission:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/build_queue.py
"""
import json
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
PLATFORM_ROOT = RUN_DIR.parent.parent
REPO_ROOT = PLATFORM_ROOT.parent
sys.path.insert(0, str(PLATFORM_ROOT / "src"))
sys.path.insert(0, str(PLATFORM_ROOT / "scripts"))

# the run this one extends, and where its winning configurations were read from
PARENT_RUN = ("2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128"
              "_envs-per-copy-4_steps-per-copy-10M"
              "_bonus-rnd-visit-count-sqrt-visit-count-linear-none"
              "_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-256"
              "_first-batch")

# the sweep's fixed knobs, identical for every arm. Every arm's winning learning rate in the parent
# run was 1e-3, so the extension carries one rate and no rate sweep.
LEARNING_RATE = "1e-3"
COPIES = 8192
ROLLOUT_STEPS = 128
ENVS_PER_COPY = 4
EPISODE_STEPS = 400
# 977 windows of 200 iterations. 200 is 8 whole turns of the 25-iteration episode clock
# (400 / gcd(128, 400)), so every window covers each part of the episode the same number of times
# and its reward sum carries no episode-clock artifact. 10^8 / 512 = 195,312.5 is not a whole
# iteration, let alone a whole window; 977 windows is 195,400 iterations and 100,044,800 steps per
# copy, which is 10^8 to within 0.045 per cent — nearer than the 976 windows below it (0.058 per
# cent short). Recorded as a deliberate deviation in experiment_background.md.
ITERATIONS = 195400
WINDOW_ITERATIONS = 200

# one unit per algorithm arm, at the configuration that arm won with in the parent run. The
# seconds per iteration are the parent's canary measurements at 8,448 copies, scaled to this run's
# 8,192; `none` was measured only at 768 copies, where per-iteration overhead dominates, so its
# entry carries the upper bound instead — the cost of the visit-count arms, which do strictly more
# work than an arm with no bonus at all. The canary phase measures the real figure.
UNITS = [
    {"order": 1, "bonus": "rnd_next_state", "intrinsic_weight": "10",
     "parent_seconds_per_iteration_h100": 0.0870, "parent_copies": 8448,
     "parent_peak_device_memory_gib": 28.2, "rate_is_upper_bound": False},
    {"order": 2, "bonus": "gt_position_velocity_sqrt", "intrinsic_weight": "1",
     "parent_seconds_per_iteration_h100": 0.0358, "parent_copies": 8448,
     "parent_peak_device_memory_gib": 7.8, "rate_is_upper_bound": False},
    {"order": 3, "bonus": "gt_position_velocity_linear", "intrinsic_weight": "1",
     "parent_seconds_per_iteration_h100": 0.0350, "parent_copies": 8448,
     "parent_peak_device_memory_gib": 7.8, "rate_is_upper_bound": False},
    {"order": 4, "bonus": "none", "intrinsic_weight": None,
     "parent_seconds_per_iteration_h100": 0.0358, "parent_copies": 8448,
     "parent_peak_device_memory_gib": 7.8, "rate_is_upper_bound": True},
]


def unit_id(unit: dict) -> str:
    """The unit's name, with every load-bearing knob spelled into it.

    before: {"order": 1, "bonus": "rnd_next_state", "intrinsic_weight": "10"}
    after:  "unit-1_rnd-next-state_learning-rate-1e-3_intrinsic-weight-10_copies-8192"
    """
    weight_part = ("no-intrinsic-weight" if unit["intrinsic_weight"] is None
                   else f"intrinsic-weight-{unit['intrinsic_weight']}")
    return (f"unit-{unit['order']}_{unit['bonus'].replace('_', '-')}"
            f"_learning-rate-{LEARNING_RATE}_{weight_part}_copies-{COPIES}")


def seconds_per_iteration(unit: dict) -> float:
    """This unit's estimated seconds per iteration on an H100, scaled to this run's copy count.

    before: 0.0870 s at 8,448 copies ; after: 0.0870 x 8192 / 8448 = 0.0844 s at 8,192 copies
    """
    return round(unit["parent_seconds_per_iteration_h100"] * COPIES / unit["parent_copies"], 5)


def peak_device_memory_gib(unit: dict) -> float:
    """The card memory this unit is expected to touch, scaled to this run's copy count."""
    return round(unit["parent_peak_device_memory_gib"] * COPIES / unit["parent_copies"], 1)


def unit_record(unit: dict) -> dict:
    """One queue entry: the identity of the unit and the exact arguments that run it."""
    arguments = ["--unit-id", unit_id(unit), "--bonus", unit["bonus"],
                 "--learning-rates", LEARNING_RATE]
    # the arm with no bonus has no weight, so its single cell is the learning rate alone
    if unit["intrinsic_weight"] is not None:
        arguments += ["--intrinsic-weights", unit["intrinsic_weight"]]
    arguments += ["--copies-per-cell", str(COPIES),
                  "--rollout-steps", str(ROLLOUT_STEPS),
                  "--envs-per-copy", str(ENVS_PER_COPY),
                  "--update-style", "full_batch",
                  "--iterations", str(ITERATIONS),
                  "--window-iterations", str(WINDOW_ITERATIONS),
                  "--episode-steps", str(EPISODE_STEPS),
                  "--base-seed", "0", "--run-seed", "0",
                  "--track-coverage"]
    return {
        "unit_id": unit_id(unit),
        "order": unit["order"],
        "bonus": unit["bonus"],
        "extends_parent_run": PARENT_RUN,
        "cells": 1,
        "copies": COPIES,
        "copies_per_cell": COPIES,
        "learning_rate": float(LEARNING_RATE),
        "intrinsic_weight": (None if unit["intrinsic_weight"] is None
                             else float(unit["intrinsic_weight"])),
        "iterations": ITERATIONS,
        "window_iterations": WINDOW_ITERATIONS,
        "windows": ITERATIONS // WINDOW_ITERATIONS,
        "env_steps_per_copy": ITERATIONS * ROLLOUT_STEPS * ENVS_PER_COPY,
        "estimated_seconds_per_iteration_h100": seconds_per_iteration(unit),
        "estimated_seconds_per_iteration_is_upper_bound": unit["rate_is_upper_bound"],
        "estimated_seconds_h100": round(ITERATIONS * seconds_per_iteration(unit), 1),
        "peak_device_memory_gib": peak_device_memory_gib(unit),
        # the card must hold the expected peak with the submission skill's 25% room on top
        "required_device_memory_gib": round(1.25 * peak_device_memory_gib(unit), 1),
        "arguments": arguments,
    }


def resolved_trainer_configs(records: list) -> dict:
    """Every unit's fully resolved trainer configuration, keyed by unit id.

    The four units share every knob but the bonus and the intrinsic weight, and this resolves them
    through the same `sweep_config` the runner calls, so the file records what will actually run
    rather than what was typed.
    """
    from exploration_platform.training.sweep import sweep_config
    resolved = {}
    for record in records:
        weights = () if record["intrinsic_weight"] is None else (record["intrinsic_weight"],)
        config = sweep_config(learning_rates=(record["learning_rate"],), betas=weights,
                              copies_per_group=COPIES, style="full_batch",
                              n_envs=ENVS_PER_COPY, num_steps=ROLLOUT_STEPS, base_seed=0,
                              track_coverage=True)
        resolved[record["unit_id"]] = {field: getattr(config, field)
                                       for field in sorted(config.__dataclass_fields__)}
    return resolved


def as_yaml_text(value, indent: int = 0) -> str:
    """Write nested dictionaries, lists and scalars as YAML text; the platform's own writer."""
    from run_training import as_yaml
    return as_yaml(value, indent)


def write_launch_files(records: list, commit: str) -> None:
    """Write command.txt, manifest.yaml and config_resolved.yaml before the first submission."""
    # the launch command of the run as a whole, so the sweep can be rebuilt without reading prose
    (RUN_DIR / "command.txt").write_text(
        "# the sweep as a whole: build the queue, then submit one job per unit\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/build_queue.py\n"
        f"# then, per unit, one independent job (see code/submit_one.sh):\n"
        f"#   RUN_DIR={RUN_DIR} bash code/submit_one.sh real <unit_id> <node> 4-00:00:00\n"
        f"# code commit at launch: {commit}\n"
        "# one unit runs as:\n"
        + "".join(
            "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
            f"{PLATFORM_ROOT}/scripts/run_training.py --run-dir {RUN_DIR} "
            + " ".join(record["arguments"]) + "\n" for record in records))

    manifest = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "kind": "extension",
        "extends": {
            "parent_run": PARENT_RUN,
            "parent_run_path": f"runs/{PARENT_RUN}",
            "what_changed": ("each arm's best configuration of the parent run, re-run at 8,192 "
                             "copies for 100,044,800 environment steps per copy: 32 times the "
                             "copies of one parent cell and 10 times its step budget"),
            # keyed by arm rather than a list of dictionaries: the platform's YAML writer renders a
            # list item as one scalar, so a list of dictionaries would come out as Python reprs
            "configurations_extended": {
                record["bonus"]: {
                    "learning_rate": record["learning_rate"],
                    "intrinsic_weight": ("none — this arm has no bonus"
                                         if record["intrinsic_weight"] is None
                                         else record["intrinsic_weight"]),
                    "parent_copies": 256, "extension_copies": record["copies"],
                    "parent_env_steps_per_copy": 10035200,
                    "extension_env_steps_per_copy": record["env_steps_per_copy"]}
                for record in records},
        },
        "description": ("The extension of train run 1.1. Four work units, one per algorithm arm, "
                        "each holding the single configuration that arm won with in the parent "
                        "run: random network distillation at intrinsic weight 10, the two oracle "
                        "visit-count bonuses at intrinsic weight 1, and no bonus at all, every one "
                        "at learning rate 1e-3. 8,192 copies per unit, seeded 0 to 8,191, "
                        "100,044,800 environment steps per copy."),
        "git": {"commit": commit},
        "specs": {
            "env": "pointmaze_large_cont400_nonoise@1",
            "agent": "ppo_full_batch@1",
            "bonus": ("rnd_next_state@1 + gt_position_velocity_sqrt@1 + "
                      "gt_position_velocity_linear@1 + none@1"),
            "update_schedule": ("post_rollout_update@1 for the visit-count arms, per_rollout@1 for "
                                "random network distillation, none@1 for the arm with no bonus"),
        },
        "compile_signature": {
            "copies": COPIES, "envs_per_copy": ENVS_PER_COPY, "rollout_steps": ROLLOUT_STEPS,
            "update_style": "full_batch",
            "epochs": "not applicable — full_batch takes one gradient step on the whole rollout",
            "minibatches": "not applicable — full_batch takes one gradient step on the whole "
                           "rollout",
            "dtype": "float32", "stats_dtype": "float64",
            "iterations": ITERATIONS, "window_iterations": WINDOW_ITERATIONS,
            "windows": ITERATIONS // WINDOW_ITERATIONS,
            "env_steps_per_copy": ITERATIONS * ROLLOUT_STEPS * ENVS_PER_COPY,
        },
        "seeding": {"base_seed": 0, "run_seed": 0, "mode": "paired",
                    "note": "one cell per unit, so the 8,192 copies of a unit differ only by seed"},
        "units": [record["unit_id"] for record in records],
        "artifacts": {"metrics": "metrics.jsonl", "summary": "summary.json",
                      "shards": "data/*.jsonl", "aggregator": "code/aggregate.py"},
        "checkpoints": "none — this platform saves no model state, so a unit resumes by being "
                       "re-run whole",
    }
    (RUN_DIR / "manifest.yaml").write_text(as_yaml_text(manifest) + "\n")

    resolved = {"units": resolved_trainer_configs(records),
                "environment": {"specification": "pointmaze_large_cont400_nonoise@1",
                                "map_name": "large", "start_cell": "(7, 1)", "goal_cell": "(1, 10)",
                                "position_noise": 0.0, "max_episode_steps": EPISODE_STEPS,
                                "continuing_task": True, "goal_radius": 0.45, "reward_shift": 0.0},
                "runtime": {"interpreter": "/p/rlprojects/RND/.venvs/platform_jax/bin/python",
                            "iterations": ITERATIONS, "window_iterations": WINDOW_ITERATIONS,
                            "run_seed": 0}}
    (RUN_DIR / "config_resolved.yaml").write_text(as_yaml_text(resolved) + "\n")


def main() -> None:
    """Write the four unit files into queue/pending/ and the run's launch-time files."""
    pending = RUN_DIR / "queue" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    records = [unit_record(unit) for unit in UNITS]
    for record in records:
        (pending / f"{record['unit_id']}.json").write_text(json.dumps(record, indent=2) + "\n")

    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    write_launch_files(records, commit)

    total = sum(record["estimated_seconds_h100"] for record in records)
    print(f"{len(records)} units written to {pending}")
    for record in records:
        bound = " (upper bound)" if record["estimated_seconds_per_iteration_is_upper_bound"] else ""
        print(f"  {record['unit_id']}: {record['copies']} copies, "
              f"{record['estimated_seconds_h100'] / 3600:.2f} h on an H100{bound}, "
              f"needs {record['required_device_memory_gib']} GiB")
    print(f"whole sweep on one H100, one unit after another: {total / 3600:.2f} h of compute")


if __name__ == "__main__":
    main()
