"""Write this sweep's four work units into `queue/pending/`, and the run's launch-time files.

One run is one sweep. A work unit here is one algorithm arm: one compiled program holding every
(learning rate, intrinsic weight) cell of that arm at once, 256 copies per cell, seeded in pairs
across cells. The arms are separate units because each is a different compiled program — the bonus
is chosen on the host before anything is traced — and because their memory and their speed differ
by a factor of four, so they belong on different cards.

Run once, before any submission:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/build_queue.py
"""
import json
import subprocess
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
PLATFORM_ROOT = RUN_DIR.parent.parent
REPO_ROOT = PLATFORM_ROOT.parent

# the sweep's fixed knobs, identical for every arm
LEARNING_RATES = "1e-3,1e-4,1e-5"
INTRINSIC_WEIGHTS = "1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000"
COPIES_PER_CELL = 256
ROLLOUT_STEPS = 128
ENVS_PER_COPY = 4
EPISODE_STEPS = 400
# 98 windows of 200 iterations. 200 is 8 whole turns of the 25-iteration episode clock
# (400 / gcd(128, 400)), so every window covers each part of the episode the same number of times
# and its reward sum carries no episode-clock artifact. 19,600 x 128 x 4 = 10,035,200 environment
# steps per copy, which is the plan's 10^7 to within 0.35%; the plan's own 19,531 would have left a
# final 131-iteration window that is not a whole number of clock turns and so could not be scored.
ITERATIONS = 19600
WINDOW_ITERATIONS = 200

# one unit per algorithm arm: the bonus, whether it has an intrinsic weight to sweep, and the
# measured seconds per iteration on an H100 from the throughput gate that preceded this run
UNITS = [
    {"order": 1, "bonus": "rnd_next_state", "sweeps_weight": True,
     "measured_seconds_per_iteration_h100": 0.0840, "peak_device_memory_gib": 28.2},
    {"order": 2, "bonus": "gt_position_velocity_sqrt", "sweeps_weight": True,
     "measured_seconds_per_iteration_h100": 0.0345, "peak_device_memory_gib": 7.8},
    {"order": 3, "bonus": "gt_position_velocity_linear", "sweeps_weight": True,
     "measured_seconds_per_iteration_h100": 0.0343, "peak_device_memory_gib": 7.8},
    {"order": 4, "bonus": "none", "sweeps_weight": False,
     "measured_seconds_per_iteration_h100": 0.0070, "peak_device_memory_gib": 0.8},
]


def unit_id(unit: dict) -> str:
    """The unit's name, with every load-bearing knob spelled into it.

    before: {"order": 1, "bonus": "rnd_next_state", "sweeps_weight": True}
    after:  "unit-1_rnd-next-state_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_
             copies-per-cell-256_copies-8448"
    """
    weight_part = ("intrinsic-weight-1e-5-to-1e5" if unit["sweeps_weight"]
                   else "no-intrinsic-weight")
    return (f"unit-{unit['order']}_{unit['bonus'].replace('_', '-')}"
            f"_learning-rate-1e-3-1e-4-1e-5_{weight_part}"
            f"_copies-per-cell-{COPIES_PER_CELL}_copies-{copies_of(unit)}")


def copies_of(unit: dict) -> int:
    """How many training copies this unit runs: cells times copies per cell."""
    cells = 3 * (11 if unit["sweeps_weight"] else 1)
    return cells * COPIES_PER_CELL


def unit_record(unit: dict) -> dict:
    """One queue entry: the identity of the unit and the exact arguments that run it."""
    arguments = ["--unit-id", unit_id(unit), "--bonus", unit["bonus"],
                 "--learning-rates", LEARNING_RATES]
    # the arm with no bonus has no weight to sweep, so its cells are the three learning rates only
    if unit["sweeps_weight"]:
        arguments += ["--intrinsic-weights", INTRINSIC_WEIGHTS]
    arguments += ["--copies-per-cell", str(COPIES_PER_CELL),
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
        "cells": 3 * (11 if unit["sweeps_weight"] else 1),
        "copies": copies_of(unit),
        "copies_per_cell": COPIES_PER_CELL,
        "iterations": ITERATIONS,
        "window_iterations": WINDOW_ITERATIONS,
        "env_steps_per_copy": ITERATIONS * ROLLOUT_STEPS * ENVS_PER_COPY,
        "measured_seconds_per_iteration_h100": unit["measured_seconds_per_iteration_h100"],
        "estimated_seconds_h100": round(
            ITERATIONS * unit["measured_seconds_per_iteration_h100"], 1),
        "peak_device_memory_gib": unit["peak_device_memory_gib"],
        # the card must hold the measured peak with the submission skill's 25% room on top
        "required_device_memory_gib": round(1.25 * unit["peak_device_memory_gib"], 1),
        "arguments": arguments,
    }


def main() -> None:
    """Write the four unit files into queue/pending/ and the run's command.txt."""
    pending = RUN_DIR / "queue" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    records = [unit_record(unit) for unit in UNITS]
    for record in records:
        (pending / f"{record['unit_id']}.json").write_text(json.dumps(record, indent=2) + "\n")

    # the launch command of the run as a whole, so the sweep can be rebuilt without reading prose
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    (RUN_DIR / "command.txt").write_text(
        "# the sweep as a whole: build the queue, then submit one job per unit\n"
        "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
        f"{RUN_DIR}/code/build_queue.py\n"
        f"bash {RUN_DIR}/code/submit.sh\n"
        f"# code commit at launch: {commit}\n"
        "# one unit runs as:\n"
        + "".join(
            "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python "
            f"{PLATFORM_ROOT}/scripts/run_training.py --run-dir {RUN_DIR} "
            + " ".join(record["arguments"]) + "\n" for record in records))

    total = sum(record["estimated_seconds_h100"] for record in records)
    print(f"{len(records)} units written to {pending}")
    for record in records:
        print(f"  {record['unit_id']}: {record['copies']} copies, "
              f"{record['estimated_seconds_h100'] / 60:.1f} min on an H100, "
              f"needs {record['required_device_memory_gib']} GiB")
    print(f"whole sweep on one H100, one unit after another: {total / 60:.1f} min of compute")


if __name__ == "__main__":
    main()
