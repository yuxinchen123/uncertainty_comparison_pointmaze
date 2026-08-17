"""Roll this run's shards up into run-level metrics, and score every configuration of every arm.

This is the run's own aggregation module. It reads only `data/*.jsonl` and `slurm/jobs/*/*.json`,
and everything downstream — the 20-minute status table, the development document's tables, the
training-curve figure — imports the functions here rather than re-deriving a number from the shards.

Two rules the scoring obeys:

1. **Only phase-blocked windows are scored.** The task is continuing, so an episode ends only at the
   400-step truncation and every copy shares one episode clock; one iteration sees 128 of those 400
   steps, so its extrinsic reward is a sample of one window of the episode and can read zero for
   every copy while copies are solving. A record here covers 200 iterations, which is 8 whole turns
   of the 25-iteration clock, so its reward sum sees every part of the episode equally. A record
   whose `phase_blocked` flag is false is never scored.
2. **Only completed units feed a table number.** A unit still running has windows on disk and they
   are fine for a truncated curve, never for a table.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/aggregate.py
"""
import json
import math
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
PLATFORM_ROOT = RUN_DIR.parent.parent
sys.path.insert(0, str(PLATFORM_ROOT / "scripts"))

# the open cells of the large map; coverage is already reported as a fraction of them, and this is
# here so a reader of a coverage number can see what the denominator was
OPEN_CELLS = 46


def read_units(run_dir: Path = RUN_DIR) -> dict:
    """Every unit's records, grouped: its identity card, its windows, and its completion record.

    before: data/unit-1_....jsonl with job / unit_start / 98 episode_window / unit_complete lines;
    after:  {"unit-1_...": {"start": {...}, "windows": [98 records], "complete": {...}}}
    A window recorded twice (a re-attempt of the same unit) keeps the later line.
    """
    units = {}
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("record")
            if kind not in ("unit_start", "episode_window", "unit_complete"):
                continue
            unit = units.setdefault(record["unit_id"],
                                    {"start": None, "windows": {}, "complete": None})
            if kind == "unit_start":
                unit["start"] = record
            elif kind == "unit_complete":
                unit["complete"] = record
            else:
                unit["windows"][record["last_iteration"]] = record
    for unit in units.values():
        unit["windows"] = [unit["windows"][key] for key in sorted(unit["windows"])]
    return units


def scored_windows(unit: dict) -> list:
    """The unit's windows that may feed a score: the phase-blocked ones, in iteration order."""
    return [window for window in unit["windows"] if window["phase_blocked"]]


def cell_copies(unit: dict) -> dict:
    """Which copy indices belong to each (learning rate, intrinsic weight) cell of this unit.

    before: cell_settings [(1e-3, 1e-5), (1e-3, 1e-4), ...], copy_cell [0,0,...,1,1,...]
    after:  {(0.001, 1e-05): [0, 1, ..., 255], (0.001, 0.0001): [256, ...], ...}
    """
    settings = [tuple(setting) for setting in unit["start"]["cell_settings"]]
    members = {setting: [] for setting in settings}
    for copy_index, cell in enumerate(unit["start"]["copy_cell"]):
        members[settings[cell]].append(copy_index)
    return members


def mean_and_standard_error(values: list) -> tuple:
    """The mean of a list and the standard error of that mean; a single value has no error."""
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def per_copy_scores(unit: dict) -> dict:
    """The four per-copy numbers this run scores a configuration by, as lists over copies.

    whole_run_reward   mean episode return over every phase-blocked window of the run
    last_window_reward mean episode return over the final phase-blocked window
    reached_goal       1.0 if the copy ever collected extrinsic reward, else 0.0
    coverage_percent   distinct open maze cells entered, as a percentage of the 46 open cells

    before: 98 windows, each holding a per-copy reward SUM over 256 episodes;
    after:  whole_run_reward[k] = sum of copy k's 98 window sums / (98 x 256) episodes.
    """
    windows = scored_windows(unit)
    if not windows:
        return {}
    copies = len(windows[0]["reward_ext_sum_per_copy"])
    total_reward = [0.0] * copies
    total_episodes = 0.0
    for window in windows:
        total_episodes += window["episodes_per_copy_in_window"]
        for index, value in enumerate(window["reward_ext_sum_per_copy"]):
            total_reward[index] += value
    last = windows[-1]
    return {
        "whole_run_reward": [value / total_episodes for value in total_reward],
        "last_window_reward": [value / last["episodes_per_copy_in_window"]
                               for value in last["reward_ext_sum_per_copy"]],
        "reached_goal": [1.0 if value > 0 else 0.0 for value in total_reward],
        "coverage_percent": [100.0 * value for value in last.get("coverage_per_copy", [])],
        "episodes_per_copy_scored": total_episodes,
        "windows_scored": len(windows),
        "last_window_episodes": last["episodes_per_copy_in_window"],
    }


def cell_table(run_dir: Path = RUN_DIR, completed_only: bool = True) -> list:
    """One row per (arm, learning rate, intrinsic weight) cell, with mean and standard error.

    completed_only keeps the table rule: a unit that has not finished contributes to curves but
    never to a table number.
    """
    rows = []
    for unit_id, unit in sorted(read_units(run_dir).items()):
        if unit["start"] is None:
            continue
        if completed_only and unit["complete"] is None:
            continue
        scores = per_copy_scores(unit)
        if not scores:
            continue
        for (rate, weight), members in cell_copies(unit).items():
            row = {"unit_id": unit_id, "bonus": unit["start"]["bonus"],
                   "learning_rate": rate, "intrinsic_weight": weight,
                   "copies": len(members),
                   "episodes_per_copy_scored": scores["episodes_per_copy_scored"],
                   "windows_scored": scores["windows_scored"],
                   # steps to goal is not recorded by this trainer: the goal step inside an episode
                   # is never read back to the host, so the column has no number in this run
                   "steps_to_goal": None}
            for name in ("whole_run_reward", "last_window_reward", "coverage_percent"):
                values = [scores[name][index] for index in members] if scores[name] else []
                mean, error = mean_and_standard_error(values)
                row[name], row[f"{name}_standard_error"] = mean, error
            row["success_rate"] = (sum(scores["reached_goal"][index] for index in members)
                                   / len(members))
            rows.append(row)
    return rows


def best_per_arm(rows: list, key: str = "whole_run_reward") -> dict:
    """The highest-scoring cell of each algorithm arm, by the column the table is sorted on."""
    best = {}
    for row in rows:
        if row[key] is None:
            continue
        current = best.get(row["bonus"])
        if current is None or row[key] > current[key]:
            best[row["bonus"]] = row
    return best


def cell_curve(unit: dict, members: list) -> dict:
    """The learning curve of one cell: mean episode return per window, with its standard error.

    Every window is used, phase-blocked or not, but a window that is not phase-blocked is flagged
    so a plot can drop it; in this run only a final short window could ever be one.
    """
    steps, means, errors, blocked = [], [], [], []
    for window in unit["windows"]:
        episodes = window["episodes_per_copy_in_window"]
        values = [window["reward_ext_sum_per_copy"][index] / episodes for index in members]
        mean, error = mean_and_standard_error(values)
        steps.append(window["env_steps_per_copy"])
        means.append(mean)
        errors.append(error)
        blocked.append(window["phase_blocked"])
    return {"env_steps_per_copy": steps, "mean_episode_return": means,
            "standard_error": errors, "phase_blocked": blocked}


def aggregate(run_dir: Path = RUN_DIR) -> dict:
    """Write metrics.jsonl and summary.json, the summary carrying this run's own cell scores."""
    from aggregate_run import aggregate as generic_aggregate
    summary = generic_aggregate(run_dir)
    rows = cell_table(run_dir)
    summary["cells_scored"] = len(rows)
    summary["cells"] = rows
    summary["best_per_arm"] = best_per_arm(rows)
    summary["scoring"] = {
        "windows": "phase-blocked episode windows only (200 iterations = 8 turns of the "
                   "25-iteration episode clock = 256 episodes per copy)",
        "open_cells": OPEN_CELLS,
        "steps_to_goal": "not recorded in this run",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Aggregate this run and print the best configuration of each arm."""
    summary = aggregate()
    print(f"{summary['units_complete']} units complete, {summary['records']} window records, "
          f"{summary['cells_scored']} cells scored")
    for bonus, row in sorted(summary["best_per_arm"].items()):
        print(f"  {bonus}: learning rate {row['learning_rate']:g}, intrinsic weight "
              f"{row['intrinsic_weight']:g} -> whole-run reward "
              f"{row['whole_run_reward']:.4f} +- {row['whole_run_reward_standard_error']:.4f}, "
              f"coverage {row['coverage_percent']:.1f}%, success {row['success_rate']:.3f}")


if __name__ == "__main__":
    main()
