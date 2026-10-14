"""Aggregate explicitly selected configurations without loading full histories into memory."""
import argparse
import json
import math
from pathlib import Path


def mean_error(values):
    """Return the mean and sample standard error over independent training copies."""
    mean = sum(values) / len(values)
    error = (math.sqrt(sum((value - mean)**2 for value in values)
                       / (len(values) - 1) / len(values)) if len(values) > 1 else 0.)
    return mean, error


def best_per_algorithm(rows):
    """Select each algorithm's highest whole-run mean among the completed settings supplied."""
    if not rows:
        raise ValueError("no completed settings to rank")
    best = {}
    for row in rows:
        previous = best.get(row["bonus"])
        if previous is None or row["whole_run_reward"] > previous["whole_run_reward"]:
            best[row["bonus"]] = row
    return best


def read_unit(path):
    """Stream the latest attempt, retaining only per-copy totals and small curve records."""
    start, complete, totals, episodes, last, curve = None, None, None, 0., None, []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("record")
            if kind == "unit_start":
                # Before: a partial attempt followed by a retry; after: only the retry is scored.
                start, complete, totals, episodes, last, curve = record, None, None, 0., None, []
            elif kind == "unit_complete":
                complete = record
            elif kind == "episode_window" and record["phase_blocked"]:
                if start is None:
                    raise ValueError(f"window without unit_start in {path}")
                rewards = record["reward_ext_sum_per_copy"]
                if totals is None:
                    totals = [0.] * len(rewards)
                totals = [before + value for before, value in zip(totals, rewards)]
                episodes += record["episodes_per_copy_in_window"]
                values = [value / record["episodes_per_copy_in_window"] for value in rewards]
                mean, error = mean_error(values)
                curve.append({"steps": record["env_steps_per_copy"], "mean": mean,
                              "standard_error": error})
                last = record
    return start, complete, totals, episodes, last, curve


def aggregate(run_dir):
    """Write complete-setting scores and interim curves from the selected run's shards."""
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "parameter_settings.json").read_text())
    curves, cells, units = {}, [], []
    for path in sorted((run_dir / "data").glob("*.jsonl")):
        start, complete, totals, episodes, last, curve = read_unit(path)
        if start is None:
            continue
        if len(start["cell_settings"]) != 1:
            raise ValueError("selected-run shards must each hold one requested setting")
        rate, weight = start["cell_settings"][0]
        key = f"{start['bonus']}:{rate:g}:{weight:g}"
        if key in curves:
            raise ValueError(f"duplicate selected setting {key}")
        if [rate, weight] not in settings[start["bonus"]]:
            raise ValueError(f"unexpected parameter setting {key}")
        if start["copy_seed_index"] != list(range(128)):
            raise ValueError(f"{key} does not cover paired seeds 0 through 127 exactly")
        curves[key] = curve
        units.append({"unit_id": start["unit_id"], "complete": complete is not None,
                      "last_iteration": last["last_iteration"] if last else 0,
                      "iterations": start["iterations"], "completion": complete})
        if complete is None:
            continue

        # Before: window reward sums; after: each copy's mean return over whole episodes.
        row = {"bonus": start["bonus"], "learning_rate": rate, "intrinsic_weight": weight,
               "copies": len(totals), "curve_key": key,
               "success_rate": sum(value > 0 for value in totals) / len(totals)}
        series = {"whole_run_reward": [value / episodes for value in totals],
                  "last_window_reward": [value / last["episodes_per_copy_in_window"]
                                         for value in last["reward_ext_sum_per_copy"]],
                  "coverage_percent": [100 * value for value in last["coverage_per_copy"]]}
        for name, values in series.items():
            row[name], row[name + "_standard_error"] = mean_error(values)
        cells.append(row)

    # A single writer publishes atomic aggregates; training processes never overwrite them.
    summary = {"run_id": run_dir.name, "units_complete": sum(u["complete"] for u in units),
               "units_seen": len(units), "expected_configurations": sum(map(len, settings.values())),
               "units": units, "cells": cells, "curves": curves,
               "scoring": "phase-blocked episode windows; standard errors over 128 paired copies"}
    temp = run_dir / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    temp.replace(run_dir / "summary.json")
    temp = run_dir / "metrics.jsonl.tmp"
    temp.write_text("".join(json.dumps(dict(row, record="configuration_result")) + "\n"
                            for row in cells))
    temp.replace(run_dir / "metrics.jsonl")
    return summary


def main():
    """Refresh the selected run's aggregate and print its completed-setting count."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = aggregate(args.run_dir)
    print(f"{summary['units_complete']}/{summary['expected_configurations']} settings complete")


if __name__ == "__main__":
    main()
