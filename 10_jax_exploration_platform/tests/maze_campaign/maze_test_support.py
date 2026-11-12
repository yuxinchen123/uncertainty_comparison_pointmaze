"""Shared pieces of the maze campaign's tests: paths, a study description, a card, and a stand-in trainer.

Standard library only: these tests import no numerical library and start no trainer. They may run
on a login node: `python -m unittest discover -s tests/maze_campaign -p "test_maze_*.py"`.
"""
import json
from pathlib import Path
import sys

PLATFORM = Path(__file__).resolve().parents[2]
MAZE = PLATFORM / "scripts/maze_campaign"
if str(MAZE) not in sys.path:
    sys.path.insert(0, str(MAZE))

import build_units  # noqa: E402
import unit_queue  # noqa: E402

POINTMAZE = ("configs/experiments/pointmaze_large_cleanrl-ppo-rnd_goal-termination_"
             "reward-arrival-plus-one_example_steps-1024000.toml")
ANTMAZE = ("configs/experiments/antmaze_umaze_cleanrl-ppo-rnd_goal-termination_"
           "reward-arrival-plus-one_example_steps-204800.toml")
CARD = {"gpu": "NVIDIA RTX A4500", "memory_mb": 20470, "driver": "580.0", "hostname": "testnode",
        "recorded_at": "2026-11-12T01:00:00-08:00", "cuda_visible_devices": "0"}

# A stand-in for scripts/run_training.py. It writes what a unit's process leaves behind (a shard,
# the unit's own log, the command it was given) and ends the way FAKE_TRAINER_MODE says.
FAKE_TRAINER = r'''
import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir")
parser.add_argument("--unit-id")
args, _ = parser.parse_known_args()
run = Path(args.run_dir)
for folder in ("data", "logs"):
    (run / folder).mkdir(exist_ok=True)
(run / "logs" / f"{args.unit_id}.log").open("a").write("the unit's own log\n")
(run / f"command_{args.unit_id}.json").write_text(json.dumps(sys.argv[1:]))
mode = os.environ["FAKE_TRAINER_MODE"]
with (run / "data" / f"{args.unit_id}.jsonl").open("a") as shard:
    shard.write(json.dumps({"record": "episode_window", "unit_id": args.unit_id, "window_index": 0,
                            "first_iteration": 1, "last_iteration": 5}) + "\n")
    if mode == "complete":
        shard.write(json.dumps({"record": "unit_complete", "unit_id": args.unit_id, "copies": 32, "iterations": 10,
                                "windows": 2, "seconds_per_iteration_steady": 0.05}) + "\n")
if mode == "stop":
    raise SystemExit(f"unit {args.unit_id} stopped at iteration 5 of 10 with its state saved; run the same "
                     "command again to continue it")
if mode == "fail":
    raise RuntimeError("the stand-in trainer was told to fail")
if mode == "out_of_memory":
    print("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1000000 bytes", file=sys.stderr)
    sys.exit(1)
'''


def description(**changes):
    """The plan's first-pass development sweep on PointMaze Large in both reward settings, with changes applied."""
    study = {
        "study": "cleanrl-ppo-rnd-baseline", "phase": "development-first-pass", "method": "cleanrl-ppo-rnd",
        "bonus": "rnd_next_state", "environments": ["pointmaze-large"],
        "reward_settings": ["arrival_plus_one", "step_cost"],
        "configuration_files": {"pointmaze": POINTMAZE, "antmaze": ANTMAZE},
        "steps_per_policy": 50331648, "windows_per_unit": 200,
        "cells": {"kind": "first_pass_grid"}, "seeds": {"first": 0, "count": 16},
        "policies_per_unit": {"card": "RTX A4500", "default": 512}, "minimum_card_memory_mb": 7000,
    }
    study.update(changes)
    return study


def units_of(study, estimates_dir):
    """Build a description's unit specifications against the real platform sources and a given estimates folder."""
    return build_units.study_units(study, estimates_dir, PLATFORM)


def queue_units(run_dir, specifications):
    """Put specifications in a run folder's queue, in order; returns their names."""
    for index, specification in enumerate(specifications, 1):
        assert unit_queue.enqueue(run_dir, specification, build_units.queue_record(specification, index))
    return [specification["unit_id"] for specification in specifications]


def small_run(run_dir, units=2):
    """Queue a few small units (2 cells x 4 seeds x both rewards = 16 policies each) in a temporary run folder."""
    cells = [{"learning_rate": 1e-4, "entropy_coefficient": 0.001, "intrinsic_weight": float(index + 1)}
             for index in range(2 * units)]
    study = description(cells={"kind": "list", "cells": cells}, seeds={"first": 0, "count": 4},
                        policies_per_unit=16, iterations=10, window_iterations=5)
    del study["steps_per_policy"], study["windows_per_unit"]
    return queue_units(run_dir, units_of(study, Path(run_dir) / "no_estimates"))


def write_shard(run_dir, name, windows, complete=False):
    """Write a unit's shard with some window records and, if asked, its unit_complete record."""
    path = Path(run_dir) / "data" / f"{name}.jsonl"
    path.parent.mkdir(exist_ok=True)
    lines = [{"record": "unit_start", "unit_id": name}]
    lines += [{"record": "episode_window", "unit_id": name, "window_index": index, "first_iteration": 5 * index + 1,
               "last_iteration": 5 * index + 5, "return_sum_per_copy": [0.0] * 16} for index in range(windows)]
    if complete:
        lines.append({"record": "unit_complete", "unit_id": name, "copies": 16, "iterations": 5 * windows,
                      "windows": windows, "seconds_per_iteration_steady": 0.05})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path
