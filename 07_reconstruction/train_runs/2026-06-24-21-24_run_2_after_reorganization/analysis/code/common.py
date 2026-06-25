#!/usr/bin/env python3
"""
Shared loader + constants for the "train run 2" aggregate.

Train run 2 is a local work-queue sweep that runs train.py 600 times
(3 algorithms x 200 seeds, 1e6 steps; one logging mode "local", no wandb). Each run writes
one per-run JSON to:

    <data_dir>/local/<run_name>.json     (run_name = "<run_id>_of_<run_total>", e.g. 042_of_600)

with this shape:

    {"run_id": 42, "run_total": 600,
     "algorithm": "rnd_state", "beta": 100.0, "a_seed": 1,
     "z_logging_mode": "local", "total_timesteps": 1000000,
     "eval_freq": 50000, "runtime_seconds": 1234.5,
     "eval_history": [ {"step": 50000, "eval/mean_extrinsic_reward": -0.1, ...}, ... ],
     "train_history": [ {"step": 50000, "train/mean_extrinsic_reward": -0.2, ...}, ... ],
     "distance_history": [ {"step": 50000, "distance_to_gt/min_c_l1_diff": ..., ...}, ... ] }

This module loads every per-run JSON from the "local" mode subdirectory of a data
directory into a list of `RunRecord`s and offers tidy views (a per-run frame and a long
reward-curve frame for either the eval or the train curve). It reads by content, not
filename, so old descriptive-name JSONs and new id-named JSONs load together. It is robust
to partial / missing files: a JSON that fails to parse (a run still being written) is
skipped and counted; an empty `eval_history` / `train_history` / `distance_history` yields
`None` / `[]` for the final reward / curve / distance instead of crashing.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import pandas as pd

# The three algorithms in this sweep, each pinned to its single best-tuned beta.
# Fixed left-to-right / legend order; tables and plots follow it.
ALGORITHMS = ["gt_position_velocity", "rnd_elliptical", "rnd_state"]
ALGO_BETA = {"gt_position_velocity": 1.0, "rnd_elliptical": 0.01, "rnd_state": 100.0}

# Single logging mode now: run 2 uses a local work queue and never calls wandb, so there is no
# full-vs-param-only comparison. All runs log locally under data/local/. The loader still iterates this
# list, so reward / distance simply pool over the one mode.
MODES = ["local"]

# Primary metric: per-eval mean extrinsic reward. STEP_KEY is the x-axis next to it.
REWARD_KEY = "eval/mean_extrinsic_reward"
# Training-episode reward at the same cadence: mean extrinsic reward over the PAST n_eval_episodes training
# episodes (logged by TrainEpisodeStatsCallback into train_history). Drawn as a DASHED line, same color as
# the algorithm's eval line. Absent on runs finished before train_history was saved -> empty train_curve.
TRAIN_REWARD_KEY = "train/mean_extrinsic_reward"
STEP_KEY = "step"

# The six distance-to-ground-truth metrics (all lower-is-better). gt_position_velocity
# is the oracle field, so its distances are ~0.
DISTANCE_METRICS = [
    "distance_to_gt/min_c_l1_diff",
    "distance_to_gt/min_c_l2_diff",
    "distance_to_gt/min_c_l1_inv",
    "distance_to_gt/min_c_l2_inv",
    "distance_to_gt/normalized_l2",
    "distance_to_gt/normalized_angle_rad",
]
# Short header label for each metric (drops the "distance_to_gt/" prefix), for tables.
DISTANCE_SHORT = {k: k.split("/", 1)[1] for k in DISTANCE_METRICS}

# One stable color per algorithm (tab10) so the bar plot and the line plot agree.
_TAB10 = matplotlib.colormaps["tab10"]
ALGO_COLOR = {a: _TAB10(i) for i, a in enumerate(ALGORITHMS)}

# Default paths, derived from this file's location:
#   code/ -> analysis/ (FOLDER) -> <run-slug>/ ; data lives at <run-slug>/data,
#   plots at <run-slug>/analysis/plots.
CODE_DIR = Path(__file__).resolve().parent
FOLDER = CODE_DIR.parent
DEFAULT_DATA_DIR = FOLDER.parent / "data"
DEFAULT_PLOTS_DIR = FOLDER / "plots"


@dataclass
class RunRecord:
    """One training run's tidy summary: identity, runtime, final reward, eval curve, final distances."""

    algorithm: str
    beta: float
    mode: str
    seed: int
    runtime_seconds: float | None
    final_reward: float | None  # last eval_history entry's eval/mean_extrinsic_reward
    eval_curve: list[tuple[int, float]] = field(default_factory=list)  # (step, eval reward), step-sorted
    train_curve: list[tuple[int, float]] = field(default_factory=list)  # (step, train reward over past N eps)
    final_distance: dict[str, float] | None = None  # last distance_history entry's six metrics


def _finite(x) -> bool:
    """True if x is a real, finite number (rejects None / NaN / inf)."""
    return isinstance(x, (int, float)) and not (isinstance(x, bool)) and math.isfinite(x)


def _parse_reward_curve(history: list[dict], reward_key: str) -> tuple[float | None, list[tuple[int, float]]]:
    """Extract (final reward, step-sorted (step, reward) curve) from a history list keyed by reward_key.

    Shared by the eval curve (reward_key=REWARD_KEY) and the training curve (reward_key=TRAIN_REWARD_KEY).
    Before: history = [{"step": 50000, "<reward_key>": -0.1, ...}, ...]
    After:  final = reward at the largest step; curve = [(50000, -0.1), (100000, 0.2), ...]
    """
    # keep only entries that carry both a finite step and a finite reward under reward_key
    curve = [
        (int(e[STEP_KEY]), float(e[reward_key]))
        for e in history
        if STEP_KEY in e and reward_key in e and _finite(e.get(STEP_KEY)) and _finite(e.get(reward_key))
    ]
    if not curve:
        return None, []
    # sort by step so "final" = largest step regardless of write order
    curve.sort(key=lambda sr: sr[0])
    return curve[-1][1], curve


def _parse_final_distance(distance_history: list[dict]) -> dict[str, float] | None:
    """Extract the six distance metrics from the last (largest-step) distance_history entry.

    Before: distance_history = [{"distance_to_gt/min_c_l1_diff": 1476.9, ..., "step": 50000}, ...]
    After:  {"distance_to_gt/min_c_l1_diff": 1476.9, ...} from the entry with the largest step,
            or None if there is no usable entry.
    """
    # keep entries with a finite step so "final" is well defined
    stepped = [e for e in distance_history if STEP_KEY in e and _finite(e.get(STEP_KEY))]
    if not stepped:
        return None
    # the entry at the largest step is the final distance snapshot
    last = max(stepped, key=lambda e: e[STEP_KEY])
    out = {m: float(last[m]) for m in DISTANCE_METRICS if m in last and _finite(last.get(m))}
    return out or None


def _record_from_json(obj: dict, mode_hint: str) -> RunRecord:
    """Build a RunRecord from one parsed per-run JSON dict (mode_hint = the subdir it came from)."""
    # parse eval + train reward curves (train_history absent on older runs -> empty curve) + final distance
    final_reward, eval_curve = _parse_reward_curve(obj.get("eval_history", []) or [], REWARD_KEY)
    _, train_curve = _parse_reward_curve(obj.get("train_history", []) or [], TRAIN_REWARD_KEY)
    final_distance = _parse_final_distance(obj.get("distance_history", []) or [])
    runtime = obj.get("runtime_seconds")
    return RunRecord(
        algorithm=str(obj["algorithm"]),
        beta=float(obj["beta"]),
        # trust the JSON's own mode field; fall back to the subdir name it was found in
        mode=str(obj.get("z_logging_mode", mode_hint)),
        seed=int(obj["a_seed"]),
        runtime_seconds=float(runtime) if _finite(runtime) else None,
        final_reward=final_reward,
        eval_curve=eval_curve,
        train_curve=train_curve,
        final_distance=final_distance,
    )


def load_records(data_dir: Path | str) -> list[RunRecord]:
    """Load every per-run JSON under both mode subdirs of data_dir into RunRecords.

    Robust to partial / missing data: a missing mode subdir is skipped; a JSON that
    fails to parse (a run still writing) is skipped and the count is reported on stderr.
    """
    data_dir = Path(data_dir)
    records: list[RunRecord] = []
    n_bad = 0
    # walk each known mode subdir (a missing one just contributes nothing)
    for mode in MODES:
        mode_dir = data_dir / mode
        if not mode_dir.is_dir():
            continue
        for jf in sorted(mode_dir.glob("*.json")):
            # a half-written JSON is an expected transient during a live sweep: skip + count it
            try:
                obj = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                n_bad += 1
                continue
            records.append(_record_from_json(obj, mode_hint=mode))
    if n_bad:
        print(f"[common] skipped {n_bad} unreadable / partial JSON file(s) under {data_dir}", file=sys.stderr)
    return records


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    """One row per run: algorithm, beta, mode, seed, runtime_seconds, final_reward, + 6 distance cols.

    Before: [RunRecord(algorithm='rnd_state', mode='wandb_full', seed=0, final_reward=-0.1, ...), ...]
    After:  DataFrame with columns
            [algorithm, beta, mode, seed, runtime_seconds, final_reward, <6 distance_to_gt/* cols>].
    """
    rows = []
    for r in records:
        # flatten the per-run record; missing final distance -> all-None distance cells
        row = {
            "algorithm": r.algorithm,
            "beta": r.beta,
            "mode": r.mode,
            "seed": r.seed,
            "runtime_seconds": r.runtime_seconds,
            "final_reward": r.final_reward,
        }
        for m in DISTANCE_METRICS:
            row[m] = (r.final_distance or {}).get(m)
        rows.append(row)
    return pd.DataFrame(rows, columns=["algorithm", "beta", "mode", "seed", "runtime_seconds", "final_reward", *DISTANCE_METRICS])


def curve_frame(records: list[RunRecord], attr: str = "eval_curve") -> pd.DataFrame:
    """Long-format reward curves pooled across modes + seeds: columns [algorithm, mode, seed, step, reward].

    attr selects which per-run curve to explode: "eval_curve" (solid eval line) or "train_curve"
    (dashed training line). Runs whose chosen curve is empty contribute no rows.
    Before: records each with <attr> = [(50000, -0.1), (100000, 0.2), ...]
    After:  one row per (run, step): algorithm, mode, seed, step, reward.
    """
    rows = []
    for r in records:
        # explode this run's chosen (step, reward) curve into one row per logged step
        for step, reward in getattr(r, attr):
            rows.append({"algorithm": r.algorithm, "mode": r.mode, "seed": r.seed, "step": step, "reward": reward})
    return pd.DataFrame(rows, columns=["algorithm", "mode", "seed", "step", "reward"])


def rank_marks(values: list[float | None], lower_is_better: bool) -> tuple[set[int], set[int]]:
    """Indices of the best and second-best values for bold / underline marking (handles ties, skips None).

    Returns (best_indices, second_indices). With lower_is_better the smallest value
    wins; otherwise the largest. Ties share the rank, e.g. two equal minima are both "best".
    """
    # collect the finite candidates with their positions
    valid = [(i, v) for i, v in enumerate(values) if _finite(v)]
    if not valid:
        return set(), set()
    distinct = sorted({v for _, v in valid}, reverse=not lower_is_better)
    best_v = distinct[0]
    second_v = distinct[1] if len(distinct) > 1 else None
    best_set = {i for i, v in valid if v == best_v}
    second_set = {i for i, v in valid if second_v is not None and v == second_v}
    return best_set, second_set


def tex_escape_algo(name: str) -> str:
    """Escape an algorithm name's underscores for LaTeX (rnd_state -> rnd\\_state)."""
    return name.replace("_", "\\_")
