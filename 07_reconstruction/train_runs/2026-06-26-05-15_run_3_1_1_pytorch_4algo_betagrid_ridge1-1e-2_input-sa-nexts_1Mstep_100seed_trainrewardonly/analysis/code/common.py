#!/usr/bin/env python3
"""Shared loader + constants for the Train-run-3.1.1 aggregate.

Run 3.1.1 is a local work-queue sweep of train.py (SB3 SAC, PyTorch) over 91 configs x 100 seeds = 9100
runs, scored on the TRAINING-episode extrinsic reward (standalone eval is OFF). Each run writes one per-run
JSON under <data_dir>/local/<NNNN_of_9100>.json with this shape:

    {"run_id":.., "run_total":9100, "algorithm":"rnd_elliptical_global", "beta":1.0, "a_seed":3,
     "eval_standalone": false, "total_timesteps":1000000, "eval_freq":50000, "runtime_seconds":..,
     "eval_history":   [ {"step":50000, "visit_counts/...":..}, ... ],          # NO eval reward (eval off)
     "train_history":  [ {"step":50000, "train/mean_extrinsic_reward":-0.2, ...}, ... ],  # the scored curve
     "train_episode_history": [ ... per completed episode ... ],
     "distance_history": [],
     # elliptical runs ALSO carry these (RND runs omit them):
     "elliptical_update_timing":"add", "elliptical_feature_input":"next_state",
     "elliptical_regularization":0.01, "elliptical_feature_normalization":"unit"}

The four METHODS compared (columns of the beta-star table, curves in the plot):
    A1 rnd_elliptical        (batch covariance, update_timing=sample)
    A2 rnd_elliptical_global (global covariance, update_timing=sample)
    A3 rnd_elliptical_global (global covariance, update_timing=add)
    A4 rnd_next_state        (RND)
A1/A2/A3 each sweep beta(7) x ridge{1,1e-2} x input{state_action,next_state}; A4 sweeps beta(7). The
beta-star table reports, per method, the single best CONFIG (the (beta, ridge, input) with the highest
mean final training reward over seeds). The loader also parses the EVAL curve, so it can load the run-2
gt_position_velocity data (eval reward only) for the oracle reference curve overlaid on the plot.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import pandas as pd

# Score on the training-episode reward (eval is OFF in run 3.1.1). For the run-2 gt overlay we use its eval
# reward. STEP_KEY is the shared x-axis.
TRAIN_REWARD_KEY = "train/mean_extrinsic_reward"
EVAL_REWARD_KEY = "eval/mean_extrinsic_reward"
STEP_KEY = "step"
MIN_SEEDS = 30  # a curve segment / table cell is only shown when pooled over at least this many seeds

# Method order (left-to-right, table-row, legend). The run-2 gt oracle is appended for the plot only.
METHODS = ["A1_batch", "A2_global_sample", "A3_global_add", "A4_rnd_next_state"]
# matplotlib legend labels: plain underscores (matplotlib renders underscores literally in text mode; a
# LaTeX-escaped "\_" would print a visible backslash). The LaTeX table uses its own escaped names.
METHOD_LABEL = {
    "A1_batch": "rnd_elliptical (batch)",
    "A2_global_sample": "rnd_elliptical_global (sample)",
    "A3_global_add": "rnd_elliptical_global (add)",
    "A4_rnd_next_state": "rnd_next_state",
}
GT_METHOD = "gt_position_velocity"  # run-2 oracle reference (eval reward), overlaid on the plot
GT_BETA = 1.0
# Run-2 reference curves overlaid on the run-3.1.1 plot (EVAL reward -- run 2 logged eval only), each at its
# run-2 best-tuned beta: the visit-count oracle, the RND-state baseline, and the run-2 batch elliptical
# (which used RAW features and ridge 1e-6 -- run 2 predates the unit-norm default). 4 methods + 3 = 7 lines.
RUN2_OVERLAYS = [("gt_position_velocity", 1.0), ("rnd_state", 100.0), ("rnd_elliptical", 0.01)]

_TAB10 = matplotlib.colormaps["tab10"]
# stable colors: the gt oracle keeps tab10 index 0 (matching run-2's Figure 5), then the four methods;
# the run-2 rnd_state / rnd_elliptical overlays get indexes 5 / 6.
ALGO_COLOR = {GT_METHOD: _TAB10(0), "A1_batch": _TAB10(1), "A2_global_sample": _TAB10(2),
              "A3_global_add": _TAB10(3), "A4_rnd_next_state": _TAB10(4), "rnd_state": _TAB10(5),
              "rnd_elliptical": _TAB10(6)}

CODE_DIR = Path(__file__).resolve().parent
FOLDER = CODE_DIR.parent
DEFAULT_PLOTS_DIR = FOLDER / "plots"
# run-2 data dir (for the run-2 eval-reward reference curves)
RUN2_DATA_DIR = FOLDER.parent.parent / "2026-06-24-21-24_run_2_after_reorganization" / "data"


@dataclass
class RunRecord:
    """One run's tidy summary: identity + config knobs, runtime, final training reward, both curves."""

    algorithm: str
    beta: float
    seed: int
    update_timing: str | None        # elliptical only ("sample"|"add"); None for RND
    feature_input: str | None        # elliptical only ("state_action"|"next_state"); None for RND
    regularization: float | None     # elliptical ridge lambda; None for RND
    runtime_seconds: float | None
    final_train_reward: float | None
    train_curve: list[tuple[int, float]] = field(default_factory=list)
    final_eval_reward: float | None = None
    eval_curve: list[tuple[int, float]] = field(default_factory=list)

    def method_id(self) -> str | None:
        """Map this record to one of the four method ids (or the gt oracle / None if unrecognized)."""
        if self.algorithm == GT_METHOD:
            return GT_METHOD
        if self.algorithm == "rnd_elliptical":
            return "A1_batch"
        if self.algorithm == "rnd_elliptical_global":
            return "A2_global_sample" if self.update_timing == "sample" else "A3_global_add"
        if self.algorithm == "rnd_next_state":
            return "A4_rnd_next_state"
        return None

    def config_id(self) -> str:
        """Canonical sub-config identity within a method, as a STRING so None (RND has no ridge/input) round-
        trips cleanly through pandas groupby (which would turn a float None into NaN and break matching).
        e.g. 'A2_global_sample|beta=1|ridge=0.01|input=state_action' or 'A4_rnd_next_state|beta=1|ridge=none|input=none'."""
        return config_id_str(self.method_id(), self.beta, self.regularization, self.feature_input)


def _finite(x) -> bool:
    """True if x is a real, finite number (rejects None / NaN / inf / bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def ridge_str(reg) -> str:
    """Ridge lambda as a stable string ('none' when absent, e.g. RND)."""
    return "none" if reg is None or not _finite(reg) else f"{reg:g}"


def input_str(inp) -> str:
    """Feature-input as a stable string ('none' when absent, e.g. RND)."""
    return inp if inp else "none"


def config_id_str(method_id, beta, reg, inp) -> str:
    """Canonical config-id string shared by the loader and the analysis so None (RND ridge/input)
    round-trips as the literal 'none' instead of becoming NaN in a pandas groupby and failing to match."""
    return f"{method_id}|beta={beta:g}|ridge={ridge_str(reg)}|input={input_str(inp)}"


def _parse_reward_curve(history, reward_key) -> tuple[float | None, list[tuple[int, float]]]:
    """Extract (final reward, step-sorted (step, reward) curve) from a history list keyed by reward_key.
    Before: history=[{"step":50000,"<key>":-0.1,...},...]; After: final=reward at max step, curve=[(step,r),...]."""
    # keep only entries with a finite step and a finite reward under reward_key
    curve = [
        (int(e[STEP_KEY]), float(e[reward_key]))
        for e in history
        if STEP_KEY in e and reward_key in e and _finite(e.get(STEP_KEY)) and _finite(e.get(reward_key))
    ]
    if not curve:
        return None, []
    curve.sort(key=lambda sr: sr[0])  # sort by step so "final" = largest step regardless of write order
    return curve[-1][1], curve


def _record_from_json(obj: dict) -> RunRecord:
    """Build a RunRecord from one parsed per-run JSON dict (elliptical knobs absent on RND runs -> None)."""
    final_train, train_curve = _parse_reward_curve(obj.get("train_history", []) or [], TRAIN_REWARD_KEY)
    final_eval, eval_curve = _parse_reward_curve(obj.get("eval_history", []) or [], EVAL_REWARD_KEY)
    reg = obj.get("elliptical_regularization")
    return RunRecord(
        algorithm=str(obj["algorithm"]),
        beta=float(obj["beta"]),
        seed=int(obj["a_seed"]),
        update_timing=obj.get("elliptical_update_timing"),
        feature_input=obj.get("elliptical_feature_input"),
        regularization=float(reg) if _finite(reg) else None,
        runtime_seconds=float(obj["runtime_seconds"]) if _finite(obj.get("runtime_seconds")) else None,
        final_train_reward=final_train,
        train_curve=train_curve,
        final_eval_reward=final_eval,
        eval_curve=eval_curve,
    )


def load_records(data_dir: Path | str) -> list[RunRecord]:
    """Load every per-run JSON under <data_dir>/local/ into RunRecords. Robust to partial/missing files:
    a JSON that fails to parse (a run still being written) is skipped and counted on stderr."""
    data_dir = Path(data_dir)
    records: list[RunRecord] = []
    n_bad = 0
    n_partial = 0
    mode_dir = data_dir / "local"
    if mode_dir.is_dir():
        for jf in sorted(mode_dir.glob("*.json")):
            # a half-written JSON is an expected transient during a live sweep: skip + count it
            try:
                obj = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                n_bad += 1
                continue
            # checkpoint records (completed=false, eval-cadence flushes since 2026-07-02) are unfinished
            # runs: exclude from final-reward aggregates (a missing flag = old write-once record = complete)
            if obj.get("completed", True) is False:
                n_partial += 1
                continue
            records.append(_record_from_json(obj))
    if n_bad or n_partial:
        print(f"[common] skipped {n_bad} unreadable + {n_partial} incomplete-checkpoint JSON(s) under {data_dir}",
              file=sys.stderr)
    return records


def train_curve_frame(records: list[RunRecord]) -> pd.DataFrame:
    """Long-format TRAINING reward curves: one row per (run, step). Columns [method, config_id, seed, step, reward].
    Records with no recognized method or empty train_curve contribute nothing."""
    rows = []
    for r in records:
        mid = r.method_id()
        if mid is None:
            continue
        for step, reward in r.train_curve:
            rows.append({"method": mid, "config_id": r.config_id(), "seed": r.seed, "step": step, "reward": reward})
    return pd.DataFrame(rows, columns=["method", "config_id", "seed", "step", "reward"])


def eval_curve_frame(records: list[RunRecord], algorithm: str, beta: float) -> pd.DataFrame:
    """Long-format EVAL reward curve for one (algorithm, beta) reference from a loaded record set
    (used for the run-2 and run-4 overlay curves): [seed, step, reward]."""
    rows = []
    for r in records:
        if r.algorithm == algorithm and abs(r.beta - beta) < 1e-9:
            for step, reward in r.eval_curve:
                rows.append({"seed": r.seed, "step": step, "reward": reward})
    return pd.DataFrame(rows, columns=["seed", "step", "reward"])


def rank_marks(values, lower_is_better: bool) -> tuple[set[int], set[int]]:
    """Indices of best and second-best values for bold / underline marking (handles ties, skips None)."""
    valid = [(i, v) for i, v in enumerate(values) if _finite(v)]
    if not valid:
        return set(), set()
    distinct = sorted({v for _, v in valid}, reverse=not lower_is_better)
    best_v = distinct[0]
    second_v = distinct[1] if len(distinct) > 1 else None
    best_set = {i for i, v in valid if v == best_v}
    second_set = {i for i, v in valid if second_v is not None and v == second_v}
    return best_set, second_set


def tex_escape(s: str) -> str:
    """Escape underscores for LaTeX (rnd_state -> rnd\\_state)."""
    return s.replace("_", "\\_")
