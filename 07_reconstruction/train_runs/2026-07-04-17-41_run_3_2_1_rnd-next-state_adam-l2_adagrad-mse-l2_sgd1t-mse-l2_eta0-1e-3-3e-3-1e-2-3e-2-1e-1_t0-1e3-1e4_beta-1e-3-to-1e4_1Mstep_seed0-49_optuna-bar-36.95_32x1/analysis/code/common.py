#!/usr/bin/env python3
"""Shared loader + helpers for the run-3.2.1 (optimizer sweep) and run-3.2.2 (fresh-seed validation) tables
and the training-reward curve figure.

Each per-run JSON (written by train.py) has this shape (only the fields used here are shown):

    {"algorithm": "rnd_next_state", "completed": true,
     "rnd_optimizer": "sgd1t", "rnd_bonus_readout": "mse",     # optimizer + bonus readout switches
     "rnd_sgd_eta0": 0.1, "rnd_sgd_t0": 1000.0,                # SGD-1/t step-size schedule (sgd1t only)
     "beta": 10.0, "a_seed": 3,
     "train_history": [{"step": 50000, "train/mean_extrinsic_reward": 0.0, ...}, ...],  # scored curve
     "eval_history":  [{"step": 50000, "eval/mean_extrinsic_reward": 44.9, ...}, ...]}  # oracle ref only

Scoring convention (matches run 3.1.1 / 3.1.2):
- A run's score R_i = the LAST train_history row's "train/mean_extrinsic_reward" (largest step; the mean
  extrinsic reward over the past 100 completed training episodes at 1e6 steps).
- A configuration pools its seeds: Rbar = mean R_i, SE = s / sqrt(n) (s = sample std, ddof=1),
  success = fraction of seeds with R_i > 5.0.
- Only completed=true runs are loaded. A curve point (per step) needs >= 10 seeds.

The R311 run-3.1.1 rnd_next_state records predate the optimizer switch and carry NO rnd_optimizer /
rnd_bonus_readout fields; they are implicitly adam / mse (the defaults below).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

TRAIN_REWARD_KEY = "train/mean_extrinsic_reward"
EVAL_REWARD_KEY = "eval/mean_extrinsic_reward"
STEP_KEY = "step"
SUCCESS_THRESHOLD = 5.0   # a seed "succeeds" if its final training reward exceeds this
MIN_SEEDS = 10            # a config is eligible / a curve point is drawn only at this many seeds or more


def _finite(x) -> bool:
    """True if x is a real, finite number (rejects None / NaN / inf / bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class RunRecord:
    """One run's tidy summary: optimizer/readout/schedule/beta identity, seed, final training reward, and
    both the training-reward curve (scored) and the eval-reward curve (used only for the run-2 oracle)."""

    optimizer: str
    readout: str
    eta0: float
    t0: float
    beta: float
    seed: int
    algorithm: str
    final_train_reward: float | None
    train_curve: list[tuple[int, float]] = field(default_factory=list)
    eval_curve: list[tuple[int, float]] = field(default_factory=list)

    def config_key(self) -> str:
        """Canonical configuration id: 'optimizer|readout|eta0|t0|beta', with eta0/t0 shown only for the
        SGD-1/t optimizer (a literal '-' otherwise), e.g. 'sgd1t|mse|0.1|1000|10' or 'adam|mse|-|-|100'."""
        return config_key(self.optimizer, self.readout, self.eta0, self.t0, self.beta)


def config_key(optimizer: str, readout: str, eta0: float, t0: float, beta: float) -> str:
    """Build the canonical config id string 'optimizer|readout|%g(eta0)|%g(t0)|%g(beta)'; eta0/t0 are the
    literal '-' unless the optimizer is SGD-1/t (only sgd1t carries a step-size schedule)."""
    e = ("%g" % eta0) if optimizer == "sgd1t" else "-"
    t = ("%g" % t0) if optimizer == "sgd1t" else "-"
    return f"{optimizer}|{readout}|{e}|{t}|{beta:g}"


def _parse_reward_curve(history, reward_key) -> tuple[float | None, list[tuple[int, float]]]:
    """Extract (final reward at the largest step, step-sorted (step, reward) list) from a history list.

    Before: history=[{"step":50000,"<key>":0.0,...}, {"step":1000000,"<key>":47.6,...}, ...]
    After:  final=47.6 (reward at the largest step), curve=[(50000,0.0),...,(1000000,47.6)].
    """
    # keep only rows with a finite step and a finite reward under reward_key
    curve = [
        (int(e[STEP_KEY]), float(e[reward_key]))
        for e in history
        if STEP_KEY in e and reward_key in e and _finite(e.get(STEP_KEY)) and _finite(e.get(reward_key))
    ]
    if not curve:
        return None, []
    curve.sort(key=lambda sr: sr[0])   # sort by step so "final" = largest step regardless of write order
    return curve[-1][1], curve


def _record_from_json(obj: dict) -> RunRecord:
    """Build a RunRecord from one parsed per-run JSON dict. Optimizer/readout default to adam/mse for the
    run-3.1.1 rnd_next_state records, which predate the optimizer switch and omit those fields."""
    # final training reward + curve (the scored metric); eval curve only for the run-2 oracle overlay
    final_train, train_curve = _parse_reward_curve(obj.get("train_history", []) or [], TRAIN_REWARD_KEY)
    _, eval_curve = _parse_reward_curve(obj.get("eval_history", []) or [], EVAL_REWARD_KEY)
    return RunRecord(
        optimizer=str(obj.get("rnd_optimizer", "adam")),
        readout=str(obj.get("rnd_bonus_readout", "mse")),
        eta0=float(obj["rnd_sgd_eta0"]) if _finite(obj.get("rnd_sgd_eta0")) else float("nan"),
        t0=float(obj["rnd_sgd_t0"]) if _finite(obj.get("rnd_sgd_t0")) else float("nan"),
        beta=float(obj["beta"]),
        seed=int(obj["a_seed"]),
        algorithm=str(obj.get("algorithm", "")),
        final_train_reward=final_train,
        train_curve=train_curve,
        eval_curve=eval_curve,
    )


def load_records(local_dir: Path | str, algorithm_filter: str | None = None) -> list[RunRecord]:
    """Load every completed per-run JSON directly under <local_dir> into RunRecords.

    Only completed=true runs are kept (a missing flag = an old write-once record = complete). Half-written
    JSONs (a transient during a live sweep) are skipped and counted on stderr. When algorithm_filter is
    given, only runs of that algorithm are kept (used to pull just rnd_next_state out of the mixed run-3.1.1
    data). Scans of the run-3.2.1 dir are slow (~9000 files); that is expected.
    """
    local_dir = Path(local_dir)
    records: list[RunRecord] = []
    n_bad = 0
    n_partial = 0
    for jf in sorted(local_dir.glob("*.json")):
        # a half-written JSON is an expected transient during a live sweep: skip + count it
        try:
            obj = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            n_bad += 1
            continue
        # checkpoint records (completed=false) are unfinished runs: excluded from every aggregate
        if obj.get("completed", True) is False:
            n_partial += 1
            continue
        if algorithm_filter is not None and obj.get("algorithm") != algorithm_filter:
            continue
        records.append(_record_from_json(obj))
    if n_bad or n_partial:
        print(f"[common] skipped {n_bad} unreadable + {n_partial} incomplete JSON(s) under {local_dir}",
              file=sys.stderr)
    return records


def rank_marks(values, lower_is_better: bool = False) -> tuple[set[int], set[int]]:
    """Indices of the best and second-best values for bold / underline marking (handles ties, skips None).
    Pass ALREADY-ROUNDED display values so ties at display precision (e.g. two configs both 0.77) are
    marked together, per the analysis convention."""
    valid = [(i, v) for i, v in enumerate(values) if _finite(v)]
    if not valid:
        return set(), set()
    distinct = sorted({v for _, v in valid}, reverse=not lower_is_better)
    best_v = distinct[0]
    second_v = distinct[1] if len(distinct) > 1 else None
    best_set = {i for i, v in valid if v == best_v}
    second_set = {i for i, v in valid if second_v is not None and v == second_v}
    return best_set, second_set
