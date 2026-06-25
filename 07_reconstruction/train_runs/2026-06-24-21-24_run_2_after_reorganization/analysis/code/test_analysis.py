#!/usr/bin/env python3
"""
Unit tests for the train-run-2 analysis pipeline (revised: single "local" logging mode, no wandb).

Covers the load + parse helpers in common.py: the shared reward-curve parser (eval and train), the
final-distance parser, rank marking, the robust loader, and the reward/distance summaries. Each per-run
JSON carries the revised convention (run_id/run_total, three groups eval/train/distance). Run with:

    conda run -n exploration python -m pytest test_analysis.py
    conda run -n exploration python test_analysis.py        # pytest-free fallback
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import make_distance_table  # noqa: E402
import make_reward_table  # noqa: E402


def _run_json(algorithm: str, seed: int, runtime: float, run_id: int = 0, run_total: int = 600,
              n_eval: int = 3, with_distance: bool = True) -> dict:
    """Build a minimal but realistic per-run JSON dict (revised local-mode record) for tests."""
    # eval_history: n_eval evals at 50k,100k,...; extrinsic reward climbs with step + a small seed offset
    eval_history = [
        {"step": 50000 * (i + 1), "eval/mean_extrinsic_reward": float(i + seed * 0.01), "eval/mean_ep_length": 50.0}
        for i in range(n_eval)
    ]
    # train_history: same cadence; train reward sits just below eval (the dashed curve), with the window size
    train_history = [
        {"step": 50000 * (i + 1), C.TRAIN_REWARD_KEY: float(i - 0.5 + seed * 0.01), "train/n_episodes_averaged": 100}
        for i in range(n_eval)
    ]
    obj = {
        "run_id": run_id, "run_total": run_total,
        "algorithm": algorithm, "beta": C.ALGO_BETA[algorithm], "a_seed": seed,
        "z_logging_mode": "local", "total_timesteps": 1000000, "eval_freq": 50000,
        "runtime_seconds": runtime,
        "eval_history": eval_history, "train_history": train_history, "distance_history": [],
    }
    if with_distance:
        # distance_history: one entry per eval step, all six metrics decreasing slightly
        obj["distance_history"] = [
            {**{m: 100.0 - i - seed * 0.001 for m in C.DISTANCE_METRICS}, "step": 50000 * (i + 1)}
            for i in range(n_eval)
        ]
    return obj


def _write_dataset(root: Path) -> None:
    """Write a small single-mode (local) dataset: full runs, one partial run, and a corrupt file."""
    # full, normal runs: 2 algorithms x seeds 0..2 (gt has near-zero distance; rnd_state has real distance)
    mode = C.MODES[0]  # "local" -- the only logging mode now
    (root / mode).mkdir(parents=True, exist_ok=True)
    rid = 0
    for seed in range(3):
        for algo in ("rnd_state", "gt_position_velocity"):
            obj = _run_json(algo, seed, runtime=100.0 + seed, run_id=rid, with_distance=True)
            if algo == "gt_position_velocity":
                # oracle field -> near-zero distances
                for e in obj["distance_history"]:
                    for m in C.DISTANCE_METRICS:
                        e[m] = 1e-9
            # id-based filename (revised convention); descriptive fields live in the JSON content
            (root / mode / f"{rid:03d}_of_600.json").write_text(json.dumps(obj))
            rid += 1
    # partial run: empty histories (a run mid-write) -> loads but has no final reward / distance / train curve
    partial = _run_json("rnd_elliptical", seed=0, runtime=42.0, run_id=rid, n_eval=0, with_distance=False)
    partial["train_history"] = []
    (root / mode / f"{rid:03d}_of_600.json").write_text(json.dumps(partial))
    # corrupt run: truncated JSON the loader must skip without crashing
    (root / mode / "corrupt.json").write_text('{"algorithm": "rnd_state", "a_seed": 0,')


def test_parse_reward_curve_eval_and_train():
    """Final reward = largest-step reward; the shared parser works for both eval and train keys; empty -> (None, [])."""
    # eval curve via REWARD_KEY
    final, curve = C._parse_reward_curve(
        [{"step": 100000, C.REWARD_KEY: 2.0}, {"step": 50000, C.REWARD_KEY: 1.0}], C.REWARD_KEY
    )
    assert final == 2.0 and curve == [(50000, 1.0), (100000, 2.0)]
    # the SAME helper parses the training curve via TRAIN_REWARD_KEY
    tfinal, tcurve = C._parse_reward_curve([{"step": 50000, C.TRAIN_REWARD_KEY: 0.5}], C.TRAIN_REWARD_KEY)
    assert tfinal == 0.5 and tcurve == [(50000, 0.5)]
    # empty / key-less history -> (None, [])
    assert C._parse_reward_curve([], C.REWARD_KEY) == (None, [])


def test_parse_final_distance_takes_largest_step():
    """Final distance is the snapshot at the largest step; missing data yields None."""
    hist = [
        {**{m: 1.0 for m in C.DISTANCE_METRICS}, "step": 50000},
        {**{m: 0.5 for m in C.DISTANCE_METRICS}, "step": 100000},
    ]
    out = C._parse_final_distance(hist)
    assert out is not None and out[C.DISTANCE_METRICS[0]] == 0.5
    assert C._parse_final_distance([]) is None


def test_rank_marks_higher_and_lower_better_with_ties():
    """rank_marks picks best/second for both directions and shares ranks on ties, skipping None."""
    best, second = C.rank_marks([1.0, 3.0, 2.0, None], lower_is_better=False)
    assert best == {1} and second == {2}
    best, second = C.rank_marks([1.0, 1.0, 2.0], lower_is_better=True)
    assert best == {0, 1} and second == {2}


def test_load_records_robust_to_partial_and_corrupt(tmp_path):
    """Loader reads all valid runs, keeps the partial run, and skips the corrupt file."""
    _write_dataset(tmp_path)
    records = C.load_records(tmp_path)
    # 6 normal (2 algos x 3 seeds) + 1 partial = 7; the corrupt file is dropped
    assert len(records) == 7
    partial = [r for r in records if r.algorithm == "rnd_elliptical"][0]
    assert partial.final_reward is None and partial.eval_curve == [] and partial.train_curve == []


def test_records_carry_train_curve_and_tolerate_id_fields(tmp_path):
    """A full run loads its train_curve (from train_history) at the eval cadence; extra run_id/run_total are ignored."""
    _write_dataset(tmp_path)
    rec = [r for r in C.load_records(tmp_path) if r.algorithm == "rnd_state"][0]
    # train_curve parsed from train_history, same length + cadence as the eval curve
    assert len(rec.train_curve) == len(rec.eval_curve) == 3
    assert rec.train_curve[0][0] == 50000 and rec.eval_curve[0][0] == 50000  # (step, reward), step-sorted


def test_reward_summary_single_mode(tmp_path):
    """Reward summary pools the single local mode: rnd_state / gt each have n = 3 seeds."""
    _write_dataset(tmp_path)
    summary = make_reward_table.reward_summary(C.load_records(tmp_path))
    n_by_algo = dict(zip(summary["algorithm"], summary["n"]))
    assert n_by_algo["rnd_state"] == 3 and n_by_algo["gt_position_velocity"] == 3
    # the partial rnd_elliptical run has no final reward, so it is absent from the summary
    assert "rnd_elliptical" not in n_by_algo


def test_distance_summary_only_algos_with_distance(tmp_path):
    """Distance summary includes only algorithms with distance data; gt is ~0 (best, first row)."""
    _write_dataset(tmp_path)
    summary = make_distance_table.distance_summary(C.load_records(tmp_path))
    assert set(summary["algorithm"]) == {"rnd_state", "gt_position_velocity"}
    # gt_position_velocity (~1e-9) is the smallest on the sort metric -> first row
    assert summary.iloc[0]["algorithm"] == "gt_position_velocity"


def _run_all_tests():
    """Run every test function in this module against a fresh tmp dir (pytest-free fallback)."""
    import tempfile

    # the no-arg tests run directly; the tmp_path ones get a fresh temp directory each
    test_parse_reward_curve_eval_and_train()
    test_parse_final_distance_takes_largest_step()
    test_rank_marks_higher_and_lower_better_with_ties()
    for fn in (
        test_load_records_robust_to_partial_and_corrupt,
        test_records_carry_train_curve_and_tolerate_id_fields,
        test_reward_summary_single_mode,
        test_distance_summary_only_algos_with_distance,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("all tests passed")


if __name__ == "__main__":
    _run_all_tests()
