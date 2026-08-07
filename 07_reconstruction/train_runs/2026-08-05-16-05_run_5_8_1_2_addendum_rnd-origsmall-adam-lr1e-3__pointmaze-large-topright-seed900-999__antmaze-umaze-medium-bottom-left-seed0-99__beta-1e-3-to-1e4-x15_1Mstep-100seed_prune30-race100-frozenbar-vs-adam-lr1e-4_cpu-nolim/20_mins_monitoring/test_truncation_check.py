#!/usr/bin/env python
"""Tests for the truncation invariant checker, on a CORRECT decision log and on deliberately wrong
ones — the sweep_prune skill requires the checker itself to be tested before launch, because a
checker that passes everything is worse than no checker.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest 20_mins_monitoring/test_truncation_check.py -q
"""
import hashlib
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue as bq        # noqa: E402
import truncation_check as chk  # noqa: E402

SWEEP = "sim-sweep"
POINTMAZE_KEY = "initial_single_large_pointmaze_max_400|origsmall|lr0.001|b0.001"
UMAZE_KEY = "AntMaze_UMaze-v5_start_bottom_left|origsmall|lr0.001|b30"
BARS = {"bars": {"initial_single_large_pointmaze_max_400": {"mean": 40.0},
                 "AntMaze_UMaze-v5_start_bottom_left": {"mean": -690.0},
                 "AntMaze_Medium-v5_start_bottom_left": {"mean": -998.0}}}


def decision(key, env, verdict, n, mean, std, bar, sha, rule):
    """One well-formed decision line with a self-consistent upper bound."""
    return {"verdict": verdict, "config_key": key, "env_setup": env, "score_rule": rule,
            "n": n, "mean": mean, "std": std,
            "upper_99": mean + 2.576 * std / math.sqrt(n),
            "bar": bar, "bars_sha256": sha}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A temporary run tree with a bars file and an empty queue, wired into the checker."""
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    (slurm / "FROZEN_BARS.json").write_text(json.dumps(BARS))
    (tmp_path / "queue" / SWEEP / "pending").mkdir(parents=True)
    monkeypatch.setattr(chk, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(chk, "SLURM", str(slurm))
    return tmp_path


def write_log(tree, entries):
    """Write a decision log from a list of dicts and return the bars sha the checker will compare to."""
    path = tree / "slurm" / f"truncation_decisions_{SWEEP}.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def sha_of(tree):
    """sha256 of the temporary tree's bars file."""
    return hashlib.sha256((tree / "slurm" / "FROZEN_BARS.json").read_bytes()).hexdigest()


def test_a_correct_log_passes(tree):
    """Golden path: one valid truncation and one valid survivor, no violations."""
    s = sha_of(tree)
    write_log(tree, [
        decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                 n=30, mean=1.0, std=0.5, bar=40.0, sha=s, rule="final_reward"),
        decision(UMAZE_KEY, "AntMaze_UMaze-v5_start_bottom_left", "survivor",
                 n=100, mean=-650.0, std=5.0, bar=-690.0, sha=s, rule="whole_run_mean"),
    ])
    violations, seen = chk.check(SWEEP, n_required=30, n_target=100)
    assert violations == []
    assert seen == {POINTMAZE_KEY: "truncated", UMAZE_KEY: "survivor"}


def test_a_decision_below_the_seed_floor_is_caught(tree):
    s = sha_of(tree)
    write_log(tree, [decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                              n=12, mean=1.0, std=0.5, bar=40.0, sha=s, rule="final_reward")])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("below the floor" in v for v in violations)


def test_a_truncation_that_does_not_satisfy_the_inequality_is_caught(tree):
    """Recorded as truncated while its upper bound sits ABOVE the bar."""
    s = sha_of(tree)
    write_log(tree, [decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                              n=40, mean=55.0, std=1.0, bar=40.0, sha=s, rule="final_reward")])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any(">= bar" in v for v in violations)


def test_a_survivor_that_should_have_been_truncated_is_caught(tree):
    s = sha_of(tree)
    write_log(tree, [decision(UMAZE_KEY, "AntMaze_UMaze-v5_start_bottom_left", "survivor",
                              n=100, mean=-900.0, std=1.0, bar=-690.0, sha=s,
                              rule="whole_run_mean")])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("survivor but upper bound" in v for v in violations)


def test_a_tampered_upper_bound_is_caught(tree):
    """A decision line whose recorded upper bound does not follow from its own n, mean and std."""
    s = sha_of(tree)
    d = decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                 n=30, mean=1.0, std=0.5, bar=40.0, sha=s, rule="final_reward")
    d["upper_99"] = -999.0
    write_log(tree, [d])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("recomputed" in v for v in violations)


def test_a_changed_bars_file_is_caught(tree):
    """The bars are frozen: a decision citing a different sha256 means the file moved mid-run."""
    write_log(tree, [decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                              n=30, mean=1.0, std=0.5, bar=40.0, sha="0" * 64,
                              rule="final_reward")])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("the bars changed mid-run" in v for v in violations)


def test_the_wrong_score_rule_is_caught(tree):
    """An AntMaze decision recorded under the PointMaze rule."""
    s = sha_of(tree)
    write_log(tree, [decision(UMAZE_KEY, "AntMaze_UMaze-v5_start_bottom_left", "truncated",
                              n=30, mean=-900.0, std=1.0, bar=-690.0, sha=s,
                              rule="final_reward")])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("expected 'whole_run_mean'" in v for v in violations)


def test_deciding_the_same_configuration_twice_is_caught(tree):
    s = sha_of(tree)
    d = decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                 n=30, mean=1.0, std=0.5, bar=40.0, sha=s, rule="final_reward")
    write_log(tree, [d, dict(d)])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("decided twice" in v for v in violations)


def test_pending_markers_left_behind_by_a_truncation_are_caught(tree):
    """A truncated configuration must have had every pending marker moved to pruned/."""
    s = sha_of(tree)
    write_log(tree, [decision(POINTMAZE_KEY, "initial_single_large_pointmaze_max_400", "truncated",
                              n=30, mean=1.0, std=0.5, bar=40.0, sha=s, rule="final_reward")])
    marker = tree / "queue" / SWEEP / "pending" / "0000_of_4500_leftover_seed900.json"
    marker.write_text(json.dumps({"config_key": POINTMAZE_KEY}))
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("still pending" in v for v in violations)


def test_a_decision_on_an_unknown_configuration_is_caught(tree):
    s = sha_of(tree)
    write_log(tree, [decision("SomeMaze|origsmall|lr0.001|b1", "AntMaze_UMaze-v5_start_bottom_left",
                              "truncated", n=30, mean=-900.0, std=1.0, bar=-690.0, sha=s,
                              rule="whole_run_mean")])
    violations, _ = chk.check(SWEEP, 30, 100)
    assert any("not one of this run's" in v for v in violations)


def test_the_checker_knows_this_run_has_ninety_configurations():
    """A guard on the checker's own idea of the sweep size: 45 per predictor learning rate."""
    assert len(bq.CONFIGS) == 90
    assert len({c["lr"] for c in bq.CONFIGS}) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
