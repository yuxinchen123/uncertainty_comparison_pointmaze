#!/usr/bin/env python
"""Tests for the truncation controller and the two score rules, on synthetic records.

Covers the golden path (a clearly-worse configuration is truncated, a clearly-better one survives)
and the edge cases that would silently corrupt a sweep: deciding below the seed floor, using the
wrong environment's score rule, deciding twice, and scoring a partial record.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_truncation_controller.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue as bq              # noqa: E402
import score_rules                    # noqa: E402
import truncation_controller as tc    # noqa: E402

POINTMAZE = "initial_single_large_pointmaze_max_400"
UMAZE = "AntMaze_UMaze-v5_start_bottom_left"
SWEEP = "test-sweep"

# bars chosen so the arithmetic in the tests is obvious, not the real frozen values
BARS = {POINTMAZE: {"mean": 40.0, "beta": "1000", "score_rule": "final_reward"},
        UMAZE: {"mean": -690.0, "beta": "10000", "score_rule": "whole_run_mean"},
        "AntMaze_Medium-v5_start_bottom_left": {"mean": -998.0, "beta": "3000",
                                                "score_rule": "whole_run_mean"}}


def make_record(env, beta, final, episodes):
    """One synthetic per-run record: `final` is the last windowed reward (train run 5's score) and
    `episodes` the per-episode returns (train run 1.2's score comes from their mean)."""
    return {
        "completed": True, "total_timesteps": bq.STEPS, "env_setup": env,
        "rnd_lr": 0.001, "beta": float(beta),
        "train_history": [{"step": 1000000, "train/mean_extrinsic_reward": final}],
        "train_episode_history": [{"train/extrinsic_reward": v} for v in episodes],
    }


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """A temporary run tree with the queue, the data dir and a bars file the controller can read."""
    monkeypatch.setattr(tc, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(tc, "HERE", str(tmp_path / "slurm"))
    (tmp_path / "slurm").mkdir()
    (tmp_path / "slurm" / "FROZEN_BARS.json").write_text(json.dumps({"schema": 1, "bars": BARS}))
    for sub in ("pending", "running", "done", "failed", "pruned"):
        (tmp_path / "queue" / SWEEP / sub).mkdir(parents=True)
    (tmp_path / "data" / SWEEP / "local").mkdir(parents=True)
    return tmp_path


def write_records(root, env, beta, n, final, episodes):
    """Write n identical-shaped records for one configuration into the sweep's local data dir."""
    local = root / "data" / SWEEP / "local"
    key = f"{env}|origsmall|lr0.001|b{'%g' % float(beta)}"
    for i in range(n):
        rec = make_record(env, beta, final, episodes)
        (local / f"{abs(hash((key, i))) % 10**9}_of_4500.json").write_text(json.dumps(rec))
    return key


def write_pending(root, env, beta, n):
    """Write n pending markers for one configuration, so a truncation has something to move."""
    spec = next(c for c in bq.CONFIGS
                if c["env_setup"] == env and float(c["beta"]) == float(beta))
    tag = bq.label(spec)
    for i in range(n):
        (root / "queue" / SWEEP / "pending" / f"{i:04d}_of_4500_{tag}_seed{i}.json").write_text(
            json.dumps({"config_key": bq.config_key(spec)}))


def test_score_rule_is_per_environment():
    """PointMaze scores on the final windowed reward; AntMaze on the mean over all episodes."""
    rec = make_record(POINTMAZE, "1000", final=42.0, episodes=[1.0, 3.0])
    assert score_rules.score_of_record(rec) == 42.0
    rec_ant = make_record(UMAZE, "10000", final=-700.0, episodes=[-600.0, -800.0])
    assert score_rules.score_of_record(rec_ant) == -700.0   # mean of the episodes, not `final`
    rec_ant2 = make_record(UMAZE, "10000", final=0.0, episodes=[-600.0, -700.0, -800.0])
    assert score_rules.score_of_record(rec_ant2) == -700.0


def test_unknown_environment_raises_rather_than_guessing():
    """A missing score rule must fail loudly — a guessed rule would corrupt every decision."""
    with pytest.raises(KeyError):
        score_rules.rule_for("SomeMaze-v9_start_top_right")


def test_no_decision_below_the_seed_floor(sweep):
    """29 completed seeds, hopeless mean: still no verdict, because the floor is 30."""
    write_records(sweep, POINTMAZE, "0.001", n=29, final=1.0, episodes=[1.0])
    write_pending(sweep, POINTMAZE, "0.001", n=5)
    assert tc.run_cycle(SWEEP, n_required=30, n_target=100) == []
    assert len(os.listdir(sweep / "queue" / SWEEP / "pending")) == 5


def test_configuration_below_the_bar_is_truncated_and_its_pending_seeds_move(sweep):
    """30 seeds at final reward 1.0 with no spread: upper bound 1.0 < bar 40.0 -> truncated."""
    key = write_records(sweep, POINTMAZE, "0.001", n=30, final=1.0, episodes=[1.0])
    write_pending(sweep, POINTMAZE, "0.001", n=7)
    lines = tc.run_cycle(SWEEP, n_required=30, n_target=100)
    assert len(lines) == 1 and lines[0].startswith("truncated")
    assert os.listdir(sweep / "queue" / SWEEP / "pending") == []
    assert len(os.listdir(sweep / "queue" / SWEEP / "pruned")) == 7
    entry = json.loads((sweep / "slurm" / f"truncation_decisions_{SWEEP}.jsonl").read_text())
    assert entry["verdict"] == "truncated" and entry["config_key"] == key
    assert entry["n"] == 30 and entry["bar"] == 40.0 and entry["pending_moved"] == 7
    assert entry["score_rule"] == "final_reward" and len(entry["bars_sha256"]) == 64


def test_configuration_above_the_bar_is_not_truncated(sweep):
    """A configuration beating the bar keeps its pending seeds and gets no verdict before n_target."""
    write_records(sweep, POINTMAZE, "1000", n=40, final=55.0, episodes=[1.0])
    write_pending(sweep, POINTMAZE, "1000", n=6)
    assert tc.run_cycle(SWEEP, n_required=30, n_target=100) == []
    assert len(os.listdir(sweep / "queue" / SWEEP / "pending")) == 6


def test_survivor_verdict_at_the_seed_target(sweep):
    """100 completed seeds above the bar -> a final survivor verdict, and nothing is moved."""
    write_records(sweep, UMAZE, "30", n=100, final=0.0, episodes=[-650.0])
    write_pending(sweep, UMAZE, "30", n=3)
    lines = tc.run_cycle(SWEEP, n_required=30, n_target=100)
    assert len(lines) == 1 and lines[0].startswith("survivor")
    assert len(os.listdir(sweep / "queue" / SWEEP / "pending")) == 3
    entry = json.loads((sweep / "slurm" / f"truncation_decisions_{SWEEP}.jsonl").read_text())
    assert entry["verdict"] == "survivor" and entry["score_rule"] == "whole_run_mean"


def test_a_decided_configuration_is_never_decided_again(sweep):
    """Restart safety: the second cycle re-reads the decision log and stays silent."""
    write_records(sweep, POINTMAZE, "0.001", n=30, final=1.0, episodes=[1.0])
    write_pending(sweep, POINTMAZE, "0.001", n=4)
    assert len(tc.run_cycle(SWEEP, 30, 100)) == 1
    assert tc.run_cycle(SWEEP, 30, 100) == []
    log = (sweep / "slurm" / f"truncation_decisions_{SWEEP}.jsonl").read_text().strip().split("\n")
    assert len(log) == 1


def test_partial_records_never_enter_a_decision(sweep):
    """completed=false checkpoints of killed attempts are invisible to the controller."""
    local = sweep / "data" / SWEEP / "local"
    for i in range(40):
        rec = make_record(POINTMAZE, "0.001", final=1.0, episodes=[1.0])
        rec["completed"] = False
        (local / f"{i}_of_4500.json").write_text(json.dumps(rec))
    write_pending(sweep, POINTMAZE, "0.001", n=4)
    assert tc.run_cycle(SWEEP, 30, 100) == []
    assert len(os.listdir(sweep / "queue" / SWEEP / "pending")) == 4


def test_the_spread_can_save_a_configuration_whose_mean_is_below_the_bar(sweep):
    """The rule is the 99% UPPER bound, not the mean: a noisy configuration is not truncated."""
    # 30 records alternating 0 and 70 -> mean 35 (below the 40 bar), sd about 35.6,
    # upper bound 35 + 2.576*35.6/sqrt(30) about 51.7, which is above the bar
    local = sweep / "data" / SWEEP / "local"
    for i in range(30):
        rec = make_record(POINTMAZE, "0.01", final=(0.0 if i % 2 else 70.0), episodes=[1.0])
        (local / f"{i}_of_4500.json").write_text(json.dumps(rec))
    write_pending(sweep, POINTMAZE, "0.01", n=5)
    assert tc.run_cycle(SWEEP, 30, 100) == []
    assert len(os.listdir(sweep / "queue" / SWEEP / "pending")) == 5


def test_records_from_another_step_budget_are_ignored(sweep):
    """Defense in depth: a record whose total_timesteps is not this run's is never scored."""
    local = sweep / "data" / SWEEP / "local"
    for i in range(35):
        rec = make_record(POINTMAZE, "0.001", final=1.0, episodes=[1.0])
        rec["total_timesteps"] = 10000000
        (local / f"{i}_of_4500.json").write_text(json.dumps(rec))
    write_pending(sweep, POINTMAZE, "0.001", n=4)
    assert tc.run_cycle(SWEEP, 30, 100) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
