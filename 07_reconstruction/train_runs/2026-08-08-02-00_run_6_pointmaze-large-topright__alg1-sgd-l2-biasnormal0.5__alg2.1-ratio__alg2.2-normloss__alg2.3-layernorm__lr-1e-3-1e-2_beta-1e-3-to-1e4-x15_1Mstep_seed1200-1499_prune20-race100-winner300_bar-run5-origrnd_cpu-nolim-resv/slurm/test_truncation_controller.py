#!/usr/bin/env python
"""Tests for the two-phase truncation controller, on a synthetic queue + records tree.

Covered: the phase-1 rule fires only from the 20-seed floor and only when the 99% upper bound is
under the bar; candidacy at 100; the per-arm winner decision waits for every sibling, crowns the
best mean, stops the others and prunes only their seed>=100 tail; winner completion at 300; and
restart-safety (a second cycle changes nothing).

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_truncation_controller.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue as bq            # noqa: E402
import truncation_controller as tc  # noqa: E402

SWEEP = "test-sweep"
BAR = 38.6412


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A miniature run tree: queue dirs, a data/local dir, and the REAL bars file symlinked in."""
    root = str(tmp_path)
    monkeypatch.setattr(tc, "RUN_DIR", root)
    monkeypatch.setattr(tc, "HERE", os.path.join(root, "slurm"))
    os.makedirs(os.path.join(root, "slurm"))
    # the real bars file, so BAR here matches the controller's bar exactly
    src = os.path.join(HERE, "FROZEN_BARS.json")
    with open(src) as fh, open(os.path.join(root, "slurm", "FROZEN_BARS.json"), "w") as out:
        out.write(fh.read())
    for sub in ("pending", "running", "done", "failed", "pruned"):
        os.makedirs(os.path.join(root, "queue", SWEEP, sub))
    os.makedirs(os.path.join(root, "data", SWEEP, "local"))
    return root


def put_marker(root, cfg_spec, seed_index):
    """One pending marker named exactly as build_queue names it."""
    cfg_index = bq.CONFIGS.index(cfg_spec)
    run_id = seed_index * len(bq.CONFIGS) + cfg_index
    name = (f"{run_id:0{len(str(bq.RUN_TOTAL))}d}_of_{bq.RUN_TOTAL}_"
            f"{bq.label(cfg_spec)}_seed{bq.a_seed_of(seed_index)}.json")
    path = os.path.join(root, "queue", SWEEP, "pending", name)
    with open(path, "w") as fh:
        json.dump({"config_key": bq.config_key(cfg_spec), "seed_index": seed_index}, fh)


def put_record(root, cfg_spec, seed_index, score):
    """One COMPLETED per-run record whose final_reward is `score`."""
    cfg_index = bq.CONFIGS.index(cfg_spec)
    run_id = seed_index * len(bq.CONFIGS) + cfg_index
    rec = {"completed": True, "total_timesteps": bq.STEPS,
           "env_setup": cfg_spec["env_setup"], "beta": float(cfg_spec["beta"]),
           "rnd_lr": float(cfg_spec["lr"]),
           **{k: v for k, v in cfg_spec["params"].items()
              if k in ("rnd_readout_norm_init", "rnd_predictor_loss", "rnd_layer_norm")},
           "train_history": [{"step": bq.STEPS, "train/mean_extrinsic_reward": score}]}
    path = os.path.join(root, "data", SWEEP, "local", f"{run_id}_of_{bq.RUN_TOTAL}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh)


def decisions(root):
    """{config_key: latest verdict} from the log the controller wrote."""
    path = os.path.join(root, "slurm", f"truncation_decisions_{SWEEP}.jsonl")
    out = {}
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            out[d["config_key"]] = d["verdict"]
    return out


CFG_A, CFG_B = bq.CONFIGS[0], bq.CONFIGS[1]      # two alg1 configurations
CFG_OTHER_ARM = bq.CONFIGS[30]                    # an alg2.1 configuration


def test_no_decision_below_the_20_seed_floor(tree):
    """19 seeds far below the bar: no verdict yet."""
    for i in range(19):
        put_record(tree, CFG_A, i, 1.0)
    tc.run_cycle(SWEEP, 20, 100, 300)
    assert decisions(tree) == {}


def test_phase1_truncates_a_config_whose_upper_bound_is_under_the_bar(tree):
    """20 identical low scores -> upper bound = mean < bar -> truncated, markers pruned."""
    for i in range(20):
        put_record(tree, CFG_A, i, 1.0)
    for i in range(20, 25):
        put_marker(tree, CFG_A, i)
    tc.run_cycle(SWEEP, 20, 100, 300)
    assert decisions(tree)[bq.config_key(CFG_A)] == "truncated"
    assert os.listdir(os.path.join(tree, "queue", SWEEP, "pending")) == []


def test_phase1_keeps_a_config_whose_upper_bound_clears_the_bar(tree):
    """20 scores straddling the bar with a wide spread: upper bound above it -> undecided."""
    for i in range(20):
        put_record(tree, CFG_A, i, BAR - 10 + 20 * (i % 2))
    tc.run_cycle(SWEEP, 20, 100, 300)
    assert decisions(tree) == {}


def test_candidacy_at_100_and_no_winner_while_a_sibling_races(tree):
    """A 100-seed above-bar config becomes candidate; no winner while its arm has an undecided
    sibling."""
    for i in range(100):
        put_record(tree, CFG_A, i, BAR + 5)
    tc.run_cycle(SWEEP, 20, 100, 300)
    d = decisions(tree)
    assert d[bq.config_key(CFG_A)] == "candidate"
    assert "winner" not in d.values()


def test_winner_per_arm_once_every_sibling_is_resolved(tree):
    """Arm alg1: every config truncated except two candidates; the better mean wins, the other
    stops at 100 and only its seed>=100 markers are pruned."""
    alg1 = [c for c in bq.CONFIGS if c["arm"] == "alg1"]
    for cfg in alg1:
        if cfg in (CFG_A, CFG_B):
            continue
        for i in range(20):
            put_record(tree, cfg, i, 1.0)          # truncated
    for i in range(100):
        put_record(tree, CFG_A, i, BAR + 8)        # the better candidate
        put_record(tree, CFG_B, i, BAR + 4)        # the worse candidate
    put_marker(tree, CFG_B, 50)                    # a not-yet-run seed BELOW 100: must survive
    put_marker(tree, CFG_B, 150)                   # a tail seed: must be pruned
    put_marker(tree, CFG_A, 150)                   # the winner's tail: must survive
    tc.run_cycle(SWEEP, 20, 100, 300)
    d = decisions(tree)
    assert d[bq.config_key(CFG_A)] == "winner"
    assert d[bq.config_key(CFG_B)] == "stopped_at_100"
    left = os.listdir(os.path.join(tree, "queue", SWEEP, "pending"))
    tags = {n.split("_of_")[0] for n in left}
    assert len(left) == 2   # CFG_B seed 50 and CFG_A seed 150 remain
    assert any(bq.label(CFG_B) in n and "seed1250" in n for n in left)
    assert any(bq.label(CFG_A) in n and "seed1350" in n for n in left)


def test_winner_completes_at_300(tree):
    """The crowned winner reaches 300 completed seeds -> winner_complete."""
    alg1 = [c for c in bq.CONFIGS if c["arm"] == "alg1"]
    for cfg in alg1:
        if cfg is CFG_A:
            continue
        for i in range(20):
            put_record(tree, cfg, i, 1.0)
    for i in range(300):
        put_record(tree, CFG_A, i, BAR + 8)
    tc.run_cycle(SWEEP, 20, 100, 300)
    assert decisions(tree)[bq.config_key(CFG_A)] == "winner_complete"


def test_a_second_cycle_changes_nothing(tree):
    """Restart-safety: re-running the controller adds no decision lines."""
    for i in range(20):
        put_record(tree, CFG_A, i, 1.0)
    tc.run_cycle(SWEEP, 20, 100, 300)
    log = os.path.join(tree, "slurm", f"truncation_decisions_{SWEEP}.jsonl")
    n1 = len(open(log).readlines())
    tc.run_cycle(SWEEP, 20, 100, 300)
    assert len(open(log).readlines()) == n1


def test_arms_decide_independently(tree):
    """A resolved alg1 does not touch alg2.1's racing configurations."""
    alg1 = [c for c in bq.CONFIGS if c["arm"] == "alg1"]
    for cfg in alg1:
        for i in range(20):
            put_record(tree, cfg, i, 1.0)
    for i in range(50):
        put_record(tree, CFG_OTHER_ARM, i, BAR + 5)   # racing, above bar, below 100
    tc.run_cycle(SWEEP, 20, 100, 300)
    d = decisions(tree)
    assert all(d[bq.config_key(c)] == "truncated" for c in alg1)
    assert bq.config_key(CFG_OTHER_ARM) not in d


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
