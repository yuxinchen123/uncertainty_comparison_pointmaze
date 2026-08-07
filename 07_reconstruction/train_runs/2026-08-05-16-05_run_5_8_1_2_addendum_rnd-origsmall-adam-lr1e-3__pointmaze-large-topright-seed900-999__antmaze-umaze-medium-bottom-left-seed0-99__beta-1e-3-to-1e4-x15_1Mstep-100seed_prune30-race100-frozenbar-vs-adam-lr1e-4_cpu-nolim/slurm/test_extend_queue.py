#!/usr/bin/env python
"""Tests for extend_queue.py — growing a LIVE queue without repeating or disturbing any run.

This script runs against a sweep with hundreds of workers claiming from the same directory, so the
tests are about the two properties that make that safe rather than about counts:
  1. work that is finished or in flight is never touched, so no completed run is ever repeated;
  2. a (configuration, seed) never exists twice in pending/, so no unit is ever claimed twice.
Plus the operational ones: re-running changes nothing, a crash mid-rename is repaired by re-running,
and the resulting names sort in run-id order (which is what makes the claim window seed-ordered).

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_extend_queue.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue as bq        # noqa: E402
import extend_queue as eq       # noqa: E402

SWEEP = "test-sweep"


def old_scheme_units():
    """The (config_key, seed_index) pairs the pre-extension queue held: Adam 1e-3, seeds 0..99."""
    lr3 = [c for c in bq.CONFIGS if c["lr"] == "0.001"]
    assert len(lr3) == 45
    return [(bq.config_key(c), s) for s in range(100) for c in lr3], lr3


def build_old_queue(root, pools_for_seed):
    """Write a pre-extension queue (45 Adam 1e-3 configurations x 100 seeds, 4-digit ids).

    `pools_for_seed(seed_index)` says which pool that seed's markers go in, so a test can place some
    seeds in done/ or running/ exactly as the live sweep had them.
    """
    queue = os.path.join(root, "queue", SWEEP)
    for sub in eq.POOLS + (eq.STAGING,):
        os.makedirs(os.path.join(queue, sub), exist_ok=True)
    _, lr3 = old_scheme_units()
    run_total_old = 4500
    for seed_index in range(100):
        pool = pools_for_seed(seed_index)
        for cfg_index, cfg_spec in enumerate(lr3):
            run_id = seed_index * 45 + cfg_index
            seed = bq.a_seed_of(cfg_spec["env_setup"], seed_index)
            cfg = {"sweep_id": SWEEP, "run_id": run_id, "run_total": run_total_old, "pool": pool,
                   "env_setup": cfg_spec["env_setup"], "algorithm": cfg_spec["algorithm"],
                   "arm": cfg_spec["arm"], "beta": cfg_spec["beta"],
                   "a_seed": seed, "seed_index": seed_index,
                   "config_key": bq.config_key(cfg_spec),
                   "params": cfg_spec["params"], "fixed": dict(bq.FIXED_COMMON)}
            name = f"{run_id:04d}_of_{run_total_old}_{bq.label(cfg_spec)}_seed{seed}.json"
            with open(os.path.join(queue, pool, name), "w") as fh:
                json.dump(cfg, fh)
    return queue


@pytest.fixture
def live_queue(tmp_path, monkeypatch):
    """A queue shaped like the real one on 2026-08-06: seeds 0-18 done, 19-20 running, 21-99 pending."""
    root = str(tmp_path)
    monkeypatch.setattr(eq, "RUN_DIR", root)

    def pool_of(seed_index):
        if seed_index < 19:
            return "done"
        if seed_index < 21:
            return "running"
        return "pending"

    return build_old_queue(root, pool_of), root


def pool_names(queue, pool):
    """The marker filenames in one pool."""
    return sorted(os.listdir(os.path.join(queue, pool)))


def units_in(queue, pool):
    """The (config_key, seed_index) pairs held by one pool."""
    d = os.path.join(queue, pool)
    return [eq.marker_unit(os.path.join(d, n)) for n in os.listdir(d)]


def test_finished_and_in_flight_work_is_never_touched(live_queue):
    """The property that keeps a completed run from being repeated: done/ and running/ are inert."""
    queue, _ = live_queue
    before_done, before_running = pool_names(queue, "done"), pool_names(queue, "running")
    eq.extend(SWEEP, apply_changes=True)
    assert pool_names(queue, "done") == before_done
    assert pool_names(queue, "running") == before_running


def test_no_unit_is_ever_queued_twice(live_queue):
    """The property that keeps a unit from being claimed twice: every (config, seed) appears once."""
    queue, _ = live_queue
    eq.extend(SWEEP, apply_changes=True)
    everywhere = [u for pool in eq.POOLS for u in units_in(queue, pool)]
    assert len(everywhere) == len(set(everywhere))
    assert len(everywhere) == bq.RUN_TOTAL   # 90 configurations x 300 seeds, each exactly once


def test_the_whole_extended_sweep_is_present_after_one_pass(live_queue):
    """Both learning rates, all 300 seeds: 27,000 units, and the counts add up."""
    queue, _ = live_queue
    counts = eq.extend(SWEEP, apply_changes=True)
    # before: 4,500 units (45 configurations x 100 seeds); after: 27,000
    assert counts["created"] == bq.RUN_TOTAL - 4500
    assert counts["renamed"] + counts["claimed_mid_rename"] == 79 * 45   # seeds 21..99, still pending
    assert counts["untouched_non_pending"] == 21 * 45                    # seeds 0..20, done/running
    assert counts["unknown_in_queue"] == 0


def test_the_new_arm_is_the_same_stack_at_the_new_learning_rate(live_queue):
    """Adam 1e-2 differs from Adam 1e-3 in rnd_lr and nothing else, and shares its seeds."""
    queue, _ = live_queue
    eq.extend(SWEEP, apply_changes=True)
    by_lr = {}
    for name in os.listdir(os.path.join(queue, "pending")):
        d = json.load(open(os.path.join(queue, "pending", name)))
        by_lr.setdefault(d["params"]["rnd_lr"], []).append(d)
    assert set(by_lr) == {"0.001", "0.01"}
    a = next(d for d in by_lr["0.001"] if d["beta"] == "1000" and d["seed_index"] == 50)
    b = next(d for d in by_lr["0.01"] if d["beta"] == "1000" and d["seed_index"] == 50
             and d["env_setup"] == a["env_setup"])
    assert a["a_seed"] == b["a_seed"]           # the two arms are paired seed by seed
    assert a["fixed"] == b["fixed"]
    differing = {k for k in a["params"] if a["params"][k] != b["params"][k]}
    assert differing == {"rnd_lr"}


def test_running_it_again_changes_nothing(live_queue):
    """Idempotent: the second pass creates nothing, renames nothing, and leaves the files identical."""
    queue, _ = live_queue
    eq.extend(SWEEP, apply_changes=True)
    snapshot = {pool: pool_names(queue, pool) for pool in eq.POOLS}
    counts = eq.extend(SWEEP, apply_changes=True)
    assert counts["created"] == 0 and counts["renamed"] == 0
    assert counts["already"] == len(snapshot["pending"])
    assert {pool: pool_names(queue, pool) for pool in eq.POOLS} == snapshot


def test_names_sort_in_run_id_order(live_queue):
    """What makes claim()'s 32-name window a seed-ordered frontier: lexical order IS numeric order."""
    queue, _ = live_queue
    eq.extend(SWEEP, apply_changes=True)
    names = pool_names(queue, "pending")
    ids = [json.load(open(os.path.join(queue, "pending", n)))["run_id"] for n in names]
    assert ids == sorted(ids)
    # and the lexically smallest markers are the lowest seeds of the arm that is behind
    first = json.load(open(os.path.join(queue, "pending", names[0])))
    assert first["seed_index"] == 0 and first["params"]["rnd_lr"] == "0.01"


def test_a_crash_mid_rename_is_repaired_by_re_running(live_queue):
    """A marker stranded in staging/ goes back to pending/ before anything else happens."""
    queue, _ = live_queue
    eq.extend(SWEEP, apply_changes=True)
    victim = pool_names(queue, "pending")[0]
    os.rename(os.path.join(queue, "pending", victim), os.path.join(queue, eq.STAGING, victim))
    assert victim not in pool_names(queue, "pending")
    counts = eq.extend(SWEEP, apply_changes=True)
    assert counts["staging_drained"] == 1
    assert victim in pool_names(queue, "pending")
    assert os.listdir(os.path.join(queue, eq.STAGING)) == []


def test_a_dry_run_changes_nothing(live_queue):
    """The default mode reports the same work it would do, without touching the queue."""
    queue, _ = live_queue
    snapshot = {pool: pool_names(queue, pool) for pool in eq.POOLS}
    counts = eq.extend(SWEEP, apply_changes=False)
    assert counts["created"] == bq.RUN_TOTAL - 4500
    assert counts["renamed"] == 79 * 45
    assert {pool: pool_names(queue, pool) for pool in eq.POOLS} == snapshot


def test_a_truncated_configuration_is_not_resurrected(tmp_path, monkeypatch):
    """A configuration already pruned keeps its pruned seeds, and only its unqueued seeds are added.

    The controller prunes by moving pending markers to pruned/; extend_queue must not undo that for
    the seeds already decided, and must not skip the seeds the controller never saw.
    """
    root = str(tmp_path)
    monkeypatch.setattr(eq, "RUN_DIR", root)
    queue = build_old_queue(root, lambda s: "pruned" if s >= 30 else "done")
    before_pruned = pool_names(queue, "pruned")
    eq.extend(SWEEP, apply_changes=True)
    assert pool_names(queue, "pruned") == before_pruned      # nothing re-pended
    pending_units = set(units_in(queue, "pending"))
    lr3_keys = {bq.config_key(c) for c in bq.CONFIGS if c["lr"] == "0.001"}
    # seeds 0..99 of the old arm are accounted for (done or pruned); only 100..299 are new
    assert not any(k in lr3_keys and s < 100 for k, s in pending_units)
    assert all((k, 150) in pending_units for k in lr3_keys)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
