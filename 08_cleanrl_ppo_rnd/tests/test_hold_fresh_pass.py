"""Unit tests for the pass that keeps the pending queue stocked with finishable runs.

The pass decides which runs a freed worker slot is allowed to claim, so its two failure modes are
both expensive: holding back a half-finished run parks work that would have completed inside one
segment, and leaving the queue empty makes a starting slot exit for the rest of its job's four days.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from hold_fresh_pass import run_pass, marker_run_id  # noqa: E402

TARGET = 2_000_000_000


def build_sweep(tmp_path, markers, steps_by_run):
    """Lay out a sweep folder with queue markers and one record per run.

    before: markers = {"pending": [1, 2], "hold_fresh": [3]}, steps_by_run = {1: 1.2e9, 2: 0, 3: 9e8}
    after:  <tmp>/queue/pending/{001,002}_of_150.json, <tmp>/queue/hold_fresh/003_of_150.json,
            and <tmp>/data/sweep/local/00N_of_150.json each carrying that run's last step
    """
    for state, ids in markers.items():
        d = tmp_path / "queue" / state
        d.mkdir(parents=True, exist_ok=True)
        for rid in ids:
            (d / f"{rid:03d}_of_150.json").write_text(json.dumps({"run_id": rid}))
    local = tmp_path / "data" / "sweep" / "local"
    local.mkdir(parents=True, exist_ok=True)
    for rid, steps in steps_by_run.items():
        (local / f"{rid:03d}_of_150.json").write_text(json.dumps(
            {"run_id": rid, "train_history": [{"step": int(steps)}]}))
    return str(tmp_path)


def states(sweep):
    """Read back which markers sit in pending and which are held."""
    out = {}
    for state in ("pending", "hold_fresh"):
        d = os.path.join(sweep, "queue", state)
        out[state] = sorted(marker_run_id(n) for n in os.listdir(d)) if os.path.isdir(d) else []
    return out


def test_near_zero_runs_are_held_and_progressed_ones_are_not(tmp_path):
    """Golden path: the queue keeps the runs a slot can finish and parks the ones it cannot."""
    sweep = build_sweep(tmp_path, {"pending": [1, 2, 3]},
                        {1: 1.2e9, 2: 6_553_600, 3: 9e8})     # run 2 is at 0.3% of target
    held, released, pending = run_pass(sweep, TARGET, 0.05, pending_floor=0)
    assert held == [2] and released == []
    assert states(sweep) == {"pending": [1, 3], "hold_fresh": [2]}
    assert pending == [1, 3]


def test_a_requeued_near_zero_run_is_held_on_the_next_pass(tmp_path):
    """The reason this is a pass and not a one-shot: requeue keeps refilling pending.

    Reported by the collaborator 2026-08-10: ten runs at 0.2-0.3% re-entered pending after the first
    hold, leaving a freed slot a 10-in-13 chance of starting one instead of resuming a 46-57% run.
    """
    sweep = build_sweep(tmp_path, {"pending": [1], "hold_fresh": [2]}, {1: 1.2e9, 2: 6_553_600})
    run_pass(sweep, TARGET, 0.05, pending_floor=0)
    # requeue puts a fresh run back into pending behind the first pass
    os.rename(os.path.join(sweep, "queue", "hold_fresh", "002_of_150.json"),
              os.path.join(sweep, "queue", "pending", "002_of_150.json"))
    held, _, pending = run_pass(sweep, TARGET, 0.05, pending_floor=0)
    assert held == [2] and pending == [1]


def test_the_floor_releases_held_runs_rather_than_starve_a_slot(tmp_path):
    """Edge case: an empty queue costs a slot four days, which is worse than starting a fresh run.

    The most-progressed held run is released first, so the floor is met at the least cost.
    """
    sweep = build_sweep(tmp_path, {"pending": [], "hold_fresh": [5, 6, 7]},
                        {5: 1_000_000, 6: 50_000_000, 7: 20_000_000})
    _, released, pending = run_pass(sweep, TARGET, 0.05, pending_floor=2)
    assert released == [6, 7]        # 50M first, then 20M; run 5 at 1M stays held
    assert pending == [6, 7]
    assert states(sweep)["hold_fresh"] == [5]


def test_a_wrongly_held_run_is_released_by_the_next_pass(tmp_path):
    """The pass is idempotent in both directions, so a bad classification is self-correcting."""
    sweep = build_sweep(tmp_path, {"pending": [], "hold_fresh": [9]}, {9: 1.4e9})
    held, released, pending = run_pass(sweep, TARGET, 0.05, pending_floor=0)
    assert held == [] and released == [9] and pending == [9]
