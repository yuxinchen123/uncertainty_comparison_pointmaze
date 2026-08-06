"""Unit tests for the matched-step comparison in analysis/compute_run_metrics.py.

The property under test is the one the whole analysis rests on: a run must contribute to a step only
if it actually reached that step. This campaign will not finish, so runs stop at whatever step their
node's speed allowed. Counting a run at a step it never reached, or dropping one that did, would let
the allocation of fast and slow nodes masquerade as a difference between the arms.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))

from compute_run_metrics import matched_step_comparison  # noqa: E402


def run(arm, series):
    """Build the minimal record shape matched_step_comparison reads."""
    # before: series = [(100, 0.0), (200, 5.0)]
    # after:  {"arm": ..., "_series": [...], "_last_step": 200}
    return {"arm": arm, "_series": series, "_last_step": max(s for s, _ in series)}


def test_a_run_counts_only_at_steps_it_reached():
    """A short run must not appear in the tally for a step it never got to."""
    # arm A: two runs reach step 200; arm B: one reaches 200, one stopped at 100
    records = ([run("A", [(100, 0.0), (200, 7.0)]) for _ in range(3)]
               + [run("B", [(100, 0.0), (200, 0.0)]) for _ in range(2)]
               + [run("B", [(100, 0.0)]) for _ in range(1)])
    curve = matched_step_comparison(records, min_runs_per_arm=1)
    at = {e["step"]: e["arms"] for e in curve}
    assert at[100]["B"]["runs_reaching_this_step"] == 3
    assert at[200]["B"]["runs_reaching_this_step"] == 2, "the run that stopped at 100 was counted"
    assert at[200]["A"]["runs_reaching_this_step"] == 3


def test_a_run_value_is_its_best_row_up_to_that_step():
    """A later collapse must not erase a score the run had already achieved by the step in question."""
    records = [run("A", [(100, 50.0), (200, 0.0)])] + [run("B", [(100, 0.0), (200, 0.0)])]
    curve = matched_step_comparison(records, min_runs_per_arm=1)
    at = {e["step"]: e["arms"] for e in curve}
    assert at[200]["A"]["max"] == 50.0
    assert at[100]["A"]["fraction_scored"] == 1.0
    assert at[200]["B"]["fraction_scored"] == 0.0


def test_a_step_is_reported_only_when_every_arm_still_has_enough_runs():
    """A step where one arm has thinned out is dropped, so no comparison is made on unequal footing."""
    # arm B's runs all stop at 100, so step 200 must not be reported at all
    records = ([run("A", [(100, 0.0), (200, 1.0)]) for _ in range(5)]
               + [run("B", [(100, 0.0)]) for _ in range(5)])
    curve = matched_step_comparison(records, min_runs_per_arm=5)
    assert [e["step"] for e in curve] == [100]


def test_empty_input_returns_an_empty_curve():
    """No records means no curve, rather than an exception at the caller."""
    assert matched_step_comparison([], min_runs_per_arm=1) == []
