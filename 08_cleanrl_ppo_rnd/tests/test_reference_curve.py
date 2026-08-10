"""Unit tests for the published-reference handling in the interim analysis.

The reference is CleanRL's own run of this file, and the whole point of drawing it beside the arms is
that it is the SAME quantity. They log a trailing mean over 20 episodes; this project logs one over
200. Re-averaging their raw per-episode returns over 200 is what makes the comparison honest, so the
window arithmetic is worth pinning: a wrong window would show a difference in smoothing and invite
reading it as a difference in performance.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "train_runs",
    "2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1"
    "_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv",
    "analysis", "code"))

from make_interim_table_and_plot import load_reference, reference_at_step  # noqa: E402


def write_reference(tmp_path, points):
    """Lay out a sweep folder holding only the reference file load_reference reads."""
    # before: points = [(100, 0.0), (200, 10.0)]
    # after:  <tmp>/analysis/data/reference/cleanrl_ppo_rnd_montezuma.json holding that series
    d = tmp_path / "analysis" / "data" / "reference"
    d.mkdir(parents=True)
    (d / "cleanrl_ppo_rnd_montezuma.json").write_text(json.dumps(
        {"source": "test", "series": {"charts/episodic_return": points}}))
    return str(tmp_path)


def test_the_running_mean_uses_the_requested_window(tmp_path):
    """With a window of 3, each point is the mean of the last three episode returns."""
    pts = [(10, 0.0), (20, 3.0), (30, 6.0), (40, 9.0), (50, 12.0)]
    ref = load_reference(write_reference(tmp_path, pts), window=3)
    # the first two points average over fewer than three, because that is all that has happened
    assert [v for _, v in ref["points"]] == [0.0, 1.5, 3.0, 6.0, 9.0]
    assert ref["window"] == 3


def test_a_wider_window_smooths_a_spike_more(tmp_path):
    """The window is the whole reason the reference is comparable, so its effect is pinned."""
    pts = [(i * 10, 0.0) for i in range(10)] + [(100, 1000.0)]
    narrow = load_reference(write_reference(tmp_path / "a", pts), window=2)
    wide = load_reference(write_reference(tmp_path / "b", pts), window=10)
    assert narrow["points"][-1][1] == pytest.approx(500.0)
    assert wide["points"][-1][1] == pytest.approx(100.0)


def test_the_series_is_sorted_by_step_before_averaging(tmp_path):
    """wandb rows arrive unordered; a trailing mean over an unsorted series would be meaningless."""
    ref = load_reference(write_reference(tmp_path, [(30, 6.0), (10, 0.0), (20, 3.0)]), window=3)
    assert [s for s, _ in ref["points"]] == [10, 20, 30]
    assert ref["points"][-1][1] == pytest.approx(3.0)


def test_reading_at_a_step_takes_the_last_point_at_or_before_it(tmp_path):
    """The reference is read at the same step as the arms, never interpolated past its own data."""
    ref = load_reference(write_reference(tmp_path, [(10, 1.0), (20, 1.0), (30, 1.0)]), window=1)
    assert reference_at_step(ref, 25) == pytest.approx(1.0)   # falls back to the step-20 row
    assert reference_at_step(ref, 5) is None                  # before the run started, no value


def test_a_missing_reference_file_is_not_an_error(tmp_path):
    """The figure and table must still build in a checkout that has not pulled the reference."""
    (tmp_path / "analysis").mkdir()
    assert load_reference(str(tmp_path)) is None
