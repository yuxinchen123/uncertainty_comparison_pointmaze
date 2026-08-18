"""Unit tests for the fixed harness: point sets, checkpoint grid, fitting, metrics.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest tests/ -q   (from code/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decay_harness.fitting import fit_power_floor
from decay_harness.metrics import compute_metrics, deviation_per_position, target_curve
from decay_harness.points import checkpoint_steps, nonuniform_probabilities, point_set


def test_checkpoint_grid_matches_prior_work():
    """The grid must reproduce convergence run 1's: 79 nonzero steps for 4096, ending at 4096."""
    cps = checkpoint_steps(4096)
    assert len(cps) == 79 and cps[0] == 1 and cps[-1] == 4096
    assert cps == sorted(set(cps))  # strictly increasing, deduplicated


def test_point_sets_shapes_and_geometry():
    """Golden path: sizes and the documented world coordinates of each set's corners."""
    cm = point_set("cell_midpoints")
    assert cm.shape == (108, 4)
    assert cm[:, 2:].sum() == 0  # vx = vy = 0 everywhere
    assert {(-5.5, 4.0), (5.5, -4.0)} <= {(float(x), float(y)) for x, y in cm[:, :2]}
    cs = point_set("center_square")
    assert cs.shape == (100, 4) and abs(cs[:, 0]).max() <= 0.45 + 1e-9
    # edge case: unknown name raises
    try:
        point_set("nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_nonuniform_probabilities_decade_span():
    """The sampling law spans one decade across x and sums to 1."""
    pts = point_set("cell_midpoints")
    pr = nonuniform_probabilities(pts)
    assert abs(pr.sum() - 1.0) < 1e-12
    left = pr[pts[:, 0] == pts[:, 0].min()].mean()
    right = pr[pts[:, 0] == pts[:, 0].max()].mean()
    assert abs(left / right - 10.0) < 1e-6


def test_fit_recovers_known_power_law():
    """Golden path: a clean floored power law is recovered to two decimals."""
    steps = np.array([0] + checkpoint_steps(4096))
    y = 0.01 + 1.0 * (steps + 2.0) ** -0.5
    fit = fit_power_floor(steps, y)
    assert abs(fit["alpha"] - 0.5) < 0.01 and abs(fit["c"] - 0.01) < 0.005


def _record_from_curves(bonus, counts, env="cell_midpoints", seed=0):
    """Assemble a metric-ready record from (T, P) bonus and visit-count arrays."""
    steps = [0] + checkpoint_steps(int(np.asarray(counts).max()))
    return {"point_set": env, "a_seed": seed, "diverged": False,
            "checkpoint_steps": steps, "bonus": np.asarray(bonus).tolist(),
            "visit_counts": np.asarray(counts).tolist()}


def test_ideal_curve_scores_zero():
    """A bonus exactly equal to the count oracle gives deviation 0 at every position."""
    steps = np.array([0] + checkpoint_steps(4096))
    counts = np.tile(steps[:, None], (1, 5))  # uniform full batch: m_i(n) = n, 5 positions
    bonus = target_curve(counts)
    rec = _record_from_curves(bonus, counts)
    assert np.allclose(deviation_per_position(rec), 0.0)
    m = compute_metrics([rec], fit_slopes=False)
    assert m["dev_worst"] == 0.0 and m["start_dev"] == 0.0


def test_one_bad_position_dominates_dev_worst():
    """A single off-rate position must show up in dev_worst but barely in dev_mean."""
    steps = np.array([0] + checkpoint_steps(4096))
    counts = np.tile(steps[:, None], (1, 10))
    bonus = target_curve(counts).copy()
    with np.errstate(divide="ignore"):
        bonus[:, 0] = np.minimum(1.0, np.where(steps > 0, steps, 1.0) ** -1.0)  # decays as 1/n
    m = compute_metrics([_record_from_curves(bonus, counts)], fit_slopes=False)
    assert m["dev_worst"] > 1.0 and m["dev_mean"] < 0.5


def test_nonfinite_bonus_marks_position_inf():
    """Edge case: a NaN checkpoint makes that position's deviation inf, not a quiet number."""
    steps = np.array([0] + checkpoint_steps(64))
    counts = np.tile(steps[:, None], (1, 3))
    bonus = target_curve(counts).copy()
    bonus[5, 1] = np.nan
    dev = deviation_per_position(_record_from_curves(bonus, counts))
    assert np.isinf(dev[1]) and np.isfinite(dev[[0, 2]]).all()


def test_diverged_records_counted_not_scored():
    """Edge case: a diverged record is excluded from scores but counted."""
    steps = np.array([0] + checkpoint_steps(64))
    counts = np.tile(steps[:, None], (1, 3))
    good = _record_from_curves(target_curve(counts), counts)
    bad = dict(good, diverged=True)
    m = compute_metrics([good, bad], fit_slopes=False)
    assert m["n_diverged"] == 1 and m["dev_worst"] == 0.0
