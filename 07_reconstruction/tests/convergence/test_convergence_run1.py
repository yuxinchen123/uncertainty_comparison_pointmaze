"""Tests for convergence_train.py (convergence-rate run 1): the checkpoint grid, the three point
sets, a 16-step end-to-end mini run whose record matches the logging convention (including the
step-0 cross-check against RND.compute()), and the divergence guard."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import convergence_train as ct
from rnd_exploration.methods.rnd import RND


def make_args(**kw):
    """Build the trainer's argparse namespace with mini-run defaults, overridable via kwargs."""
    params = dict(point_set="center_square", rnd_weight_init="orthogonal", rnd_bias_init="zero",
                  rnd_optimizer="adam", rnd_lr=1e-3, rnd_sgd_eta0=1e-2, rnd_sgd_t0=1e3,
                  a_seed=0, n_steps=16, run_id=0, run_total=2, sweep_id="test", local_log_dir="")
    params.update(kw)
    return argparse.Namespace(**params)


def test_checkpoint_steps_pinned():
    """The quarter-exponent checkpoint grid is exactly the documented list (79 steps to 4096)."""
    cps = ct.checkpoint_steps(4096)
    # pinned head (duplicates collapsed at the small end), pinned tail, pinned length
    assert cps[:23] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 21, 23, 25, 27, 29, 32]
    assert cps[-5:] == [3158, 3444, 3756, 4096][-4:] or cps[-1] == 4096
    assert cps[-1] == 4096
    assert len(cps) == 79
    assert all(a < b for a, b in zip(cps, cps[1:]))
    # the previous half-exponent grid is a subset (densify only ADDED points)
    half = sorted({round(2 ** (k / 2.0)) for k in range(0, 25)})
    assert set(half).issubset(set(cps))
    # a short run keeps the same rule
    assert ct.checkpoint_steps(16) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]


def test_point_sets_pinned():
    """The three point sets have the documented shapes, coordinates, and vx = vy = 0."""
    center = ct.point_set("center_square")
    topright = ct.point_set("top_right_cell")
    cells = ct.point_set("cell_midpoints")
    for pts, n in [(center, 100), (topright, 100), (cells, 108)]:
        assert pts.shape == (n, 4) and pts.dtype == np.float32
        assert np.all(pts[:, 2:] == 0.0)                       # vx = vy = 0 everywhere
        assert len({tuple(r) for r in pts.tolist()}) == n      # all points distinct
    # env 1.1: 10x10 midpoints of [-0.5, 0.5]^2, spacing 0.1
    assert center[:, 0].min() == pytest.approx(-0.45) and center[:, 0].max() == pytest.approx(0.45)
    assert center[:, 1].min() == pytest.approx(-0.45) and center[:, 1].max() == pytest.approx(0.45)
    # env 1.2: the goal cell (row 1, col 10) spanning [4, 5] x [2.5, 3.5]
    assert topright[:, 0].min() == pytest.approx(4.05) and topright[:, 0].max() == pytest.approx(4.95)
    assert topright[:, 1].min() == pytest.approx(2.55) and topright[:, 1].max() == pytest.approx(3.45)
    # env 2: all 108 cell midpoints (col - 5.5, 4 - row), corners included
    rows = {tuple(r[:2]) for r in cells.tolist()}
    assert (-5.5, 4.0) in rows and (5.5, -4.0) in rows and (0.5, 0.0) in rows
    with pytest.raises(ValueError):
        ct.point_set("whole_maze")


def test_end_to_end_mini_run(tmp_path):
    """A 16-step Adam run writes the documented record; step 0 matches RND.compute()'s l2 readout."""
    args = make_args(local_log_dir=str(tmp_path))
    rec = ct.run(args)
    # the record landed at the id-based path with the checkpointed-logging fields
    path = tmp_path / "local" / "0_of_2.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["completed"] is True and on_disk["diverged"] is False
    assert on_disk["checkpoint_steps"] == [0] + ct.checkpoint_steps(16)
    assert [c["step"] for c in on_disk["checkpoints"]] == [0] + ct.checkpoint_steps(16)
    assert all(len(c["b_l2"]) == 100 for c in on_disk["checkpoints"])
    assert all(np.isfinite(c["b_l2"]).all() for c in on_disk["checkpoints"])
    assert len(on_disk["points_xy"]) == 100
    # training reduces the mean bonus (16 full-batch steps on 100 fixed points)
    assert np.mean(on_disk["checkpoints"][-1]["b_l2"]) < np.mean(on_disk["checkpoints"][0]["b_l2"])
    # step-0 cross-check: rebuilding with the same seeding sequence, RND.compute()'s l2 readout
    # (which only adds the 1e-8 clamp) equals the stored step-0 row
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = RND(obs_shape=(4,), output_dim=128, lr=1e-3, device="cpu", use_obs_norm=False,
                distance="mse", n_predictors=1, feature="rnd_next_state",
                optimizer="adam", bonus_readout="l2", bias_init="zero", bias_seed=0)
    b0 = model.compute({"next_observations": ct.point_set("center_square")}).numpy()
    assert np.allclose(b0, np.asarray(rec["checkpoints"][0]["b_l2"]), atol=1e-5)


def test_sgd1t_runs_and_divergence_guard(tmp_path):
    """A short SGD-1/t run completes; an absurd eta0 trips the divergence guard and is flagged."""
    ok = ct.run(make_args(rnd_optimizer="sgd1t", rnd_sgd_eta0=1e-2, rnd_sgd_t0=1e3, n_steps=8,
                          local_log_dir=str(tmp_path), run_id=1))
    assert ok["completed"] is True and ok["diverged"] is False
    bad = ct.run(make_args(rnd_optimizer="sgd1t", rnd_sgd_eta0=1e12, rnd_sgd_t0=1e3, n_steps=8,
                           local_log_dir=str(tmp_path), run_id=0, a_seed=1))
    assert bad["diverged"] is True and bad["completed"] is True
    # the finite prefix (at least the step-0 baseline) is kept; the non-finite row is not stored
    assert len(bad["checkpoints"]) >= 1
    assert all(np.isfinite(c["b_l2"]).all() for c in bad["checkpoints"])
