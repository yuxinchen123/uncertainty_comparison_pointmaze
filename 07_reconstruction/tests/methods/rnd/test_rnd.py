"""Tests for rnd_exploration.methods.rnd (RND intrinsic reward model).

Covers: compute() output shape; _get_feature_tensor column slicing per
FEATURE_CHOICES mode; single vs ensemble predictor construction; linear_rnd
construction; mse vs abs distance; and that update() reduces the bonus when
trained repeatedly on the same batch. Small shapes / few samples keep it fast.
"""
import numpy as np
import pytest
import torch

from rnd_exploration.methods.rnd import (
    RND,
    FEATURE_CHOICES,
    ObservationEncoder,
    EnsembleObservationEncoder,
)

OBS_DIM = 4
ACTION_DIM = 2
N = 8


def make_samples(n=N, obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=0):
    """Build a samples dict (observations / next_observations / actions) of float32 numpy arrays."""
    # Deterministic random batch so distances are non-trivial but reproducible.
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "next_observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions": rng.standard_normal((n, action_dim)).astype(np.float32),
    }


def make_rnd(feature="rnd_next_state", **kw):
    """Construct an RND with small obs/action dims on CPU, overridable via kwargs."""
    # output_dim kept small to stay fast; action_dim wired so action features have a width.
    params = dict(
        obs_shape=(OBS_DIM,),
        output_dim=16,
        device="cpu",
        action_dim=ACTION_DIM,
        feature=feature,
    )
    params.update(kw)
    return RND(**params)


# Maps each feature mode to its expected input width given OBS_DIM / ACTION_DIM.
EXPECTED_INPUT_DIM = {
    "rnd_next_state": OBS_DIM,
    "rnd_next_state_position_only": 2,
    "rnd_state": OBS_DIM,
    "rnd_state_action": OBS_DIM + ACTION_DIM,
    "rnd_state_action_next_state": OBS_DIM + ACTION_DIM + OBS_DIM,
}


def test_get_feature_tensor_slices_per_feature():
    """_get_feature_tensor selects the right source columns/values for each FEATURE_CHOICES mode."""
    samples = make_samples()
    obs = torch.as_tensor(samples["observations"])
    next_obs = torch.as_tensor(samples["next_observations"])
    actions = torch.as_tensor(samples["actions"])
    # Golden path: every feature mode yields the documented shape and exact column values.
    for feature in FEATURE_CHOICES:
        rnd = make_rnd(feature=feature)
        x = rnd._get_feature_tensor(samples)
        assert x.shape == (N, EXPECTED_INPUT_DIM[feature]), feature
        assert x.dtype == torch.float32
        if feature == "rnd_next_state":
            assert torch.allclose(x, next_obs)
        elif feature == "rnd_next_state_position_only":
            assert torch.allclose(x, next_obs[:, 0:2])
        elif feature == "rnd_state":
            assert torch.allclose(x, obs)
        elif feature == "rnd_state_action":
            assert torch.allclose(x[:, :OBS_DIM], obs)
            assert torch.allclose(x[:, OBS_DIM:], actions)
        elif feature == "rnd_state_action_next_state":
            assert torch.allclose(x[:, :OBS_DIM], obs)
            assert torch.allclose(x[:, OBS_DIM:OBS_DIM + ACTION_DIM], actions)
            assert torch.allclose(x[:, OBS_DIM + ACTION_DIM:], next_obs)
    # Edge case: position_only must take exactly the first two columns, never more, even though obs has 4.
    rnd_pos = make_rnd(feature="rnd_next_state_position_only")
    x_pos = rnd_pos._get_feature_tensor(samples)
    assert x_pos.shape == (N, 2)
    assert not torch.allclose(x_pos, next_obs[:, 0:2] * 0 + next_obs[:, 2:4])


def test_get_feature_tensor_unsqueezes_1d_actions():
    """state_action handles 1-D actions by unsqueezing them to a single column before concat."""
    # Golden path: a 2-D (N, 1) action array concatenates cleanly to width obs_dim + 1.
    rng = np.random.default_rng(1)
    samples_2d = {
        "observations": rng.standard_normal((N, OBS_DIM)).astype(np.float32),
        "actions": rng.standard_normal((N, 1)).astype(np.float32),
    }
    rnd = make_rnd(feature="rnd_state_action", action_dim=1)
    x2d = rnd._get_feature_tensor(samples_2d)
    assert x2d.shape == (N, OBS_DIM + 1)
    # Edge case: a 1-D (N,) action vector must be unsqueezed to (N, 1) and give the same result.
    samples_1d = dict(samples_2d)
    samples_1d["actions"] = samples_2d["actions"].reshape(N)
    x1d = rnd._get_feature_tensor(samples_1d)
    assert x1d.shape == (N, OBS_DIM + 1)
    assert torch.allclose(x1d, x2d)


def test_compute_output_shape_and_range():
    """compute() returns a per-sample (N,) or (N,1) tensor of finite, non-negative bonuses."""
    # Golden path: default single-predictor RND over a normal-sized batch.
    rnd = make_rnd()
    out = rnd.compute(make_samples())
    assert out.shape == (N,) or out.shape == (N, 1)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    # Edge case: a single-sample batch still yields one finite, non-negative bonus.
    one = make_samples(n=1)
    out1 = rnd.compute(one)
    assert out1.shape == (1,) or out1.shape == (1, 1)
    assert torch.isfinite(out1).all()
    assert (out1 >= 0).all()


def test_single_vs_ensemble_construction_and_compute():
    """n_predictors=1 builds a single encoder; n_predictors>1 builds the batched ensemble encoder."""
    # Golden path: single predictor is an ObservationEncoder and compute returns (N,).
    single = make_rnd(n_predictors=1)
    assert isinstance(single.predictor, ObservationEncoder)
    assert single.n_predictors == 1
    out_single = single.compute(make_samples())
    assert out_single.shape == (N,) or out_single.shape == (N, 1)
    # Ensemble: predictor is EnsembleObservationEncoder; its raw forward stacks n_predictors outputs.
    ens = make_rnd(n_predictors=5)
    assert isinstance(ens.predictor, EnsembleObservationEncoder)
    assert ens.n_predictors == 5
    feat = ens._get_feature_tensor(make_samples())
    raw = ens.predictor(feat)
    assert raw.shape == (N, 5, ens.output_dim)
    out_ens = ens.compute(make_samples())
    assert out_ens.shape == (N,) or out_ens.shape == (N, 1)
    assert torch.isfinite(out_ens).all()
    # Edge case: beta_std>0 changes the ensemble bonus (it adds the per-sample disagreement std).
    samples = make_samples()
    ens_nostd = make_rnd(n_predictors=5, beta_std=0.0)
    ens_std = make_rnd(n_predictors=5, beta_std=10.0)
    # Copy weights so only beta_std differs between the two ensembles being compared.
    ens_std.predictor.load_state_dict(ens_nostd.predictor.state_dict())
    ens_std.target.load_state_dict(ens_nostd.target.state_dict())
    b0 = ens_nostd.compute(samples)
    b1 = ens_std.compute(samples)
    assert not torch.allclose(b0, b1)


def test_linear_rnd_construction():
    """linear_rnd=True builds body/head encoders, freezes body+target, trains only the head, forces n=1."""
    # Golden path: predictor body is copied from (and frozen like) the target; only head is trainable.
    rnd = make_rnd(linear_rnd=True, n_predictors=7)
    assert rnd.linear_rnd is True
    assert rnd.n_predictors == 1  # edge case: requested n_predictors=7 is overridden to 1 for linear RND
    assert rnd.predictor.linear_mode and rnd.target.linear_mode
    # Target is entirely frozen.
    assert all(not p.requires_grad for p in rnd.target.parameters())
    # Predictor body is frozen and equal to the target body; predictor head trains.
    assert all(not p.requires_grad for p in rnd.predictor.body.parameters())
    assert all(p.requires_grad for p in rnd.predictor.head.parameters())
    for pp, tp in zip(rnd.predictor.body.parameters(), rnd.target.body.parameters()):
        assert torch.allclose(pp, tp)
    # compute() returns a finite, non-negative per-sample bonus on this path too.
    out = rnd.compute(make_samples())
    assert out.shape == (N,) or out.shape == (N, 1)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_distance_mse_vs_abs():
    """_dist_ensemble computes 0.5*sum(diff^2) for mse and sum(|diff|) for abs; bad distance raises."""
    # Build known src (N, n_pred, out) and tgt (N, out) so the two reductions are hand-checkable.
    src = torch.tensor([[[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]])  # (2, 1, 3)
    tgt = torch.tensor([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])      # (2, 3); diff is all-ones per row
    # Golden path: mse reduction = 0.5 * sum(1^2 * 3) = 1.5 per row.
    rnd_mse = make_rnd(distance="mse")
    d_mse = rnd_mse._dist_ensemble(src, tgt)
    assert d_mse.shape == (2, 1)
    assert torch.allclose(d_mse, torch.tensor([[1.5], [1.5]]))
    # abs reduction = sum(|1| * 3) = 3 per row, and differs from the mse result.
    rnd_abs = make_rnd(distance="abs")
    d_abs = rnd_abs._dist_ensemble(src, tgt)
    assert torch.allclose(d_abs, torch.tensor([[3.0], [3.0]]))
    assert not torch.allclose(d_mse, d_abs)
    # Edge case: an unsupported distance name is rejected at construction time.
    with pytest.raises(ValueError):
        make_rnd(distance="l2")


def test_invalid_feature_raises():
    """The constructor accepts every FEATURE_CHOICES value and rejects an unknown feature name."""
    # Golden path: each documented feature constructs without error.
    for feature in FEATURE_CHOICES:
        make_rnd(feature=feature)
    # Edge case: an unknown feature string raises ValueError.
    with pytest.raises(ValueError):
        make_rnd(feature="not_a_feature")


def test_update_reduces_bonus_single_predictor():
    """Repeated update() on one fixed batch lowers the single-predictor bonus (predictor learns target)."""
    torch.manual_seed(0)
    # use_obs_norm=False so normalization does not shift between the before/after compute calls.
    rnd = make_rnd(feature="rnd_state", use_obs_norm=False, lr=0.01)
    samples = make_samples(seed=3)
    before = rnd.compute(samples).mean().item()
    # Train the predictor toward the frozen target on the same batch many times.
    for _ in range(150):
        rnd.update(samples)
    after = rnd.compute(samples).mean().item()
    # Golden path + edge: the mean bonus must strictly decrease and stay non-negative.
    assert after < before
    assert after >= 0.0


def test_update_reduces_bonus_linear_rnd():
    """Repeated update() also lowers the bonus on the linear_rnd path (only the head is fit)."""
    torch.manual_seed(0)
    rnd = make_rnd(feature="rnd_state", linear_rnd=True, use_obs_norm=False, lr=0.01)
    samples = make_samples(seed=4)
    before = rnd.compute(samples).mean().item()
    # The trainable head learns to match the target output on this batch.
    for _ in range(150):
        rnd.update(samples)
    after = rnd.compute(samples).mean().item()
    assert after < before
    assert after >= 0.0
