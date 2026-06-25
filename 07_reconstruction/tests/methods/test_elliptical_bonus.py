"""Tests for the EllipticalBonus intrinsic reward model (Mahalanobis / UCB bonus)."""
import numpy as np
import torch

from rnd_exploration.methods.elliptical_bonus import EllipticalBonus

# Small, fast shapes so every test runs in well under a second.
OBS_SHAPE = (2,)
ACTION_DIM = 2
FEATURE_DIM = 8
REG = 1e-6


def _make_model(seed: int = 0) -> EllipticalBonus:
    """Build a CPU EllipticalBonus with a fixed seed so the frozen phi network is deterministic."""
    torch.manual_seed(seed)
    return EllipticalBonus(
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        feature_dim=FEATURE_DIM,
        device="cpu",
        regularization=REG,
    )


def _batch(n: int, seed: int = 1) -> dict:
    """Build a samples dict with n random observations/actions of the configured shapes."""
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OBS_SHAPE[0])).astype(np.float32),
        "actions": rng.standard_normal((n, ACTION_DIM)).astype(np.float32),
    }


def test_compute_bonus_nonnegative_and_shape():
    """compute returns a finite, nonnegative bonus of N entries for a batch and for a single transition."""
    model = _make_model()
    # Golden path: a batch of 5 transitions -> N=5 nonnegative finite values, shape (N,) or (N,1).
    n = 5
    bonus = model.compute(_batch(n))
    assert bonus.numel() == n
    assert bonus.reshape(-1).shape == (n,)
    assert bonus.shape in ((n,), (n, 1))
    assert torch.all(bonus >= 0)
    assert torch.all(torch.isfinite(bonus))
    # Edge case: a single transition passed as 1D arrays -> exactly one nonnegative value.
    single = {
        "observations": np.array([0.3, -0.7], dtype=np.float32),
        "actions": np.array([0.1, 0.2], dtype=np.float32),
    }
    bonus_single = model.compute(single)
    assert bonus_single.numel() == 1
    assert float(bonus_single.reshape(-1)[0]) >= 0


def test_compute_matches_manual_quadratic_form():
    """compute equals the row-wise Mahalanobis form sqrt(phi^T Lambda^{-1} phi) before and after an update."""
    model = _make_model()
    samples = _batch(4, seed=7)

    # Golden path: with the initial (diagonal) Lambda^{-1}, recompute the quadratic form by hand.
    def manual_bonus() -> torch.Tensor:
        phi = model._samples_to_features(samples)
        quad = ((phi @ model._cov_inv) * phi).sum(dim=1).clamp(min=1e-8)
        return torch.sqrt(quad)

    assert torch.allclose(model.compute(samples), manual_bonus(), atol=1e-6)
    # Edge case: after an update Lambda^{-1} is no longer diagonal; the identity must still hold.
    model.update(_batch(16, seed=99))
    assert torch.allclose(model.compute(samples), manual_bonus(), atol=1e-5)


def test_update_reduces_bonus_for_repeated_state():
    """update lowers the bonus of a state that appears in the update batch; an empty batch is a no-op."""
    model = _make_model()
    # A single fixed state-action pair whose bonus we track across an update.
    x = {
        "observations": np.array([[0.5, -0.2]], dtype=np.float32),
        "actions": np.array([[0.3, 0.4]], dtype=np.float32),
    }
    bonus_before = float(model.compute(x).reshape(-1)[0])
    # Golden path: update on a batch that includes the repeated state x -> its bonus must drop.
    batch = _batch(8, seed=3)
    batch["observations"] = np.concatenate([batch["observations"], np.tile(x["observations"], (4, 1))], axis=0)
    batch["actions"] = np.concatenate([batch["actions"], np.tile(x["actions"], (4, 1))], axis=0)
    model.update(batch)
    bonus_after = float(model.compute(x).reshape(-1)[0])
    assert bonus_after < bonus_before
    # Edge case: updating with an empty batch (n=0) must leave Lambda^{-1} and the bonus unchanged.
    cov_inv_snapshot = model._cov_inv.clone()
    empty = {
        "observations": np.zeros((0, OBS_SHAPE[0]), dtype=np.float32),
        "actions": np.zeros((0, ACTION_DIM), dtype=np.float32),
    }
    model.update(empty)
    assert torch.equal(model._cov_inv, cov_inv_snapshot)
    assert float(model.compute(x).reshape(-1)[0]) == bonus_after


def test_update_changes_lambda_and_its_inverse():
    """update replaces Lambda and its inverse consistently; a seen state gets a smaller bonus than a novel one."""
    model = _make_model()
    cov_before = model._cov.clone()
    cov_inv_before = model._cov_inv.clone()
    # Golden path: update with a diverse full-rank batch (n >> feature_dim) so Lambda is well-conditioned.
    model.update(_batch(64, seed=5))
    # Lambda and its inverse must both have moved away from the initial reg*I / (1/reg)*I.
    assert not torch.allclose(model._cov, cov_before)
    assert not torch.allclose(model._cov_inv, cov_inv_before)
    # Lambda must be symmetric and its stored inverse must actually invert it.
    assert torch.allclose(model._cov, model._cov.T, atol=1e-6)
    identity = model._cov_inv @ model._cov
    assert torch.allclose(identity, torch.eye(FEATURE_DIM), atol=1e-3)
    # Edge case: update a fresh model with many copies of one state x so Lambda is rank-1 + reg*I.
    # The seen state sits in the high-eigenvalue subspace, so its bonus is far smaller than a
    # novel state's, which still has large-inverse (reg^{-1}) components orthogonal to phi(x).
    model_rank1 = _make_model()
    x_obs = np.array([0.5, -0.2], dtype=np.float32)
    x_act = np.array([0.3, 0.4], dtype=np.float32)
    model_rank1.update(
        {
            "observations": np.tile(x_obs, (32, 1)).astype(np.float32),
            "actions": np.tile(x_act, (32, 1)).astype(np.float32),
        }
    )
    seen = {"observations": x_obs[None, :], "actions": x_act[None, :]}
    novel = {
        "observations": np.array([[-3.0, 2.5]], dtype=np.float32),
        "actions": np.array([[-1.5, 1.0]], dtype=np.float32),
    }
    bonus_seen = float(model_rank1.compute(seen).reshape(-1)[0])
    bonus_novel = float(model_rank1.compute(novel).reshape(-1)[0])
    assert bonus_novel > bonus_seen
