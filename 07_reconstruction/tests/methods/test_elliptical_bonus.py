"""Tests for the EllipticalBonus intrinsic reward model (Mahalanobis / UCB bonus).

Covers both quadratic-form paths: the default inverse-free Cholesky solve and the kept explicit
inverse. A dedicated test pins that the two paths produce the same bonus (the validity check).
"""
import numpy as np
import pytest
import torch

from rnd_exploration.methods.elliptical_bonus import (
    EllipticalBonus,
    GlobalEllipticalBonus,
    cholesky_quadratic_form,
    inverse_quadratic_form,
    normalize_features,
)

# Small, fast shapes so every test runs in well under a second.
OBS_SHAPE = (2,)
ACTION_DIM = 2
FEATURE_DIM = 8
REG = 1e-6


def _make_model(seed: int = 0, bonus_method: str = "cholesky", **kwargs) -> EllipticalBonus:
    """Build a CPU batch EllipticalBonus with a fixed seed so the frozen phi network is deterministic.
    bonus_clip defaults to +inf here so the math-identity tests below see the unclipped bonus."""
    torch.manual_seed(seed)
    kwargs.setdefault("bonus_clip", float("inf"))
    kwargs.setdefault("regularization", REG)
    return EllipticalBonus(
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        feature_dim=FEATURE_DIM,
        device="cpu",
        bonus_method=bonus_method,
        **kwargs,
    )


def _make_global_model(seed: int = 0, **kwargs) -> GlobalEllipticalBonus:
    """Build a CPU GlobalEllipticalBonus (cumulative covariance) with a fixed seed; bonus_clip defaults
    to +inf so the count-law identity is not masked by the clip."""
    torch.manual_seed(seed)
    kwargs.setdefault("bonus_clip", float("inf"))
    kwargs.setdefault("regularization", REG)
    return GlobalEllipticalBonus(
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        feature_dim=FEATURE_DIM,
        device="cpu",
        **kwargs,
    )


def _make_next_state_model(seed: int = 0, **kwargs) -> EllipticalBonus:
    """Build a batch EllipticalBonus whose encoder input is the next state only (no action)."""
    return _make_model(seed=seed, feature_input="next_state", **kwargs)


def _batch(n: int, seed: int = 1) -> dict:
    """Build a samples dict with n random observations/actions/next_observations of the configured shapes.
    next_observations is drawn last so the observations/actions draws are byte-identical to before this
    field was added (the state_action path ignores it; the next_state path reads it)."""
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OBS_SHAPE[0])).astype(np.float32),
        "actions": rng.standard_normal((n, ACTION_DIM)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OBS_SHAPE[0])).astype(np.float32),
    }


def test_default_bonus_method_is_cholesky():
    """The default elliptical bonus uses the inverse-free Cholesky path, not the explicit inverse."""
    model = _make_model()
    # Default construction must select cholesky and expose its factor, not an explicit inverse.
    assert model.bonus_method == "cholesky"
    assert hasattr(model, "_chol_L")
    assert not hasattr(model, "_cov_inv")
    # Default numerical knobs: Cholesky jitter 0, quadratic-form floor 1e-12 (the chosen code defaults).
    assert model.cholesky_jitter == 0.0
    assert model.quadratic_floor == 1e-12


def test_default_feature_normalization_is_unit():
    """The elliptical family defaults to unit-norm feature normalization with ε_φ = 1e-8."""
    model = _make_model()
    assert model.feature_normalization == "unit"
    assert model.feature_norm_eps == 1e-8
    # unit mode is stateless, so no running feature-statistics object is created.
    assert model._feat_rms is None


def test_unit_norm_bounds_feature_norm():
    """unit-norm features satisfy ||φ||_2 <= 1, so the ridge λ is interpretable (main.tex)."""
    model = _make_model(feature_normalization="unit")
    phi = model._samples_to_features(_batch(16, seed=4))
    norms = phi.norm(dim=-1)
    # Every normalized feature vector has length at most 1 (up to the ε_φ denominator slack).
    assert torch.all(norms <= 1.0 + 1e-6)
    # The normalization is non-trivial: the raw features are not already unit length (here they are
    # well below 1, so unit-norm rescales them up toward 1).
    raw_norms = model._raw_features(_batch(16, seed=4)).norm(dim=-1)
    assert torch.any(torch.abs(raw_norms - 1.0) > 1e-2)


def test_none_mode_returns_raw_features():
    """The 'none' diagnostic mode leaves the raw frozen-encoder features unchanged."""
    model = _make_model(feature_normalization="none")
    samples = _batch(8, seed=2)
    assert torch.allclose(model._samples_to_features(samples), model._raw_features(samples))


def test_rms_unit_runs_and_tracks_running_stats():
    """rms_unit keeps running per-coordinate stats, updates them on update(), and gives valid bonuses."""
    model = _make_model(feature_normalization="rms_unit")
    assert model._feat_rms is not None
    samples = _batch(6, seed=8)
    # Before any update the running stats are at their init (mean 0, var 1), so rms_unit == unit.
    unit_model = _make_model(feature_normalization="unit")  # same seed -> same frozen phi
    assert torch.allclose(model._samples_to_features(samples), unit_model._samples_to_features(samples), atol=1e-6)
    # An update must move the running mean off zero and keep the bonus finite and nonnegative.
    mean_before = model._feat_rms.mean.copy()
    model.update(_batch(64, seed=5))
    assert not np.allclose(model._feat_rms.mean, mean_before)
    bonus = model.compute(samples)
    assert torch.all(torch.isfinite(bonus)) and torch.all(bonus >= 0)


def test_normalize_features_helper_modes():
    """The module-level normalize_features matches the closed-form for each mode on a fixed input."""
    z = torch.tensor([[3.0, 4.0], [0.0, 0.0]])  # row 0 has norm 5; row 1 is the zero vector
    eps = 1e-8
    # none: identity.
    assert torch.allclose(normalize_features(z, "none", eps), z)
    # unit: row 0 -> (0.6, 0.8); row 1 -> 0/(0+eps) = 0 (the ε_φ denominator avoids a divide-by-zero).
    unit = normalize_features(z, "unit", eps)
    assert torch.allclose(unit[0], torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(unit[1], torch.zeros(2), atol=1e-6)
    # rms_unit: standardize by mean/std then unit-normalize; with mean 0 / std 1 it reduces to unit.
    mean, std = torch.zeros(2), torch.ones(2)
    assert torch.allclose(normalize_features(z, "rms_unit", eps, mean, std), unit, atol=1e-6)


def test_invalid_feature_normalization_raises():
    """An unknown feature_normalization mode fails loud at construction."""
    with pytest.raises(ValueError):
        _make_model(feature_normalization="zscore")


@pytest.mark.parametrize("bonus_method", ["cholesky", "inverse"])
def test_compute_bonus_nonnegative_and_shape(bonus_method):
    """compute returns a finite, nonnegative bonus of N entries for a batch and for a single transition."""
    model = _make_model(bonus_method=bonus_method)
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


def test_cholesky_matches_inverse():
    """The inverse-free Cholesky path and the explicit-inverse path give the same bonus (validity check)."""
    # Two models with the SAME seed share identical frozen phi weights, so only the solve path differs.
    chol = _make_model(seed=0, bonus_method="cholesky")
    inv = _make_model(seed=0, bonus_method="inverse")
    samples = _batch(6, seed=7)

    # Before any update: Λ_0 = λ I for both; the two bonuses must already agree.
    assert torch.allclose(chol.compute(samples), inv.compute(samples), atol=1e-5, rtol=1e-4)

    # After the same full-rank update both store the same Λ (well-conditioned), so bonuses still agree.
    upd = _batch(64, seed=5)
    chol.update(upd)
    inv.update(upd)
    assert torch.allclose(chol._cov, inv._cov, atol=1e-6)
    assert torch.allclose(chol.compute(samples), inv.compute(samples), atol=1e-5, rtol=1e-4)

    # The two module-level helpers must agree on the same Λ via Λ^{-1} and the Cholesky factor of Λ.
    # φ is cast to float64 to match the float64 covariance/solve structure.
    phi = chol._samples_to_features(samples).double()
    cov_inv = torch.linalg.inv(chol._cov)
    chol_L = torch.linalg.cholesky(chol._cov)
    q_inverse = inverse_quadratic_form(cov_inv, phi)
    q_cholesky = cholesky_quadratic_form(chol_L, phi)
    assert torch.allclose(q_inverse, q_cholesky, atol=1e-5, rtol=1e-4)


def test_compute_matches_manual_quadratic_form():
    """compute equals sqrt(max(φ^T Λ^{-1} φ, ε_q)) computed by hand from Λ, before and after an update."""
    model = _make_model()
    samples = _batch(4, seed=7)

    # Manual reference: invert Λ directly (method-agnostic) and apply the same floor + sqrt. The covariance
    # and solve are float64, so φ is cast to float64 and compute()'s float32 bonus is compared in float64.
    def manual_bonus() -> torch.Tensor:
        phi = model._samples_to_features(samples).double()
        cov_inv = torch.linalg.inv(model._cov)
        quad = ((phi @ cov_inv) * phi).sum(dim=1).clamp(min=model.quadratic_floor)
        return torch.sqrt(quad)

    assert torch.allclose(model.compute(samples).double(), manual_bonus(), atol=1e-5, rtol=1e-4)
    # Edge case: after an update Λ is no longer diagonal; the identity must still hold.
    model.update(_batch(16, seed=99))
    assert torch.allclose(model.compute(samples).double(), manual_bonus(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("bonus_method", ["cholesky", "inverse"])
def test_update_reduces_bonus_for_repeated_state(bonus_method):
    """update lowers the bonus of a state that appears in the update batch; an empty batch is a no-op."""
    model = _make_model(bonus_method=bonus_method)
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
    # Edge case: updating with an empty batch (n=0) must leave Λ and the bonus unchanged.
    cov_snapshot = model._cov.clone()
    empty = {
        "observations": np.zeros((0, OBS_SHAPE[0]), dtype=np.float32),
        "actions": np.zeros((0, ACTION_DIM), dtype=np.float32),
    }
    model.update(empty)
    assert torch.equal(model._cov, cov_snapshot)
    assert float(model.compute(x).reshape(-1)[0]) == bonus_after


def test_update_changes_lambda_and_factor():
    """update replaces Λ and its solve structure consistently; a seen state gets a smaller bonus than a novel one."""
    model = _make_model()  # default cholesky path
    cov_before = model._cov.clone()
    chol_before = model._chol_L.clone()
    # Golden path: update with a diverse full-rank batch (n >> feature_dim) so Λ is well-conditioned.
    model.update(_batch(64, seed=5))
    # Λ and its Cholesky factor must both have moved away from the initial λ I / sqrt(λ) I.
    assert not torch.allclose(model._cov, cov_before)
    assert not torch.allclose(model._chol_L, chol_before)
    # Λ must be symmetric and the stored factor must reconstruct it: L L^T ≈ Λ (jitter is 0 by default).
    assert torch.allclose(model._cov, model._cov.T, atol=1e-6)
    assert torch.allclose(model._chol_L @ model._chol_L.T, model._cov, atol=1e-4)
    # Edge case: update a fresh model with many copies of one state x so Λ is rank-1 + λ I.
    # The seen state sits in the high-eigenvalue subspace, so its bonus is far smaller than a
    # novel state's, which still has large components orthogonal to phi(x).
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


def test_cholesky_jitter_option_enables_factorization():
    """The 1e-8 Cholesky jitter option adds a diagonal correction and still matches the inverse path."""
    # Build a cholesky model with the documented non-default jitter; it must factor and stay valid.
    jittered = _make_model(seed=0, bonus_method="cholesky", cholesky_jitter=1e-8)
    inv = _make_model(seed=0, bonus_method="inverse")
    samples = _batch(6, seed=11)
    assert jittered.cholesky_jitter == 1e-8
    # After a full-rank update Λ has O(1) eigenvalues, so the 1e-8 jitter is a negligible diagonal
    # bump and the jittered Cholesky bonus matches the (unjittered) explicit-inverse bonus.
    # (At the initial Λ_0 = λ I = 1e-6 I, by contrast, a 1e-8 jitter would be ~1% of λ — not negligible.)
    upd = _batch(64, seed=5)
    jittered.update(upd)
    inv.update(upd)
    assert torch.allclose(jittered.compute(samples), inv.compute(samples), atol=1e-4, rtol=1e-3)


def test_invalid_bonus_method_raises():
    """An unknown bonus_method fails loud at construction rather than silently picking a path."""
    with pytest.raises(ValueError):
        _make_model(bonus_method="pseudoinverse")


def test_invalid_update_timing_raises():
    """An unknown update_timing fails loud at construction."""
    with pytest.raises(ValueError):
        _make_model(update_timing="step")


def test_global_default_timing_accumulates_on_update_not_observe():
    """Default (sample timing): the global covariance grows on update() and observe() is a no-op."""
    model = _make_global_model()  # default update_timing="sample"
    assert model.update_timing == "sample"
    cov0 = model._cov.clone()
    # observe() must NOT change Λ under sample timing.
    model.observe(_batch(16, seed=1))
    assert torch.equal(model._cov, cov0)
    # Two update() calls each ADD outer products, so Λ keeps growing (trace strictly increases).
    model.update(_batch(16, seed=2))
    cov1 = model._cov.clone()
    assert torch.trace(cov1) > torch.trace(cov0) + 1e-6
    model.update(_batch(16, seed=3))
    assert torch.trace(model._cov) > torch.trace(cov1) + 1e-6


def test_global_add_timing_accumulates_on_observe_not_update():
    """add timing: the global covariance grows on observe() and update() is a no-op for Λ."""
    model = _make_global_model(update_timing="add")
    assert model.update_timing == "add"
    cov0 = model._cov.clone()
    # update() must NOT change Λ under add timing.
    model.update(_batch(16, seed=2))
    assert torch.equal(model._cov, cov0)
    # observe() ADDS outer products, so Λ grows.
    model.observe(_batch(16, seed=4))
    assert torch.trace(model._cov) > torch.trace(cov0) + 1e-6


def test_global_is_cumulative_not_replaced():
    """The global rule keeps a running total: Λ after two batches = λI + Φ1^TΦ1 + Φ2^TΦ2 (not replaced)."""
    model = _make_global_model(update_timing="add")
    b1, b2 = _batch(12, seed=5), _batch(20, seed=6)
    model.observe(b1)
    model.observe(b2)
    # Reconstruct the expected cumulative covariance directly from the two batches' features (float64, to
    # match the float64 running covariance).
    phi1 = model._samples_to_features(b1).double()
    phi2 = model._samples_to_features(b2).double()
    expected = REG * torch.eye(FEATURE_DIM, dtype=torch.float64) + phi1.T @ phi1 + phi2.T @ phi2
    assert torch.allclose(model._cov, expected, atol=1e-5)


def test_global_count_law_one_repeated_feature():
    """Repeating one (s,a) n times gives the count-decay bonus sqrt(c/(λ+n·c)), c=||φ||² (Sherman-Morrison).
    Uses a well-conditioned ridge λ=1 so the float32 quadratic form is accurate (a tiny ridge makes
    Λ = λI + nφφ^T condition number ~n/λ, where float32 cannot resolve the orthogonal directions)."""
    lam = 1.0
    model = _make_global_model(update_timing="add", regularization=lam)
    x = {
        "observations": np.array([[0.5, -0.2]], dtype=np.float32),
        "actions": np.array([[0.3, 0.4]], dtype=np.float32),
    }
    n = 50
    # Λ = λI + n·φφ^T after observing n identical copies of the same (s,a).
    repeated = {
        "observations": np.tile(x["observations"], (n, 1)).astype(np.float32),
        "actions": np.tile(x["actions"], (n, 1)).astype(np.float32),
    }
    model.observe(repeated)
    # Sherman-Morrison closed form for q = φ^T(λI + n φφ^T)^{-1}φ with c = ||φ||² is c/(λ + n·c).
    phi = model._samples_to_features(x)
    c = float((phi ** 2).sum())
    expected = (c / (lam + n * c)) ** 0.5
    bonus = float(model.compute(x).reshape(-1)[0])
    assert abs(bonus - expected) < 1e-4
    # The bonus must also decay as more visits accumulate: another n copies lowers it further.
    bonus_n = bonus
    model.observe(repeated)
    bonus_2n = float(model.compute(x).reshape(-1)[0])
    assert bonus_2n < bonus_n


def test_batch_add_timing_switch_replaces_from_added_batch():
    """The update_timing switch works for the whole family: a BATCH model on add timing replaces Λ in observe()."""
    model = _make_model(update_timing="add")  # batch rule, but fired on add
    cov0 = model._cov.clone()
    # update() is now a no-op (add timing); Λ stays at λI.
    model.update(_batch(16, seed=2))
    assert torch.equal(model._cov, cov0)
    # observe() REPLACES Λ with the added batch covariance (batch rule), so Λ moves off λI.
    model.observe(_batch(16, seed=7))
    assert not torch.allclose(model._cov, cov0)


def test_bonus_clip_default_is_five():
    """The elliptical family clips the bonus to 5 by default (the family default for both rules)."""
    # Construct directly (the test helper overrides bonus_clip to +inf) to read the real constructor default.
    torch.manual_seed(0)
    batch_default = EllipticalBonus(obs_shape=OBS_SHAPE, action_dim=ACTION_DIM, feature_dim=FEATURE_DIM, device="cpu")
    global_default = GlobalEllipticalBonus(obs_shape=OBS_SHAPE, action_dim=ACTION_DIM, feature_dim=FEATURE_DIM, device="cpu")
    assert batch_default.bonus_clip == 5.0
    assert global_default.bonus_clip == 5.0


def test_bonus_clip_caps_the_bonus():
    """The bonus is clipped to bonus_clip; an inf clip recovers the unclipped value (same Λ, same phi)."""
    # Λ_0 = REG·I = 1e-6·I, so an unseen unit feature has bonus ≈ 1/sqrt(1e-6) ≈ 1000 -> the clip bites.
    clipped = _make_model(seed=0, bonus_clip=5.0)
    unclipped = _make_model(seed=0, bonus_clip=float("inf"))
    samples = _batch(6, seed=4)
    bonus_clipped = clipped.compute(samples)
    bonus_unclipped = unclipped.compute(samples)
    # Every clipped value is at most 5, and at least one raw value actually exceeded 5 (so the clip acted).
    assert torch.all(bonus_clipped <= 5.0 + 1e-6)
    assert torch.any(bonus_unclipped > 5.0)
    # Where the unclipped bonus is below the cap the two agree exactly.
    below = bonus_unclipped <= 5.0
    assert torch.allclose(bonus_clipped[below], bonus_unclipped[below], atol=1e-6)


def test_elliptical_input_is_raw_state_no_obs_rms():
    """The elliptical encoder consumes the RAW [s;a]: scaling the observation scales the encoder input,
    proving no observation-RMS standardization is applied upstream (only the in-family feature norm)."""
    model = _make_model(feature_normalization="none")  # 'none' exposes the raw frozen-encoder output
    base = {
        "observations": np.array([[0.5, -0.2]], dtype=np.float32),
        "actions": np.array([[0.0, 0.0]], dtype=np.float32),
    }
    scaled = {
        "observations": base["observations"] * 10.0,
        "actions": base["actions"],
    }
    # If the state were obs-RMS-standardized, the encoder input (and feature) would be invariant to a
    # constant rescale; because it is raw, the feature changes when the observation is scaled.
    feat_base = model._raw_features(base)
    feat_scaled = model._raw_features(scaled)
    assert not torch.allclose(feat_base, feat_scaled)


def test_default_feature_input_is_state_action():
    """The elliptical encoder consumes (s, a) by default: input dim = obs_dim + action_dim."""
    model = _make_model()
    assert model.feature_input == "state_action"
    assert model._input_dim == OBS_SHAPE[0] + ACTION_DIM
    # the first Linear layer's in_features must match the (s, a) input width.
    assert model.phi.network[0].in_features == OBS_SHAPE[0] + ACTION_DIM


def test_next_state_feature_input_uses_next_obs_only():
    """feature_input='next_state' builds φ from s' alone: input dim = obs_dim, and the action is ignored."""
    model = _make_next_state_model(feature_normalization="none")  # 'none' exposes the raw encoder output
    assert model.feature_input == "next_state"
    # the encoder input drops the action dims (obs_dim only).
    assert model._input_dim == OBS_SHAPE[0]
    assert model.phi.network[0].in_features == OBS_SHAPE[0]
    # changing the next observation changes the feature; changing the action does NOT.
    base = {
        "observations": np.array([[0.5, -0.2]], dtype=np.float32),
        "actions": np.array([[0.1, 0.2]], dtype=np.float32),
        "next_observations": np.array([[0.3, 0.7]], dtype=np.float32),
    }
    diff_action = {**base, "actions": base["actions"] * 5.0}
    diff_next = {**base, "next_observations": base["next_observations"] * -2.0}
    feat_base = model._raw_features(base)
    assert torch.allclose(model._raw_features(diff_action), feat_base)    # action not consumed
    assert not torch.allclose(model._raw_features(diff_next), feat_base)  # next state is consumed


def test_next_state_bonus_and_update_work():
    """A next-state elliptical model gives a finite nonnegative bonus, and update() lowers a seen state's bonus."""
    model = _make_next_state_model()
    # Golden path: a batch yields finite nonnegative bonuses.
    bonus = model.compute(_batch(6, seed=4))
    assert torch.all(torch.isfinite(bonus)) and torch.all(bonus >= 0)
    # A state whose next_observation appears (repeated) in the update batch gets a smaller bonus afterward.
    x = {
        "observations": np.zeros((1, OBS_SHAPE[0]), dtype=np.float32),
        "actions": np.zeros((1, ACTION_DIM), dtype=np.float32),
        "next_observations": np.array([[0.4, -0.3]], dtype=np.float32),
    }
    before = float(model.compute(x).reshape(-1)[0])
    batch = _batch(8, seed=3)
    # before: each field has 8 rows; after: 8 + 4 repeats of x -> 12 rows in every field (kept consistent).
    for k, v in x.items():
        batch[k] = np.concatenate([batch[k], np.tile(v, (4, 1))], axis=0)
    model.update(batch)
    after = float(model.compute(x).reshape(-1)[0])
    assert after < before


def test_invalid_feature_input_raises():
    """An unknown feature_input fails loud at construction."""
    with pytest.raises(ValueError):
        _make_model(feature_input="state_only")


def test_diagnostics_tracks_feature_scale_and_ridge_ratio():
    """diagnostics() reports the mean raw / normalized feature squared norms from the covariance's input
    stream and the effective ridge ratio ρ = λ d / mean ||φ||²; the eigenvalue range brackets Λ."""
    lam = 0.5
    model = _make_model(feature_normalization="unit", regularization=lam)
    model.update(_batch(64, seed=5))
    d = model.diagnostics()
    # golden path: unit-norm features have mean ||φ||² ≈ 1, so ρ ≈ λ·d / 1
    assert 0.9 <= d["mean_normalized_feature_sq_norm"] <= 1.0 + 1e-6
    assert abs(d["effective_ridge_ratio"] - lam * FEATURE_DIM / d["mean_normalized_feature_sq_norm"]) < 1e-9
    assert d["mean_raw_feature_sq_norm"] > 0
    assert d["n_update_features"] == 64
    # Λ = λI + data/n: its eigenvalues are at least λ, and the trace identity bounds the max eigenvalue
    assert d["cov_eig_min"] >= lam - 1e-9
    assert d["cov_eig_max"] > d["cov_eig_min"]
    # edge case: before any update the ratio is infinite (no feature stream yet)
    fresh = _make_model()
    assert fresh.diagnostics()["effective_ridge_ratio"] == float("inf")


def test_diagnostics_counts_clip_hits():
    """The clip-hit fraction counts bonuses above the cap; it is zero when the clip is +inf."""
    # tiny ridge -> unseen states have bonus ≈ 1/sqrt(1e-6) = 1000 >> clip 5, so every bonus is clipped
    clipped = _make_model(seed=0, bonus_clip=5.0)
    n = 6
    clipped.compute(_batch(n, seed=4))
    d = clipped.diagnostics()
    assert d["n_bonuses_computed"] == n
    assert d["clip_hit_fraction"] == 1.0
    # edge case: with the clip disabled (+inf) nothing can exceed the cap
    unclipped = _make_model(seed=0, bonus_clip=float("inf"))
    unclipped.compute(_batch(n, seed=4))
    assert unclipped.diagnostics()["clip_hit_fraction"] == 0.0
