"""Tests for the run-8.1.2 switches on the RND class.

Covers: the ratio bonus vs the frozen initial predictor (readout_norm_init, algorithms 2.1-2.3) —
exactly 1 at update 0, frozen copy untouched by training; the initialization-normalized training
loss (predictor_loss='mse_init_normalized', algorithm 2.2); the plain constant-rate SGD optimizer
option; the LayerNorm architecture (rnd_layer_norm, algorithm 2.3) including the keyed-init-stream
equality with the LayerNorm-free twin; and the loud rejection of invalid flag combinations.
Small shapes keep it fast.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from rnd_exploration.methods.rnd import RND

OBS_DIM = 4
OUT = 16
N = 8


def make_samples(n=N, seed=0):
    """Build a deterministic samples dict (observations / next_observations / actions)."""
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OBS_DIM)).astype(np.float32),
        "actions": rng.standard_normal((n, 2)).astype(np.float32),
    }


def make_rnd(**kw):
    """Construct a small CPU RND, overridable via kwargs (obs-norm off so readouts see raw input)."""
    params = dict(obs_shape=(OBS_DIM,), output_dim=OUT, device="cpu", action_dim=2, use_obs_norm=False)
    params.update(kw)
    return RND(**params)


def make_arm(**kw):
    """Construct the run-8.1.2 algorithm-2 base arm: ratio bonus, l2 readout, normal_0.5 bias,
    leaky_relu, deeper predictor, plain constant-rate SGD — the sweep's actual knob bundle."""
    params = dict(
        bonus_readout="l2", readout_norm_init=True, bias_init="normal_0.5",
        activation="leaky_relu", predictor_extra_layers=1, optimizer="sgd", lr=1e-2,
    )
    params.update(kw)
    return make_rnd(**params)


# ---------------------------------------------------------------------------------------------------
# ratio bonus (algorithms 2.1-2.3)
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layer_norm", [False, True])
def test_ratio_bonus_is_one_at_init(layer_norm):
    """Golden property: at update 0 the predictor IS its frozen copy, so the ratio bonus is 1
    everywhere (up to the 1e-8 eps guard), with and without LayerNorm."""
    model = make_arm(layer_norm=layer_norm)
    bonus = model.compute(make_samples())
    assert bonus.shape == (N,)
    assert torch.allclose(bonus, torch.ones(N), atol=1e-5)


def test_ratio_bonus_frozen_copy_untouched_by_training():
    """After training updates the live predictor moves and the bonus leaves 1, but the frozen
    initial predictor's parameters stay byte-identical and grad-free."""
    model = make_arm()
    init_before = {k: v.clone() for k, v in model.init_predictor.state_dict().items()}
    pred_before = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    # several full-batch updates on fixed samples move the live predictor toward the target
    samples = make_samples()
    for _ in range(20):
        model.update(samples)
    # golden path: live predictor changed, bonus no longer ~1 on the trained batch
    assert any(not torch.equal(pred_before[k], v) for k, v in model.predictor.state_dict().items())
    bonus = model.compute(samples)
    assert not torch.allclose(bonus, torch.ones(N), atol=1e-3)
    # edge case that must NEVER happen: the frozen copy moving with training
    for k, v in model.init_predictor.state_dict().items():
        assert torch.equal(init_before[k], v), f"init_predictor tensor {k} changed during training"
    assert all(not p.requires_grad for p in model.init_predictor.parameters())


def test_ratio_bonus_finite_when_init_error_is_tiny():
    """eps guard: with predictor == init == target the raw errors are ~0 on both sides of the ratio;
    the bonus must stay finite (clamp + eps), never NaN/inf."""
    model = make_arm()
    with torch.no_grad():
        model.predictor.network.load_state_dict(model.target.network.state_dict(), strict=False)
    # the deeper predictor has an extra Linear the target lacks; zero it so outputs match exactly
    with torch.no_grad():
        last = model.predictor.network[-1]
        last.weight.zero_()
        last.bias.zero_()
    bonus = model.compute(make_samples())
    assert torch.isfinite(bonus).all()


# ---------------------------------------------------------------------------------------------------
# initialization-normalized training loss (algorithm 2.2)
# ---------------------------------------------------------------------------------------------------

def test_normalized_loss_is_one_at_init():
    """At update 0, ||e_theta||^2 / (||e_0||^2 + delta) == 1 per sample (up to delta), so the
    batch-mean loss the update uses is ~1. Verified on the model's own nets."""
    model = make_arm(predictor_loss="mse_init_normalized")
    x = torch.as_tensor(make_samples()["next_observations"])
    with torch.no_grad():
        e_live = model.target(x) - model.predictor(x)
        e_init = model.target(x) - model.init_predictor(x)
    per_sample = e_live.pow(2).sum(dim=1) / (e_init.pow(2).sum(dim=1) + model.predictor_loss_delta)
    assert torch.allclose(per_sample, torch.ones(N), atol=1e-5)


def test_normalized_loss_update_moves_only_the_live_predictor():
    """update() under the normalized loss trains the live predictor only: target and frozen copy
    stay byte-identical, and the live predictor's parameters move."""
    model = make_arm(predictor_loss="mse_init_normalized")
    tgt_before = {k: v.clone() for k, v in model.target.state_dict().items()}
    init_before = {k: v.clone() for k, v in model.init_predictor.state_dict().items()}
    pred_before = {k: v.clone() for k, v in model.predictor.state_dict().items()}
    samples = make_samples()
    for _ in range(5):
        model.update(samples)
    assert any(not torch.equal(pred_before[k], v) for k, v in model.predictor.state_dict().items())
    for k, v in model.target.state_dict().items():
        assert torch.equal(tgt_before[k], v)
    for k, v in model.init_predictor.state_dict().items():
        assert torch.equal(init_before[k], v)


# ---------------------------------------------------------------------------------------------------
# plain constant-rate SGD
# ---------------------------------------------------------------------------------------------------

def test_plain_sgd_constant_lr():
    """optimizer='sgd' builds torch SGD at lr=rnd_lr with momentum 0 and NO schedule: the lr is
    unchanged after updates (contrast sgd1t, whose lr follows the 1/t schedule)."""
    model = make_arm(lr=1e-3)
    assert isinstance(model.opt, torch.optim.SGD)
    assert model.opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert model.opt.param_groups[0]["momentum"] == 0
    samples = make_samples()
    for _ in range(5):
        model.update(samples)
    # golden path: constant rate survives updates
    assert model.opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    # edge contrast: sgd1t's lr has decayed after the same number of updates
    sched = make_rnd(optimizer="sgd1t", sgd_eta0=1e-3, sgd_t0=2.0)
    for _ in range(5):
        sched.update(samples)
    assert sched.opt.param_groups[0]["lr"] < 1e-3


# ---------------------------------------------------------------------------------------------------
# LayerNorm architecture (algorithm 2.3)
# ---------------------------------------------------------------------------------------------------

def test_layer_norm_architecture_and_identity_init():
    """layer_norm=True: LayerNorm after each hidden Linear (before its activation), the predictor's
    extra block carries its own, never after the output Linear; affine params start at identity."""
    model = make_arm(layer_norm=True)
    # target: [Linear, LN, act, Linear]; predictor (extra_layers=1): [Linear, LN, act, Linear, LN, act, Linear]
    tgt_types = [type(m) for m in model.target.network]
    pred_types = [type(m) for m in model.predictor.network]
    assert tgt_types == [nn.Linear, nn.LayerNorm, nn.LeakyReLU, nn.Linear]
    assert pred_types == [nn.Linear, nn.LayerNorm, nn.LeakyReLU, nn.Linear,
                          nn.LayerNorm, nn.LeakyReLU, nn.Linear]
    # no output normalization: the last module of both nets is the output Linear
    assert isinstance(model.target.network[-1], nn.Linear)
    assert isinstance(model.predictor.network[-1], nn.Linear)
    for net in (model.target.network, model.predictor.network):
        for m in net:
            if isinstance(m, nn.LayerNorm):
                assert torch.equal(m.weight, torch.ones_like(m.weight))
                assert torch.equal(m.bias, torch.zeros_like(m.bias))


def test_layer_norm_keyed_draws_match_layer_norm_free_twin():
    """The keyed weight/bias streams skip LayerNorm positions, so the 2.3 net draws the SAME Linear
    biases (normal_0.5) as its LayerNorm-free twin at the same seed — the ablation isolates
    LayerNorm alone."""
    plain = make_arm(bias_seed=7)
    normed = make_arm(bias_seed=7, layer_norm=True)
    plain_linears = [m for m in plain.predictor.network if isinstance(m, nn.Linear)]
    normed_linears = [m for m in normed.predictor.network if isinstance(m, nn.Linear)]
    assert len(plain_linears) == len(normed_linears) == 3
    for a, b in zip(plain_linears, normed_linears):
        assert torch.equal(a.bias, b.bias)
    plain_tgt = [m for m in plain.target.network if isinstance(m, nn.Linear)]
    normed_tgt = [m for m in normed.target.network if isinstance(m, nn.Linear)]
    for a, b in zip(plain_tgt, normed_tgt):
        assert torch.equal(a.bias, b.bias)


# ---------------------------------------------------------------------------------------------------
# loud rejection of invalid combinations
# ---------------------------------------------------------------------------------------------------

def test_invalid_combinations_raise():
    """Every unsupported combination fails loud at construction (code-style rule 1)."""
    # ratio bonus needs the l2 readout
    with pytest.raises(ValueError):
        make_rnd(readout_norm_init=True, bonus_readout="mse")
    with pytest.raises(ValueError):
        make_rnd(readout_norm_init=True, bonus_readout="mse_mean")
    # normalized loss needs the frozen copy
    with pytest.raises(ValueError):
        make_rnd(predictor_loss="mse_init_normalized")
    # unknown loss name
    with pytest.raises(ValueError):
        make_rnd(predictor_loss="l2_normalized")
    # ensemble/linear paths are unsupported for all three switches
    with pytest.raises(ValueError):
        make_rnd(readout_norm_init=True, bonus_readout="l2", n_predictors=2)
    with pytest.raises(ValueError):
        make_rnd(layer_norm=True, n_predictors=2)
    with pytest.raises(ValueError):
        make_rnd(layer_norm=True, linear_rnd=True)
    # unknown optimizer name still rejected
    with pytest.raises(ValueError):
        make_rnd(optimizer="sgd_momentum")
