"""Tests for the train-run-5 original-RND switches on the RND class.

Covers: the mse_mean readout formula; the deeper-predictor architecture (rnd_predictor_extra_layers)
shapes + zero-is-bit-identical + keyed-bias-stream stability; update_proportion (full-batch
bit-identity, no-RNG-consumption, mask statistics, empty-mask edge); and the leaky_relu activation.
Small shapes keep it fast.
"""
import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from rnd_exploration.methods.rnd import RND, ObservationEncoder

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


# ---------------------------------------------------------------------------------------------------
# mse_mean readout
# ---------------------------------------------------------------------------------------------------

def test_bonus_readout_mse_mean_formula():
    """mse_mean = (1/m)*sum e_j^2 = (2/m)*B_mse on identical weights; and requires distance='mse'."""
    samples = make_samples()
    mse = make_rnd()
    mm = make_rnd(bonus_readout="mse_mean")
    # identical nets so both readouts see the same residual e = target(x) - predictor(x)
    mm.predictor.load_state_dict(mse.predictor.state_dict())
    mm.target.load_state_dict(mse.target.state_dict())
    b_mse = mse.compute(samples)
    b_mm = mm.compute(samples)
    # golden path: mse_mean == (2/m) * mse readout == (1/m) * sum_j e_j^2
    assert torch.allclose(b_mm, (2.0 / OUT) * b_mse, atol=1e-6)
    x = torch.as_tensor(samples["next_observations"])
    with torch.no_grad():
        e = mse.target(x) - mse.predictor(x)
    assert torch.allclose(b_mm, e.pow(2).sum(dim=1) / OUT, atol=1e-6)
    # edge case: mse_mean on the abs distance is rejected (needs the squared-error distance)
    with pytest.raises(ValueError):
        make_rnd(bonus_readout="mse_mean", distance="abs")


# ---------------------------------------------------------------------------------------------------
# deeper predictor (rnd_predictor_extra_layers)
# ---------------------------------------------------------------------------------------------------

def test_predictor_extra_layers_shapes():
    """extra_layers deepens the PREDICTOR only; target keeps 2 Linears; output dim m is preserved."""
    rnd = make_rnd(predictor_extra_layers=2)
    # golden path: predictor Linears at positions 0,2,4,6 (2 base + 2 extra); target only 0,2
    pred_linears = [m for m in rnd.predictor.network if isinstance(m, nn.Linear)]
    tgt_linears = [m for m in rnd.target.network if isinstance(m, nn.Linear)]
    assert len(pred_linears) == 4 and len(tgt_linears) == 2
    # widths: obs->256, 256->m, then m->m, m->m; final output dim stays m so the distillation error matches
    assert pred_linears[0].in_features == OBS_DIM and pred_linears[0].out_features == 256
    assert pred_linears[-1].in_features == OUT and pred_linears[-1].out_features == OUT
    with torch.no_grad():
        assert rnd.predictor(torch.zeros(1, OBS_DIM)).shape == (1, OUT)
    out = rnd.compute(make_samples())
    assert torch.isfinite(out).all() and (out >= 0).all()
    # edge cases: the deeper predictor is only defined for the standard single-predictor architecture
    with pytest.raises(ValueError):
        make_rnd(predictor_extra_layers=1, linear_rnd=True)
    with pytest.raises(ValueError):
        make_rnd(predictor_extra_layers=1, n_predictors=3)


def test_extra_layers_zero_is_bit_identical():
    """extra_layers=0 (default) builds the exact historical net: same predictor and target params."""
    torch.manual_seed(0)
    base = make_rnd()
    torch.manual_seed(0)
    zero = make_rnd(predictor_extra_layers=0)
    # golden path: identical construction (RNG order unchanged) -> byte-equal predictor and target
    for pb, pz in zip(base.predictor.parameters(), zero.predictor.parameters()):
        assert torch.equal(pb, pz)
    for tb, tz in zip(base.target.parameters(), zero.target.parameters()):
        assert torch.equal(tb, tz)


def test_keyed_bias_streams_stable_under_extra_layers():
    """A deeper predictor keeps the base Linears' keyed bias draws (positions 0,2) byte-identical."""
    def expected_normal(seed, net_name, layer_idx, shape):
        """Recompute the keyed normal bias draw with the ablation's exact key format."""
        key = "::".join(str(p) for p in (seed, "bias-normal", net_name, layer_idx))
        g = torch.Generator()
        g.manual_seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0x7FFFFFFF)
        return 0.5 * torch.randn(shape, generator=g)
    torch.manual_seed(0)
    shallow = make_rnd(bias_init="normal_0.5", bias_seed=111)
    torch.manual_seed(0)
    deep = make_rnd(bias_init="normal_0.5", bias_seed=111, predictor_extra_layers=2)
    # golden path: predictor positions 0,2 keyed identically whether or not extra layers were appended
    for layer_idx in (0, 2):
        assert torch.equal(shallow.predictor.network[layer_idx].bias,
                           deep.predictor.network[layer_idx].bias)
        assert torch.equal(deep.predictor.network[layer_idx].bias,
                           expected_normal(111, "predictor", layer_idx,
                                           deep.predictor.network[layer_idx].bias.shape))
    # edge: the extra Linears (positions 4,6) get their OWN keyed draws (nonzero, distinct keys)
    for layer_idx in (4, 6):
        assert torch.equal(deep.predictor.network[layer_idx].bias,
                           expected_normal(111, "predictor", layer_idx,
                                           deep.predictor.network[layer_idx].bias.shape))
    # target biases (positions 0,2) are unaffected by the predictor's depth
    for layer_idx in (0, 2):
        assert torch.equal(shallow.target.network[layer_idx].bias,
                           deep.target.network[layer_idx].bias)


# ---------------------------------------------------------------------------------------------------
# update_proportion
# ---------------------------------------------------------------------------------------------------

def test_update_proportion_full_batch_bit_identical_and_no_rng():
    """update_proportion=1.0 (default) trains on the whole batch AND draws no torch.rand."""
    samples = make_samples(seed=3)
    # golden path: explicit 1.0 and the default produce byte-identical predictor params after an update
    torch.manual_seed(0)
    a = make_rnd(update_proportion=1.0)
    torch.manual_seed(0)
    b = make_rnd()  # default is 1.0
    a.update(samples)
    b.update(samples)
    for pa, pb in zip(a.predictor.parameters(), b.predictor.parameters()):
        assert torch.equal(pa, pb)
    # no-RNG-consumption: construction consumes the same RNG both times; if update() drew any
    # torch.rand, the post-update stream position would differ from the construct-only position.
    # Same seed -> same construction -> (construct + update) probe == (construct only) probe iff
    # update consumed zero global RNG (the >=1.0 branch never calls _masked_mean's torch.rand).
    torch.manual_seed(0)
    make_rnd(update_proportion=1.0).update(samples)
    after_update = torch.rand(1).item()
    torch.manual_seed(0)
    make_rnd(update_proportion=1.0)  # construct only (identical construction RNG), no update
    bare = torch.rand(1).item()
    assert after_update == bare
    # contrast: a proportion < 1.0 DOES draw torch.rand in update(), so it shifts the stream
    torch.manual_seed(0)
    make_rnd(update_proportion=0.5).update(samples)
    masked_after = torch.rand(1).item()
    assert masked_after != bare


def test_update_proportion_mask_statistics_and_empty_edge():
    """update_proportion<1.0 keeps ~that fraction of samples and never yields a NaN loss."""
    rnd = make_rnd(update_proportion=0.25)
    # golden path: over a large batch and many draws, the kept fraction is ~0.25
    torch.manual_seed(0)
    kept = []
    for _ in range(200):
        mask = (torch.rand(256) < rnd.update_proportion).float()
        kept.append(mask.mean().item())
    assert abs(float(np.mean(kept)) - 0.25) < 0.02
    # the masked loss is finite when trained on a real batch
    samples = make_samples(seed=5)
    for _ in range(10):
        rnd.update(samples)
    assert torch.isfinite(rnd.compute(samples)).all()
    # edge case: an all-False mask must not divide by zero (clamp(min=1.0)) -> finite zero loss
    per_sample = torch.tensor([1.0, 2.0, 3.0])
    tiny = make_rnd(update_proportion=1e-12)  # effectively never keeps a sample
    torch.manual_seed(1)
    val = tiny._masked_mean(per_sample)
    assert torch.isfinite(val) and val.item() == 0.0
    # update_proportion out of (0,1] is rejected
    with pytest.raises(ValueError):
        make_rnd(update_proportion=0.0)
    with pytest.raises(ValueError):
        make_rnd(update_proportion=1.5)


# ---------------------------------------------------------------------------------------------------
# activation
# ---------------------------------------------------------------------------------------------------

def test_activation_relu_default_and_leaky_relu_slope():
    """activation defaults to nn.ReLU; leaky_relu builds nn.LeakyReLU with the original's slope 0.2."""
    # golden path: default net uses ReLU activations
    default = make_rnd()
    assert any(isinstance(m, nn.ReLU) for m in default.target.network)
    assert not any(isinstance(m, nn.LeakyReLU) for m in default.target.network)
    # leaky_relu: both nets use nn.LeakyReLU(negative_slope=0.2), compute stays finite/non-negative
    leaky = make_rnd(activation="leaky_relu")
    acts = [m for m in leaky.predictor.network if isinstance(m, nn.LeakyReLU)]
    assert acts and all(abs(m.negative_slope - 0.2) < 1e-12 for m in acts)
    assert any(isinstance(m, nn.LeakyReLU) for m in leaky.target.network)
    out = leaky.compute(make_samples())
    assert torch.isfinite(out).all() and (out >= 0).all()
    # edge case: an unknown activation is rejected
    with pytest.raises(ValueError):
        make_rnd(activation="gelu")
    # edge case: non-relu activation on the ensemble/linear architectures is rejected
    with pytest.raises(ValueError):
        make_rnd(activation="leaky_relu", n_predictors=3)


def test_run5_original_small_architecture():
    """The run-5 original-small net: target state->256->128, predictor +1 layer (128->128), leaky 0.2."""
    rnd = RND(obs_shape=(OBS_DIM,), output_dim=128, device="cpu", use_obs_norm=False,
              action_dim=2, activation="leaky_relu", predictor_extra_layers=1,
              bonus_readout="mse_mean", lr=1e-4, update_proportion=1.0,
              reward_norm=True, reward_norm_gamma=0.99)
    pred_linears = [m for m in rnd.predictor.network if isinstance(m, nn.Linear)]
    tgt_linears = [m for m in rnd.target.network if isinstance(m, nn.Linear)]
    # target: 2 Linears (state->256, 256->128); predictor: 3 Linears (+ one 128->128), output dim 128
    assert [l.out_features for l in tgt_linears] == [256, 128]
    assert [l.out_features for l in pred_linears] == [256, 128, 128]
    assert pred_linears[-1].in_features == 128
    # activation is leaky 0.2; Adam lr is the exposed 1e-4; readout is mean-over-dims
    assert isinstance(rnd.opt, torch.optim.Adam)
    assert rnd.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert rnd.bonus_readout == "mse_mean"
    out = rnd.compute(make_samples())
    assert torch.isfinite(out).all() and (out >= 0).all()
