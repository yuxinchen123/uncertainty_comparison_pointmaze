"""Tests for the RND weight_init switch (convergence run 1).

Covers: the default reproduces the historical orthogonal init bit-for-bit; pytorch_default draws
the documented uniform law from keyed streams (deterministic in bias_seed, independent of the
global torch RNG); the bias schemes are unaffected by the weight scheme; and invalid or
unsupported combinations fail loud.
"""
import pytest
import torch

from rnd_exploration.methods.rnd import RND

OBS_DIM = 4


def make_rnd(**kw):
    """Construct a small standard single-predictor RND on CPU, overridable via kwargs."""
    params = dict(obs_shape=(OBS_DIM,), output_dim=16, device="cpu", use_obs_norm=False)
    params.update(kw)
    return RND(**params)


def _weights(model):
    """Return the four Linear weight tensors of a standard RND as a flat list."""
    return [model.target.network[0].weight, model.target.network[2].weight,
            model.predictor.network[0].weight, model.predictor.network[2].weight]


def test_default_orthogonal_is_bit_identical_to_plain_construction():
    """weight_init='orthogonal' (and the omitted default) leave every parameter bit-identical."""
    # same global torch seed -> the orthogonal draws consume the global RNG identically
    torch.manual_seed(0)
    plain = make_rnd()
    torch.manual_seed(0)
    explicit = make_rnd(weight_init="orthogonal")
    for a, b in zip(_weights(plain), _weights(explicit)):
        assert torch.equal(a, b)
    assert plain.weight_init == "orthogonal"


def test_pytorch_default_law_and_keyed_determinism():
    """pytorch_default weights follow U(-1/sqrt(fan_in), +1/sqrt(fan_in)) and depend only on
    bias_seed (not on the global torch RNG); different seeds and nets give different draws."""
    torch.manual_seed(0)
    m1 = make_rnd(weight_init="pytorch_default", bias_seed=7)
    torch.manual_seed(999)  # different global stream, same keyed seed
    m2 = make_rnd(weight_init="pytorch_default", bias_seed=7)
    torch.manual_seed(0)
    m3 = make_rnd(weight_init="pytorch_default", bias_seed=8)  # different keyed seed
    # keyed determinism: identical across global seeds, different across bias_seed values
    for w1, w2, w3 in zip(_weights(m1), _weights(m2), _weights(m3)):
        assert torch.equal(w1, w2)
        assert not torch.equal(w1, w3)
    # the uniform bound per layer: fan_in=4 -> 0.5; fan_in=256 -> 0.0625
    for w, bound in [(m1.target.network[0].weight, 0.5), (m1.target.network[2].weight, 0.0625),
                     (m1.predictor.network[0].weight, 0.5), (m1.predictor.network[2].weight, 0.0625)]:
        assert w.abs().max().item() <= bound + 1e-9
        assert w.abs().max().item() > 0.5 * bound  # actually filling the range, not degenerate
    # target and predictor draw from different keyed streams (net name is part of the key)
    assert not torch.equal(m1.target.network[0].weight, m1.predictor.network[0].weight)
    # and the law differs from the orthogonal default (orthogonal rows have unit-ish norms * sqrt(2))
    torch.manual_seed(0)
    orth = make_rnd(bias_seed=7)
    assert not torch.equal(orth.target.network[0].weight, m1.target.network[0].weight)


def test_bias_schemes_unaffected_by_weight_scheme():
    """The keyed bias draws are identical whichever weight scheme is active; zero stays zero."""
    torch.manual_seed(0)
    m_orth = make_rnd(bias_init="normal_0.5", bias_seed=3)
    torch.manual_seed(0)
    m_pt = make_rnd(weight_init="pytorch_default", bias_init="normal_0.5", bias_seed=3)
    for net in ("target", "predictor"):
        for idx in (0, 2):
            b1 = getattr(m_orth, net).network[idx].bias
            b2 = getattr(m_pt, net).network[idx].bias
            assert torch.equal(b1, b2)
    # zero-bias default stays exactly zero under pytorch_default weights
    torch.manual_seed(0)
    m_zero = make_rnd(weight_init="pytorch_default")
    for w in [m_zero.target.network[0].bias, m_zero.target.network[2].bias,
              m_zero.predictor.network[0].bias, m_zero.predictor.network[2].bias]:
        assert torch.equal(w, torch.zeros_like(w))


def test_invalid_and_unsupported_weight_init_raise():
    """Unknown scheme names and the non-standard architectures fail loud."""
    with pytest.raises(ValueError):
        make_rnd(weight_init="kaiming")
    with pytest.raises(ValueError):
        make_rnd(weight_init="pytorch_default", linear_rnd=True)
    with pytest.raises(ValueError):
        make_rnd(weight_init="pytorch_default", n_predictors=2)
