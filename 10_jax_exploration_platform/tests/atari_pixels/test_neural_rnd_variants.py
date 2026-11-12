"""CPU-safe tests for count-free pixel-RND exploration ablations.

Use PYTHONPATH=10_jax_exploration_platform/src pytest -q tests/atari_pixels/test_neural_rnd_variants.py
No Atari ROM/sprite pack, remote GPU or Slurm access is required.
"""
from dataclasses import asdict
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.rnd_variants import (
    target_features, predictor_error, initial_bonus)
from exploration_platform.atari_pixels.state import AdamState, adam_step, initialize_adam


def test_uniform_start_is_exact_for_nonzero_frozen_target_outputs():
    features = jnp.asarray([[[2., 0., 0.], [0., -5., 0.]],
                            [[1., 2., 2.], [1., -2., 3.]]])
    zeros = jnp.zeros_like(features)
    normalized = target_features(features, uniform_start=True)
    np.testing.assert_allclose(initial_bonus(zeros, normalized),
                               np.full((2, 2), 0.5), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(target_features(features), features)


def test_loss_default_is_cleanrl_mean_feature_mse():
    target = jnp.asarray([[[0.25, 1., -2.], [2., 1., 0.]]])
    guess = jnp.asarray([[[-1., 1.5, 0.], [-2., 0., 1.]]])
    expected = jnp.square(guess - target).mean(axis=-1)
    np.testing.assert_array_equal(predictor_error(guess, target), expected)


@pytest.mark.parametrize("exponent", (2., 3.))
def test_power_loss_is_count_free_and_gives_finite_gradients(exponent):
    target = jnp.array([[[.2, -.5, .1], [0.4, 0.1, -.3]]])
    guess = jnp.zeros_like(target)
    fun = lambda x: predictor_error(x, target, mode="power", exponent=exponent).sum()
    gradient = jax.grad(fun)(guess)
    assert np.isfinite(np.asarray(gradient)).all()
    sq = np.sum(np.square(np.asarray(target - guess)), axis=-1)
    np.testing.assert_allclose(fun(guess), np.sum(sq ** exponent / exponent))


@pytest.mark.parametrize("a", (0.5, 1., 2.))
def test_shrink_regression_exact_fit_one_step_has_analytical_residual(a):
    # The exact-fit law applies ONLY to one independently fitted state; the
    # shared CNN is not guaranteed to show the same decay.
    target = jnp.array([[[1., 0., 0.]]])
    guess = jnp.zeros_like(target)
    e0 = np.sum(np.square(np.asarray(target - guess)))
    q = (1 + e0 ** (1 / a)) ** (-a / 2)
    exact_next_guess = target - q * (target - guess)
    loss = predictor_error(guess, target, mode="shrink", decay_exponent=a)
    np.testing.assert_allclose(loss, np.array([[((1-q)**2)/3]]), rtol=2e-6)
    np.testing.assert_allclose(
        initial_bonus(exact_next_guess, target),
        0.5 * e0 / (1 + e0 ** (1/a)) ** a, rtol=1e-6)


def test_shrink_regression_target_is_detached_from_predictor():
    target = jnp.array([[[1., 0.]]])
    p0 = jnp.array([[[0.2, 0.1]]])
    gradient = jax.grad(lambda p: predictor_error(
        p, target, mode="shrink", decay_exponent=1.).sum())(p0)
    e = target - p0
    q = (1 + jnp.sum(e * e, axis=-1, keepdims=True)) ** -0.5
    # Mean squared error on 2 output features.  This is the gradient
    # treating the adaptive regression target as a CONSTANT.
    expected = -(1 - q) * e
    np.testing.assert_allclose(gradient, expected, rtol=3e-6, atol=2e-7)


def test_sgd_predictor_does_not_change_policy_adam_update():
    params = {"agent": {"w": jnp.array([[0.1, 0.3]])},
              "predictor": {"w": jnp.array([[0.2, -0.1]])}}
    grads = {"agent": {"w": jnp.array([[1., -2.]])},
             "predictor": {"w": jnp.array([[0.4, -0.6]])}}
    opt = initialize_adam(params)
    args = (params, grads, opt, jnp.asarray([1e-4]), 100., 1e-5)
    default, _, norm1 = adam_step(*args)
    alt, _, norm2 = adam_step(*args, predictor_optimizer="sgd",
                              predictor_learning_rate=1e-2)
    np.testing.assert_array_equal(default["agent"]["w"], alt["agent"]["w"])
    np.testing.assert_array_equal(norm1, norm2)
    np.testing.assert_allclose(alt["predictor"]["w"],
        params["predictor"]["w"] - .01 * grads["predictor"]["w"], rtol=1e-6)


def test_1b_screen_preserves_2b_lr_annealing_horizon_and_identity():
    c = PixelPPOConfig(n_copies=4, rnd_uniform_start=True,
                       rnd_predictor_loss="shrink", rnd_predictor_optimizer="sgd")
    assert c.total_iterations == 122070
    assert 61035 * c.batch_size == 999997440
    assert 61035 < c.total_iterations
    same = PixelPPOConfig(**json.loads(json.dumps(asdict(c))))
    assert same == c
    with pytest.raises(ValueError, match="named parameter"):
        PixelPPOConfig(rnd_predictor_optimizer="sgd", flat_parameters=True)
    with pytest.raises(ValueError, match="rnd_predictor_loss"):
        PixelPPOConfig(rnd_predictor_loss="count")
    with pytest.raises(ValueError, match="rnd_decay_exponent"):
        PixelPPOConfig(rnd_decay_exponent=-1)
