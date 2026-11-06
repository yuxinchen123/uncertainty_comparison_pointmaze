"""The predictor evaluated on its selected rows only gives the all-rows loss and gradients."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.networks import initialize_networks
from exploration_platform.atari_pixels.training import minibatch_loss

jax.config.update("jax_default_matmul_precision", "highest")


def synthetic_batch(copies, rows, key):
    keys = jax.random.split(key, 10)
    shape = (copies, rows)
    return {"observations": jax.random.randint(keys[0], shape + (4, 84, 84), 0, 256).astype(jnp.uint8),
            "actions": jax.random.randint(keys[1], shape, 0, 18),
            "log_probabilities": jax.random.normal(keys[2], shape, jnp.float32) * 0.1 - 2.9,
            "advantage_ext": jax.random.normal(keys[3], shape, jnp.float32),
            "advantage_int": jax.random.normal(keys[4], shape, jnp.float32),
            "values_ext": jax.random.normal(keys[5], shape, jnp.float32),
            "return_ext": jax.random.normal(keys[6], shape, jnp.float32),
            "return_int": jax.random.normal(keys[7], shape, jnp.float32),
            "normalized_frames": jax.random.normal(keys[8], shape + (1, 84, 84), jnp.float32)}


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("selected_rows,probability", [(96, 0.25), (16, 0.25), (96, 0.0)])
def test_selected_rows_match_all_rows(copies, selected_rows, probability):
    """96 rows hold every selected row of 256 (p = 0.25); 16 rows force the exact fallback."""
    rows = 256
    parameters, target = initialize_networks(tuple(range(1, copies + 1)), 18)
    batch = synthetic_batch(copies, rows, jax.random.PRNGKey(3))
    mask = (jax.random.uniform(jax.random.PRNGKey(4), (copies, rows)) < probability).astype(jnp.float32)
    coefficients = jnp.ones((copies,), jnp.float32)

    def gradient(config):
        loss = lambda values: minibatch_loss(values, target, batch, mask, config, coefficients)[0]
        return jax.jit(jax.value_and_grad(loss))(parameters)

    all_value, all_gradient = gradient(PixelPPOConfig(n_copies=copies))
    some_value, some_gradient = gradient(PixelPPOConfig(n_copies=copies, predictor_selected_rows=selected_rows))
    np.testing.assert_allclose(float(some_value), float(all_value), rtol=1e-5, atol=1e-6)
    for expected, actual in zip(jax.tree.leaves(all_gradient), jax.tree.leaves(some_gradient)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("copies", [1, 2])
def test_merged_head_layers_match_separate_layers(copies):
    """Merged actor/critic and value-head matrix products give the same outputs and gradients."""
    from exploration_platform.atari_pixels.networks import policy_forward

    parameters, _ = initialize_networks(tuple(range(1, copies + 1)), 18)
    observations = jax.random.randint(jax.random.PRNGKey(6), (copies, 32, 4, 84, 84), 0, 256).astype(jnp.uint8)

    def total(values, config):
        logits, ext, intr = policy_forward(values["agent"], observations, config)
        return jnp.sum(jnp.sin(logits)) + jnp.sum(ext ** 2) + jnp.sum(jnp.cos(intr))

    separate = jax.value_and_grad(total)(parameters, PixelPPOConfig(n_copies=copies))
    merged = jax.value_and_grad(total)(parameters, PixelPPOConfig(n_copies=copies, merged_head_layers=True))
    np.testing.assert_allclose(float(merged[0]), float(separate[0]), rtol=1e-5, atol=1e-5)
    for expected, actual in zip(jax.tree.leaves(separate[1]), jax.tree.leaves(merged[1])):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-5)
