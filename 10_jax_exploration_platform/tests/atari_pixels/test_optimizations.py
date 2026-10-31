"""Independent numerical checks of layouts, state packing and equivalent scan forms."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.networks import convolution, dense
from exploration_platform.atari_pixels.state import (FlatParameterLayout, adam_step,
    initialize_adam, initialize_statistics, update_observation_statistics)
from exploration_platform.atari_pixels.training import generalized_advantages, normalize_intrinsic_rewards


@pytest.mark.parametrize("layout,batching", [("nchw", "vmap"), ("nhwc", "vmap"),
                                             ("nchw", "grouped"), ("nhwc", "grouped")])
def test_convolution_layouts_and_copy_isolation(layout, batching):
    """Equivalent convolution layouts retain exact policy separation at two independent copies."""
    images = (jnp.arange(2 * 3 * 12 * 12 * 2, dtype=jnp.float32) % 31).reshape(2, 3, 12, 12, 2) / 31
    layer = {"weight": jnp.arange(2 * 3 * 3 * 2 * 4, dtype=jnp.float32).reshape(2, 3, 3, 2, 4) / 1000,
             "bias": jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 100}
    reference = convolution(layer, images, 2, "nchw", "vmap")
    actual = convolution(layer, images, 2, layout, batching)
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    changed = convolution(layer, images.at[1].set(100), 2, layout, batching)
    np.testing.assert_array_equal(actual[0], changed[0])


def test_dense_forms_and_flat_parameter_layout():
    """Dense alternatives agree and packing/unpacking preserves every named tensor."""
    values = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 24
    layer = {"weight": jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 40,
             "bias": jnp.zeros((2, 5), jnp.float32)}
    np.testing.assert_allclose(dense(layer, values, "matmul"), dense(layer, values, "einsum"),
                               rtol=1e-6, atol=1e-6)
    parameters = {"layer": layer, "extra": jnp.ones((2, 3), jnp.float32)}
    layout = FlatParameterLayout(parameters)
    restored = layout.unpack(layout.pack(parameters))
    for expected, actual in zip(jax.tree.leaves(parameters), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(actual, expected)
    gradients = jax.tree.map(lambda value: value * 3 + .1, parameters)
    rates = jnp.asarray([1e-4, 2e-4], jnp.float32)
    reference, _, norms = adam_step(parameters, gradients, initialize_adam(parameters), rates, .5, 1e-5)
    fused, _, fused_norms = adam_step(parameters, gradients, initialize_adam(parameters), rates, .5, 1e-5,
                                     fuse_optimizer=True)
    for expected, actual in zip(jax.tree.leaves(reference), jax.tree.leaves(fused)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fused_norms, norms, rtol=1e-6)


def test_associative_advantages_and_reward_filter():
    """Parallel recurrence composition agrees at masked boundaries and nonzero filter state."""
    configuration = PixelPPOConfig(n_copies=2, n_envs=2, num_steps=4, num_minibatches=1)
    rewards = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2) / 16
    rollout = SimpleNamespace(values_ext=rewards / 3, values_int=rewards / 7,
        done_before=jnp.asarray([[[0, 0], [0, 0]], [[0, 1], [0, 0]],
                                 [[0, 0], [1, 0]], [[0, 0], [0, 0]]], jnp.bool_))
    final_values = (jnp.ones((2, 2), jnp.float32), jnp.ones((2, 2), jnp.float32) / 2)
    done = jnp.asarray([[False, True], [False, False]])
    expected = generalized_advantages(rewards, rewards / 2, rollout, final_values, done, configuration)
    actual = generalized_advantages(rewards, rewards / 2, rollout, final_values, done,
                                     replace(configuration, associative_gae=True))
    for before, after in zip(expected, actual):
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)
    state = SimpleNamespace(reward_statistics=initialize_statistics(2, ()),
                             reward_filter=jnp.ones((2, 4), jnp.float32) * .25)
    expected = normalize_intrinsic_rewards(rewards, state, configuration)
    actual = normalize_intrinsic_rewards(rewards, state, replace(configuration, associative_reward_filter=True))
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)


def test_float64_raw_moments_match_two_pass_on_pixel_values():
    """The alternative moment reduction retains float64 accuracy even for constant pixels."""
    frames = (jnp.arange(2 * 5 * 84 * 84).reshape(2, 5, 1, 84, 84) % 256).astype(jnp.uint8)
    for value in (frames, jnp.ones_like(frames) * 255):
        statistics = initialize_statistics(2, (1, 84, 84))
        expected = update_observation_statistics(statistics, value, "two_pass")
        actual = update_observation_statistics(statistics, value, "raw_moments")
        for before, after in zip(expected, actual):
            np.testing.assert_allclose(after, before, rtol=1e-11, atol=1e-11)
