"""Validate additional per-policy scalar settings without permitting different batch structures."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.settings import scalar_setting
from exploration_platform.atari_pixels.state import adam_step, initialize_adam, initialize_statistics
from exploration_platform.atari_pixels.training import generalized_advantages, normalize_intrinsic_rewards


def test_scalar_mapping_roundtrip_and_structural_rejection():
    """Accept clear mappings, preserve checkpoint identity and reject incompatible copy shapes."""
    configuration = PixelPPOConfig(n_copies=2, per_copy_scalar_settings={
        "entropy_coefficient": (0.001, 0.01), "learning_rate": (1e-4, 3e-4)})
    assert isinstance(configuration.per_copy_scalar_settings, tuple)
    assert configuration.per_copy_values("learning_rate") == (1e-4, 3e-4)
    restored = PixelPPOConfig(**json.loads(json.dumps(asdict(configuration))))
    assert restored == configuration
    with pytest.raises(ValueError, match="structural"):
        PixelPPOConfig(n_copies=2, per_copy_scalar_settings={"n_envs": (128, 256)})
    with pytest.raises(ValueError, match="only one"):
        replace(configuration, per_copy_learning_rates=(1e-4, 3e-4))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        PixelPPOConfig(n_copies=2, per_copy_scalar_settings={"gamma_ext": (0.99, 1.1)})
    with pytest.raises(ValueError, match="n_copies"):
        PixelPPOConfig(n_copies=2, per_copy_scalar_settings={"value_coefficient": (0.5,)})
    with pytest.raises(ValueError, match="float32"):
        PixelPPOConfig(per_copy_scalar_settings={"learning_rate": (1e100,)})
    with pytest.raises(ValueError, match="float32"):
        PixelPPOConfig(per_copy_scalar_settings={"adam_epsilon": (1e-100,)})


def test_discount_vectors_agree_with_separate_scalar_recurrences():
    """Each policy's GAE and intrinsic filter match the same settings run on that policy alone."""
    configuration = PixelPPOConfig(n_copies=2, n_envs=3, num_steps=4, num_minibatches=1,
        per_copy_scalar_settings={"gamma_ext": (.999, .9), "gamma_int": (.99, .8),
                                  "gae_lambda": (.95, .7)})
    rewards = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3) / 24
    rollout = SimpleNamespace(values_ext=rewards / 3, values_int=rewards / 7,
                               done_before=jnp.zeros(rewards.shape, jnp.bool_))
    final = (jnp.ones((2, 3), jnp.float32), jnp.ones((2, 3), jnp.float32) / 2)
    done = jnp.asarray([[False, True, False], [False, False, True]])
    actual = generalized_advantages(rewards, rewards / 2, rollout, final, done, configuration)
    state = SimpleNamespace(reward_statistics=initialize_statistics(2, ()),
                             reward_filter=jnp.ones((2, 4), jnp.float32) / 4)
    actual_rewards = normalize_intrinsic_rewards(rewards, state, configuration)
    for index in range(2):
        settings = {name: values[index] for name, values in configuration.per_copy_scalar_settings}
        scalar_configuration = replace(configuration, n_copies=1, per_copy_scalar_settings=(), **settings)
        scalar_rollout = SimpleNamespace(values_ext=rollout.values_ext[:, index:index + 1],
            values_int=rollout.values_int[:, index:index + 1], done_before=rollout.done_before[:, index:index + 1])
        expected = generalized_advantages(rewards[:, index:index + 1], rewards[:, index:index + 1] / 2,
            scalar_rollout, tuple(value[index:index + 1] for value in final), done[index:index + 1], scalar_configuration)
        for complete, single in zip(actual, expected):
            np.testing.assert_allclose(complete[:, index:index + 1], single, rtol=1e-6, atol=1e-6)
        scalar_state = SimpleNamespace(reward_statistics=initialize_statistics(1, ()),
                                       reward_filter=state.reward_filter[index:index + 1])
        expected_rewards = normalize_intrinsic_rewards(rewards[:, index:index + 1], scalar_state, scalar_configuration)
        np.testing.assert_allclose(actual_rewards[0][:, index:index + 1], expected_rewards[0], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(actual_rewards[2][index:index + 1], expected_rewards[2], rtol=1e-6, atol=1e-6)


def test_gradient_caps_and_adam_epsilon_broadcast_only_within_each_policy():
    """A zero cap disables clipping, while other policies retain their own cap and Adam epsilon."""
    configuration = PixelPPOConfig(n_copies=2, per_copy_scalar_settings={
        "max_gradient_norm": (.5, 0.), "adam_epsilon": (1e-5, .01)})
    parameters = {"weights": jnp.ones((2, 2, 3), jnp.float32)}
    gradients = {"weights": jnp.ones((2, 2, 3), jnp.float32) * 100}
    actual, moments, _ = adam_step(parameters, gradients, initialize_adam(parameters),
        jnp.ones((2,), jnp.float32) * 1e-4, scalar_setting(configuration, "max_gradient_norm"),
        scalar_setting(configuration, "adam_epsilon"))
    for index, cap, epsilon in ((0, .5, 1e-5), (1, 0., .01)):
        single_parameters = {"weights": parameters["weights"][index:index + 1]}
        single_gradients = {"weights": gradients["weights"][index:index + 1]}
        expected, expected_moments, _ = adam_step(single_parameters, single_gradients,
            initialize_adam(single_parameters), jnp.asarray([1e-4], jnp.float32), cap, epsilon)
        np.testing.assert_array_equal(actual["weights"][index:index + 1], expected["weights"])
        np.testing.assert_array_equal(moments.first["weights"][index:index + 1], expected_moments.first["weights"])
    assert float(moments.first["weights"][1, 0, 0]) == 10.0
