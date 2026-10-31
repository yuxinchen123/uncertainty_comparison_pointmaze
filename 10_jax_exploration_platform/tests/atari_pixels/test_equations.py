"""Numerical gates against the untouched original CleanRL classes; run on a compute node."""

import ast
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.networks import (categorical_statistics, policy_forward,
                                                        quantity_keys, rnd_forward)
from exploration_platform.atari_pixels.state import (adam_step, initialize_adam,
    initialize_statistics, update_observation_statistics)
from exploration_platform.atari_pixels.training import minibatch_loss, normalize_intrinsic_rewards


def original_classes():
    """Compile only the original model classes, avoiding its unrelated EnvPool imports."""
    source = Path(os.environ.get("CLEANRL_REFERENCE_FILE",
        "/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/cleanrl/ppo_rnd_envpool.py"))
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in {"layer_init", "Agent", "RNDModel"}]
    assert len(definitions) == 3
    namespace = {"np": np, "torch": torch, "nn": nn, "Categorical": Categorical}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Agent"], namespace["RNDModel"]


def layer_arrays(layer, gradients=False):
    """Convert original OIHW/dense weights or gradients into the learner's copy-first layout."""
    weight = layer.weight.grad if gradients else layer.weight
    bias = layer.bias.grad if gradients else layer.bias
    weight, bias = weight.detach().cpu().numpy(), bias.detach().cpu().numpy()
    # Before: [out,in,H,W]; after: [one copy,H,W,in,out]. Dense becomes [one copy,in,out].
    weight = weight.transpose(2, 3, 1, 0) if weight.ndim == 4 else weight.T
    return {"weight": jnp.asarray(weight[None]), "bias": jnp.asarray(bias[None])}


def network_arrays(agent, rnd, gradients=False):
    """Extract precisely the upstream policy/predictor tensors and frozen target."""
    def layer(module):
        """Convert one original module using the selected parameter/gradient source."""
        return layer_arrays(module, gradients)

    policy = {"convolutions": tuple(layer(agent.network[i]) for i in (0, 2, 4)),
              "dense": tuple(layer(agent.network[i]) for i in (7, 9)),
              "extra": layer(agent.extra_layer[0]),
              "actor": tuple(layer(agent.actor[i]) for i in (0, 2)),
              "value_ext": layer(agent.critic_ext), "value_int": layer(agent.critic_int)}
    predictor = {"convolutions": tuple(layer(rnd.predictor[i]) for i in (0, 2, 4)),
                 "dense": tuple(layer(rnd.predictor[i]) for i in (7, 9, 11))}
    target = (None if gradients else {
        "convolutions": tuple(layer(rnd.target[i]) for i in (0, 2, 4)),
        "dense": (layer(rnd.target[7]),)})
    return {"agent": policy, "predictor": predictor}, target


@pytest.fixture(scope="module")
def reference():
    """Initialize the actual original CleanRL networks once, on the allocated compute node."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    agent_class, rnd_class = original_classes()
    agent = agent_class(SimpleNamespace(single_action_space=SimpleNamespace(n=18)))
    rnd = rnd_class(4, 18)
    return agent, rnd


def fixture_batch(reference, cache=False):
    """Use deterministic varied inputs that exercise clipped values, normalization and masks."""
    agent, rnd = reference
    parameters, target = network_arrays(agent, rnd)
    images = (np.arange(4 * 4 * 84 * 84).reshape(1, 4, 4, 84, 84) % 256).astype(np.uint8)
    normalized = ((np.arange(4 * 84 * 84).reshape(1, 4, 1, 84, 84) % 97) - 48).astype(np.float32) / 12.0
    logits, values_ext, values_int = policy_forward(parameters["agent"], jnp.asarray(images))
    actions = jnp.asarray([[0, 4, 7, 17]], jnp.int32)
    old_log, _ = categorical_statistics(logits, actions)
    batch = {"observations": jnp.asarray(images), "normalized_frames": jnp.asarray(normalized),
             "actions": actions, "log_probabilities": old_log + jnp.asarray([[-0.3, 0.2, -0.1, 0.4]]),
             "values_ext": values_ext, "advantage_ext": jnp.asarray([[-1., 2., .5, -.25]], jnp.float32),
             "advantage_int": jnp.asarray([[.3, .1, -.2, .4]], jnp.float32),
             "return_ext": values_ext + jnp.asarray([[.4, -.5, .8, -.1]], jnp.float32),
             "return_int": values_int + jnp.asarray([[.2, -.7, .3, .6]], jnp.float32)}
    if cache:
        batch["target_features"] = rnd_forward(target, batch["normalized_frames"])
    # Explicit float32 avoids x64-enabled weak-scalar promotion in the test fixture itself.
    batch = jax.tree.map(lambda value: value.astype(jnp.float32) if value.dtype.kind == "f" else value, batch)
    return parameters, target, batch


def torch_loss(reference, batch, mask):
    """Evaluate the original loss statements with supplied actions and predictor mask."""
    agent, rnd = reference
    values = {name: torch.from_numpy(np.array(value[0])) for name, value in batch.items()}
    _, log_probability, entropy, value_ext, value_int = agent.get_action_and_value(
        values["observations"].float(), values["actions"].long())
    ratio = (log_probability - values["log_probabilities"]).exp()
    advantages = values["advantage_ext"] * 2.0 + values["advantage_int"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = torch.maximum(-advantages * ratio, -advantages * torch.clamp(ratio, .9, 1.1)).mean()
    value_ext, value_int = value_ext[:, 0], value_int[:, 0]
    clipped_value = values["values_ext"] + torch.clamp(value_ext - values["values_ext"], -.1, .1)
    value_loss = .5 * torch.maximum((value_ext - values["return_ext"]) ** 2,
                                    (clipped_value - values["return_ext"]) ** 2).mean()
    value_loss = value_loss + .5 * ((value_int - values["return_int"]) ** 2).mean()
    predictor, target = rnd(values["normalized_frames"])
    forward_errors = ((predictor - target.detach()) ** 2).mean(-1)
    mask = torch.from_numpy(np.array(mask[0]))
    predictor_loss = (forward_errors * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return policy_loss - .001 * entropy.mean() + .5 * value_loss + predictor_loss


def test_forward_against_original_cleanrl(reference):
    """All policy outputs and both RND networks match the untouched original model classes."""
    agent, rnd = reference
    parameters, target, batch = fixture_batch(reference)
    observations = torch.from_numpy(np.array(batch["observations"][0])).float()
    actions = torch.from_numpy(np.array(batch["actions"][0])).long()
    with torch.no_grad():
        _, log_probability, entropy, value_ext, value_int = agent.get_action_and_value(observations, actions)
        predictor, frozen = rnd(torch.from_numpy(np.array(batch["normalized_frames"][0])))
    logits, got_ext, got_int = policy_forward(parameters["agent"], batch["observations"])
    got_log_probability, got_entropy = categorical_statistics(logits, batch["actions"])
    pairs = ((got_log_probability[0], log_probability), (got_entropy[0], entropy),
             (got_ext[0], value_ext[:, 0]), (got_int[0], value_int[:, 0]),
             (rnd_forward(parameters["predictor"], batch["normalized_frames"])[0], predictor),
             (rnd_forward(target, batch["normalized_frames"])[0], frozen))
    for actual, expected in pairs:
        np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=3e-4, atol=1e-5)


@pytest.mark.parametrize("cache", [False, True])
def test_loss_gradients_and_adam_against_original(reference, cache):
    """Compare losses, every trainable gradient, joint clipping and one Adam step."""
    agent, rnd = reference
    parameters, target, batch = fixture_batch(reference, cache)
    configuration = PixelPPOConfig(n_envs=2, num_steps=2, num_minibatches=1,
                                    update_epochs=1, cache_target_features=cache)
    mask = jnp.asarray([[1., 0., 1., 0.]], jnp.float32)
    combined = list(agent.parameters()) + list(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(combined, lr=1e-4, eps=1e-5)
    optimizer.zero_grad()
    expected_loss = torch_loss(reference, batch, mask)
    expected_loss.backward()
    (_, actual_metrics), actual_gradients = jax.value_and_grad(minibatch_loss, has_aux=True)(
        parameters, target, batch, mask, configuration, jnp.ones((1,), jnp.float32))
    expected_gradients, _ = network_arrays(agent, rnd, gradients=True)
    np.testing.assert_allclose(float(actual_metrics["loss"][0]), expected_loss.item(), rtol=3e-4, atol=1e-5)
    for actual, expected in zip(jax.tree.leaves(actual_gradients), jax.tree.leaves(expected_gradients)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-3, atol=3e-5)
    next_parameters, _, _ = adam_step(parameters, actual_gradients, initialize_adam(parameters),
                                     jnp.asarray([1e-4], jnp.float32), .5, 1e-5)
    torch.nn.utils.clip_grad_norm_(combined, .5)
    optimizer.step()
    expected_parameters, _ = network_arrays(agent, rnd)
    for actual, expected in zip(jax.tree.leaves(next_parameters), jax.tree.leaves(expected_parameters)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-3, atol=2e-5)


def test_empty_predictor_mask(reference):
    """An all-zero predictor mask produces a finite zero predictor loss."""
    parameters, target, batch = fixture_batch(reference)
    configuration = PixelPPOConfig(n_envs=2, num_steps=2, num_minibatches=1)
    _, metrics = minibatch_loss(parameters, target, batch, jnp.zeros((1, 4), jnp.float32),
                                configuration, jnp.ones((1,), jnp.float32))
    assert float(metrics["predictor_loss"][0]) == 0.0
    assert np.isfinite(float(metrics["loss"][0]))


def test_independent_clipping_and_key_stability():
    """Changing one policy cannot change another policy's update or keyed random quantities."""
    parameters = {"weights": jnp.asarray([[1., 2.], [3., 4.]], jnp.float32)}
    gradients = {"weights": jnp.asarray([[.1, .2], [100., 200.]], jnp.float32)}
    updated, _, _ = adam_step(parameters, gradients, initialize_adam(parameters),
                              jnp.asarray([1e-4, 1e-4], jnp.float32), .5, 1e-5)
    gradients["weights"] = gradients["weights"].at[1].set(0.)
    changed, _, _ = adam_step(parameters, gradients, initialize_adam(parameters),
                              jnp.asarray([1e-4, 1e-4], jnp.float32), .5, 1e-5)
    np.testing.assert_array_equal(updated["weights"][0], changed["weights"][0])
    np.testing.assert_array_equal(quantity_keys(jnp.asarray([17]), "actions", 3)[0],
                                   quantity_keys(jnp.asarray([17, 33]), "actions", 3)[0])
    assert not np.array_equal(quantity_keys(jnp.asarray([17]), "actions", 3),
                              quantity_keys(jnp.asarray([17]), "predictor-mask", 3))


def test_statistics_and_original_reward_filter():
    """Pin float64 per-pixel moments and upstream's unusual filter axis/sample count."""
    statistics = initialize_statistics(2, (1, 84, 84))
    frames = jnp.broadcast_to(jnp.asarray([[[[[0.]]], [[[2.]]]], [[[[4.]]], [[[6.]]]]], jnp.uint8),
                              (2, 2, 1, 84, 84))
    statistics = update_observation_statistics(statistics, frames)
    assert statistics.mean.dtype == jnp.float64
    np.testing.assert_allclose(statistics.count, [2.0001, 2.0001])
    np.testing.assert_allclose(np.asarray(statistics.mean[:, 0, 0, 0]),
                               np.asarray([2., 10.]) / 2.0001, rtol=1e-12)
    configuration = PixelPPOConfig(n_copies=2, n_envs=2, num_steps=3, num_minibatches=1)
    rewards = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2) + 1
    fake_state = SimpleNamespace(reward_statistics=initialize_statistics(2, ()),
                                 reward_filter=jnp.zeros((2, 3), jnp.float32))
    normalized, reward_statistics, filtered = normalize_intrinsic_rewards(rewards, fake_state, configuration)
    expected_filtered = np.asarray(rewards).transpose(2, 1, 0).copy()
    expected_filtered[1] += .99 * expected_filtered[0]
    np.testing.assert_allclose(filtered, expected_filtered[-1], rtol=1e-6)
    np.testing.assert_allclose(reward_statistics.count, [2.0001, 2.0001])
    assert np.isfinite(np.asarray(normalized)).all()


def test_configuration_separates_original_atari_defaults():
    """Keep the original Atari values and reject invalid fixed minibatch shapes."""
    configuration = PixelPPOConfig()
    assert configuration.batch_size == 16384 and configuration.minibatch_size == 4096
    assert configuration.total_iterations == 122070
    assert not configuration.cache_target_features and not configuration.hoist_rollout_bonus
    assert configuration.entropy_coefficient == .001 and configuration.learning_rate == 1e-4
    with pytest.raises(ValueError, match="minibatches"):
        replace(configuration, n_envs=1, num_steps=1)
