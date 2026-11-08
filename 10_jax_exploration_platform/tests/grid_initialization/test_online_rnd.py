"""Online RND, input statistics and final-policy evaluation; run only on an allocated compute node."""

import jax
import jax.numpy as jnp
import numpy as np

from exploration_platform.atari_pixels.state import initialize_statistics, update_observation_statistics
from exploration_platform.grid_initialization.config import GridPPOConfig
from exploration_platform.grid_initialization.environments import make_geometry
from exploration_platform.grid_initialization.online_rnd import (
    frozen_network_field, numpy_forward, network_weights, normalize_coordinates, stack_networks,
    warmup_statistics)
from exploration_platform.grid_initialization.training import GridTrainer, minibatch_loss

jax.config.update("jax_enable_x64", True)


def online_trainer(normalize, policy_kind="tabular", copies=2, width=8):
    """Build a small online-RND trainer whose copies use field seeds 0 and 1."""
    world = make_geometry("straight", width)
    episode_length = 64
    field_seeds = np.arange(copies) % 2
    rnd = stack_networks(field_seeds)
    statistics = [warmup_statistics(world, episode_length, int(seed)) for seed in field_seeds]
    if normalize:
        rnd["statistics"] = {name: np.stack([np.asarray(row[name], np.float64) for row in statistics])
                             for name in ("mean", "variance", "count")}
    configuration = GridPPOConfig(n_copies=copies, policy_kind=policy_kind, hidden_width=8, num_steps=16,
        num_minibatches=2, update_epochs=2, total_timesteps=256, episode_length=episode_length,
        intrinsic_source="online_rnd", normalize_rnd_inputs=normalize)
    fields = np.ones((copies, world.state_count), np.float32)
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates, fields, rnd)
    return world, trainer, trainer.initialize(np.arange(copies) + 11), statistics, field_seeds


def test_step_zero_error_equals_frozen_field_raw_and_whitened():
    """At step 0 the online error of every cell equals the frozen field before mean scaling."""
    for normalize in (False, True):
        world, trainer, state, statistics, seeds = online_trainer(normalize)
        errors = np.asarray(trainer.all_cell_errors(state))
        for copy, seed in enumerate(seeds):
            field = frozen_network_field(world, int(seed), statistics[copy] if normalize else None)
            np.testing.assert_allclose(errors[copy], field.raw, rtol=2e-5)
            np.testing.assert_allclose(field.normalized.mean(), 1.0, rtol=1e-6)


def test_frozen_field_weights_match_original_generator_seeds():
    """The processor weights use the same named seeds and shapes as the original field generator."""
    weights = network_weights(3)
    assert [w.shape for w in weights["target"]] == [(2, 32), (32, 64), (64, 64), (64, 512)]
    assert [w.shape for w in weights["predictor"]][-3:] == [(64, 512), (512, 512), (512, 512)]
    # Orthogonal with gain sqrt(2): W^T W = 2 I for every tall or square layer.
    np.testing.assert_allclose(weights["predictor"][4].T @ weights["predictor"][4], 2 * np.eye(512), atol=1e-4)
    np.testing.assert_array_equal(network_weights(3)["target"][0], weights["target"][0])


def test_observation_statistics_match_numpy_running_mean():
    """The per-copy statistics update equals gym RunningMeanStd's parallel formula."""
    values = np.random.default_rng(0).normal(size=(3, 16, 2))
    statistics = update_observation_statistics(initialize_statistics(3, (2,)), jnp.asarray(values))
    count = 1e-4 + 16
    delta = values.mean(1)
    mean = delta * 16 / count
    variance = (1e-4 + values.var(1) * 16 + delta ** 2 * 1e-4 * 16 / count) / count
    np.testing.assert_allclose(statistics.mean, mean, rtol=1e-12)
    np.testing.assert_allclose(statistics.variance, variance, rtol=1e-12)


def test_warmup_is_seeded_by_field_and_clips_inputs():
    """Warm-up statistics depend only on the field seed, and whitened inputs stay within [-5, 5]."""
    world = make_geometry("snake", 4, 2)
    first, second = warmup_statistics(world, 64, 2), warmup_statistics(world, 64, 2)
    np.testing.assert_array_equal(first["mean"], second["mean"])
    assert first["count"] == 50 * 128 + 1e-4
    inputs = normalize_coordinates(world.coordinates, first["mean"], first["variance"])
    assert np.abs(inputs).max() <= 5.0


def test_online_iteration_trains_predictor_keeps_target_and_updates_statistics():
    """One iteration changes the predictor and statistics, never the target, and logs its loss."""
    for policy_kind in ("tabular", "neural"):
        world, trainer, state, _, _ = online_trainer(True, policy_kind)
        after, metrics = trainer.train_iteration(state)
        before_predictor = jax.tree.leaves(state.parameters["predictor"])
        after_predictor = jax.tree.leaves(after.parameters["predictor"])
        assert any(not np.allclose(a, b) for a, b in zip(before_predictor, after_predictor))
        for a, b in zip(jax.tree.leaves(state.target), jax.tree.leaves(after.target)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(after.observation_statistics.count, state.observation_statistics.count + 16)
        assert np.all(np.asarray(metrics["predictor_loss"]) > 0)
        # The bonus of each arrival is the cell's error before the update, with no count division.
        errors = np.asarray(trainer.all_cell_errors(state))
        _, rollout = trainer.capture(state)
        arrivals = np.asarray(rollout.arrivals)
        expected = errors[np.arange(2)[None], arrivals]
        np.testing.assert_allclose(rollout.intrinsic_rewards, expected, rtol=1e-6)


def test_predictor_loss_is_masked_mean_squared_feature_error():
    """The added loss equals CleanRL's masked mean of per-sample mean squared feature errors."""
    world, trainer, state, _, seeds = online_trainer(False)
    configuration = trainer.configuration
    states = jnp.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], jnp.int32)
    mask = jnp.asarray([[True, False, True, False], [False, False, False, True]])
    inputs = jnp.asarray(world.coordinates)[states]
    batch = {"states": states, "actions": jnp.zeros((2, 4), jnp.int32),
             "log_probabilities": jnp.full((2, 4), -np.log(4), jnp.float32),
             "values_ext": jnp.zeros((2, 4)), "advantage_ext": jnp.arange(8.).reshape(2, 4),
             "advantage_int": jnp.zeros((2, 4)), "return_ext": jnp.zeros((2, 4)),
             "return_int": jnp.zeros((2, 4)), "rnd_inputs": inputs, "predictor_mask": mask}
    _, diagnostics = minibatch_loss(state.parameters, batch, trainer.coordinates, configuration, state.target)
    for copy, seed in enumerate(seeds):
        weights = network_weights(int(seed))
        rows = np.asarray(inputs[copy])
        errors = ((numpy_forward(weights["predictor"], rows) - numpy_forward(weights["target"], rows)) ** 2).mean(-1)
        kept = np.asarray(mask[copy])
        np.testing.assert_allclose(diagnostics["predictor_loss"][copy], errors[kept].mean(), rtol=2e-5)


def test_count_configuration_rejects_rnd_networks_and_input_normalization():
    """Switches that act only on online RND are refused for count bonuses."""
    try:
        GridPPOConfig(normalize_rnd_inputs=True)
    except ValueError:
        pass
    else:
        raise AssertionError("count bonuses must refuse RND input normalization")


def test_evaluation_counts_goal_steps_and_failures():
    """A policy that always moves right reaches the goal in width-1 steps; always left never does."""
    world = make_geometry("straight", 4)
    configuration = GridPPOConfig(n_copies=2, num_steps=8, num_minibatches=2, update_epochs=1,
                                  total_timesteps=8, episode_length=16)
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates,
                          np.ones((2, 4), np.float32))
    state = trainer.initialize([1, 2])
    table = np.zeros((2, 4, 6), np.float32)
    table[0, :, 1] = 100.0   # right
    table[1, :, 3] = 100.0   # left
    state = state._replace(parameters={"table": jnp.asarray(table)})
    greedy = np.asarray(trainer.evaluate(state, 1, True))
    sampled = np.asarray(trainer.evaluate(state, 32, False))
    np.testing.assert_array_equal(greedy[:, 0], [3, -1])
    np.testing.assert_array_equal(sampled[0], np.full(32, 3))
    np.testing.assert_array_equal(sampled[1], np.full(32, -1))
