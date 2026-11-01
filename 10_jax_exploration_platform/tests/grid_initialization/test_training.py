"""Exact-count and PPO checks; execute only inside a compute-node allocation."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.grid_initialization.config import GridPPOConfig
from exploration_platform.grid_initialization.training import GridTrainer, generalized_advantages


def fixture(policy_kind="tabular", **changes):
    """Create a three-state deterministic path with four compass actions."""
    configuration = GridPPOConfig(n_copies=2, policy_kind=policy_kind, hidden_width=8,
        num_steps=8, num_minibatches=2, update_epochs=1, total_timesteps=64,
        episode_length=4, **changes)
    transitions = np.array([[0, 1, 0, 0], [1, 2, 1, 0], [2, 2, 2, 1]], np.int32)
    coordinates = np.array([[-1., 0.], [0., 0.], [1., 0.]], np.float32)
    fields = np.array([[1., 2., 3.], [1., 2., 3.]], np.float32)
    trainer = GridTrainer(configuration, transitions, coordinates, fields)
    return trainer, trainer.initialize([17, 31])


def assert_trees_close(left, right, tolerance=2e-5):
    """Compare matching public states without ignoring integer or boolean differences."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        if np.asarray(a).dtype.kind in "biu":
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def test_count_each_arrival_before_bonus_and_persist_across_resets():
    """Repeated wall actions count arrivals, and timeout resets do not erase them."""
    trainer, state = fixture()
    # A very strong up-action preference forces every transition to remain in state zero.
    parameters = {"table": state.parameters["table"].at[:, :, 0].set(100.)}
    state = state._replace(parameters=parameters)
    after, rollout = trainer.capture(state)
    np.testing.assert_array_equal(rollout.arrival_counts[:, 0], np.arange(1, 9))
    np.testing.assert_allclose(rollout.intrinsic_rewards[:, 0], 1 / np.sqrt(np.arange(1, 9)), rtol=1e-6)
    np.testing.assert_array_equal(after.counts, [[8, 0, 0], [8, 0, 0]])
    np.testing.assert_array_equal(after.episodes, [2, 2])
    np.testing.assert_array_equal(after.first_goal_step, [-1, -1])
    np.testing.assert_array_equal(after.blocked_moves, [8, 8])
    np.testing.assert_array_equal(state.counts, np.zeros((2, 3), np.int32))


def test_goal_bonus_uses_arrived_state_before_reset():
    """Goal arrival pays the goal's field and the following decision starts at zero."""
    trainer, state = fixture()
    state = state._replace(parameters={"table": state.parameters["table"].at[:, :, 1].set(100.)})
    after, rollout = trainer.capture(state)
    np.testing.assert_array_equal(rollout.states[:, 0], [0, 1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(rollout.arrivals[:, 0], [1, 2, 1, 2, 1, 2, 1, 2])
    np.testing.assert_allclose(rollout.intrinsic_rewards[:2, 0], [2., 3.])
    np.testing.assert_array_equal(after.first_goal_step, [2, 2])
    np.testing.assert_array_equal(after.successes, [4, 4])
    np.testing.assert_array_equal(after.positions, [0, 0])


@pytest.mark.parametrize("policy_kind", ["tabular", "neural"])
def test_complete_iteration_and_independent_settings(policy_kind):
    """Both policy types train, and changing another policy leaves the first unchanged."""
    trainer, state = fixture(policy_kind)
    expected, metrics = trainer.train_iteration(state)
    jax.block_until_ready(expected)
    assert int(expected.iteration) == 1
    assert int(expected.optimizer.step) == 2
    assert np.asarray(expected.counts).sum() == 16
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(expected.parameters))
    changed = GridTrainer(replace(trainer.configuration, per_copy_scalar_settings={
        "learning_rate": (2 ** -8, 2 ** -5), "intrinsic_coefficient": (1., 16.)}),
        trainer.transition_table, trainer.coordinates, trainer.fields)
    actual, _ = changed.train_iteration(state)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        if a.ndim and a.shape[0] == 2:
            np.testing.assert_array_equal(a[0], b[0])
    assert np.isfinite(np.asarray(metrics["loss"])).all()


@pytest.mark.parametrize("option,value", [
    ("rollout_unroll", 2), ("update_unroll", 2), ("precompute_all_states", True),
    ("all_states_minibatch", True), ("hoist_values", True),
    ("hoist_log_probabilities", True), ("hoist_bonus", True), ("batch_random_keys", True),
    ("flat_parameters", True), ("flatten_update_scan", True),
    ("pre_gather_minibatches", True), ("associative_gae", True),
    ("rsqrt_bonus", True), ("fuse_optimizer", True), ("flat_state_gather", True)])
def test_optimization_complete_update(option, value):
    """Every candidate computes the same short tabular trajectory and update within tolerance."""
    trainer, state = fixture()
    expected, expected_metrics = trainer.train_iteration(state)
    candidate = GridTrainer(replace(trainer.configuration, **{option: value}),
        trainer.transition_table, trainer.coordinates, trainer.fields)
    actual, actual_metrics = candidate.train_iteration(state)
    assert_trees_close(expected, actual)
    assert_trees_close(expected_metrics, actual_metrics)


def test_intrinsic_bootstraps_reset_state_and_task_return_stops():
    """A terminal transition's intrinsic return includes its reset value."""
    trainer, state = fixture(gamma_ext=0.5, gamma_int=0.5, gae_lambda=0.)
    state = state._replace(parameters={"table": state.parameters["table"].at[:, :, 1].set(100.)})
    _, rollout = trainer.capture(state)
    rollout = rollout._replace(values_ext=jnp.zeros((8, 2), jnp.float32), values_int=jnp.zeros((8, 2), jnp.float32))
    ext, intrinsic = generalized_advantages(rollout.rewards, jnp.ones((8, 2), jnp.float32), rollout,
        (jnp.full((2,), 10., jnp.float32), jnp.full((2,), 10., jnp.float32)), trainer.configuration)
    np.testing.assert_allclose(ext[-1], [1., 1.])
    np.testing.assert_allclose(intrinsic[-1], [6., 6.])


@pytest.mark.parametrize("upstream", [False, True])
def test_explicit_cleanrl_normalization(upstream):
    """The selected sensitivity supports either recorded normalization-axis convention."""
    trainer, state = fixture(normalize_intrinsic_rewards=True, upstream_reward_filter_axis=upstream)
    after, metrics = trainer.train_iteration(state)
    assert np.isfinite(np.asarray(metrics["loss"])).all()
    assert np.all(np.asarray(after.reward_statistics.count) > 1e-4)


def test_donation_and_zero_field():
    """Donated buffers survive successive updates and zero bonuses stay finite."""
    trainer, _ = fixture(donate_state=True)
    trainer = GridTrainer(trainer.configuration, trainer.transition_table,
        trainer.coordinates, jnp.zeros_like(trainer.fields))
    state = trainer.initialize([17, 31])
    state, _ = trainer.train_iteration(state)
    state, metrics = trainer.train_iteration(state)
    np.testing.assert_array_equal(metrics["raw_intrinsic_reward"], [0., 0.])
    assert int(state.iteration) == 2
    assert np.isfinite(np.asarray(metrics["loss"])).all()


@pytest.mark.parametrize("policy_kind", ["tabular", "neural"])
def test_copy_batch_size_preserves_first_policy(policy_kind):
    """Packing one extra policy preserves the first policy's randomness and update."""
    trainer, paired = fixture(policy_kind)
    expected, _ = trainer.train_iteration(paired)
    single = GridTrainer(replace(trainer.configuration, n_copies=1),
        trainer.transition_table, trainer.coordinates, trainer.fields[:1])
    actual, _ = single.train_iteration(single.initialize([17]))
    for first, second in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        if first.ndim and first.shape[0] == 2:
            np.testing.assert_allclose(first[:1], second, rtol=2e-5, atol=2e-5)


def test_invalid_configuration_and_inputs():
    """Invalid settings and non-finite frozen fields fail before tracing."""
    with pytest.raises(ValueError, match="minibatches"):
        GridPPOConfig(num_steps=3, num_minibatches=2)
    with pytest.raises(ValueError, match="one value"):
        GridPPOConfig(n_copies=2, per_copy_scalar_settings={"learning_rate": [0.1]})
    trainer, _ = fixture()
    invalid = trainer.fields.at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="fields"):
        GridTrainer(trainer.configuration, trainer.transition_table, trainer.coordinates, invalid).initialize([1, 2])
