"""Count decay exponents, one-hot (tabular-RND) networks and the predictor options; run only on a compute node."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.grid_initialization.config import GridPPOConfig
from exploration_platform.grid_initialization.environments import make_geometry
from exploration_platform.grid_initialization.online_rnd import (
    NetworkOptions, frozen_network_field, input_table, network_layers, network_weights, stack_networks)
from exploration_platform.grid_initialization.training import GridTrainer, count_bonus

jax.config.update("jax_enable_x64", True)
CAMPAIGN = Path(__file__).resolve().parents[2]/'scripts/grid_initialization_campaign'


def test_count_bonus_follows_each_exponent():
    """Before: field 2 and visits 1..8; after: 2 n^-e for every exponent, with 0.5 the square-root rule."""
    visits = jnp.arange(1, 9, dtype=jnp.int32)
    for exponent in (2.0, 1.0, 0.5, 0.25):
        configuration = GridPPOConfig(n_copies=1, count_decay_exponent=exponent)
        values = np.asarray(count_bonus(jnp.full((8,), 2.0, jnp.float32), visits, configuration))
        np.testing.assert_allclose(values, 2.0 * np.arange(1, 9) ** -exponent, rtol=1e-6)
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, count_decay_exponent=0.0)


def test_one_hot_networks_differ_from_coordinate_networks_only_in_the_input_layer():
    """The one-hot input layer is [S,32] under its own seed name; the deeper layers are the coordinate layers."""
    coordinate = network_weights(3)
    one_hot = network_layers(3, NetworkOptions(input_kind="one_hot"), 32)
    assert [weight.shape for weight, _ in one_hot["target"]] == [(32, 32), (32, 64), (64, 64), (64, 512)]
    for name in ("target", "predictor"):
        for layer in range(1, len(coordinate[name])):
            np.testing.assert_array_equal(one_hot[name][layer][0], coordinate[name][layer])
        assert all(np.all(bias == 0) for _, bias in one_hot[name])
    # The default options reproduce the coordinate networks with zero biases.
    default = network_layers(3, NetworkOptions(), 2)
    for name in ("target", "predictor"):
        for layer, weight in enumerate(coordinate[name]):
            np.testing.assert_array_equal(default[name][layer][0], weight)


def tabular_trainer(world, copies=2, **changes):
    """Build a small online tabular-RND trainer whose copies use field seeds 0 and 1."""
    options = NetworkOptions(input_kind="one_hot", **changes.pop("network", {}))
    rnd = stack_networks(np.arange(copies) % 2, options, world.state_count)
    configuration = GridPPOConfig(n_copies=copies, hidden_width=8, num_steps=16, num_minibatches=2,
        update_epochs=2, total_timesteps=256, episode_length=64, intrinsic_source="online_rnd",
        rnd_input="one_hot", **changes)
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates,
                          np.ones((copies, world.state_count), np.float32), rnd)
    return trainer, trainer.initialize(np.arange(copies) + 11), options


def test_online_tabular_rnd_starts_at_the_frozen_tabular_field():
    """At step 0 the online tabular-RND error of every cell equals the frozen tabular-RND field."""
    world = make_geometry("snake", 4, 2)
    trainer, state, options = tabular_trainer(world)
    errors = np.asarray(trainer.all_cell_errors(state))
    for copy in range(2):
        np.testing.assert_allclose(errors[copy], frozen_network_field(world, copy % 2, None, options).raw, rtol=2e-5)
    assert input_table(world, options).shape == (world.state_count, world.state_count)


def test_unit_start_and_initial_ratio_give_equal_starting_bonuses():
    """A zero predictor output with unit-length targets starts at 1/2 everywhere; the ratio readout starts at 1."""
    world = make_geometry("straight", 8)
    trainer, state, _ = tabular_trainer(world, rnd_unit_length_target=True, network={"zero_predictor_output": True})
    np.testing.assert_allclose(np.asarray(trainer.all_cell_errors(state)), 0.5, rtol=1e-6)
    trainer, state, _ = tabular_trainer(world, rnd_readout="initial_ratio_length")
    np.testing.assert_allclose(np.asarray(trainer.all_cell_errors(state)), 1.0, rtol=1e-6)


@pytest.mark.parametrize("optimizer", ["sgd", "adagrad", "adam"])
def test_own_optimizers_train_the_predictor_and_input_layer_only_freezes_the_rest(optimizer):
    """One iteration moves the predictor; with input-layer training only layer 0 moves; the target never moves."""
    world = make_geometry("straight", 8)
    for trainable in ("all", "input_layer"):
        trainer, state, _ = tabular_trainer(world, rnd_optimizer=optimizer, rnd_learning_rate=1e-2,
                                            rnd_loss="power", rnd_loss_power=4.0, rnd_trainable=trainable)
        before = jax.tree.map(np.asarray, state.parameters["predictor"])
        target = jax.tree.map(np.asarray, state.target)
        after_state, _ = trainer.train_iteration(state)
        after = jax.tree.map(np.asarray, after_state.parameters["predictor"])
        moved = [not np.array_equal(b["weight"], a["weight"]) for b, a in zip(before, after)]
        assert moved[0]
        if trainable == "input_layer":
            assert not any(moved[1:])
        jax.tree.map(np.testing.assert_array_equal, target, jax.tree.map(np.asarray, after_state.target))
        assert int(after_state.predictor_optimizer.step) == 2 * 2


def test_shrink_target_and_target_copy_trunk_run():
    """The shrink-target loss with the target-copy trunk lowers the start cell's error and leaves untrained rows at 1."""
    world = make_geometry("straight", 8)
    trainer, state, _ = tabular_trainer(world, rnd_optimizer="sgd", rnd_learning_rate=1e-2, rnd_loss="shrink_target",
        rnd_trainable="input_layer", rnd_readout="initial_ratio_length", network={"predictor_trunk": "target_copy"})
    assert len(state.parameters["predictor"]) == 4
    for _ in range(4):
        state, _ = trainer.train_iteration(state)
    # The predictor trains on the states actions are taken in, so the start cell is always trained.
    errors = np.asarray(trainer.all_cell_errors(state))
    assert np.all(errors[:, 0] < 1.0) and np.all(errors <= 1.0 + 1e-6) and np.all(np.isfinite(errors))


def test_invalid_combinations_are_refused():
    """One-hot inputs are never whitened, and count runs refuse online RND settings."""
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, intrinsic_source="online_rnd", rnd_input="one_hot", normalize_rnd_inputs=True)
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, rnd_readout="length")
    with pytest.raises(ValueError):
        NetworkOptions(input_kind="pixels")


def test_condition_names_keep_the_original_names_and_add_unique_new_ones():
    """The original names are unchanged; the extension's ten names are new and distinct."""
    sys.path.insert(0, str(CAMPAIGN))
    from fixed_settings_campaign import condition_name, conditions
    original = [condition_name(row) for row in conditions('original')]
    extension = [condition_name(row) for row in conditions('extension')]
    assert 'uniform_state-normalization-off_reward-normalization-on' in original and len(original) == 13
    assert 'uniform-decay-exponent-2_state-normalization-off_reward-normalization-on' in extension
    assert 'online-tabular-rnd_state-normalization-off_reward-normalization-on' in extension
    assert len(set(original + extension)) == 23


def test_training_budget_reads_an_early_stop():
    """The full budget by default; an early stop on the same schedule when given."""
    sys.path.insert(0, str(CAMPAIGN))
    from train_unit import training_budget
    assert training_budget({'total_timesteps': 2 ** 24}) == 2 ** 24
    assert training_budget({'total_timesteps': 2 ** 24, 'stop_after_timesteps': 2 ** 22}) == 2 ** 22
    with pytest.raises(ValueError):
        training_budget({'total_timesteps': 2 ** 22, 'stop_after_timesteps': 2 ** 24})


def test_generalized_shrink_power_matches_its_closed_form_and_stays_bounded():
    """The shrink factor is (1 + r^p)^(-1/p): p = 2 is the original 1/sqrt(1+r^2); the loss is finite for huge errors."""
    from exploration_platform.grid_initialization.training import predictor_sample_losses
    world = make_geometry("straight", 8)
    for power in (1.0, 2.0, 4.0):
        trainer, state, _ = tabular_trainer(world, rnd_optimizer="sgd", rnd_learning_rate=0.5, rnd_loss="shrink_target",
            rnd_shrink_power=power, rnd_readout="length", rnd_unit_length_target=True,
            network={"zero_predictor_output": True})
        ids = jnp.broadcast_to(jnp.arange(8), (2, 8))
        inputs = state.rnd_table[ids]
        losses = np.asarray(predictor_sample_losses(state.parameters["predictor"], state.target, inputs,
                                                    trainer.configuration))
        # Unit start: r = 1 in every cell, so the loss is 0.5 (1 - 2^(-1/p))^2 everywhere.
        np.testing.assert_allclose(losses, 0.5 * (1.0 - 2.0 ** (-1.0 / power)) ** 2, rtol=1e-5)
        for _ in range(3):
            state, _ = trainer.train_iteration(state)
        assert np.isfinite(np.asarray(trainer.all_cell_errors(state))).all()
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, intrinsic_source="online_rnd", rnd_shrink_power=1.0)


def exact_trainer(world, iterations=1, **changes):
    """A unit-start trainer with the exact-step optimizer and the length readout."""
    return tabular_trainer(world, rnd_optimizer="exact_step", rnd_loss="shrink_target", rnd_readout="length",
        rnd_unit_length_target=True, rnd_exact_step_iterations=iterations, network={"zero_predictor_output": True},
        **changes)


def test_exact_step_counts_kept_visits_of_one_cell():
    """Eight kept visits of one cell in a minibatch take its error length from 1 to 1/sqrt(1+8), then on to 1/sqrt(1+10)."""
    from exploration_platform.grid_initialization.training import exact_shrink_step
    world = make_geometry("straight", 8)
    trainer, state, _ = exact_trainer(world)
    cells = jnp.zeros((2, 8), jnp.int32)
    inputs = state.rnd_table[cells]
    predictor = exact_shrink_step(state.parameters["predictor"], state.target, inputs, jnp.ones((2, 8)), trainer.configuration)
    errors = np.asarray(trainer.all_cell_errors(state._replace(parameters=dict(state.parameters, predictor=predictor))))
    np.testing.assert_allclose(errors[:, 0], 1.0 / np.sqrt(9.0), rtol=0.02)
    # Two more kept visits; unkept samples of the minibatch do not count.
    mask = jnp.asarray(np.tile([1, 1, 0, 0, 0, 0, 0, 0], (2, 1)))
    predictor = exact_shrink_step(predictor, state.target, inputs, mask, trainer.configuration)
    errors = np.asarray(trainer.all_cell_errors(state._replace(parameters=dict(state.parameters, predictor=predictor))))
    np.testing.assert_allclose(errors[:, 0], 1.0 / np.sqrt(11.0), rtol=0.03)
    # A minibatch with nothing kept changes nothing.
    same = exact_shrink_step(predictor, state.target, inputs, jnp.zeros((2, 8)), trainer.configuration)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), same, predictor)


def test_exact_step_counts_each_cell_by_its_own_visits_and_other_exponents():
    """Cells with 6 and 2 kept visits end near 1/sqrt(7) and 1/sqrt(3); p = 1 and p = 4 follow (1 + n)^(-1/p).

    Two cells share the deeper layers, so a few steps along the gradient realize both targets only approximately;
    the rarer cell is moved a little less than its target (measured 0.63-0.64 against 0.577 after four steps).
    """
    from exploration_platform.grid_initialization.training import exact_shrink_step
    world = make_geometry("straight", 8)
    cells = jnp.asarray(np.tile([0, 0, 0, 0, 0, 0, 1, 1], (2, 1)))
    for power in (2.0, 1.0, 4.0):
        trainer, state, _ = exact_trainer(world, iterations=4, **({} if power == 2.0 else {"rnd_shrink_power": power}))
        predictor = exact_shrink_step(state.parameters["predictor"], state.target, state.rnd_table[cells],
                                      jnp.ones((2, 8)), trainer.configuration)
        errors = np.asarray(trainer.all_cell_errors(state._replace(parameters=dict(state.parameters, predictor=predictor))))
        # The rarer cell is under-moved more when its target is further away (p = 1: 0.43 against 0.33).
        tolerance = 0.15 if power == 2.0 else 0.35
        np.testing.assert_allclose(errors[:, 0], 7.0 ** (-1.0 / power), rtol=tolerance)
        np.testing.assert_allclose(errors[:, 1], 3.0 ** (-1.0 / power), rtol=tolerance)
        assert np.all(errors[:, 0] < errors[:, 1]) and np.all(errors[:, 1] < 1.0)
        assert np.isfinite(errors).all() and errors[:, 2:].min() > 0.5


def test_exact_step_trains_inside_the_ppo_iteration_and_refuses_other_losses():
    """A full iteration with the exact-step optimizer lowers the start cell's bonus and stays finite."""
    world = make_geometry("straight", 8)
    trainer, state, _ = exact_trainer(world)
    for _ in range(3):
        state, _ = trainer.train_iteration(state)
    errors = np.asarray(trainer.all_cell_errors(state))
    assert np.isfinite(errors).all() and np.all(errors[:, 0] < 0.9)
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, intrinsic_source="online_rnd", rnd_input="one_hot", rnd_optimizer="exact_step")


def test_diagnostic_count_forms_follow_their_formulas():
    """Per-step with rate 1 is the count rule; per-rollout uses the visits before the rollout; the rate scales them."""
    world = make_geometry("straight", 4)
    fields = np.ones((1, world.state_count), np.float32)

    def bonuses(**changes):
        configuration = GridPPOConfig(n_copies=1, num_steps=16, num_minibatches=2, total_timesteps=64, episode_length=64,
                                      **changes)
        trainer = GridTrainer(configuration, world.transition_table, world.coordinates, fields)
        state = trainer.initialize([5])
        state, first = trainer.capture(state)
        _, second = trainer.capture(state)
        return first, second

    rule_first, rule_second = bonuses()
    step_first, _ = bonuses(count_rate=0.5)
    rollout_first, rollout_second = bonuses(count_update="per_rollout")
    # The same policy seed gives the same path, so arrival counts match across the three runs.
    counts = np.asarray(rule_first.arrival_counts, np.float64)[:, 0]
    np.testing.assert_allclose(np.asarray(rule_first.intrinsic_rewards)[:, 0], counts ** -0.5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(step_first.intrinsic_rewards)[:, 0], (1 + 0.5 * (counts - 1)) ** -0.5, rtol=1e-6)
    # In the first rollout nothing was visited before it began, so every bonus is 1.
    np.testing.assert_allclose(np.asarray(rollout_first.intrinsic_rewards), 1.0, rtol=1e-6)
    assert np.asarray(rollout_second.intrinsic_rewards).max() <= 1.0 and np.asarray(rollout_second.intrinsic_rewards).min() < 1.0
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, intrinsic_source="online_rnd", count_update="per_rollout")


def test_count_mixture_is_a_weighted_sum_of_power_laws():
    """A mixture ((1, 2), (0.1, 0.5)) gives (1+m)^-2 + 0.1 (1+m)^-0.5 of the earlier visits m."""
    world = make_geometry("straight", 4)
    configuration = GridPPOConfig(n_copies=1, num_steps=16, num_minibatches=2, total_timesteps=64, episode_length=64,
                                  count_mixture=((1.0, 2.0), (0.1, 0.5)))
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates, np.ones((1, world.state_count), np.float32))
    _, rollout = trainer.capture(trainer.initialize([5]))
    earlier = np.asarray(rollout.arrival_counts, np.float64)[:, 0] - 1
    np.testing.assert_allclose(np.asarray(rollout.intrinsic_rewards)[:, 0],
                               (1 + earlier) ** -2.0 + 0.1 * (1 + earlier) ** -0.5, rtol=1e-6)
    with pytest.raises(ValueError):
        GridPPOConfig(n_copies=1, count_mixture=((1.0, -1.0),))
