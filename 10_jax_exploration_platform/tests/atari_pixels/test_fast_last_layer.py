"""The fast last layer in the pixel trainer; numerical execution belongs on an allocated compute node.

The arithmetic tests use small feature widths; the trainer tests use a stand-in environment
whose environments show different pixels, so the rows of one step differ.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.training import (PixelPPORND, fast_last_layer_bonus,
                                                        predictor_bonus_from_features, row_alignment)
from exploration_platform.bonuses.rnd.last_layer import fast_layer_scores, unit_rows_with_bias

ARCHIVED_METRIC_NAMES = {"approximate_kl", "entropy", "episode_count", "episode_return_sum", "gradient_norm",
                         "intrinsic_reward_mean", "learning_rate", "loss", "policy_loss", "predictor_loss",
                         "reward_ext_sum", "rooms_visited", "steps_per_policy", "value_loss"}


def random_rollout(steps=5, copies=2, envs=8, features=6, outputs=5, seed=0):
    """Stand-in features after a ReLU, targets and an output layer, in the trainer's layout."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    hidden = jax.nn.relu(jax.random.normal(keys[0], (steps, copies, envs, features), jnp.float32))
    targets = jax.random.normal(keys[1], (steps, copies, envs, outputs), jnp.float32)
    layer = {"weight": jax.random.normal(keys[2], (copies, features, outputs), jnp.float32) * 0.3,
             "bias": jax.random.normal(keys[3], (copies, outputs), jnp.float32) * 0.1}
    return hidden, targets, layer


def plain_bonus(hidden, targets, layer):
    """Half the squared error of the unchanged output layer, in float64: [T, C, N]."""
    predicted = jnp.einsum("tcnf,cfk->tcnk", hidden.astype(jnp.float64), layer["weight"].astype(jnp.float64))
    error = predicted + layer["bias"].astype(jnp.float64)[None, :, None, :] - targets.astype(jnp.float64)
    return 0.5 * (error * error).sum(axis=-1)


def test_step_size_zero_gives_the_plain_bonus():
    """A temporary layer that never changes scores every transition like the predictor itself."""
    hidden, targets, layer = random_rollout()
    for group in (1, 4, 8):
        actual = fast_last_layer_bonus(hidden, targets, layer, (0.0, 0.0), group)
        assert actual.shape == (5, 2, 8) and actual.dtype == jnp.float32
        np.testing.assert_allclose(actual, plain_bonus(hidden, targets, layer), rtol=1e-6)


def test_first_step_is_scored_before_the_layer_changes():
    """Whatever the step size, the bonus of the first rollout step is the plain bonus."""
    hidden, targets, layer = random_rollout()
    actual = fast_last_layer_bonus(hidden, targets, layer, (0.5, 1.0), 4)
    expected = plain_bonus(hidden, targets, layer)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6)
    assert not np.allclose(actual[1:], expected[1:], rtol=1e-3)


@pytest.mark.parametrize("envs,group", [(1, 1), (4, 4), (4, 2), (4, 1)])
def test_one_state_repeated_at_step_one_half_gives_b_then_a_quarter_then_a_sixteenth(envs, group):
    """Each visit halves the error, so half the squared error falls by four; duplicates in a group share the change."""
    hidden, targets, layer = random_rollout(steps=1, copies=1, envs=1)
    # before: one state at one step ; after: the same state in every environment at three steps
    hidden = jnp.broadcast_to(hidden, (3, 1, envs, hidden.shape[-1]))
    targets = jnp.broadcast_to(targets, (3, 1, envs, targets.shape[-1]))
    bonus = np.asarray(fast_last_layer_bonus(hidden, targets, layer, (0.5,), group), np.float64)
    assert np.isfinite(bonus).all() and bonus[0, 0, 0] > 0
    np.testing.assert_allclose(bonus[1], bonus[0] / 4, rtol=1e-5)
    np.testing.assert_allclose(bonus[2], bonus[0] / 16, rtol=1e-5)


def test_a_group_of_all_environments_equals_the_direct_call():
    """With one group per policy the trainer's function is last_layer.py's own scoring."""
    hidden, targets, layer = random_rollout()
    rows, length = unit_rows_with_bias(hidden)
    whole_layer = jnp.concatenate([layer["weight"].astype(jnp.float64),
                                   layer["bias"].astype(jnp.float64)[:, None, :]], axis=1)
    direct = fast_layer_scores(rows, length, targets.astype(jnp.float64), whole_layer, 0.3,
                               lambda error: 0.5 * (error * error).sum(axis=-1))
    actual = fast_last_layer_bonus(hidden, targets, layer, (0.3, 0.3), 8)
    np.testing.assert_allclose(actual, direct, rtol=1e-6)


def test_changing_one_group_leaves_the_other_groups_bit_equal():
    """Environments 0 to 3 and 4 to 7 form two groups; new data in the second cannot reach the first."""
    hidden, targets, layer = random_rollout()
    other_hidden, other_targets, _ = random_rollout(seed=1)
    changed_hidden = hidden.at[:, :, 4:].set(other_hidden[:, :, 4:])
    changed_targets = targets.at[:, :, 4:].set(other_targets[:, :, 4:])
    expected = np.asarray(fast_last_layer_bonus(hidden, targets, layer, (0.2, 0.7), 4))
    actual = np.asarray(fast_last_layer_bonus(changed_hidden, changed_targets, layer, (0.2, 0.7), 4))
    np.testing.assert_array_equal(actual[:, :, :4], expected[:, :, :4])
    assert not np.array_equal(actual[:, :, 4:], expected[:, :, 4:])
    # With one group of eight the same change does reach the first four environments.
    whole = np.asarray(fast_last_layer_bonus(hidden, targets, layer, (0.2, 0.7), 8))
    whole_changed = np.asarray(fast_last_layer_bonus(changed_hidden, changed_targets, layer, (0.2, 0.7), 8))
    assert not np.array_equal(whole_changed[1:, :, :4], whole[1:, :, :4])


def test_per_copy_step_sizes_equal_separate_calls():
    """Each copy's bonus under its own step size is what that copy gets when scored alone."""
    hidden, targets, layer = random_rollout()
    together = fast_last_layer_bonus(hidden, targets, layer, (0.1, 0.5), 4)
    for copy, step in enumerate((0.1, 0.5)):
        alone = fast_last_layer_bonus(hidden[:, copy:copy + 1], targets[:, copy:copy + 1],
                                      {name: value[copy:copy + 1] for name, value in layer.items()}, (step,), 4)
        np.testing.assert_allclose(together[:, copy], alone[:, 0], rtol=1e-6)
    assert not np.allclose(together[1:, 0], fast_last_layer_bonus(hidden, targets, layer, (0.5, 0.5), 4)[1:, 0], rtol=1e-3)


def test_duplicate_and_all_zero_rows_stay_finite():
    """Identical states in one group and features that are all zero have a defined bonus."""
    hidden, targets, layer = random_rollout()
    same = jnp.broadcast_to(hidden[:, :, :1], hidden.shape)
    for features in (same, jnp.zeros_like(hidden)):
        for group in (2, 8):
            bonus = np.asarray(fast_last_layer_bonus(features, targets, layer, (0.9, 0.1), group))
            assert np.isfinite(bonus).all() and (bonus >= 0).all()


def test_profile_values_from_stored_features():
    """The predictor's own bonus is the plain bonus; the row alignment is 1 for equal rows."""
    hidden, targets, layer = random_rollout()
    np.testing.assert_allclose(predictor_bonus_from_features(hidden, targets, layer),
                               plain_bonus(hidden, targets, layer), rtol=1e-4)
    same = jnp.broadcast_to(hidden[:, :, :1], hidden.shape)
    np.testing.assert_allclose(row_alignment(same), [1.0, 1.0], rtol=1e-12)
    alignment = np.asarray(row_alignment(hidden))
    assert alignment.shape == (2,) and (alignment > 1 / 8).all() and (alignment < 1).all()


def test_configuration_accepts_and_rejects_the_fast_last_layer_settings():
    """The step is in [0, 1], the group divides the environments, and both are on or off together."""
    assert not PixelPPOConfig().fast_last_layer_enabled
    assert PixelPPOConfig().fast_last_layer_steps == (0.0,)
    packed = PixelPPOConfig(n_copies=4, fast_last_layer_sharing_group=16, per_copy_scalar_settings={
        "fast_last_layer_step": (0.01, 0.03, 0.1, 0.3), "intrinsic_coefficient": (0.25, 0.5, 1.0, 2.0)})
    assert packed.fast_last_layer_enabled and packed.fast_last_layer_steps == (0.01, 0.03, 0.1, 0.3)
    for group in (128, 16, 4, 1):
        assert PixelPPOConfig(fast_last_layer_step=0.1, fast_last_layer_sharing_group=group).fast_last_layer_enabled
    assert not PixelPPOConfig(n_copies=2, per_copy_scalar_settings={"fast_last_layer_step": (0.0, 0.0)}).fast_last_layer_enabled
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        PixelPPOConfig(fast_last_layer_step=1.5, fast_last_layer_sharing_group=16)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        PixelPPOConfig(fast_last_layer_step=-0.1)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        PixelPPOConfig(n_copies=2, fast_last_layer_sharing_group=16,
                       per_copy_scalar_settings={"fast_last_layer_step": (0.1, 1.1)})
    with pytest.raises(ValueError, match="positive divisor"):
        PixelPPOConfig(fast_last_layer_step=0.1)
    with pytest.raises(ValueError, match="positive divisor"):
        PixelPPOConfig(fast_last_layer_step=0.1, fast_last_layer_sharing_group=5)
    with pytest.raises(ValueError, match="must be 0 when"):
        PixelPPOConfig(fast_last_layer_sharing_group=16)
    with pytest.raises(ValueError, match="all zero or all positive"):
        PixelPPOConfig(n_copies=2, fast_last_layer_sharing_group=16,
                       per_copy_scalar_settings={"fast_last_layer_step": (0.0, 0.1)})
    with pytest.raises(ValueError, match="hoist_rollout_bonus or paired_rnd_rollout_trunks"):
        PixelPPOConfig(fast_last_layer_step=0.1, fast_last_layer_sharing_group=16, hoist_rollout_bonus=True)
    with pytest.raises(ValueError, match="hoist_rollout_bonus or paired_rnd_rollout_trunks"):
        PixelPPOConfig(fast_last_layer_step=0.1, fast_last_layer_sharing_group=16, paired_rnd_rollout_trunks=True)
    with pytest.raises(ValueError, match="integer"):
        PixelPPOConfig(fast_last_layer_step=0.1, fast_last_layer_sharing_group=16.0)


class VariedPixelEnvironment:
    """A deterministic transition source whose environments show different pixels at every step."""

    action_count = 18

    def __init__(self, num_envs):
        """Hold the fixed number of vector environments for each independent policy."""
        self.num_envs = num_envs

    def observe(self, counters):
        """Make four distinct uint8 frames from each environment's integer counter."""
        offsets = jnp.arange(4, dtype=jnp.int32)[None, None, :, None, None]
        return jnp.broadcast_to(((counters[..., None, None, None] * 9 + offsets) % 256).astype(jnp.uint8),
                                 counters.shape + (4, 84, 84))

    def reset(self, seeds):
        """Start environment n of every policy at counter 5 * n, so no two environments agree."""
        counters = jnp.broadcast_to(jnp.arange(self.num_envs, dtype=jnp.int32) * 5, (len(seeds), self.num_envs))
        return counters, self.observe(counters)

    def step(self, counters, actions):
        """Advance fixed pixels, ending an episode whenever a counter is a multiple of three."""
        counters = counters + 1
        done = counters % 3 == 0
        rewards = (actions % 2).astype(jnp.float32)
        return counters, self.observe(counters), rewards, done, {
            "episode_return": counters.astype(jnp.float32), "episode_done": done,
            "final_observation": self.observe(counters), "full_game_reset": done}


@pytest.fixture(scope="module")
def trainer_fixture():
    """Two warmed policies with four environments each, and the plain trainer's rollout and update."""
    configuration = PixelPPOConfig(n_copies=2, n_envs=4, num_steps=3, num_minibatches=1, update_epochs=1,
                                   observation_warmup_iterations=1, total_timesteps=256)
    environment = VariedPixelEnvironment(4)
    trainer = PixelPPORND(configuration, environment)
    state = trainer.warmup(trainer.initialize((19, 37)))
    _, rollout = jax.jit(trainer.collect)(state)
    result, metrics = trainer.train_iteration(state)
    jax.block_until_ready(result)
    return configuration, environment, state, rollout, result, metrics


def test_default_trainer_reports_exactly_the_archived_metric_names(trainer_fixture):
    """With every new option off, an update returns the fields of the archived baseline and no other."""
    _, _, _, rollout, _, metrics = trainer_fixture
    assert set(metrics) == ARCHIVED_METRIC_NAMES
    assert rollout.bonus_hidden is None and rollout.bonus_targets is None
    assert rollout.predictor_bonus is None and rollout.row_alignment is None and rollout.game_events is None


def test_trainer_rollout_with_a_tiny_step_size_gives_the_plain_bonus(trainer_fixture):
    """The trainer feeds the fast layer the same next frames and statistics as the plain bonus."""
    configuration, environment, state, plain_rollout, _, _ = trainer_fixture
    for group in (4, 2):
        fast = PixelPPORND(replace(configuration, fast_last_layer_step=1e-12,
                                   fast_last_layer_sharing_group=group), environment)
        _, rollout = jax.jit(fast.collect)(state)
        np.testing.assert_allclose(rollout.intrinsic_rewards, plain_rollout.intrinsic_rewards, rtol=2e-4)
        assert rollout.bonus_hidden is None and rollout.bonus_targets is None
        # Everything else of the rollout does not depend on the bonus.
        np.testing.assert_array_equal(rollout.actions, plain_rollout.actions)
        np.testing.assert_array_equal(rollout.next_frames, plain_rollout.next_frames)


def test_trainer_rollout_first_step_is_plain_and_later_steps_differ(trainer_fixture):
    """With a real step size the first step keeps the plain bonus and later steps see a changed layer."""
    configuration, environment, state, plain_rollout, _, _ = trainer_fixture
    fast = PixelPPORND(replace(configuration, fast_last_layer_step=0.5, fast_last_layer_sharing_group=2), environment)
    _, rollout = jax.jit(fast.collect)(state)
    np.testing.assert_allclose(rollout.intrinsic_rewards[0], plain_rollout.intrinsic_rewards[0], rtol=2e-4)
    assert np.isfinite(np.asarray(rollout.intrinsic_rewards)).all()
    assert not np.allclose(rollout.intrinsic_rewards[1:], plain_rollout.intrinsic_rewards[1:], rtol=1e-3)


def test_complete_update_with_the_fast_last_layer_and_its_profile(trainer_fixture):
    """A whole update runs, stays finite, and reports the bonus profile beside the archived fields."""
    configuration, environment, state, plain_rollout, _, plain_metrics = trainer_fixture
    fast = PixelPPORND(replace(configuration, fast_last_layer_step=0.5, fast_last_layer_sharing_group=2,
                               record_bonus_profile=True), environment)
    result, metrics = fast.train_iteration(state)
    jax.block_until_ready(result)
    assert set(metrics) == ARCHIVED_METRIC_NAMES | {"bonus_first_steps_mean", "bonus_last_steps_mean",
                                                    "bonus_divisor", "predictor_bonus_mean", "row_alignment"}
    for name, value in metrics.items():
        assert np.isfinite(np.asarray(value, np.float64)).all(), name
    for value in jax.tree.leaves(result.parameters):
        assert value.dtype == jnp.float32 and np.isfinite(np.asarray(value)).all()
    # The predictor's own bonus is what the plain trainer reports as its bonus on the same rollout.
    np.testing.assert_allclose(metrics["predictor_bonus_mean"], plain_metrics["intrinsic_reward_mean"], rtol=2e-4)
    assert not np.allclose(metrics["intrinsic_reward_mean"], plain_metrics["intrinsic_reward_mean"], rtol=1e-3)
    # The rollout has three steps, fewer than 32, so both ends cover the whole rollout.
    np.testing.assert_allclose(metrics["bonus_first_steps_mean"], metrics["intrinsic_reward_mean"], rtol=1e-6)
    np.testing.assert_allclose(metrics["bonus_last_steps_mean"], metrics["intrinsic_reward_mean"], rtol=1e-6)
    np.testing.assert_allclose(metrics["bonus_divisor"], np.sqrt(np.asarray(result.reward_statistics.variance)))
    alignment = np.asarray(metrics["row_alignment"])
    assert (alignment > 0.25).all() and (alignment <= 1.0 + 1e-12).all()
    # Running the same update again gives the same bits.
    again, again_metrics = fast.train_iteration(state)
    for before, after in zip(jax.tree.leaves(result), jax.tree.leaves(again)):
        np.testing.assert_array_equal(before, after)
    for name in metrics:
        np.testing.assert_array_equal(metrics[name], again_metrics[name])


def test_bonus_profile_without_the_fast_last_layer_changes_no_other_number(trainer_fixture):
    """The profile of the plain bonus adds three fields and leaves the state and the other fields as they are."""
    configuration, environment, state, _, plain_result, plain_metrics = trainer_fixture
    profiled = PixelPPORND(replace(configuration, record_bonus_profile=True), environment)
    result, metrics = profiled.train_iteration(state)
    jax.block_until_ready(result)
    assert set(metrics) == ARCHIVED_METRIC_NAMES | {"bonus_first_steps_mean", "bonus_last_steps_mean", "bonus_divisor"}
    for before, after in zip(jax.tree.leaves(plain_result), jax.tree.leaves(result)):
        np.testing.assert_array_equal(before, after)
    for name in ARCHIVED_METRIC_NAMES:
        np.testing.assert_array_equal(plain_metrics[name], metrics[name])


def test_changing_one_copys_step_size_leaves_the_other_copy_bit_equal(trainer_fixture):
    """Per-copy step sizes: a new value in policy one cannot alter policy zero's complete update."""
    configuration, environment, state, _, _, _ = trainer_fixture
    results = []
    for steps in ((0.1, 0.1), (0.1, 0.5)):
        trainer = PixelPPORND(replace(configuration, fast_last_layer_sharing_group=2,
                                      per_copy_scalar_settings={"fast_last_layer_step": steps}), environment)
        result, metrics = trainer.train_iteration(state)
        jax.block_until_ready(result)
        results.append((result, metrics))
    (expected, expected_metrics), (actual, actual_metrics) = results
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        if before.ndim and before.shape[0] == 2:
            np.testing.assert_array_equal(before[0], after[0])
        else:
            np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(expected_metrics["intrinsic_reward_mean"][0], actual_metrics["intrinsic_reward_mean"][0])
    assert float(expected_metrics["intrinsic_reward_mean"][1]) != float(actual_metrics["intrinsic_reward_mean"][1])
    assert any(not np.array_equal(np.asarray(before[1]), np.asarray(after[1]))
               for before, after in zip(jax.tree.leaves(expected.parameters), jax.tree.leaves(actual.parameters)))
