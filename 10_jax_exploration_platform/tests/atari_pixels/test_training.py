"""Small complete learner checks; numerical execution belongs on an allocated compute node."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.training import PixelPPORND


class FixedPixelEnvironment:
    """A deterministic transition source that exercises PPO boundaries without game assets."""

    action_count = 18

    def __init__(self, num_envs):
        """Hold the fixed number of vector environments for each independent policy."""
        self.num_envs = num_envs

    def observe(self, counters):
        """Make four distinct uint8 frames from each environment's integer counter."""
        offsets = jnp.arange(4, dtype=jnp.int32)[None, None, :, None, None]
        return jnp.broadcast_to(((counters[..., None, None, None] + offsets) % 256).astype(jnp.uint8),
                                 counters.shape + (4, 84, 84))

    def reset(self, seeds):
        """Start every vector environment at zero while retaining an independent policy axis."""
        counters = jnp.zeros((len(seeds), self.num_envs), jnp.int32)
        return counters, self.observe(counters)

    def step(self, counters, actions):
        """Advance fixed pixels, ending an episode every third step."""
        counters = counters + 1
        done = counters % 3 == 0
        rewards = (actions % 2).astype(jnp.float32)
        return counters, self.observe(counters), rewards, done, {
            "episode_return": counters.astype(jnp.float32), "episode_done": done,
            "final_observation": self.observe(counters), "full_game_reset": done}


def test_warmup_complete_iteration_and_optimized_equivalence():
    """A complete rollout/update retains frozen targets and agrees after optional batching."""
    configuration = PixelPPOConfig(n_envs=2, num_steps=2, num_minibatches=1,
                                    update_epochs=1, observation_warmup_iterations=1,
                                    total_timesteps=64)
    environment = FixedPixelEnvironment(2)
    trainer = PixelPPORND(configuration, environment)
    starting = trainer.initialize((17,))
    warmed = trainer.warmup(starting)
    jax.block_until_ready(warmed)
    np.testing.assert_array_equal(starting.observations, warmed.observations)
    np.testing.assert_allclose(warmed.observation_statistics.count, [4.0001])
    assert int(warmed.environment[0, 0]) == 2
    expected, metrics = trainer.train_iteration(warmed)
    jax.block_until_ready(expected)
    assert int(expected.iteration) == 1 and int(expected.optimizer.step) == 1
    for before, after in zip(jax.tree.leaves(warmed.target), jax.tree.leaves(expected.target)):
        np.testing.assert_array_equal(before, after)
    for value in jax.tree.leaves(expected.parameters):
        assert value.dtype == jnp.float32 and np.isfinite(np.asarray(value)).all()
    assert np.isfinite(float(metrics["loss"][0]))
    assert int(metrics["steps_per_policy"]) == 4

    # Cache only frozen training targets; batched rollout bonuses use the old statistics.
    optimized = PixelPPORND(replace(configuration, cache_target_features=True,
                                    hoist_rollout_bonus=True), environment)
    actual, actual_metrics = optimized.train_iteration(warmed)
    jax.block_until_ready(actual)
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_allclose(before, after, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(metrics["loss"], actual_metrics["loss"], rtol=1e-4, atol=1e-5)


def test_donation_and_zero_warmup():
    """The donated-state path accepts the environment/observation buffers on successive calls."""
    configuration = PixelPPOConfig(n_envs=2, num_steps=2, num_minibatches=1,
                                    update_epochs=1, observation_warmup_iterations=0,
                                    total_timesteps=64, donate_state=True)
    trainer = PixelPPORND(configuration, FixedPixelEnvironment(2))
    state = trainer.warmup(trainer.initialize((23,)))
    state, _ = trainer.train_iteration(state)
    state, metrics = trainer.train_iteration(state)
    jax.block_until_ready(state)
    assert int(state.iteration) == 2 and int(state.optimizer.step) == 2
    assert np.isfinite(np.asarray(metrics["loss"])).all()


@pytest.fixture(scope="module")
def paired_policy_fixture():
    """Initialize two independently seeded policies once for complete-iteration isolation tests."""
    configuration = PixelPPOConfig(n_copies=2, n_envs=2, num_steps=2, num_minibatches=1,
                                    update_epochs=1, observation_warmup_iterations=1,
                                    total_timesteps=256)
    environment = FixedPixelEnvironment(2)
    trainer = PixelPPORND(configuration, environment)
    state = trainer.warmup(trainer.initialize((19, 37)))
    result, metrics = trainer.train_iteration(state)
    jax.block_until_ready(result)
    return configuration, environment, state, result, metrics


@pytest.mark.parametrize("setting,values", [
    ("per_copy_learning_rates", (1e-4, 3e-4)),
    ("per_copy_intrinsic_coefficients", (1.0, 10.0)),
])
def test_complete_iteration_hyperparameter_copy_isolation(paired_policy_fixture, setting, values):
    """Changing only policy one's rate or intrinsic weight leaves policy zero bit-identical."""
    configuration, environment, state, expected, expected_metrics = paired_policy_fixture
    changed_trainer = PixelPPORND(replace(configuration, **{setting: values}), environment)
    actual, actual_metrics = changed_trainer.train_iteration(state)
    jax.block_until_ready(actual)
    # Every mutable field, including parameters, Adam, filters and statistics, retains its copy axis.
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        if before.ndim and before.shape[0] == 2:
            np.testing.assert_array_equal(before[0], after[0])
        else:
            np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(np.asarray(before[1]), np.asarray(after[1]))
               for before, after in zip(jax.tree.leaves(expected.parameters),
                                         jax.tree.leaves(actual.parameters)))
    np.testing.assert_array_equal(expected_metrics["loss"][0], actual_metrics["loss"][0])
    if setting == "per_copy_learning_rates":
        np.testing.assert_allclose(actual_metrics["learning_rate"], values, rtol=1e-7)


def test_checkpoint_restores_exact_next_update_and_rejects_changed_environment(
        paired_policy_fixture, tmp_path):
    """The production checkpoint writer/reader reproduce uninterrupted training exactly."""
    from exploration_platform.atari_pixels.checkpointing import load_checkpoint

    configuration, environment, _, state, _ = paired_policy_fixture
    # Import the real campaign writer, without invoking its command-line entry point.
    script = Path(__file__).resolve().parents[2] / "scripts/atari_pixel_campaign/run_unit.py"
    specification = importlib.util.spec_from_file_location("pixel_checkpoint_writer", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    path = tmp_path / "latest.pkl"
    seeds, environment_configuration = (19, 37), {"num_envs": 2}
    module.save_checkpoint(path, state, configuration, seeds, environment_configuration)
    restored = load_checkpoint(path, configuration, seeds, environment_configuration)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(before, after)
    assert path.exists() and not path.with_suffix(".tmp").exists()

    # Replay one actual next rollout and update rather than checking serialized bytes alone.
    trainer = PixelPPORND(configuration, environment)
    uninterrupted, expected_metrics = trainer.train_iteration(state)
    resumed, actual_metrics = trainer.train_iteration(restored)
    jax.block_until_ready(resumed)
    for before, after in zip(jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(before, after)
    for name in expected_metrics:
        np.testing.assert_array_equal(expected_metrics[name], actual_metrics[name])

    with pytest.raises(ValueError, match="environment settings"):
        load_checkpoint(path, configuration, seeds, {"num_envs": 3})
    with pytest.raises(ValueError, match="policy seeds"):
        load_checkpoint(path, configuration, (37, 19), environment_configuration)
    with pytest.raises(ValueError, match="learner configuration"):
        load_checkpoint(path, replace(configuration, learning_rate=3e-4), seeds, environment_configuration)
    # Old or incomplete files cannot silently resume with an unrecorded environment protocol.
    with path.open("rb") as stream:
        missing_environment = pickle.load(stream)
    del missing_environment["environment_configuration"]
    incomplete = tmp_path / "missing_environment.pkl"
    with incomplete.open("wb") as stream:
        pickle.dump(missing_environment, stream)
    with pytest.raises(ValueError, match="no recorded environment"):
        load_checkpoint(incomplete, configuration, seeds, environment_configuration)
