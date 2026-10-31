"""Small complete learner checks; numerical execution belongs on an allocated compute node."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

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
            "episode_return": counters.astype(jnp.float32), "episode_done": done}


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
