"""Check preprocessing and reset semantics without loading game sprites."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exploration_platform.atari_pixels.environment import (
    AtariPixelEnvironment,
    area_resize_frame,
    area_resize_weights,
    grayscale_frame,
)


class SmallGameState(NamedTuple):
    """Retain only the position and lives needed by the adapter fixture."""

    position: jax.Array
    lives: jax.Array


class SmallGame:
    """Give reset and held-frame tests a deterministic game with one life boundary."""

    def reset(self, key):
        """Start the fixture at position zero and three lives."""
        # The seed is unused because this fixture's transitions are deterministic.
        del key
        state = SmallGameState(jnp.int32(0), jnp.int32(3))
        return state.position, state

    def step(self, state, action):
        """Move by the action value and lose one life when first crossing position four."""
        # A four-frame action of two moves 1 -> 3 -> 5 -> 7 -> 9 and loses exactly one life.
        position = state.position + action
        crossed = (state.position < 4) & (position >= 4)
        state = SmallGameState(position, state.lives - crossed.astype(jnp.int32))
        return position, state, action.astype(jnp.float32), position >= 100, {}


class SmallPixelEnvironment(AtariPixelEnvironment):
    """Exercise adapter control flow while keeping renderer compilation inexpensive."""

    def __init__(self, **overrides):
        """Configure the fixture without importing JAXtari or allocating game assets."""
        # The default reset contains no NOOPs and one FIRE frame.
        self.num_envs = 2
        self.frame_skip = 4
        self.max_episode_steps = 108000
        self.noop_max = 0
        self.episodic_life = True
        self.sticky_action_probability = 0.0
        self.game = SmallGame()
        for name, value in overrides.items():
            setattr(self, name, value)

    def _frame(self, game):
        """Represent the fixture's position as a single grayscale pixel."""
        # Using [1,1] images isolates reset and frame-skip behavior from resizing cost.
        return jnp.asarray([[game.position]], dtype=jnp.uint8)

    def _resize(self, gray):
        """Expand the fixture's pixel into the learner's required 84 by 84 shape."""
        # Before: [1,1]; after: an [84,84] constant image with the same pixel value.
        return jnp.broadcast_to(gray, (84, 84))


def test_grayscale_matches_recorded_ale_coefficients():
    """Check primary colors and prove grayscale must precede maximum pooling."""
    # Red/green/blue/white should produce ALE's rounded 76/150/29/255 bytes.
    rgb = jnp.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
                      dtype=jnp.uint8)
    np.testing.assert_array_equal(grayscale_frame(rgb), [76, 150, 29, 255])
    assert int(jnp.maximum(*grayscale_frame(rgb[:2]))) == 150
    assert int(grayscale_frame(jnp.maximum(rgb[0], rgb[1]))) == 226


def test_area_resize_preserves_constants_and_averages_regions():
    """Check the area estimator on one exact average and a constant Atari-sized frame."""
    # Before: [[0,20],[40,60]]; after: the single area average [[30]].
    weights = area_resize_weights(2, 1)
    result = area_resize_frame(jnp.asarray([[0, 20], [40, 60]], dtype=jnp.uint8),
                               weights, weights)
    np.testing.assert_array_equal(result, [[30]])
    constant = area_resize_frame(jnp.full((210, 160), 73, jnp.uint8),
                                 area_resize_weights(210), area_resize_weights(160))
    np.testing.assert_array_equal(constant, np.full((84, 84), 73, np.uint8))


def test_hold_four_frames_and_continue_game_after_life_loss():
    """A life boundary masks PPO returns, preserves game progress, and runs no extra row."""
    # Reset's FIRE puts position at 1; four action-2 frames reach 9, reset FIRE reaches 10.
    env = SmallPixelEnvironment()
    state, obs = jax.jit(env.reset)(jnp.asarray([7, 11], dtype=jnp.int32))
    assert obs.shape == (2, 2, 4, 84, 84)
    next_state, next_obs, rewards, done, info = jax.jit(env.step)(
        state, jnp.full((2, 2), 2, jnp.int32))
    np.testing.assert_array_equal(next_state.game.position, np.full(4, 10))
    np.testing.assert_array_equal(info["raw_reward"], np.full((2, 2), 8))
    np.testing.assert_array_equal(rewards, np.ones((2, 2)))
    np.testing.assert_array_equal(done, np.ones((2, 2), bool))
    assert not bool(jnp.any(info["episode_done"]))
    np.testing.assert_array_equal(next_obs[:, :, -1, 0, 0], np.full((2, 2), 10))


def test_time_limit_resets_full_game_and_retains_terminal_return():
    """A time-limit boundary fills a new frame stack and reports the completed return."""
    # The first action reaches the one-step limit; returned pixels already belong to reset.
    env = SmallPixelEnvironment(max_episode_steps=1)
    state, _ = jax.jit(env.reset)(jnp.asarray([7], dtype=jnp.int32))
    state, obs, _, done, info = jax.jit(env.step)(state, jnp.full((1, 2), 2, jnp.int32))
    np.testing.assert_array_equal(state.game.position, [1, 1])
    np.testing.assert_array_equal(state.step_count, [0, 0])
    np.testing.assert_array_equal(state.episode_return, [0, 0])
    np.testing.assert_array_equal(info["episode_return"], [[8, 8]])
    assert bool(jnp.all(done & info["truncated"] & info["episode_done"]))
    np.testing.assert_array_equal(obs, np.ones((1, 2, 4, 84, 84), np.uint8))


def test_noop_stream_does_not_change_sticky_stream_identifier():
    """Named streams give identical lane seeds under paired policy groups."""
    # Matching policy seeds pair trajectories; distinct environment indices separate keys.
    env = SmallPixelEnvironment()
    state, _ = jax.jit(env.reset)(jnp.asarray([19, 19], dtype=jnp.int32))
    np.testing.assert_array_equal(state.key[0], state.key[2])
    np.testing.assert_array_equal(state.key[1], state.key[3])
    assert not bool(jnp.all(state.key[0] == state.key[1]))
