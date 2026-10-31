"""Check preprocessing and reset semantics without loading game sprites."""

from typing import NamedTuple
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from exploration_platform.atari_pixels.environment import (
    AtariPixelEnvironment,
    area_resize_frame,
    area_resize_weights,
    grayscale_frame,
    _install_room_background_cache,
    NOOP_STREAM,
)


def test_room_background_cache_uses_complete_state_and_each_room_geometry():
    """Caching accepts the added fire-history field and keeps rooms distinct."""
    # A complete reset template carries prev_is_fire; old renderer templates omitted it.
    class CacheState(NamedTuple):
        """Retain complete-state evidence and one room's static geometry."""

        geometry: jax.Array
        prev_is_fire: jax.Array

    class CacheRenderer:
        """Draw small rasters using only the room geometry supplied by the loader."""

        def _load_room_geometry(self, room, state):
            """Keep fire history while replacing the selected room geometry."""
            # Room zero becomes geometry 10; room two becomes geometry 12.
            return state._replace(geometry=room + jnp.int32(10))

        def _build_room_background(self, room, state):
            """Expose incorrect geometry loads or incomplete reset states directly."""
            # Before: one geometry value; after: a constant two-by-two palette raster.
            assert state.prev_is_fire.dtype == jnp.bool_
            return jnp.full((2, 2), state.geometry + room, dtype=jnp.int32)

    class CacheGame:
        """Use a valid reset state instead of the upstream renderer's incomplete template."""

        def __init__(self, room_count):
            """Give the fixture a separate renderer and explicit room count."""
            # One room also checks that the leading cache axis is retained.
            self.renderer = CacheRenderer()
            self.consts = SimpleNamespace(MAX_ROOMS=room_count)

        def reset(self, key):
            """Return all game fields without consuming a learner's training key."""
            # The test fixture has no random game quantities.
            del key
            return None, CacheState(jnp.int32(-1), jnp.bool_(False))

    # Check both one room and several distinct room backgrounds.
    for count in (1, 3):
        game = CacheGame(count)
        _install_room_background_cache(game)
        expected = np.broadcast_to((10 + 2 * np.arange(count))[:, None, None], (count, 2, 2))
        np.testing.assert_array_equal(game.renderer.room_backgrounds, expected)
        assert game.renderer.pre_load_rooms is True


def test_area_ratio_byte_rounding_keeps_even_ties_and_adjacent_values():
    """Integer rounding keeps half-even ties without unsupported Triton primitives."""
    from exploration_platform.atari_pixels.pixel_preprocessing import _round_area_ratio_to_bytes

    # Numerators cover exact half bytes and the immediately neighboring rational values.
    denominator = jnp.int32(33600)
    totals = jnp.asarray([0, 16800, 50400, 84000, 8551200, 8568000,
                          16799, 16801, 50399, 50401, 83999, 84001], jnp.int32)
    expected = np.asarray([0, 0, 2, 2, 254, 255, 0, 1, 1, 2, 2, 3], np.uint8)
    np.testing.assert_array_equal(_round_area_ratio_to_bytes(totals, denominator), expected)


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
        self.render_final_two_only = False
        self.frame_skip_unroll = 1
        self.batch_layout = "flat"
        self.stack_update = "concatenate"
        self.reset_only_when_needed = True
        self.fuse_pixel_preprocessing = False
        self.combine_palette_and_sprite_drawing = False
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


def test_grayscale_rounds_exact_half_values_up_before_area_reduction():
    """ALE rounds grayscale halfway values up; area-resize rounding is a separate operation."""
    # Exact luminances are 28.5, 21.5, 14.5, 7.5 and 22.5, using decimal ALE coefficients.
    rgb = jnp.asarray([[0, 0, 250], [0, 4, 168], [0, 8, 86], [0, 12, 4], [10, 1, 166]],
                      dtype=jnp.uint8)
    np.testing.assert_array_equal(grayscale_frame(rgb), [29, 22, 15, 8, 23])


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
    np.testing.assert_array_equal(info["final_observation"][:, :, -1, 0, 0],
                                   np.full((2, 2), 9))
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


def test_nonterminal_life_loss_from_one_to_zero_still_masks_ppo_returns():
    """JAXtari's zero-lives state remains playable until the final loss below zero."""
    # The pinned game's own AtariWrapper checks old lives > 0, not new lives > 0.
    environment = SmallPixelEnvironment()
    state, _ = jax.jit(environment.reset)(jnp.asarray([7], dtype=jnp.int32))
    state = state._replace(game=state.game._replace(lives=jnp.ones_like(state.game.lives)))
    state, _, _, done, info = jax.jit(environment.step)(state, jnp.full((1, 2), 2, jnp.int32))
    np.testing.assert_array_equal(state.game.lives, [0, 0])
    assert bool(jnp.all(done & info["life_lost"]))
    assert not bool(jnp.any(info["episode_done"] | info["full_game_reset"]))


def test_game_over_stops_held_frames_and_immediately_resets_the_full_stack():
    """A real game-over mask stops further held frames and preserves its terminal pixels."""
    # Position 98 plus one action-2 frame reaches game over; later held frames must not execute.
    environment = SmallPixelEnvironment(stack_update="circular")
    state, _ = jax.jit(environment.reset)(jnp.asarray([7], dtype=jnp.int32))
    state = state._replace(game=state.game._replace(
        position=jnp.full_like(state.game.position, 98), lives=jnp.zeros_like(state.game.lives)))
    state, observations, rewards, done, info = jax.jit(environment.step)(
        state, jnp.full((1, 2), 2, jnp.int32))
    assert bool(jnp.all(done & info["game_over"] & info["episode_done"] & info["full_game_reset"]))
    np.testing.assert_array_equal(info["raw_reward"], [[2, 2]])
    np.testing.assert_array_equal(info["final_observation"][..., -1, 0, 0], [[100, 100]])
    np.testing.assert_array_equal(rewards, [[1, 1]])
    np.testing.assert_array_equal(state.frame_cursor, [0, 0])
    np.testing.assert_array_equal(observations, np.ones((1, 2, 4, 84, 84), np.uint8))


def test_policy_seed_pairing_preserves_distinct_environment_keys():
    """Paired policy seeds retain matching lanes and distinct environment keys."""
    # Matching policy seeds pair trajectories; distinct environment indices separate keys.
    env = SmallPixelEnvironment()
    state, _ = jax.jit(env.reset)(jnp.asarray([19, 19], dtype=jnp.int32))
    np.testing.assert_array_equal(state.key[0], state.key[2])
    np.testing.assert_array_equal(state.key[1], state.key[3])
    assert not bool(jnp.all(state.key[0] == state.key[1]))


def test_environment_optimization_options_preserve_every_output():
    """Compare alternate batching, rendering, stacking and scans on a life-reset transition."""
    # Every candidate starts from the exact same saved state and receives the same held actions.
    seeds = jnp.asarray([7, 11], dtype=jnp.int32)
    actions = jnp.full((2, 2), 2, jnp.int32)
    baseline = SmallPixelEnvironment()
    state, _ = jax.jit(baseline.reset)(seeds)
    expected = jax.jit(baseline.step)(state, actions)
    candidates = [
        {"render_final_two_only": True},
        {"frame_skip_unroll": 2},
        {"frame_skip_unroll": 4},
        {"batch_layout": "nested"},
        {"stack_update": "gather"},
        {"stack_update": "circular"},
        {"reset_only_when_needed": False},
    ]
    for options in candidates:
        candidate = SmallPixelEnvironment(**options)
        actual = jax.jit(candidate.step)(state, actions)
        normalized = (candidate.canonical_state(actual[0]), *actual[1:])
        for wanted, observed in zip(jax.tree.leaves(expected), jax.tree.leaves(normalized)):
            np.testing.assert_array_equal(observed, wanted, err_msg=str(options))


def test_circular_storage_preserves_multiple_steps_and_life_reset():
    """A real circular buffer keeps chronological observations across cursor wraparound."""
    # Repeated appends change physical order; canonical frames, game state and pixels stay equal.
    baseline = SmallPixelEnvironment()
    circular = SmallPixelEnvironment(stack_update="circular")
    reference, _ = jax.jit(baseline.reset)(jnp.asarray([3], dtype=jnp.int32))
    current, _ = jax.jit(circular.reset)(jnp.asarray([3], dtype=jnp.int32))
    for action in [2, 1, 2, 1, 2, 1]:
        actions = jnp.full((1, 2), action, jnp.int32)
        expected = jax.jit(baseline.step)(reference, actions)
        actual = jax.jit(circular.step)(current, actions)
        reference, current = expected[0], actual[0]
        normalized = (circular.canonical_state(current), *actual[1:])
        for wanted, observed in zip(jax.tree.leaves(expected), jax.tree.leaves(normalized)):
            np.testing.assert_array_equal(observed, wanted)
        np.testing.assert_array_equal(circular.observe(current), actual[1])


def test_game_end_during_life_reset_reports_the_actual_full_stack_reset():
    """Indexed frame reconstruction must recognize a full reset caused by reset NOOPs."""
    class NoopEndsGame(SmallGame):
        """End the fixture on the first reset NOOP following a life loss."""

        def step(self, state, action):
            """Preserve the original action transition, then end during reset NOOP processing."""
            observation, next_state, reward, done, info = super().step(state, action)
            done = done | ((action == 0) & (state.lives == 2))
            return observation, next_state, reward, done, info

    class ForcedNoopEnvironment(SmallPixelEnvironment):
        """Retain named random streams while ensuring the fixture performs one reset NOOP."""

        def _named_key(self, key, stream, index):
            """Use a supplied deterministic NOOP-count key and preserve all other streams."""
            return self.noop_count_key if stream == NOOP_STREAM else super()._named_key(key, stream, index)

    # Select a fixed key whose pinned JAX generator requests one NOOP rather than zero.
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(32, dtype=jnp.uint32))
    counts = jax.vmap(lambda key: jax.random.randint(key, (), 0, 2, dtype=jnp.int32))(keys)
    selection = np.flatnonzero(np.asarray(counts) == 1)[0]
    environment = ForcedNoopEnvironment(noop_max=2, stack_update="circular")
    environment.noop_count_key = keys[selection]
    environment.game = NoopEndsGame()
    state, _ = jax.jit(environment.reset)(jnp.asarray([7], dtype=jnp.int32))
    state, observations, _, done, info = jax.jit(environment.step)(
        state, jnp.full((1, 2), 2, jnp.int32))
    assert bool(jnp.all(done & info["life_lost"] & info["full_game_reset"]))
    assert not bool(jnp.any(info["episode_done"]))
    np.testing.assert_array_equal(state.frame_cursor, [0, 0])
    np.testing.assert_array_equal(observations, np.ones((1, 2, 4, 84, 84), np.uint8))


def test_fused_area_kernel_matches_reference_after_rounding_correction():
    """Interpret the GPU kernel on deterministic RGB patterns and check every output byte."""
    from exploration_platform.atari_pixels.pixel_preprocessing import fused_area_preprocess

    # A full color pattern exercises fractional area weights and both grayscale/max orderings.
    indices = jnp.arange(210 * 160 * 3, dtype=jnp.int32).reshape(210, 160, 3)
    first = ((indices * 17 + 3) % 256).astype(jnp.uint8)
    second = ((indices * 31 + 119) % 256).astype(jnp.uint8)
    row_weights, column_weights = area_resize_weights(210), area_resize_weights(160)
    reference = area_resize_frame(jnp.maximum(grayscale_frame(first), grayscale_frame(second)),
                                   row_weights, column_weights)
    result, ambiguous = fused_area_preprocess(first, second, interpret=True)
    corrected = jnp.where(ambiguous, reference, result)
    np.testing.assert_array_equal(corrected, reference)
