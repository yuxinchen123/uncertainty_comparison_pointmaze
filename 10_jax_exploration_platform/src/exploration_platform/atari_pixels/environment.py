"""Pixel Montezuma adapter with the recorded EnvPool 0.6.6 preprocessing.

The game is JAXtari at commit 04ab9aec318c72162e2e53503a594a489d7e162b.
Its game logic is a reimplementation; this module does not claim ALE trajectories.
Grayscale precedes the final-two-frame maximum, followed by area resizing and
four-frame stacking. Resets happen inside the transition, so no action is lost
to a separate reset-only call.
"""

import hashlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


def _stream_identifier(name: str) -> int:
    """Give each random quantity a stable identifier independent of draw order."""
    # Adding another identifier never changes an existing quantity's stream.
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


GAME_STREAM = _stream_identifier("montezuma_pixels.game_initialization")
NOOP_STREAM = _stream_identifier("montezuma_pixels.reset_noop_count")
STICKY_STREAM = _stream_identifier("montezuma_pixels.sticky_action")
ENVIRONMENT_STREAM = _stream_identifier("montezuma_pixels.environment_index")


class PixelEnvironmentState(NamedTuple):
    """Checkpointable game, observation, counter, and random state for every lane."""

    game: Any
    frames: jax.Array
    key: jax.Array
    frame_count: jax.Array
    step_count: jax.Array
    reset_count: jax.Array
    previous_action: jax.Array
    episode_return: jax.Array


def grayscale_frame(rgb: jax.Array) -> jax.Array:
    """Use ALE's rounded 0.2989/0.5870/0.1140 grayscale conversion."""
    # Before: RGB [..., 3] uint8; after: one rounded grayscale byte per pixel.
    rgb = rgb.astype(jnp.int32)
    luminance = rgb[..., 0] * 2989 + rgb[..., 1] * 5870 + rgb[..., 2] * 1140
    return ((luminance + 5000) // 10000).astype(jnp.uint8)


def area_resize_weights(source_size: int, destination_size: int = 84) -> jax.Array:
    """Build exact pixel-overlap weights for a one-dimensional area reduction."""
    # Before: 210 source rows; after: [84, 210] weights whose rows each sum to one.
    scale = source_size / destination_size
    rows = []
    for output_index in range(destination_size):
        left, right = output_index * scale, (output_index + 1) * scale
        rows.append([
            max(0.0, min(right, index + 1) - max(left, index)) / scale
            for index in range(source_size)
        ])
    return jnp.asarray(rows, dtype=jnp.float32)


def area_resize_frame(gray: jax.Array, row_weights: jax.Array,
                      column_weights: jax.Array) -> jax.Array:
    """Area-average a grayscale image and round its result to observation bytes."""
    # Before: [210, 160] uint8; after: [84, 84] uint8, with no bilinear filtering.
    rows = jnp.matmul(row_weights, gray.astype(jnp.float32),
                      precision=jax.lax.Precision.HIGHEST)
    reduced = jnp.matmul(rows, column_weights.T, precision=jax.lax.Precision.HIGHEST)
    return jnp.clip(jnp.rint(reduced), 0, 255).astype(jnp.uint8)


class AtariPixelEnvironment:
    """Expose independent [policy copy, environment] lanes to the PPO learner."""

    action_count = 18
    observation_shape = (4, 84, 84)

    def __init__(self, num_envs: int = 128, frame_skip: int = 4,
                 max_episode_steps: int = 108000, noop_max: int = 30,
                 episodic_life: bool = True,
                 sticky_action_probability: float = 0.25,
                 render_final_two_only: bool = False, frame_skip_unroll: int = 1,
                 batch_layout: str = "flat", stack_update: str = "concatenate",
                 reset_only_when_needed: bool = True,
                 cache_room_backgrounds: bool = False):
        """Load the pinned game lazily and retain only static preprocessing settings."""
        # Keep package imports out of learner-only tests that use a small fake environment.
        import jaxtari

        if num_envs < 1 or frame_skip < 1 or max_episode_steps < 1 or noop_max < 0:
            raise ValueError("environment counts and limits must be positive; noop_max may be zero")
        if not 0.0 <= sticky_action_probability <= 1.0:
            raise ValueError("sticky_action_probability must be between zero and one")
        if frame_skip_unroll < 1:
            raise ValueError("frame_skip_unroll must be positive")
        if batch_layout not in ("flat", "nested"):
            raise ValueError("batch_layout must be 'flat' or 'nested'")
        if stack_update not in ("concatenate", "gather"):
            raise ValueError("stack_update must be 'concatenate' or 'gather'")
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.noop_max = noop_max
        self.episodic_life = episodic_life
        self.sticky_action_probability = sticky_action_probability
        self.render_final_two_only = render_final_two_only
        self.frame_skip_unroll = frame_skip_unroll
        self.batch_layout = batch_layout
        self.stack_update = stack_update
        self.reset_only_when_needed = reset_only_when_needed
        self.cache_room_backgrounds = cache_room_backgrounds

        # All calls into this game must use int32; the older platform enables x64 globally.
        with jax.enable_x64(False):
            if cache_room_backgrounds:
                # Upstream caches only static room geometry; actors, items, and darkness stay dynamic.
                from jaxtari.games.jax_montezumarevenge import JaxMontezumaRevenge
                from jaxtari.games.montezuma_revenge.core import MontezumaRevengeConstants

                jaxtari.check_ownership()
                constants = MontezumaRevengeConstants(RENDERER_PRELOAD_ROOMS=True)
                self.game = JaxMontezumaRevenge(constants)
            else:
                self.game = jaxtari.make("montezumarevenge")
            height, width, channels = self.game.image_space().shape
            if channels != 3 or self.game.action_space().n != self.action_count:
                raise ValueError("the pinned Montezuma game must expose RGB pixels and 18 actions")
            self.row_weights = area_resize_weights(height)
            self.column_weights = area_resize_weights(width)

    def _named_key(self, key, stream, index):
        """Derive one event's key without advancing any other random quantity."""
        # For example, a new logging stream leaves every sticky-action draw unchanged.
        return jax.random.fold_in(jax.random.fold_in(key, stream), index)

    def _frame(self, game):
        """Render and grayscale one game state before any pooling or resizing."""
        # The renderer returns [210, 160, 3]; grayscale must happen before maximum pooling.
        return grayscale_frame(self.game.render(game))

    def _resize(self, gray):
        """Apply the recorded area resize to one raw grayscale image."""
        # Keep the final observation small while the game keeps its native coordinate system.
        return area_resize_frame(gray, self.row_weights, self.column_weights)

    def _append_frame(self, frames, frame):
        """Append identical pixels using either concatenation or an explicit gather."""
        # Before: [a,b,c,d]; after: [b,c,d,new], under either implementation.
        if self.stack_update == "gather":
            indices = jnp.asarray([1, 2, 3, 3], dtype=jnp.int32)
            return jnp.take(frames, indices, axis=0).at[3].set(frame)
        return jnp.concatenate((frames[1:], frame[None]), axis=0)

    def _map_lanes(self, function, shape, *arguments):
        """Map the same scalar transition with flat or nested policy/environment axes."""
        # Before: [C*N,...]; nested form temporarily uses [C,N,...], then returns flat lanes.
        if self.batch_layout == "flat":
            return jax.vmap(function)(*arguments)
        nested = jax.tree.map(lambda value: value.reshape(shape + value.shape[1:]), arguments)
        result = jax.vmap(jax.vmap(function))(*nested)
        return jax.tree.map(lambda value: value.reshape((-1,) + value.shape[2:]), result)

    def _raw_frame(self, game, key, frame_count, previous_action, requested_action):
        """Advance exactly one native game frame, including its sticky-action draw."""
        # A frame's draw depends on its own index, independent of reset-noop randomness.
        sticky_key = self._named_key(key, STICKY_STREAM, frame_count)
        sticky = jax.random.bernoulli(sticky_key, self.sticky_action_probability)
        action = jnp.where(sticky, previous_action, requested_action).astype(jnp.int32)
        _, next_game, reward, done, _ = self.game.step(game, action)
        return next_game, frame_count + jnp.int32(1), action, reward, done

    def _reset_single(self, state, reset_game):
        """Apply NOOP/FIRE initialization, preserving the game after a life loss."""
        # Full-game resets use a new named key; a life-loss reset keeps score and inventory.
        reset_index = state.reset_count + jnp.int32(1)
        game_key = self._named_key(state.key, GAME_STREAM, reset_index)
        game = jax.lax.cond(reset_game, lambda _: self.game.reset(game_key)[1],
                            lambda _: state.game, operand=None)
        frame_count = state.frame_count
        previous_action = jnp.where(reset_game, jnp.int32(0), state.previous_action)
        noop_key = self._named_key(state.key, NOOP_STREAM, reset_index)
        noop_count = (jax.random.randint(noop_key, (), 0, self.noop_max, dtype=jnp.int32)
                      if self.noop_max else jnp.int32(0))

        # EnvPool's FIRE games take 0..29 raw NOOP frames, then one raw FIRE frame.
        def noop_body(index, carry):
            """Advance one reset NOOP, starting a fresh game if the old game ends."""
            current, frames, action, full_reset = carry
            current, frames, action, _, ended = self._raw_frame(
                current, state.key, frames, action, jnp.int32(0))
            restart_key = self._named_key(game_key, GAME_STREAM, index)
            current = jax.lax.cond(ended, lambda _: self.game.reset(restart_key)[1],
                                   lambda _: current, operand=None)
            return current, frames, action, full_reset | ended

        game, frame_count, previous_action, reset_game = jax.lax.fori_loop(
            0, noop_count, noop_body, (game, frame_count, previous_action, reset_game))
        game, frame_count, previous_action, _, _ = self._raw_frame(
            game, state.key, frame_count, previous_action, jnp.int32(1))
        frame = self._resize(self._frame(game))

        # Before: four older frames; after: a new full stack, or one appended life-reset frame.
        full_stack = jnp.broadcast_to(frame, self.observation_shape)
        continued_stack = self._append_frame(state.frames, frame)
        frames = jnp.where(reset_game, full_stack, continued_stack)
        return PixelEnvironmentState(
            game, frames, state.key, frame_count,
            jnp.where(reset_game, jnp.int32(0), state.step_count), reset_index,
            previous_action, jnp.where(reset_game, jnp.float32(0), state.episode_return))

    def _initial_single(self, key):
        """Initialize one lane, including the first NOOP/FIRE reset and frame stack."""
        # The placeholder stack is replaced completely by the first full-game reset.
        game = self.game.reset(self._named_key(key, GAME_STREAM, jnp.int32(0)))[1]
        state = PixelEnvironmentState(
            game, jnp.zeros(self.observation_shape, jnp.uint8), key,
            jnp.int32(0), jnp.int32(0), jnp.int32(-1), jnp.int32(0), jnp.float32(0))
        return self._reset_single(state, jnp.bool_(True))

    def reset(self, seed_indices):
        """Reset C policies' N independent environments and return state plus byte pixels."""
        # Before: seeds [C]; after: unique (seed, environment-index) keys [C*N, 2].
        with jax.enable_x64(False):
            seeds = jnp.asarray(seed_indices, dtype=jnp.uint32)
            lane_indices = jnp.arange(self.num_envs, dtype=jnp.uint32)
            keys = jax.vmap(lambda seed: jax.vmap(lambda lane: self._named_key(
                jax.random.PRNGKey(seed), ENVIRONMENT_STREAM, lane))(lane_indices))(seeds)
            shape = (seeds.shape[0], self.num_envs)
            state = self._map_lanes(self._initial_single, shape, keys.reshape(-1, 2))
            state = jax.tree.map(jnp.copy, state)
            return state, state.frames.reshape(shape + self.observation_shape)

    def observe(self, state):
        """Read the current stored frame history without advancing game or random state."""
        # Before: [C*N,4,84,84]; after: the same bytes viewed as [C,N,4,84,84].
        copies = state.frames.shape[0] // self.num_envs
        return state.frames.reshape((copies, self.num_envs) + self.observation_shape)

    def _step_single(self, state, action):
        """Advance one held action and report masks before an immediate reset."""
        # Stop native frames at game over; life-loss masks are decided after the held action.
        initial_frame = state.game if self.render_final_two_only else self._frame(state.game)

        def frame_body(carry, _):
            """Advance one held frame and retain the latest two grayscale screens."""
            game, count, previous, total_reward, ended, prior, latest = carry

            def active(_):
                """Compute a game frame only while this game remains active."""
                next_game, next_count, next_action, reward, done = self._raw_frame(
                    game, state.key, count, previous, action)
                latest_frame = (next_game if self.render_final_two_only
                                else self._frame(next_game))
                return (next_game, next_count, next_action, total_reward + reward,
                        done, latest, latest_frame)

            next_carry = jax.lax.cond(ended, lambda _: carry, active, operand=None)
            return next_carry, None

        carry = (state.game, state.frame_count, state.previous_action, jnp.float32(0),
                 jnp.bool_(False), initial_frame, initial_frame)
        (game, count, previous, reward, game_done, prior, latest), _ = jax.lax.scan(
            frame_body, carry, None, length=self.frame_skip, unroll=self.frame_skip_unroll)
        if self.render_final_two_only:
            # Render only the retained last two game states instead of all held-action frames.
            prior, latest = self._frame(prior), self._frame(latest)
        pooled = jnp.maximum(prior, latest) if self.frame_skip > 1 else latest
        frame = self._resize(pooled)
        frames = self._append_frame(state.frames, frame)
        step_count = state.step_count + jnp.int32(1)
        truncated = step_count >= self.max_episode_steps
        life_lost = ((game.lives < state.game.lives) & (game.lives > 0)
                     if self.episodic_life else jnp.bool_(False))
        done = game_done | truncated | life_lost
        episode_return = state.episode_return + reward.astype(jnp.float32)
        next_state = PixelEnvironmentState(game, frames, state.key, count, step_count,
                                            state.reset_count, previous, episode_return)

        # Keep full-game episode statistics separate from the PPO life-loss boundary.
        info = {"episode_return": episode_return, "episode_done": game_done | truncated,
                "raw_reward": reward.astype(jnp.float32), "game_over": game_done,
                "life_lost": life_lost, "truncated": truncated}
        return next_state, jnp.sign(reward).astype(jnp.float32), done, info

    def step(self, state, actions):
        """Step all [C,N] lanes, resetting finished lanes without an extra transition."""
        # Before: actions [C,N]; after: flat state lanes plus pixels and statistics [C,N,...].
        with jax.enable_x64(False):
            actions = jnp.asarray(actions, dtype=jnp.int32)
            shape = actions.shape
            next_state, rewards, done, info = self._map_lanes(
                self._step_single, shape, state, actions.reshape(-1))
            # RND scores the state reached by the action, before any immediate reset.
            # Before: terminal stack [B,4,84,84]; after: info keeps that stack for this transition.
            info["final_observation"] = next_state.frames

            def reset_finished(current):
                """Run NOOP/FIRE resets only on transitions that ended a life or game."""
                reset_lane = lambda lane, ended, full: jax.lax.cond(
                    ended, lambda _: self._reset_single(lane, full), lambda _: lane,
                    operand=None)
                return self._map_lanes(reset_lane, shape, current, done, info["episode_done"])

            if self.reset_only_when_needed:
                next_state = jax.lax.cond(jnp.any(done), reset_finished,
                                           lambda current: current, next_state)
            else:
                next_state = reset_finished(next_state)
            observation = next_state.frames.reshape(shape + self.observation_shape)
            info = jax.tree.map(lambda value: value.reshape(shape + value.shape[1:]), info)
            return next_state, observation, rewards.reshape(shape), done.reshape(shape), info
