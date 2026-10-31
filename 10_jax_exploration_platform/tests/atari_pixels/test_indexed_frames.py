"""Check frame-bank reconstruction across warmup and same-step episode boundaries."""

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.training import (PixelPPORND, flatten_rollout,
                                                        gather_indexed_frames)


class FrameState(NamedTuple):
    """Small deterministic frame stack with a known terminal/reset image distinction."""
    count: object
    frames: object


class FrameEnvironment:
    """Append integer frames and replace every third terminal stack by reset image seven."""

    action_count = 18

    def reset(self, seeds):
        """Start all lanes with the policy seed repeated across the full initial stack."""
        count = jnp.zeros((len(seeds), 2), jnp.int32)
        frames = jnp.broadcast_to(seeds[:, None, None, None, None], (len(seeds), 2, 4, 84, 84)).astype(jnp.uint8)
        return FrameState(count, frames), frames

    def observe(self, state):
        """Expose the real post-warmup frame history without advancing it."""
        return state.frames

    def step(self, state, actions):
        """Report pre-reset terminal images while returning the reset stack for the next action."""
        count = state.count + 1
        frame = jnp.broadcast_to(count[..., None, None, None].astype(jnp.uint8), (count.shape[0], 2, 1, 84, 84))
        final = jnp.concatenate((state.frames[..., 1:, :, :], frame), axis=2)
        done = count % 3 == 0
        frames = jnp.where(done[..., None, None, None], jnp.uint8(7), final)
        return FrameState(count, frames), frames, jnp.zeros(count.shape, jnp.float32), done, {
            "episode_return": count.astype(jnp.float32), "episode_done": done,
            "final_observation": final, "full_game_reset": done}


def test_indexed_storage_preserves_stale_first_pixels_and_reset_stacks():
    """Reconstruct all four current frames exactly despite warmup and repeated full resets."""
    configuration = PixelPPOConfig(n_envs=2, num_steps=4, num_minibatches=1,
                                    update_epochs=1, observation_warmup_iterations=1,
                                    total_timesteps=64)
    reference = PixelPPORND(configuration, FrameEnvironment())
    state = reference.warmup(reference.initialize((17,)))
    _, expected = reference.collect(state)
    indexed = PixelPPORND(replace(configuration, rollout_frame_storage="indexed_frames"), FrameEnvironment())
    after, actual = indexed.collect(state)
    lanes = jnp.broadcast_to(jnp.tile(jnp.arange(2, dtype=jnp.int32), 4), (1, 8))
    reconstructed = gather_indexed_frames(actual.frame_bank, lanes, flatten_rollout(actual.observations))
    np.testing.assert_array_equal(reconstructed, flatten_rollout(expected.observations).astype(jnp.uint8))
    # The upstream first observation is seed 17, while the actual warmed game has already advanced.
    assert int(reconstructed[0, 0, 0, 0, 0]) == 17
    assert np.any(np.asarray(actual.next_frames) == 6)
    _, expected_batch = reference.prepare_batch(after, expected)
    _, actual_batch = indexed.prepare_batch(after, actual)
    np.testing.assert_array_equal(expected_batch["normalized_frames"], actual_batch["normalized_frames"])
