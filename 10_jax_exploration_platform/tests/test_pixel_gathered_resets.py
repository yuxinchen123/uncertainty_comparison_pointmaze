"""Resetting only the ended lanes, gathered into a small batch, gives exactly the per-lane reset.

Run on a compute node. Uses the corrected sprite pack, where lives are lost often under random
actions, and a short episode limit so that time-limit resets also occur. A budget of two lanes
exercises the gathered path and the fallback for steps where more lanes ended.
"""

import os

from exploration_platform.atari_pixels.environment import CORRECTED_ASSET_ROOT

os.environ.setdefault("XDG_DATA_HOME", CORRECTED_ASSET_ROOT)
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import numpy as np
import pytest

SETTINGS = dict(num_envs=32, max_episode_steps=150, cache_room_backgrounds=True, render_final_two_only=True,
                separable_rectangle_drawing=True, unroll_fixed_render_loops=True, masked_conditional_draws=True,
                two_way_masked_draws=True, render_final_two_together=True)


def test_stored_start_state_equals_game_reset():
    """The pinned reset only stores its key: the stored start state with the key is game.reset(key)."""
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    environment = AtariPixelEnvironment(num_envs=2, constant_game_reset=True)
    with jax.enable_x64(False):
        for index in range(16):
            key = jax.random.fold_in(jax.random.PRNGKey(5), index)
            expected = jax.jit(lambda k: environment.game.reset(k)[1])(key)
            actual = jax.jit(environment._game_reset)(key)
            for wanted, observed in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
                assert observed.dtype == wanted.dtype
                np.testing.assert_array_equal(np.asarray(observed), np.asarray(wanted))


@pytest.mark.parametrize("options", [{"reset_lane_budget": 2},
                                     {"reset_lane_budget": 2, "constant_game_reset": True}])
def test_gathered_resets_match_per_lane_resets(options):
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    per_lane = AtariPixelEnvironment(**SETTINGS)
    gathered = AtariPixelEnvironment(**options, **SETTINGS)
    state, _ = per_lane.reset([3, 4])
    other, _ = gathered.reset([3, 4])
    step_per_lane, step_gathered = jax.jit(per_lane.step), jax.jit(gathered.step)
    key = jax.random.PRNGKey(17)
    ended_counts = []
    for index in range(400):
        actions = jax.random.randint(jax.random.fold_in(key, index), (2, 32), 0, 18)
        state, *outputs = step_per_lane(state, actions)
        other, *other_outputs = step_gathered(other, actions)
        for wanted, observed in zip(jax.tree.leaves((state, outputs)), jax.tree.leaves((other, other_outputs))):
            np.testing.assert_array_equal(np.asarray(observed), np.asarray(wanted), err_msg=f"step {index}")
        ended_counts.append(int(np.asarray(outputs[2]).sum()))
    # Both paths ran: steps with one or two ended lanes, and steps with more than the budget.
    assert any(0 < count <= 2 for count in ended_counts)
    assert any(count > 2 for count in ended_counts)
