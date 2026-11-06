"""The corrected sprite pack gives Montezuma's start room its real geometry. Run on a compute node.

With the alternative pack's own .npy room images the start room is solid below its top floor: the
conveyor carries the player to x = 54, where no action moves it, and no game ever loses a life.
"""

import os

from exploration_platform.atari_pixels.environment import CORRECTED_ASSET_ROOT

os.environ.setdefault("XDG_DATA_HOME", CORRECTED_ASSET_ROOT)
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

RIGHT, LEFT = 3, 4


@pytest.fixture(scope="module")
def game():
    """The pinned game, loaded from the corrected pack."""
    import jaxtari

    with jax.enable_x64(False):
        return jaxtari.make("montezumarevenge")


def hold(game, state, action, frames):
    """Hold one action; return the final state and the per-frame lives."""
    def body(current, _):
        _, current, _, _, _ = game.step(current, jnp.int32(action))
        return current, current.lives
    return jax.lax.scan(body, state, None, length=frames)


def on_conveyor(game):
    """A fresh game with the player standing on the start room's conveyor (top at y 88)."""
    _, state = game.reset(jax.random.PRNGKey(0))
    return state.replace(player_x=jnp.int32(70), player_y=jnp.int32(68))


def test_environment_accepts_the_corrected_pack():
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment

    environment = AtariPixelEnvironment(num_envs=2)
    assert environment.room_count == 33
    assert environment.max_episode_steps == 27000


def test_player_walks_right_against_the_conveyor(game):
    with jax.enable_x64(False):
        state, _ = jax.jit(lambda s: hold(game, s, RIGHT, 32))(on_conveyor(game))
    assert int(state.player_x) > 70 and int(state.lives) == 5


def test_walking_off_the_conveyor_costs_a_life(game):
    with jax.enable_x64(False):
        state, lives = jax.jit(lambda s: hold(game, s, LEFT, 200))(on_conveyor(game))
    assert int(np.asarray(lives).min()) < 5


def test_random_play_loses_lives_and_ends_games(game):
    games, frames = 32, 3000
    with jax.enable_x64(False):
        _, state = jax.vmap(game.reset)(jax.random.split(jax.random.PRNGKey(1), games))

        def body(carry, key):
            current, lost, ended = carry
            actions = jax.random.randint(key, (games,), 0, 18)
            _, after, _, done, _ = jax.vmap(game.step)(current, actions)
            lost = lost + (after.lives < current.lives).sum()
            fresh = jax.vmap(game.reset)(jax.random.split(key, games))[1]
            after = jax.tree.map(lambda new, old: jnp.where(
                done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old), fresh, after)
            return (after, lost, ended + done.sum()), None

        (_, lost, ended), _ = jax.jit(lambda s, k: jax.lax.scan(body, (s, 0, 0), k))(
            state, jax.random.split(jax.random.PRNGKey(2), frames))
    assert int(lost) > games and int(ended) > 0
