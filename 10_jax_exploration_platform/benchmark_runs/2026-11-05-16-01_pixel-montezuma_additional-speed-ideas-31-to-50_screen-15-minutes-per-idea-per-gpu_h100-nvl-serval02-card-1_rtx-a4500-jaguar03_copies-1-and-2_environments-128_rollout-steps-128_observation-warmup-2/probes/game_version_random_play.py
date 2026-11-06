"""Long uniform-random play of whichever JAXtari Montezuma is importable: stuck spots, deaths, game ends, rooms.

Run once with the installed (pinned) package and once with PYTHONPATH pointing at another source tree.
Each game that ends is reset with its own key, as a training environment would be.
"""
import os
import sys
import time

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import numpy as np
import jax
import jax.numpy as jnp
import jaxtari

GAMES, CHUNK, CHUNKS = 128, 5000, 9
print("jaxtari source:", jaxtari.__file__, flush=True)
with jax.enable_x64(False):
    game = jaxtari.make("montezumarevenge")
    root = jax.random.PRNGKey(0)
    _, state = jax.vmap(game.reset)(jax.random.split(jax.random.fold_in(root, 1), GAMES))

    def body(carry, inputs):
        """One random action per game; count life losses and game ends, then reset ended games."""
        state, lost, ended = carry
        action_key, reset_key = inputs
        actions = jax.random.randint(action_key, (GAMES,), 0, 18)
        _, after, _, done, _ = jax.vmap(game.step)(state, actions)
        lost = lost + (after.lives < state.lives).astype(jnp.int32)
        ended = ended + done.astype(jnp.int32)
        _, fresh = jax.vmap(game.reset)(jax.random.split(reset_key, GAMES))
        after = jax.tree.map(lambda new, old: jnp.where(
            done.reshape(done.shape + (1,) * (old.ndim - 1)), new, old), fresh, after)
        return (after, lost, ended), (after.player_x, after.room_id)

    run = jax.jit(lambda carry, keys: jax.lax.scan(body, carry, keys))
    carry = (state, jnp.zeros(GAMES, jnp.int32), jnp.zeros(GAMES, jnp.int32))
    rooms_seen = set()
    started = time.time()
    for chunk in range(CHUNKS):
        keys = jax.random.split(jax.random.fold_in(root, 100 + chunk), (CHUNK, 2))
        carry, (xs, rooms) = run(carry, (keys[:, 0], keys[:, 1]))
        state, lost, ended = carry
        jax.block_until_ready(lost)
        xs, rooms = np.asarray(xs), np.asarray(rooms)
        rooms_seen |= set(np.unique(rooms).tolist())
        x = np.asarray(state.player_x)
        motionless = int(np.sum(xs.max(0) == xs.min(0)))
        print(f"after {(chunk + 1) * CHUNK} actions: most common x {np.bincount(x).argmax()} "
              f"({np.bincount(x).max()} of {GAMES} games); games with no horizontal movement in the last "
              f"{CHUNK} actions {motionless}; life losses so far total {int(lost.sum())}, games with one or "
              f"more {int((lost > 0).sum())}; game ends so far {int(ended.sum())}; rooms visited so far "
              f"{sorted(rooms_seen)} [{time.time() - started:.0f} s]", flush=True)
