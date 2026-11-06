"""Unrolled fixed-count render loops give exactly the original pixels. Run on a compute node.

States come from random play and from every room (the room number is set directly, so the
platforms, conveyors, lasers, items, doors and enemies of all rooms are drawn).
"""

import os

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp
import numpy as np

from exploration_platform.atari_pixels.environment import _install_unrolled_render_loops


def test_unrolled_render_matches_original():
    """Same state, two renderers: the original and the unrolled one."""
    import jaxtari

    with jax.enable_x64(False):
        original, unrolled = jaxtari.make("montezumarevenge"), jaxtari.make("montezumarevenge")
        _install_unrolled_render_loops(unrolled)
        step = jax.jit(original.step)
        render_original, render_unrolled = jax.jit(original.render), jax.jit(unrolled.render)
        states = []
        for seed in range(4):
            key = jax.random.PRNGKey(seed)
            _, state = original.reset(key)
            for index in range(300):
                action = jax.random.randint(jax.random.fold_in(key, index), (), 0, 18)
                _, state, _, _, _ = step(state, action)
                if index % 30 == 0:
                    states.append(state)
        _, template = original.reset(jax.random.PRNGKey(99))
        for room in range(original.consts.MAX_ROOMS):
            states.append(template.replace(room_id=jnp.asarray(room, jnp.int32)))
        for state in states:
            np.testing.assert_array_equal(np.asarray(render_unrolled(state)),
                                          np.asarray(render_original(state)))
        assert len(states) >= 40
