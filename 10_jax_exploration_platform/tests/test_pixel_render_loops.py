"""Unrolled fixed-count render loops give exactly the original pixels. Run on a compute node.

States come from random play and from every room (the room number is set directly, so the
platforms, conveyors, lasers, items, doors and enemies of all rooms are drawn).
"""

import os

os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")

import jax
import jax.numpy as jnp
import numpy as np

import pytest

from exploration_platform.atari_pixels.environment import (_install_masked_conditional_draws,
                                                           _install_unrolled_render_loops)


INSTALLERS = {
    "unrolled loops": _install_unrolled_render_loops,
    "masked conditional draws": _install_masked_conditional_draws,
    "masked conditional draws and unrolled loops":
        lambda game: _install_masked_conditional_draws(game, unroll_loops=True),
    "two-way masked draws and unrolled loops":
        lambda game: _install_masked_conditional_draws(game, unroll_loops=True, two_way=True),
}


@pytest.mark.parametrize("variant", sorted(INSTALLERS))
def test_changed_render_matches_original(variant):
    """Same state, two renderers: the original and the changed one."""
    import jaxtari

    with jax.enable_x64(False):
        original, unrolled = jaxtari.make("montezumarevenge"), jaxtari.make("montezumarevenge")
        INSTALLERS[variant](unrolled)
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
            # Torch, amulet, keys and swords in the inventory; lasers on at several cycle phases.
            carried = template.inventory.at[0].set(2).at[1].set(1).at[2].set(1).at[3].set(1)
            for cycle in (0, 3, 50, 95):
                states.append(template.replace(
                    room_id=jnp.asarray(room, jnp.int32), inventory=carried,
                    lasers_active=jnp.ones_like(template.lasers_active),
                    laser_cycle=jnp.asarray(cycle, template.laser_cycle.dtype)))
        for state in states:
            np.testing.assert_array_equal(np.asarray(render_unrolled(state)),
                                          np.asarray(render_original(state)))
        # Every item type in every item slot, and every enemy type, all active.
        for kind in range(5):
            states.append(template.replace(
                items_active=jnp.ones_like(template.items_active),
                items_type=jnp.full_like(template.items_type, kind),
                enemies_active=jnp.ones_like(template.enemies_active),
                enemies_type=jnp.full_like(template.enemies_type, kind % 3)))
        assert len(states) >= 175
        # Training renders all environments at once under vmap, where conditionals become selects.
        batch = jax.tree.map(lambda *values: jnp.stack(values), *states)
        np.testing.assert_array_equal(np.asarray(jax.jit(jax.vmap(unrolled.render))(batch)),
                                      np.asarray(jax.jit(jax.vmap(original.render))(batch)))
