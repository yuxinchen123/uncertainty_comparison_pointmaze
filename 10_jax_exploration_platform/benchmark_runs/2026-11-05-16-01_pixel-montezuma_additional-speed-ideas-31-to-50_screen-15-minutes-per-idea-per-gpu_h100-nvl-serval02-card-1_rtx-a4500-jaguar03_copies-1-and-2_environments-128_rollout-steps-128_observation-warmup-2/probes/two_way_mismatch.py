"""Which two-way conversion (conditional or switch) changes pixels, and on which states?"""
import os
os.environ.setdefault("JAXTARI_SKIP_SPRITE_UPDATE", "1")
import jax, jax.numpy as jnp, numpy as np
import jaxtari
from exploration_platform.atari_pixels import environment as E

class OnlySwitch(E._MaskedDrawLax):
    def cond(self, predicate, true_fun, false_fun, *operands, **kwargs):
        if len(operands) == 1 and not kwargs and false_fun(operands[0]) is operands[0]:
            return self._masked(predicate, true_fun, operands[0])
        return jax.lax.cond(predicate, true_fun, false_fun, *operands, **kwargs)

class OnlyCond(E._MaskedDrawLax):
    def switch(self, index, branches, *operands, **kwargs):
        return jax.lax.switch(index, branches, *operands, **kwargs)

with jax.enable_x64(False):
    original = jaxtari.make("montezumarevenge")
    _, template = original.reset(jax.random.PRNGKey(99))
    states = [("room %d" % room, template.replace(room_id=jnp.asarray(room, jnp.int32))) for room in range(original.consts.MAX_ROOMS)]
    for kind in range(5):
        states.append(("items type %d" % kind, template.replace(items_active=jnp.ones_like(template.items_active), items_type=jnp.full_like(template.items_type, kind), enemies_active=jnp.ones_like(template.enemies_active), enemies_type=jnp.full_like(template.enemies_type, kind % 3))))
    reference = jax.jit(original.render)
    for label, cls in (("two-way conditionals only", OnlyCond), ("switches only", OnlySwitch)):
        game = jaxtari.make("montezumarevenge")
        saved = E._MaskedDrawLax
        E._MaskedDrawLax = cls
        E._install_masked_conditional_draws(game, unroll_loops=True, two_way=True)
        E._MaskedDrawLax = saved
        render = jax.jit(game.render)
        bad = [name for name, state in states if not np.array_equal(np.asarray(render(state)), np.asarray(reference(state)))]
        print(label, "mismatching states:", bad, flush=True)
