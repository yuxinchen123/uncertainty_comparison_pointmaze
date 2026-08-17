"""The optimizer: a per-copy gradient-norm clip and a hand-rolled Adam, both elementwise.

Every parameter carries a leading copy axis, so one Adam here is C independent Adams and one
clip is C independent clips. The two update styles are built as two separate functions and the
style is chosen before the program is compiled — there is no traced branch on it anywhere.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ... import F32


class OptState(NamedTuple):
    """Adam's state: first moments, second moments, and the step counter."""
    m: dict              # same tree as the parameters
    v: dict              # same tree as the parameters
    t: jnp.ndarray       # int32 scalar


def opt_init(params) -> OptState:
    """Zero moments shaped like the parameters, step counter at zero."""
    return OptState(jax.tree.map(jnp.zeros_like, params),
                    jax.tree.map(jnp.zeros_like, params),
                    jnp.zeros((), jnp.int32))


def clip_per_copy(grads, n_copies: int, max_grad_norm: float):
    """Per-copy gradient-norm clip: the norm runs over each copy's own slice of every tensor."""
    # before: grads is a tree of [C, ...] arrays; after: each is scaled by that copy's own factor
    leaves = jax.tree.leaves(grads)
    g2 = sum((leaf.reshape(n_copies, -1).astype(F32) ** 2).sum(axis=1) for leaf in leaves)
    scale = jnp.minimum(1.0, max_grad_norm / (jnp.sqrt(g2) + 1e-6))
    return jax.tree.map(lambda g: g * scale.reshape((n_copies,) + (1,) * (g.ndim - 1)), grads)


def adam_step(params, grads, opt: OptState, lr, adam_eps: float, n_copies: int, lr_per_copy):
    """One Adam step (torch.optim.Adam formula): moments, bias correction, parameter update.

    Without a sweep, `lr` is the rate for every copy and `lr_per_copy` is None, so the arithmetic
    is a scalar times a tree. With a sweep, `lr` is the annealing multiplier and `lr_per_copy` is
    a [C] device constant broadcast along the copy axis — a vector reshaped to [C, 1, ...]
    against each parameter, which the compiler folds into the same elementwise update.
    """
    b1, b2, eps = 0.9, 0.999, adam_eps
    t = opt.t + 1
    tf_ = t.astype(F32)
    bc1 = 1.0 - b1 ** tf_
    bc2 = 1.0 - b2 ** tf_
    m = jax.tree.map(lambda mm, g: b1 * mm + (1 - b1) * g, opt.m, grads)
    v = jax.tree.map(lambda vv, g: b2 * vv + (1 - b2) * g * g, opt.v, grads)
    if lr_per_copy is None:
        step = lambda p, mm, vv: p - (lr / bc1) * mm / (jnp.sqrt(vv / bc2) + eps)
    else:
        def step(p, mm, vv):
            # before: lr_per_copy [C]; after: [C, 1, ...] matching this parameter's rank
            rate = lr_per_copy.reshape((n_copies,) + (1,) * (p.ndim - 1)) * lr
            return p - (rate / bc1) * mm / (jnp.sqrt(vv / bc2) + eps)
    return jax.tree.map(step, params, m, v), OptState(m, v, t)


def build_update(cfg, loss_fn, lr_per_copy):
    """Return the update function for this configuration's style, chosen here and not at run time.

    loss_fn(params, batch) -> scalar loss summed over copies. The returned function has the
    signature update(params, opt, batch, lr, key) -> (params, opt, reported loss).
    """
    if cfg.update_style == "full_batch":
        return _build_full_batch(cfg, loss_fn, lr_per_copy)
    if cfg.update_style == "epoch_minibatch":
        return _build_epoch_minibatch(cfg, loss_fn, lr_per_copy)
    raise ValueError(f"unknown update style {cfg.update_style!r}; "
                     "use 'full_batch' or 'epoch_minibatch'")


def _build_full_batch(cfg, loss_fn, lr_per_copy):
    """Style A: one gradient step on all T*N rows per copy (spec 11 + correction)."""
    def update(params, opt, batch, lr, key):
        """One gradient, one clip, one Adam step."""
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, style_a=True))(params)
        grads = clip_per_copy(grads, cfg.n_copies, cfg.max_grad_norm)
        params, opt = adam_step(params, grads, opt, lr, cfg.adam_eps, cfg.n_copies, lr_per_copy)
        return params, opt, loss
    return update


def _build_epoch_minibatch(cfg, loss_fn, lr_per_copy):
    """Style B: epochs x minibatches of shuffled steps, run as one scan over the 16 steps."""
    C = cfg.n_copies
    Brows = cfg.num_steps * cfg.n_envs
    mb_size = Brows // cfg.num_minibatches
    n_steps = cfg.update_epochs * cfg.num_minibatches

    def update(params, opt, batch, lr, key):
        """Draw all permutations up front, gather every minibatch, then scan the steps."""
        # permutations, independent per copy AND per epoch
        # before: uniform draws (E, C, B); after: minibatch row indices (E*K, C, mb)
        perm = jnp.argsort(jax.random.uniform(key, (cfg.update_epochs, C, Brows), F32), axis=-1)
        idx = perm.reshape(cfg.update_epochs, C, cfg.num_minibatches, mb_size)
        idx = idx.transpose(0, 2, 1, 3).reshape(n_steps, C, mb_size)

        # gather all minibatches up front: field (C,B,k) -> (S,C,mb,k); (C,B) -> (S,C,mb)
        def gather(t):
            """One batch field, cut into the S minibatches the scan will walk."""
            if t.ndim == 3:
                return jnp.take_along_axis(t[None], idx[..., None], axis=2)
            return jnp.take_along_axis(t[None], idx, axis=2)

        mbs = {k: gather(batch[k]) for k in sorted(batch)}

        def body(carry, mb):
            """One of the sixteen gradient steps."""
            params, opt = carry
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, mb, style_a=False))(params)
            grads = clip_per_copy(grads, C, cfg.max_grad_norm)
            params, opt = adam_step(params, grads, opt, lr, cfg.adam_eps, C, lr_per_copy)
            return (params, opt), loss

        (params, opt), losses = jax.lax.scan(body, (params, opt), mbs,
                                             unroll=cfg.update_unroll)
        return params, opt, losses[-1]
    return update
