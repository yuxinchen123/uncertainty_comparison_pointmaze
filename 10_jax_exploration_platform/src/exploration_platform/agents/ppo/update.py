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


def per_copy_norm(tree, n_copies: int):
    """The gradient norm of each copy over every tensor of a tree, [C]; zeros for an empty tree."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.zeros((n_copies,), F32)
    return jnp.sqrt(sum((leaf.reshape(n_copies, -1).astype(F32) ** 2).sum(axis=1)
                        for leaf in leaves))


def clip_and_report(cfg, grads, n_copies: int):
    """Clip the gradient as the configuration says, and report each network's norm before it.

    The default clips the whole tree jointly, exactly as `clip_per_copy` always has. With
    clip_per_network the actor, the critic and the bonus's networks are clipped separately, so a
    large critic gradient no longer shrinks the actor's step.

    before: grads {"agent": {"actor", "critic"}, "bonus": {"predictor"} or {}}
    after:  the same tree clipped, and {"grad_norm_actor": [C], "grad_norm_critic": [C],
            "grad_norm_bonus": [C], "clip_scale": [C]} where clip_scale is the smallest factor
            any clip applied to that copy (1 when nothing was clipped)
    """
    # each network's norm before clipping, per copy
    parts = {"actor": grads["agent"]["actor"], "critic": grads["agent"]["critic"],
             "bonus": grads["bonus"]}
    norms = {name: per_copy_norm(tree, n_copies) for name, tree in parts.items()}
    report = {f"grad_norm_{name}": norm for name, norm in norms.items()}

    # the joint clip (default) or one clip per network
    if not cfg.clip_per_network:
        total = jnp.sqrt(sum(norm ** 2 for norm in norms.values()))
        report["clip_scale"] = jnp.minimum(1.0, cfg.max_grad_norm / (total + 1e-6))
        return clip_per_copy(grads, n_copies, cfg.max_grad_norm), report
    scales = [jnp.minimum(1.0, cfg.max_grad_norm / (norm + 1e-6)) for norm in norms.values()]
    report["clip_scale"] = jnp.minimum(jnp.minimum(scales[0], scales[1]), scales[2])
    clipped = {"agent": {"actor": clip_per_copy(parts["actor"], n_copies, cfg.max_grad_norm),
                         "critic": clip_per_copy(parts["critic"], n_copies, cfg.max_grad_norm)},
               "bonus": (clip_per_copy(parts["bonus"], n_copies, cfg.max_grad_norm)
                         if jax.tree.leaves(parts["bonus"]) else parts["bonus"])}
    return clipped, report


def adam_step(params, grads, opt: OptState, lr, adam_eps: float, n_copies: int, lr_per_copy,
              bonus_rate=None):
    """One Adam step (torch.optim.Adam formula): moments, bias correction, parameter update.

    Without a sweep, `lr` is the rate for every copy and `lr_per_copy` is None, so the arithmetic
    is a scalar times a tree. With a sweep, `lr` is the annealing multiplier and `lr_per_copy` is
    a [C] device constant broadcast along the copy axis — a vector reshaped to [C, 1, ...]
    against each parameter, which the compiler folds into the same elementwise update.

    bonus_rate, when given, is the constant rate of the bonus's parameters (the tree under
    "bonus"), in place of the agent's rate: the bonus then has its own Adam. Its moments sit in
    the same two trees and it shares the step counter, because both optimizers take exactly one
    step per gradient step.
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
    if bonus_rate is None:
        return jax.tree.map(step, params, m, v), OptState(m, v, t)
    # the agent at its own (annealed, possibly per-copy) rate, the bonus at its constant rate
    bonus_step = lambda p, mm, vv: p - (bonus_rate / bc1) * mm / (jnp.sqrt(vv / bc2) + eps)
    stepped = {"agent": jax.tree.map(step, params["agent"], m["agent"], v["agent"]),
               "bonus": jax.tree.map(bonus_step, params["bonus"], m["bonus"], v["bonus"])}
    return stepped, OptState(m, v, t)


def clip_and_step(cfg, params, grads, opt: OptState, lr, lr_per_copy, bonus_rate):
    """Clip one gradient and take one Adam step with it.

    bonus_rate None: the joint clip over the agent and the bonus and one rate, as always.
    bonus_rate given: the clip covers the agent's gradient only, and the bonus's parameters are
    stepped with their unclipped gradient at that constant rate.
    """
    if bonus_rate is None:
        grads = clip_per_copy(grads, cfg.n_copies, cfg.max_grad_norm)
        return adam_step(params, grads, opt, lr, cfg.adam_eps, cfg.n_copies, lr_per_copy)
    grads = {"agent": clip_per_copy(grads["agent"], cfg.n_copies, cfg.max_grad_norm),
             "bonus": grads["bonus"]}
    return adam_step(params, grads, opt, lr, cfg.adam_eps, cfg.n_copies, lr_per_copy, bonus_rate)


def build_update(cfg, loss_fn, lr_per_copy, bonus_rate=None):
    """Return the update function for this configuration's style, chosen here and not at run time.

    loss_fn(params, batch) -> scalar loss summed over copies. The returned function has the
    signature update(params, opt, batch, lr, key) -> (params, opt, reported loss). bonus_rate is
    the constant learning rate of a bonus that has its own Adam, or None (see `clip_and_step`).
    """
    reporting = (cfg.diagnostics or cfg.clip_per_network or cfg.critic_lr_scale != 1.0
                 or cfg.logstd_floor > float("-inf"))
    if bonus_rate is not None and (cfg.flat_params or reporting):
        raise ValueError("a bonus with its own learning rate needs the named parameter tree and "
                         "the plain update: turn off flat_params, diagnostics, clip_per_network, "
                         "critic_lr_scale and logstd_floor")
    if cfg.flat_params and reporting:
        raise ValueError("diagnostics and clip_per_network need the named parameter tree; "
                         "turn flat_params off")
    if cfg.gather_minibatches_per_epoch and cfg.update_style != "epoch_minibatch":
        raise ValueError("gather_minibatches_per_epoch applies to epoch_minibatch updates only; "
                         f"this configuration's update style is {cfg.update_style!r}")
    if reporting:
        return _build_reporting(cfg, loss_fn, lr_per_copy)
    if cfg.update_style == "full_batch":
        return _build_full_batch(cfg, loss_fn, lr_per_copy, bonus_rate)
    if cfg.update_style == "epoch_minibatch":
        return _build_epoch_minibatch(cfg, loss_fn, lr_per_copy, bonus_rate)
    raise ValueError(f"unknown update style {cfg.update_style!r}; "
                     "use 'full_batch' or 'epoch_minibatch'")


def _build_full_batch(cfg, loss_fn, lr_per_copy, bonus_rate=None):
    """Style A: one gradient step on all T*N rows per copy (spec 11 + correction)."""
    def update(params, opt, batch, lr, key):
        """One gradient, one clip, one Adam step."""
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, style_a=True))(params)
        params, opt = clip_and_step(cfg, params, grads, opt, lr, lr_per_copy, bonus_rate)
        return params, opt, loss
    return update


def draw_permutations(cfg, key, n_rows: int):
    """A shuffled order of a copy's rows for every epoch and copy, [epochs, C, n_rows] integers.

    "argsort_uniform" draws one uniform number per row and takes the positions that sort them,
    one sort per epoch and copy. "sort_packed_keys" gives every row a 64-bit key holding its
    (epoch, copy) group in the high bits, 32 random bits in the middle and its own row number in
    the low bits; ONE sort of all keys leaves each group's rows together, in the order of their
    random bits, and the low bits of the sorted keys are the shuffled row numbers.

    The two schemes read the same 32 random bits per row from the key. A uniform float32 number
    keeps the leading 23 of them, and rows with equal numbers stay in row order (about
    n_rows^2 / 2^24 pairs per group: 16 pairs at 16,384 rows). The packed keys sort on all 32
    bits, so only those tied rows can come out in another order (still tied on 32 bits: about
    n_rows^2 / 2^33 pairs per group). Where a group has no tie the two orders are the same.

    before (sort_packed_keys): 2 groups of 4 rows, random bits [[7, 2, 9, 5], [1, 8, 3, 6]]
    after:  keys sorted -> group 0 rows [1, 3, 0, 2], group 1 rows [0, 2, 3, 1]
    """
    groups = cfg.update_epochs * cfg.n_copies
    if cfg.permutation_draw == "argsort_uniform":
        return jnp.argsort(jax.random.uniform(key, (cfg.update_epochs, cfg.n_copies, n_rows), F32),
                           axis=-1)
    if cfg.permutation_draw != "sort_packed_keys":
        raise ValueError(f"unknown permutation_draw {cfg.permutation_draw!r}; use "
                         "'argsort_uniform' or 'sort_packed_keys'")
    row_bits = max(1, (n_rows - 1).bit_length())
    group_bits = max(1, (groups - 1).bit_length())
    if group_bits + 32 + row_bits > 64:
        raise ValueError(f"sort_packed_keys holds {groups} (epoch, copy) groups, 32 random bits "
                         f"and {n_rows} rows in one 64-bit key, which needs "
                         f"{group_bits + 32 + row_bits} bits")
    draws = jax.random.bits(key, (groups, n_rows), jnp.uint32).astype(jnp.uint64)
    group = jnp.arange(groups, dtype=jnp.uint64)[:, None]
    row = jnp.arange(n_rows, dtype=jnp.uint64)[None, :]
    packed = (group << (32 + row_bits)) | (draws << row_bits) | row
    ordered = jnp.sort(packed.reshape(-1))
    return (ordered & jnp.uint64((1 << row_bits) - 1)).astype(jnp.int32).reshape(
        cfg.update_epochs, cfg.n_copies, n_rows)


def scan_minibatches(cfg, batch, key, carry, body):
    """Shuffle each copy's rows once per epoch and run `body` over every minibatch, in order.

    body(carry, minibatch) -> (carry, output). Returns (carry, the outputs stacked over the
    epochs x minibatches gradient steps). By default every minibatch of every epoch is gathered
    before the first step, which holds one shuffled copy of the batch per epoch. With
    gather_minibatches_per_epoch the same rows are gathered one epoch at a time inside an outer
    scan over the epochs, so one shuffled copy is held; the draws and the step order are the same.
    """
    C = cfg.n_copies
    Brows = cfg.num_steps * cfg.n_envs
    mb_size = Brows // cfg.num_minibatches
    n_steps = cfg.update_epochs * cfg.num_minibatches

    # permutations, independent per copy AND per epoch
    # before: shuffled row orders (E, C, B); after: minibatch row indices (E, C, K, mb)
    perm = draw_permutations(cfg, key, Brows)
    idx = perm.reshape(cfg.update_epochs, C, cfg.num_minibatches, mb_size)

    def gather(t, rows):
        """One batch field cut into minibatches: (C,B,k) -> (S,C,mb,k); (C,B) -> (S,C,mb)."""
        if t.ndim == 3:
            return jnp.take_along_axis(t[None], rows[..., None], axis=2)
        return jnp.take_along_axis(t[None], rows, axis=2)

    if not cfg.gather_minibatches_per_epoch:
        # gather all minibatches up front, then one scan over the E*K steps
        idx = idx.transpose(0, 2, 1, 3).reshape(n_steps, C, mb_size)
        mbs = {k: gather(batch[k], idx) for k in sorted(batch)}
        return jax.lax.scan(body, carry, mbs, unroll=cfg.update_unroll)

    def epoch(carry, rows):
        """One epoch: gather its K minibatches, then scan them."""
        mbs = {k: gather(batch[k], rows) for k in sorted(batch)}
        return jax.lax.scan(body, carry, mbs, unroll=cfg.update_unroll)

    # before: idx (E, C, K, mb); after: (E, K, C, mb), one epoch's rows per outer step
    carry, outputs = jax.lax.scan(epoch, carry, idx.transpose(0, 2, 1, 3))
    # before: outputs (E, K, ...); after: (E*K, ...), the shape the single scan returns
    return carry, jax.tree.map(lambda x: x.reshape((n_steps,) + x.shape[2:]), outputs)


def _build_epoch_minibatch(cfg, loss_fn, lr_per_copy, bonus_rate=None):
    """Style B: epochs x minibatches of shuffled steps, run as one scan over the 16 steps."""
    def update(params, opt, batch, lr, key):
        """Draw all permutations up front, gather the minibatches, then scan the steps."""
        def body(carry, mb):
            """One of the sixteen gradient steps."""
            params, opt = carry
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, mb, style_a=False))(params)
            params, opt = clip_and_step(cfg, params, grads, opt, lr, lr_per_copy, bonus_rate)
            return (params, opt), loss

        (params, opt), losses = scan_minibatches(cfg, batch, key, (params, opt), body)
        return params, opt, losses[-1]
    return update


def after_step(cfg, before, after):
    """The study's adjustments to a finished Adam step: a shorter critic step, a log-std floor.

    before / after: the trainable tree before and after the step. With critic_lr_scale s the
    critic moves s times as far as Adam proposed: before 0.10, Adam 0.12, s = 0.5 -> 0.11.
    With a floor f the log standard deviation becomes max(logstd, f): -1.3 with f = -1 -> -1.
    """
    if cfg.critic_lr_scale != 1.0:
        s = cfg.critic_lr_scale
        critic = jax.tree.map(lambda b, a: b + s * (a - b),
                              before["agent"]["critic"], after["agent"]["critic"])
        after = {**after, "agent": {**after["agent"], "critic": critic}}
    if cfg.logstd_floor > float("-inf"):
        actor = {**after["agent"]["actor"],
                 "logstd": jnp.maximum(after["agent"]["actor"]["logstd"], cfg.logstd_floor)}
        after = {**after, "agent": {**after["agent"], "actor": actor}}
    return after


def _build_reporting(cfg, loss_fn, lr_per_copy):
    """Either update style, clipping through `clip_and_report`, adjusting through `after_step`,
    and returning the clip report as well.

    The signature becomes update(params, opt, batch, lr, key) -> (params, opt, loss, report);
    under epoch-minibatch updates each report field is the mean over the gradient steps. Kept
    apart from the two plain builders so a configuration that asks for neither switch compiles the
    program it always did.
    """
    C = cfg.n_copies
    if cfg.update_style == "full_batch":
        def update(params, opt, batch, lr, key):
            """One gradient, the configured clip, one Adam step, and the clip's report."""
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, style_a=True))(params)
            grads, report = clip_and_report(cfg, grads, C)
            stepped, opt = adam_step(params, grads, opt, lr, cfg.adam_eps, C, lr_per_copy)
            return after_step(cfg, params, stepped), opt, loss, report
        return update
    if cfg.update_style != "epoch_minibatch":
        raise ValueError(f"unknown update style {cfg.update_style!r}")

    def update(params, opt, batch, lr, key):
        """Style B as in `_build_epoch_minibatch`, with the clip report averaged over the steps."""
        def body(carry, mb):
            """One gradient step, with its clip report as a scan output."""
            params, opt = carry
            loss, grads = jax.value_and_grad(lambda p: loss_fn(p, mb, style_a=False))(params)
            grads, report = clip_and_report(cfg, grads, C)
            stepped, opt = adam_step(params, grads, opt, lr, cfg.adam_eps, C, lr_per_copy)
            return (after_step(cfg, params, stepped), opt), (loss, report)

        # the same permutations and minibatches as the plain style B builder; the reports come
        # back stacked [S, C], and the mean over the S steps is what is recorded
        (params, opt), (losses, reports) = scan_minibatches(cfg, batch, key, (params, opt), body)
        return params, opt, losses[-1], jax.tree.map(lambda x: x.mean(axis=0), reports)
    return update
