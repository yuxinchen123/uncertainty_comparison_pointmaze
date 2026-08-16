"""Where the one-array parameter form differs from the twenty-one-tensor form.

The equivalence test shows a relative parameter deviation of about 3e-5 after one iteration,
above the gate the earlier rollout change had to meet. This asks whether the two forms compute a
different function or the same one with different rounding, by comparing quantities upstream of
the optimizer, where nothing has had a chance to amplify yet:

  loss           one loss value on identical inputs
  gradient       the raw gradient of that loss, before the clip
  clip norm      the per-copy gradient norm, the one reduction whose order really changed
  after Adam     the parameters after a single optimizer step on that gradient

A gradient that agrees to a few parts in ten million while the parameters after Adam disagree by
more says the function is the same and the optimizer amplified last-bit differences, which is
what happens when the second moment is small: the update is a ratio, so a tiny change in a small
denominator moves it much further than it moves the gradient.

Run: PYTHONNOUSERSITE=1 <jax python> diagnose_flat_params_deviation.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
sys.path.insert(0, str(BASE / "benchmarks"))
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402
from profile_jax_phases import make_batch  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1)


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return float(np.abs(a - b).max() / scale)


def main():
    ta = JaxPPORND(PPOConfig(flat_params=False, **CFG))
    tb = JaxPPORND(PPOConfig(flat_params=True, **CFG))
    pa = ta.init_params
    pb = tb.pack(ta.init_params)
    batch = make_batch(ta.cfg, jax.random.PRNGKey(3))

    # one loss and one gradient on identical inputs, with no optimizer involved
    la, ga = jax.jit(jax.value_and_grad(lambda p: ta._losses(p, batch, style_a=False)))(pa)
    lb, gb = jax.jit(jax.value_and_grad(lambda p: tb._losses(p, batch, style_a=False)))(pb)
    ga_leaves = jax.tree.leaves(ga)
    gb_leaves = jax.tree.leaves(tb.unpack(gb))
    print(f"loss                     {float(la):.10f} vs {float(lb):.10f}  "
          f"relative {rel(la, lb):.3e}")
    print(f"gradient, worst tensor   relative {max(rel(x, y) for x, y in zip(ga_leaves, gb_leaves)):.3e}")

    # the clip's per-copy norm: one long reduction against twenty-one summed partial reductions
    C = ta.cfg.n_copies
    na = np.sqrt(sum(np.asarray(leaf).reshape(C, -1).astype(np.float32).__pow__(2).sum(axis=1)
                     for leaf in ga_leaves))
    nb = np.sqrt(np.asarray(gb).reshape(C, -1).astype(np.float32).__pow__(2).sum(axis=1))
    print(f"clip norm per copy       relative {rel(na, nb):.3e}   "
          f"values {np.asarray(na)} vs {np.asarray(nb)}")

    # one Adam step on that gradient, so the amplification is measured on its own
    lr = jnp.asarray(3e-4, jnp.float32)
    za = jax.tree.map(jnp.zeros_like, pa)
    zb = jax.tree.map(jnp.zeros_like, pb)
    t0 = jnp.zeros((), jnp.int32)
    ua, _, _, _ = ta._adam_step(pa, ta._clip_per_copy(ga), za, za, t0, lr)
    ub, _, _, _ = tb._adam_step(pb, tb._clip_per_copy(gb), zb, zb, t0, lr)
    ua_leaves, ub_leaves = jax.tree.leaves(ua), jax.tree.leaves(tb.unpack(ub))
    worst = max((rel(x, y), i) for i, (x, y) in enumerate(zip(ua_leaves, ub_leaves)))
    print(f"after one Adam step      relative {worst[0]:.3e}  (tensor index {worst[1]})")

    # the step each form actually took, so the difference can be read against the step size
    da = np.asarray(ua_leaves[worst[1]]) - np.asarray(jax.tree.leaves(pa)[worst[1]])
    print(f"  that tensor's step size  max |change| {np.abs(da).max():.3e}, "
          f"difference between forms {np.abs(np.asarray(ua_leaves[worst[1]]) - np.asarray(ub_leaves[worst[1]])).max():.3e}")


def double_precision_check():
    """The same loss and gradient in double precision, which settles same-function-or-not.

    If the two forms compute the same function, the only thing separating them in single
    precision is rounding, and in double precision they must agree to about fifteen digits.
    If instead they compute different things, the disagreement stays about the same size.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    ta = JaxPPORND(PPOConfig(flat_params=False, **CFG))
    tb = JaxPPORND(PPOConfig(flat_params=True, **CFG))
    to64 = lambda x: np.asarray(x).astype(np.float64)
    pa = jax.tree.map(lambda x: jnp.asarray(to64(x)), ta.init_params)
    pb = jnp.concatenate([jnp.asarray(to64(l)).reshape(l.shape[0], -1)
                          for l in jax.tree.leaves(ta.init_params)], axis=1)
    batch = {k: jnp.asarray(to64(v))
             for k, v in make_batch(ta.cfg, jax.random.PRNGKey(3)).items()}

    la, ga = jax.jit(jax.value_and_grad(lambda p: ta._losses(p, batch, style_a=False)))(pa)
    lb, gb = jax.jit(jax.value_and_grad(lambda p: tb._losses(p, batch, style_a=False)))(pb)
    ga_leaves = jax.tree.leaves(ga)
    gb_leaves = jax.tree.leaves(tb.unpack(gb))
    worst = max(rel(x, y) for x, y in zip(ga_leaves, gb_leaves))
    print(f"\ndouble precision:")
    print(f"  loss                   relative {rel(la, lb):.3e}")
    print(f"  gradient, worst tensor relative {worst:.3e}")
    print("  verdict:", "same function, single-precision rounding only"
          if worst < 1e-12 else "the two forms compute different things")


if __name__ == "__main__":
    main()
    double_precision_check()
