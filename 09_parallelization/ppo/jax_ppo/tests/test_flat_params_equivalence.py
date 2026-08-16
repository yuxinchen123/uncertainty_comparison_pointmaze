"""Holding every parameter in one array must not change what the trainer computes.

Round five replaced the twenty-one named parameter tensors with one array of shape
[copies, parameters per copy], cut back into the named tensors before every forward pass. The
claim is that this is exact up to floating-point reordering: the networks see the same numbers,
and only two reductions change shape — the gradient-norm clip, which now sums one long row
instead of adding up twenty-one per-tensor sums, and Adam, which now runs on one array instead
of twenty-one.

Three checks, in the spirit of the round-two hoist test next to this file:
  1. Same function — the loss and the gradient computed in DOUBLE precision, which is the gate.
     Agreement there to about machine precision, with disagreement in single precision, is what
     distinguishes "same function, different rounding" from "different function".
  2. Isolation — one iteration in single precision, reported with both an absolute and a
     relative figure. The relative figure is dominated by tensors that start at zero, where the
     parameter's whole magnitude is the step just taken.
  3. Drift — three chained iterations, reported not gated: a last-bit difference changes an
     action, which changes the next observation, so chained differences say nothing about
     whether the two forms compute the same function.

Run: PYTHONNOUSERSITE=1 <jax python> test_flat_params_equivalence.py
"""
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1)


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return float(np.abs(a - b).max() / scale)


def build(flat):
    """A trainer and its primed state, identical apart from the parameter-layout knob."""
    t = JaxPPORND(PPOConfig(flat_params=flat, **CFG))
    st = t.init_state()
    st = t.prime_obs_rms(st, jax.random.PRNGKey(5))
    return t, st


def named(trainer, params):
    """The parameters as the 21 named tensors, whichever form they are stored in."""
    return jax.tree.leaves(trainer.unpack(params) if trainer.cfg.flat_params else params)


def test_start_identical():
    """Both forms must start from the same weights — otherwise the comparison means nothing."""
    ta, sa = build(False)
    tb, sb = build(True)
    worst = max(rel(x, y) for x, y in zip(named(ta, sa.params), named(tb, sb.params)))
    print(f"starting weights: worst relative deviation {worst:.3e}")
    assert worst == 0.0, "the two forms did not start from identical weights"
    print("ok test_start_identical")


def test_same_function_in_double_precision():
    """The gate: in double precision the two forms must agree to about machine precision.

    This is what decides whether the one-array form computes the same function. In single
    precision the backward pass accumulates in a different order, so the two forms differ in
    the last bits and the optimizer amplifies that (see the reported check below); in double
    precision, the same reordering is far below anything that matters, so agreement there and
    disagreement in single precision together mean "same function, different rounding".
    """
    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "benchmarks"))
    from profile_jax_phases import make_batch

    ta = JaxPPORND(PPOConfig(flat_params=False, **CFG))
    tb = JaxPPORND(PPOConfig(flat_params=True, **CFG))
    to64 = lambda x: jax.numpy.asarray(np.asarray(x).astype(np.float64))
    pa = jax.tree.map(to64, ta.init_params)
    pb = jax.numpy.concatenate(
        [to64(l).reshape(l.shape[0], -1) for l in jax.tree.leaves(ta.init_params)], axis=1)
    batch = {k: to64(v) for k, v in make_batch(ta.cfg, jax.random.PRNGKey(3)).items()}

    la, ga = jax.jit(jax.value_and_grad(lambda p: ta._losses(p, batch, style_a=False)))(pa)
    lb, gb = jax.jit(jax.value_and_grad(lambda p: tb._losses(p, batch, style_a=False)))(pb)
    worst = max(rel(x, y) for x, y in
                zip(jax.tree.leaves(ga), jax.tree.leaves(tb.unpack(gb))))
    print(f"double precision: loss relative {rel(la, lb):.3e}, "
          f"gradient worst relative {worst:.3e}")
    assert rel(la, lb) == 0.0, "the two forms do not even agree on the loss"
    assert worst <= 1e-12, "the one-array form computes a different function"
    print("ok test_same_function_in_double_precision")


def test_isolation_report():
    """One iteration in single precision; reported so the rounding difference is on the record.

    Not gated: the deviation is dominated by tensors that start at zero, where the parameter's
    whole magnitude IS the step just taken, so a last-bit difference in the step reads as a
    large relative difference in the parameter. The absolute difference is around 1e-7 on a
    step of 3e-4.
    """
    ta, sa = build(False)
    tb, sb = build(True)
    key = jax.random.PRNGKey(21)
    sa2, _ = ta._iterate(sa, key, 3e-4)
    sb2, _ = tb._iterate(sb, key, 3e-4)
    pa, pb = named(ta, sa2.params), named(tb, sb2.params)
    worst_rel = max(rel(x, y) for x, y in zip(pa, pb))
    worst_abs = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(pa, pb))
    print(f"single precision, one iteration: worst deviation {worst_abs:.3e} absolute, "
          f"{worst_rel:.3e} relative")
    assert worst_abs <= 1e-5, "the deviation is far larger than single-precision rounding"
    print("ok test_isolation_report")


def test_drift_report():
    """Three chained iterations; reported so the accumulation is on the record."""
    ta, sa = build(False)
    tb, sb = build(True)
    for it in range(3):
        key = jax.random.PRNGKey(100 + it)
        sa, _ = ta._iterate(sa, key, 3e-4)
        sb, _ = tb._iterate(sb, key, 3e-4)
    pa, pb = named(ta, sa.params), named(tb, sb.params)
    worst = max(float(np.abs(np.asarray(x) - np.asarray(y)).max()) for x, y in zip(pa, pb))
    worst_rel = max(rel(x, y) for x, y in zip(pa, pb))
    print(f"drift after 3 iterations: worst parameter deviation {worst:.3e} "
          f"({worst_rel:.3e} relative)")
    assert worst <= 1e-4, "parameters drifted more than accumulation explains"
    print("ok test_drift_report")


if __name__ == "__main__":
    test_start_identical()
    test_isolation_report()
    test_drift_report()
    test_same_function_in_double_precision()   # last: it switches on double precision
