"""Holding every parameter in one array must not change what the trainer computes.

Round five replaced the twenty-one named parameter tensors with one array of shape
[copies, parameters per copy], cut back into the named tensors before every forward pass. The
claim is that this is exact up to floating-point reordering: the networks see the same numbers,
and only two reductions change shape — the gradient-norm clip, which now sums one long row
instead of adding up twenty-one per-tensor sums, and Adam, which now runs on one array instead
of twenty-one.

Two checks, matching the round-two hoist test next to this file:
  1. Isolation — both forms start from a byte-identical state and run ONE iteration, so every
     difference comes from that iteration alone. Gated relative to each parameter's own
     magnitude, because the clip's sum accumulates in a different order.
  2. Drift — three chained iterations, reported not gated: a last-bit difference changes an
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


def test_isolation():
    """One iteration from identical state: the two forms must agree relatively."""
    ta, sa = build(False)
    tb, sb = build(True)
    key = jax.random.PRNGKey(21)
    sa2, _ = ta._iterate(sa, key, 3e-4)
    sb2, _ = tb._iterate(sb, key, 3e-4)
    worst = max(rel(x, y) for x, y in zip(named(ta, sa2.params), named(tb, sb2.params)))
    print(f"isolation: worst relative parameter deviation after one iteration {worst:.3e}")
    assert worst <= 1e-5, "the one-array form changed the computation"
    print("ok test_isolation")


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
    assert worst_rel <= 1e-4, "parameters drifted more than accumulation explains"
    print("ok test_drift_report")


if __name__ == "__main__":
    test_start_identical()
    test_isolation()
    test_drift_report()
