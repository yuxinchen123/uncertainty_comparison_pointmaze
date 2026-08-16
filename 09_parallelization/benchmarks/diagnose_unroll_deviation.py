"""Does the rollout scan's unroll factor change what the trainer computes?

Round five raised the unroll factor from 4. Unrolling is a compiler instruction about how many
copies of the loop body to emit, not a change to the arithmetic, so the values should be
untouched — but a longer body gives the compiler more to fuse, and fusing can change the order
in which a sum accumulates. This measures it directly, and also re-measures the round-two
rollout-hoist comparison at each unroll factor, because that gate is evaluated at whatever
unroll the trainer currently defaults to.

Run: PYTHONNOUSERSITE=1 <jax python> diagnose_unroll_deviation.py
"""
import sys
from pathlib import Path

import jax
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1)


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-12))


def build(unroll, hoist=True, update_unroll=1):
    """A trainer and its primed state at one rollout unroll factor and one update unroll."""
    t = JaxPPORND(PPOConfig(scan_unroll=unroll, hoist_rollout=hoist,
                            update_unroll=update_unroll, **CFG))
    return t, t.prime_obs_rms(t.init_state(), jax.random.PRNGKey(5))


def one_iteration(trainer, state):
    """The parameters after a single iteration from the given state."""
    st, _ = trainer._iterate(state, jax.random.PRNGKey(21), 3e-4)
    return jax.tree.leaves(st.params)


def main():
    print("changing only the unroll factor:")
    for lo, hi in [(4, 16), (4, 32), (16, 32)]:
        ta, sa = build(lo)
        tb, sb = build(hi)
        worst = max(rel(x, y) for x, y in zip(one_iteration(ta, sa), one_iteration(tb, sb)))
        print(f"  unroll {lo:>2} against {hi:>2}: worst relative parameter deviation {worst:.3e}")

    # the same check on the non-hoisted path, which computes the critic values and the
    # log-probability INSIDE the scan, so unrolling gives the compiler more to fuse there
    print("\nchanging only the unroll factor, with the rollout hoist turned off:")
    for lo, hi in [(4, 16), (4, 32)]:
        ta, sa = build(lo, hoist=False)
        tb, sb = build(hi, hoist=False)
        worst = max(rel(x, y) for x, y in zip(one_iteration(ta, sa), one_iteration(tb, sb)))
        print(f"  unroll {lo:>2} against {hi:>2}: worst relative parameter deviation {worst:.3e}")

    # the sixteen-step update scan, unrolled: the steps are sequentially dependent, so a
    # longer body should not let the compiler reorder any sum
    print("\nchanging only the UPDATE scan's unroll factor:")
    for lo, hi in [(1, 2), (1, 4)]:
        ta, sa = build(0, update_unroll=lo)
        tb, sb = build(0, update_unroll=hi)
        worst = max(rel(x, y) for x, y in zip(one_iteration(ta, sa), one_iteration(tb, sb)))
        print(f"  update unroll {lo} against {hi}: worst relative parameter deviation {worst:.3e}")

    print("\nthe round-two rollout-hoist comparison, measured at each unroll factor:")
    for u in (4, 16, 32):
        ta, sa = build(u, hoist=False)
        tb, sb = build(u, hoist=True)
        worst = max(rel(x, y) for x, y in zip(one_iteration(ta, sa), one_iteration(tb, sb)))
        print(f"  unroll {u:>2}: worst relative parameter deviation {worst:.3e}")


if __name__ == "__main__":
    main()
