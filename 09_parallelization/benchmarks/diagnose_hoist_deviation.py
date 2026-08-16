"""Every field's deviation between the hoisted and unhoisted rollout, not only the worst one.

The gate `test_hoist_equivalence.py` reports a single number and fails on it. To tell float32
reassociation from a genuinely changed computation, the useful question is WHICH fields move and
by how much relative to one another: a small deviation in the stored critic values that appears
much larger in the advantages is the signature of the advantage scan amplifying it, not of the
two forms computing different things.

Run (on the graphics-processor host, under the lock):
  bash locks/gpu_run.sh "PYTHONNOUSERSITE=1 <jax python> benchmarks/diagnose_hoist_deviation.py"
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo" / "tests"))

import jax  # noqa: E402

import test_hoist_equivalence as T  # noqa: E402


def deviations():
    """One row per compared field: deviation against the field's own scale, and in absolute terms.

    before: two trainers differing only in the hoist knob, run one iteration from identical state
    after:  {"adv": {"rel": 2.6e-05, "abs": 1.4e-04, "scale": 5.3}, ...}
    """
    # goal: the same construction the gate uses, so the numbers are comparable with its verdict
    unhoisted, state_u = T.build(False)
    hoisted, state_h = T.build(True)
    key = jax.random.PRNGKey(21)
    batch_u, _ = T.batch_of(unhoisted, state_u, key)
    batch_h, _ = T.batch_of(hoisted, state_h, key)

    # goal: for each field, the absolute movement and what it is relative to
    out = {}
    for name in T.FIELDS:
        a, b = np.asarray(batch_u[name]), np.asarray(batch_h[name])
        scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
        out[name] = {"rel": float(np.abs(a - b).max() / scale),
                     "abs": float(np.abs(a - b).max()), "scale": scale}
    return out


def main():
    """Print the per-field table the single-number gate hides."""
    rows = deviations()
    print(f"{'field':12} {'deviation / scale':>18} {'max absolute':>14} {'field scale':>14}")
    for name, r in rows.items():
        print(f"{name:12} {r['rel']:18.3e} {r['abs']:14.3e} {r['scale']:14.3e}")
    # goal: state the amplification the advantage scan applies, which is what the comparison is for
    gamma_lambda = 0.999 * 0.95
    print(f"\nadvantage scan sums about 1/(1 - gamma*lambda) = {1/(1-gamma_lambda):.0f} terms, "
          f"so a deviation in the stored values appears that many times larger in adv")


if __name__ == "__main__":
    main()
