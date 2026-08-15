"""The rollout hoist (round 2, J1) must not change what the trainer computes.

Round two moved the critic forward, the log-probability and the RND bonus out of the rollout
scan into single wide passes afterwards. The claim is that this is exact up to floating-point
reordering: each is a pure function of data the scan already stores and of parameters that do
not change during a rollout.

Two checks, mirroring the torch side:
  1. Isolation — both variants start from a byte-identical state and run ONE iteration, so
     every difference comes from that iteration alone. Gated, relative to each quantity's own
     magnitude, because a wider matmul accumulates in a different order.
  2. Drift — three chained iterations, reported not gated: a last-bit difference changes an
     action, which changes the next observation, so chained absolute differences say nothing
     about whether the two forms compute the same function.

Run: PYTHONNOUSERSITE=1 <jax python> test_hoist_equivalence.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, obs_norm_init_iters=1)
FIELDS = ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int", "vext_old", "rnd_input"]


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return float(np.abs(a - b).max() / scale)


def build(hoist):
    """A trainer and its primed state, identical apart from the hoist knob."""
    t = JaxPPORND(PPOConfig(hoist_rollout=hoist, **CFG))
    st = t.init_state()
    st = t.prime_obs_rms(st, jax.random.PRNGKey(5))
    return t, st


def batch_of(trainer, state, key):
    """The update batch one iteration produces, without applying the update."""
    cfg = trainer.cfg
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
    captured = {}

    def spy(st, batch, lr, k):
        # the batch is a traced value here; returning it from the jitted function below is
        # what makes it a concrete array the test can compare
        captured["batch"] = batch
        return st, jnp.zeros(())

    def run(s, k, lr):
        st, _ = trainer._iterate_impl(spy, s, k, lr)
        return st, captured["batch"]

    state2, batch = jax.jit(run)(state, key, 3e-4)
    return batch, state2


def test_isolation():
    """One iteration from identical state: the two forms must agree relatively."""
    ta, sa = build(False)
    tb, sb = build(True)
    key = jax.random.PRNGKey(21)
    ba, _ = batch_of(ta, sa, key)
    bb, _ = batch_of(tb, sb, key)
    worst = max((rel(ba[k], bb[k]), k) for k in FIELDS)
    print(f"isolation: worst relative field deviation {worst[0]:.3e} ({worst[1]})")
    assert worst[0] <= 1e-5, "the hoist changed the computation"
    print("ok test_isolation")


def test_drift_report():
    """Three chained iterations; reported so the accumulation is on the record."""
    ta, sa = build(False)
    tb, sb = build(True)
    for it in range(3):
        key = jax.random.PRNGKey(100 + it)
        sa, _ = ta._iterate(sa, key, 3e-4)
        sb, _ = tb._iterate(sb, key, 3e-4)
    leaves_a = jax.tree.leaves(sa.params)
    leaves_b = jax.tree.leaves(sb.params)
    worst = max(float(np.abs(np.asarray(x) - np.asarray(y)).max())
                for x, y in zip(leaves_a, leaves_b))
    worst_rel = max(rel(x, y) for x, y in zip(leaves_a, leaves_b))
    print(f"drift after 3 iterations: worst parameter deviation {worst:.3e} "
          f"({worst_rel:.3e} relative)")
    assert worst_rel <= 1e-4, "parameters drifted more than accumulation explains"
    print("ok test_drift_report")


if __name__ == "__main__":
    test_isolation()
    test_drift_report()
