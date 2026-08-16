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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                       / "src" / "exploration_platform" / "agents" / "ppo"))
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


def build_unroll(update_unroll):
    """A trainer and primed state differing only in the update scan's unroll factor.

    The reference for what harmless reassociation costs: unrolling instructs the compiler how
    many loop bodies to emit, and unroll 1 against 2 agrees to 3.7e-16 in double precision.
    """
    t = JaxPPORND(PPOConfig(update_unroll=update_unroll, **CFG))
    return t, t.prime_obs_rms(t.init_state(), jax.random.PRNGKey(5))


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


# Fields the two forms must reproduce EXACTLY: the hoist moves the critic and bonus passes out
# of the rollout loop, and touches nothing that produces these.
EXACT_FIELDS = ["obs", "actions", "old_logprob", "rnd_input"]
# The advantage scan sums about 1/(1 - gamma*lambda) ~ 20 terms, so any float32 difference in the
# stored values reappears in adv and the returns multiplied by up to that. Judging those by the
# same bound as the values themselves demands the values agree to 5e-7, which is below float32
# accumulation noise for these matmul shapes — the gate would be failing arithmetic, not code.
VALUE_TOL = 1e-5
SCAN_AMPLIFICATION = 20


def test_isolation():
    """One iteration from identical state: exact where the hoist cannot matter, bounded elsewhere."""
    ta, sa = build(False)
    tb, sb = build(True)
    key = jax.random.PRNGKey(21)
    ba, _ = batch_of(ta, sa, key)
    bb, _ = batch_of(tb, sb, key)
    for k in EXACT_FIELDS:
        d = rel(ba[k], bb[k])
        print(f"isolation: {k:12} {d:.3e} (must be exactly zero)")
        assert d == 0.0, f"the hoist changed {k}, which it cannot touch"
    derived = max((rel(ba[k], bb[k]), k) for k in FIELDS if k not in EXACT_FIELDS)
    limit = VALUE_TOL * SCAN_AMPLIFICATION
    print(f"isolation: worst derived field {derived[0]:.3e} ({derived[1]}), limit {limit:.0e}")
    assert derived[0] <= limit, "the hoist changed the computation beyond float32 reassociation"
    print("ok test_isolation")


def chained_drift(ta, sa, tb, sb, iterations=3):
    """Worst relative parameter deviation after running the same iterations through both."""
    for it in range(iterations):
        key = jax.random.PRNGKey(100 + it)
        sa, _ = ta._iterate(sa, key, 3e-4)
        sb, _ = tb._iterate(sb, key, 3e-4)
    return max(rel(x, y) for x, y in
               zip(jax.tree.leaves(sa.params), jax.tree.leaves(sb.params)))


def test_drift_against_a_control():
    """Chained drift, judged against a change that provably computes the same function.

    Training is a feedback loop, so ANY difference in float32 rounding grows from iteration to
    iteration and a fixed bound says nothing on its own. Measured: a change already established
    as the same function — the update scan at unroll 1 against unroll 2, which agree to 3.7e-16
    in double precision — drifts 1.7e-03 over three iterations, seventeen times the 1e-4 bound
    this test used to assert. So the control is measured here in the same run and the hoist is
    required only to stay in its neighbourhood.
    """
    control = chained_drift(*build_unroll(1), *build_unroll(2))
    hoisted = chained_drift(*build(False), *build(True))
    print(f"drift after 3 iterations: hoist {hoisted:.3e}, "
          f"same-function control {control:.3e} ({hoisted/control:.2f}x)")
    assert hoisted <= 5 * control, "the hoist drifts far beyond float32 reassociation"
    print("ok test_drift_against_a_control")


if __name__ == "__main__":
    test_isolation()
    test_drift_against_a_control()
