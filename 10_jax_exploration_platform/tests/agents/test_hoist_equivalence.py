"""The rollout hoist (round 2, J1) must not change what the trainer computes.

Round two moved the critic forward, the log-probability and the intrinsic reward out of the
rollout scan into single wide passes afterwards. The claim is that this is exact up to
floating-point reordering: each is a pure function of data the scan already stores and of
parameters that do not change during a rollout.

Two checks, mirroring the torch side:
  1. Isolation — both variants start from a byte-identical state and run ONE iteration, so every
     difference comes from that iteration alone.
  2. Drift — three chained iterations, judged against a change already established as the same
     function, because a last-bit difference changes an action, which changes the next
     observation, so chained absolute differences say nothing on their own.

Run: PYTHONNOUSERSITE=1 <jax python> test_hoist_equivalence.py
"""
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

CFG = dict(n_copies=4, n_envs=4, num_steps=32, prime_iterations=1)
FIELDS = ["obs", "actions", "old_logprob", "adv", "ret_ext", "ret_int", "vext_old", "rnd_input"]


def rel(a, b):
    """Largest deviation relative to the magnitude of the quantity itself."""
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    return float(np.abs(a - b).max() / scale)


def build(hoist):
    """A runner and its primed state, identical apart from the hoist knob."""
    r = Runner(PPOConfig(hoist_rollout=hoist, **CFG))
    return r, r.prime(r.init_state(run_seed=5))


def build_unroll(update_unroll):
    """A runner and primed state differing only in the update scan's unroll factor.

    The reference for what harmless reassociation costs: unrolling instructs the compiler how
    many loop bodies to emit, and unroll 1 against 2 agrees to 3.7e-16 in double precision.
    """
    r = Runner(PPOConfig(update_unroll=update_unroll, **CFG))
    return r, r.prime(r.init_state(run_seed=5))


def batch_of(runner, state):
    """The update batch one iteration produces, alongside the state it leaves behind."""
    fn = jax.jit(runner.composition.iteration_capturing_batch())
    state2, _metrics, batch = fn(state, 3e-4)
    return batch, state2


# The log-probability is the one stored field the hoist provably cannot move: both forms compute
# it from the same action noise and the same log standard deviation, and from nothing else.
EXACT_FIELDS = ["old_logprob"]
# Everything else is downstream of the sampled action. Moving the critic and the bonus out of the
# scan body leaves the compiler a different loop to fuse, so the action can differ in the last
# float32 bit; the environment then carries that difference into every later observation, and the
# advantage scan sums about 1/(1 - gamma*lambda) ~ 20 terms of it. Measured on the processor at
# this configuration: the action moves by 1.7e-08 relative, the observation by 1.3e-07, and the
# advantage — the worst field — by 2.9e-07, against a limit of 2e-04.
VALUE_TOL = 1e-5
SCAN_AMPLIFICATION = 20


def test_isolation():
    """One iteration from identical state: exact where the hoist cannot matter, bounded elsewhere.

    Rewritten on 2026-08-16. The check used to demand that the observations, the actions and the
    bonus's whitened input be bit-identical between the two forms, and reported a failure once the
    environment's position noise went to zero. Re-measured against the frozen 09_parallelization
    baseline with the same interpreter, the deviation is there in the baseline too, at about
    5e-08 relative on the observations — it is float32 reassociation of the sampled action, not
    anything the zero-noise environment did and not anything the platform changed. So the exact
    requirement now names the one field that is genuinely a pure function of shared inputs, the
    first rollout step (where both forms still act from a byte-identical state) is required to be
    exact, and the rest is bounded.
    """
    ta, sa = build(False)
    tb, sb = build(True)
    ba, _ = batch_of(ta, sa)
    bb, _ = batch_of(tb, sb)

    for k in EXACT_FIELDS:
        d = rel(ba[k], bb[k])
        print(f"isolation: {k:12} {d:.3e} (must be exactly zero)")
        assert d == 0.0, f"the hoist changed {k}, which is a pure function of shared inputs"

    # the first rollout step runs from a byte-identical state in both forms, so the action taken
    # there — and the observation it was taken at — cannot yet have drifted
    # before: obs [C, T*N, 4] with row t*N + n; after: the N rows of step t = 0
    n_envs = CFG["n_envs"]
    for k in ["obs", "actions"]:
        d = rel(np.asarray(ba[k])[:, :n_envs], np.asarray(bb[k])[:, :n_envs])
        print(f"isolation: {k:12} {d:.3e} at the first rollout step (must be exactly zero)")
        assert d == 0.0, f"the hoist changed {k} at the very first step, before any drift"

    rest = sorted((rel(ba[k], bb[k]), k) for k in FIELDS if k not in EXACT_FIELDS)
    for d, k in rest:
        print(f"isolation: {k:12} {d:.3e} over the whole rollout")
    limit = VALUE_TOL * SCAN_AMPLIFICATION
    print(f"isolation: worst field {rest[-1][0]:.3e} ({rest[-1][1]}), limit {limit:.0e}")
    assert rest[-1][0] <= limit, "the hoist changed the computation beyond float32 reassociation"
    print("ok test_isolation")


def chained_drift(ta, sa, tb, sb, iterations=3):
    """Worst relative parameter deviation after running the same iterations through both."""
    for _ in range(iterations):
        sa, _ = ta.iterate(sa, 3e-4)
        sb, _ = tb.iterate(sb, 3e-4)
    return max(rel(x, y) for x, y in
               zip(jax.tree.leaves(sa.agent_params), jax.tree.leaves(sb.agent_params)))


def test_drift_against_a_control():
    """Chained drift, judged against a change that provably computes the same function.

    Training is a feedback loop, so ANY difference in float32 rounding grows from iteration to
    iteration and a fixed bound says nothing on its own. Measured: a change already established
    as the same function — the update scan at unroll 1 against unroll 2, which agree to 3.7e-16
    in double precision — drifts about 1.7e-03 over three iterations, seventeen times the 1e-4
    bound this test used to assert. So the control is measured here in the same run and the hoist
    is required only to stay in its neighbourhood.
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
