"""What parameter drift does three chained iterations produce from a change known to be benign?

`test_hoist_equivalence.py` asserts that hoisting the critic pass out of the rollout leaves the
parameters within 1e-4 relative after three iterations. That bound was chosen without a reference:
training is a feedback loop, so ANY difference in float32 rounding grows from iteration to
iteration, and the question is not whether the two runs drift but whether they drift more than an
arithmetically harmless change does.

This measures the control. It reruns the same three iterations for a pair that differs only in the
update scan's unroll factor — a change already established as the same function (the two agree to
3.7e-16 in double precision) — and prints its drift beside the hoist's. If the two are the same
size, the hoist's drift is what reassociation costs here and the gate's bound is the wrong
instrument; if the hoist drifts far more, something really did change.

Run (on the graphics-processor host, under the lock):
  bash locks/gpu_run.sh "PYTHONNOUSERSITE=1 <jax python> benchmarks/diagnose_drift_control.py"
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo" / "tests"))

import jax  # noqa: E402

import test_hoist_equivalence as T  # noqa: E402
from jax_ppo_rnd import PPOConfig, JaxPPORND  # noqa: E402


def drift(trainer_a, state_a, trainer_b, state_b, iterations=3):
    """Worst relative parameter deviation after chaining the same iterations through both."""
    # goal: identical keys and learning rate, so the only difference is the arithmetic order
    for i in range(iterations):
        key = jax.random.PRNGKey(100 + i)
        state_a, _ = trainer_a._iterate(state_a, key, 3e-4)
        state_b, _ = trainer_b._iterate(state_b, key, 3e-4)
    return max(T.rel(x, y) for x, y in
               zip(jax.tree.leaves(state_a.params), jax.tree.leaves(state_b.params)))


def build_unroll(update_unroll):
    """A trainer and primed state differing from the default only in the update scan's unroll."""
    # goal: mirror the gate's own construction exactly, so the control is comparable with it
    trainer = JaxPPORND(PPOConfig(update_unroll=update_unroll, **T.CFG))
    state = trainer.prime_obs_rms(trainer.init_state(), jax.random.PRNGKey(5))
    return trainer, state


def main():
    """Print the hoist's drift beside a benign change's drift over the same three iterations."""
    hoist = drift(*T.build(False), *T.build(True))
    control = drift(*build_unroll(1), *build_unroll(2))
    print(f"{'change':38} {'relative parameter drift, 3 iterations':>40}")
    print(f"{'hoist off against hoist on':38} {hoist:40.3e}")
    print(f"{'update unroll 1 against 2 (same function)':38} {control:40.3e}")
    print(f"\nratio: {hoist / control:.2f}x the drift of a change known to compute the same thing")


if __name__ == "__main__":
    main()
