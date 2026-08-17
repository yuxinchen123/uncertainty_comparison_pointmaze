"""One iteration must compute the same thing whether or not it was compiled.

Compiling is an optimisation, so running the same function with compilation switched off has to
give the same answer. It is not required to give the same BITS: without compilation there is no
fusion, so a sum is accumulated in a different order and float32 rounds differently. Integer state
— a visit-count table, the optimizer's step counter — has no such excuse and is required to be
exactly equal.

Floating-point arrays are judged by the mixed rule this project already uses for its
cross-framework check: the difference has to satisfy `|a - b| <= atol + rtol*|b|` with rtol 1e-5
and atol 1e-6, reported as `max |a - b| / (atol/rtol + |b|)` so one number covers every array. A
purely relative rule cannot work here, because the state holds quantities on wildly different
scales AND quantities that start at zero: the variance of the intrinsic returns under random
network distillation is of order 1e8 after one iteration, while the log standard deviation starts
at exactly zero, so its whole magnitude after one iteration IS the step just taken and a last-bit
difference in that step reads as a large relative difference in the parameter.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_jit_eager_equivalence.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import BONUS_REGISTRY  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

SMALL = dict(n_copies=3, n_envs=2, num_steps=8, prime_iterations=1, update_style="full_batch")
RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE = 1e-5, 1e-6


def arrays(state):
    """Every array of a state, with a readable name."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return [(jax.tree_util.keystr(k), np.asarray(v)) for k, v in flat]


def check(bonus_name: str):
    """Run one iteration compiled and one uncompiled from the same primed state."""
    runner = Runner(PPOConfig(**SMALL), bonus=bonus_name)
    start = runner.prime(runner.init_state(run_seed=4))
    # the compiled iteration donates its state argument, so its buffers are gone afterwards; each
    # call gets its own copy of the same numbers
    copy_of = lambda tree: jax.tree.map(jnp.copy, tree)
    compiled, _ = runner.iterate(copy_of(start), runner.lr_argument(1, 1))
    with jax.disable_jit():
        eager, _ = runner.iterate(copy_of(start), runner.lr_argument(1, 1))

    worst, worst_name, worst_absolute = 0.0, "", 0.0
    for (name, a), (_n, b) in zip(arrays(compiled), arrays(eager)):
        assert a.shape == b.shape and a.dtype == b.dtype, f"{bonus_name} {name}: shape or type"
        if a.dtype.kind in "iub":
            assert (a == b).all(), f"{bonus_name} {name}: integer state differs"
            continue
        if not a.size:
            continue
        x, y = a.astype(np.float64), b.astype(np.float64)
        absolute = float(np.abs(x - y).max())
        # the mixed rule, written so one number can be compared against rtol
        mixed = float(np.max(np.abs(x - y)
                             / (ABSOLUTE_TOLERANCE / RELATIVE_TOLERANCE + np.abs(y))))
        if mixed > worst:
            worst, worst_name, worst_absolute = mixed, name, absolute
    print(f"{bonus_name:28} integer state exactly equal; worst float difference {worst:.2e} "
          f"by the mixed rule ({worst_absolute:.2e} absolute, at {worst_name or 'nothing'})")
    assert worst <= RELATIVE_TOLERANCE, (
        f"{bonus_name}: uncompiled and compiled disagree beyond float32 reassociation")


def test_every_bonus_agrees_compiled_and_not():
    """Every registered family."""
    for bonus_name in sorted(BONUS_REGISTRY):
        check(bonus_name)
    print("ok test_every_bonus_agrees_compiled_and_not")


if __name__ == "__main__":
    test_every_bonus_agrees_compiled_and_not()
