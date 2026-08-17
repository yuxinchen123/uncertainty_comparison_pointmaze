"""One copy's training must never move another copy's — with any bonus.

The copies share a program, not a run: every parameter, every statistic and every count table
carries a leading copy axis, and one iteration is C independent iterations. The check is the
strongest one available: perturb copy 2's weights before training, run both, and require copies 0,
1 and 3 to end BIT-identical while copy 2 does not.

The bonus's own state is checked too, because that is where a new family is most likely to leak —
a scatter that indexes the table without the copy axis, or a statistic reduced over the wrong
axis, would join the copies together silently and leave every parameter test still passing.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_bonus_copy_isolation.py
"""
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import BONUS_REGISTRY  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

SMALL = dict(n_copies=4, n_envs=2, num_steps=16, prime_iterations=1)
KEEP = np.array([0, 1, 3])


def run(runner, state, iterations=2):
    """A few deterministic iterations from the given state."""
    state = runner.prime(state)
    for it in range(1, iterations + 1):
        state, _ = runner.iterate(state, runner.lr_argument(it, iterations))
    return state


def perturbed(runner):
    """A starting state with only copy 2's first actor weight moved."""
    state = runner.init_state(run_seed=7)
    moved = state.agent_params["actor"]["W0"].at[2].add(0.05)
    return state._replace(agent_params={**state.agent_params,
                                        "actor": {**state.agent_params["actor"], "W0": moved}})


def named_arrays(tree, prefix):
    """Every array under a tree, with a readable name, as host arrays."""
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return [(prefix + jax.tree_util.keystr(k), np.asarray(v)) for k, v in flat]


def check(bonus_name: str, style: str):
    """Train an untouched run and a perturbed one, and compare copy by copy."""
    a = Runner(PPOConfig(update_style=style, **SMALL), bonus=bonus_name)
    b = Runner(PPOConfig(update_style=style, **SMALL), bonus=bonus_name)
    sa = run(a, a.init_state(run_seed=7))
    sb = run(b, perturbed(b))

    parts = lambda s: (named_arrays(s.agent_params, "agent_params")
                       + named_arrays(s.bonus_params, "bonus_params")
                       + named_arrays(s.bonus_state, "bonus_state")
                       + named_arrays(s.agent_state, "agent_state"))
    moved_somewhere = False
    for (ka, va), (_kb, vb) in zip(parts(sa), parts(sb)):
        assert (va[KEEP] == vb[KEEP]).all(), f"{bonus_name}/{style} {ka}: copies 0/1/3 diverged"
        moved_somewhere |= not (va[2] == vb[2]).all()
    assert moved_somewhere, (f"{bonus_name}/{style}: the perturbation changed nothing at all, so "
                            "the check proves nothing")
    print(f"ok {bonus_name:28} {style:16} copies 0/1/3 bit-identical, copy 2 differs")


def test_copy_isolation_for_every_bonus():
    """Every registered family, in both update styles."""
    for bonus_name in sorted(BONUS_REGISTRY):
        for style in ("full_batch", "epoch_minibatch"):
            check(bonus_name, style)
    print("ok test_copy_isolation_for_every_bonus")


def test_visit_count_tables_are_per_copy():
    """A visit-count table must differ between copies that went to different places.

    Distinct seeding gives every copy its own weights, so the copies take different paths and
    their tables have to end up different. A table shared across copies — the mistake a scatter
    that forgets the copy axis makes — would show up here as four identical tables.
    """
    runner = Runner(PPOConfig(update_style="full_batch", **SMALL),
                    bonus="gt_position_velocity_sqrt")
    state = run(runner, runner.init_state(run_seed=7))
    counts = np.asarray(state.bonus_state["counts"])
    identical = [(i, j) for i in range(4) for j in range(i + 1, 4)
                 if (counts[i] == counts[j]).all()]
    per_copy = [int((row > 0).sum()) for row in counts]
    print(f"distinct states counted per copy: {per_copy}; pairs of copies with identical "
          f"tables: {identical}")
    assert not identical, "two copies ended with the same count table"
    assert all(n > 0 for n in per_copy), "a copy counted nothing at all"
    print("ok test_visit_count_tables_are_per_copy")


if __name__ == "__main__":
    test_copy_isolation_for_every_bonus()
    test_visit_count_tables_are_per_copy()
