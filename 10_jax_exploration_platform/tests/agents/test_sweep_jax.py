"""Tests for the sweep across copy groups (round 4).

A sweep trains several groups of copies at once, each group at its own learning rate. The
properties that make the result trustworthy:

  1. a sweep whose rates are all equal reproduces the ordinary uniform-rate run,
  2. a group whose rate is zero never moves, while the others do,
  3. changing one group's rate leaves every other group's parameters bitwise unchanged,
  4. paired seeding gives copy k of every group the same initial weights, so the two copies act
     identically at the shared starting observation,
  5. distinct seeding gives every copy its own weights, so they do not.

Run with the platform's canonical JAX environment registered in
/p/rlprojects/RND/.venvs/ENVS.md (currently /p/rlprojects/RND/.venvs/platform_jax):
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python test_sweep_jax.py  (CPU)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.networks import actor_mean  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.sweep import sweep_config  # noqa: E402

import jax  # noqa: E402

SMALL = dict(n_envs=2, num_steps=8, prime_iterations=1)


def run(runner, iters=3, run_seed=17):
    """A few deterministic iterations from a fresh state; returns the final state."""
    state = runner.prime(runner.init_state(run_seed=run_seed))
    for it in range(1, iters + 1):
        state, _ = runner.iterate(state, runner.lr_argument(it, iters))
    return state


def leaves(params):
    """Flat list of arrays, in a stable order."""
    return [np.asarray(v) for v in jax.tree.leaves(params)]


def start_actions(runner):
    """What each copy's untrained policy does at the observation every episode begins from.

    before: the environment's reset state, identical for every copy (position noise is zero)
    after:  [C, 1, 2] action means — they can differ only because the policies differ
    """
    reset = runner.env.reset()
    obs = np.concatenate([np.asarray(reset.pos), np.asarray(reset.vel)], -1)[:, :1, :]
    actor = runner.init_trainable()["agent"]["actor"]
    return np.asarray(actor_mean(actor, obs))


def test_uniform_sweep_matches_plain_run():
    """All rates equal: the sweep path must reproduce the ordinary path."""
    plain = Runner(PPOConfig(n_copies=4, learning_rate=3e-4, **SMALL))
    swept = Runner(sweep_config([3e-4, 3e-4], 2, sweep_seed_mode="distinct", **SMALL))
    sp, ss = run(plain), run(swept)
    worst = max(np.abs(a - b).max()
                for a, b in zip(leaves(sp.agent_params), leaves(ss.agent_params)))
    print(f"uniform sweep vs plain run: worst parameter deviation {worst:.3e}")
    assert worst <= 1e-6, "the sweep path changed the uniform-rate result"
    print("ok test_uniform_sweep_matches_plain_run")


def test_zero_rate_group_is_frozen():
    """A group at rate zero must not move; a group at a real rate must."""
    t = Runner(sweep_config([0.0, 3e-3], 2, **SMALL))
    before = leaves(t.init_state().agent_params)
    after = leaves(run(t).agent_params)
    frozen = max(np.abs(a[:2] - b[:2]).max() for a, b in zip(before, after))
    moved = max(np.abs(a[2:] - b[2:]).max() for a, b in zip(before, after))
    print(f"zero-rate group movement {frozen:.3e}; trained group movement {moved:.3e}")
    assert frozen == 0.0, "the zero-rate group moved"
    assert moved > 0.0, "the trained group did not move"
    print("ok test_zero_rate_group_is_frozen")


def test_groups_do_not_influence_each_other():
    """Changing one group's rate must leave the other group's parameters bitwise identical."""
    a = Runner(sweep_config([3e-4, 1e-3], 2, **SMALL))
    b = Runner(sweep_config([3e-4, 5e-2], 2, **SMALL))   # only the second group differs
    la, lb = leaves(run(a).agent_params), leaves(run(b).agent_params)
    same = all((x[:2] == y[:2]).all() for x, y in zip(la, lb))
    differ = any((x[2:] != y[2:]).any() for x, y in zip(la, lb))
    print(f"unchanged group identical: {same}; changed group differs: {differ}")
    assert same, "changing one group's learning rate moved another group"
    assert differ, "changing the learning rate had no effect, so the test proves nothing"
    print("ok test_groups_do_not_influence_each_other")


def test_paired_seeding_gives_groups_the_same_start():
    """Paired seeding: copy k of every group starts from the same weights and acts the same.

    The environment cannot show this. Its position noise is zero, so every episode of every copy
    begins at exactly the same point and the starting observation is identical whatever the seed
    — the first assertion below states that, so the fact is on the record rather than assumed.
    What seeding controls is the POLICY, and the visible consequence is the action each copy's
    policy takes at that shared starting observation.
    """
    t = Runner(sweep_config([3e-4, 1e-3], 3, sweep_seed_mode="paired", **SMALL))
    reset = t.env.reset()
    obs = np.concatenate([np.asarray(reset.pos), np.asarray(reset.vel)], -1)
    assert (obs[:3] == obs[3:]).all(), "the zero-noise environment did not start every copy alike"
    for p in jax.tree.leaves(t.init_trainable()):
        assert (np.asarray(p)[:3] == np.asarray(p)[3:]).all(), \
            "paired groups did not start from the same weights"
    actions = start_actions(t)
    assert (actions[:3] == actions[3:]).all(), "paired groups did not act identically"
    print("ok test_paired_seeding_gives_groups_the_same_start")


def test_distinct_seeding_separates_every_copy():
    """Distinct seeding: every copy gets its own weights, so no two copies act alike.

    Rewritten on 2026-08-16 for the zero-noise environment. The check used to read the
    observation right after a reset and require two differently-seeded copies to differ there.
    With the position noise set to zero that is impossible — every episode begins at exactly the
    same point — so the check was asserting a property of the environment's randomness rather
    than of the seeding. What distinct seeding actually does is give every copy its own weight
    draw, and the way to see it is the action each copy's policy takes at the one starting
    observation they all share.
    """
    t = Runner(sweep_config([3e-4, 1e-3], 3, sweep_seed_mode="distinct", **SMALL))
    w = np.asarray(t.init_trainable()["agent"]["actor"]["W0"])
    assert not (w[:3] == w[3:]).all(), "distinct seeding still paired the groups' weights"
    actions = start_actions(t)
    assert not (actions[:3] == actions[3:]).all(), "distinct seeding still paired the policies"
    # every copy differs from every other, not merely group from group
    for i in range(6):
        for j in range(i + 1, 6):
            assert not (actions[i] == actions[j]).all(), f"copies {i} and {j} act identically"
    print("ok test_distinct_seeding_separates_every_copy")


def test_coverage_recording():
    """The visited map accumulates across iterations, stays inside [0, 1], and never shrinks.

    Uses a long enough rollout that the ball actually leaves its starting cell: with a rollout
    too short to move, coverage would sit at one cell and a map that reset every iteration would
    pass unnoticed.
    """
    t = Runner(PPOConfig(n_copies=4, n_envs=2, num_steps=128, prime_iterations=1,
                         track_coverage=True))
    state = t.prime(t.init_state(run_seed=5))
    seen = []
    for it in range(1, 4):
        state, _ = t.iterate(state, t.lr_argument(it, 3))
        seen.append(t.coverage(state))
    assert all(0.0 <= c.min() and c.max() <= 1.0 for c in seen), "coverage left [0, 1]"
    assert all((b >= a - 1e-12).all() for a, b in zip(seen, seen[1:])), \
        "a cumulative visited map shrank"
    assert seen[-1].mean() > seen[0].mean(), \
        "coverage did not grow, so the map is not accumulating across iterations"
    print(f"coverage after 1 and 3 iterations: {seen[0].mean():.3f} -> {seen[-1].mean():.3f}")
    print("ok test_coverage_recording")


if __name__ == "__main__":
    test_uniform_sweep_matches_plain_run()
    test_zero_rate_group_is_frozen()
    test_groups_do_not_influence_each_other()
    test_paired_seeding_gives_groups_the_same_start()
    test_distinct_seeding_separates_every_copy()
    test_coverage_recording()
