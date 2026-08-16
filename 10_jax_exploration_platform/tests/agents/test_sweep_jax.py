"""Tests for the JAX learning-rate sweep across copy groups (round 4).

Mirrors ../../torch_ppo/tests/test_sweep.py. A sweep trains several groups of copies at once,
each group at its own learning rate. The properties that make the result trustworthy:

  1. a sweep whose rates are all equal reproduces the ordinary uniform-rate run,
  2. a group whose rate is zero never moves, while the others do,
  3. changing one group's rate leaves every other group's parameters bitwise unchanged,
  4. paired seeding gives copy k of every group the same initial weights and the same
     environments, so a difference between groups is the rate's doing and nothing else,
  5. distinct seeding gives every copy its own stream, as before.

Run with the platform's canonical JAX environment registered in
/p/rlprojects/RND/.venvs/ENVS.md (currently /p/rlprojects/RND/.venvs/platform_jax):
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python test_sweep_jax.py  (CPU)
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent / "src" / "exploration_platform"
sys.path.insert(0, str(BASE / "agents" / "ppo"))
from jax_ppo_rnd import PPOConfig, JaxPPORND, sweep_config  # noqa: E402

import jax  # noqa: E402

SMALL = dict(n_envs=2, num_steps=8, obs_norm_init_iters=1)


def run(trainer, iters=3, run_seed=17):
    """A few deterministic iterations from a fresh state; returns the final state."""
    state = trainer.init_state()
    key = jax.random.PRNGKey(run_seed)
    state = trainer.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    for it in range(1, iters + 1):
        state, _ = trainer._iterate(state, jax.random.fold_in(key, it),
                                    trainer.lr_argument(it, iters))
    return state


def leaves(params):
    """Flat list of arrays, in a stable order."""
    return [np.asarray(v) for v in jax.tree.leaves(params)]


def test_uniform_sweep_matches_plain_run():
    """All rates equal: the sweep path must reproduce the ordinary path."""
    plain = JaxPPORND(PPOConfig(n_copies=4, learning_rate=3e-4, **SMALL))
    swept = JaxPPORND(sweep_config([3e-4, 3e-4], 2, sweep_seed_mode="distinct", **SMALL))
    sp, ss = run(plain), run(swept)
    worst = max(np.abs(a - b).max() for a, b in zip(leaves(sp.params), leaves(ss.params)))
    print(f"uniform sweep vs plain run: worst parameter deviation {worst:.3e}")
    assert worst <= 1e-6, "the sweep path changed the uniform-rate result"
    print("ok test_uniform_sweep_matches_plain_run")


def test_zero_rate_group_is_frozen():
    """A group at rate zero must not move; a group at a real rate must."""
    t = JaxPPORND(sweep_config([0.0, 3e-3], 2, **SMALL))
    before = leaves(t.init_state().params)
    after = leaves(run(t).params)
    frozen = max(np.abs(a[:2] - b[:2]).max() for a, b in zip(before, after))
    moved = max(np.abs(a[2:] - b[2:]).max() for a, b in zip(before, after))
    print(f"zero-rate group movement {frozen:.3e}; trained group movement {moved:.3e}")
    assert frozen == 0.0, "the zero-rate group moved"
    assert moved > 0.0, "the trained group did not move"
    print("ok test_zero_rate_group_is_frozen")


def test_groups_do_not_influence_each_other():
    """Changing one group's rate must leave the other group's parameters bitwise identical."""
    a = JaxPPORND(sweep_config([3e-4, 1e-3], 2, **SMALL))
    b = JaxPPORND(sweep_config([3e-4, 5e-2], 2, **SMALL))   # only the second group differs
    la, lb = leaves(run(a).params), leaves(run(b).params)
    same = all((x[:2] == y[:2]).all() for x, y in zip(la, lb))
    differ = any((x[2:] != y[2:]).any() for x, y in zip(la, lb))
    print(f"unchanged group identical: {same}; changed group differs: {differ}")
    assert same, "changing one group's learning rate moved another group"
    assert differ, "changing the learning rate had no effect, so the test proves nothing"
    print("ok test_groups_do_not_influence_each_other")


def test_paired_seeding_gives_groups_the_same_start():
    """Paired seeding: copy k of every group starts identical and sees the same environments."""
    t = JaxPPORND(sweep_config([3e-4, 1e-3], 3, sweep_seed_mode="paired", **SMALL))
    for p in leaves(t.init_params):
        assert (p[:3] == p[3:]).all(), "paired groups did not start from the same weights"
    st = t.env.reset()
    obs = np.asarray(np.concatenate([st.pos, st.vel], -1))
    assert (obs[:3] == obs[3:]).all(), "paired groups did not get the same environments"
    print("ok test_paired_seeding_gives_groups_the_same_start")


def test_distinct_seeding_separates_every_copy():
    """Distinct seeding: no two copies share weights or environments."""
    t = JaxPPORND(sweep_config([3e-4, 1e-3], 3, sweep_seed_mode="distinct", **SMALL))
    w = np.asarray(t.init_params["actor"]["W0"])
    assert not (w[:3] == w[3:]).all(), "distinct seeding still paired the groups"
    st = t.env.reset()
    obs = np.asarray(np.concatenate([st.pos, st.vel], -1))
    assert not (obs[:3] == obs[3:]).all(), "distinct seeding still paired the environments"
    print("ok test_distinct_seeding_separates_every_copy")


def test_coverage_recording():
    """The visited map accumulates across iterations, stays inside [0, 1], and never shrinks.

    Uses a long enough rollout that the ball actually leaves its starting cell: with a rollout
    too short to move, coverage would sit at one cell and a map that reset every iteration
    would pass unnoticed.
    """
    t = JaxPPORND(PPOConfig(n_copies=4, n_envs=2, num_steps=128, obs_norm_init_iters=1,
                            track_coverage=True))
    state = t.init_state()
    key = jax.random.PRNGKey(5)
    state = t.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    seen = []
    for it in range(1, 4):
        state, _ = t._iterate(state, jax.random.fold_in(key, it), t.lr_argument(it, 3))
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
