"""The speed keys of the PointMaze and AntMaze study, checked on the processor.

Each key is off by default, and a program with every key off is the earlier program (the golden
parity test and the per-copy tests cover that). Here each key is turned on and compared with the
program that has it off:

  target_features_once_per_iteration  the fixed target's features of the update batch, computed
                                      once and gathered, are the features the loss recomputed;
                                      three training iterations end in the same state
  permutation_draw="sort_packed_keys" every epoch and copy gets each row exactly once; the order
                                      is the stable sort of the scheme's own random bits, which
                                      is the default scheme's order wherever no two rows tie in
                                      their leading 23 bits; row positions are uniform; the
                                      default scheme is unchanged
  next_values_from_shifted_values     training with truncations inside the rollouts, and with a
                                      termination on every step, ends in the same state; the
                                      configurations the key cannot serve are refused

  skip_reset_draws_at_zero_noise      same final state, bit for bit; refused with a noise that
                                      is not zero
  intrinsic_sums_by_matrix_product    the two matrices are the discount powers; the products
                                      equal the loops on random buffers; training ends in the
                                      same state
  advantage_scan_unroll               training ends in the same state
  action_noise_generator="rbg"        the default generator is unchanged; the other one's draws
                                      have mean 0, variance 1 and a normal distribution, and
                                      differ from the default's; training stays finite
  actor_critic_joint_trunk            the joint forward equals the two separate forwards;
                                      training ends in the same state

"The same state" is reported two ways: how many arrays are bit-identical, and the largest
difference among the others, which must stay at rounding size.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform import F32  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.networks import (actor_mean,  # noqa: E402
                                                      actor_mean_and_critic_values,
                                                      critic_values)
from exploration_platform.agents.ppo.update import draw_permutations  # noqa: E402
from exploration_platform.bonuses.rnd.networks import target_features  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.train_step import (action_noise,  # noqa: E402
                                                      intrinsic_sum_matrices)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# the study's update on a small shape: clipped, 4 epochs x 4 minibatches of 8 rows
SMALL = dict(n_copies=4, n_envs=2, num_steps=16, prime_iterations=1,
             update_style="epoch_minibatch", clip_coef=0.1, ent_coef=0.001, episode_log=True)
# the episode ends at the goal or after 20 steps: 16-step rollouts 2, 3 and 4 hold a truncation
TRUNCATIONS = EnvConfig(continuing_task=False, max_episode_steps=20)
# the goal is the start cell, so every step ends its episode at the goal
TERMINATIONS = EnvConfig(continuing_task=False, max_episode_steps=20, goal_cell=(7, 1))


def state_arrays(state) -> dict:
    """Every array of a training state, by its path in the tree."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def run(cfg, iterations=4, env_cfg=None):
    """Train a few iterations from a fresh state; return every array of the final state."""
    runner = Runner(cfg, env_cfg=env_cfg)
    state = runner.prime(runner.init_state(run_seed=17))
    for k in range(1, iterations + 1):
        state, _ = runner.iterate(state, runner.lr_argument(k, iterations))
    return state_arrays(state)


def compare_states(off: dict, on: dict, label: str, tolerance: float = 2e-5) -> None:
    """Require two final states to agree to rounding size, and say how many arrays are identical.

    Integer and boolean arrays (step counts, reset counts, the coverage map, the episode log's
    counts) must be equal exactly. A float array may differ by at most `tolerance` x (1 + |value|).
    """
    assert off.keys() == on.keys(), label
    identical, worst, worst_name = 0, 0.0, ""
    for name in off:
        a, b = off[name], on[name]
        if a.tobytes() == b.tobytes():
            identical += 1
            continue
        assert a.dtype.kind == "f", f"{label}: the non-float array {name} differs"
        difference = float(np.max(np.abs(a - b) / (1.0 + np.abs(a))))
        if difference > worst:
            worst, worst_name = difference, name
    assert worst <= tolerance, f"{label}: {worst_name} differs by {worst:.3e}"
    print(f"PASS {label}: {identical} of {len(off)} arrays bit-identical; largest difference "
          f"of the others {worst:.3e}" + (f" in {worst_name}" if worst_name else ""))


def test_target_features_gathered_equal_target_features_recomputed():
    """The target's features of gathered rows are the gathered features of all rows."""
    runner = Runner(PPOConfig(**SMALL))
    target = runner.bonus.constants["target"]
    x = jax.random.normal(jax.random.PRNGKey(3), (4, 32, 4), F32)
    rows = jnp.argsort(jax.random.uniform(jax.random.PRNGKey(4), (4, 8)), axis=-1)
    # before: features of all 32 rows, then rows picked; after: rows picked, then their features
    once = jnp.take_along_axis(target_features(target, x), rows[..., None], axis=1)
    again = target_features(target, jnp.take_along_axis(x, rows[..., None], axis=1))
    np.testing.assert_allclose(np.asarray(once), np.asarray(again), rtol=1e-5, atol=1e-6)
    print("PASS gathered target features equal recomputed ones; bit-identical: "
          f"{np.asarray(once).tobytes() == np.asarray(again).tobytes()}")


def test_target_features_once_per_iteration_trains_to_the_same_state():
    """Four iterations with the target's features computed once equal four with them recomputed,
    with minibatches gathered up front and one epoch at a time."""
    for extra in ({}, {"gather_minibatches_per_epoch": True}):
        off = run(PPOConfig(**SMALL, **extra), env_cfg=TRUNCATIONS)
        on = run(PPOConfig(target_features_once_per_iteration=True, **SMALL, **extra),
                 env_cfg=TRUNCATIONS)
        compare_states(off, on, f"target features once per iteration {extra or ''}".strip())


def test_default_permutation_draw_is_the_argsort_of_uniform_numbers():
    """The default scheme's orders are the ones the update always drew."""
    cfg = PPOConfig(n_copies=3, n_envs=2, num_steps=8, update_epochs=4, num_minibatches=4)
    key = jax.random.PRNGKey(11)
    drawn = draw_permutations(cfg, key, 16)
    before = jnp.argsort(jax.random.uniform(key, (4, 3, 16), F32), axis=-1)
    np.testing.assert_array_equal(np.asarray(drawn), np.asarray(before))
    print("PASS the default permutation draw is unchanged")


def test_packed_key_permutations_are_the_stable_sort_of_their_random_bits():
    """Every epoch and copy holds each row once, in the order of the scheme's own random bits."""
    cfg = PPOConfig(n_copies=3, n_envs=2, num_steps=8, update_epochs=4, num_minibatches=4,
                    permutation_draw="sort_packed_keys")
    key = jax.random.PRNGKey(11)
    drawn = np.asarray(draw_permutations(cfg, key, 16))
    assert drawn.shape == (4, 3, 16) and drawn.dtype == np.int32
    # before: the scheme's random bits, one row of 16 per (epoch, copy) group, [12, 16]
    # after:  each group's rows in the order of their bits, equal bits keeping row order
    bits = np.asarray(jax.random.bits(key, (12, 16), jnp.uint32))
    expected = np.argsort(bits, axis=-1, kind="stable").reshape(4, 3, 16)
    np.testing.assert_array_equal(drawn, expected)
    for order in drawn.reshape(12, 16):
        assert sorted(order.tolist()) == list(range(16))
    # a row count that is not a power of two (12 rows need 4 bits, values 12 to 15 unused)
    odd = np.asarray(draw_permutations(replace(cfg, n_envs=3, num_steps=4), key, 12))
    for order in odd.reshape(12, 12):
        assert sorted(order.tolist()) == list(range(12))
    print("PASS packed-key permutations: each row once, in the stable order of the random bits")


def test_packed_key_permutations_put_every_row_in_every_position_equally_often():
    """Over 8,000 orders of 8 rows, the row found at each position is uniform (chi-square)."""
    counts = {}
    for scheme in ("argsort_uniform", "sort_packed_keys"):
        cfg = PPOConfig(n_copies=2000, n_envs=1, num_steps=8, update_epochs=4,
                        num_minibatches=4, permutation_draw=scheme)
        orders = np.asarray(draw_permutations(cfg, jax.random.PRNGKey(5), 8)).reshape(-1, 8)
        # before: 8,000 orders [8000, 8]; after: table[position, row] of how often
        table = np.zeros((8, 8))
        for position in range(8):
            table[position] = np.bincount(orders[:, position], minlength=8)
        counts[scheme] = float(((table - 1000.0) ** 2 / 1000.0).sum())
    # 49 degrees of freedom: a uniform scheme stays below 85.4 with probability 0.999
    assert counts["sort_packed_keys"] < 85.4, counts
    print(f"PASS row positions are uniform: chi-square {counts['sort_packed_keys']:.1f} for "
          f"packed keys, {counts['argsort_uniform']:.1f} for the default scheme (49 degrees "
          "of freedom, 0.999 quantile 85.4)")


def test_packed_keys_refuse_a_shape_that_does_not_fit_64_bits():
    """2^20 (epoch, copy) groups, 32 random bits and 2^14 rows need 66 bits: refused."""
    cfg = PPOConfig(n_copies=2 ** 18, update_epochs=4, permutation_draw="sort_packed_keys")
    try:
        draw_permutations(cfg, jax.random.PRNGKey(0), 2 ** 14)
    except ValueError as error:
        assert "66 bits" in str(error)
        print(f"PASS refused: {error}")
        return
    raise AssertionError("a key of 66 bits was accepted")


def test_an_unknown_permutation_draw_is_refused():
    """A scheme name the update does not know is an error, not the default."""
    try:
        draw_permutations(PPOConfig(permutation_draw="shuffle"), jax.random.PRNGKey(0), 16)
    except ValueError as error:
        assert "permutation_draw" in str(error)
        print(f"PASS refused: {error}")
        return
    raise AssertionError("an unknown permutation_draw was accepted")


def test_packed_key_permutations_train():
    """Four iterations under the packed-key scheme stay finite.

    The scheme reads the same random bits as the default one, which sorts on the leading 23 of
    them. With 32 rows no two rows of a group share those 23 bits, so both schemes give the same
    orders and the same training here; the count of differing arrays is printed, not required.
    """
    off = run(PPOConfig(**SMALL), env_cfg=TRUNCATIONS)
    on = run(PPOConfig(permutation_draw="sort_packed_keys", **SMALL), env_cfg=TRUNCATIONS)
    assert all(np.isfinite(value).all() for value in on.values() if value.dtype.kind == "f")
    changed = [name for name in off if off[name].tobytes() != on[name].tobytes()]
    print(f"PASS the packed-key scheme trains; {len(changed)} of {len(off)} arrays differ from "
          "the default scheme's")


def test_packed_keys_order_rows_as_the_default_scheme_does_except_at_ties():
    """Both schemes sort the same random bits: the default on the leading 23 bits with equal
    values kept in row order, the packed keys on all 32. So wherever a group has no two rows
    with equal leading 23 bits, the two orders are the same."""
    cfg = PPOConfig(n_copies=5, n_envs=4, num_steps=128, update_epochs=4, num_minibatches=4)
    key = jax.random.PRNGKey(23)
    default = np.asarray(draw_permutations(cfg, key, 512)).reshape(20, 512)
    packed = np.asarray(draw_permutations(replace(cfg, permutation_draw="sort_packed_keys"),
                                          key, 512)).reshape(20, 512)
    # before: 32 random bits per row [20, 512]; after: their leading 23 bits, the default's keys
    leading = np.asarray(jax.random.bits(key, (20, 512), jnp.uint32)) >> 9
    same, tied = 0, 0
    for group in range(20):
        if len(np.unique(leading[group])) == 512:
            np.testing.assert_array_equal(default[group], packed[group])
            same += 1
        else:
            tied += 1
            assert sorted(packed[group].tolist()) == list(range(512))
    assert same > 0
    print(f"PASS {same} of 20 groups have no tie and the same order under both schemes; "
          f"{tied} groups have a tie")


def test_next_values_from_shifted_values_train_to_the_same_state():
    """With truncations inside the rollouts, and with a termination on every step, four
    iterations with next values read from the following step equal four with the second pass."""
    for label, env_cfg in (("truncations inside the rollouts", TRUNCATIONS),
                           ("a termination on every step", TERMINATIONS)):
        off = run(PPOConfig(**SMALL), env_cfg=env_cfg)
        on = run(PPOConfig(next_values_from_shifted_values=True, **SMALL), env_cfg=env_cfg)
        # the episodes the check is about did happen
        episodes = off[".agent_state['episode_log']['episodes']"]
        arrivals = off[".agent_state['episode_log']['arrivals']"]
        assert episodes.min() > 0
        assert (arrivals.min() > 0) == (env_cfg is TERMINATIONS)
        compare_states(off, on, f"next values from shifted values, {label}")


def test_next_values_from_shifted_values_refuse_what_they_cannot_serve():
    """Refused: the historical rule that reads final observations for the intrinsic return, and
    a step limit shorter than a rollout (two truncations of one environment in a rollout)."""
    cases = ((PPOConfig(next_values_from_shifted_values=True,
                        configuration_preset="pointmaze_historical", **SMALL), TRUNCATIONS),
             (PPOConfig(next_values_from_shifted_values=True, **SMALL),
              EnvConfig(continuing_task=False, max_episode_steps=6)))
    for cfg, env_cfg in cases:
        try:
            Runner(cfg, env_cfg=env_cfg)
        except ValueError as error:
            assert "next_values_from_shifted_values" in str(error)
            print(f"PASS refused: {error}")
            continue
        raise AssertionError("next_values_from_shifted_values accepted a case it cannot serve")


def test_skipping_reset_draws_at_zero_noise_changes_nothing():
    """With no reset noise, a reset without the keyed draws gives the same state bit for bit,
    with truncations and with a termination on every step; a noise that is not zero is refused."""
    for label, env_cfg in (("truncations", TRUNCATIONS), ("a termination on every step",
                                                          TERMINATIONS)):
        off = run(PPOConfig(**SMALL), env_cfg=env_cfg)
        on = run(PPOConfig(skip_reset_draws_at_zero_noise=True, **SMALL), env_cfg=env_cfg)
        assert off.keys() == on.keys()
        for name in off:
            assert off[name].tobytes() == on[name].tobytes(), f"{label}: {name} differs"
        print(f"PASS no reset draws at zero noise, {label}: {len(off)} arrays bit-identical")
    try:
        Runner(PPOConfig(skip_reset_draws_at_zero_noise=True, **SMALL),
               env_cfg=EnvConfig(position_noise=0.25))
    except ValueError as error:
        assert "position_noise" in str(error)
        print(f"PASS refused: {error}")
        return
    raise AssertionError("skipping the draws was accepted with a noise that is not zero")


def test_intrinsic_sum_matrices_hold_the_discount_powers():
    """The docstring's example, and the products against the loops on random buffers."""
    filter_matrix, carry_powers, advantage_matrix = intrinsic_sum_matrices(0.5, 1.0, 3)
    np.testing.assert_array_equal(np.asarray(filter_matrix),
                                  [[1, 0, 0], [0.5, 1, 0], [0.25, 0.5, 1]])
    np.testing.assert_array_equal(np.asarray(carry_powers), [0.5, 0.25, 0.125])
    np.testing.assert_array_equal(np.asarray(advantage_matrix),
                                  [[1, 0.5, 0.25], [0, 1, 0.5], [0, 0, 1]])
    # before: rewards and one-step differences [T, K]; after: the loops' results, in float64
    gamma, lam, T = 0.99, 0.95, 128
    filter_matrix, carry_powers, advantage_matrix = intrinsic_sum_matrices(gamma, lam, T)
    rng = np.random.default_rng(0)
    r, d, carried = rng.normal(size=(T, 6)), rng.normal(size=(T, 6)), rng.normal(size=6)
    filt, f = np.zeros((T, 6)), carried.copy()
    for t in range(T):
        f = gamma * f + r[t]
        filt[t] = f
    adv, a = np.zeros((T, 6)), np.zeros(6)
    for t in reversed(range(T)):
        a = d[t] + gamma * lam * a
        adv[t] = a
    by_product = (np.asarray(filter_matrix, np.float64) @ r
                  + np.asarray(carry_powers, np.float64)[:, None] * carried)
    np.testing.assert_allclose(by_product, filt, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(advantage_matrix, np.float64) @ d, adv, rtol=1e-5,
                               atol=1e-5)
    print("PASS the filter and advantage matrices reproduce the loops on random buffers")


def test_intrinsic_sums_by_matrix_product_and_unrolled_loops_train_to_the_same_state():
    """Four iterations with the intrinsic sums as matrix products, with the loops unrolled 16
    steps, and with both, against four iterations with the plain loops."""
    off = run(PPOConfig(**SMALL), env_cfg=TRUNCATIONS)
    for label, keys in (("intrinsic sums by matrix product",
                         {"intrinsic_sums_by_matrix_product": True}),
                        ("advantage loops unrolled 16 steps", {"advantage_scan_unroll": 16}),
                        ("matrix product and unrolled extrinsic loop",
                         {"intrinsic_sums_by_matrix_product": True,
                          "advantage_scan_unroll": 16})):
        compare_states(off, run(PPOConfig(**SMALL, **keys), env_cfg=TRUNCATIONS), label)


def test_action_noise_generators():
    """The default generator is unchanged; the other one draws standard normal numbers."""
    key, shape = jax.random.PRNGKey(9), (128, 64, 4, 2)
    default = np.asarray(action_noise(PPOConfig(), key, shape))
    np.testing.assert_array_equal(default, np.asarray(jax.random.normal(key, shape, F32)))
    other = np.asarray(action_noise(PPOConfig(action_noise_generator="rbg"), key, shape))
    assert other.shape == shape and other.dtype == np.float32
    assert not np.array_equal(other, default)
    again = np.asarray(action_noise(PPOConfig(action_noise_generator="rbg"),
                                    jax.random.PRNGKey(10), shape))
    assert not np.array_equal(other, again), "another key must give other numbers"
    # 65,536 draws: mean within 4 standard errors, variance within 3%, and the largest gap
    # between the empirical and the normal distribution function below the 0.001 level
    flat = np.sort(other.ravel().astype(np.float64))
    n = flat.size
    assert abs(flat.mean()) < 4.0 / np.sqrt(n) and abs(flat.var() - 1.0) < 0.03
    normal_cdf = np.asarray(jax.scipy.stats.norm.cdf(jnp.asarray(flat)))
    gap = np.max(np.abs(normal_cdf - (np.arange(n) + 0.5) / n))
    assert gap < 1.95 / np.sqrt(n), gap
    # the two action dimensions of a step are not correlated
    pairs = other.reshape(-1, 2).astype(np.float64)
    assert abs(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) < 4.0 / np.sqrt(len(pairs))
    try:
        action_noise(PPOConfig(action_noise_generator="other"), key, shape)
        raise AssertionError("an unknown generator was accepted")
    except ValueError as error:
        assert "action_noise_generator" in str(error)
    on = run(PPOConfig(action_noise_generator="rbg", **SMALL), env_cfg=TRUNCATIONS)
    assert all(np.isfinite(value).all() for value in on.values() if value.dtype.kind == "f")
    print(f"PASS action noise: default unchanged; rbg mean {flat.mean():+.4f}, variance "
          f"{flat.var():.4f}, largest distribution gap {gap:.4f} (limit "
          f"{1.95 / np.sqrt(n):.4f}); training stays finite")


def test_joint_trunk_equals_the_two_separate_forwards():
    """The batched forward of both networks gives each network's own outputs, and training with
    it ends in the same state."""
    runner = Runner(PPOConfig(**SMALL))
    agent = runner.init_trainable()["agent"]
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 40, 4), F32)
    mean, vext, vint = actor_mean_and_critic_values(agent["actor"], agent["critic"], x)
    separate = (actor_mean(agent["actor"], x), *critic_values(agent["critic"], x))
    identical = all(np.asarray(a).tobytes() == np.asarray(b).tobytes()
                    for a, b in zip((mean, vext, vint), separate))
    for a, b in zip((mean, vext, vint), separate):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
    print(f"PASS the joint forward equals the separate forwards; bit-identical: {identical}")
    off = run(PPOConfig(**SMALL), env_cfg=TRUNCATIONS)
    on = run(PPOConfig(actor_critic_joint_trunk=True, **SMALL), env_cfg=TRUNCATIONS)
    compare_states(off, on, "actor and critic hidden layers as one batched network")


if __name__ == "__main__":
    test_target_features_gathered_equal_target_features_recomputed()
    test_target_features_once_per_iteration_trains_to_the_same_state()
    test_default_permutation_draw_is_the_argsort_of_uniform_numbers()
    test_packed_key_permutations_are_the_stable_sort_of_their_random_bits()
    test_packed_key_permutations_put_every_row_in_every_position_equally_often()
    test_packed_keys_refuse_a_shape_that_does_not_fit_64_bits()
    test_an_unknown_permutation_draw_is_refused()
    test_packed_key_permutations_train()
    test_packed_keys_order_rows_as_the_default_scheme_does_except_at_ties()
    test_next_values_from_shifted_values_train_to_the_same_state()
    test_next_values_from_shifted_values_refuse_what_they_cannot_serve()
    test_skipping_reset_draws_at_zero_noise_changes_nothing()
    test_intrinsic_sum_matrices_hold_the_discount_powers()
    test_intrinsic_sums_by_matrix_product_and_unrolled_loops_train_to_the_same_state()
    test_action_noise_generators()
    test_joint_trunk_equals_the_two_separate_forwards()
