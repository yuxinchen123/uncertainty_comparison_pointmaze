"""The two output-layer fits of `bonuses/rnd/last_layer.py`, on hand-made features and targets.

Checked: the smallest joint change of the layer (against NumPy's pseudo-inverse, and with states
that share one feature row), the temporary layer of the fast last layer (a state seen again in
one rollout), and the kept layer of the scheduled last layer (one state seen k times follows
(1 + k)^(-a); a state with orthogonal features keeps its error).

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.bonuses.rnd.last_layer import (by_step, fast_layer_scores,  # noqa: E402
                                                         fit_rows, flat_rollout,
                                                         scheduled_layer_scores,
                                                         unit_rows_with_bias)

import jax.numpy as jnp  # noqa: E402

FEATURES, OUTPUTS = 5, 3
half_squared = lambda error: 0.5 * (error ** 2).sum(-1)
length_of = lambda error: jnp.sqrt((error ** 2).sum(-1))


def unit(vectors):
    """Float64 vectors scaled to unit length along the last axis."""
    vectors = np.asarray(vectors, np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def rollout_of(states, order):
    """Stack per-state arrays into a rollout of one policy.

    states: an array [S, k]; order: per step, the state index of each environment.
    before: order [[0, 1], [0, 1]] ; after: [T = 2, C = 1, N = 2, k]
    """
    return jnp.asarray(np.asarray(states)[np.asarray(order)][:, None])


def test_unit_rows_with_bias_and_the_rollout_shapes():
    """Features get the bias input and unit length; a rollout is cut into steps and back."""
    rows, length = unit_rows_with_bias(jnp.asarray([[3.0, 0.0], [0.0, 0.0]], jnp.float32))
    assert rows.dtype == jnp.float64 and length.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(rows), [np.array([3, 0, 1]) / np.sqrt(10), [0, 0, 1]],
                               rtol=1e-15)
    np.testing.assert_allclose(np.asarray(length), [[np.sqrt(10)], [1.0]], rtol=1e-15)

    # 2 policies, 3 steps, 2 environments: row t*N + n of the flat form is environment n at step t
    flat = jnp.arange(2 * 6 * 1, dtype=jnp.float64).reshape(2, 6, 1)
    steps = by_step(flat, 3, 2)
    assert steps.shape == (3, 2, 2, 1)
    np.testing.assert_array_equal(np.asarray(steps[1, 0, :, 0]), [2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(flat_rollout(steps[..., 0])), np.asarray(flat[..., 0]))
    try:
        by_step(flat, 4, 2)
        raise AssertionError("a rollout of the wrong length was accepted")
    except ValueError as error:
        assert "4 steps x 2" in str(error)
    print("PASS unit rows with the bias input; rollout cut into steps and back")


def test_fit_rows_give_the_smallest_joint_change():
    """The change built from `fit_rows` moves every state's error by the wanted amount and is the
    smallest such change; states with one feature row share the change."""
    rng = np.random.default_rng(0)
    rows = unit(rng.normal(size=(3, FEATURES + 1)))
    wanted = rng.normal(size=(3, OUTPUTS))
    change = np.asarray(fit_rows(jnp.asarray(rows))).T @ wanted
    np.testing.assert_allclose(rows @ change, wanted, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(change, np.linalg.pinv(rows) @ wanted, rtol=1e-6, atol=1e-8)

    # one state: the row itself; orthogonal rows: the same rows; one row twice: half each
    one = jnp.asarray(unit([[0.6, 0.8, 0.0]]))
    np.testing.assert_array_equal(np.asarray(fit_rows(one)), np.asarray(one))
    orthogonal = jnp.asarray(np.eye(3)[:2])
    np.testing.assert_allclose(np.asarray(fit_rows(orthogonal)), np.eye(3)[:2], atol=1e-8)
    # one row twice: the two together make the row (what moves the layer), each about half
    # (how the row is split between two equal states is decided by rounding, to about 1e-7)
    twice = jnp.asarray(unit([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]))
    shared = np.asarray(fit_rows(twice))
    np.testing.assert_allclose(shared.sum(axis=0), np.asarray(twice)[0], rtol=1e-8)
    np.testing.assert_allclose(shared, np.asarray(twice) / 2, rtol=1e-5)
    print("PASS fit rows: joint, smallest, and shared between states with the same features")


def fast_inputs(order, seed=1):
    """A rollout of the fast last layer for one policy: rows, lengths, targets, and a layer."""
    rng = np.random.default_rng(seed)
    hidden = np.abs(rng.normal(size=(2, FEATURES))).astype(np.float32)   # two states, A and B
    rows, length = unit_rows_with_bias(jnp.asarray(hidden))
    target = rng.normal(size=(2, OUTPUTS))
    layer = jnp.asarray(rng.normal(size=(1, FEATURES + 1, OUTPUTS)))
    return (rollout_of(rows, order), rollout_of(length, order), rollout_of(target, order), layer)


def test_fast_layer_lowers_a_state_seen_again_in_the_rollout():
    """Exact fit (eta 1): a second visit pays 0. Half fit: the error halves at every visit."""
    # one environment sees state A three times
    rows, length, target, layer = fast_inputs([[0], [0], [0]])
    first = float(half_squared(length[0, 0, 0] * (rows[0, 0, 0] @ layer[0]) - target[0, 0, 0]))
    bonus = np.asarray(fast_layer_scores(rows, length, target, layer, 1.0, half_squared))
    assert bonus.shape == (3, 1, 1) and abs(bonus[0, 0, 0] - first) < 1e-12 * first
    assert bonus[1, 0, 0] < 1e-24 * first and bonus[2, 0, 0] < 1e-24 * first, bonus.ravel()
    bonus = np.asarray(fast_layer_scores(rows, length, target, layer, 0.5, half_squared))
    np.testing.assert_allclose(bonus.ravel(), [first, first / 4, first / 16], rtol=1e-10)
    # the length of the error halves where its half square quarters
    lengths = np.asarray(fast_layer_scores(rows, length, target, layer, 0.5, length_of)).ravel()
    np.testing.assert_allclose(lengths, np.sqrt(2 * first) * np.array([1, 0.5, 0.25]), rtol=1e-10)

    # two environments, states A and B at both steps: both are fitted at once, so both pay
    # (almost) nothing at the second step although their features overlap
    rows, length, target, layer = fast_inputs([[0, 1], [0, 1]])
    assert float(rows[0, 0, 0] @ rows[0, 0, 1]) > 0.1, "the two states should share features"
    bonus = np.asarray(fast_layer_scores(rows, length, target, layer, 1.0, half_squared))
    assert (bonus[1] < 1e-14 * bonus[0]).all(), bonus

    # both environments in the same state: the system has no unique solution, the result is
    # finite, and the state still pays (almost) nothing at the next step
    rows, length, target, layer = fast_inputs([[0, 0], [0, 0]])
    bonus = np.asarray(fast_layer_scores(rows, length, target, layer, 1.0, half_squared))
    assert np.isfinite(bonus).all() and (bonus[1] < 1e-14 * bonus[0]).all(), bonus
    print("PASS fast last layer: exact fit pays 0 on a second visit; half fit quarters the bonus")


def test_scheduled_layer_follows_the_schedule_exactly():
    """One state seen k times has squared error (1 + k)^(-a); the layer carries on between
    rollouts; a state with orthogonal features keeps its error of 1."""
    rng = np.random.default_rng(2)
    # three states: A and B with orthogonal feature rows, C between them; unit-length targets
    feature_rows = unit([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
    targets = unit(rng.normal(size=(3, OUTPUTS)))
    squared_error = lambda layer, s: float(((feature_rows[s] @ np.asarray(layer)[0]
                                             - targets[s]) ** 2).sum())
    visits = 6
    for decay in (0.5, 0.75, 1.0):
        layer = jnp.zeros((1, 4, OUTPUTS), jnp.float64)
        order = [[0]] * visits
        layer, squared = scheduled_layer_scores(rollout_of(feature_rows, order),
                                                rollout_of(targets, order), layer, decay)
        # the bonus at visit k (from 1) is the error after k - 1 visits
        np.testing.assert_allclose(np.asarray(squared).ravel(),
                                   (1.0 + np.arange(visits)) ** -decay, rtol=1e-12)
        np.testing.assert_allclose(squared_error(layer, 0), (1.0 + visits) ** -decay, rtol=1e-12)
        # the state never visited, with orthogonal features, is where it started
        np.testing.assert_allclose(squared_error(layer, 1), 1.0, rtol=1e-12)
        # the state that shares features with the visited one has moved
        assert abs(squared_error(layer, 2) - 1.0) > 1e-3

        # a second rollout continues from the kept layer
        layer, squared = scheduled_layer_scores(rollout_of(feature_rows, order),
                                                rollout_of(targets, order), layer, decay)
        np.testing.assert_allclose(np.asarray(squared).ravel(),
                                   (1.0 + visits + np.arange(visits)) ** -decay, rtol=1e-11)
    print("PASS scheduled last layer: (1 + k)^(-a) after k visits for a = 0.5, 0.75, 1; "
          "orthogonal features keep error 1")


def test_scheduled_layer_fits_the_states_of_one_step_jointly():
    """Two environments with overlapping features: after a step, EACH state's squared error is
    the schedule's next value, which fitting them one after the other would not give."""
    rng = np.random.default_rng(3)
    feature_rows = unit([[1, 0.5, 0, 0], [0.5, 1, 0.2, 0]])
    targets = unit(rng.normal(size=(2, OUTPUTS)))
    order = [[0, 1]] * 4
    for decay in (0.5, 1.0):
        _, squared = scheduled_layer_scores(rollout_of(feature_rows, order),
                                            rollout_of(targets, order),
                                            jnp.zeros((1, 4, OUTPUTS), jnp.float64), decay)
        expected = (1.0 + np.arange(4)) ** -decay
        np.testing.assert_allclose(np.asarray(squared)[:, 0, 0], expected, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(squared)[:, 0, 1], expected, rtol=1e-6)
    print("PASS scheduled last layer: the states of one step are fitted jointly")


if __name__ == "__main__":
    test_unit_rows_with_bias_and_the_rollout_shapes()
    test_fit_rows_give_the_smallest_joint_change()
    test_fast_layer_lowers_a_state_seen_again_in_the_rollout()
    test_scheduled_layer_follows_the_schedule_exactly()
    test_scheduled_layer_fits_the_states_of_one_step_jointly()
