"""The faster forms of the fast last layer's bonus against the reference form; run on an allocated compute node.

Each form equals last_layer.fast_layer_scores in exact arithmetic, so the float64 forms must agree
with it to float64 rounding, and the forms that read float32 values (plain errors, float32
products) within the stated tolerances.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.fast_layer_forms import (REFERENCE_FORM, FastLayerForm, chunk_steps_for,
                                                                chunked_fast_layer_scores, small_solve_for,
                                                                unrolled_fit_rows)
from exploration_platform.atari_pixels.training import PixelPPORND, fast_last_layer_bonus
from exploration_platform.bonuses.rnd.last_layer import fast_layer_scores, fit_rows, unit_rows_with_bias

# Largest relative difference allowed between a float64 form and the reference (float64 rounding
# through a few hundred sums); the forms that read float32 numbers have their own tolerances.
FLOAT64_TOLERANCE = 1e-9
PLAIN_ERROR_TOLERANCE = 1e-4
FLOAT32_PRODUCT_TOLERANCE = 1e-3
# The learner settings of the reference form, which the defaults no longer give.
REFERENCE_SETTINGS = {"fast_last_layer_chunk_rows": 0, "fast_last_layer_plain_errors": False,
                      "fast_last_layer_small_solve": "cholesky"}


def random_rollout(steps=8, copies=2, envs=8, features=6, outputs=5, seed=0, repeated=False):
    """Stand-in features after a ReLU, targets and an output layer, in the trainer's layout.

    With repeated, environments 0 and 1 of every step and step 2 and step 5 show the same state,
    so the systems have the nearly singular blocks that game starts produce.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    hidden = jax.nn.relu(jax.random.normal(keys[0], (steps, copies, envs, features), jnp.float32))
    targets = jax.random.normal(keys[1], (steps, copies, envs, outputs), jnp.float32)
    if repeated:
        # before: independent states ; after: environment 1 equals environment 0, step 5 equals step 2
        hidden = hidden.at[:, :, 1].set(hidden[:, :, 0]).at[5].set(hidden[2])
        targets = targets.at[:, :, 1].set(targets[:, :, 0]).at[5].set(targets[2])
    layer = {"weight": jax.random.normal(keys[2], (copies, features, outputs), jnp.float32) * 0.3,
             "bias": jax.random.normal(keys[3], (copies, outputs), jnp.float32) * 0.1}
    return hidden, targets, layer


def plain_errors(hidden, targets, layer):
    """The predictor's float32 output minus the target's, as the rollout stores it: [T, C, N, K]."""
    return jnp.einsum("tcnf,cfk->tcnk", hidden, layer["weight"]) + layer["bias"][None, :, None, :] - targets


def relative_difference(actual, expected):
    """Largest |actual - expected| / |expected| over all transitions."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float((np.abs(actual - expected) / np.abs(expected)).max())


def float64_reference(hidden, targets, layer, steps, group):
    """The reference form's bonus before the float32 cast: [T, C*groups, group] float64."""
    rows, length = unit_rows_with_bias(hidden)
    whole = jnp.concatenate([layer["weight"].astype(jnp.float64), layer["bias"].astype(jnp.float64)[:, None, :]], axis=1)
    count, copies, envs = hidden.shape[:3]
    shape = lambda values: values.reshape(count, copies * (envs // group), group, values.shape[-1])
    return fast_layer_scores(shape(rows), shape(length), shape(targets.astype(jnp.float64)),
                             jnp.repeat(whole, envs // group, axis=0),
                             jnp.repeat(jnp.asarray(steps, jnp.float64), envs // group)[:, None, None],
                             lambda error: 0.5 * (error * error).sum(axis=-1))


def float64_chunked(hidden, targets, layer, steps, group, form):
    """The chunked form's bonus before the float32 cast, from targets in float64: [T, C*groups, group]."""
    rows, length = unit_rows_with_bias(hidden)
    whole = jnp.concatenate([layer["weight"].astype(jnp.float64), layer["bias"].astype(jnp.float64)[:, None, :]], axis=1)
    right = jnp.einsum("tcnf,cfk->tcnk", rows, whole) - targets.astype(jnp.float64) / length
    count, copies, envs = hidden.shape[:3]
    shape = lambda values: values.reshape(count, copies * (envs // group), group, values.shape[-1])
    eta = jnp.repeat(jnp.asarray(steps, jnp.float64), envs // group)[:, None, None]
    return chunked_fast_layer_scores(shape(rows), shape(length), shape(right), eta, form)


def test_chunk_steps_follow_the_rows_per_chunk():
    """Steps per chunk: the whole rollout when it is small, one step for groups of more than 16
    environments, else the largest divisor of the rollout length that keeps the chunk within its rows."""
    # The defaults: whole rollouts of at most 512 rows, chunks of at most 128 rows.
    assert chunk_steps_for(128, 512, 128, 4) == 128
    assert chunk_steps_for(128, 512, 128, 16) == 8
    assert chunk_steps_for(128, 512, 128, 128) == 1
    # Without the whole-rollout rule.
    assert chunk_steps_for(512, 0, 128, 4) == 128
    assert chunk_steps_for(512, 0, 128, 16) == 32
    assert chunk_steps_for(512, 0, 128, 128) == 1
    assert chunk_steps_for(4096, 0, 128, 32) == 1
    assert chunk_steps_for(8, 0, 128, 16) == 1
    assert chunk_steps_for(512, 0, 3, 2) == 3
    assert chunk_steps_for(4, 0, 6, 1) == 3
    assert chunk_steps_for(1, 8, 4, 2) == 4


@pytest.mark.parametrize("envs", [1, 2, 4, 16])
def test_unrolled_fit_rows_equal_the_library_factorization(envs):
    """The written-out factorization gives the library's fit rows, also for two equal rows.

    The rows have more entries than there are rows, as in the trainer (513 entries, at most 128
    rows), so only equal rows make the system nearly singular. There the fit rows depend on
    float64 rounding at about 1e-7, the inverse ridge times the rounding of one entry.
    """
    rows, _ = unit_rows_with_bias(jax.nn.relu(jax.random.normal(jax.random.PRNGKey(envs), (3, 5, envs, envs + 7))))
    np.testing.assert_allclose(unrolled_fit_rows(rows), fit_rows(rows), rtol=1e-9, atol=1e-12)
    if envs > 1:
        same = rows.at[..., 1, :].set(rows[..., 0, :])
        np.testing.assert_allclose(unrolled_fit_rows(same), fit_rows(same), rtol=1e-5, atol=1e-7)
    if envs == 2:
        # two equal rows share the change: each fit row is half the row
        np.testing.assert_allclose(unrolled_fit_rows(same)[..., 0, :], same[..., 0, :] / 2, rtol=1e-6)


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("group,chunk", [(8, 1), (8, 2), (8, 8), (4, 4), (2, 8), (1, 8), (4, 3)])
@pytest.mark.parametrize("small_solve", ["cholesky", "unrolled"])
def test_chunked_form_equals_the_reference_to_float64_rounding(group, chunk, small_solve, repeated):
    """Every chunk length, with either factorization, gives the reference bonus in float64."""
    steps = 6 if chunk == 3 else 8
    # More entries per row than rows per group, as in the trainer.
    hidden, targets, layer = random_rollout(steps=steps, features=12, repeated=repeated)
    expected = float64_reference(hidden, targets, layer, (0.3, 0.9), group)
    form = FastLayerForm(chunk_steps=chunk, small_solve=small_solve)
    actual = float64_chunked(hidden, targets, layer, (0.3, 0.9), group, form)
    assert actual.shape == expected.shape and actual.dtype == jnp.float64
    assert relative_difference(actual, expected) < FLOAT64_TOLERANCE


@pytest.mark.parametrize("chunk", [1, 4, 8])
def test_one_state_repeated_at_step_one_half_gives_b_then_a_quarter_then_a_sixteenth(chunk):
    """The halving property of the reference holds in the chunked form."""
    hidden, targets, layer = random_rollout(steps=1, copies=1, envs=1)
    hidden = jnp.broadcast_to(hidden, (8, 1, 4, hidden.shape[-1]))
    targets = jnp.broadcast_to(targets, (8, 1, 4, targets.shape[-1]))
    form = FastLayerForm(chunk_steps=chunk)
    bonus = np.asarray(fast_last_layer_bonus(hidden, targets, layer, (0.5,), 4, form), np.float64)
    for step in range(1, 8):
        np.testing.assert_allclose(bonus[step], bonus[0] / 4 ** step, rtol=1e-5)


def test_plain_errors_and_float32_products_stay_within_their_tolerances():
    """The forms that read float32 numbers agree with the reference within the stated tolerances."""
    hidden, targets, layer = random_rollout(steps=8, repeated=True)
    expected = fast_last_layer_bonus(hidden, targets, layer, (0.3, 0.9), 4)
    errors = plain_errors(hidden, targets, layer)
    for chunk in (2, 8):
        plain = fast_last_layer_bonus(hidden, None, layer, (0.3, 0.9), 4,
                                      FastLayerForm(chunk_steps=chunk, plain_errors=True), errors)
        assert relative_difference(plain, expected) < PLAIN_ERROR_TOLERANCE
        for precision in ("changes", "changes_and_applications"):
            products = fast_last_layer_bonus(hidden, targets, layer, (0.3, 0.9), 4,
                                             FastLayerForm(chunk_steps=chunk, float32_products=precision))
            assert relative_difference(products, expected) < FLOAT32_PRODUCT_TOLERANCE


def test_chunked_form_keeps_the_groups_apart():
    """New data in the second group of four environments cannot reach the first group."""
    hidden, targets, layer = random_rollout()
    other_hidden, other_targets, _ = random_rollout(seed=1)
    changed_hidden = hidden.at[:, :, 4:].set(other_hidden[:, :, 4:])
    changed_targets = targets.at[:, :, 4:].set(other_targets[:, :, 4:])
    form = FastLayerForm(chunk_steps=4)
    expected = np.asarray(fast_last_layer_bonus(hidden, targets, layer, (0.2, 0.7), 4, form))
    actual = np.asarray(fast_last_layer_bonus(changed_hidden, changed_targets, layer, (0.2, 0.7), 4, form))
    np.testing.assert_array_equal(actual[:, :, :4], expected[:, :, :4])
    assert not np.array_equal(actual[:, :, 4:], expected[:, :, 4:])


def test_step_size_zero_gives_the_plain_bonus_in_the_chunked_form():
    """With step size zero no chunk changes anything: the bonus is the plain bonus."""
    hidden, targets, layer = random_rollout()
    plain = 0.5 * (np.asarray(plain_errors(hidden, targets, layer), np.float64) ** 2).sum(axis=-1)
    for chunk in (1, 8):
        bonus = fast_last_layer_bonus(hidden, targets, layer, (0.0, 0.0), 4, FastLayerForm(chunk_steps=chunk))
        np.testing.assert_allclose(bonus, plain, rtol=1e-5)


def test_configuration_of_the_execution_options():
    """The options are read only with the fast last layer on, and need the chunked form."""
    on = {"fast_last_layer_step": 0.1, "fast_last_layer_sharing_group": 16}
    assert PixelPPOConfig(fast_last_layer_chunk_rows=512, fast_last_layer_plain_errors=True,
                          fast_last_layer_float32_products="changes", fast_last_layer_small_solve="unrolled",
                          **on).fast_last_layer_enabled
    with pytest.raises(ValueError, match="'none', 'changes' or 'changes_and_applications'"):
        PixelPPOConfig(fast_last_layer_float32_products="all", **on)
    assert PixelPPOConfig(**REFERENCE_SETTINGS, **on).fast_last_layer_enabled
    # Off, the options are not read.
    PixelPPOConfig(fast_last_layer_chunk_rows=0, fast_last_layer_plain_errors=True)
    with pytest.raises(ValueError, match="need a positive fast_last_layer_chunk_rows"):
        PixelPPOConfig(fast_last_layer_chunk_rows=0, **on)
    with pytest.raises(ValueError, match="need a positive fast_last_layer_chunk_rows"):
        PixelPPOConfig(fast_last_layer_chunk_rows=0, fast_last_layer_plain_errors=False,
                       fast_last_layer_small_solve="unrolled", **on)
    with pytest.raises(ValueError, match="'cholesky', 'unrolled' or 'unrolled_for_small_groups'"):
        PixelPPOConfig(fast_last_layer_small_solve="direct", **on)
    with pytest.raises(ValueError, match="nonnegative integer"):
        PixelPPOConfig(fast_last_layer_chunk_rows=-1, **on)


def test_the_defaults_give_the_accepted_form_and_the_reference_stays_available():
    """With the fast last layer on, the defaults give the accepted form of each sharing group."""
    environment = VariedPixelEnvironment(128)
    expected = {4: FastLayerForm(chunk_steps=128, plain_errors=True, small_solve="unrolled"),
                16: FastLayerForm(chunk_steps=8, plain_errors=True, small_solve="unrolled"),
                128: FastLayerForm(chunk_steps=1, plain_errors=True, small_solve="cholesky")}
    for group, form in expected.items():
        on = {"fast_last_layer_step": 0.1, "fast_last_layer_sharing_group": group}
        assert PixelPPORND(PixelPPOConfig(**on), environment).fast_layer_form == form
        assert PixelPPORND(PixelPPOConfig(**REFERENCE_SETTINGS, **on), environment).fast_layer_form == REFERENCE_FORM
    assert PixelPPORND(PixelPPOConfig(), environment).fast_layer_form == REFERENCE_FORM


def test_the_factorization_by_group_size_writes_out_only_small_groups():
    """unrolled_for_small_groups writes out groups of up to 16 environments; larger ones use the library."""
    assert small_solve_for("unrolled_for_small_groups", 4) == "unrolled"
    assert small_solve_for("unrolled_for_small_groups", 16) == "unrolled"
    assert small_solve_for("unrolled_for_small_groups", 128) == "cholesky"
    assert small_solve_for("cholesky", 4) == "cholesky" and small_solve_for("unrolled", 128) == "unrolled"
    for group, expected in ((2, "unrolled"), (32, "cholesky")):
        configuration = PixelPPOConfig(n_envs=32, num_steps=4, num_minibatches=1, fast_last_layer_step=0.1,
                                       fast_last_layer_sharing_group=group, fast_last_layer_chunk_rows=1,
                                       fast_last_layer_whole_rollout_rows=0,
                                       fast_last_layer_small_solve="unrolled_for_small_groups")
        form = PixelPPORND(configuration, VariedPixelEnvironment(32)).fast_layer_form
        assert form == FastLayerForm(chunk_steps=1, plain_errors=True, small_solve=expected)


class VariedPixelEnvironment:
    """A deterministic transition source whose environments show different pixels at every step."""

    action_count = 18

    def __init__(self, num_envs):
        """Hold the fixed number of vector environments for each independent policy."""
        self.num_envs = num_envs

    def observe(self, counters):
        """Make four distinct uint8 frames from each environment's integer counter."""
        offsets = jnp.arange(4, dtype=jnp.int32)[None, None, :, None, None]
        return jnp.broadcast_to(((counters[..., None, None, None] * 9 + offsets) % 256).astype(jnp.uint8),
                                 counters.shape + (4, 84, 84))

    def reset(self, seeds):
        """Start environment n of every policy at counter 5 * n, so no two environments agree."""
        counters = jnp.broadcast_to(jnp.arange(self.num_envs, dtype=jnp.int32) * 5, (len(seeds), self.num_envs))
        return counters, self.observe(counters)

    def step(self, counters, actions):
        """Advance fixed pixels, ending an episode whenever a counter is a multiple of three."""
        counters = counters + 1
        done = counters % 3 == 0
        rewards = (actions % 2).astype(jnp.float32)
        return counters, self.observe(counters), rewards, done, {
            "episode_return": counters.astype(jnp.float32), "episode_done": done,
            "final_observation": self.observe(counters), "full_game_reset": done}


def test_trainer_with_the_chunked_form_and_plain_errors_matches_the_reference_trainer():
    """A whole update with the faster options gives the reference bonus and the same predictor bonus."""
    configuration = PixelPPOConfig(n_copies=2, n_envs=4, num_steps=4, num_minibatches=1, update_epochs=1,
                                   observation_warmup_iterations=1, total_timesteps=256,
                                   fast_last_layer_step=0.5, fast_last_layer_sharing_group=2,
                                   record_bonus_profile=True, **REFERENCE_SETTINGS)
    environment = VariedPixelEnvironment(4)
    reference = PixelPPORND(configuration, environment)
    assert reference.fast_layer_form == REFERENCE_FORM
    state = reference.warmup(reference.initialize((19, 37)))
    _, expected = jax.jit(reference.collect)(state)
    expected_result, expected_metrics = reference.train_iteration(state)
    # Chunks of two steps and of the whole rollout in float64, then the defaults (the accepted form).
    for options in ({"fast_last_layer_chunk_rows": 4, "fast_last_layer_whole_rollout_rows": 0},
                    {"fast_last_layer_chunk_rows": 8, "fast_last_layer_whole_rollout_rows": 0},
                    {name: getattr(PixelPPOConfig(), name) for name in REFERENCE_SETTINGS}):
        trainer = PixelPPORND(replace(configuration, **options), environment)
        _, rollout = jax.jit(trainer.collect)(state)
        tolerance = PLAIN_ERROR_TOLERANCE if trainer.fast_layer_form.plain_errors else 1e-6
        assert relative_difference(rollout.intrinsic_rewards, expected.intrinsic_rewards) < tolerance
        assert rollout.bonus_hidden is None and rollout.bonus_targets is None and rollout.bonus_errors is None
        np.testing.assert_array_equal(rollout.actions, expected.actions)
        result, metrics = trainer.train_iteration(state)
        assert set(metrics) == set(expected_metrics)
        for name, value in metrics.items():
            assert np.isfinite(np.asarray(value, np.float64)).all(), name
        if trainer.fast_layer_form.plain_errors:
            # The predictor's own bonus from stored plain errors is the plain trainer's bonus formula.
            np.testing.assert_allclose(metrics["predictor_bonus_mean"], expected_metrics["predictor_bonus_mean"],
                                       rtol=1e-6)
        else:
            np.testing.assert_array_equal(metrics["predictor_bonus_mean"], expected_metrics["predictor_bonus_mean"])
        assert jax.tree.structure(result) == jax.tree.structure(expected_result)
