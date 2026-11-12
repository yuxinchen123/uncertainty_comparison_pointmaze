"""The options of the random-network-distillation bonus added for batch 2 of the PointMaze and
AntMaze study (2026-11-12): the bound of the input clip, input statistics frozen after the
priming rollouts, random Fourier features read by both networks, the scheduled last layer fitted
on random Fourier features, and a reward divisor held after a fraction of the run.

Each option is checked for the exact property it is built to have. That the options at their
defaults change nothing, and that one policy never moves another, is checked for all options
together in `test_rnd_bonus_options.py`.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "agents"))
sys.path.insert(0, str(ROOT / "tests" / "bonuses"))
from exploration_platform import F32  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import (bonus_options_record,  # noqa: E402
                                                   family_configuration, make_bonus)
from exploration_platform.bonuses.rnd.config import RNDConfig  # noqa: E402
from exploration_platform.bonuses.rnd.networks import (fourier_features,  # noqa: E402
                                                       init_fourier_features,
                                                       predictor_features, target_features,
                                                       unit_length, whiten)
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.statistics import rms_init, rms_update  # noqa: E402
from exploration_platform.training import maze_grid  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.train_step import intrinsic_divisor  # noqa: E402
from test_checkpoint_resume import check_resume  # noqa: E402
from test_rnd_bonus_options import (SHORT_EPISODES, SMALL, assert_same_arrays,  # noqa: E402
                                    family, random_observations, rnd, same_bytes, state_arrays,
                                    train)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# the first draws of the random Fourier features for base seed 0 and seed index 0: the four
# frequencies of feature 0 (length scale 1) and the phases of features 0 and 1, as drawn on
# 2026-11-12. A change of these numbers means the two streams were given other keys
PINNED_FREQUENCIES = [0.2209448218345642, 0.1878988891839981, -0.27810996770858765,
                      0.9456480741500854]
PINNED_PHASES = [1.169372797012329, 2.4445204734802246]


def train_each(cfg, bonus, iterations: int, planned: int):
    """Prime and train the first `iterations` iterations of a run planned for `planned`; returns
    (trainer, the state after every iteration, the metrics of every iteration) as host arrays."""
    runner = Runner(cfg, bonus=bonus)
    state = runner.prime(runner.init_state(0))
    states, metrics = [], []
    for k in range(1, iterations + 1):
        state, one = runner.iterate(state, runner.lr_argument(k, planned))
        states.append(jax.device_get(state))
        metrics.append(jax.device_get(one))
    return runner, states, metrics


def assert_refused(options: dict, message: str) -> None:
    """The family must refuse these options with a message that holds the given words."""
    try:
        make_bonus("rnd_next_state", options)
        raise AssertionError(f"accepted: {options}")
    except ValueError as error:
        assert message in str(error), (message, str(error))


def test_options_are_checked_and_what_they_imply_is_recorded():
    """The defaults are the bonus as it was; values are read in the option's type; random Fourier
    features set frozen statistics and no clip in the record; the two slower predictor rates of
    the batch are accepted; values and combinations that are not built are refused."""
    record = bonus_options_record("rnd_next_state")
    assert {name: record[name] for name in (
        "input_clip", "input_statistics", "input_features", "fourier_feature_count",
        "fourier_length_scale", "scheduled_last_layer_features", "divisor_hold_fraction")} == {
        "input_clip": 5.0, "input_statistics": "running", "input_features": "raw",
        "fourier_feature_count": 256, "fourier_length_scale": 1.0,
        "scheduled_last_layer_features": "body", "divisor_hold_fraction": 0.0}

    # text from a command line; statistics named as running are replaced by what the features need
    read = family_configuration("rnd_next_state", {
        "input_features": "random_fourier", "fourier_feature_count": "512",
        "fourier_length_scale": "0.5", "input_statistics": "running"})
    assert read == RNDConfig(input_features="random_fourier", fourier_feature_count=512,
                             fourier_length_scale=0.5, input_statistics="frozen_after_priming",
                             input_clip=0.0)
    assert isinstance(read.fourier_feature_count, int)
    assert family_configuration("rnd_next_state", {"input_clip": "10"}).input_clip == 10.0
    record = bonus_options_record("rnd_next_state", {
        "scheduled_last_layer_decay": 0.5, "scheduled_last_layer_features": "random_fourier",
        "fourier_feature_count": 512})
    assert (record["input_statistics"], record["input_clip"], record["even_start"],
            record["reward_divisor"]) == ("frozen_after_priming", 0.0, True, "none")

    # the slower predictor of the batch needs no new option
    cfg = PPOConfig(**SMALL)
    for multiple in (0.0625, 0.125):
        slower = family(cfg, predictor_learning_rate_multiple=str(multiple))
        assert slower.own_learning_rate == multiple * 1e-4

    for options, message in (
            ({"input_clip": -1}, "input_clip must be zero"),
            ({"input_statistics": "frozen"}, "unknown input_statistics"),
            ({"input_features": "fourier"}, "unknown input_features"),
            ({"scheduled_last_layer_features": "fourier"}, "unknown scheduled_last_layer_features"),
            ({"input_features": "random_fourier", "fourier_feature_count": 0}, "at least 1"),
            ({"input_features": "random_fourier", "fourier_length_scale": 0}, "must be positive"),
            ({"scheduled_last_layer_features": "random_fourier"}, "above zero"),
            ({"input_features": "random_fourier", "scheduled_last_layer_decay": 0.5},
             "built for the trained predictor"),
            ({"input_features": "random_fourier", "input_clip": 10}, "no clip"),
            ({"fourier_feature_count": 512}, "read only with"),
            ({"fourier_length_scale": 0.5}, "read only with")):
        assert_refused(options, message)
    print("PASS the options of batch 2 are read, checked, and recorded with what they imply")


def test_input_clip_bounds_the_whitened_input():
    """Clip 0 is the whitening with no clip, clip 10 bounds at +-10, the default at +-5; the
    starting bonus of an input beyond 5 differs between clip 5 and clip 10; no draw moves."""
    cfg = PPOConfig(**SMALL)
    C = cfg.n_copies
    values = np.array([-12.0, -7.0, 3.0, 8.0], np.float32)
    observations = jnp.broadcast_to(jnp.asarray(values), (C, 1, 4))

    # fresh statistics (mean 0, variance 1): the whitened input is the observation itself
    fresh = rms_init(C, 4)
    np.testing.assert_array_equal(np.asarray(whiten(observations, fresh))[0, 0], [-5, -5, 3, 5])
    np.testing.assert_array_equal(np.asarray(whiten(observations, fresh, 10.0))[0, 0],
                                  [-10, -7, 3, 8])
    np.testing.assert_array_equal(np.asarray(whiten(observations, fresh, 0.0))[0, 0], values)
    # statistics of a batch: clip 0 is (observation - mean) / standard deviation, nothing else
    moved = rms_update(fresh, random_observations(cfg, 64, seed=8) * 3 + 1)
    mean = np.asarray(moved.mean, np.float32)[:, None, :]
    deviation = np.sqrt(np.asarray(moved.var) + 1e-8).astype(np.float32)[:, None, :]
    unclipped = (np.asarray(observations) - mean) / deviation
    np.testing.assert_allclose(np.asarray(whiten(observations, moved, 0.0)), unclipped, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(whiten(observations, moved, 2.0)),
                               np.clip(unclipped, -2.0, 2.0), rtol=1e-6)

    # the starting bonus along the first coordinate, at 5, 8, 9, 12 and 15 with the rest at 0
    along = np.zeros((C, 5, 4), np.float32)
    along[:, :, 0] = [5.0, 8.0, 9.0, 12.0, 15.0]
    along = jnp.asarray(along)
    control, wider, unbounded = family(cfg), family(cfg, input_clip=10), family(cfg, input_clip=0)
    params, state = control.init()
    for other in (wider, unbounded):
        other_params, other_state = other.init()
        assert same_bytes(params, other_params) and same_bytes(state, other_state)
        assert same_bytes(control.constants, other.constants)
    at_5 = np.asarray(control.score_alone(params, state, along))
    at_10 = np.asarray(wider.score_alone(params, state, along))
    at_none = np.asarray(unbounded.score_alone(params, state, along))
    same = lambda one, other: np.testing.assert_allclose(one, other, rtol=1e-5)
    apart = lambda one, other: (np.abs(one - other) > 0.1 * np.abs(other)).all()
    # clip 5: every input beyond 5 has the bonus of 5
    for column in (1, 2, 3, 4):
        same(at_5[:, column], at_5[:, 0])
    # clip 10: 8 and 9 differ from each other and from clip 5; 12 and 15 have the bonus of 10
    same(at_10[:, 0], at_5[:, 0])
    assert apart(at_10[:, 1], at_5[:, 1]) and apart(at_10[:, 2], at_10[:, 1])
    same(at_10[:, 3], at_10[:, 4])
    # no clip: equal to clip 10 up to 9, and 12 and 15 differ
    same(at_none[:, :3], at_10[:, :3])
    assert apart(at_none[:, 3], at_10[:, 3]) and apart(at_none[:, 4], at_none[:, 3])
    # neither network has a bias at the start, so an input 3 times as large has 9 times the
    # bonus: with no clip the starting bonus keeps rising with the distance from the mean
    same(at_none[:, 4], 9.0 * at_none[:, 0])
    print("PASS input clip: 0 is no clip, 10 bounds at +-10; the starting bonus beyond 5 "
          f"differs (at 9: {at_5[0, 2]:.3g} with clip 5, {at_10[0, 2]:.3g} with clip 10)")


def test_input_statistics_frozen_after_priming():
    """Frozen input statistics: priming moves them exactly as the control's, training leaves them
    bit for bit where priming left them, the update batch is whitened with them, and zero
    priming rollouts are refused."""
    cfg = PPOConfig(**SMALL)
    frozen = rnd(input_statistics="frozen_after_priming", input_clip=0)
    _, primed, state, metrics = train(cfg, frozen)
    _, control_primed, control_state, control_metrics = train(cfg, "rnd_next_state")

    # priming: the statistics of the priming rollouts, the same as the control's
    statistics = primed.bonus_state["obs_rms"]
    np.testing.assert_allclose(statistics.count,
                               1e-4 + cfg.prime_iterations * cfg.num_steps * cfg.n_envs)
    assert same_bytes(statistics, control_primed.bonus_state["obs_rms"])
    assert not same_bytes(statistics, jax.device_get(rms_init(cfg.n_copies, 4)))
    # training: not moved, while the control's move
    assert same_bytes(statistics, state.bonus_state["obs_rms"])
    assert not same_bytes(control_primed.bonus_state["obs_rms"],
                          control_state.bonus_state["obs_rms"])
    assert all(np.isfinite(one["rint_mean"]).all() for one in metrics)
    # the predictor is trained
    assert not same_bytes(primed.bonus_params, state.bonus_params)

    # the hooks alone: priming moves the statistics, a training rollout returns the same
    # statistics and whitens the update batch with them
    alone = family(cfg, input_statistics="frozen_after_priming", input_clip=20)
    params, start = alone.init()
    rows = cfg.num_steps * cfg.n_envs
    after_priming = alone.prime(params, start, random_observations(cfg, rows, seed=9) * 2)
    assert not same_bytes(after_priming["obs_rms"], start["obs_rms"])
    rollout = random_observations(cfg, rows, seed=10) * 40
    new_state, reward, extra = alone.post_rollout(params, after_priming, rollout, None)
    assert same_bytes(new_state["obs_rms"], after_priming["obs_rms"])
    expected = whiten(rollout, after_priming["obs_rms"], 20.0)
    np.testing.assert_array_equal(np.asarray(extra["rnd_input"]), np.asarray(expected))
    assert float(np.abs(np.asarray(expected)).max()) == 20.0
    np.testing.assert_allclose(np.asarray(reward),
                               np.asarray(alone.score_alone(params, after_priming, rollout)),
                               rtol=1e-6)

    # with no priming rollout there is nothing to freeze
    try:
        Runner(replace(cfg, prime_iterations=0), bonus=frozen)
        raise AssertionError("frozen input statistics were accepted with no priming rollout")
    except ValueError as error:
        assert "priming rollout" in str(error)
    print("PASS frozen input statistics: moved by priming only, bit-identical through training")


def test_fourier_features_have_the_gaussian_inner_product():
    """For D = 16384 features the inner product of the features of two inputs at distance d is
    exp(-d^2 / (2 l^2)) within 0.04 (five times 1 / sqrt(D), the size of the error of D random
    features), for l = 0.5, 1 and 2 and d from 0 to 5 l. For the batch's D = 512 the inner
    products of inputs 5 l apart and more have root mean square 1 / sqrt(512) = 0.044."""
    count, copies = 16384, 2
    rng = np.random.default_rng(4)
    largest = 0.0
    for length_scale in (0.5, 1.0, 2.0):
        fourier = init_fourier_features(
            RNDConfig(fourier_feature_count=count, fourier_length_scale=length_scale),
            copies, 0, [0, 1])
        # a base input per copy, and inputs at the given distances from it in random directions
        distances = length_scale * np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
        base = rng.uniform(-3, 3, (copies, 1, 4))
        direction = rng.normal(size=(copies, len(distances), 4))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        inputs = base + distances[None, :, None] * direction
        features = np.asarray(fourier_features(fourier, jnp.asarray(inputs, F32)), np.float64)
        at_base = np.asarray(fourier_features(fourier, jnp.asarray(base, F32)), np.float64)
        assert features.shape == (copies, len(distances), count)
        assert np.abs(features).max() <= np.sqrt(2.0 / count) * (1 + 1e-6)
        inner = (features * at_base).sum(axis=-1)
        expected = np.exp(-distances ** 2 / (2 * length_scale ** 2))
        np.testing.assert_allclose(inner, np.broadcast_to(expected, inner.shape), atol=0.04)
        largest = max(largest, float(np.abs(inner - expected).max()))
    # 512 features: 64 pairs of inputs per copy, the two inputs of a pair 5 to 10 length scales
    # apart, where exp(-d^2 / (2 l^2)) is below 4e-6; what is left is the error of 512 features
    count, pairs, length_scale = 512, 64, 0.5
    fourier = init_fourier_features(
        RNDConfig(fourier_feature_count=count, fourier_length_scale=length_scale),
        copies, 0, [0, 1])
    one = rng.uniform(-10, 10, (copies, pairs, 4))
    direction = rng.normal(size=(copies, pairs, 4))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    other = one + length_scale * rng.uniform(5, 10, (copies, pairs, 1)) * direction
    inner = (np.asarray(fourier_features(fourier, jnp.asarray(one, F32)), np.float64)
             * np.asarray(fourier_features(fourier, jnp.asarray(other, F32)), np.float64)).sum(-1)
    spread = float(np.sqrt((inner ** 2).mean()))
    assert abs(spread * np.sqrt(count) - 1.0) < 0.25, spread
    print("PASS random Fourier features: inner product exp(-d^2 / (2 l^2)) within 0.04 for "
          f"D = 16384 (largest difference {largest:.4f}); for D = 512 the inner products of "
          f"far inputs have root mean square {spread:.4f}")


def test_fourier_draws_are_keyed_by_seed_index_and_feature():
    """The frequencies and phases of a policy depend on its seed index only: the same in
    programs with different copy counts, different for other seed indices and base seeds; a
    larger feature count appends features; the length scale rescales the same normal draws."""
    config = RNDConfig(fourier_feature_count=8, fourier_length_scale=0.5)
    few = jax.device_get(init_fourier_features(config, 3, 0, [0, 1, 2]))
    many = jax.device_get(init_fourier_features(config, 5, 0, [2, 7, 0, 1, 9]))
    assert few["frequencies"].shape == (3, 4, 8) and few["phases"].shape == (3, 8)
    assert few["frequencies"].dtype == np.float32 and few["phases"].dtype == np.float32
    for name in ("frequencies", "phases"):
        # seed index 2 is copy 2 of the first program and copy 0 of the second, and so on
        for one, other in ((2, 0), (0, 2), (1, 3)):
            np.testing.assert_array_equal(few[name][one], many[name][other])
        assert (few[name][0] != few[name][1]).all() and (many[name][1] != many[name][4]).all()
    other_base = jax.device_get(init_fourier_features(config, 3, 1, [0, 1, 2]))
    assert (other_base["frequencies"] != few["frequencies"]).all()
    assert (other_base["phases"] != few["phases"]).all()
    # the two streams are not one stream read twice
    assert not np.allclose(few["frequencies"][:, 0, :], few["phases"])

    # more features: the first 8 of 32 are the 8; another length scale: the same normals
    more = jax.device_get(init_fourier_features(replace(config, fourier_feature_count=32),
                                                3, 0, [0, 1, 2]))
    np.testing.assert_allclose(more["frequencies"][:, :, :8], few["frequencies"], rtol=1e-6)
    np.testing.assert_allclose(more["phases"][:, :8], few["phases"], rtol=1e-6)
    wide = jax.device_get(init_fourier_features(replace(config, fourier_length_scale=2.0),
                                                3, 0, [0, 1, 2]))
    np.testing.assert_allclose(wide["frequencies"] * 2.0, few["frequencies"] * 0.5, rtol=1e-6)
    np.testing.assert_array_equal(wide["phases"], few["phases"])

    # many draws: frequencies with standard deviation 1 / l and mean 0, phases over [0, 2 pi)
    large = jax.device_get(init_fourier_features(
        RNDConfig(fourier_feature_count=4096, fourier_length_scale=0.5), 1, 0, [0]))
    assert abs(large["frequencies"].std() - 2.0) < 0.06 and abs(large["frequencies"].mean()) < 0.08
    phases = large["phases"]
    assert phases.min() >= 0 and phases.max() <= np.float32(2 * np.pi)
    assert abs(phases.mean() - np.pi) < 0.15 and phases.min() < 0.1 and phases.max() > 6.18

    # the first draws, pinned
    first = jax.device_get(init_fourier_features(RNDConfig(fourier_feature_count=2), 1, 0, [0]))
    np.testing.assert_allclose(first["frequencies"][0, :, 0], PINNED_FREQUENCIES, rtol=1e-5)
    np.testing.assert_allclose(first["phases"][0], PINNED_PHASES, rtol=1e-5)
    print("PASS random Fourier draws: keyed by seed index and feature, one stream each")


def test_random_fourier_features_as_the_input_of_both_networks():
    """Both networks read the features of the whitened input (frozen statistics, no clip): the
    bonus is their half squared error on the features; the first layers have the features'
    width and every later layer is the control's own draw; saved constants replace the draw."""
    cfg = PPOConfig(**SMALL)
    C, count = cfg.n_copies, 64
    options = dict(input_features="random_fourier", fourier_feature_count=count,
                   fourier_length_scale=0.5)
    control, local = family(cfg), family(cfg, **options)
    control_params, _ = control.init()
    params, start = local.init()

    # the constants and the draws
    assert sorted(control.constants) == ["target"]
    assert sorted(local.constants) == ["fourier", "target"]
    drawn = init_fourier_features(family_configuration("rnd_next_state", options), C,
                                  cfg.base_seed, list(range(C)))
    assert same_bytes(local.constants["fourier"], drawn)
    assert params["predictor"]["W0"].shape == (C, count, 256)
    assert local.constants["target"]["W0"].shape == (C, count, 256)
    for name in ("b0", "W1", "b1", "W2", "b2"):
        np.testing.assert_array_equal(np.asarray(params["predictor"][name]),
                                      np.asarray(control_params["predictor"][name]))
    for name in ("b0", "W1", "b1"):
        np.testing.assert_array_equal(np.asarray(local.constants["target"][name]),
                                      np.asarray(control.constants["target"][name]))

    # the bonus of inputs far beyond the default clip, after one priming rollout
    rows = cfg.num_steps * cfg.n_envs
    state = local.prime(params, start, random_observations(cfg, rows, seed=6))
    observations = random_observations(cfg, 20, seed=5) * 4
    features = fourier_features(local.constants["fourier"],
                                whiten(observations, state["obs_rms"], 0.0))
    error = (predictor_features(params["predictor"], features)
             - target_features(local.constants["target"], features))
    np.testing.assert_allclose(np.asarray(local.score_alone(params, state, observations)),
                               0.5 * np.asarray((error ** 2).sum(-1)), rtol=1e-5)

    # a training rollout: the statistics stay, and the update batch holds the features
    rollout = random_observations(cfg, rows, seed=7) * 4
    new_state, reward, extra = local.post_rollout(params, state, rollout, None)
    assert same_bytes(new_state["obs_rms"], state["obs_rms"])
    assert extra["rnd_input"].shape == (C, rows, count)
    np.testing.assert_allclose(
        np.asarray(extra["rnd_input"]),
        np.asarray(fourier_features(local.constants["fourier"],
                                    whiten(rollout, state["obs_rms"], 0.0))), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(reward),
                               np.asarray(local.score_alone(params, state, rollout)), rtol=1e-6)

    # constants from a checkpoint are what the program holds: with frequencies and phases of
    # zero every feature is sqrt(2 / D) at every input
    saved = {"target": local.constants["target"],
             "fourier": {"frequencies": np.zeros((C, 4, count), np.float32),
                         "phases": np.zeros((C, count), np.float32)}}
    restored = rnd(**options)(cfg, EnvConfig(), C, cfg.base_seed, list(range(C)),
                              saved_constants=saved)
    _, _, extra = restored.post_rollout(params, state, rollout, None)
    np.testing.assert_allclose(np.asarray(extra["rnd_input"]), np.sqrt(2.0 / count), rtol=1e-6)
    try:
        rnd(**options)(cfg, EnvConfig(), C, cfg.base_seed, list(range(C)),
                       saved_constants={"target": local.constants["target"]})
        raise AssertionError("saved constants without the Fourier features were accepted")
    except ValueError as error:
        assert "shapes" in str(error)

    # in training: the predictor is trained, the statistics stay where priming left them
    _, primed, final, metrics = train(cfg, rnd(**options))
    assert same_bytes(primed.bonus_state["obs_rms"], final.bonus_state["obs_rms"])
    assert not same_bytes(primed.bonus_params, final.bonus_params)
    assert all(np.isfinite(one["rint_mean"]).all() and (one["rint_mean"] > 0).all()
               for one in metrics)
    print("PASS random Fourier features as the input: bonus on the features, later layers "
          "keep the control's draws, saved constants used, statistics frozen in training")


def test_scheduled_last_layer_on_random_fourier_features():
    """Scheduled last layer on random Fourier features: one input visited k times has squared
    error (1 + k)^(-a) for a = 0.5 and 1; an input 5 length scales away moves only by the
    overlap of the two inputs' features, which leaves its squared error above 0.99 for a large
    D; nothing is trainable, there is no predictor body, and the target is the control's."""
    visits = 5
    control = family(PPOConfig(**SMALL))

    def fitted_and_far(copies: int, count: int, length_scale: float, decay: float):
        """Visit one input per policy `visits` times; returns the squared errors of the visits
        [C, visits], then of the visited input and of an input 5 length scales away [C, 2],
        the overlap of their unit-length feature rows [C], and that of their targets [C]."""
        cfg = PPOConfig(**{**SMALL, "n_copies": copies, "n_envs": 1, "num_steps": visits})
        local = family(cfg, scheduled_last_layer_decay=decay, fourier_feature_count=count,
                       scheduled_last_layer_features="random_fourier",
                       fourier_length_scale=length_scale)
        assert local.reward_divisor == "none" and local.rollout_step is None
        params, state = local.init()
        assert params == {} and state["output_layer"].dtype == jnp.float64
        assert state["output_layer"].shape == (copies, count, 128)
        assert sorted(local.constants) == ["fourier", "target"]
        assert_same_arrays(
            {k: np.asarray(v)[:copies] for k, v in control.constants["target"].items()},
            {k: np.asarray(v) for k, v in local.constants["target"].items()}, "target")

        # fresh statistics, so an observation is its own whitened input: the visited input, and
        # one at distance 5 l from it in a random direction
        rng = np.random.default_rng(11)
        visited = rng.uniform(-5, 5, (copies, 1, 4))
        direction = rng.normal(size=(copies, 1, 4))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        pair = jnp.asarray(np.concatenate([visited, visited + 5 * length_scale * direction],
                                          axis=1), F32)
        np.testing.assert_allclose(np.asarray(local.score_alone(params, state, pair)), 1.0,
                                   rtol=1e-6)
        rollout = jnp.repeat(pair[:, :1], visits, axis=1)
        new_state, reward, extra = local.post_rollout(params, state, rollout, None)
        assert extra == {} and float(local.loss(params, {})) == 0.0
        assert same_bytes(new_state["obs_rms"], state["obs_rms"])
        after = np.asarray(local.score_alone(params, new_state, pair))
        # a second rollout continues from the kept layer
        _, again, _ = local.post_rollout(params, new_state, rollout, None)
        np.testing.assert_allclose(np.asarray(again)[:, -1], (2.0 * visits) ** -decay, rtol=1e-6)

        x = whiten(pair, state["obs_rms"], 0.0)
        rows = np.asarray(unit_length(
            fourier_features(local.constants["fourier"], x).astype(jnp.float64)))
        targets = np.asarray(unit_length(
            target_features(local.constants["target"], x).astype(jnp.float64)))
        return (np.asarray(reward), after, (rows[:, 0] * rows[:, 1]).sum(-1),
                (targets[:, 0] * targets[:, 1]).sum(-1))

    lines = []
    for decay in (0.5, 1.0):
        # the batch's 512 features: the schedule at the visited input; the input 5 l away is
        # where the overlap of the features puts it. After k visits the layer predicts
        # (1 - (1 + k)^(-a / 2)) x overlap x the visited input's target there
        reward, after, overlap, target_overlap = fitted_and_far(3, 512, 1.0, decay)
        np.testing.assert_allclose(reward, np.tile((1.0 + np.arange(visits)) ** -decay, (3, 1)),
                                   rtol=1e-6)
        np.testing.assert_allclose(after[:, 0], (1.0 + visits) ** -decay, rtol=1e-6)
        fitted = 1.0 - (1.0 + visits) ** (-decay / 2)
        expected = 1.0 - 2.0 * fitted * overlap * target_overlap + (fitted * overlap) ** 2
        np.testing.assert_allclose(after[:, 1], expected, rtol=1e-6)
        assert (np.abs(overlap) < 0.25).all(), overlap
        lines.append(f"a = {decay}, D = 512: overlaps {np.round(overlap, 4).tolist()}, "
                     f"squared error 5 l away {np.round(after[:, 1], 4).tolist()}")

        # many features: the overlap is exp(-12.5) within about 1 / sqrt(D) = 0.002, and the
        # input 5 l away keeps a squared error above 0.99
        for length_scale in (0.5, 2.0):
            reward, after, overlap, _ = fitted_and_far(1, 262144, length_scale, decay)
            np.testing.assert_allclose(after[:, 0], (1.0 + visits) ** -decay, rtol=1e-6)
            assert abs(overlap[0]) < 0.01, overlap
            assert 0.99 < after[0, 1] < 1.01, after
            lines.append(f"a = {decay}, D = 262144, l = {length_scale}: overlap "
                         f"{overlap[0]:.5f}, squared error 5 l away {after[0, 1]:.5f}")

    # in training: nothing of the bonus is trained, and the reward is at most about 1
    cfg = PPOConfig(**SMALL)
    runner, primed, state, metrics = train(cfg, rnd(
        scheduled_last_layer_decay=0.75, scheduled_last_layer_features="random_fourier",
        fourier_feature_count=512))
    assert state.bonus_params == {} and runner.bonus.reward_divisor == "none"
    assert np.asarray(state.bonus_state["output_layer"]).any()
    assert same_bytes(primed.bonus_state["obs_rms"], state.bonus_state["obs_rms"])
    assert all((one["rint_mean"] > 0).all() and (one["rint_mean"] < 1.05).all()
               for one in metrics)
    print("PASS scheduled last layer on random Fourier features: (1 + k)^(-a) after k visits; "
          "an input 5 l away moves only by the features' overlap\n  " + "\n  ".join(lines))


def test_divisor_held_after_a_fraction_of_the_run():
    """A divisor held after 3/8 of 8 planned iterations: the first 3 iterations are the
    control's bit for bit; from the 4th on the divisor's statistics are those the 3rd left, bit
    for bit; the learner's reward is the bonus over the held value; the maze-grid record shows
    the held value; values that are not built are refused."""
    planned, fraction, hold = 8, 0.375, 3
    cfg = PPOConfig(**SMALL, total_iterations=planned, diagnostics=True)
    C = cfg.n_copies
    bonus = rnd(reward_divisor="held_after_fraction", divisor_hold_fraction=fraction,
                record_bonus_on_maze_grid=True)
    _, control, _ = train_each(cfg, "rnd_next_state", planned, planned)
    runner, held, metrics = train_each(cfg, bonus, planned, planned)
    assert runner.bonus.divisor_hold_fraction == fraction

    # iterations numbered 0, 1 and 2 are below 3/8 x 8 = 3: the running divisor of the control
    for index in range(hold):
        assert_same_arrays(state_arrays(control[index]),
                           state_arrays(held[index], without=("maze_grid_visits",)),
                           f"iteration {index + 1}")
    # from iteration number 3 on: the statistics and the filter value of iteration number 2
    last_moved = held[hold - 1].agent_state
    for index in range(hold, planned):
        for name in ("int_rms", "int_filter"):
            assert same_bytes(held[index].agent_state[name], last_moved[name]), (index, name)
        assert not same_bytes(control[index].agent_state["int_rms"], last_moved["int_rms"])

    # the learner's reward: the bonus over the divisor of the iteration's own statistics before
    # the hold, over the held value afterwards
    held_value = np.asarray(intrinsic_divisor("held_after_fraction", last_moved["int_rms"], C))
    np.testing.assert_allclose(held_value,
                               np.sqrt(np.asarray(last_moved["int_rms"].var)[:, 0] + 1e-8),
                               rtol=1e-6)
    for index, one in enumerate(metrics):
        divisor = held_value if index >= hold else np.asarray(intrinsic_divisor(
            "held_after_fraction", held[index].agent_state["int_rms"], C))
        np.testing.assert_array_equal(one["diagnostics"]["int_std"], divisor)
        np.testing.assert_allclose(one["diagnostics"]["rint_hat_mean"],
                                   one["rint_mean"] / divisor, rtol=1e-5)
    assert (np.asarray(metrics[0]["diagnostics"]["int_std"]) != held_value).all()

    # the maze-grid record: the running value before the hold, the held value afterwards
    probe = maze_grid.build_probe(runner.cfg, runner.env, runner.bonus, runner.layout)
    for index in (0, hold - 1, hold, planned - 1):
        fields = maze_grid.read_window(held[index], probe)
        expected = np.asarray(intrinsic_divisor(
            "held_after_fraction", held[min(index, hold - 1)].agent_state["int_rms"], C))
        np.testing.assert_allclose(fields["intrinsic_reward_divisor_per_copy"], expected,
                                   rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(fields["probe_intrinsic_reward_per_copy"]),
            np.asarray(fields["probe_bonus_per_copy"]) / expected[:, None], rtol=1e-4)

    # what is refused: fractions outside (0, 1), the fraction with another divisor, the
    # scheduled last layer with it, and a run that does not say how long it is planned to be
    held_options = {"reward_divisor": "held_after_fraction"}
    for options, message in (
            (held_options, "above 0 and below 1"),
            ({**held_options, "divisor_hold_fraction": 1}, "above 0 and below 1"),
            ({**held_options, "divisor_hold_fraction": -0.25}, "above 0 and below 1"),
            ({"divisor_hold_fraction": 0.5}, "read only with"),
            ({"reward_divisor": "frozen_after_priming", "divisor_hold_fraction": 0.5},
             "read only with"),
            ({**held_options, "divisor_hold_fraction": 0.5, "scheduled_last_layer_decay": 1},
             "no divisor")):
        assert_refused(options, message)
    try:
        Runner(replace(cfg, total_iterations=0), bonus=bonus)
        raise AssertionError("a held divisor was accepted without the run's planned length")
    except ValueError as error:
        assert "planned number of iterations" in str(error)
    print("PASS divisor held after a fraction of the run: the control's state before the hold, "
          "the statistics unchanged from it on, the reward is the bonus over the held value")


def test_saved_units_continue_exactly():
    """A unit saved and restored equals the uninterrupted unit bit for bit: with random Fourier
    features read by both networks (the checkpoint holds the frequencies and phases), with the
    scheduled last layer on them, with frozen input statistics under a wider clip, and with a
    divisor held from an iteration that lies after the save."""
    cfg = PPOConfig(**SMALL, episode_log=True)
    check_resume(cfg, SHORT_EPISODES,
                 rnd(input_features="random_fourier", fourier_feature_count=32,
                     fourier_length_scale=0.5, record_bonus_on_maze_grid=True),
                 total=4, stop_after=2)
    check_resume(cfg, SHORT_EPISODES,
                 rnd(scheduled_last_layer_decay=0.75, fourier_feature_count=32,
                     scheduled_last_layer_features="random_fourier"), total=4, stop_after=2)
    check_resume(cfg, SHORT_EPISODES,
                 rnd(input_statistics="frozen_after_priming", input_clip=20),
                 total=4, stop_after=2)
    # planned for 6 iterations and held from iteration number 3; saved after 2 iterations
    check_resume(replace(cfg, total_iterations=6), SHORT_EPISODES,
                 rnd(reward_divisor="held_after_fraction", divisor_hold_fraction=0.5),
                 total=6, stop_after=2)
    print("PASS saved units with the options of batch 2 continue exactly")


if __name__ == "__main__":
    test_options_are_checked_and_what_they_imply_is_recorded()
    test_input_clip_bounds_the_whitened_input()
    test_input_statistics_frozen_after_priming()
    test_fourier_features_have_the_gaussian_inner_product()
    test_fourier_draws_are_keyed_by_seed_index_and_feature()
    test_random_fourier_features_as_the_input_of_both_networks()
    test_scheduled_last_layer_on_random_fourier_features()
    test_divisor_held_after_a_fraction_of_the_run()
    test_saved_units_continue_exactly()
