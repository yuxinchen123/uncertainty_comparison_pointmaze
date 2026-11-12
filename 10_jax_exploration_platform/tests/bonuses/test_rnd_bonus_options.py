"""The options of the random-network-distillation bonus added for batch 1 of the PointMaze and
AntMaze study (2026-11-12): the start, the scale and the decay of the bonus.

Each option is checked for the exact property it is built to have, and all of them together for
what they must not change: options at their defaults leave the training state bit for bit as it
was, recording the bonus on the maze grid changes nothing in training, a saved unit continues
exactly, and one policy's training never moves another's.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "agents"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import (BONUS_REGISTRY, bonus_options_record,  # noqa: E402
                                                   family_configuration, make_bonus)
from exploration_platform.bonuses.rnd.config import RNDConfig  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.statistics import rms_init, rms_update  # noqa: E402
from exploration_platform.training import maze_grid  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.train_step import intrinsic_divisor  # noqa: E402
from test_checkpoint_resume import check_resume  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SMALL = dict(n_copies=3, n_envs=2, num_steps=16, prime_iterations=2,
             update_style="epoch_minibatch", clip_coef=0.1)
# episodes of at most 6 steps, so the environments of a policy are often in the same state
SHORT_EPISODES = EnvConfig(continuing_task=False, max_episode_steps=6)


def rnd(**options):
    """The random-network-distillation family with the given options changed."""
    return make_bonus("rnd_next_state", options)


def family(cfg, **options):
    """The family's functions bound to a configuration, without a trainer around them."""
    return rnd(**options)(cfg, EnvConfig(), cfg.n_copies, cfg.base_seed,
                          list(range(cfg.n_copies)))


def train(cfg, bonus, iterations=3, env_cfg=None):
    """Prime and train a few iterations; returns (trainer, the state after priming, the final
    state, the metrics of every iteration), states and metrics as host arrays."""
    runner = Runner(cfg, bonus=bonus, env_cfg=env_cfg)
    state = runner.prime(runner.init_state(0))
    primed = jax.device_get(state)
    metrics = []
    for k in range(1, iterations + 1):
        state, one = runner.iterate(state, runner.lr_argument(k, iterations))
        metrics.append(jax.device_get(one))
    return runner, primed, jax.device_get(state), metrics


def same_bytes(one, other) -> bool:
    """Whether two trees of arrays hold the same leaves, bit for bit."""
    return ([np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(one)]
            == [np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(other)])


def state_arrays(state, without=()) -> dict:
    """Every array of a training state by its path, except the named entries of the agent's state."""
    kept = state._replace(agent_state={name: value for name, value in state.agent_state.items()
                                       if name not in without})
    flat, _ = jax.tree_util.tree_flatten_with_path(kept)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def assert_same_arrays(one: dict, other: dict, what: str) -> None:
    """Two sets of named arrays must hold the same names and the same bytes."""
    assert one.keys() == other.keys(), what
    for name in one:
        assert one[name].tobytes() == other[name].tobytes(), f"{what}: {name} differs"


def random_observations(cfg, rows: int, seed: int = 0):
    """Random PointMaze observations [C, rows, 4]: positions and velocities within +-5."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-5, 5, (cfg.n_copies, rows, 4)).astype(np.float32))


def test_options_are_read_checked_and_recorded():
    """Names and values are checked; text from a command line is read in the option's type; the
    record holds every option; without options the registered preset itself is returned."""
    assert make_bonus("rnd_next_state") is BONUS_REGISTRY["rnd_next_state"]
    assert make_bonus("rnd_state", {}) is BONUS_REGISTRY["rnd_next_state"]
    assert family_configuration("rnd_next_state") == RNDConfig()
    assert family_configuration("none") is None and bonus_options_record("none") == {}
    read = family_configuration("rnd_next_state", {
        "even_start": "true", "fast_last_layer_step": "0.3", "reward_divisor": "none",
        "predictor_learning_rate_multiple": 4})
    assert read == RNDConfig(even_start=True, fast_last_layer_step=0.3, reward_divisor="none",
                             predictor_learning_rate_multiple=4.0)
    assert isinstance(read.predictor_learning_rate_multiple, float)

    record = bonus_options_record("rnd_next_state", {"scheduled_last_layer_decay": 0.5})
    assert set(record) == set(RNDConfig.__dataclass_fields__)
    # the scheduled last layer is defined with an even start and no divisor
    assert record["even_start"] is True and record["reward_divisor"] == "none"
    assert bonus_options_record("rnd_next_state")["reward_divisor"] == "running_std"

    refused = (("rnd_next_state", {"even_starts": True}, "unknown options"),
               ("rnd_next_state", {"even_start": "yes"}, "true or false"),
               ("rnd_next_state", {"fast_last_layer_step": "fast"}, "a number"),
               ("rnd_next_state", {"fast_last_layer_step": 1.5}, "between 0"),
               ("rnd_next_state", {"reward_divisor": "frozen"}, "unknown reward_divisor"),
               ("rnd_next_state", {"bonus_form": 3}, "a word"),
               ("rnd_next_state", {"scheduled_last_layer_decay": 1, "fast_last_layer_step": 1},
                "cannot be"),
               ("rnd_next_state", {"scheduled_last_layer_decay": 1,
                                   "reward_divisor": "frozen_after_priming"}, "no divisor"),
               ("none", {"even_start": True}, "has no options"),
               ("elliptical", {}, "unknown bonus"))
    for name, options, message in refused:
        try:
            make_bonus(name, options)
            raise AssertionError(f"accepted: {name} {options}")
        except ValueError as error:
            assert message in str(error), (message, str(error))
    print("PASS options are read in their types, checked, and recorded in full")


def test_options_at_their_defaults_change_nothing():
    """Every option given at its default value: the training state after three iterations is
    bit-identical to the state of the registered preset, in both update styles."""
    defaults = {"even_start": False, "reward_divisor": "running_std",
                "predictor_learning_rate_multiple": 0, "bonus_form": "half_squared_error",
                "fast_last_layer_step": "0", "scheduled_last_layer_decay": 0.0,
                "record_bonus_on_maze_grid": "false"}
    for style in ("epoch_minibatch", "full_batch"):
        cfg = PPOConfig(**{**SMALL, "update_style": style}, episode_log=True)
        _, _, plain, _ = train(cfg, "rnd_next_state", env_cfg=SHORT_EPISODES)
        _, _, with_defaults, _ = train(cfg, rnd(**defaults), env_cfg=SHORT_EPISODES)
        assert_same_arrays(state_arrays(plain), state_arrays(with_defaults), style)
    print("PASS options at their defaults: bit-identical training state in both update styles")


def test_recording_on_the_maze_grid_changes_nothing_in_training():
    """The diagnostic: the training state is bit-identical with it on; the visit counts add up
    and mark the cells the coverage map marks; the probes are scored as the bonus stands."""
    cfg = PPOConfig(**SMALL, track_coverage=True)
    iterations = 3
    _, _, plain, _ = train(cfg, "rnd_next_state", iterations)
    runner, _, recorded, _ = train(cfg, rnd(record_bonus_on_maze_grid=True), iterations)
    assert_same_arrays(state_arrays(plain), state_arrays(recorded, without=("maze_grid_visits",)),
                       "recording on the maze grid")

    # every state the rollouts reached is counted once, in the cell the coverage map marks
    visits = np.asarray(recorded.agent_state["maze_grid_visits"])
    assert visits.shape == (cfg.n_copies, runner.env.n_cells) and visits.dtype == np.int32
    assert (visits.sum(axis=1) == iterations * cfg.num_steps * cfg.n_envs).all()
    np.testing.assert_array_equal(visits > 0, np.asarray(recorded.visited))

    # one probe per open cell, at the cell's centre with zero velocity
    cells = maze_grid.open_cell_list(runner.env)
    probes = maze_grid.probe_observations(runner.env)
    assert len(cells) == int(np.asarray(runner.env.open_cells).sum()) == len(probes)
    assert (7, 1) in cells and (0, 0) not in cells
    np.testing.assert_array_equal(probes[cells.index((7, 1))], [-4.5, -3.0, 0.0, 0.0])
    index = np.asarray(runner.env.cell_index(jnp.asarray(probes)[None]))[0]
    np.testing.assert_array_equal(index, [row * runner.env.cols + col for row, col in cells])

    # the window's fields: the bonus of each probe as the bonus stands, the divisor, the
    # reward (bonus / divisor), and the visits of the open cells
    probe = maze_grid.build_probe(runner.cfg, runner.env, runner.bonus, runner.layout)
    fields = maze_grid.read_window(recorded, probe)
    assert tuple(fields) == maze_grid.WINDOW_FIELDS
    expected = np.asarray(runner.bonus.score_alone(
        recorded.bonus_params, recorded.bonus_state,
        jnp.broadcast_to(jnp.asarray(probes)[None], (cfg.n_copies,) + probes.shape)))
    np.testing.assert_allclose(fields["probe_bonus_per_copy"], expected, rtol=1e-5)
    divisor = np.sqrt(np.asarray(recorded.agent_state["int_rms"].var)[:, 0] + 1e-8)
    np.testing.assert_allclose(fields["intrinsic_reward_divisor_per_copy"], divisor, rtol=1e-5)
    np.testing.assert_allclose(fields["probe_intrinsic_reward_per_copy"],
                               expected / divisor[:, None], rtol=1e-4)
    np.testing.assert_array_equal(fields["cell_visits_per_copy"],
                                  visits[:, np.asarray(runner.env.open_cells)])
    cleared = maze_grid.clear_window(recorded.agent_state)
    assert not np.asarray(cleared["maze_grid_visits"]).any()
    assert cleared["int_rms"] is recorded.agent_state["int_rms"]
    print("PASS recording on the maze grid: training unchanged; visits and probes as defined")


def test_even_start_gives_every_input_the_same_first_bonus():
    """Even start: the first bonus at 100 random inputs is 0.5 (length 1) with largest over
    smallest equal to 1 within rounding; the control's ratio is well above 2. The draws of the
    hidden layers and of the target are the control's."""
    cfg = PPOConfig(**SMALL)
    observations = random_observations(cfg, 100)
    ratio = lambda values: float(values.max(axis=1).max() / values.min(axis=1).min())

    control = family(cfg)
    control_params, state = control.init()
    assert ratio(np.asarray(control.score_alone(control_params, state, observations))) > 2

    even = family(cfg, even_start=True)
    params, state = even.init()
    first = np.asarray(even.score_alone(params, state, observations))
    assert abs(ratio(first) - 1) < 1e-5, ratio(first)
    np.testing.assert_allclose(first, 0.5, rtol=1e-5)
    length = family(cfg, even_start=True, bonus_form="error_length")
    np.testing.assert_allclose(np.asarray(length.score_alone(params, state, observations)), 1.0,
                               rtol=1e-5)

    # the output layer starts at zero; no existing draw has moved
    assert not np.asarray(params["predictor"]["W2"]).any()
    assert not np.asarray(params["predictor"]["b2"]).any()
    for name in ("W0", "b0", "W1", "b1"):
        np.testing.assert_array_equal(np.asarray(params["predictor"][name]),
                                      np.asarray(control_params["predictor"][name]))
    assert_same_arrays({k: np.asarray(v) for k, v in control.constants["target"].items()},
                       {k: np.asarray(v) for k, v in even.constants["target"].items()}, "target")
    print(f"PASS even start: first bonus 0.5 at 100 inputs, largest over smallest {ratio(first)}")


def test_error_length_is_the_length_of_the_same_error():
    """Error length: the bonus is sqrt(2 x the control's bonus) at every input."""
    cfg = PPOConfig(**SMALL)
    observations = random_observations(cfg, 50, seed=1)
    control, length = family(cfg), family(cfg, bonus_form="error_length")
    params, state = control.init()
    np.testing.assert_allclose(
        np.asarray(length.score_alone(params, state, observations)),
        np.sqrt(2 * np.asarray(control.score_alone(params, state, observations))), rtol=1e-5)
    print("PASS error length: the square root of twice the half squared error")


def test_frozen_divisor_is_set_by_priming_and_does_not_move():
    """Fixed scale: the divisor after priming is the running standard deviation of the priming
    rollouts' discounted bonus (recomputed here in NumPy for one priming rollout), and three
    training iterations leave it bit for bit where it was."""
    cfg = PPOConfig(**{**SMALL, "prime_iterations": 1}, diagnostics=True)
    runner, primed, state, metrics = train(cfg, rnd(reward_divisor="frozen_after_priming"))
    C, T, N = cfg.n_copies, cfg.num_steps, cfg.n_envs

    # the priming rollout again, step by step, in a fresh trainer: random actions from the
    # run's priming key
    runner = Runner(cfg, bonus=rnd(reward_divisor="frozen_after_priming"))
    start = runner.init_state(0)
    key = jax.random.fold_in(jax.random.fold_in(start.rng, 999999937), 0)
    actions = jax.random.uniform(key, (T, C, N, runner.env.act_dim), jnp.float32, -1.0, 1.0)
    env_state, reached = start.env_state, []
    for t in range(T):
        env_state, _, _, _, _, final_obs = runner.env.step(env_state, actions[t])
        reached.append(np.asarray(final_obs))
    flat = jnp.asarray(np.stack(reached).transpose(1, 0, 2, 3).reshape(C, T * N, 4))
    statistics = rms_update(rms_init(C, 4), flat)
    bonus = np.asarray(runner.bonus.score_alone(start.bonus_params, {"obs_rms": statistics}, flat),
                       np.float64).reshape(C, T, N)

    # the discounted sum forward in time, then the running variance from its starting values
    # (mean 0, variance 1, count 1e-4), as the statistics module merges a batch
    filtered, carried = np.zeros((C, T, N)), np.zeros((C, N))
    for t in range(T):
        carried = cfg.gamma_int * carried + bonus[:, t]
        filtered[:, t] = carried
    batch = filtered.reshape(C, T * N)
    count, total = 1e-4, 1e-4 + T * N
    variance = (count + batch.var(axis=1) * T * N
                + batch.mean(axis=1) ** 2 * count * T * N / total) / total
    np.testing.assert_allclose(primed.agent_state["int_rms"].var[:, 0], variance, rtol=1e-4)
    np.testing.assert_allclose(primed.agent_state["int_rms"].count, total)
    assert (np.abs(variance - 1) > 1e-3).all(), "priming did not move the divisor's statistics"

    # training leaves the statistics untouched, and the learner's reward is the bonus over it
    for name in ("int_rms", "int_filter"):
        assert same_bytes(state.agent_state[name], primed.agent_state[name]), name
    divisor = np.asarray(intrinsic_divisor("frozen_after_priming", state.agent_state["int_rms"], C))
    np.testing.assert_allclose(divisor, np.sqrt(variance + 1e-8), rtol=1e-4)
    for one in metrics:
        np.testing.assert_array_equal(one["diagnostics"]["int_std"], divisor)
        np.testing.assert_allclose(one["diagnostics"]["rint_hat_mean"],
                                   one["rint_mean"] / divisor, rtol=1e-5)

    # the control's divisor does move
    _, primed, state, _ = train(cfg, "rnd_next_state")
    assert (state.agent_state["int_rms"].var != primed.agent_state["int_rms"].var).all()

    # with no priming rollout there is nothing to freeze
    try:
        Runner(replace(cfg, prime_iterations=0), bonus=rnd(reward_divisor="frozen_after_priming"))
        raise AssertionError("a frozen divisor was accepted with no priming rollout")
    except ValueError as error:
        assert "priming rollout" in str(error)
    print("PASS frozen divisor: set by the priming rollouts, unchanged by training")


def test_no_divisor_hands_the_learner_the_bonus_itself():
    """Even start with no divisor: the learner's reward is the bonus, 0.5 at the first rollout
    and never above it by more than rounding; the divisor's statistics are never touched."""
    cfg = PPOConfig(**SMALL, diagnostics=True)
    _, primed, state, metrics = train(cfg, rnd(even_start=True, reward_divisor="none"))
    np.testing.assert_allclose(metrics[0]["rint_mean"], 0.5, rtol=1e-5)
    for one in metrics:
        np.testing.assert_array_equal(one["diagnostics"]["rint_hat_mean"], one["rint_mean"])
        np.testing.assert_array_equal(one["diagnostics"]["int_std"], 1.0)
    np.testing.assert_array_equal(state.agent_state["int_rms"].var, 1.0)
    np.testing.assert_array_equal(state.agent_state["int_rms"].count, 1e-4)
    assert not state.agent_state["int_filter"].any()
    print("PASS no divisor: the learner's reward is the bonus; first rollout 0.5")


def test_separate_predictor_step_size():
    """The predictor's own Adam: (1) with the clip out of reach, no anneal and the predictor's
    rate equal to the policy's, training is the joint optimizer's (the same arithmetic in
    another compiled program, so equal to rounding, not to the bit); (2) the policy's update
    does not depend on the predictor's rate, bit for bit; (3) one step of the predictor scales
    with its rate."""
    # (1) both rates 1e-4, a clip that never binds: the numbers of one Adam over both
    cfg = PPOConfig(**SMALL, learning_rate=1e-4, anneal_lr=False, max_grad_norm=1e9)
    for iterations, tolerance in ((1, 1e-5), (3, 1e-4)):
        _, _, joint, _ = train(cfg, "rnd_next_state", iterations)
        _, _, separate, _ = train(cfg, rnd(predictor_learning_rate_multiple=1.0), iterations)
        joint, separate = state_arrays(joint), state_arrays(separate)
        assert joint.keys() == separate.keys()
        for name in joint:
            if joint[name].dtype.kind == "f":
                np.testing.assert_allclose(separate[name], joint[name], rtol=tolerance,
                                           atol=tolerance * 1e-2, err_msg=name)
            else:
                np.testing.assert_array_equal(separate[name], joint[name], err_msg=name)

    # (2) and (3): the usual clip and anneal, one full-batch update, rates 0.25e-4 and 4e-4
    cfg = PPOConfig(**{**SMALL, "update_style": "full_batch"})
    runner, primed, slow, _ = train(cfg, rnd(predictor_learning_rate_multiple=0.25), iterations=1)
    assert runner.bonus.own_learning_rate == 0.25e-4
    _, _, fast, _ = train(cfg, rnd(predictor_learning_rate_multiple=4.0), iterations=1)
    assert same_bytes(slow.agent_params, fast.agent_params), "the policy's update moved"
    for name in ("W0", "W1", "W2"):
        before = primed.bonus_params["predictor"][name]
        slow_step = slow.bonus_params["predictor"][name] - before
        fast_step = fast.bonus_params["predictor"][name] - before
        # a step of one or two float32 units of the parameter is 16 times coarser when scaled
        np.testing.assert_allclose(fast_step, 16 * slow_step, rtol=2e-2, atol=2e-6)
        assert np.abs(slow_step).max() > 1e-6, name

    # the options this needs
    for changes in (dict(flat_params=True), dict(diagnostics=True)):
        try:
            Runner(replace(cfg, **changes), bonus=rnd(predictor_learning_rate_multiple=1.0))
            raise AssertionError(f"accepted: {changes}")
        except ValueError as error:
            assert "own learning rate" in str(error)
    print("PASS separate predictor step size: joint numbers at equal rates; the policy's update "
          "does not depend on it; one predictor step scales with the rate")


def test_fast_last_layer_inside_the_rollout():
    """Fast last layer: with the exact fit a state seen again in one rollout pays 0, with half
    the fit a quarter; scoring changes no parameter; the predictor's gradient update is the
    control's (compared with the clip out of reach, where the two updates do not interact)."""
    # one environment, the rollout A, A, B, A
    cfg = PPOConfig(**{**SMALL, "n_envs": 1, "num_steps": 4})
    two = random_observations(cfg, 2, seed=2)
    rollout = two[:, np.array([0, 0, 1, 0])]
    exact = family(cfg, fast_last_layer_step=1.0)
    assert exact.rollout_step is None
    params, state = exact.init()
    alone = np.asarray(exact.score_alone(params, state, rollout))
    new_state, reward, extra = exact.post_rollout(params, state, rollout, None)
    reward = np.asarray(reward)
    np.testing.assert_allclose(reward[:, 0], alone[:, 0], rtol=1e-5)
    assert (reward[:, 1] < 1e-12 * reward[:, 0]).all(), reward
    assert (reward[:, 2] > 0).all()
    assert sorted(new_state) == ["obs_rms"] and sorted(extra) == ["rnd_input"]
    np.testing.assert_array_equal(np.asarray(exact.score_alone(params, state, rollout)), alone)
    half = family(cfg, fast_last_layer_step=0.5)
    _, reward, _ = half.post_rollout(params, state, rollout, None)
    np.testing.assert_allclose(np.asarray(reward)[:, 1], alone[:, 0] / 4, rtol=1e-5)

    # two environments in states A and B at both steps: fitted at once, both pay nothing again
    cfg = PPOConfig(**{**SMALL, "n_envs": 2, "num_steps": 2})
    exact = family(cfg, fast_last_layer_step=1.0)
    params, state = exact.init()
    _, reward, _ = exact.post_rollout(params, state, two[:, np.array([0, 1, 0, 1])], None)
    reward = np.asarray(reward)
    assert (reward[:, 2:] < 1e-10 * reward[:, :2]).all(), reward

    # in training: the predictor after one iteration is the control's, bit for bit
    cfg = PPOConfig(**SMALL, max_grad_norm=1e9)
    _, _, control, control_metrics = train(cfg, "rnd_next_state", iterations=1,
                                           env_cfg=SHORT_EPISODES)
    _, _, fitted, metrics = train(cfg, rnd(fast_last_layer_step=1.0), iterations=1,
                                  env_cfg=SHORT_EPISODES)
    assert same_bytes(control.bonus_params, fitted.bonus_params), "the predictor's update moved"
    assert same_bytes(control.bonus_state, fitted.bonus_state)
    assert (metrics[0]["rint_mean"] < control_metrics[0]["rint_mean"]).all()

    # a rollout scored one step at a time has no order to fit in
    try:
        Runner(replace(cfg, hoist_rollout=False), bonus=rnd(fast_last_layer_step=1.0))
        raise AssertionError("the fast last layer was composed with the in-scan rollout")
    except ValueError as error:
        assert "hoist" in str(error)
    print("PASS fast last layer: a second visit pays 0 at the exact fit; parameters unchanged "
          "by scoring; the predictor's update is the control's")


def test_scheduled_last_layer_on_fixed_features():
    """Scheduled last layer: one state visited k times has squared error (1 + k)^(-a) for
    a = 0.5 and 1; nothing is trainable; the body is the predictor's own draw; no divisor."""
    visits = 5
    cfg = PPOConfig(**{**SMALL, "n_envs": 1, "num_steps": visits})
    one = random_observations(cfg, 1, seed=3)
    rollout = jnp.repeat(one, visits, axis=1)
    control_params, _ = family(cfg).init()
    for decay in (0.5, 1.0):
        scheduled = family(cfg, scheduled_last_layer_decay=decay)
        assert scheduled.reward_divisor == "none" and scheduled.rollout_step is None
        params, state = scheduled.init()
        assert params == {} and state["output_layer"].dtype == jnp.float64
        assert state["output_layer"].shape == (cfg.n_copies, 129, 128)
        for name in ("W0", "b0", "W1", "b1"):
            np.testing.assert_array_equal(np.asarray(scheduled.constants["predictor_body"][name]),
                                          np.asarray(control_params["predictor"][name]))
        np.testing.assert_allclose(np.asarray(scheduled.score_alone(params, state, one)), 1.0,
                                   rtol=1e-6)

        # the bonus at visit k is the error after k - 1 visits; the kept layer carries on
        new_state, reward, extra = scheduled.post_rollout(params, state, rollout, None)
        assert extra == {} and float(scheduled.loss(params, {})) == 0.0
        np.testing.assert_allclose(np.asarray(reward),
                                   np.tile((1.0 + np.arange(visits)) ** -decay, (cfg.n_copies, 1)),
                                   rtol=1e-6)
        # the same input statistics, so the state's features are the ones that were fitted
        kept = {"obs_rms": state["obs_rms"], "output_layer": new_state["output_layer"]}
        np.testing.assert_allclose(np.asarray(scheduled.score_alone(params, kept, one)),
                                   (1.0 + visits) ** -decay, rtol=1e-6)
        _, reward, _ = scheduled.post_rollout(params, kept, rollout, None)
        np.testing.assert_allclose(np.asarray(reward)[:, -1], (2.0 * visits) ** -decay, rtol=1e-6)
        # priming moves the input statistics and leaves the layer alone
        primed = scheduled.prime(params, kept, rollout)
        assert primed["output_layer"] is kept["output_layer"]

    # in training: nothing of the bonus is trained, and the reward is at most about 1
    cfg = PPOConfig(**SMALL)
    runner, _, state, metrics = train(cfg, rnd(scheduled_last_layer_decay=0.75))
    assert state.bonus_params == {} and runner.bonus.reward_divisor == "none"
    assert np.asarray(state.bonus_state["output_layer"]).any()
    assert all((one["rint_mean"] > 0).all() and (one["rint_mean"] < 1.05).all()
               for one in metrics)
    print("PASS scheduled last layer: (1 + k)^(-a) after k visits for a = 0.5 and 1")


def test_saved_units_continue_exactly():
    """A unit saved and restored equals the uninterrupted unit bit for bit: with the scheduled
    last layer (its kept layer and its fixed body are in the checkpoint), and with the other
    options together (frozen divisor, the predictor's own rate, fast last layer, maze grid)."""
    cfg = PPOConfig(**SMALL, episode_log=True)
    check_resume(cfg, SHORT_EPISODES, rnd(scheduled_last_layer_decay=0.75), total=4, stop_after=2)
    check_resume(cfg, SHORT_EPISODES,
                 rnd(reward_divisor="frozen_after_priming", predictor_learning_rate_multiple=4,
                     fast_last_layer_step=0.3, record_bonus_on_maze_grid=True, even_start=True,
                     bonus_form="error_length"), total=4, stop_after=2)
    scheduled = family(cfg, scheduled_last_layer_decay=0.75)
    assert sorted(scheduled.constants) == ["predictor_body", "target"]
    print("PASS saved units with the new options continue exactly")


def test_one_policy_never_moves_another():
    """With the two last-layer fits, moving one policy's actor before training leaves every
    other policy's parameters and bonus state bit-identical."""
    cfg = PPOConfig(**{**SMALL, "n_copies": 4})
    keep = np.array([0, 1, 3])
    for options in (dict(fast_last_layer_step=0.3), dict(scheduled_last_layer_decay=0.75)):
        finals = []
        for move in (False, True):
            runner = Runner(cfg, bonus=rnd(**options), env_cfg=SHORT_EPISODES)
            state = runner.init_state(7)
            if move:
                moved = state.agent_params["actor"]["W0"].at[2].add(0.05)
                state = state._replace(agent_params={
                    **state.agent_params, "actor": {**state.agent_params["actor"], "W0": moved}})
            state = runner.prime(state)
            for k in (1, 2):
                state, _ = runner.iterate(state, runner.lr_argument(k, 2))
            finals.append({**state_arrays(state._replace(agent_state={}, env_state=(), opt=()))})
        plain, moved = finals
        differs = False
        for name in plain:
            if plain[name].ndim == 0 or plain[name].shape[0] != cfg.n_copies:
                continue
            assert plain[name][keep].tobytes() == moved[name][keep].tobytes(), (options, name)
            differs = differs or plain[name][2].tobytes() != moved[name][2].tobytes()
        assert differs, "the moved policy did not change"
    print("PASS the last-layer fits keep the policies apart")


if __name__ == "__main__":
    test_options_are_read_checked_and_recorded()
    test_options_at_their_defaults_change_nothing()
    test_recording_on_the_maze_grid_changes_nothing_in_training()
    test_even_start_gives_every_input_the_same_first_bonus()
    test_error_length_is_the_length_of_the_same_error()
    test_frozen_divisor_is_set_by_priming_and_does_not_move()
    test_no_divisor_hands_the_learner_the_bonus_itself()
    test_separate_predictor_step_size()
    test_fast_last_layer_inside_the_rollout()
    test_scheduled_last_layer_on_fixed_features()
    test_saved_units_continue_exactly()
    test_one_policy_never_moves_another()
