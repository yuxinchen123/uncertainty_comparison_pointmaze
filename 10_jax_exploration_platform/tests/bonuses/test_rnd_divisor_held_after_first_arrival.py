"""The reward divisor held after a policy's first arrival (`reward_divisor=held_after_first_arrival`,
added 2026-11-13 for the PointMaze and AntMaze study).

For each policy on its own, the divisor of the intrinsic reward (the running standard deviation of
the reward's discounted sum) is moved on by every rollout up to and including the one in which the
policy first reaches the goal, and by no later rollout. A policy that never arrives keeps the
running divisor. The tests use a small PointMaze whose goal lies next to the start, so that some
policies arrive within a few iterations and others do not.

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
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import make_bonus  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.train_step import intrinsic_divisor  # noqa: E402
from test_checkpoint_resume import check_resume  # noqa: E402
from test_rnd_bonus_options import SMALL, rnd, same_bytes, state_arrays  # noqa: E402

import jax  # noqa: E402

OPTION = "held_after_first_arrival"
# six policies, two environments each, rollouts of 16 steps: 32 environment steps per policy and
# iteration. The goal is the cell to the right of the start and counts from 0.8 m, so a policy
# has to move 0.2 m; episodes end there or after 32 steps
SIX = {**SMALL, "n_copies": 6}
NEAR_GOAL = EnvConfig(continuing_task=False, max_episode_steps=32, goal_radius=0.8, goal_cell=(7, 2))
PLANNED = 8
STEPS_PER_ITERATION = SIX["n_envs"] * SIX["num_steps"]
# the iteration (from 1) in which each policy first arrives with run seed 0 and CleanRL's bonus, as
# found on 2026-11-13; None: no arrival in 8 iterations. A change means the set-up no longer
# exercises what the tests below need
FIRST_ARRIVALS_WITH_SEED_0 = [None, 4, 4, 2, 6, None]


def train_each(cfg, bonus, env_cfg, iterations: int = PLANNED, seed: int = 0):
    """Prime and train; returns (trainer, the state after every iteration, the metrics of every
    iteration) as host arrays."""
    runner = Runner(cfg, bonus=bonus, env_cfg=env_cfg)
    state = runner.prime(runner.init_state(seed))
    states, metrics = [], []
    for k in range(1, iterations + 1):
        state, one = runner.iterate(state, runner.lr_argument(k, iterations))
        states.append(jax.device_get(state))
        metrics.append(jax.device_get(one))
    return runner, states, metrics


def arrival_iterations(state) -> list:
    """The iteration (from 1) of each policy's first arrival, None where the log holds none.

    before: first arrival steps [-1, 126, 60] with 32 steps per iteration ; after: [None, 4, 2]
    """
    steps = np.asarray(state.agent_state["episode_log"]["first_goal_step"])
    return [None if step < 0 else int(np.ceil(step / STEPS_PER_ITERATION)) for step in steps]


def rows_of_copy(state, copy: int, n_copies: int) -> dict:
    """Every array of a training state that has one row per policy, cut to one policy's row."""
    return {name: leaf[copy] for name, leaf in state_arrays(state).items()
            if leaf.ndim >= 1 and leaf.shape[0] == n_copies}


def divisor_rows(state, copy: int) -> list:
    """One policy's rows of the divisor's statistics (mean, variance, count) and of the filter value."""
    return [np.asarray(leaf)[copy] for leaf in (*state.agent_state["int_rms"],
                                                state.agent_state["int_filter"])]


def test_each_policy_holds_its_divisor_from_its_own_first_arrival():
    """Up to and including the iteration of its first arrival a policy is the control's (the
    running divisor) bit for bit; from the next iteration on its divisor's statistics and its
    filter value are those that iteration left, while the control's and the other policies' keep
    moving; a policy with no arrival is the control's for the whole run; the learner's reward is
    the bonus over the divisor of the state."""
    cfg = PPOConfig(**SIX, episode_log=True, diagnostics=True)
    C = cfg.n_copies
    _, control, _ = train_each(cfg, "rnd_next_state", NEAR_GOAL)
    runner, held, metrics = train_each(cfg, rnd(reward_divisor=OPTION), NEAR_GOAL)
    assert runner.bonus.reward_divisor == OPTION and runner.bonus.divisor_hold_fraction is None

    # the set-up: policies that arrive at iteration 2 or later with iterations left after it, and
    # policies that never arrive; the two runs agree on every first arrival
    arrivals = arrival_iterations(control[-1])
    assert arrivals == FIRST_ARRIVALS_WITH_SEED_0, arrivals
    assert arrival_iterations(held[-1]) == arrivals
    np.testing.assert_array_equal(held[-1].agent_state["episode_log"]["first_goal_step"],
                                  control[-1].agent_state["episode_log"]["first_goal_step"])

    for copy, arrival in enumerate(arrivals):
        last_moved = PLANNED if arrival is None else arrival
        # the control's policy, bit for bit, through the iteration of the first arrival: the hold
        # does not begin before the rollout with the arrival has moved the divisor
        for index in range(last_moved):
            one, other = rows_of_copy(control[index], copy, C), rows_of_copy(held[index], copy, C)
            assert one.keys() == other.keys() and len(one) >= 10, len(one)
            for name in one:
                assert one[name].tobytes() == other[name].tobytes(), (copy, index + 1, name)
        if arrival is None:
            continue
        # the rollout with the arrival still moved the divisor
        assert not same_bytes(divisor_rows(held[arrival - 1], copy),
                              divisor_rows(held[arrival - 2], copy))
        # from the next iteration on: the statistics and the filter value of the arrival's iteration
        kept = divisor_rows(held[arrival - 1], copy)
        for index in range(arrival, PLANNED):
            assert same_bytes(divisor_rows(held[index], copy), kept), (copy, index + 1)
            assert not same_bytes(divisor_rows(control[index], copy), kept), (copy, index + 1)

    # a policy with no arrival keeps moving in the run with the hold, in every iteration
    for copy in (index for index, arrival in enumerate(arrivals) if arrival is None):
        for index in range(1, PLANNED):
            assert not same_bytes(divisor_rows(held[index], copy),
                                  divisor_rows(held[index - 1], copy)), (copy, index + 1)

    # the learner's reward: the bonus over the divisor read from the state the iteration left
    # (the value just moved for a policy not yet held, the held value for the others)
    for index, one in enumerate(metrics):
        divisor = np.asarray(intrinsic_divisor(OPTION, held[index].agent_state["int_rms"], C))
        np.testing.assert_array_equal(one["diagnostics"]["int_std"], divisor)
        np.testing.assert_allclose(one["diagnostics"]["rint_hat_mean"],
                                   one["rint_mean"] / divisor, rtol=1e-5)
    print(f"PASS divisor held after the first arrival: first arrivals at iterations {arrivals}; "
          "each policy is the control's through its arrival's iteration and holds its divisor "
          "from the next one; policies with no arrival are the control's for the whole run")


def test_the_hold_works_with_the_other_options_of_the_bonus():
    """With the predictor's own learning rate, the fast last layer and the error length as the
    bonus, each policy's divisor still moves through the iteration of its own first arrival and
    stays from the next one on."""
    cfg = PPOConfig(**SIX, episode_log=True)
    _, states, _ = train_each(cfg, rnd(reward_divisor=OPTION, predictor_learning_rate_multiple=0.5,
                                       fast_last_layer_step=0.1, bonus_form="error_length"),
                              NEAR_GOAL)
    arrivals = arrival_iterations(states[-1])
    assert any(arrival is not None and 2 <= arrival < PLANNED for arrival in arrivals), arrivals
    assert any(arrival is None for arrival in arrivals), arrivals
    for copy, arrival in enumerate(arrivals):
        last_moved = PLANNED if arrival is None else arrival
        for index in range(1, PLANNED):
            moved = not same_bytes(divisor_rows(states[index], copy),
                                   divisor_rows(states[index - 1], copy))
            assert moved == (index < last_moved), (copy, index + 1, arrival)
    print(f"PASS the hold with the predictor's own rate, the fast last layer and the error "
          f"length: first arrivals at iterations {arrivals}")


def test_saved_units_continue_exactly_before_and_after_the_hold():
    """A unit saved before any policy holds its divisor, and a unit saved after one policy holds
    it and before another arrives, both continue to the uninterrupted unit bit for bit."""
    cfg = PPOConfig(**SIX, episode_log=True)
    bonus = rnd(reward_divisor=OPTION)
    # the first arrivals of the unit the resume check trains (its run seed is 17)
    _, states, _ = train_each(cfg, bonus, NEAR_GOAL, seed=17)
    arrivals = [arrival for arrival in arrival_iterations(states[-1]) if arrival is not None]
    first, last = min(arrivals), max(arrivals)
    assert 2 <= first and first + 1 < last <= PLANNED, arrivals
    # saved one iteration before the first arrival: no policy holds its divisor yet
    check_resume(cfg, NEAR_GOAL, bonus, total=PLANNED, stop_after=first - 1)
    # saved one iteration after the first arrival: that policy holds, the last arrival is to come
    check_resume(cfg, NEAR_GOAL, bonus, total=PLANNED, stop_after=first + 1)
    print(f"PASS saved units continue exactly: saved after iteration {first - 1} (before any "
          f"hold) and after iteration {first + 1} (first arrivals at iterations {sorted(arrivals)})")


def test_what_is_refused():
    """Without the episode log, in an environment whose episodes do not end at the goal, with the
    fraction of the other hold, and with the scheduled last layer, the option is refused."""
    bonus = rnd(reward_divisor=OPTION)
    for cfg, env_cfg, message in (
            (PPOConfig(**SIX), NEAR_GOAL, "needs episode_log"),
            (PPOConfig(**SIX, episode_log=True), EnvConfig(), "end at the goal")):
        try:
            Runner(cfg, bonus=bonus, env_cfg=env_cfg)
            raise AssertionError(f"accepted where it must be refused: {message}")
        except ValueError as error:
            assert message in str(error), (message, str(error))
    for options, message in (({"divisor_hold_fraction": 0.5}, "read only with"),
                             ({"scheduled_last_layer_decay": 1}, "no divisor")):
        try:
            make_bonus("rnd_next_state", {"reward_divisor": OPTION, **options})
            raise AssertionError(f"accepted: {options}")
        except ValueError as error:
            assert message in str(error), (message, str(error))
    print("PASS refused without the episode log, without episodes that end at the goal, with a "
          "hold fraction, and with the scheduled last layer")


def test_antmaze_takes_the_option():
    """AntMaze: with no arrival the run is the running divisor's bit for bit; a policy whose first
    arrival is in the log keeps its divisor in the next iteration while the other's moves."""
    env_cfg = replace(preset("umaze"), continuing_task=False, max_episode_steps=6)
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=1, num_steps=4,
                    prime_iterations=1, update_style="epoch_minibatch", num_minibatches=2,
                    episode_log=True)
    _, control, _ = train_each(cfg, "rnd_next_state", env_cfg, iterations=2)
    runner = Runner(cfg, bonus=rnd(reward_divisor=OPTION), env_cfg=env_cfg)
    state = runner.prime(runner.init_state(0))
    first = jax.device_get(runner.iterate(state, runner.lr_argument(1, 2))[0])
    # the second iteration twice, each from the host copy of the first one's state (an iteration
    # uses up the state it is given): as it is, and with an arrival written into policy 1's log
    log = first.agent_state["episode_log"]
    arrived = np.array(log["first_goal_step"])
    arrived[1] = 3
    marked = first._replace(agent_state={**first.agent_state,
                                         "episode_log": {**log, "first_goal_step": arrived}})
    second = jax.device_get(runner.iterate(first, runner.lr_argument(2, 2))[0])
    after_mark = jax.device_get(runner.iterate(marked, runner.lr_argument(2, 2))[0])

    # no arrival in two iterations of six-step episodes: the running divisor's run, bit for bit
    assert (np.asarray(second.agent_state["episode_log"]["first_goal_step"]) < 0).all()
    for mine, theirs in ((first, control[0]), (second, control[1])):
        one, other = state_arrays(mine), state_arrays(theirs)
        assert one.keys() == other.keys()
        assert all(one[name].tobytes() == other[name].tobytes() for name in one)
    # with the arrival in the log: policy 1 keeps the first iteration's divisor, policy 0 moves
    assert same_bytes(divisor_rows(after_mark, 1), divisor_rows(first, 1))
    assert same_bytes(divisor_rows(after_mark, 0), divisor_rows(second, 0))
    assert not same_bytes(divisor_rows(after_mark, 0), divisor_rows(first, 0))
    print("PASS AntMaze: the running divisor's run without an arrival; a policy with an arrival "
          "in the log keeps its divisor while the other moves")


if __name__ == "__main__":
    test_each_policy_holds_its_divisor_from_its_own_first_arrival()
    test_the_hold_works_with_the_other_options_of_the_bonus()
    test_saved_units_continue_exactly_before_and_after_the_hold()
    test_what_is_refused()
    test_antmaze_takes_the_option()
