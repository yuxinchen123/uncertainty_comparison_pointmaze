"""The fast last layer that stops at a policy's first arrival
(`fast_last_layer_until_first_arrival=true` with `fast_last_layer_step` above 0, added 2026-11-13
for the PointMaze and AntMaze study).

For each policy on its own, the rollouts up to and including the one in which the policy first
reaches the goal are scored with the fast last layer; every later rollout is scored by the
predictor alone, exactly as random network distillation scores it. A policy that never arrives
keeps the fast last layer. The temporary layer is rebuilt from the predictor at every rollout and
dropped after it, so the option adds nothing to the training state. The tests use the small
PointMaze of the divisor held after the first arrival, whose goal lies next to the start.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "agents"))
sys.path.insert(0, str(ROOT / "tests" / "bonuses"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import bonus_options_record, make_bonus  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.checkpoint import check_identity  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from test_checkpoint_resume import check_resume  # noqa: E402
from test_rnd_bonus_options import (SMALL, family, random_observations, rnd, same_bytes,  # noqa: E402
                                    state_arrays)
from test_rnd_divisor_held_after_first_arrival import (NEAR_GOAL, PLANNED, SIX,  # noqa: E402
                                                       arrival_iterations, divisor_rows,
                                                       rows_of_copy, train_each)

import jax  # noqa: E402

STEP = 0.3
FAST = dict(fast_last_layer_step=STEP)
UNTIL = dict(fast_last_layer_step=STEP, fast_last_layer_until_first_arrival=True)
HOLD = "held_after_first_arrival"


def same_rows(one: dict, other: dict) -> bool:
    """Whether two cuts of a training state to one policy hold the same arrays, bit for bit."""
    return one.keys() == other.keys() and all(
        one[name].tobytes() == other[name].tobytes() for name in one)


def largest_difference(one: dict, other: dict) -> float:
    """The largest absolute difference between two cuts of a training state to one policy, over
    their floating-point arrays; every other array (counts, keys, flags) must be the same.

    before: {"w": [1.0, 2.0], "n": [3]} and {"w": [1.0, 2.5], "n": [3]} ; after: 0.5
    """
    assert one.keys() == other.keys()
    largest = 0.0
    for name in one:
        if np.issubdtype(one[name].dtype, np.floating):
            if one[name].size:
                largest = max(largest, float(np.abs(one[name].astype(np.float64)
                                                    - other[name].astype(np.float64)).max()))
        else:
            np.testing.assert_array_equal(one[name], other[name], err_msg=name)
    return largest


def next_from(runner, state, iteration: int):
    """One further iteration from a host copy of a state -> (state, metrics) as host arrays."""
    after, metrics = runner.iterate(state, runner.lr_argument(iteration, PLANNED))
    return jax.device_get(after), jax.device_get(metrics)


def test_the_hook_scores_arrived_policies_by_the_predictor_alone():
    """The bonus's own hook, without a trainer: a policy marked as arrived gets exactly the
    scores of the family without the fast last layer, a policy not marked exactly the scores of
    the fast last layer; the state and the update batch are the same in all three families."""
    cfg = PPOConfig(**{**SMALL, "n_envs": 2, "num_steps": 4})
    rollout = random_observations(cfg, 8, seed=3)
    plain, fast, until = family(cfg), family(cfg, **FAST), family(cfg, **UNTIL)
    assert until.reads_first_arrival and not fast.reads_first_arrival
    assert until.rollout_step is None
    params, state = until.init()
    _, plain_reward, plain_extra = plain.post_rollout(params, state, rollout, None)
    _, fast_reward, _ = fast.post_rollout(params, state, rollout, None)
    plain_reward, fast_reward = np.asarray(plain_reward), np.asarray(fast_reward)
    # the two scorings differ for every policy, or the checks below would show nothing
    assert all(not np.array_equal(plain_reward[copy], fast_reward[copy]) for copy in range(3))

    # before: arrived [False, True, False]
    # after:  policies 0 and 2 hold the fast scores, policy 1 the predictor's own scores
    arrived = np.array([False, True, False])
    new_state, reward, extra = until.post_rollout(params, state, rollout, None, arrived)
    reward = np.asarray(reward)
    np.testing.assert_array_equal(reward[1], plain_reward[1])
    np.testing.assert_array_equal(reward[[0, 2]], fast_reward[[0, 2]])
    assert reward.dtype == plain_reward.dtype
    assert sorted(new_state) == ["obs_rms"] and sorted(extra) == ["rnd_input"]
    np.testing.assert_array_equal(np.asarray(extra["rnd_input"]),
                                  np.asarray(plain_extra["rnd_input"]))
    # edge cases: nobody has arrived (the fast last layer's rollout), everybody has (the plain one)
    for marks, expected in ((np.zeros(3, bool), fast_reward), (np.ones(3, bool), plain_reward)):
        _, reward, _ = until.post_rollout(params, state, rollout, None, marks)
        np.testing.assert_array_equal(np.asarray(reward), expected)
    print("PASS the hook: an arrived policy is scored by the predictor alone, the others by the "
          "fast last layer; state and update batch unchanged")


def test_each_policy_keeps_the_fast_layer_through_its_first_arrival_and_no_longer():
    """In training: through the iteration of its own first arrival a policy is the fast last
    layer's bit for bit (the rollout with the arrival is still scored with the fast layer); from
    the next iteration on its rollout is scored as random network distillation scores it, shown
    by continuing one state with three programs; a policy with no arrival is the fast last
    layer's for the whole run; the option adds no array to the training state.

    The logged mean reward is compared to rounding only: it is a sum formed for the record, and
    two compiled programs may order that sum differently (it differed in its last digit between
    the program with the option and the fast last layer's on 2026-11-13, while the two training
    states were the same bit for bit). Against the plain program the state is compared to
    rounding as well, since nothing makes two programs order their float32 arithmetic alike."""
    cfg = PPOConfig(**SIX, episode_log=True)
    C = cfg.n_copies
    fast_runner, fast, _ = train_each(cfg, rnd(**FAST), NEAR_GOAL)
    runner, until, _ = train_each(cfg, rnd(**UNTIL), NEAR_GOAL)
    assert runner.bonus.reads_first_arrival and not fast_runner.bonus.reads_first_arrival

    # nothing is added to the training state: the same arrays, in the same shapes and types
    one, other = state_arrays(fast[-1]), state_arrays(until[-1])
    assert one.keys() == other.keys()
    assert all(one[name].shape == other[name].shape and one[name].dtype == other[name].dtype
               for name in one)

    # the set-up: the two runs agree on every first arrival, one policy arrives with iterations
    # left after it, and another has not arrived by the iteration after that
    arrivals = arrival_iterations(fast[-1])
    assert arrival_iterations(until[-1]) == arrivals, (arrival_iterations(until[-1]), arrivals)
    np.testing.assert_array_equal(until[-1].agent_state["episode_log"]["first_goal_step"],
                                  fast[-1].agent_state["episode_log"]["first_goal_step"])
    reached = [arrival for arrival in arrivals if arrival is not None]
    assert reached and min(reached) < PLANNED, arrivals
    switch = min(reached)
    early = [copy for copy, arrival in enumerate(arrivals) if arrival == switch]
    late = [copy for copy, arrival in enumerate(arrivals) if arrival is None or arrival > switch]
    assert early and late, arrivals

    # the fast last layer's policy, bit for bit, through the iteration of the first arrival
    for copy, arrival in enumerate(arrivals):
        last_fast = PLANNED if arrival is None else arrival
        for index in range(last_fast):
            assert same_rows(rows_of_copy(fast[index], copy, C),
                             rows_of_copy(until[index], copy, C)), (copy, index + 1)
        # and no longer: the next iteration leaves another state than the fast last layer's
        if arrival is not None and arrival < PLANNED:
            assert not same_rows(rows_of_copy(fast[arrival], copy, C),
                                 rows_of_copy(until[arrival], copy, C)), (copy, arrival + 1)

    # the iteration after the first arrival, three times from the same state: with the option,
    # with the fast last layer for everybody, and with the predictor alone for everybody
    start = until[switch - 1]
    plain_runner = Runner(cfg, bonus="rnd_next_state", env_cfg=NEAR_GOAL)
    mine, mine_metrics = next_from(runner, start, switch + 1)
    as_fast, fast_metrics = next_from(fast_runner, start, switch + 1)
    as_plain, plain_metrics = next_from(plain_runner, start, switch + 1)
    for copy in range(C):
        # the two programs must differ clearly for this policy, or the comparison shows nothing
        assert not np.isclose(fast_metrics["rint_mean"][copy], plain_metrics["rint_mean"][copy],
                              rtol=1e-3), copy
    distances = []
    for copy in early:
        # arrived in the rollout before: scored and updated as the plain program does it, to
        # rounding, and far from what the fast last layer would have left
        np.testing.assert_allclose(mine_metrics["rint_mean"][copy],
                                   plain_metrics["rint_mean"][copy], rtol=1e-6)
        to_plain = largest_difference(rows_of_copy(mine, copy, C), rows_of_copy(as_plain, copy, C))
        to_fast = largest_difference(rows_of_copy(mine, copy, C), rows_of_copy(as_fast, copy, C))
        distances.append((copy, to_plain, to_fast))
        assert to_plain <= 1e-5 and to_plain <= 1e-3 * to_fast, (copy, to_plain, to_fast)
    for copy in late:
        # not arrived yet, in the same batch: still the fast last layer, bit for bit in the state
        np.testing.assert_allclose(mine_metrics["rint_mean"][copy],
                                   fast_metrics["rint_mean"][copy], rtol=1e-6)
        assert same_rows(rows_of_copy(mine, copy, C), rows_of_copy(as_fast, copy, C)), copy
        assert not same_rows(rows_of_copy(mine, copy, C), rows_of_copy(as_plain, copy, C)), copy
    print(f"PASS fast last layer until the first arrival: first arrivals at iterations "
          f"{arrivals}; each policy is the fast last layer's through its arrival's iteration; in "
          f"iteration {switch + 1} policies {early} are scored by the predictor alone and "
          f"policies {late} by the fast last layer; largest difference of an arrived policy's "
          f"state from the plain program's and from the fast last layer's (policy, plain, fast): "
          f"{[(copy, float(f'{a:.3g}'), float(f'{b:.3g}')) for copy, a, b in distances]}")


def test_it_works_with_the_divisor_held_after_the_first_arrival():
    """Together with `reward_divisor=held_after_first_arrival`: the program builds and trains,
    each policy is bit for bit the run of the fast last layer with the hold through the iteration
    of its first arrival, and its divisor moves through that iteration and stays afterwards."""
    cfg = PPOConfig(**SIX, episode_log=True)
    C = cfg.n_copies
    _, reference, _ = train_each(cfg, rnd(**FAST, reward_divisor=HOLD), NEAR_GOAL)
    runner, states, _ = train_each(cfg, rnd(**UNTIL, reward_divisor=HOLD), NEAR_GOAL)
    assert runner.bonus.reads_first_arrival and runner.bonus.reward_divisor == HOLD
    arrivals = arrival_iterations(states[-1])
    assert arrival_iterations(reference[-1]) == arrivals
    assert any(arrival is not None and arrival < PLANNED for arrival in arrivals), arrivals
    for copy, arrival in enumerate(arrivals):
        last_moved = PLANNED if arrival is None else arrival
        for index in range(last_moved):
            assert same_rows(rows_of_copy(reference[index], copy, C),
                             rows_of_copy(states[index], copy, C)), (copy, index + 1)
        for index in range(1, PLANNED):
            moved = not same_bytes(divisor_rows(states[index], copy),
                                   divisor_rows(states[index - 1], copy))
            assert moved == (index < last_moved), (copy, index + 1, arrival)
    print(f"PASS with the divisor held after the first arrival: first arrivals at iterations "
          f"{arrivals}; the fast last layer and the moving divisor both end with the arrival")


def test_saved_units_continue_exactly_before_and_after_the_switch():
    """A unit saved before any policy has arrived, and a unit saved after the first policy
    switched to the predictor alone, both continue to the uninterrupted unit bit for bit."""
    cfg = PPOConfig(**SIX, episode_log=True)
    bonus = rnd(**UNTIL)
    # the first arrivals of the unit the resume check trains (its run seed is 17)
    _, states, _ = train_each(cfg, bonus, NEAR_GOAL, seed=17)
    arrivals = [arrival for arrival in arrival_iterations(states[-1]) if arrival is not None]
    assert arrivals and min(arrivals) + 1 < PLANNED, arrivals
    first = min(arrivals)
    before, after = max(first - 1, 1), first + 1
    check_resume(cfg, NEAR_GOAL, bonus, total=PLANNED, stop_after=before)
    check_resume(cfg, NEAR_GOAL, bonus, total=PLANNED, stop_after=after)
    print(f"PASS saved units continue exactly: saved after iteration {before} and after iteration "
          f"{after} (first arrivals at iterations {sorted(arrivals)})")


def test_a_checkpoint_from_before_the_option_continues():
    """A unit that changed a bonus option and was saved before this option existed has no entry
    for it in its record. It continues under the code with the option at its default, and is
    refused when the current unit has set the option."""
    defaults = bonus_options_record("rnd_next_state")
    assert defaults["fast_last_layer_until_first_arrival"] is False
    record = lambda **options: {"trainer_config": asdict(PPOConfig(n_copies=4)), "run_seed": 17,
                                "bonus": "rnd_next_state",
                                "bonus_options": bonus_options_record("rnd_next_state", options)}
    # before: the saved record of a unit with the fast last layer, written by the earlier code
    # (a checkpoint holds its record as JSON)
    older = json.loads(json.dumps(record(**FAST)))
    del older["bonus_options"]["fast_last_layer_until_first_arrival"]
    check_identity(older, record(**FAST), bonus_option_defaults=defaults)
    for current in (record(**UNTIL), record(fast_last_layer_step=0.5)):
        try:
            check_identity(older, current, bonus_option_defaults=defaults)
            raise AssertionError("a changed bonus option was accepted")
        except ValueError as error:
            assert "bonus_options" in str(error), str(error)
    # without the family's defaults the later option is refused, as before
    try:
        check_identity(older, record(**FAST))
        raise AssertionError("a later bonus option was accepted without the family's defaults")
    except ValueError as error:
        assert "bonus_options" in str(error), str(error)
    # a unit that changed no option has no bonus_options entry on either side
    plain = {"trainer_config": asdict(PPOConfig(n_copies=4)), "run_seed": 17,
             "bonus": "rnd_next_state"}
    check_identity(json.loads(json.dumps(plain)), plain, bonus_option_defaults=defaults)
    print("PASS a checkpoint written before the option existed continues at the option's default "
          "and is refused when the option is set")


def test_what_is_refused():
    """Without a fast last layer, with the scheduled last layer, without the episode log, in an
    environment whose episodes do not end at the goal, and for a caller of the hook that hands
    over no arrivals, the option is refused."""
    for options, message in (
            ({"fast_last_layer_until_first_arrival": "true"}, "fast_last_layer_step above zero"),
            ({"fast_last_layer_until_first_arrival": True, "fast_last_layer_step": 0},
             "fast_last_layer_step above zero"),
            ({**UNTIL, "scheduled_last_layer_decay": 1}, "cannot be")):
        try:
            make_bonus("rnd_next_state", options)
            raise AssertionError(f"accepted: {options}")
        except ValueError as error:
            assert message in str(error), (message, str(error))
    bonus = rnd(**UNTIL)
    for cfg, env_cfg, message in (
            (PPOConfig(**SIX), NEAR_GOAL, "needs episode_log"),
            (PPOConfig(**SIX, episode_log=True), EnvConfig(), "end at the goal")):
        try:
            Runner(cfg, bonus=bonus, env_cfg=env_cfg)
            raise AssertionError(f"accepted where it must be refused: {message}")
        except ValueError as error:
            assert message in str(error), (message, str(error))
    cfg = PPOConfig(**{**SMALL, "n_envs": 2, "num_steps": 4})
    until = family(cfg, **UNTIL)
    params, state = until.init()
    try:
        until.post_rollout(params, state, random_observations(cfg, 8), None)
        raise AssertionError("scored without the first arrivals")
    except ValueError as error:
        assert "first arrival" in str(error), str(error)
    print("PASS refused without a fast last layer, with the scheduled last layer, without the "
          "episode log, without episodes that end at the goal, and without the first arrivals")


def test_antmaze_takes_the_option():
    """AntMaze: with no arrival the run is the fast last layer's bit for bit; a policy whose first
    arrival is in the log is scored in the next iteration as the plain program scores it, while
    the other policy is still scored with the fast last layer."""
    env_cfg = replace(preset("umaze"), continuing_task=False, max_episode_steps=6)
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=1, num_steps=4,
                    prime_iterations=1, update_style="epoch_minibatch", num_minibatches=2,
                    episode_log=True)
    lr = lambda runner, k: runner.lr_argument(k, 2)
    _, fast, _ = train_each(cfg, rnd(**FAST), env_cfg, iterations=2)
    runner = Runner(cfg, bonus=rnd(**UNTIL), env_cfg=env_cfg)
    state = runner.prime(runner.init_state(0))
    first = jax.device_get(runner.iterate(state, lr(runner, 1))[0])
    # the second iteration from the host copy of the first one's state: as it is, and with an
    # arrival written into policy 1's log, the latter also under the plain program
    log = first.agent_state["episode_log"]
    arrived = np.array(log["first_goal_step"])
    arrived[1] = 3
    marked = first._replace(agent_state={**first.agent_state,
                                         "episode_log": {**log, "first_goal_step": arrived}})
    second, second_metrics = (jax.device_get(part) for part in runner.iterate(first, lr(runner, 2)))
    after_mark, mark_metrics = (jax.device_get(part)
                                for part in runner.iterate(marked, lr(runner, 2)))
    plain = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
    _, plain_metrics = (jax.device_get(part) for part in plain.iterate(marked, lr(plain, 2)))

    # no arrival in two iterations of six-step episodes: the fast last layer's run, bit for bit
    assert (np.asarray(second.agent_state["episode_log"]["first_goal_step"]) < 0).all()
    for mine, theirs in ((first, fast[0]), (second, fast[1])):
        one, other = state_arrays(mine), state_arrays(theirs)
        assert one.keys() == other.keys()
        assert all(one[name].tobytes() == other[name].tobytes() for name in one)
    # with the arrival in the log: policy 1 gets the plain program's reward, policy 0 keeps the
    # fast last layer's; the two programs differ for both policies
    assert not np.isclose(second_metrics["rint_mean"], plain_metrics["rint_mean"],
                          rtol=1e-3).any()
    np.testing.assert_allclose(mark_metrics["rint_mean"][1], plain_metrics["rint_mean"][1],
                               rtol=1e-6)
    np.testing.assert_allclose(mark_metrics["rint_mean"][0], second_metrics["rint_mean"][0],
                               rtol=1e-6)
    assert same_bytes(divisor_rows(after_mark, 0), divisor_rows(second, 0))
    assert not same_bytes(divisor_rows(after_mark, 1), divisor_rows(second, 1))
    print("PASS AntMaze: the fast last layer's run without an arrival; a policy with an arrival "
          "in the log is scored by the predictor alone while the other keeps the fast last layer")


if __name__ == "__main__":
    test_the_hook_scores_arrived_policies_by_the_predictor_alone()
    test_each_policy_keeps_the_fast_layer_through_its_first_arrival_and_no_longer()
    test_it_works_with_the_divisor_held_after_the_first_arrival()
    test_saved_units_continue_exactly_before_and_after_the_switch()
    test_a_checkpoint_from_before_the_option_continues()
    test_what_is_refused()
    test_antmaze_takes_the_option()
