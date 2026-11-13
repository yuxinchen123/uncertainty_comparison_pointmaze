"""The second divisor for policies whose fast last layer has stopped
(`predictor_divisor_after_first_arrival=true` with `fast_last_layer_until_first_arrival=true`,
added 2026-11-13 after the diagnosis in the auto research folder,
`diagnosis/2026-11-13_slow-settling-after-the-fast-last-layer/notes.md`).

Without the option one discounted-sum filter and one running standard deviation follow whatever a
policy is rewarded with, so after its first arrival a policy is divided by a statistic built on the
fast last layer's scores, which is many times smaller than the one random network distillation
keeps. With the option a second filter and running standard deviation follow the predictor's own
scores of every rollout from the first one on, and a policy that arrived in an earlier rollout is
divided by that second statistic. Before its arrival a policy is unchanged. The tests use the small
PointMaze of the divisor held after the first arrival, whose goal lies next to the start.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "agents"))
sys.path.insert(0, str(ROOT / "tests" / "bonuses"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import make_bonus  # noqa: E402
from test_checkpoint_resume import check_resume  # noqa: E402
from test_rnd_bonus_options import rnd, same_bytes, state_arrays  # noqa: E402
from test_rnd_divisor_held_after_first_arrival import (NEAR_GOAL, PLANNED, SIX,  # noqa: E402
                                                       arrival_iterations, rows_of_copy,
                                                       train_each)

STEP = 0.3
UNTIL = dict(fast_last_layer_step=STEP, fast_last_layer_until_first_arrival=True)
SECOND = dict(**UNTIL, predictor_divisor_after_first_arrival=True)
HOLD = "held_after_first_arrival"


def divisor_of(statistics, copy: int) -> float:
    """The divisor the trainer forms from one policy's running statistics: sqrt(variance + 1e-8).

    before: variance 4.0 ; after: 2.0 (to rounding)
    """
    return float(np.sqrt(np.asarray(statistics.var, np.float64)[copy].reshape(()) + 1e-8))


def second_rows(state, copy: int) -> list:
    """One policy's rows of the second divisor's statistics (mean, variance, count) and filter."""
    return [np.asarray(leaf)[copy] for leaf in (*state.agent_state["int_rms_predictor"],
                                                state.agent_state["int_filter_predictor"])]


def shared_rows_equal(one, other, copy: int, n_copies: int) -> bool:
    """Whether two training states agree bit for bit on one policy's rows of every array both hold.

    before: one holds arrays {a, b}, other holds {a, b, c} ; after: a and b are compared, c is not
    """
    mine, theirs = rows_of_copy(one, copy, n_copies), rows_of_copy(other, copy, n_copies)
    return all(mine[name].tobytes() == theirs[name].tobytes() for name in mine.keys() & theirs.keys())


def test_a_policy_is_unchanged_before_its_arrival_and_divided_by_the_second_divisor_after():
    """In training with the running divisor: the option adds the second statistics to the state
    and nothing else; through the iteration of its own first arrival a policy is bit for bit the
    policy of the run without the option; from the next iteration on it is divided by the second
    divisor, while a policy that has not arrived is still divided by the first; the second
    statistics are the ones random network distillation keeps on the same scores."""
    cfg = PPOConfig(**SIX, episode_log=True, diagnostics=True)
    C = cfg.n_copies
    _, reference, _ = train_each(cfg, rnd(**UNTIL), NEAR_GOAL)
    runner, states, metrics = train_each(cfg, rnd(**SECOND), NEAR_GOAL)
    assert runner.bonus.keeps_predictor_divisor and runner.bonus.reads_first_arrival

    # what the option adds to the state: the second filter and the three arrays of its statistics
    added = sorted(set(state_arrays(states[-1])) - set(state_arrays(reference[-1])))
    assert len(added) == 4 and all("predictor" in name for name in added), added
    assert set(state_arrays(reference[-1])) <= set(state_arrays(states[-1]))

    # the set-up: one policy arrives with iterations left, another has not arrived by then
    arrivals = arrival_iterations(reference[-1])
    reached = [arrival for arrival in arrivals if arrival is not None and arrival < PLANNED]
    assert reached, arrivals
    switch = min(reached)
    early = [copy for copy, arrival in enumerate(arrivals) if arrival == switch]
    late = [copy for copy, arrival in enumerate(arrivals) if arrival is None or arrival > switch]
    assert early and late, arrivals

    # unchanged through the iteration of the first arrival, changed by the next one
    for copy, arrival in enumerate(arrivals):
        last_same = PLANNED if arrival is None else arrival
        for index in range(last_same):
            assert shared_rows_equal(states[index], reference[index], copy, C), (copy, index + 1)
        if arrival is not None and arrival < PLANNED:
            assert not shared_rows_equal(states[arrival], reference[arrival], copy, C), copy

    # the divisor used in the iteration after the first arrival, read from the diagnostics
    after = metrics[switch]["diagnostics"]
    for copy in early:
        second = divisor_of(states[switch].agent_state["int_rms_predictor"], copy)
        first = divisor_of(states[switch].agent_state["int_rms"], copy)
        np.testing.assert_allclose(after["int_std"][copy], second, rtol=1e-5)
        assert not np.isclose(first, second, rtol=1e-2), (copy, first, second)
        # the reward the learner saw is the predictor's own score over the second divisor
        np.testing.assert_allclose(after["rint_hat_mean"][copy],
                                   metrics[switch]["rint_mean"][copy] / second, rtol=1e-4)
    for copy in late:
        first = divisor_of(states[switch].agent_state["int_rms"], copy)
        np.testing.assert_allclose(after["int_std"][copy], first, rtol=1e-5)

    # the second statistics after the first iteration are random network distillation's own:
    # both programs start from the same weights, so their first rollouts are the same
    _, plain, _ = train_each(cfg, "rnd_next_state", NEAR_GOAL, iterations=1)
    for mine, theirs in zip(states[0].agent_state["int_rms_predictor"],
                            plain[0].agent_state["int_rms"]):
        np.testing.assert_allclose(np.asarray(mine), np.asarray(theirs), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(states[0].agent_state["int_filter_predictor"]),
                               np.asarray(plain[0].agent_state["int_filter"]), rtol=1e-5, atol=1e-9)
    # and they move in every iteration for every policy, arrived or not
    for copy in range(C):
        for index in range(1, PLANNED):
            assert not same_bytes(second_rows(states[index], copy),
                                  second_rows(states[index - 1], copy)), (copy, index + 1)
    print(f"PASS second divisor with the running statistics: first arrivals at iterations "
          f"{arrivals}; in iteration {switch + 1} policies {early} are divided by the divisor of "
          f"the predictor's own scores and policies {late} by the divisor of the fast last "
          f"layer's scores; the state gains {added}")


def test_with_the_hold_the_second_divisor_stays_from_the_first_arrival():
    """Together with `reward_divisor=held_after_first_arrival`: a policy's second statistics move
    through the iteration of its first arrival and stay afterwards, and that held value is what
    it is divided by; a policy with no arrival keeps moving statistics."""
    cfg = PPOConfig(**SIX, episode_log=True, diagnostics=True)
    C = cfg.n_copies
    runner, states, metrics = train_each(cfg, rnd(**SECOND, reward_divisor=HOLD), NEAR_GOAL)
    assert runner.bonus.keeps_predictor_divisor and runner.bonus.reward_divisor == HOLD
    arrivals = arrival_iterations(states[-1])
    assert any(arrival is not None and arrival < PLANNED for arrival in arrivals), arrivals
    assert any(arrival is None for arrival in arrivals), arrivals
    for copy, arrival in enumerate(arrivals):
        last_moved = PLANNED if arrival is None else arrival
        for index in range(1, PLANNED):
            moved = not same_bytes(second_rows(states[index], copy),
                                   second_rows(states[index - 1], copy))
            assert moved == (index < last_moved), (copy, index + 1, arrival)
        if arrival is not None and arrival < PLANNED:
            held = divisor_of(states[arrival - 1].agent_state["int_rms_predictor"], copy)
            for index in range(arrival, PLANNED):
                np.testing.assert_allclose(metrics[index]["diagnostics"]["int_std"][copy], held,
                                           rtol=1e-5)
    print(f"PASS second divisor with the hold: first arrivals at iterations {arrivals}; each "
          f"policy's second statistics stay from the iteration after its arrival ({C} policies)")


def test_saved_units_continue_exactly_before_and_after_the_switch():
    """A unit saved before any policy has arrived, and a unit saved after one policy has switched
    to the second divisor and before another arrives, both continue to the uninterrupted unit bit
    for bit, with the running statistics and with the hold."""
    cfg = PPOConfig(**SIX, episode_log=True)
    for bonus in (rnd(**SECOND), rnd(**SECOND, reward_divisor=HOLD)):
        # the first arrivals of the unit the resume check trains (its run seed is 17)
        _, states, _ = train_each(cfg, bonus, NEAR_GOAL, seed=17)
        arrivals = [arrival for arrival in arrival_iterations(states[-1]) if arrival is not None]
        first, last = min(arrivals), max(arrivals)
        assert 2 <= first and first + 1 < last <= PLANNED, arrivals
        check_resume(cfg, NEAR_GOAL, bonus, total=PLANNED, stop_after=first - 1)
        check_resume(cfg, NEAR_GOAL, bonus, total=PLANNED, stop_after=first + 1)
    print("PASS saved units continue exactly before and after the switch, with the running "
          "statistics and with the hold")


def test_what_is_refused():
    """Without the fast last layer that stops at the first arrival, with a divisor the second
    statistics are not built for, and with the record on the maze grid, the option is refused."""
    option = {"predictor_divisor_after_first_arrival": True}
    until = {"fast_last_layer_step": STEP, "fast_last_layer_until_first_arrival": True}
    for options, message in (
            (option, "fast_last_layer_until_first_arrival with it"),
            ({**option, "fast_last_layer_step": STEP}, "fast_last_layer_until_first_arrival with it"),
            ({**option, **until, "reward_divisor": "frozen_after_priming"}, "is built for"),
            ({**option, **until, "reward_divisor": "none"}, "is built for"),
            ({**option, **until, "reward_divisor": "held_after_fraction",
              "divisor_hold_fraction": 0.5}, "is built for"),
            ({**option, **until, "record_bonus_on_maze_grid": True}, "keeps two")):
        try:
            make_bonus("rnd_next_state", options)
            raise AssertionError(f"accepted: {options}")
        except ValueError as error:
            assert message in str(error), (message, str(error))
    # the accepted combinations build
    for options in ({**option, **until}, {**option, **until, "reward_divisor": HOLD}):
        make_bonus("rnd_next_state", options)
    print("PASS refused without the fast last layer that stops at the first arrival, with the "
          "divisors frozen after priming, held after a fraction and none, and with the maze-grid "
          "record")


if __name__ == "__main__":
    test_a_policy_is_unchanged_before_its_arrival_and_divided_by_the_second_divisor_after()
    test_with_the_hold_the_second_divisor_stays_from_the_first_arrival()
    test_saved_units_continue_exactly_before_and_after_the_switch()
    test_what_is_refused()
