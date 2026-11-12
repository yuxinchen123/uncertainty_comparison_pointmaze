"""Tests of the study's measures on hand-made policies whose answers are computed by hand."""

import numpy as np
import pytest

from maze_record_examples import flat_policy, make_policy
from maze_campaign.analysis import measures


def example():
    """Four windows of 100 steps: no episode, 3 of 4 arrive, no episode, 2 of 2 arrive."""
    return make_policy([100, 200, 300, 400], [0, 4, 0, 2], [0, 3, 0, 2], steps_to_goal=[0, 90, 0, 40],
                       coverage=[0.1, 0.5, 0.5, 1.0], first_goal_step=150)


def test_training_curve_fills_empty_windows():
    """A window with no finished episode takes the last value; before any episode, the never-reached return."""
    assert measures.training_curve(example()).tolist() == [0.0, 0.75, 0.75, 1.0]
    # With the step cost an episode that never reaches the goal returns -1, so that is the starting value.
    cost = make_policy([100, 200, 300], [0, 0, 2], [0, 0, -0.5], arrivals=[0, 0, 2], reward_setting='step_cost')
    assert measures.training_curve(cost).tolist() == [-1.0, -1.0, -0.25]


def test_whole_curve_mean_weights_windows_by_their_steps_up_to_the_budget():
    """W is the area under the step curve divided by the budget, also when the budget cuts a window."""
    policy = example()
    assert measures.whole_curve_mean(policy, 400) == pytest.approx((0 + 75 + 75 + 100) / 400)
    # Budget 250: all of windows 1 and 2, and half of window 3.
    assert measures.whole_curve_mean(policy, 250) == pytest.approx((0 + 75 + 0.75 * 50) / 250)
    # Windows of unequal length: 100 steps at 0.2, then 300 steps at 0.6.
    uneven = make_policy([100, 400], [10, 10], [2, 6])
    assert measures.whole_curve_mean(uneven, 400) == pytest.approx((100 * 0.2 + 300 * 0.6) / 400)


def test_final_level_is_the_mean_over_the_last_tenth_of_the_budget():
    """L reads steps (0.9 B, B]; a window that straddles 0.9 B counts by its steps inside."""
    policy = example()
    assert measures.final_level(policy, 400) == pytest.approx(1.0)
    assert measures.final_level(policy, 200) == pytest.approx(0.75)
    # Budget 200 with windows ending at 100, 190, 200: the last tenth is steps 181 to 200, half in each window.
    straddling = make_policy([100, 190, 200], [5, 5, 5], [0, 1, 4])
    assert measures.final_level(straddling, 200) == pytest.approx((0.2 * 10 + 0.8 * 10) / 20)


def test_a_budget_the_records_do_not_reach_is_refused():
    """Every measure raises when the policy has not trained to the budget."""
    with pytest.raises(ValueError, match='cannot be read'):
        measures.whole_curve_mean(example(), 401)
    with pytest.raises(ValueError, match='cannot be read'):
        measures.capped_first_goal_step(example(), 500)


def test_first_goal_step_is_capped_at_the_budget():
    """An arrival after the budget, or none at all, counts as the budget."""
    policy = example()
    assert measures.capped_first_goal_step(policy, 400) == 150.0 and measures.reached_goal(policy, 400)
    assert measures.capped_first_goal_step(policy, 100) == 100.0 and not measures.reached_goal(policy, 100)
    never = make_policy([100, 200], [1, 1], [0, 0])
    assert measures.capped_first_goal_step(never, 200) == 200.0 and not measures.reached_goal(never, 200)


def test_coverage_is_read_at_the_last_window_end_inside_the_budget():
    """Coverage at a budget inside window 3 is the value recorded at the end of window 2."""
    policy = example()
    assert measures.coverage_at(policy, 250) == 0.5 and measures.coverage_at(policy, 400) == 1.0
    assert measures.coverage_at(make_policy([100], [1], [0]), 100) is None
    with pytest.raises(ValueError, match='before the first logging window'):
        measures.coverage_at(policy, 50)


def test_episode_sums_split_a_straddling_window_in_proportion():
    """Half of a window inside the range counts half of its episodes and arrivals."""
    policy = make_policy([100, 200], [10, 20], [4, 8], steps_to_goal=[40, 160])
    sums = measures.episode_sums(policy, 150, 200)
    assert sums == {'episodes': 10.0, 'arrivals': 4.0, 'steps_to_goal_sum': 80.0}
    # The last tenth of 200 steps lies in window 2: 8 of its 20 episodes reach the goal.
    assert measures.final_goal_share(policy, 200) == pytest.approx(0.4)
    assert measures.final_goal_share(make_policy([100, 200], [10, 0], [4, 0]), 200) is None


def test_mean_curve_averages_policies_on_the_union_of_their_window_ends():
    """Two policies with different windows are averaged at every window end of either, up to the budget."""
    first = make_policy([100, 200, 300], [1, 1, 1], [0.0, 1.0, 1.0])
    second = make_policy([150, 300], [2, 2], [1.0, 2.0])
    grid, mean, error = measures.mean_curve([first, second], 250)
    assert grid.tolist() == [100, 150, 200, 250]
    assert mean.tolist() == [0.25, 0.75, 1.0, 1.0]
    assert error.tolist() == pytest.approx([0.25, 0.25, 0.0, 0.0])
    # Without a budget the curve runs to the largest step both policies have.
    assert measures.common_steps([first, make_policy([150], [1], [1])]) == 150


def test_configuration_measures_on_two_seeds():
    """W, L, the peak, the capped first-goal step and the shares of a two-seed configuration."""
    first = flat_policy([0.0, 0.5, 1.0, 0.8], seed=0, first_goal_step=120, coverage=[0.2, 0.6, 1.0, 1.0],
                        steps_to_goal=[0, 250, 400, 320])
    second = flat_policy([0.0, 0.0, 0.0, 0.0], seed=1, coverage=[0.2, 0.2, 0.2, 0.2])
    row = measures.configuration_measures([first, second], 400)
    assert row['seeds'] == 2 and row['whole_curve_mean'] == pytest.approx((0.575 + 0.0) / 2)
    assert row['whole_curve_mean_error'] == pytest.approx(np.std([0.575, 0.0], ddof=1) / np.sqrt(2))
    assert row['final_level'] == pytest.approx(0.4) and row['final_level_at'][0.5] == pytest.approx(0.25)
    # The seed-averaged curve is [0, 0.25, 0.5, 0.4]: its peak is 0.5 at step 300, and L is 80% of it.
    assert row['peak'] == pytest.approx(0.5) and row['peak_step'] == 300
    assert row['final_over_peak'] == pytest.approx(0.8)
    assert row['capped_first_goal_step'] == pytest.approx((120 + 400) / 2) and row['reached_goal_share'] == 0.5
    assert row['coverage'] == pytest.approx(0.6) and row['final_goal_share'] == pytest.approx(0.4)
    # In the last tenth the first seed's 8 arrivals of window 4 took 320 steps together: 40 steps each.
    assert row['steps_to_goal'] == pytest.approx(40.0)


def test_final_over_peak_is_measured_from_the_never_reached_return():
    """With the step cost the floor is -1: a curve from -1 up to -0.2 and back to -0.6 keeps half its rise."""
    policy = make_policy([100, 200, 300], [10, 10, 10], [-10, -2, -6], arrivals=[0, 10, 5], reward_setting='step_cost',
                         steps_to_goal=[0, 800, 600])
    row = measures.configuration_measures([policy], 300)
    assert row['peak'] == pytest.approx(-0.2) and row['final_level'] == pytest.approx(-0.6)
    assert row['final_over_peak'] == pytest.approx(0.4 / 0.8)
    assert row['whole_curve_mean_error'] is None and row['steps_to_goal'] == pytest.approx(120.0)
    # A curve that never leaves the floor has no ratio.
    flat = make_policy([100], [10], [-10], arrivals=[0], reward_setting='step_cost')
    assert measures.configuration_measures([flat], 100)['final_over_peak'] is None
