"""Tests of the choice of a configuration, the paired interval, each verdict, and the success rule."""

import hashlib

import numpy as np
import pytest

from maze_record_examples import flat_policy
from maze_campaign.analysis import compare, measures, records


def configuration(values, weight, width=4):
    """A two-seed configuration whose seeds both follow the given curve, named by its intrinsic weight."""
    settings = (('ent_coef', 0.001), ('int_coef', weight), ('learning_rate', 1e-4))
    return records.configurations([flat_policy(values, seed=seed, settings=settings, width=width) for seed in (0, 1)])[0]


def seeds(values_per_seed, first_seed=30000, **more):
    """One policy per listed curve, on consecutive confirmation seeds."""
    return [flat_policy(values, seed=first_seed + index, **more) for index, values in enumerate(values_per_seed)]


def test_choice_takes_the_highest_whole_curve_mean():
    """The configuration with the largest W is chosen, whatever its final level."""
    low, high = configuration([0.1] * 10, 1.0), configuration([0.9] * 9 + [0.0], 4.0)
    assert compare.choose_configuration([low, high], 1000) is high
    assert [row[2] for row in compare.rank_configurations([low, high], 1000)] == [high, low]


def test_choice_breaks_a_tie_in_w_by_the_final_level_then_by_the_settings():
    """Equal W: the higher L wins; equal W and L: the configuration whose settings sort first."""
    rising, falling = configuration([0.0] * 5 + [1.0] * 5, 1.0), configuration([1.0] * 5 + [0.0] * 5, 4.0)
    assert measures.whole_curve_mean(rising.policies[0], 1000) == measures.whole_curve_mean(falling.policies[0], 1000)
    assert compare.choose_configuration([falling, rising], 1000) is rising
    same_a, same_b = configuration([0.5] * 10, 16.0), configuration([0.5] * 10, 4.0)
    assert compare.choose_configuration([same_a, same_b], 1000) is same_b
    assert compare.choose_configuration([same_b, same_a], 1000) is same_b


def test_best_environments_per_policy_is_the_width_of_the_best_configuration():
    """Among widths 4, 16 and 128 with two weights each, the width holding the highest W is the best."""
    groups = [configuration([0.2] * 10, 1.0, width=4), configuration([0.3] * 10, 4.0, width=4),
              configuration([0.1] * 10, 1.0, width=16), configuration([0.6] * 10, 4.0, width=16),
              configuration([0.5] * 10, 1.0, width=128), configuration([0.4] * 10, 4.0, width=128)]
    width, best = compare.best_environments_per_policy(groups, 1000)
    assert width == 16 and best is groups[3]


def rising(first_window, windows, weight):
    """A two-seed configuration whose curve is 0 before window `first_window` (from 0) and 1 from it on."""
    return configuration([0.0] * first_window + [1.0] * (windows - first_window), weight)


def test_budget_rule_doubles_the_largest_step_and_rounds_up_to_a_power_of_two():
    """Runs of 4,000 steps: the three best reach 95% at steps 200, 300 and 400, so the budget is 1,024."""
    groups = [rising(0, 40, 1.0), rising(1, 40, 2.0), rising(2, 40, 4.0), rising(20, 40, 8.0)]
    proposal = compare.propose_budget(groups, 4000)
    # The range of 200 steps holds two windows; the configuration that rises at window 21 is not among the three.
    assert [(group, step) for group, step in proposal['configurations']] == [
        (groups[0], 200), (groups[1], 300), (groups[2], 400)]
    assert (proposal['largest'], proposal['rounded'], proposal['budget']) == (400, 1024, 1024)
    assert proposal['capped'] is False and proposal['never'] == []


def test_budget_rule_is_capped_at_the_run_and_says_when_a_configuration_never_arrives():
    """A budget above the run's length is cut to it; a configuration at the never-reached return counts as the run."""
    groups = [rising(0, 10, 1.0), rising(1, 10, 2.0), rising(6, 10, 4.0)]
    proposal = compare.propose_budget(groups, 1000)
    assert [step for _, step in proposal['configurations']] == [100, 200, 700]
    assert (proposal['largest'], proposal['rounded'], proposal['budget'], proposal['capped']) == (700, 2048, 1000, True)
    # Twice 256 would stay 512; here twice 300 is 600 and rounds up to 1,024, still above the run of 1,000 steps.
    assert compare.propose_budget([rising(2, 10, 1.0)], 1000)['rounded'] == 1024
    flat = configuration([0.0] * 10, 8.0)
    proposal = compare.propose_budget([groups[0], groups[1], flat], 1000)
    assert proposal['never'] == [flat] and proposal['configurations'][2] == (flat, None)
    assert (proposal['largest'], proposal['budget'], proposal['capped']) == (1000, 1000, True)


def test_seed_for_is_the_low_32_bits_of_the_sha256_of_the_joined_parts():
    """The seed depends only on the names, so adding another quantity never moves an existing one."""
    expected = int(hashlib.sha256(b'maze_campaign::W::30000').hexdigest(), 16) & 0xFFFFFFFF
    assert compare.seed_for('maze_campaign', 'W', 30000) == expected
    assert compare.seed_for('maze_campaign', 'L', 30000) != expected


def test_paired_differences_use_only_the_seeds_both_rows_have():
    """Method seeds 30000-30003 and baseline seeds 30002-30005 share two seeds; differences are method minus baseline."""
    method = seeds([[0.5] * 10] * 4)
    baseline = seeds([[0.1] * 10, [0.2] * 10, [0.3] * 10, [0.4] * 10], first_seed=30002)
    common, differences = compare.paired_differences(method, baseline, measures.whole_curve_mean, 1000)
    assert [seed for _, seed in common] == [30002, 30003]
    assert differences.tolist() == pytest.approx([0.4, 0.3])


def test_interval_of_a_constructed_case():
    """Differences 0 and 1: a resample's mean is 0, 0.5 or 1 with chances 1/4, 1/2, 1/4, so the interval is [0, 1]."""
    result = compare.difference_interval([0.0, 1.0], 'constructed case')
    assert result == {'mean': 0.5, 'interval': [0.0, 1.0]}
    # Equal differences give an interval of zero width; the same names give the same resamples.
    assert compare.difference_interval([0.25] * 8, 'equal')['interval'] == [0.25, 0.25]
    noisy = np.linspace(-1.0, 2.0, 40)
    assert compare.difference_interval(noisy, 'noisy') == compare.difference_interval(noisy, 'noisy')
    low, high = compare.difference_interval(noisy, 'noisy')['interval']
    assert low < noisy.mean() < high and high - low < 3.0
    # One seed has a mean and no interval; no seed has neither.
    assert compare.difference_interval([0.3], 'one') == {'mean': 0.3, 'interval': None}
    assert compare.difference_interval([], 'none') == {'mean': None, 'interval': None}


def test_side_of_an_interval():
    """Entirely above zero is ahead, entirely below is behind, anything touching zero is level."""
    assert compare.side([0.01, 0.2]) == 'ahead' and compare.side([-0.2, -0.01]) == 'behind'
    assert compare.side([-0.1, 0.1]) == 'level' and compare.side([0.0, 0.2]) == 'level'


BASELINE = [[0.2] * 10, [0.3] * 10, [0.4] * 10]


def verdict_of(method_curves):
    """Compare three method seeds with the three baseline seeds at 1,000 steps."""
    return compare.compare_with_baseline(seeds(method_curves), seeds(BASELINE), 1000, 'PointMaze Large', 'test')


def test_verdict_beats_when_w_is_ahead_and_l_is_ahead_or_level():
    """Ahead in W on every seed: beats when L is also ahead, and when L is level."""
    ahead = verdict_of([[0.3] * 10, [0.4] * 10, [0.5] * 10])
    assert ahead['seeds'] == 3 and ahead['whole_curve_mean']['mean'] == pytest.approx(0.1)
    assert ahead['whole_curve_mean']['interval'] == pytest.approx([0.1, 0.1])
    assert (ahead['whole_curve_mean']['side'], ahead['final_level']['side']) == ('ahead', 'ahead')
    assert ahead['beats'] is True and ahead['verdict'] == 'beats'
    # Higher for nine windows, equal to the baseline in the last tenth: L is level.
    level = verdict_of([[0.5] * 9 + [0.2], [0.5] * 9 + [0.3], [0.5] * 9 + [0.4]])
    assert (level['whole_curve_mean']['side'], level['final_level']['side']) == ('ahead', 'level')
    assert level['beats'] is True


def test_verdict_does_not_beat_when_l_is_behind_or_w_is_not_ahead():
    """Ahead in W but behind in L does not beat; neither does a W that is level or behind."""
    falls = verdict_of([[0.9] * 9 + [0.0], [0.9] * 9 + [0.1], [0.9] * 9 + [0.2]])
    assert (falls['whole_curve_mean']['side'], falls['final_level']['side']) == ('ahead', 'behind')
    assert falls['beats'] is False and falls['verdict'] == 'does not beat'
    # One seed higher, one lower, one equal: the interval of the W difference holds zero.
    mixed = verdict_of([[0.4] * 10, [0.1] * 10, [0.4] * 10])
    assert mixed['whole_curve_mean']['side'] == 'level' and mixed['beats'] is False
    behind = verdict_of([[0.1] * 10, [0.2] * 10, [0.3] * 10])
    assert behind['whole_curve_mean']['side'] == 'behind' and behind['beats'] is False


def test_verdict_needs_two_seeds_in_common():
    """With one common seed there is no interval and no verdict."""
    result = compare.compare_with_baseline(seeds([[0.9] * 10]), seeds(BASELINE), 1000, 'one seed')
    assert result['seeds'] == 1 and result['beats'] is None and result['verdict'] == 'fewer than two seeds in common'


def test_study_success_needs_all_five_environments_in_one_reward_setting():
    """Five of five in one setting succeeds; four of five in each does not; a missing case is not compared."""
    every = {(name, 'arrival_plus_one'): True for name in records.ENVIRONMENT_ORDER}
    assert compare.study_success(every)['success'] is True
    four = {**every, ('AntMaze Large', 'arrival_plus_one'): False,
            **{(name, 'step_cost'): True for name in records.ENVIRONMENT_ORDER[:4]}}
    outcome = compare.study_success(four)
    assert outcome['success'] is False
    assert outcome['arrival_plus_one']['not_beaten'] == ['AntMaze Large']
    assert outcome['step_cost']['not_compared'] == ['AntMaze Large'] and len(outcome['step_cost']['beaten']) == 4
    assert compare.study_success({**four, ('AntMaze Large', 'step_cost'): True})['success'] is True


def arrivals(first_steps, weight=1.0, first_seed=0):
    """One policy per listed first-arrival step (-1: none) with four windows of 100 steps, named by its intrinsic weight."""
    settings = (('ent_coef', 0.001), ('int_coef', weight), ('learning_rate', 1e-4))
    return [flat_policy([0, 0, 0, 0], seed=first_seed + index, first_goal_step=step, settings=settings)
            for index, step in enumerate(first_steps)]


def test_discovery_ranks_by_the_mean_capped_first_goal_step_and_counts_no_arrival_as_the_budget():
    """M is the mean of the first-arrival steps with a policy that never arrives counted as the 400 steps of the budget."""
    groups = records.configurations(arrivals([100, -1], 1.0) + arrivals([200, 300], 4.0) + arrivals([120, 380], 16.0))
    ranked = compare.rank_for_discovery(groups, 400)
    # Weight 1: (100 + 400) / 2 = 250 with one of two policies arriving; weights 4 and 16: 250 with both arriving.
    assert [(capped, share, group.setting('int_coef')) for capped, share, group in ranked] == [
        (250.0, 1.0, 4.0), (250.0, 1.0, 16.0), (250.0, 0.5, 1.0)]
    # Equal M: the larger share of policies that reach the goal stands first, then the smaller setting value.
    assert compare.choose_for_discovery(groups, 400).setting('int_coef') == 4.0
    # At a budget of 150 steps only the arrivals at steps 100 and 120 count: weight 1 has M = 125, weight 16 has 135.
    assert [group.setting('int_coef') for _, _, group in compare.rank_for_discovery(groups, 150)] == [1.0, 16.0, 4.0]
    assert compare.rank_for_discovery(groups, 150)[2][:2] == (150.0, 0.0)
    # The choice by M can differ from the choice by W: here every curve is flat at zero, so W takes the smallest setting.
    assert compare.choose_configuration(groups, 400).setting('int_coef') == 1.0


def test_a_difference_in_the_first_goal_step_is_ahead_when_its_interval_is_entirely_below_zero():
    """The difference is method minus baseline, so a negative one means the method reached the goal earlier."""
    baseline = arrivals([300, 320, 340], first_seed=30000)
    earlier = compare.compare_discovery(arrivals([100, 220, 140], first_seed=30000), baseline, 400, 'test')
    assert earlier['seeds'] == 3 and earlier['capped_first_goal_step']['mean'] == pytest.approx(-500 / 3)
    low, high = earlier['capped_first_goal_step']['interval']
    assert -200 <= low <= high <= -100 and earlier['capped_first_goal_step']['side'] == 'ahead'
    assert earlier['ahead'] is True and earlier['verdict'] == 'ahead on discovery'
    # Later on every seed (one policy never arrives and counts as the budget): entirely above zero, so behind.
    later = compare.compare_discovery(arrivals([350, -1, 390], first_seed=30000), baseline, 400, 'test')
    assert later['capped_first_goal_step']['mean'] == pytest.approx((50 + 80 + 50) / 3)
    assert later['capped_first_goal_step']['interval'][0] > 0 and later['ahead'] is False
    assert later['verdict'] == 'behind on discovery'
    # Earlier on one seed and later on another: the interval holds zero.
    mixed = compare.compare_discovery(arrivals([200, 420, 340], first_seed=30000), baseline, 400, 'test')
    assert mixed['capped_first_goal_step']['side'] == 'level' and mixed['verdict'] == 'within the interval'
    assert mixed['ahead'] is False
    # One seed in common: no interval and no verdict.
    one = compare.compare_discovery(arrivals([100], first_seed=30002), baseline, 400, 'test')
    assert one['seeds'] == 1 and one['ahead'] is None and one['verdict'] == 'fewer than two seeds in common'
    assert compare.discovery_side([-120.0, -30.0]) == 'ahead' and compare.discovery_side([10.0, 90.0]) == 'behind'
    assert compare.discovery_side([-50.0, 40.0]) == 'level' and compare.discovery_side([-50.0, 0.0]) == 'level'
    # The same count as the success rule serves "ahead on discovery in all five environments".
    every = {(name, 'step_cost'): True for name in records.ENVIRONMENT_ORDER}
    assert compare.study_success(every)['success'] is True
    assert compare.study_success({**every, ('AntMaze Large', 'step_cost'): None})['success'] is False
