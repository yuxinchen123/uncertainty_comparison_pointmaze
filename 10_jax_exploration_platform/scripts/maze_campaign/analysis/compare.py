"""Choose a configuration on the development seeds, and compare a method with the baseline on fresh seeds.

- Choice: the highest whole-curve mean W; ties go to the higher final level L, then to the smaller number of
  environments per policy, then to the smaller value of the first setting that differs (settings in name
  order), so the choice never depends on the order the records were read in.
- Comparison: on the seeds both rows have, the per-seed difference (method minus baseline) in W and in L. Its
  95% interval is the 2.5% and 97.5% points of the means of 10,000 resamples of those seeds, drawn with
  replacement. Each quantity has its own generator, keyed by what is compared (rule rng-seeding.md).
- Verdict: the method beats the baseline when the W interval is entirely above zero and the L interval is not
  entirely below zero.
- Success: the method beats the baseline in all five environments in one reward setting.
- Secondary metric (discovery alone): the mean capped first-goal step M. Its choice is the smallest M; ties go to
  the larger share of policies that reach the goal, then to the same last keys as above. Its comparison is the
  per-seed difference in the capped first-goal step (method minus baseline) with the same interval; the method
  is ahead on discovery when that interval is entirely below zero, behind when it is entirely above zero. A
  method ahead on discovery in all five environments in one reward setting is noted; it is not part of success.
- Budget rule: of the three configurations with the highest W, the largest first step at which the smoothed
  seed-averaged curve reaches 95% of the final level; twice that, rounded up to a power of two, and at most the
  run's own length. A configuration that never reaches 95% counts as the whole run.
"""

import hashlib

import numpy as np

from . import measures
from .records import ENVIRONMENT_ORDER, REWARD_SETTING_ORDER

RESAMPLES = 10_000
MEASURES = {'whole_curve_mean': measures.whole_curve_mean, 'final_level': measures.final_level}
DISCOVERY_VERDICTS = {'ahead': 'ahead on discovery', 'level': 'within the interval', 'behind': 'behind on discovery'}


def seed_for(*parts):
    """A stable 32-bit seed for one named random quantity, independent of the order anything is computed in."""
    # before: ('maze_campaign', 'paired_difference', 'PointMaze Large'); after: the low 32 bits of the SHA-256 of
    # 'maze_campaign::paired_difference::PointMaze Large'.
    key = '::'.join(str(part) for part in parts)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def settings_order(settings):
    """A sort key for a configuration's settings: by name, then by value, values of different kinds never compared."""
    # before: (('int_coef', 4.0), ('preset', 'x')); after: (('int_coef', 'float', 4.0), ('preset', 'str', 'x')).
    return tuple((name, type(value).__name__, value) for name, value in settings)


def rank_configurations(configurations, budget):
    """Configurations from best to worst: by W, then L, then the smaller rollout width and settings."""
    scored = [(float(np.mean([measures.whole_curve_mean(policy, budget) for policy in group.policies])),
               float(np.mean([measures.final_level(policy, budget) for policy in group.policies])), group)
              for group in configurations]
    return sorted(scored, key=lambda row: (-row[0], -row[1], row[2].environments_per_policy,
                                           settings_order(row[2].settings)))


def choose_configuration(configurations, budget):
    """The configuration with the highest whole-curve mean at the budget (ties: the higher final level)."""
    return rank_configurations(configurations, budget)[0][2]


def rank_for_discovery(configurations, budget):
    """Configurations from the earliest discovery on: by M, then the larger share of policies that reach the goal."""
    # before: three configurations with M 300, 300 and 120 steps, of which 2, 4 and 4 of 4 policies reach the goal
    # after:  [(120.0, 1.0, third), (300.0, 1.0, second), (300.0, 0.5, first)]: equal M, so the larger share decides.
    scored = [(float(np.mean([measures.capped_first_goal_step(policy, budget) for policy in group.policies])),
               float(np.mean([measures.reached_goal(policy, budget) for policy in group.policies])), group)
              for group in configurations]
    return sorted(scored, key=lambda row: (row[0], -row[1], row[2].environments_per_policy,
                                           settings_order(row[2].settings)))


def choose_for_discovery(configurations, budget):
    """The configuration with the smallest mean capped first-goal step at the budget (ties: more policies arrive)."""
    return rank_for_discovery(configurations, budget)[0][2]


def best_environments_per_policy(configurations, budget):
    """The rollout width whose best configuration has the highest whole-curve mean at equal environment steps."""
    # before: configurations of 4, 16 and 128 environments per policy, three intrinsic weights each;
    # after:  (16, its best configuration) when the best of the three widths' own best W belongs to 16.
    best = choose_configuration(configurations, budget)
    return best.environments_per_policy, best


def propose_budget(configurations, run_steps, most=3):
    """The plan's budget rule on the configurations of one sweep that have trained the run's whole length."""
    # before: three configurations whose smoothed curves reach 95% of their final level at steps 200, 350 and 100
    #         in runs of 4,000 steps; after: largest 350, doubled 700, rounded up to 1,024, not capped: budget 1,024.
    ranked = [row[2] for row in rank_configurations(configurations, run_steps)][:most]
    reached = [measures.steps_to_final_level_share(group.policies, run_steps) for group in ranked]
    never = [group for group, step in zip(ranked, reached) if step is None]
    # A configuration that never reaches the share counts as the whole run, which puts the budget at the cap.
    largest = int(run_steps) if never else max(reached)
    rounded = 1 << (2 * largest - 1).bit_length()
    return {'configurations': list(zip(ranked, reached)), 'never': never, 'largest': largest, 'rounded': rounded,
            'capped': rounded > run_steps, 'budget': min(rounded, int(run_steps))}


def paired_differences(method_policies, baseline_policies, measure, budget):
    """Per-seed differences (method minus baseline) of one measure on the seeds both have, in seed order."""
    # before: method seeds {30000, 30001, 30002}, baseline seeds {30001, 30002, 30003};
    # after:  seeds [30001, 30002] and the two differences.
    first = {policy.seed_key: policy for policy in method_policies}
    second = {policy.seed_key: policy for policy in baseline_policies}
    common = sorted(first.keys() & second.keys())
    differences = np.asarray([measure(first[seed], budget) - measure(second[seed], budget) for seed in common])
    return common, differences


def difference_interval(differences, *name, resamples=RESAMPLES):
    """The mean of paired differences and its 95% interval from resampling the seeds; no interval under two seeds."""
    differences = np.asarray(differences, np.float64)
    if len(differences) < 2:
        return {'mean': float(differences.mean()) if len(differences) else None, 'interval': None}
    # One row of seed indices per resample, drawn with replacement; the interval is read off the row means.
    generator = np.random.default_rng(seed_for('maze_campaign', 'paired_difference', *name))
    chosen = generator.integers(0, len(differences), size=(resamples, len(differences)))
    low, high = np.quantile(differences[chosen].mean(axis=1), [0.025, 0.975])
    return {'mean': float(differences.mean()), 'interval': [float(low), float(high)]}


def side(interval):
    """Where a difference stands: 'ahead' when its interval is entirely above zero, 'behind' when below, else 'level'."""
    low, high = interval
    return 'ahead' if low > 0 else 'behind' if high < 0 else 'level'


def compare_with_baseline(method_policies, baseline_policies, budget, *name):
    """The paired differences in W and in L with their intervals and sides, and whether the method beats the baseline."""
    result = {}
    for quantity, measure in MEASURES.items():
        seeds, differences = paired_differences(method_policies, baseline_policies, measure, budget)
        result[quantity] = difference_interval(differences, *name, quantity, budget)
        result['seeds'] = len(seeds)
    if result['seeds'] < 2:
        return {**result, 'beats': None, 'verdict': 'fewer than two seeds in common'}
    for quantity in MEASURES:
        result[quantity]['side'] = side(result[quantity]['interval'])
    # Ahead on the whole curve, and not behind at the end.
    beats = result['whole_curve_mean']['side'] == 'ahead' and result['final_level']['side'] != 'behind'
    return {**result, 'beats': beats, 'verdict': 'beats' if beats else 'does not beat'}


def discovery_side(interval):
    """Where a difference in the first-goal step stands: 'ahead' when its interval is entirely below zero (earlier)."""
    # before: [-120.0, -30.0], [-50.0, 40.0], [10.0, 90.0]; after: 'ahead', 'level', 'behind'.
    low, high = interval
    return 'ahead' if high < 0 else 'behind' if low > 0 else 'level'


def compare_discovery(method_policies, baseline_policies, budget, *name):
    """The paired difference in the capped first-goal step with its interval and side, and the verdict on discovery."""
    seeds, differences = paired_differences(method_policies, baseline_policies, measures.capped_first_goal_step, budget)
    found = difference_interval(differences, *name, 'capped_first_goal_step', budget)
    if len(seeds) < 2:
        return {'capped_first_goal_step': found, 'seeds': len(seeds), 'ahead': None,
                'verdict': 'fewer than two seeds in common'}
    # A negative difference means the method first reached the goal earlier than the baseline.
    found['side'] = discovery_side(found['interval'])
    return {'capped_first_goal_step': found, 'seeds': len(seeds), 'ahead': found['side'] == 'ahead',
            'verdict': DISCOVERY_VERDICTS[found['side']]}


def study_success(beats):
    """Whether a method succeeds: it beats the baseline in all five environments in one reward setting.

    The secondary metric counts the same way, with "ahead on discovery" in place of "beats".
    """
    # before: {('PointMaze Large', 'arrival_plus_one'): True, ...}, a case not compared yet being absent;
    # after:  per reward setting the environments beaten, not beaten and not compared, and the overall answer.
    result = {}
    for setting in REWARD_SETTING_ORDER:
        result[setting] = {
            'beaten': [name for name in ENVIRONMENT_ORDER if beats.get((name, setting)) is True],
            'not_beaten': [name for name in ENVIRONMENT_ORDER if beats.get((name, setting)) is False],
            'not_compared': [name for name in ENVIRONMENT_ORDER if beats.get((name, setting)) is None]}
        result[setting]['success'] = len(result[setting]['beaten']) == len(ENVIRONMENT_ORDER)
    result['success'] = any(result[setting]['success'] for setting in REWARD_SETTING_ORDER)
    return result
