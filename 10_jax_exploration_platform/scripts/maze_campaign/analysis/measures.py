"""The study's measures, per policy and per configuration, at any budget up to the steps a policy has trained.

A policy's training curve is a step function: on the steps of logging window w it has the value R_w, the mean
return (in the reward the learner sees) of the training episodes that ended in that window. Every mean over a
range of steps weights each window by its steps inside the range, so a window that straddles an end of the
range counts in proportion. Sums of episodes are split the same way.

- Whole-curve mean W(B): the mean of the curve over steps (0, B].
- Final level L(B): the mean of the curve over the last 10% of the budget, steps (0.9 B, B].
- A budget shorter than the run reads the same records up to that step, so L(0.5 B) is the mean over
  (0.45 B, 0.5 B] of the run trained for B steps.
- A smoothed curve is the seed-averaged curve averaged over a fixed number of steps that end at each logged
  step. The peak of the decline check is the highest value of the curve smoothed over 10% of the budget, the
  same width as the final level, so the smoothed value at B is L(B) and the peak is never below it. The budget
  rule smooths over 5% of the run's steps and looks for the first step that reaches 95% of the final level.
"""

import numpy as np

FINAL_SHARE = 0.1
LENGTH_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
# The budget rule of the plan: the curve smoothed over this share of the run's steps, and the share of the
# final level (measured from the return of an episode that never reaches the goal) it has to reach.
BUDGET_SMOOTHING_SHARE = 0.05
BUDGET_LEVEL_SHARE = 0.95


def require_steps(policy, budget):
    """Refuse a budget the policy's records do not reach."""
    if budget <= 0 or policy.steps[-1] < budget:
        raise ValueError(f'Policy seed {policy.seed} of unit {policy.unit_id} has records to step '
                         f'{int(policy.steps[-1])}; a budget of {budget} steps cannot be read from them.')


def steps_inside(steps, low, high):
    """How many environment steps of each logging window lie in the range (low, high]."""
    # before: window ends [100, 200, 300], range (150, 250]; after: [0, 50, 50].
    starts = np.concatenate([[0], steps[:-1]])
    return np.clip(np.minimum(steps, high) - np.maximum(starts, low), 0, None).astype(np.float64)


def training_curve(policy):
    """Mean return of the episodes that end in each window; a window with none takes the last value."""
    # before: return sums [0, 3, 0, 2], episodes [0, 4, 0, 2], reward 1 on arrival
    # after:  [0, 0.75, 0.75, 1]: window 1 has no episode yet and takes the return of an episode that never
    #         reaches the goal (0 here, -1 with the step cost); window 3 repeats window 2.
    filled = policy.episodes > 0
    value = np.where(filled, policy.return_sum / np.maximum(policy.episodes, 1), 0.0)
    last = np.maximum.accumulate(np.where(filled, np.arange(len(filled)), -1))
    return np.where(last >= 0, value[np.maximum(last, 0)], policy.never_reached_return)


def curve_mean(policy, low, high):
    """Step-weighted mean of the policy's training curve over the steps in (low, high]."""
    require_steps(policy, high)
    return float(steps_inside(policy.steps, low, high) @ training_curve(policy) / (high - low))


def whole_curve_mean(policy, budget):
    """W(B) of one policy: the area under its training curve up to the budget, divided by the budget."""
    return curve_mean(policy, 0, budget)


def final_level(policy, budget):
    """L(B) of one policy: the mean of its training curve over the last 10% of the budget."""
    return curve_mean(policy, budget - FINAL_SHARE * budget, budget)


def reached_goal(policy, budget):
    """Whether the policy's first arrival at the goal is at or before the budget."""
    require_steps(policy, budget)
    return 0 <= policy.first_goal_step <= budget


def capped_first_goal_step(policy, budget):
    """The training step of the first arrival; a policy with none by the budget counts as the budget."""
    return float(policy.first_goal_step) if reached_goal(policy, budget) else float(budget)


def coverage_at(policy, budget):
    """Share of open maze cells reached by the last window end at or before the budget; None when not recorded."""
    require_steps(policy, budget)
    if policy.coverage is None:
        return None
    index = int(np.searchsorted(policy.steps, budget, side='right')) - 1
    if index < 0:
        raise ValueError(f'A budget of {budget} steps ends before the first logging window of unit {policy.unit_id}.')
    return float(policy.coverage[index])


def episode_sums(policy, low, high):
    """Episodes, arrivals and steps-to-goal sum of the range (low, high], a straddling window in proportion."""
    # before: windows of 100 steps with arrivals [4, 8], range (150, 200]; after: arrivals 4 (half of window 2).
    require_steps(policy, high)
    lengths = np.diff(policy.steps, prepend=0)
    share = steps_inside(policy.steps, low, high) / lengths
    return {name: float(share @ getattr(policy, name)) for name in ('episodes', 'arrivals', 'steps_to_goal_sum')}


def final_goal_share(policy, budget):
    """Share of the policy's training episodes in the last 10% of the budget that reach the goal; None with no episode."""
    sums = episode_sums(policy, budget - FINAL_SHARE * budget, budget)
    return sums['arrivals'] / sums['episodes'] if sums['episodes'] > 0 else None


def common_steps(policies):
    """The largest step count every one of the policies has records for."""
    return int(min(policy.steps[-1] for policy in policies))


def mean_curve(policies, budget=None):
    """The seed-averaged training curve up to the budget (default: the common steps): steps, mean, standard error."""
    limit = common_steps(policies) if budget is None else budget
    for policy in policies:
        require_steps(policy, limit)
    # The grid is every window end of any policy up to the limit, and the limit itself.
    # before: policies with window ends [100, 200, 300] and [150, 300], limit 250; after: [100, 150, 200, 250].
    grid = np.unique(np.concatenate([policy.steps[policy.steps <= limit] for policy in policies] + [[limit]]))
    # A policy's value at a grid step is the value of its window that contains the step.
    values = np.stack([training_curve(policy)[np.searchsorted(policy.steps, grid, side='left')]
                       for policy in policies])
    error = (values.std(axis=0, ddof=1) / np.sqrt(len(policies)) if len(policies) > 1
             else np.full(len(grid), np.nan))
    return grid, values.mean(axis=0), error


def smoothed_curve(grid, curve, width):
    """A step curve averaged over the `width` steps that end at each of its steps from `width` on: steps, values."""
    # before: window ends [100, 200, 300, 400] with values [0, 1, 1, 0], width 200
    # after:  steps [200, 300, 400], values [0.5, 1.0, 0.5]; no value before step 200, where the range is not full.
    # The small allowance keeps a step that equals the width when the width was computed as a share of a budget.
    ends = grid[grid >= width * (1 - 1e-12)]
    values = np.asarray([steps_inside(grid, end - width, end) @ curve / width for end in ends])
    return ends, values


def steps_to_final_level_share(policies, run_steps, share=BUDGET_LEVEL_SHARE, smoothing=BUDGET_SMOOTHING_SHARE):
    """The first logged step at which the smoothed seed-averaged curve reaches a share of the final level; None if never."""
    # Both sides are measured from the return of an episode that never reaches the goal, so the answer is the
    # same with reward 1 on arrival (floor 0) and with the step cost (floor -1). A configuration whose final
    # level is that floor has no rise to measure.
    floor = policies[0].never_reached_return
    level = float(np.mean([final_level(policy, run_steps) for policy in policies]))
    if level <= floor:
        return None
    grid, curve, _ = mean_curve(policies, run_steps)
    ends, values = smoothed_curve(grid, curve, smoothing * run_steps)
    reached = np.flatnonzero(values - floor >= share * (level - floor))
    return int(ends[reached[0]]) if len(reached) else None


def earliest_smoothed_step(policies, run_steps, smoothing=BUDGET_SMOOTHING_SHARE):
    """The first logged step with a full smoothing range before it: the budget rule cannot read a rise earlier."""
    # before: windows of 262,144 steps in a run of 50,331,648 (the range is 2,516,582 steps); after: 2,621,440.
    grid, curve, _ = mean_curve(policies, run_steps)
    return int(smoothed_curve(grid, curve, smoothing * run_steps)[0][0])


def standard_error(values):
    """Standard error of the mean over seeds; None with fewer than two."""
    values = np.asarray(values, np.float64)
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None


def configuration_measures(policies, budget):
    """Every measure of one configuration at the budget; all of its policies must have trained that far."""
    whole = [whole_curve_mean(policy, budget) for policy in policies]
    final = [final_level(policy, budget) for policy in policies]
    level, floor = float(np.mean(final)), policies[0].never_reached_return
    # Two peaks of the seed-averaged curve: of the curve smoothed over the last-10% width (the decline check's),
    # and of its single windows (stricter, because one window of a few seeds is raised by noise).
    grid, curve, _ = mean_curve(policies, budget)
    ends, smoothed = smoothed_curve(grid, curve, FINAL_SHARE * budget)
    peak_index, window_index = int(np.argmax(smoothed)), int(np.argmax(curve))
    peak, window_peak = float(smoothed[peak_index]), float(curve[window_index])

    # Episode measures of the last 10%: the goal share is a mean over policies, the steps to the goal a mean
    # over all arrivals of all policies.
    shares = [share for share in (final_goal_share(policy, budget) for policy in policies) if share is not None]
    tail = [episode_sums(policy, budget - FINAL_SHARE * budget, budget) for policy in policies]
    arrivals = sum(sums['arrivals'] for sums in tail)
    coverage = [coverage_at(policy, budget) for policy in policies]
    return {
        'seeds': len(policies), 'budget': budget,
        'whole_curve_mean': float(np.mean(whole)), 'whole_curve_mean_error': standard_error(whole),
        'final_level': level, 'final_level_error': standard_error(final),
        'final_level_at': {fraction: float(np.mean([final_level(policy, fraction * budget) for policy in policies]))
                           for fraction in LENGTH_FRACTIONS},
        # The ratios are measured from the return of an episode that never reaches the goal, so they mean the
        # same with reward 1 on arrival (floor 0) and with the step cost (floor -1).
        'peak': peak, 'peak_step': int(ends[peak_index]),
        'final_over_peak': (level - floor) / (peak - floor) if peak > floor else None,
        'single_window_peak': window_peak, 'single_window_peak_step': int(grid[window_index]),
        'final_over_single_window_peak': (level - floor) / (window_peak - floor) if window_peak > floor else None,
        'capped_first_goal_step': float(np.mean([capped_first_goal_step(policy, budget) for policy in policies])),
        'reached_goal_share': float(np.mean([reached_goal(policy, budget) for policy in policies])),
        'coverage': None if any(value is None for value in coverage) else float(np.mean(coverage)),
        'steps_to_goal': sum(sums['steps_to_goal_sum'] for sums in tail) / arrivals if arrivals > 0 else None,
        'final_goal_share': float(np.mean(shares)) if shares else None}
