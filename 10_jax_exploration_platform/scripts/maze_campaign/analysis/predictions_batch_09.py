"""The written predictions of batch 9 (its ideas.md: section "Predictions" and the addendum of 10:53 PT).

Batch 9 has no screen: the fast last layer and RND are read for the mean capped first-goal time $M$ on the
validation seeds case by case (`screen.validation_cases`): the first block is the phase `discovery-validation`, the
second the phase `discovery-block-two`.
"""

from .prediction_readings import (LARGE_ARRIVAL, LARGE_STEP_COST, MEDIUM_ARRIVAL, MEDIUM_STEP_COST, best_entry,
                                  m_text, second_row, thousands)

FAST = 'fast-last-layer-step-0-point-1'
FAST_SECOND = 'fast-layer-step-0-point-1'
# "At least 40 thousand steps", "a few thousand steps at most" (read as at most 6 thousand, the addendum's upper
# end), and the addendum's "2 to 6 thousand".
LARGE_GAP, FEW, MEDIUM_RANGE = 40_000, 6_000, (2_000, 6_000)

READING = ('Where a prediction leaves a choice, it is read as follows. Every reading is of $M$. On the first block '
           'each side stands at its best configuration for $M$ among those run there; on the second block each '
           'side stands at the configuration that was its best for $M$ on the first block (equal rules). "Ahead with '
           'the interval below zero" is read as: the 95% interval of the paired difference (fast last layer minus '
           'RND) is entirely below zero. "By at least 40 thousand steps" and "by more than 40 thousand" are read on '
           'the mean difference. "A few thousand steps at most" is read as: the mean difference is below zero and '
           'at most 6 thousand steps in size. "Level" is read as: the interval holds zero.')


def first_block(case, verdict):
    """A part on the first block: the fast last layer's difference in M from RND, each at its best, judged by `verdict`."""
    def check(screen):
        """Read best against best for M."""
        entry, waiting = best_entry(screen, case, FAST, 'discovery')
        if entry is None:
            return None, waiting
        return verdict(entry['result']['capped_first_goal_step']), (
            f"$M$ {thousands(entry['measures']['capped_first_goal_step'])}; difference from RND's best "
            f"{m_text(entry['result'])}.")
    return check


def second_block(case, verdict):
    """A part on the second block: the difference in M from RND under equal rules, judged by `verdict`."""
    def check(screen):
        """Read each side at its first-block best for M."""
        row, waiting = second_row(screen, case, FAST_SECOND, 'discovery')
        if row is None:
            return None, waiting
        return verdict(row['equal']['capped_first_goal_step']), (
            f"$M$ {thousands(row['measures']['capped_first_goal_step'])}; difference from RND under equal rules "
            f"{m_text(row['equal'])}.")
    return check


large = lambda found: found['side'] == 'ahead' and found['mean'] <= -LARGE_GAP
few = lambda found: -FEW <= found['mean'] < 0
below = lambda found: found['side'] == 'ahead'
level = lambda found: found['side'] == 'level'
medium = lambda found: found['side'] == 'ahead' and -MEDIUM_RANGE[1] <= found['mean'] <= -MEDIUM_RANGE[0]


def below_in_one_medium_case(screen):
    """Whether the interval of the difference is below zero in at least one of the two PointMaze Medium cases."""
    found = [first_block(case, below)(screen) for case in (MEDIUM_ARRIVAL, MEDIUM_STEP_COST)]
    if any(held is None for held, _ in found):
        return None, ' '.join(words for _, words in found)
    return any(held for held, _ in found), ' '.join(f'{name}: {words}' for name, (_, words) in
                                                    zip(('Reward 1 on arrival', 'Step cost'), found))


PREDICTIONS = [
    {'title': 'PointMaze Large, step cost (first block)',
     'parts': [('Ahead of RND by at least 40 thousand steps, interval below zero', [], first_block(LARGE_STEP_COST, large))]},
    {'title': 'the two PointMaze Medium cases (first block)',
     'parts': [('Reward 1 on arrival: ahead by a few thousand steps at most', [], first_block(MEDIUM_ARRIVAL, few)),
               ('Step cost: ahead by a few thousand steps at most', [], first_block(MEDIUM_STEP_COST, few)),
               ('The interval is below zero in at least one of the two', [], below_in_one_medium_case)]},
    {'title': 'the second block for discovery (addendum, written 2026-11-13 10:53 PT)',
     'parts': [('PointMaze Large with reward 1 on arrival: ahead by more than 40 thousand steps, interval below zero', [],
                second_block(LARGE_ARRIVAL, large)),
               ('PointMaze Large with the step cost: ahead by more than 40 thousand steps, interval below zero', [],
                second_block(LARGE_STEP_COST, large)),
               ('PointMaze Medium with reward 1 on arrival: ahead by 2 to 6 thousand steps, interval below zero', [],
                second_block(MEDIUM_ARRIVAL, medium)),
               ('PointMaze Medium with the step cost: level', [], second_block(MEDIUM_STEP_COST, level))]},
]
