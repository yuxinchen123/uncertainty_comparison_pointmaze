"""The written predictions of batch 8 (its ideas.md, section "Predictions"), read on the first validation block.

Batch 8 has no screen: the divisor held from each policy's first arrival and the error length with a held divisor
are read from the validation seeds case by case (`screen.validation_cases`), each side at its best configuration
on the first block, against RND and against the divisor held after one sixteenth of the run.
"""

from .prediction_readings import (LARGE_ARRIVAL, MEDIUM_ARRIVAL, best_against_arm, best_entry, m_text, thousands,
                                  w_text)

ARRIVAL_HOLD = 'divisor-held-from-first-arrival'
FRACTION_HOLD = 'divisor-held-after-one-sixteenth'
ERROR_LENGTH = 'error-length-with-held-divisor'
WITHIN = 0.005

READING = ('Where a prediction leaves a choice, it is read as follows. Both are read on the first validation block '
           '(seeds 10000 to 10255), as the batch\'s ideas file says, with every side at its own best configuration '
           'there: for the whole-curve mean $W$ the highest $W$, for the mean capped first-goal time $M$ the smallest '
           '$M$. "Level with the held divisor alone (within 0.005)" is read as: the mean paired difference in $W$ is '
           'at most 0.005 in size. "Ahead" and "sooner" are read as: the 95% interval of the paired difference is '
           'entirely above zero ($W$) or entirely below zero ($M$). "Level" in prediction 2 is read as: the interval '
           'of the difference in $W$ holds zero.')


def against_the_fraction_hold(case, slug, metric, verdict):
    """A part: an arm at its best against the held divisor alone at its best on the first block, judged by `verdict`."""
    def check(screen):
        """Read best against best on the first block."""
        entry, waiting = best_against_arm(screen, case, slug, FRACTION_HOLD, metric)
        if entry is None:
            return None, waiting
        mine, theirs, result = entry['measures'], entry['baseline_measures'], entry['result']
        if metric == 'whole':
            return verdict(result['whole_curve_mean']), (
                f"$W$ {mine['whole_curve_mean']:.4f} against {theirs['whole_curve_mean']:.4f}; difference {w_text(result)}.")
        return verdict(result['capped_first_goal_step']), (
            f"$M$ {thousands(mine['capped_first_goal_step'])} against {thousands(theirs['capped_first_goal_step'])}; "
            f'difference {m_text(result)}.')
    return check


def ahead_of_rnd(case, slug):
    """A part: an arm at its best W is ahead of RND at its best, the interval above zero."""
    def check(screen):
        """Read best against best on the first block."""
        entry, waiting = best_entry(screen, case, slug)
        if entry is None:
            return None, waiting
        return entry['result']['whole_curve_mean']['side'] == 'ahead', (
            f"$W$ {entry['measures']['whole_curve_mean']:.4f}; difference from RND's best {w_text(entry['result'])}.")
    return check


within = lambda found: abs(found['mean']) <= WITHIN
ahead = lambda found: found['side'] == 'ahead'
level = lambda found: found['side'] == 'level'

PREDICTIONS = [
    {'title': "the divisor held from each policy's first arrival",
     'parts': [('PointMaze Large with reward 1 on arrival: level with the held divisor alone (within 0.005)', [],
                against_the_fraction_hold(LARGE_ARRIVAL, ARRIVAL_HOLD, 'whole', within)),
               ('PointMaze Large with reward 1 on arrival: ahead of RND with its interval above zero', [],
                ahead_of_rnd(LARGE_ARRIVAL, ARRIVAL_HOLD)),
               ('PointMaze Medium with reward 1 on arrival: ahead of the held divisor after one sixteenth', [],
                against_the_fraction_hold(MEDIUM_ARRIVAL, ARRIVAL_HOLD, 'whole', ahead))]},
    {'title': 'the error length with a held divisor',
     'parts': [('PointMaze Large with reward 1 on arrival: reaches the goal sooner than the held divisor alone', [],
                against_the_fraction_hold(LARGE_ARRIVAL, ERROR_LENGTH, 'discovery', ahead)),
               ('PointMaze Medium with reward 1 on arrival: reaches the goal sooner than the held divisor alone', [],
                against_the_fraction_hold(MEDIUM_ARRIVAL, ERROR_LENGTH, 'discovery', ahead)),
               ('PointMaze Large with reward 1 on arrival: level with the held divisor alone on $W$', [],
                against_the_fraction_hold(LARGE_ARRIVAL, ERROR_LENGTH, 'whole', level)),
               ('PointMaze Medium with reward 1 on arrival: level with the held divisor alone on $W$', [],
                against_the_fraction_hold(MEDIUM_ARRIVAL, ERROR_LENGTH, 'whole', level))]},
]
