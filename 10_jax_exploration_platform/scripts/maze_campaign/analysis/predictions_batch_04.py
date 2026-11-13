"""The written predictions of batch 4 (its ideas.md, section "Predictions"), each as parts the sweeps' numbers decide.

Batch 4 has no screen: its predictions are read from the development sweeps case by case (`screen.cases`, as
case_sweeps.read_development gives them). The form is that of predictions_batch_01.py, with no method named in a
part's list of needed units: a part whose case is not complete says so itself and cannot be read yet.
"""

import numpy as np

from . import batch_summary, measures, report

MEDIUM_ARRIVAL = ('pointmaze-medium', 'arrival_plus_one')
MEDIUM_STEP_COST = ('pointmaze-medium', 'step_cost')
LARGE_STEP_COST = ('pointmaze-large', 'step_cost')
FINALISTS = ('fast-last-layer-step-0-point-1-slower-predictor-held-divisor', 'slower-predictor-with-held-divisor')
# Prediction 3 names two groups of seeds by their whole-curve mean, about -0.72 and about -0.90: the worse group
# is read as the seeds below the value halfway between the two.
WORSE_GROUP_BELOW = -0.81

READING = ('Where a prediction leaves a choice, it is read as follows. Each side is read at its own best configuration '
           'on the development seeds: for the whole-curve mean $W$ the one with the highest $W$, for "reach the goal '
           'sooner" the one with the smallest mean capped first-goal time $M$. "Sooner" is read as a negative mean '
           'difference in $M$ from the baseline; the interval is given beside it. "Level with or slightly ahead" is '
           'read as: the 95% interval of the difference in $W$ is not entirely below zero. "Not ahead" is read as: '
           'that interval is not entirely above zero. "The worse group" of prediction 3 is read as the seeds whose '
           f'whole-curve mean is below {WORSE_GROUP_BELOW:g}, halfway between the two group values the prediction '
           'names.')


def compared(screen, case):
    """The finalists' comparisons with the baseline in one case, or the words for why they cannot be read yet."""
    # before: a case in which one finalist's unit is not complete; after: (None, 'Waiting for: <its name> ...').
    found = screen.cases[case]
    waiting = [found['sides'][slug].name for slug in FINALISTS
               if found['comparisons'].get(slug) is None or found['comparisons'][slug]['whole']['result'] is None]
    if waiting:
        return None, (f"Waiting for complete units in {found['words']}: {'; '.join(waiting)}"
                      + ('' if found['baseline'].groups else ' (the baseline has no complete configuration either)') + '.')
    return {found['sides'][slug].name: found['comparisons'][slug] for slug in FINALISTS}, None


def whole_words(entry):
    """A finalist's best W against the baseline's, with the paired difference."""
    return (f"$W$ {entry['measures']['whole_curve_mean']:.3f}, difference from the baseline "
            f"{batch_summary.fine_difference(entry['result']['whole_curve_mean'], 3)}")


def sooner(case):
    """A part: both finalists first reach the goal sooner than the baseline in a case (mean difference in M below zero)."""
    def check(screen):
        """Read at each side's best configuration for M."""
        found, waiting = compared(screen, case)
        if found is None:
            return None, waiting
        results = {name: comparison['discovery']['result']['capped_first_goal_step'] for name, comparison in found.items()}
        return all(result['mean'] < 0 for result in results.values()), (
            'Difference in $M$ from the baseline, in thousands of steps: '
            + '; '.join(f'{name} {report.discovery_interval_text(result)}' for name, result in results.items()) + '.')
    return check


def whole_side(case, refused, claim_words):
    """A part: no finalist's difference in W from the baseline lies on the refused side in a case."""
    def check(screen):
        """Read at each side's best configuration for W."""
        found, waiting = compared(screen, case)
        if found is None:
            return None, waiting
        sides = {name: comparison['whole']['result']['whole_curve_mean']['side'] for name, comparison in found.items()}
        return all(side != refused for side in sides.values()), (
            f'{claim_words}: ' + '; '.join(f"{name} {whole_words(comparison['whole'])}" for name, comparison in found.items()) + '.')
    return check


def smallest_weight(screen):
    """Whether each finalist's best configuration for W in the step-cost case of PointMaze Medium has the smallest weight of its grid."""
    found, waiting = compared(screen, MEDIUM_STEP_COST)
    if found is None:
        return waiting
    rows = []
    for name, comparison in found.items():
        entry = comparison['whole']
        at_end = ('int_coef', 'smallest') in entry['ends']
        rows.append(f"{name}: best intrinsic weight {entry['best'].setting('int_coef'):g}, "
                    f"{'the smallest' if at_end else 'not the smallest'} of its grid")
    return ('; '.join(rows) + '. The prediction says that a best weight at the smallest of the grid would mean the case '
            'needs a change outside the bonus.')


def worse_share(policies, budget):
    """The share of policies whose whole-curve mean is below the split between the two groups."""
    # before: whole-curve means [-0.70, -0.92, -0.74, -0.88] and the split -0.81; after: 0.5.
    return float(np.mean([measures.whole_curve_mean(policy, budget) < WORSE_GROUP_BELOW for policy in policies]))


def fewer_in_the_worse_group(screen):
    """Whether each finalist has a smaller share of seeds in the worse group than the baseline, in PointMaze Large with the step cost."""
    found, waiting = compared(screen, LARGE_STEP_COST)
    if found is None:
        return None, waiting
    budget = screen.cases[LARGE_STEP_COST]['budget']
    theirs = worse_share(next(iter(found.values()))['whole']['baseline'].policies, budget)
    mine = {name: worse_share(comparison['whole']['best'].policies, budget) for name, comparison in found.items()}
    return all(share < theirs for share in mine.values()), (
        f'Share of seeds with $W$ below {WORSE_GROUP_BELOW:g}, each side at its best configuration for $W$: '
        + '; '.join(f'{name} {100 * share:.0f}%' for name, share in mine.items()) + f'; baseline {100 * theirs:.0f}%.')


def large_step_cost_beside(screen):
    """The finalists' whole-curve means against the baseline's in PointMaze Large with the step cost."""
    found, waiting = compared(screen, LARGE_STEP_COST)
    return waiting if found is None else '; '.join(f"{name} {whole_words(comparison['whole'])}" for name, comparison in found.items()) + '.'


PREDICTIONS = [
    {'title': 'PointMaze Medium, reward 1 on arrival',
     'parts': [('Both finalists reach the goal sooner than the baseline', [], sooner(MEDIUM_ARRIVAL)),
               ('Both are level with or slightly ahead of the baseline on the whole-curve mean', [],
                whole_side(MEDIUM_ARRIVAL, 'behind', 'At the best configurations for $W$'))]},
    {'title': 'PointMaze Medium, step cost',
     'parts': [('No finalist is ahead of the baseline on the whole-curve mean', [],
                whole_side(MEDIUM_STEP_COST, 'ahead', 'At the best configurations for $W$'))],
     'beside': [([], smallest_weight)]},
    {'title': 'PointMaze Large, step cost',
     'parts': [('Each finalist has a smaller share of seeds in the worse group than the baseline', [], fewer_in_the_worse_group)],
     'beside': [([], large_step_cost_beside)]},
]
