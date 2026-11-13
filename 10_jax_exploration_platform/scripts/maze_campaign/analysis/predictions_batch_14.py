"""The written predictions of batch 14 (its ideas.md, section "Predictions").

The ant U-maze screen of the fast last layer with the first-arrival hold at steps 0.03 and 0.01 is read against the
RND unit of the AntMaze baseline folder's check at the same configuration and seeds (`screen.same_configuration`);
the second block on PointMaze Medium from the validation seeds (`screen.validation_cases`).
"""

from . import validation_seeds
from .prediction_readings import (ANT, MEDIUM_ARRIVAL, MEDIUM_STEP_COST, group_case, means, pairs_of, respawns,
                                  second_row, signs_words, w_text)

STEP_01 = 'fast-step-0-point-01-with-hold'
HOLD_SECOND = 'held-from-first-arrival'
# Prediction 1: "within 0.003 ... in at least 6 of 8" and "behind in most" (at least 5 of 8); prediction 3: "within
# 0.005"; prediction 4: "a few per unit", read as at most 10 in a unit of 128 policies x 16,777,216 steps.
CLOSE, MOST, MAJORITY, WITHIN, FEW = 0.003, 6, 5, 0.005, 10

READING = ('Where a prediction leaves a choice, it is read as follows. On the ant (16 development seeds, 16,777,216 '
           'steps per policy) every comparison is with RND of the check at the same configuration and seeds, and the '
           'configurations are counted by the sign of the mean paired difference in $W$: "ahead" is above RND in at '
           'least 6 of the 8, "behind in most" below RND in at least 5 of the 8, "level" neither above nor below in 6 '
           'or more, and "within 0.003" is read on the mean difference. "By less than step 0.1" is read as: the mean '
           'over the 8 configurations of the paired difference in $W$ between step 0.03 and step 0.1 is below zero. '
           'On PointMaze Medium the second block is read with each side at the configuration fixed on the first '
           'block; "within 0.005 of the hold alone" is read on the mean paired difference, and "ahead of RND" as: the '
           '95% interval of the difference under equal rules is entirely above zero. "A few per unit" is read as at '
           'most 10.')


def title(step, count):
    """The same-configuration group of one step at one count."""
    return f'AntMaze U-maze, {count} environments per policy: fast last layer at step {step} with the first-arrival hold against RND'


def rnd(count):
    """RND of the check at one count."""
    return f'cleanrl-ppo-rnd-with-{count}-environments-per-policy'


def differences(screen, step, count, other=None):
    """The mean paired differences in W of one step from RND (or from `other`) over its configurations, or None."""
    found = pairs_of(screen, title(step, count), ANT, other or rnd(count))
    arm = group_case(screen, title(step, count), ANT)['arm']
    return None if found is None or len(arm.groups) < arm.planned else means(found, 'whole_curve_mean')


def ant_part(step, count, verdict, other=None):
    """A part: the signs or sizes of one step's differences in W, judged by `verdict`."""
    def check(screen):
        """Read the step's pairs."""
        values = differences(screen, step, count, other)
        if values is None:
            return None, f'Waiting for complete units of step {step} and of RND at {count} environments per policy.'
        against = 'step 0.1' if other else 'RND'
        return verdict(values), f'$W$ difference from {against}: {signs_words(values)}.'
    return check


def no_step_ahead(screen):
    """Whether neither step is above RND in 6 or more of 8 configurations with 128 environments per policy."""
    found = {step: differences(screen, step, 128) for step in ('0.03', '0.01')}
    if any(values is None for values in found.values()):
        return None, 'Waiting for complete units of both steps and of RND at 128 environments per policy.'
    return all(sum(value > 0 for value in values) < MOST for values in found.values()), ' '.join(
        f'Step {step}: {signs_words(values)}.' for step, values in found.items())


def second_block_hold(case):
    """A part: on the second block, step 0.01 with the hold is within 0.005 of the hold alone on W."""
    def check(screen):
        """Pair the two rows fixed on the first block."""
        mine, waiting = second_row(screen, case, STEP_01)
        theirs, other = second_row(screen, case, HOLD_SECOND)
        if mine is None or theirs is None:
            return None, waiting or other
        second = screen.validation_cases[case]['second']
        result = validation_seeds.compared(mine['group'], theirs['group'], second['budget'], 'batch 14 prediction 3', *case)
        return abs(result['whole_curve_mean']['mean']) <= WITHIN, (
            f"$W$ {mine['measures']['whole_curve_mean']:.4f} against the hold alone's {theirs['measures']['whole_curve_mean']:.4f}; "
            f'paired difference {w_text(result)}.')
    return check


def second_block_ahead(screen):
    """Whether step 0.01 with the hold is ahead of RND under equal rules on the second block, reward 1 on arrival."""
    row, waiting = second_row(screen, MEDIUM_ARRIVAL, STEP_01)
    if row is None:
        return None, waiting
    return row['equal']['whole_curve_mean']['side'] == 'ahead', (
        f"$W$ {row['measures']['whole_curve_mean']:.4f}; difference from RND under equal rules {w_text(row['equal'])}.")


def few_respawns(screen):
    """Whether every unit of the screen has at most 10 fault-guard respawns."""
    found = {}
    for step in ('0.03', '0.01'):
        for count in (128, 16):
            arm = group_case(screen, title(step, count), ANT)['arm']
            if len(arm.groups) < arm.planned:
                return None, f'Waiting for the unit of step {step} at {count} environments per policy.'
            found[(step, count)] = respawns(arm)[0]
    return all(value <= FEW for value in found.values()), 'Respawns: ' + '; '.join(
        f'step {step} with {count} environments per policy {value}' for (step, count), value in found.items()) + '.'


above = lambda values, least=MOST: sum(value > 0 for value in values) >= least
below = lambda values, least=MAJORITY: sum(value < 0 for value in values) >= least
level = lambda values: sum(value > 0 for value in values) < MOST and sum(value < 0 for value in values) < MOST

PREDICTIONS = [
    {'title': 'ant, 128 environments per policy',
     'parts': [('Step 0.01: within 0.003 of RND in at least 6 of 8 configurations', [],
                ant_part('0.01', 128, lambda values: sum(abs(value) <= CLOSE for value in values) >= MOST)),
               ('Step 0.03: behind RND in most configurations', [], ant_part('0.03', 128, below)),
               ('No step is ahead of RND at this count', [], no_step_ahead)]},
    {'title': 'ant, 16 environments per policy',
     'parts': [('Step 0.03: ahead of RND in at least 6 of 8 configurations', [], ant_part('0.03', 16, above)),
               ('Step 0.03: by less than step 0.1', [],
                ant_part('0.03', 16, lambda values: sum(values) / len(values) < 0,
                         'fast-last-layer-and-first-arrival-hold-with-16-environments-per-policy')),
               ('Step 0.01: level with RND', [], ant_part('0.01', 16, level))]},
    {'title': 'PointMaze Medium, second block',
     'parts': [('Reward 1 on arrival: step 0.01 with the hold within 0.005 of the hold alone', [], second_block_hold(MEDIUM_ARRIVAL)),
               ('Step cost: step 0.01 with the hold within 0.005 of the hold alone', [], second_block_hold(MEDIUM_STEP_COST)),
               ('Reward 1 on arrival: ahead of RND', [], second_block_ahead)]},
    {'title': 'fault-guard respawns',
     'parts': [('At most a few per unit', [], few_respawns)]},
]
