"""The written predictions of batch 7 (its ideas.md, section "Predictions"), each as parts the validation seeds decide.

Batch 7 has no screen: the held divisor alone is read on the first validation block of the two step-cost cases
(`screen.validation_cases`, as validation_seeds.read_cases gives them). The form is that of predictions_batch_04.py.
"""

from . import batch_summary, validation_seeds

MEDIUM_STEP_COST = ('pointmaze-medium', 'step_cost')
LARGE_STEP_COST = ('pointmaze-large', 'step_cost')
HELD = 'divisor-held-after-one-sixteenth'
# Prediction 2 says "within 0.005", and names one larger weight at the learning rate of the batch's first units.
WITHIN = 0.005
LARGER_WEIGHT = {'learning_rate': 0.006, 'int_coef': 0.03125}

READING = ('Where a prediction leaves a choice, it is read as follows. The held divisor alone stands at its best '
           'configuration for the whole-curve mean $W$ among those run on the first validation block, and the '
           'control at its own; a part waits until every configuration planned there for both is read. "Ahead with its interval above zero" is read as: the 95% interval of the difference '
           'in $W$ is entirely above zero. "Level with RND (within 0.005)" is read as: the mean difference in $W$ is '
           'at most 0.005 in size. "Behind RND\'s best" at the weight 0.03125 is read as: the configuration at '
           'learning rate 0.006 and that weight has a mean difference in $W$ below zero from the control at its '
           'best configuration.')


def entry_of(screen, case):
    """The held divisor's first-block comparison in a case, or the words for why it cannot be read yet."""
    found, waiting = validation_seeds.first_entries(screen.validation_cases, case, HELD)
    return (None, waiting) if found is None else (found[HELD], None)


def whole_words(result, digits=4):
    """A difference in W with its interval."""
    return batch_summary.fine_difference(result['whole_curve_mean'], digits)


def ahead_in_the_large_case(screen):
    """Whether the held divisor alone is ahead of the control in PointMaze Large with the step cost."""
    entry, waiting = entry_of(screen, LARGE_STEP_COST)
    if entry is None:
        return None, waiting
    return entry['result']['whole_curve_mean']['side'] == 'ahead', (
        f"$W$ {entry['measures']['whole_curve_mean']:.4f}, difference from the control {whole_words(entry['result'])}.")


def level_in_the_medium_case(screen):
    """Whether the held divisor alone is within 0.005 of the control in PointMaze Medium with the step cost."""
    entry, waiting = entry_of(screen, MEDIUM_STEP_COST)
    if entry is None:
        return None, waiting
    result = entry['result']['whole_curve_mean']
    return abs(result['mean']) <= WITHIN, (
        f"$W$ {entry['measures']['whole_curve_mean']:.4f}, difference from the control {whole_words(entry['result'])}; "
        f"the interval stands {'entirely above zero' if result['side'] == 'ahead' else 'entirely below zero' if result['side'] == 'behind' else 'across zero'}.")


def behind_at_the_larger_weight(screen):
    """Whether the held divisor at the weight 0.03125 is behind the control's best in PointMaze Medium with the step cost."""
    entry, waiting = entry_of(screen, MEDIUM_STEP_COST)
    if entry is None:
        return None, waiting
    first = screen.validation_cases[MEDIUM_STEP_COST]['first']
    # The one configuration of the arm at the named learning rate and weight.
    groups = [group for group in first['sides'][HELD].groups
              if all(group.setting(name) == value for name, value in LARGER_WEIGHT.items())]
    if len(groups) != 1:
        return None, 'Waiting for the configuration at learning rate 0.006 and intrinsic weight 0.03125.'
    result = validation_seeds.compared(groups[0], entry['baseline'], first['budget'], 'batch 7 prediction 2', *MEDIUM_STEP_COST,
                                       HELD, 'at the larger weight 0.03125')
    return result['whole_curve_mean']['mean'] < 0, f"Difference from the control at its best configuration {whole_words(result)}."


def best_weight(screen):
    """The held divisor's best intrinsic weight in PointMaze Medium with the step cost, beside the control's."""
    entry, waiting = entry_of(screen, MEDIUM_STEP_COST)
    if entry is None:
        return waiting
    return (f"The held divisor's best intrinsic weight on the first block is {entry['best'].setting('int_coef'):g}; "
            f"the control's is {entry['baseline'].setting('int_coef'):g}.")


PREDICTIONS = [
    {'title': 'PointMaze Large, step cost',
     'parts': [('The held divisor alone is ahead of the control with its interval above zero', [], ahead_in_the_large_case)]},
    {'title': 'PointMaze Medium, step cost',
     'parts': [('The held divisor alone is level with the control (within 0.005)', [], level_in_the_medium_case),
               ('At the larger weight 0.03125 it is behind the control\'s best', [], behind_at_the_larger_weight)],
     'beside': [([], best_weight)]},
]
