"""The written predictions of batch 6 (its ideas.md, section "Predictions"), each as parts the validation seeds decide.

Batch 6 has no screen: it is read from the validation seeds case by case (`screen.validation_cases`, as
validation_seeds.read_cases gives them), on the first block and, in PointMaze Medium with reward 1 on arrival, on
the second block. The form is that of predictions_batch_04.py.
"""

from . import batch_summary, validation_seeds

MEDIUM_ARRIVAL = ('pointmaze-medium', 'arrival_plus_one')
LARGE_ARRIVAL = ('pointmaze-large', 'arrival_plus_one')
LARGE_STEP_COST = ('pointmaze-large', 'step_cost')
# The arm of this batch under its method name on each block, and the two arms it is read beside.
HALF_SPEED = 'predictor-half-speed-held-divisor'
HALF_SPEED_SECOND = 'half-speed-predictor-held'
SLOWER = 'slower-predictor-with-held-divisor'
THREE_PARTS = 'three-parts-held-one-sixteenth'
# Prediction 2 says "within 0.005".
WITHIN = 0.005

READING = ('Where a prediction leaves a choice, it is read as follows. On the first block every arm stands at its '
           'best configuration for the whole-curve mean $W$ among those run there, and the control at its own; a '
           'part waits until every configuration planned there for its arms and for the control is read. On '
           'the second block the arm stands at the configuration that was its best on the first block and is '
           'compared under equal rules, that is with the control at the configuration that was its best on the '
           'first block. "Above RND" is read as a mean difference in $W$ above zero. "Ahead with its interval above '
           'zero" is read as: the 95% interval of the difference is entirely above zero. "Level" is read as: that '
           'interval holds zero. Prediction 1 expects the interval on the second block to reach zero; that part '
           'holds when it does.')


def first(screen, case, *slugs):
    """The first-block comparisons of the named arms in a case, or the words for why they cannot be read yet."""
    return validation_seeds.first_entries(screen.validation_cases, case, *slugs)


def second_equal(screen):
    """The half-speed arm's row on the second block with its comparison under equal rules, or the waiting words."""
    rows, waiting = validation_seeds.second_rows(screen.validation_cases, MEDIUM_ARRIVAL, HALF_SPEED_SECOND)
    return (None, waiting) if rows is None else (rows[HALF_SPEED_SECOND], None)


def whole_words(result, digits=4):
    """A difference in W with its interval."""
    return batch_summary.fine_difference(result['whole_curve_mean'], digits)


def above_on_the_second_block(screen):
    """Whether the half-speed arm's mean difference in W from the control is above zero on the second block."""
    row, waiting = second_equal(screen)
    if row is None:
        return None, waiting
    return row['equal']['whole_curve_mean']['mean'] > 0, (
        f"$W$ {row['measures']['whole_curve_mean']:.4f} on the second block, difference under equal rules {whole_words(row['equal'])}.")


def by_less_than_on_the_first_block(screen):
    """Whether the difference on the second block is smaller than the one on the first block."""
    row, waiting = second_equal(screen)
    found, earlier = first(screen, MEDIUM_ARRIVAL, HALF_SPEED)
    if row is None or found is None:
        return None, waiting or earlier
    before, after = found[HALF_SPEED]['result']['whole_curve_mean']['mean'], row['equal']['whole_curve_mean']['mean']
    return after < before, f'First block {before:+.4f}; second block {after:+.4f}.'


def interval_reaches_zero(screen):
    """Whether the interval of the difference on the second block is not entirely above zero, as the prediction expected."""
    row, waiting = second_equal(screen)
    if row is None:
        return None, waiting
    return row['equal']['whole_curve_mean']['side'] != 'ahead', f"Difference under equal rules {whole_words(row['equal'])}."


def within_of_the_slower_predictor(screen):
    """Whether the half-speed arm's best W on PointMaze Large is within 0.005 of the slower predictor with held divisor."""
    found, waiting = first(screen, LARGE_ARRIVAL, HALF_SPEED, SLOWER)
    if found is None:
        return None, waiting
    mine, theirs = (found[slug]['measures']['whole_curve_mean'] for slug in (HALF_SPEED, SLOWER))
    return abs(mine - theirs) <= WITHIN, (
        f'Best $W$ on the first block: the arm {mine:.4f}, the slower predictor with held divisor {theirs:.4f} '
        f'(apart by {abs(mine - theirs):.4f}).')


def side_is(case, slug, wanted):
    """A part: an arm's difference in W from the control on the first block of a case stands on the wanted side."""
    def check(screen):
        """Read at each side's best configuration for W."""
        found, waiting = first(screen, case, slug)
        if found is None:
            return None, waiting
        entry = found[slug]
        return entry['result']['whole_curve_mean']['side'] == wanted, (
            f"$W$ {entry['measures']['whole_curve_mean']:.4f}, difference from the control {whole_words(entry['result'])}.")
    return check


PREDICTIONS = [
    {'title': 'second block, PointMaze Medium with reward 1 on arrival',
     'parts': [('The predictor at half speed with held divisor is again above the control', [], above_on_the_second_block),
               ('By less than on the first block', [], by_less_than_on_the_first_block),
               ('Its interval reaches zero, as the prediction expected', [], interval_reaches_zero)]},
    {'title': 'PointMaze Large, reward 1 on arrival',
     'parts': [('The predictor at half speed with held divisor is within 0.005 of the slower predictor with held divisor', [],
                within_of_the_slower_predictor),
               ('It is ahead of the control with its interval above zero', [], side_is(LARGE_ARRIVAL, HALF_SPEED, 'ahead'))]},
    {'title': 'PointMaze Large, step cost',
     'parts': [('All three parts stays ahead of the control with its interval above zero', [],
                side_is(LARGE_STEP_COST, THREE_PARTS, 'ahead')),
               ('The slower predictor with held divisor is level with the control', [], side_is(LARGE_STEP_COST, SLOWER, 'level'))]},
]
