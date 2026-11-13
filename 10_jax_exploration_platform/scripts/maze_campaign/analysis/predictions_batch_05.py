"""The written predictions of batch 5 (its ideas.md, section "Predictions"), each as parts the validation seeds decide.

Batch 5 has no screen. Its arms are read on the first validation block of PointMaze Medium with reward 1 on arrival
(`screen.validation_cases`, as validation_seeds.read_cases gives them): each arm at its best configuration for the
whole-curve mean among those run there, and the control at its own. The form is that of predictions_batch_04.py.
"""

import numpy as np

from . import batch_summary, compare, measures, validation_seeds

CASE = ('pointmaze-medium', 'arrival_plus_one')
CONTROL = 'control-cleanrl-ppo-rnd'
HELD_SIXTEENTH = 'divisor-held-after-one-sixteenth'
HELD_SIXTY_FOURTH = 'divisor-held-one-sixty-fourth'
SLOWER_SIXTEENTH = 'slower-predictor-with-held-divisor'    # the finalist of batch 4
SLOWER_SIXTY_FOURTH = 'slow-predictor-held-sixty-fourth'
HALF_SPEED = 'predictor-half-speed-held-divisor'
# Prediction 1 says "within 0.01"; prediction 2 names the stretch of the run in environment steps.
LEVEL_WITHIN = 0.01
STRETCH = (66_000, 200_000)

READING = ('Where a prediction leaves a choice, it is read as follows. Every arm is read on the first validation '
           'block at its best configuration for the whole-curve mean $W$ among those run there, and the control '
           'at its own; a part waits until every configuration planned there for its arms and for the control is '
           'read. "Level with RND (within 0.01)" is read as: the mean difference in $W$ from the control is at '
           'most 0.01 in size. "Ahead" is read as: the 95% interval of the difference is entirely above zero. '
           '"Raises the return between 66 and 200 thousand steps" is read as: the mean training return over those '
           'steps, paired by seed, is higher with the divisor held after 1/64 of the run than with it held after '
           '1/16, at the same predictor speed (mean difference above zero; the interval is given beside it, and a '
           'part that held with an interval across zero is a difference the seeds do not separate from none). '
           '"Between RND and the finalist at every point of the curve" is read as: at every logging window the '
           'seed-averaged return of the arm lies between that of the control and that of the slower predictor '
           'with held divisor.')


def entries(screen, *slugs):
    """The first-block comparisons of the named arms, or the words for why they cannot be read yet."""
    return validation_seeds.first_entries(screen.validation_cases, CASE, *slugs)


def difference_words(entry, quantity='whole_curve_mean'):
    """A difference from the control with its interval, at four decimals."""
    return batch_summary.fine_difference(entry['result'][quantity], 4)


def level_with_the_control(screen):
    """Whether the held divisor alone is within 0.01 of the control on the whole-curve mean."""
    found, waiting = entries(screen, HELD_SIXTEENTH)
    if found is None:
        return None, waiting
    entry = found[HELD_SIXTEENTH]
    return abs(entry['result']['whole_curve_mean']['mean']) <= LEVEL_WITHIN, (
        f"$W$ {entry['measures']['whole_curve_mean']:.4f}, difference from the control {difference_words(entry)}.")


def ahead_on_the_final_level(screen):
    """Whether the held divisor alone is ahead of the control on the final level."""
    found, waiting = entries(screen, HELD_SIXTEENTH)
    if found is None:
        return None, waiting
    entry = found[HELD_SIXTEENTH]
    return entry['result']['final_level']['side'] == 'ahead', (
        f"$L$ {entry['measures']['final_level']:.4f}, difference from the control {difference_words(entry, 'final_level')}.")


def stretch_mean(policy, budget):
    """A policy's mean training return over the stretch of the run that prediction 2 names."""
    return measures.curve_mean(policy, *STRETCH)


def raises_the_stretch(later, earlier):
    """A part: the arm with the earlier hold has the higher return over the stretch than the arm with the later hold."""
    def check(screen):
        """Read at each arm's best configuration for W, paired by seed."""
        found, waiting = entries(screen, later, earlier)
        if found is None:
            return None, waiting
        budget = screen.validation_cases[CASE]['first']['budget']
        seeds, differences = compare.paired_differences(found[later]['best'].policies, found[earlier]['best'].policies,
                                                        stretch_mean, budget)
        result = compare.difference_interval(differences, 'batch 5 prediction 2', later, earlier)
        if result['interval'] is None:
            return None, 'Fewer than two seeds in common.'
        low, high = result['interval']
        stands = 'entirely above zero' if low > 0 else 'entirely below zero' if high < 0 else 'across zero'
        return result['mean'] > 0, (
            f'Mean return over steps {STRETCH[0]:,} to {STRETCH[1]:,}, held after 1/64 minus held after 1/16, on '
            f"{len(seeds)} seeds: {batch_summary.fine_difference(result, 4)}; the interval stands {stands}.")
    return check


def ahead_of_the_control(slug):
    """A part: an arm's whole-curve mean is ahead of the control's (interval entirely above zero)."""
    def check(screen):
        """Read at each side's best configuration for W."""
        found, waiting = entries(screen, slug)
        if found is None:
            return None, waiting
        entry = found[slug]
        return entry['result']['whole_curve_mean']['side'] == 'ahead', (
            f"$W$ {entry['measures']['whole_curve_mean']:.4f}, difference from the control {difference_words(entry)}.")
    return check


def between_at_every_window(screen):
    """Whether the half-speed arm's curve lies between the control's and the finalist's at every logging window."""
    found, waiting = entries(screen, HALF_SPEED, SLOWER_SIXTEENTH)
    if found is None:
        return None, waiting
    budget = screen.validation_cases[CASE]['first']['budget']
    # The three seed-averaged curves on their common logging windows.
    # before: three configurations of 256 policies each; after: three arrays of one mean return per window.
    curves = [measures.mean_curve(group.policies, budget) for group in
              (found[HALF_SPEED]['best'], found[HALF_SPEED]['baseline'], found[SLOWER_SIXTEENTH]['best'])]
    if any(len(grid) != len(curves[0][0]) or np.any(grid != curves[0][0]) for grid, _, _ in curves):
        raise ValueError('The three curves of batch 5, prediction 3, are not on the same logging windows.')
    half, control, finalist = (curve for _, curve, _ in curves)
    low, high = np.minimum(control, finalist), np.maximum(control, finalist)
    inside, above, below = (half >= low) & (half <= high), half > high, half < low
    return bool(inside.all()), (
        f'Of {len(half)} logging windows the arm is between the two in {int(inside.sum())}, above both in '
        f"{int(above.sum())} and below both in {int(below.sum())}. Whole-curve means: the arm "
        f"{found[HALF_SPEED]['measures']['whole_curve_mean']:.4f}, the control "
        f"{float(np.mean([measures.whole_curve_mean(policy, budget) for policy in found[HALF_SPEED]['baseline'].policies])):.4f}, "
        f"the slower predictor with held divisor {found[SLOWER_SIXTEENTH]['measures']['whole_curve_mean']:.4f}.")


PREDICTIONS = [
    {'title': 'the held divisor alone, held after 1/16 of the run',
     'parts': [('It is level with the control on the whole-curve mean (within 0.01)', [], level_with_the_control),
               ('It is ahead of the control on the final level', [], ahead_on_the_final_level)]},
    {'title': 'holding the divisor after 1/64 instead of 1/16',
     'parts': [('It raises the return between 66 and 200 thousand steps with the predictor at RND\'s speed', [],
                raises_the_stretch(HELD_SIXTY_FOURTH, HELD_SIXTEENTH)),
               ('It raises the return between 66 and 200 thousand steps with the slower predictor', [],
                raises_the_stretch(SLOWER_SIXTY_FOURTH, SLOWER_SIXTEENTH)),
               ('With the slower predictor it is ahead of the control on the whole-curve mean', [],
                ahead_of_the_control(SLOWER_SIXTY_FOURTH))]},
    {'title': 'the predictor at half speed',
     'parts': [('Its curve lies between the control\'s and the finalist\'s at every logging window', [], between_at_every_window)]},
]
