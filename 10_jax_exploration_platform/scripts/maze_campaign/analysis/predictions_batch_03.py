"""The written predictions of batch 3 (its ideas.md, section "Predictions"), each as parts the screen's numbers decide.

The form is that of predictions_batch_01.py. The batch trains no control of its own: its control is the unit of
batch 2, which the summary's settings name.
"""

from . import report
from .predictions_batch_01 import CONTROL, difference_words, measure
from .predictions_batch_02 import validation_words

SLOW_HELD = 'slower-predictor-with-held-divisor'
FAST_HELD = {0.1: 'fast-last-layer-step-0-point-1-with-held-divisor', 0.3: 'fast-last-layer-step-0-point-3-with-held-divisor'}
FAST_SLOW = 'fast-last-layer-step-0-point-1-with-slower-predictor'
THREE = {'one sixteenth': 'fast-last-layer-step-0-point-1-slower-predictor-held-divisor',
         'one eighth': 'fast-last-layer-step-0-point-1-slower-predictor-divisor-held-after-one-eighth'}
IDEAS = (SLOW_HELD, *FAST_HELD.values(), FAST_SLOW, *THREE.values())
# The methods of the validation block that the combinations were trained as.
IN_THE_VALIDATION_BLOCK = {SLOW_HELD: 'slower-predictor-with-held-divisor', THREE['one sixteenth']: 'three-parts-held-one-sixteenth',
                           THREE['one eighth']: 'three-parts-held-one-eighth'}
RECOVERED_WEIGHTS = (1.0, 2.0)

READING = ('Where a prediction leaves a choice, it is read as follows. Every method is read at its best intrinsic '
           'weight for the metric named: the whole-curve mean $W$ at the best weight for $W$, the mean capped '
           'first-goal time $M$ at the best weight for $M$. "The fast last layer with held divisor" is read at the '
           'step 0.1, the step whose final levels without the held divisor the prediction names. "Of the batch" is '
           'read among the six combinations of this batch; the control is named beside. "All three" is read with '
           'the divisor held after one sixteenth of the run, except where the two hold fractions are compared.')


def discovery(screen, slug, name):
    """One measure of a method at its best intrinsic weight for M."""
    return screen.methods[slug].discovery_measures[name]


# ---- prediction 1: slower predictor with held divisor ---------------------------------------------------------------

def slow_held_whole(screen):
    """Whether the whole-curve mean at the best weight is at least 0.015 above the control's best."""
    mean = screen.differences[SLOW_HELD]['whole_curve_mean']['mean']
    return mean >= 0.015, f"Difference in $W$ from the control: {difference_words(screen, SLOW_HELD, 'whole_curve_mean')}."


def slow_held_level(screen):
    """Whether the final level is at least the control's."""
    mine, theirs = measure(screen, SLOW_HELD, 'final_level'), measure(screen, CONTROL, 'final_level')
    return mine >= theirs, f'Final level {mine:.3f} against {theirs:.3f} for the control.'


def validation_beside(slug):
    """A beside entry: the method's row of the validation block at the best weights, when the block holds one."""
    def words(screen):
        """What the validation block says of the method, or that it has no complete unit there."""
        found = validation_words(screen, IN_THE_VALIDATION_BLOCK[slug])
        return f"{screen.names[slug]}: {found or 'the validation block has no complete unit of this method yet.'}"
    return ([slug], words)


# ---- prediction 2: fast last layer with held divisor ----------------------------------------------------------------

def fast_held_levels(screen):
    """Whether the final level at the weights 1 and 2 is 0.99 or more, at the step 0.1."""
    scores = screen.methods[FAST_HELD[0.1]].scores
    return all(scores[weight][1] >= 0.99 for weight in RECOVERED_WEIGHTS), (
        'Final level: ' + ', '.join(f'{scores[weight][1]:.3f} at weight {weight:g}' for weight in RECOVERED_WEIGHTS) + '.')


def fast_held_discovery(screen):
    """Whether the mean capped first-goal time at its best weight is below 100 thousand steps."""
    capped = discovery(screen, FAST_HELD[0.1], 'capped_first_goal_step')
    return capped < 100e3, (f'$M$ {report.thousands(capped)} thousand steps at the intrinsic weight '
                            f'{screen.methods[FAST_HELD[0.1]].discovery_weight:g}.')


def fast_held_whole(screen):
    """Whether the whole-curve mean at its best weight is above the control's best."""
    mine, theirs = measure(screen, FAST_HELD[0.1], 'whole_curve_mean'), measure(screen, CONTROL, 'whole_curve_mean')
    return mine > theirs, (f"$W$ {mine:.3f} against {theirs:.3f} for the control; difference "
                           f"{difference_words(screen, FAST_HELD[0.1], 'whole_curve_mean')}.")


def fast_held_larger_step(screen):
    """The same three numbers at the step 0.3."""
    slug = FAST_HELD[0.3]
    scores = screen.methods[slug].scores
    return ('at the step 0.3: final level ' + ', '.join(f'{scores[weight][1]:.3f} at weight {weight:g}' for weight in RECOVERED_WEIGHTS)
            + f"; $M$ {report.thousands(discovery(screen, slug, 'capped_first_goal_step'))} thousand steps at its best weight for $M$; "
            f"$W$ {measure(screen, slug, 'whole_curve_mean'):.3f} at its best weight for $W$.")


# ---- prediction 3: all three ----------------------------------------------------------------------------------------

def best_of_the_batch(value, better, words, text_of):
    """A part: all three (divisor held after one sixteenth) has the best value among the six combinations; open while units are missing."""
    def check(screen):
        """True or False once it is settled; None while a missing unit could still change the answer."""
        mine = value(screen, THREE['one sixteenth'])
        others = {slug: value(screen, slug) for slug in IDEAS if slug in screen.methods and slug != THREE['one sixteenth']}
        waiting = [screen.names[slug] for slug in IDEAS if slug not in screen.methods]
        numbers = f'{words} {text_of(mine)}'
        if others:
            first = min(others, key=lambda slug: -others[slug] if better == 'highest' else others[slug])
            numbers += f'; the {better} of the other complete combinations is {screen.names[first]} at {text_of(others[first])}'
            if (others[first] > mine) if better == 'highest' else (others[first] < mine):
                return False, numbers + '.'
        return (None, f"{numbers}. Waiting for: {'; '.join(waiting)}.") if waiting else (True, numbers + '.')
    return check


def three_hold_change(screen):
    """Whether holding after one eighth instead of one sixteenth changes the whole-curve mean by less than 0.01."""
    values = {words: measure(screen, slug, 'whole_curve_mean') for words, slug in THREE.items()}
    return abs(values['one eighth'] - values['one sixteenth']) < 0.01, (
        '$W$ ' + ' and '.join(f'{value:.3f} with the divisor held after {words}' for words, value in values.items()) + '.')


def control_beside(screen):
    """The control's two numbers, for "of the batch"."""
    return (f"control: $W$ {measure(screen, CONTROL, 'whole_curve_mean'):.3f} at its best weight for $W$, $M$ "
            f"{report.thousands(discovery(screen, CONTROL, 'capped_first_goal_step'))} thousand steps at its best weight for $M$.")


# ---- prediction 4: combinations that do not run -----------------------------------------------------------------------

def all_ran(screen):
    """Whether every unit of the batch ran to its end; None while one has not."""
    units = [unit for unit in screen.units if unit['slug'] in IDEAS]
    waiting = [unit['name'] for unit in units if unit['state'] != 'complete']
    if waiting:
        return None, f"Not complete: {'; '.join(waiting)}."
    return True, f'All {len(units)} combinations are complete: none was refused or failed, so none had to be reported.'


PREDICTIONS = [
    {'title': 'slower predictor with held divisor',
     'parts': [('The whole-curve mean at the best weight is at least 0.015 above the control\'s best', [SLOW_HELD, CONTROL], slow_held_whole),
               ('The final level is at least the control\'s', [SLOW_HELD, CONTROL], slow_held_level)],
     'beside': [validation_beside(SLOW_HELD)]},
    {'title': 'fast last layer with held divisor',
     'parts': [('The final level at the weights 1 and 2 is 0.99 or more', [FAST_HELD[0.1]], fast_held_levels),
               ('The mean capped first-goal time at its best weight is below 100 thousand steps', [FAST_HELD[0.1]], fast_held_discovery),
               ('The whole-curve mean at its best weight is above the control\'s best', [FAST_HELD[0.1], CONTROL], fast_held_whole)],
     'beside': [([FAST_HELD[0.3]], fast_held_larger_step)]},
    {'title': 'all three',
     'parts': [('The smallest mean capped first-goal time of the batch', [THREE['one sixteenth']],
                best_of_the_batch(lambda screen, slug: discovery(screen, slug, 'capped_first_goal_step'), 'smallest',
                                  '$M$ in thousands of steps:', report.thousands)),
               ('The highest whole-curve mean of the batch', [THREE['one sixteenth']],
                best_of_the_batch(lambda screen, slug: measure(screen, slug, 'whole_curve_mean'), 'highest', '$W$',
                                  lambda value: f'{value:.3f}')),
               ('Holding after one eighth instead of one sixteenth changes the whole-curve mean by less than 0.01',
                list(THREE.values()), three_hold_change)],
     'beside': [([CONTROL], control_beside), validation_beside(THREE['one sixteenth']), validation_beside(THREE['one eighth'])]},
    {'title': 'combinations that do not run',
     'parts': [('Every combination ran to its end, so none had to be reported as not runnable', [], all_ran)]},
]
