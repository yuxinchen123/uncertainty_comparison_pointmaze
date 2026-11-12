"""The written predictions of batch 1 (its ideas.md, section "Predictions"), each as parts the screen's numbers decide.

PREDICTIONS has one entry per numbered prediction of ideas.md, in its order:

- `title`: the idea the prediction is about, in words;
- `parts`: (the claim in words, the methods whose complete units it needs, a function of the screen that returns
  whether the claim held and the numbers it was read from; the function returns None in place of True or False
  when the claim cannot be read yet);
- `beside` (optional): (the methods needed, a function of the screen that returns numbers shown beside the parts
  with no word of their own).

Every method is read at its best intrinsic weight unless a weight is named. READING says how the words of the
predictions are read where they leave a choice; the summary prints it.
"""

from . import batch_summary, report

CONTROL = 'control-cleanrl-ppo-rnd'
EVEN_START = 'even-start'
FIXED_SCALE = 'fixed-scale'
NO_DIVISOR = 'even-start-with-no-divisor'
ERROR_LENGTH = 'error-length-as-the-bonus'
# The own setting of the three ideas that have one: its value -> the method.
PREDICTOR = {0.25: 'separate-predictor-step-size-multiple-0-point-25', 1: 'separate-predictor-step-size-multiple-1',
             4: 'separate-predictor-step-size-multiple-4'}
FAST = {0.1: 'fast-last-layer-inside-the-rollout-step-0-point-1', 0.3: 'fast-last-layer-inside-the-rollout-step-0-point-3',
        1: 'fast-last-layer-inside-the-rollout-step-1'}
SCHEDULED = {0.5: 'scheduled-last-layer-on-fixed-features-decay-0-point-5',
             0.75: 'scheduled-last-layer-on-fixed-features-decay-0-point-75',
             1: 'scheduled-last-layer-on-fixed-features-decay-1'}
LARGE_WEIGHTS = (16.0, 64.0)
ARRIVAL_DROP = 0.15

READING = ('Where a prediction leaves a choice, it is read as follows. Every method is read at its best intrinsic '
           'weight unless a weight is named. "Large weights" are 16 and 64. "Well above 2" is read as above 2. '
           '"Same final level" and "changes by less than its interval" are read as: the 95% interval of the '
           'difference from the control contains zero. The three separate predictor step sizes are compared with '
           'the middle one (1 times). The fast last layer is read at the step size with the highest $W$. "The '
           'highest final level at $a$ = 0.5" is read among the three decay values.')


def measure(screen, slug, name):
    """One measure of a method at its best intrinsic weight."""
    return screen.methods[slug].measures[name]


def arrival(screen, slug):
    """A method's median first-arrival step, as a number and in words."""
    value = measure(screen, slug, 'median_first_goal_step')
    return value, batch_summary.steps_text(value)


def difference_words(screen, slug, quantity):
    """A method's paired difference from the control with its 95% interval."""
    return report.interval_text(screen.differences[slug][quantity]).rsplit(',', 1)[0]


def ratio_words(value):
    """A start ratio in words."""
    return 'no usable value' if value is None else f'{value:.2f}'


def slope_words(value):
    """A decay slope in words."""
    return 'no slope' if value is None else f'{value:.2f}'


def named(screen, methods):
    """The values of an idea's own setting with one measure list each: 'name W 0.812, L 0.990, ...' per method."""
    return '; '.join(
        f"{screen.names[slug]}: $W$ {measure(screen, slug, 'whole_curve_mean'):.3f}, $L$ "
        f"{measure(screen, slug, 'final_level'):.3f}, median first arrival {arrival(screen, slug)[1]}"
        for slug in methods.values())


# ---- prediction 1: even start ---------------------------------------------------------------------------------------

def even_start_inside(screen):
    """Whether the interval of even start's difference in W from the control contains zero."""
    result = screen.differences[EVEN_START]['whole_curve_mean']
    return result['side'] == 'level', f"Difference in $W$ from the control: {difference_words(screen, EVEN_START, 'whole_curve_mean')}."


def even_start_ratio(screen):
    """Whether even start's median start ratio rounds to 1.00."""
    ratio = screen.methods[EVEN_START].grid['start_ratio']
    return ratio is not None and round(ratio, 2) == 1.0, f'Median start ratio: {ratio_words(ratio)}.'


def control_ratio(screen):
    """Whether the control's median start ratio is above 2."""
    ratio = screen.methods[CONTROL].grid['start_ratio']
    return ratio is not None and ratio > 2, f'Median start ratio: {ratio_words(ratio)}.'


# ---- prediction 2: fixed scale, and even start with no divisor --------------------------------------------------------

def large_weight_parts(slug, name):
    """The three parts of prediction 2 for one of its two methods."""
    def higher_levels(screen):
        """Whether the method's final level is above the control's at each large weight."""
        mine, theirs = screen.methods[slug].scores, screen.methods[CONTROL].scores
        words = '; '.join(f'at weight {weight:g}: {mine[weight][1]:.3f} against {theirs[weight][1]:.3f} for the control'
                          for weight in LARGE_WEIGHTS)
        return all(mine[weight][1] > theirs[weight][1] for weight in LARGE_WEIGHTS), f'Final level {words}.'

    def higher_weight(screen):
        """Whether the method's best weight is above the control's."""
        mine, theirs = screen.methods[slug].best_weight, screen.methods[CONTROL].best_weight
        return mine > theirs, f'Best intrinsic weight {mine:g} against {theirs:g} for the control.'

    def rise(screen):
        """Whether the method's W is 0.02 to 0.05 above the control's."""
        mean = screen.differences[slug]['whole_curve_mean']['mean']
        return 0.02 <= mean <= 0.05, f"Difference in $W$ from the control: {difference_words(screen, slug, 'whole_curve_mean')}."

    return [(f'{name}: the final level at the large weights is above the control\'s', [slug, CONTROL], higher_levels),
            (f'{name}: the best intrinsic weight is above the control\'s', [slug, CONTROL], higher_weight),
            (f'{name}: at the best weights, $W$ is 0.02 to 0.05 above the control\'s', [slug, CONTROL], rise)]


# ---- prediction 3: separate predictor step size ----------------------------------------------------------------------

def predictor_part(multiple, quantity, lower):
    """One part of prediction 3: a step size against the middle one, in the first-arrival step or the final level."""
    def check(screen):
        """Whether the quantity of this step size is lower (or higher) than the middle step size's."""
        if quantity == 'arrival':
            (mine, mine_words), (middle, middle_words) = arrival(screen, PREDICTOR[multiple]), arrival(screen, PREDICTOR[1])
            words = f'Median first-arrival step {mine_words} against {middle_words} at 1 times.'
        else:
            mine, middle = measure(screen, PREDICTOR[multiple], 'final_level'), measure(screen, PREDICTOR[1], 'final_level')
            words = f'Final level {mine:.3f} against {middle:.3f} at 1 times.'
        return (mine < middle if lower else mine > middle), words

    what = 'median first-arrival step' if quantity == 'arrival' else 'final level'
    return (f"{multiple:g} times: {'lower' if lower else 'higher'} {what} than 1 times",
            [PREDICTOR[multiple], PREDICTOR[1]], check)


def predictor_best(screen):
    """Whether the highest W of the three step sizes is at 1 or 4 times."""
    whole = {multiple: measure(screen, slug, 'whole_curve_mean') for multiple, slug in PREDICTOR.items()}
    best = max(whole, key=whole.get)
    return best in (1, 4), '$W$ ' + '; '.join(f'{value:.3f} at {multiple:g} times' for multiple, value in whole.items()) + '.'


# ---- prediction 4: error length -------------------------------------------------------------------------------------

def error_length_later(screen):
    """Whether error length's median first arrival is later than the control's."""
    (mine, mine_words), (theirs, theirs_words) = arrival(screen, ERROR_LENGTH), arrival(screen, CONTROL)
    return mine > theirs, f'Median first-arrival step {mine_words} against {theirs_words} for the control.'


def error_length_level(screen):
    """Whether the interval of error length's difference in L from the control contains zero."""
    result = screen.differences[ERROR_LENGTH]['final_level']
    return result['side'] == 'level', f"Difference in $L$ from the control: {difference_words(screen, ERROR_LENGTH, 'final_level')}."


def error_length_slopes(screen):
    """The decay slopes of error length and the control, for the words "like a slower decay"."""
    mine, theirs = screen.methods[ERROR_LENGTH].grid['decay_slope'], screen.methods[CONTROL].grid['decay_slope']
    return (f'median decay slope {slope_words(mine)} against {slope_words(theirs)} for the control (a slope nearer '
            'zero is a slower decay).')


# ---- prediction 5: fast last layer inside the rollout -----------------------------------------------------------------

def fast_arrival(screen):
    """Whether the best of the three step sizes (by W) arrives at least 15% earlier than the control."""
    best = max(FAST.values(), key=lambda slug: measure(screen, slug, 'whole_curve_mean'))
    (mine, mine_words), (theirs, theirs_words) = arrival(screen, best), arrival(screen, CONTROL)
    words = f'{screen.names[best]}: median first-arrival step {mine_words} against {theirs_words} for the control'
    if mine == float('inf') or theirs == float('inf'):
        return mine < theirs, words + '.'
    return mine <= (1 - ARRIVAL_DROP) * theirs, f'{words}, {100 * (1 - mine / theirs):.0f}% lower.'


# ---- prediction 6: scheduled last layer on fixed features --------------------------------------------------------------

def scheduled_earliest(screen):
    """Whether decay 1 has the smallest median first-arrival step of the batch; None while that is still open."""
    # One complete method that arrives earlier settles the claim; without one, it holds only once every unit is complete.
    mine, mine_words = arrival(screen, SCHEDULED[1])
    others = {slug: arrival(screen, slug) for slug in screen.methods if slug != SCHEDULED[1]}
    waiting = [screen.names[slug] for slug in screen.order if slug not in screen.methods]
    words = f'Median first-arrival step {mine_words}.'
    if others:
        earliest = min(others, key=lambda slug: others[slug][0])
        words = (f'Median first-arrival step {mine_words}; the earliest of the other complete methods is '
                 f'{screen.names[earliest]} at {others[earliest][1]}.')
        if others[earliest][0] < mine:
            return False, words
    return (None, f"{words} Waiting for: {'; '.join(waiting)}.") if waiting else (True, words)


def scheduled_level(screen):
    """Whether decay 0.5 has the highest final level of the three decay values."""
    level = {decay: measure(screen, slug, 'final_level') for decay, slug in SCHEDULED.items()}
    return level[0.5] >= max(level.values()), 'Final level ' + '; '.join(
        f'{value:.3f} at decay {decay:g}' for decay, value in level.items()) + '.'


def scheduled_spill(screen):
    """The differences in W of the three decay values, and the bonus at unvisited cells, for the last clause."""
    whole = '; '.join(f"decay {decay:g}: {difference_words(screen, slug, 'whole_curve_mean')}" for decay, slug in SCHEDULED.items())
    cells = []
    for words, slug in [('control', CONTROL)] + [(f'decay {decay:g}', slug) for decay, slug in SCHEDULED.items()]:
        shares = screen.methods[slug].grid['unvisited']
        cells.append(f'{words}: ' + ', '.join(f'{batch_summary.unvisited_cell(shares[share])} at {100 * share:g}%'
                                              for share in batch_summary.UNVISITED_SHARES[:2]))
    return (f'difference in $W$ from the control, {whole}. Bonus at unvisited cells over its start (how much one '
            f"input's fit lowers the error at inputs not yet visited), {'; '.join(cells)}.")


PREDICTIONS = [
    {'title': 'even start',
     'parts': [('The interval of the difference in $W$ from the control contains zero', [EVEN_START, CONTROL], even_start_inside),
               ('The start ratio of even start is 1.00', [EVEN_START], even_start_ratio),
               ('The start ratio of the control is above 2', [CONTROL], control_ratio)]},
    {'title': 'fixed scale, and even start with no divisor',
     'parts': large_weight_parts(FIXED_SCALE, 'Fixed scale') + large_weight_parts(NO_DIVISOR, 'Even start with no divisor')},
    {'title': 'separate predictor step size',
     'parts': [predictor_part(4, 'arrival', True), predictor_part(4, 'level', True),
               predictor_part(0.25, 'arrival', False), predictor_part(0.25, 'level', False),
               ('The highest $W$ of the three step sizes is at 1 or 4 times', list(PREDICTOR.values()), predictor_best)],
     'beside': [(list(PREDICTOR.values()), lambda screen: named(screen, PREDICTOR) + '.')]},
    {'title': 'error length as the bonus',
     'parts': [('Later median first arrival than the control', [ERROR_LENGTH, CONTROL], error_length_later),
               ('Same final level as the control', [ERROR_LENGTH, CONTROL], error_length_level)],
     'beside': [([ERROR_LENGTH, CONTROL], error_length_slopes)]},
    {'title': 'fast last layer inside the rollout',
     'parts': [('The median first-arrival step is at least 15% below the control\'s, at the step size with the highest $W$',
                list(FAST.values()) + [CONTROL], fast_arrival)],
     'beside': [(list(FAST.values()), lambda screen: named(screen, FAST) + '.')]},
    {'title': 'scheduled last layer on fixed features',
     'parts': [('Decay 1 has the earliest median first arrival of the batch', [SCHEDULED[1]], scheduled_earliest),
               ('Decay 0.5 has the highest final level of the three decay values', list(SCHEDULED.values()), scheduled_level)],
     'beside': [(list(SCHEDULED.values()) + [CONTROL], scheduled_spill)]},
]
