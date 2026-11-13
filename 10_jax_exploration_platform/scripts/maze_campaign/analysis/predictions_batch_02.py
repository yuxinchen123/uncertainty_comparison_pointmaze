"""The written predictions of batch 2 (its ideas.md, section "Predictions"), each as parts the screen's numbers decide.

The form is that of predictions_batch_01.py: PREDICTIONS has one entry per numbered prediction of ideas.md, in
its order, with `title`, `parts` and optionally `beside`. READING says how the words of the predictions are read
where they leave a choice; the summary prints it.
"""

from . import batch_summary, report
from .predictions_batch_01 import CONTROL, arrival, difference_words, measure, ratio_words, slope_words

CLIP = {'clip at 10': 'wider-input-clip-10', 'clip at 20': 'wider-input-clip-20', 'no clip': 'no-input-clip'}
KEPT = {'no clip': 'input-statistics-kept-from-the-random-action-rollouts-no-clip',
        'clip at 20': 'input-statistics-kept-from-the-random-action-rollouts-clip-20'}
SLOWER = {0.0625: 'slower-predictor-0-point-0625-times-1e-4', 0.125: 'slower-predictor-0-point-125-times-1e-4'}
LOCAL = {0.5: 'local-features-as-the-input-of-rnd-length-0-point-5', 1: 'local-features-as-the-input-of-rnd-length-1',
         2: 'local-features-as-the-input-of-rnd-length-2'}
# (decay a, length) -> the method.
SCHEDULED = {(0.5, 0.5): 'scheduled-last-layer-on-local-features-decay-0-point-5-length-0-point-5',
             (0.5, 1): 'scheduled-last-layer-on-local-features-decay-0-point-5-length-1',
             (0.5, 2): 'scheduled-last-layer-on-local-features-decay-0-point-5-length-2',
             (1, 1): 'scheduled-last-layer-on-local-features-decay-1-length-1'}
# The variants prediction 5 speaks of: the lengths 0.5 and 1.
SCHEDULED_READ = {key: slug for key, slug in SCHEDULED.items() if key[1] in (0.5, 1)}
HELD = {'one sixteenth': 'divisor-held-after-one-sixteenth-of-the-run', 'one eighth': 'divisor-held-after-one-eighth-of-the-run',
        'one quarter': 'divisor-held-after-one-quarter-of-the-run'}
# The method of the validation block that a hold fraction was trained as.
HELD_IN_THE_VALIDATION_BLOCK = {'one sixteenth': 'divisor-held-after-one-sixteenth', 'one quarter': 'divisor-held-after-one-quarter'}
ARRIVAL_DROP = 0.15
COMPARED_WEIGHT = 2.0
LARGER_WEIGHTS = (4.0, 16.0)

READING = ('Where a prediction leaves a choice, it is read as follows. Every method is read at its best intrinsic '
           'weight for $W$ unless a weight is named. "Arrivals at weight 2 are earlier" is read on the mean capped '
           'first-goal time $M$ at the intrinsic weight 2. "Within the interval" is read as: the 95% interval of the '
           'difference in $W$ from the control contains zero. A prediction that names two possible outcomes holds '
           'when one of the two occurred, and the numbers say which. "Stays above 0.9 until they are visited" is '
           'read on the bonus at unvisited cells over its start at 5% and at 25% of the budget, where enough pairs '
           'are left to print a median. "More than 10 times the bonus at visited cells through the first quarter" is '
           'read on the median bonus of the (policy, cell) pairs with no visit so far over the median of the pairs '
           'with a visit, at 5% and at 25% of the budget, where both kinds have enough pairs; with no such share the '
           'part cannot be read. The held divisor\'s whole-curve mean is read at the hold fraction with the '
           'highest $W$. "The highest of the batch" is read among the methods of this batch other than the control.')


def validation_words(screen, slug):
    """A method's row of the validation block at the best weights, in words; empty without such a row."""
    rows = (screen.validation or {}).get('best') or {}
    row = next((row for row in rows.get('whole', ()) if row['slug'] == slug), None)
    if row is None or row['result'] is None or row['result']['seeds'] < 2:
        return ''
    found = row['result']
    return (f"On the validation block ({found['seeds']} seeds), at its best weight {row['weight']:g} against the control "
            f"at the control's best weight: difference in $W$ {batch_summary.fine_difference(found['whole_curve_mean'], 4)}, "
            f"in $L$ {batch_summary.fine_difference(found['final_level'], 4)}.")


# ---- prediction 1: wider input clip -------------------------------------------------------------------------------

def start_ratio_above(slug, bound):
    """A part: the method's median start ratio is above a bound."""
    def check(screen):
        """Whether the largest over the smallest starting bonus (median over policies) is above the bound."""
        ratio = screen.methods[slug].grid['start_ratio']
        return ratio is not None and ratio > bound, f'Median start ratio: {ratio_words(ratio)}.'
    return check


def clip_arrival(screen):
    """Whether the clip at 20 or no clip arrives at least 15% earlier than the control, at the best weights."""
    theirs, theirs_words = arrival(screen, CONTROL)
    found = {words: arrival(screen, CLIP[words]) for words in ('clip at 20', 'no clip')}
    held = any(mine != float('inf') and mine <= (1 - ARRIVAL_DROP) * theirs for mine, _ in found.values())
    return held, ('Median first-arrival step: ' + '; '.join(f'{words} {mine_words}' for words, (_, mine_words) in found.items())
                  + f'; control {theirs_words}.')


def clip_whole(screen):
    """Whether the whole-curve mean of each clip value is within 0.02 of the control's."""
    means = {words: screen.differences[slug]['whole_curve_mean']['mean'] for words, slug in CLIP.items()}
    return all(abs(mean) < 0.02 for mean in means.values()), (
        'Difference in $W$ from the control: '
        + '; '.join(f"{words} {difference_words(screen, CLIP[words], 'whole_curve_mean')}" for words in CLIP) + '.')


# ---- prediction 2: input statistics kept from the random-action rollouts --------------------------------------------

def apart_words(entry):
    """One share's reading of the bonus at unvisited cells over the bonus at visited cells, in words."""
    pairs = f"{entry['unvisited_pairs']} pairs without a visit, {entry['visited_pairs']} with one"
    return f'too few pairs for a median ({pairs})' if entry['ratio'] is None else f"{batch_summary.short(entry['ratio'])} ({pairs})"


def kept_unvisited(screen):
    """Whether the median bonus at unvisited cells is more than 10 times the median at visited cells, at 5% and 25% of the budget."""
    # A share at which too few pairs of either kind are left has no median: it is named and decides nothing.
    shares = batch_summary.UNVISITED_SHARES[:2]
    found = {words: screen.methods[slug].grid['unvisited_over_visited'] for words, slug in KEPT.items()}
    ratios = [entry[share]['ratio'] for entry in found.values() for share in shares if entry[share]['ratio'] is not None]
    words = '; '.join(f'{name}: ' + ', '.join(f'{apart_words(entry[share])} at {100 * share:g}%' for share in shares)
                      for name, entry in found.items())
    numbers = f'Median bonus at unvisited cells over the median at visited cells, at the best weight: {words}.'
    return (None if not ratios else all(ratio > 10 for ratio in ratios)), numbers


def kept_outcome(screen):
    """Whether each variant shows one of the two written outcomes: 0.01 to 0.03 ahead with earlier arrivals, or behind."""
    theirs, theirs_words = arrival(screen, CONTROL)
    held, words = True, []
    for name, slug in KEPT.items():
        mean = screen.differences[slug]['whole_curve_mean']['mean']
        mine, mine_words = arrival(screen, slug)
        outcome = ('ahead with earlier arrivals' if 0.01 <= mean <= 0.03 and mine < theirs else 'behind' if mean < 0
                   else 'neither written outcome')
        held = held and outcome != 'neither written outcome'
        grid = screen.methods[slug].grid
        words.append(f"{name}: {outcome} (difference in $W$ {difference_words(screen, slug, 'whole_curve_mean')}; median first "
                     f"arrival {mine_words} against {theirs_words} for the control; divisor after the first logging window "
                     f"{batch_summary.short(grid['divisor']['first_window'])}; final level {measure(screen, slug, 'final_level'):.3f})")
    return held, '; '.join(words) + '.'


# ---- prediction 3: slower predictor -----------------------------------------------------------------------------------

def slower_arrivals(screen):
    """Whether both slower predictors have a smaller M than the control at the intrinsic weight 2."""
    theirs = screen.methods[CONTROL].discovery_scores[COMPARED_WEIGHT][0]
    mine = {multiple: screen.methods[slug].discovery_scores[COMPARED_WEIGHT][0] for multiple, slug in SLOWER.items()}
    return all(value < theirs for value in mine.values()), (
        f'$M$ at the intrinsic weight {COMPARED_WEIGHT:g}, in thousands of steps: '
        + '; '.join(f'{report.thousands(value)} at {multiple:g} times 1e-4' for multiple, value in mine.items())
        + f'; control {report.thousands(theirs)}.')


def slower_inside(screen):
    """Whether the interval of each slower predictor's difference in W from the control contains zero."""
    sides = {multiple: screen.differences[slug]['whole_curve_mean']['side'] for multiple, slug in SLOWER.items()}
    return all(side == 'level' for side in sides.values()), (
        'Difference in $W$ from the control: '
        + '; '.join(f"{multiple:g} times 1e-4: {difference_words(screen, slug, 'whole_curve_mean')}" for multiple, slug in SLOWER.items()) + '.')


# ---- prediction 4: local features as the input of RND -------------------------------------------------------------------

def local_ratio(screen):
    """Whether the median start ratio of every length is below 3."""
    ratios = {length: screen.methods[slug].grid['start_ratio'] for length, slug in LOCAL.items()}
    return all(ratio is not None and ratio < 3 for ratio in ratios.values()), (
        'Median start ratio: ' + '; '.join(f'{ratio_words(ratio)} at length {length:g}' for length, ratio in ratios.items()) + '.')


def local_outcome(screen):
    """Whether each length shows one of the two written outcomes: all policies arrive with W above 0.8, or 12 of 16 or fewer arrive."""
    held, words = True, []
    for length, slug in LOCAL.items():
        row = screen.methods[slug].measures
        reached = round(row['reached_goal_share'] * row['seeds'])
        outcome = ('all policies reach the goal with $W$ above 0.8' if reached == row['seeds'] and row['whole_curve_mean'] > 0.8
                   else 'three quarters of the policies or fewer reach the goal' if reached <= 0.75 * row['seeds']
                   else 'neither written outcome')
        held = held and outcome != 'neither written outcome'
        words.append(f"length {length:g}: {outcome} ({reached} of {row['seeds']} policies reach the goal, $W$ {row['whole_curve_mean']:.3f})")
    return held, '; '.join(words) + '.'


# ---- prediction 5: scheduled last layer on local features -----------------------------------------------------------------

def scheduled_words(key):
    """A variant of the scheduled last layer in words."""
    return f'decay {key[0]:g}, length {key[1]:g}'


def scheduled_unvisited(screen):
    """Whether, at the lengths 0.5 and 1, the bonus at unvisited cells stays above 0.9 of its start where a median is printed."""
    held, words = True, []
    for key, slug in SCHEDULED_READ.items():
        shares = screen.methods[slug].grid['unvisited']
        medians = [shares[share]['median'] for share in batch_summary.UNVISITED_SHARES[:2] if shares[share]['median'] is not None]
        held = held and all(median > 0.9 for median in medians)
        words.append(f'{scheduled_words(key)}: ' + ', '.join(
            f'{batch_summary.unvisited_cell(shares[share])} at {100 * share:g}%' for share in batch_summary.UNVISITED_SHARES[:2]))
    return held, 'Bonus at unvisited cells over its start: ' + '; '.join(words) + '.'


def scheduled_slope(screen):
    """Whether, at the lengths 0.5 and 1, the median decay slope is within 0.1 of minus the decay."""
    slopes = {key: screen.methods[slug].grid['decay_slope'] for key, slug in SCHEDULED_READ.items()}
    return all(slope is not None and abs(slope + key[0]) <= 0.1 for key, slope in slopes.items()), (
        'Median decay slope: ' + '; '.join(f'{slope_words(slope)} at {scheduled_words(key)}' for key, slope in slopes.items()) + '.')


def scheduled_reach(screen):
    """Whether, at the lengths 0.5 and 1, every policy reaches the goal at some intrinsic weight."""
    found = {}
    for key, slug in SCHEDULED_READ.items():
        weights = [weight for weight, (_, share) in screen.methods[slug].discovery_scores.items() if share == 1.0]
        found[key] = sorted(weights)
    return all(found.values()), 'Intrinsic weights at which every policy reaches the goal: ' + '; '.join(
        f"{scheduled_words(key)}: {', '.join(f'{weight:g}' for weight in weights) if weights else 'none'}" for key, weights in found.items()) + '.'


def scheduled_whole(screen):
    """The whole-curve mean of every variant against the control and against the batch's highest, for the last sentence."""
    ideas = {slug: measure(screen, slug, 'whole_curve_mean') for slug in screen.methods if slug != CONTROL}
    highest = max(ideas, key=ideas.get)
    rows = '; '.join(f"{scheduled_words(key)}: $W$ {ideas[slug]:.3f}, difference from the control "
                     f"{difference_words(screen, slug, 'whole_curve_mean')}, median first arrival {arrival(screen, slug)[1]}"
                     for key, slug in SCHEDULED.items() if slug in ideas)
    return (f'{rows}. The highest $W$ among the complete methods of the batch other than the control: '
            f'{screen.names[highest]} ({ideas[highest]:.3f}); the control has {measure(screen, CONTROL, "whole_curve_mean"):.3f}.')


# ---- prediction 6: divisor held after a part of the run -------------------------------------------------------------------

def held_reward(screen):
    """Whether, for every hold fraction, the weighted intrinsic reward at the end is below its value at a quarter of the budget."""
    rewards = {words: screen.methods[slug].grid['intrinsic_reward'] for words, slug in HELD.items()}
    return all(reward[1.0] < reward[0.25] for reward in rewards.values()), (
        'Weighted intrinsic reward at a quarter of the budget and at the end: '
        + '; '.join(f'held after {words}: {batch_summary.short(reward[0.25])} and {batch_summary.short(reward[1.0])}'
                    for words, reward in rewards.items()) + '.')


def control_reward(screen):
    """The control's weighted intrinsic reward at a quarter of the budget and at the end."""
    reward = screen.methods[CONTROL].grid['intrinsic_reward']
    return (f'control: {batch_summary.short(reward[0.25])} at a quarter of the budget and {batch_summary.short(reward[1.0])} at '
            f'the end, {reward[1.0] / reward[0.25]:.1f} times as much.')


def held_levels(screen):
    """Whether, for every hold fraction, the final level at the weights 4 and 16 is above the control's at the same weight."""
    theirs = screen.methods[CONTROL].scores
    held, words = True, []
    for name, slug in HELD.items():
        mine = screen.methods[slug].scores
        held = held and all(mine[weight][1] > theirs[weight][1] for weight in LARGER_WEIGHTS)
        words.append(f'held after {name}: ' + ', '.join(f'{mine[weight][1]:.3f} at weight {weight:g}' for weight in LARGER_WEIGHTS))
    return held, ('Final level: ' + '; '.join(words) + '; control: '
                  + ', '.join(f'{theirs[weight][1]:.3f} at weight {weight:g}' for weight in LARGER_WEIGHTS) + '.')


def held_whole(screen):
    """Whether the hold fraction with the highest W is 0.005 to 0.015 ahead of the control."""
    best = max(HELD, key=lambda words: measure(screen, HELD[words], 'whole_curve_mean'))
    mean = screen.differences[HELD[best]]['whole_curve_mean']['mean']
    return 0.005 <= mean <= 0.015, (f"Held after {best} (the highest $W$ of the three): difference in $W$ from the control "
                                    f"{difference_words(screen, HELD[best], 'whole_curve_mean')}.")


def held_beside(screen):
    """Arrivals per hold fraction, and what the validation block says of the held divisor."""
    rows = '; '.join(
        f"held after {words}: {round(measure(screen, slug, 'reached_goal_share') * measure(screen, slug, 'seeds'))} of "
        f"{measure(screen, slug, 'seeds')} policies reach the goal, median first arrival {arrival(screen, slug)[1]}"
        for words, slug in HELD.items())
    block = ' '.join(f'Held after {words}: {text}' for words, slug in HELD_IN_THE_VALIDATION_BLOCK.items()
                     if (text := validation_words(screen, slug)))
    return f"{rows}; control: median first arrival {arrival(screen, CONTROL)[1]}." + (f' {block}' if block else '')


PREDICTIONS = [
    {'title': 'wider input clip',
     'parts': [('The start ratio is above 300 with the clip at 10', [CLIP['clip at 10']], start_ratio_above(CLIP['clip at 10'], 300)),
               ('The start ratio is above 1,000 with no clip', [CLIP['no clip']], start_ratio_above(CLIP['no clip'], 1000)),
               ('With the clip at 20 or with no clip, the median first arrival is at least 15% earlier than the control\'s',
                [CLIP['clip at 20'], CLIP['no clip'], CONTROL], clip_arrival),
               ('The whole-curve mean moves by less than 0.02 either way', list(CLIP.values()) + [CONTROL], clip_whole)]},
    {'title': 'input statistics kept from the random-action rollouts',
     'parts': [('The bonus at cells never visited stays more than 10 times the bonus at visited cells through the first quarter of the budget',
                list(KEPT.values()), kept_unvisited),
               ('One of the two written outcomes occurs: 0.01 to 0.03 ahead with earlier arrivals, or behind',
                list(KEPT.values()) + [CONTROL], kept_outcome)]},
    {'title': 'slower predictor',
     'parts': [('Arrivals at the intrinsic weight 2 are earlier than the control\'s', list(SLOWER.values()) + [CONTROL], slower_arrivals),
               ('The whole-curve mean is within the interval', list(SLOWER.values()) + [CONTROL], slower_inside)]},
    {'title': 'local features as the input of RND',
     'parts': [('The start ratio is below 3', list(LOCAL.values()), local_ratio),
               ('One of the two written outcomes occurs: every policy reaches the goal with $W$ above 0.8, or 12 of 16 or fewer reach it',
                list(LOCAL.values()), local_outcome)]},
    {'title': 'scheduled last layer on local features',
     'parts': [('At the lengths 0.5 and 1 the bonus at cells never visited stays above 0.9 of its start', list(SCHEDULED_READ.values()),
                scheduled_unvisited),
               ('At the lengths 0.5 and 1 the decay slope is within 0.1 of minus the decay', list(SCHEDULED_READ.values()), scheduled_slope),
               ('At the lengths 0.5 and 1 every policy reaches the goal at some intrinsic weight', list(SCHEDULED_READ.values()),
                scheduled_reach)],
     'beside': [(list(SCHEDULED.values()) + [CONTROL], scheduled_whole)]},
    {'title': 'divisor held after a part of the run',
     'parts': [('The weighted intrinsic reward at the end of the run is below its value at a quarter of the budget',
                list(HELD.values()), held_reward),
               ('The final level at the weights 4 and 16 is higher than the control\'s at the same weights',
                list(HELD.values()) + [CONTROL], held_levels),
               ('At the best weight the whole-curve mean is 0.005 to 0.015 ahead of the control', list(HELD.values()) + [CONTROL],
                held_whole)],
     'beside': [([CONTROL], control_reward), (list(HELD.values()) + [CONTROL], held_beside)]},
]
