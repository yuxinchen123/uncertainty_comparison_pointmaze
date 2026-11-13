"""The written predictions of batch 11 (its ideas.md: section "Predictions" and addendum 1), read on the first block.

Batch 11 has no screen: the fast last layer until each policy's first arrival is read on the first validation block
against the first-arrival hold alone, batch 10's arm (the fast last layer throughout with the hold) and RND at the
same configuration (`screen.same_configuration`), and against RND's best (`screen.validation_cases`).
"""

from . import measures
from .prediction_readings import (LARGE_ARRIVAL, MEDIUM_ARRIVAL, MEDIUM_STEP_COST, CONTROL, against_control_best,
                                  cell_text, pair_words, paired_check, pairs_of, thousands, w_text)

THEN_HOLD = 'Fast last layer until the first arrival, then the first-arrival hold'
RUNNING = "Fast last layer until the first arrival, then RND's running divisor (addendum 1)"
FRACTION = 'Fast last layer until the first arrival, divisor held after one sixteenth of the run (addendum 1)'
ARM = 'fast-until-arrival-then-hold'
NO_HOLD = 'fast-until-arrival-no-hold'
HOLD = 'divisor-held-from-first-arrival'
BATCH_10 = 'fast-layer-first-arrival-hold'
# Prediction 2: "not behind by more than 0.003"; prediction 4: "by 0.01 to 0.02"; prediction 6: "within 0.03";
# prediction 8: "below 150 thousand steps".
SMALL, GAIN, CLOSE, EARLY = 0.003, (0.01, 0.02), 0.03, 150_000

READING = ('Where a prediction leaves a choice, it is read as follows. Every reading is on the first validation block '
           '(seeds 10000 to 10255), paired by seed. "The hold\'s own best configuration" in a case is, among the '
           'configurations both this arm and the hold alone trained, the one at which the hold alone has its highest '
           '$W$. "Within the interval of the difference" and "level" are read as: the 95% interval of the paired '
           'difference holds zero; "ahead" and "behind" as: the interval is entirely above or below zero; "at least '
           'as good" as: not behind. "Not behind by more than 0.003", "by 0.01 to 0.02" and "within 0.03" are read on '
           'the mean difference. "RND\'s best" is RND at its best configuration for that metric among all those run on '
           'the same seeds.')


def at(learning_rate, weight):
    """A configuration as keyword settings."""
    return {'learning_rate': learning_rate, 'int_coef': weight}


def hold_best_pair(case):
    """Among the arm's pairs with the hold alone in a case, the one where the hold alone has its highest W."""
    def find(screen):
        """Return (the pair, None) or (None, the waiting words)."""
        found = pairs_of(screen, THEN_HOLD, case, HOLD)
        if not found:
            return None, 'Waiting for complete units of the arm and of the hold alone at a common configuration.'
        return max(found, key=lambda pair: pair['other_measures']['whole_curve_mean']), None
    return find


def at_hold_best(case, verdict):
    """A part: the arm's difference in W from the hold alone at the hold's best common configuration, by `verdict`."""
    def check(screen):
        """Read that pair."""
        found, waiting = hold_best_pair(case)(screen)
        if found is None:
            return None, waiting
        return verdict(found['result']['whole_curve_mean']), pair_words(found, 'whole_curve_mean')
    return check


def ahead_of_rnd_best(case):
    """A part: the arm at its best W on the first block is ahead of RND's best, the interval above zero."""
    def check(screen):
        """Read every configuration of the arm against RND's best for W and keep its best."""
        found = [against_control_best(screen, case, ARM, 'whole', **at(group.setting('learning_rate'), group.setting('int_coef')))
                 for group in screen.validation_cases[case]['first']['sides'][ARM].groups]
        if not found or any(entry is None for entry, _ in found):
            return None, 'Waiting for complete units of the arm.'
        best = max((entry for entry, _ in found), key=lambda entry: entry['measures']['whole_curve_mean'])
        return best['result']['whole_curve_mean']['side'] == 'ahead', (
            f"At {cell_text(best['group'])}: $W$ {best['measures']['whole_curve_mean']:.4f}; difference from RND's best "
            f"{w_text(best['result'])}.")
    return check


def large_gain(screen):
    """Whether the arm is ahead of RND's best on Large with reward 1 on arrival by 0.01 to 0.02 at weight 4 or 16."""
    found = [against_control_best(screen, LARGE_ARRIVAL, ARM, 'whole', **at(0.002, weight)) for weight in (4.0, 16.0)]
    if any(entry is None for entry, _ in found):
        return None, ' '.join(words for entry, words in found if entry is None)
    held = any(entry['result']['whole_curve_mean']['side'] == 'ahead' and
               GAIN[0] <= entry['result']['whole_curve_mean']['mean'] <= GAIN[1] for entry, _ in found)
    return held, ' '.join(f"At {cell_text(entry['group'])}: $W$ {entry['measures']['whole_curve_mean']:.4f}; difference "
                          f"from RND's best {w_text(entry['result'])}." for entry, _ in found)


def running_keeps_the_early_arrival(weight):
    """A part: on Large with reward 1 on arrival the arm with RND's running divisor first arrives within 150 thousand steps."""
    def check(screen):
        """Read the arm's configuration at the weight."""
        side = screen.validation_cases[LARGE_ARRIVAL]['first']['sides'][NO_HOLD]
        group = next((group for group in side.groups if group.setting('int_coef') == weight), None)
        if group is None or len(side.groups) < side.planned:
            return None, 'Waiting for complete units of the arm with the running divisor.'
        found = measures.configuration_measures(group.policies, side.budget)
        return found['capped_first_goal_step'] < EARLY, f"At {cell_text(group)}: $M$ {thousands(found['capped_first_goal_step'])}."
    return check


def running_ahead_somewhere(screen):
    """Whether the arm with the running divisor is ahead of RND's best W at one of its three weights on Large."""
    found = [against_control_best(screen, LARGE_ARRIVAL, NO_HOLD, 'whole', **at(0.002, weight)) for weight in (2.0, 4.0, 16.0)]
    if any(entry is None for entry, _ in found):
        return None, ' '.join(words for entry, words in found if entry is None)
    return any(entry['result']['whole_curve_mean']['side'] == 'ahead' for entry, _ in found), ' '.join(
        f"At {cell_text(entry['group'])}: $W$ {entry['measures']['whole_curve_mean']:.4f}; difference from RND's best "
        f"{w_text(entry['result'])}." for entry, _ in found)


level = lambda found: found['side'] == 'level'
behind = lambda found: found['side'] == 'behind'
not_behind = lambda found: found['side'] != 'behind'
small = lambda found: found['mean'] >= -SMALL
close = lambda found: abs(found['mean']) <= CLOSE
M, W, L = 'capped_first_goal_step', 'whole_curve_mean', 'final_level'

PREDICTIONS = [
    {'title': "the first-goal time is that of batch 10's arm at the same configuration",
     'parts': [('Medium with reward 1 on arrival (6e-3, weight 4)', [], paired_check(THEN_HOLD, MEDIUM_ARRIVAL, BATCH_10, M, level, **at(0.006, 4.0))),
               ('Medium with reward 1 on arrival (1e-2, weight 4)', [], paired_check(THEN_HOLD, MEDIUM_ARRIVAL, BATCH_10, M, level, **at(0.01, 4.0))),
               ('Medium with the step cost (6e-3, weight 2)', [], paired_check(THEN_HOLD, MEDIUM_STEP_COST, BATCH_10, M, level, **at(0.006, 2.0))),
               ('Large with reward 1 on arrival (2e-3, weight 16)', [], paired_check(THEN_HOLD, LARGE_ARRIVAL, BATCH_10, M, level, **at(0.002, 16.0)))]},
    {'title': 'the cost after the arrival is gone',
     'parts': [(f"{name}: $W$ not behind the hold alone by more than 0.003 at the hold's best configuration", [], at_hold_best(case, small))
               for name, case in (('Medium with reward 1 on arrival', MEDIUM_ARRIVAL), ('Medium with the step cost', MEDIUM_STEP_COST),
                                  ('Large with reward 1 on arrival', LARGE_ARRIVAL))]},
    {'title': 'PointMaze Medium',
     'parts': [part for name, case in (('reward 1 on arrival', MEDIUM_ARRIVAL), ('step cost', MEDIUM_STEP_COST)) for part in (
         (f'{name}: level with the hold alone on $W$', [], at_hold_best(case, level)),
         (f"{name}: ahead of RND's best, as the hold is", [], ahead_of_rnd_best(case)))]},
    {'title': 'PointMaze Large with reward 1 on arrival',
     'parts': [("Ahead of RND's best with the interval above zero at weight 4 or 16, by 0.01 to 0.02", [], large_gain)]},
    {'title': 'without the hold',
     'parts': [('Medium with reward 1 on arrival (6e-3, weight 4): behind the arm with the hold on $L$', [],
                paired_check(RUNNING, MEDIUM_ARRIVAL, ARM, L, behind, **at(0.006, 4.0))),
               ('Medium with reward 1 on arrival (6e-3, weight 4): level with it on $M$', [],
                paired_check(RUNNING, MEDIUM_ARRIVAL, ARM, M, level, **at(0.006, 4.0))),
               ('Large with reward 1 on arrival (2e-3, weight 4): behind the arm with the hold on $L$', [],
                paired_check(RUNNING, LARGE_ARRIVAL, ARM, L, behind, **at(0.002, 4.0))),
               ('Large with reward 1 on arrival (2e-3, weight 4): level with it on $M$', [],
                paired_check(RUNNING, LARGE_ARRIVAL, ARM, M, level, **at(0.002, 4.0)))]},
    {'title': "addendum 1: the arm with RND's running divisor against RND at the same configuration",
     'parts': [('Medium with reward 1 on arrival (6e-3, weight 2): within 0.03 of RND on $W$', [],
                paired_check(RUNNING, MEDIUM_ARRIVAL, CONTROL, W, close, **at(0.006, 2.0)))],
     'beside': [([], lambda screen: 'At weight 4: ' + paired_check(RUNNING, MEDIUM_ARRIVAL, CONTROL, W, close,
                                                                   **at(0.006, 4.0))(screen)[1])]},
    {'title': 'addendum 1: the divisor held after one sixteenth against the running divisor',
     'parts': [('Medium with reward 1 on arrival (6e-3, weight 4): at least as good on $W$', [],
                paired_check(FRACTION, MEDIUM_ARRIVAL, NO_HOLD, W, not_behind, **at(0.006, 4.0))),
               ('Large with reward 1 on arrival (2e-3, weight 4): at least as good on $W$', [],
                paired_check(FRACTION, LARGE_ARRIVAL, NO_HOLD, W, not_behind, **at(0.002, 4.0)))]},
    {'title': 'addendum 1: the running divisor on PointMaze Large with reward 1 on arrival',
     'parts': [('First arrival within 150 thousand steps at weight 4', [], running_keeps_the_early_arrival(4.0)),
               ('First arrival within 150 thousand steps at weight 16', [], running_keeps_the_early_arrival(16.0)),
               ("$W$ ahead of RND's best at one of the three weights", [], running_ahead_somewhere)]},
]
