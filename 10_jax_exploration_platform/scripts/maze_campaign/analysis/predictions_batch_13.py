"""The written predictions of batch 13 (its ideas.md, section "Predictions"), read on the first validation block.

Batch 13 has no screen: the fast last layer until each policy's first arrival, then the divisor of the predictor's
own scores (held, or left running), is read against the hold alone, the arms of batches 10 and 11 and RND at the
same configuration (`screen.same_configuration`), and each side at its best (`screen.validation_cases`).
"""

from .prediction_readings import (LARGE_ARRIVAL, LARGE_STEP_COST, MEDIUM_ARRIVAL, MEDIUM_STEP_COST,
                                  against_control_best, best_against_arm, best_entry, pair_words, paired_check, pairs_of,
                                  w_text)

HELD_TITLE = "Fast last layer until the first arrival, then the predictor's own divisor, held"
RUNNING_TITLE = "The predictor's own divisor left running"
HELD = 'fast-then-predictor-divisor-held'
HOLD = 'divisor-held-from-first-arrival'
UNTIL = 'fast-until-arrival-then-hold'
BATCH_10 = 'fast-layer-first-arrival-hold'
WITHIN = 0.01

READING = ('Where a prediction leaves a choice, it is read as follows. Every reading is on the first validation block '
           '(seeds 10000 to 10255), paired by seed. "The first-goal time as the fast last layer\'s at the same '
           'configuration" is read against an arm that runs the same program until the first arrival, at the same '
           'configuration: batch 11\'s arm (the fast last layer until the first arrival, then the hold) where it was '
           'trained there, else batch 10\'s arm (the fast last layer throughout, with the hold); it holds when the '
           '95% interval of the paired difference in $M$ holds zero. "Within 0.01" is read on the mean difference, '
           'at every configuration the arm and the hold alone both trained (final level, prediction 2) or at each '
           'side\'s best (whole-curve mean, prediction 5). "Ahead with the interval above zero" is read on the interval; '
           '"ahead of or level with" and "not behind" as: the interval is not entirely below zero; "the higher final '
           'level" on the mean difference. "RND\'s best" is RND at its best configuration for $W$ among all those run '
           'on the same seeds.')


def at(learning_rate, weight):
    """A configuration as keyword settings."""
    return {'learning_rate': learning_rate, 'int_coef': weight}


def final_level_close(case):
    """A part: at every configuration both trained, the held arm's final level is within 0.01 of the hold alone's."""
    def check(screen):
        """Read every pair with the hold alone in the case."""
        found = pairs_of(screen, HELD_TITLE, case, HOLD)
        if not found:
            return None, 'Waiting for complete units of the arm and of the hold alone at a common configuration.'
        return all(abs(pair['result']['final_level']['mean']) <= WITHIN for pair in found), ' '.join(
            pair_words(pair, 'final_level') for pair in found)
    return check


def ahead_of_rnd_at(case, *cells):
    """A part: at one of the named configurations the held arm is ahead of RND's best W, the interval above zero."""
    def check(screen):
        """Read each configuration against RND's best for W."""
        found = [against_control_best(screen, case, HELD, 'whole', **settings) for settings in cells]
        if any(entry is None for entry, _ in found):
            return None, ' '.join(words for entry, words in found if entry is None)
        return any(entry['result']['whole_curve_mean']['side'] == 'ahead' for entry, _ in found), ' '.join(
            f"At (learning rate {entry['group'].setting('learning_rate'):g}, weight {entry['group'].setting('int_coef'):g}): "
            f"$W$ {entry['measures']['whole_curve_mean']:.4f}, difference from RND's best {w_text(entry['result'])}."
            for entry, _ in found)
    return check


def against_arm_at_best(case, other, verdict):
    """A part: the held arm at its best W against another arm at its best W on the first block, judged by `verdict`."""
    def check(screen):
        """Read best against best."""
        entry, waiting = best_against_arm(screen, case, HELD, other)
        if entry is None:
            return None, waiting
        return verdict(entry['result']['whole_curve_mean']), (
            f"$W$ {entry['measures']['whole_curve_mean']:.4f} against {entry['baseline_measures']['whole_curve_mean']:.4f}; "
            f"difference {w_text(entry['result'])}.")
    return check


def not_behind_rnd(case):
    """A part: the held arm at its best W is not behind RND's best."""
    def check(screen):
        """Read best against best."""
        entry, waiting = best_entry(screen, case, HELD)
        if entry is None:
            return None, waiting
        return entry['result']['whole_curve_mean']['side'] != 'behind', (
            f"$W$ {entry['measures']['whole_curve_mean']:.4f}; difference from RND's best {w_text(entry['result'])}.")
    return check


level = lambda found: found['side'] == 'level'
not_behind = lambda found: found['side'] != 'behind'
close = lambda found: abs(found['mean']) <= WITHIN
lower = lambda found: found['mean'] < 0
M = 'capped_first_goal_step'
CASES = (('Medium with reward 1 on arrival', MEDIUM_ARRIVAL), ('Medium with the step cost', MEDIUM_STEP_COST),
         ('Large with reward 1 on arrival', LARGE_ARRIVAL), ('Large with the step cost', LARGE_STEP_COST))

PREDICTIONS = [
    {'title': "the first-goal time as the fast last layer's at the same configuration",
     'parts': [("Medium with reward 1 on arrival (6e-3, weight 4), against batch 11's arm", [],
                paired_check(HELD_TITLE, MEDIUM_ARRIVAL, UNTIL, M, level, **at(0.006, 4.0))),
               ("Large with reward 1 on arrival (2e-3, weight 2), against batch 11's arm", [],
                paired_check(HELD_TITLE, LARGE_ARRIVAL, UNTIL, M, level, **at(0.002, 2.0))),
               ("Large with reward 1 on arrival (2e-3, weight 4), against batch 11's arm", [],
                paired_check(HELD_TITLE, LARGE_ARRIVAL, UNTIL, M, level, **at(0.002, 4.0))),
               ("Large with the step cost (2e-3, weight 8), against batch 10's arm", [],
                paired_check(HELD_TITLE, LARGE_STEP_COST, BATCH_10, M, level, **at(0.002, 8.0)))]},
    {'title': 'the failure of batch 11 is gone at full length',
     'parts': [(f"{name}: the final level within 0.01 of the hold alone's at every common configuration", [],
                final_level_close(case)) for name, case in CASES]},
    {'title': 'PointMaze Large with reward 1 on arrival',
     'parts': [("Ahead of RND's best with the interval above zero at weight 1 or 2", [],
                ahead_of_rnd_at(LARGE_ARRIVAL, at(0.002, 1.0), at(0.002, 2.0)))]},
    {'title': 'PointMaze Large with the step cost',
     'parts': [("Ahead of RND's best with the interval above zero at one of its three configurations", [],
                ahead_of_rnd_at(LARGE_STEP_COST, at(0.002, 4.0), at(0.002, 8.0), at(0.003, 8.0))),
               ("Not behind batch 10's arm (each at its best)", [], against_arm_at_best(LARGE_STEP_COST, BATCH_10, not_behind))]},
    {'title': 'PointMaze Medium',
     'parts': [part for name, case in CASES[:2] for part in (
         (f'{name}: within 0.01 of the hold alone on $W$ (each at its best)', [], against_arm_at_best(case, HOLD, close)),
         (f"{name}: ahead of or level with RND's best", [], not_behind_rnd(case)))]},
    {'title': 'running against held',
     'parts': [(f'{name}: the held arm has the higher final level', [],
                paired_check(RUNNING_TITLE, case, HELD, 'final_level', lower, **settings))
               for name, case, settings in (('Medium with reward 1 on arrival (6e-3, weight 2)', MEDIUM_ARRIVAL, at(0.006, 2.0)),
                                            ('Large with reward 1 on arrival (2e-3, weight 2)', LARGE_ARRIVAL, at(0.002, 2.0)),
                                            ('Large with the step cost (2e-3, weight 8)', LARGE_STEP_COST, at(0.002, 8.0)))]},
]
