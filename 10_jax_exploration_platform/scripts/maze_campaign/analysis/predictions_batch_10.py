"""The written predictions of batch 10 (its ideas.md, section "Predictions"), read from pairs at one configuration.

PointMaze: the fast last layer (step 0.1) with the first-arrival hold on the first validation block, against the
first-arrival hold alone, the fast last layer alone (phase `discovery-validation`, same seeds) and RND, each at the
same configuration (`screen.same_configuration`), and against RND's best (`screen.validation_cases`). AntMaze
U-maze: each arm of the screen in the AntMaze baseline folder against the RND unit of that folder's check at the
same configuration and seeds, at 128 and 16 environments per policy.
"""

from . import measures, validation_seeds
from .prediction_readings import (ANT, LARGE_ARRIVAL, LARGE_STEP_COST, MEDIUM_ARRIVAL, MEDIUM_STEP_COST,
                                  against_control_best, cell_text, first_of, first_waiting, group_case, m_text, means,
                                  paired_check, pairs_of, respawns, signs_words, w_text)
from .same_configuration import cell_key

TITLE = 'Fast last layer (step 0.1) with the first-arrival hold, first validation block'
ARM = 'fast-layer-first-arrival-hold'
HOLD = 'divisor-held-from-first-arrival'
FAST = 'fast-last-layer-step-0-point-1'
# The two configurations of each case: the first paired with the hold alone, the second with the fast last layer alone.
FIRST = {MEDIUM_ARRIVAL: (0.006, 4.0), MEDIUM_STEP_COST: (0.006, 2.0), LARGE_ARRIVAL: (0.002, 2.0), LARGE_STEP_COST: (0.002, 8.0)}
SECOND = {MEDIUM_ARRIVAL: (0.01, 4.0), MEDIUM_STEP_COST: (0.01, 4.0), LARGE_ARRIVAL: (0.002, 16.0), LARGE_STEP_COST: (0.0045, 8.0)}
ANT_ARMS = {'hold': "divisor held from each policy's first arrival", 'fast': 'fast last layer (step 0.1)',
            'both': 'fast last layer (step 0.1) with the first-arrival hold'}
COUNTS = (128, 16)
# Prediction 3: "within 0.005" and "behind by more than 0.01"; prediction 4: "at least 6 of the 8", "at least 0.06"
# and "at least 0.09"; prediction 7 is read as at most twice the check's respawns per step.
WITHIN, BEHIND, MOST, ANT_BEST, ANT_FINAL, RATE = 0.005, 0.01, 6, 0.06, 0.09, 2.0

READING = ('Where a prediction leaves a choice, it is read as follows. On PointMaze every comparison is paired on the '
           'first validation block (seeds 10000 to 10255) at one configuration; "RND\'s best" is RND at its best '
           'configuration for that metric among all those run on the same seeds. "Equals within its interval" and '
           '"level" are read as: the 95% interval of the paired difference holds zero; "above with the interval above '
           'zero", "behind" and "lowers the first-goal time" as: the interval is entirely on that side; "within 0.005" '
           'and "behind by more than 0.01" are read on the mean difference. "Ahead of RND\'s best on both metrics" is '
           'read as: ahead on $W$ against RND\'s best for $W$ and ahead on $M$ against RND\'s best for $M$, each with '
           'the interval on its side of zero. On the ant (16 development seeds) "above" or "below RND at the same '
           'configuration" counts the configurations by the sign of the mean paired difference; "the whole-curve mean '
           'is not ahead" is read as: above RND in fewer than 6 of the 8 configurations; "between the two single '
           'arms" as: at the same configuration, between the fast last layer\'s and the hold\'s $W$; "as rare as in '
           'the check" as: at most twice the check\'s respawns per environment step at the same number of '
           'environments per policy.')


def cell(case, which):
    """The named configuration of a case as keyword settings."""
    learning_rate, weight = which[case]
    return {'learning_rate': learning_rate, 'int_coef': weight}


def paired(case, which, other, quantity, verdict):
    """A part: the arm against another side at one of the case's two configurations (prediction_readings.paired_check)."""
    return paired_check(TITLE, case, other, quantity, verdict, **cell(case, which))


def behind_rnd_best(case):
    """A part: the arm at the case's second configuration is behind RND's best W, the interval below zero."""
    def check(screen):
        """Read the configuration against RND at its best for W on the first block."""
        found, waiting = against_control_best(screen, case, ARM, 'whole', **cell(case, SECOND))
        if found is None:
            return None, waiting
        return found['result']['whole_curve_mean']['side'] == 'behind', (
            f"At {cell_text(found['group'])}: $W$ {found['measures']['whole_curve_mean']:.4f} against RND's best "
            f"{found['control_measures']['whole_curve_mean']:.4f}; difference {w_text(found['result'])}.")
    return check


def none_ahead_on_both(case):
    """A part: no configuration of the arm is ahead of RND's best on both metrics in a case."""
    def check(screen):
        """Read every configuration of the arm on the first block against RND's best for each metric."""
        waiting = first_waiting(screen, case, ARM)
        if waiting:
            return None, waiting
        words, both = [], 0
        for group in first_of(screen, case)['sides'][ARM].groups:
            settings = {'learning_rate': group.setting('learning_rate'), 'int_coef': group.setting('int_coef')}
            whole = against_control_best(screen, case, ARM, 'whole', **settings)[0]['result']
            discovery = against_control_best(screen, case, ARM, 'discovery', **settings)[0]['result']
            both += whole['whole_curve_mean']['side'] == 'ahead' and discovery['capped_first_goal_step']['side'] == 'ahead'
            words.append(f'{cell_text(group)}: $W$ {w_text(whole)}, $M$ {m_text(discovery)}')
        return both == 0, 'Against RND\'s best for each metric: ' + '; '.join(words) + '.'
    return check


def ant_case(screen, arm, count):
    """One ant group as read: the arm, RND of the check, and their pairs; or the words for what it waits for."""
    title = f'AntMaze U-maze, {count} environments per policy: {ANT_ARMS[arm]} against RND'
    found = group_case(screen, title, ANT)
    other = f'cleanrl-ppo-rnd-with-{count}-environments-per-policy'
    pairs = pairs_of(screen, title, ANT, other)
    if pairs is None or len(found['arm'].groups) < found['arm'].planned:
        return None, f'Waiting for complete units of {ANT_ARMS[arm]} and of RND at {count} environments per policy.'
    return {'arm': found['arm'], 'rnd': found['others'][0], 'pairs': pairs, 'budget': found['budget']}, None


def ant_signs(arm, count, quantity, verdict):
    """A part: on the ant, the signs of the mean differences from RND over the 8 configurations, judged by `verdict`."""
    def check(screen):
        """Count the configurations above and below RND at the same configuration."""
        found, waiting = ant_case(screen, arm, count)
        if found is None:
            return None, waiting
        values = means(found['pairs'], quantity)
        if quantity == 'capped_first_goal_step':
            # before: differences in steps [-273644.0, ...]; after: the same in thousand steps, '-274'.
            words = '$M$ difference from RND: ' + signs_words([value / 1e3 for value in values], 0, ' thousand steps')
        else:
            words = '$W$ difference from RND: ' + signs_words(values)
        return verdict(values), words + '.'

    return check


def ant_best(arm, count, quantity, least):
    """A part: on the ant, the arm's best W (or its final level there) is at least `least`."""
    def check(screen):
        """Read the arm's best configuration for W among its 8."""
        found, waiting = ant_case(screen, arm, count)
        if found is None:
            return None, waiting
        budget = found['budget']
        best = max(found['arm'].groups, key=lambda group: measures.configuration_measures(group.policies, budget)['whole_curve_mean'])
        rnd = max(measures.configuration_measures(group.policies, budget)['whole_curve_mean'] for group in found['rnd'].groups)
        mine = measures.configuration_measures(best.policies, budget)
        return mine[quantity] >= least, (
            f"At its best configuration for $W$ {cell_text(best)}: $W$ {mine['whole_curve_mean']:.4f}, $L$ "
            f"{mine['final_level']:.4f}; RND's best $W$ over its {len(found['rnd'].groups)} configurations of the check {rnd:.4f}.")
    return check


def ant_both(count, quantity):
    """A part: the combined arm on the ant against the two single arms at the same configurations."""
    def check(screen):
        """Pair the combined arm with the fast last layer (M), or place its W between the two single arms (W)."""
        sides = {arm: ant_case(screen, arm, count) for arm in ANT_ARMS}
        if any(found is None for found, _ in sides.values()):
            return None, ' '.join(waiting for found, waiting in sides.values() if found is None)
        budget = sides['both'][0]['budget']
        groups = {arm: {cell_key(group): group for group in found['arm'].groups} for arm, (found, _) in sides.items()}
        kept = []
        for key, group in sorted(groups['both'].items()):
            if quantity == 'capped_first_goal_step':
                result = validation_seeds.compared(group, groups['fast'][key], budget, 'batch 10 prediction 6', count, key)
                kept.append(result[quantity]['side'] == 'level')
            else:
                value = lambda found: measures.configuration_measures(found.policies, budget)['whole_curve_mean']
                low, high = sorted((value(groups['fast'][key]), value(groups['hold'][key])))
                kept.append(low <= value(group) <= high)
        words = ('the interval of the paired difference in $M$ from the fast last layer alone holds zero' if quantity ==
                 'capped_first_goal_step' else '$W$ lies between the fast last layer\'s and the hold\'s')
        return sum(kept) >= MOST, f'{words} in {sum(kept)} of {len(kept)} configurations.'
    return check


def ant_respawns(count):
    """A part: every ant arm's fault-guard respawns per step at most twice the check's at the same count."""
    def check(screen):
        """Count the respawns of each arm's unit and of the check's RND units."""
        sides = {arm: ant_case(screen, arm, count) for arm in ANT_ARMS}
        if any(found is None for found, _ in sides.values()):
            return None, ' '.join(waiting for found, waiting in sides.values() if found is None)
        check_count, check_steps = respawns(sides['hold'][0]['rnd'])
        limit = RATE * check_count / check_steps
        rates = {arm: respawns(found['arm']) for arm, (found, _) in sides.items()}
        words = '; '.join(f'{ANT_ARMS[arm]} {found} in {steps / 1e9:.1f} billion steps' for arm, (found, steps) in rates.items())
        return all(found / steps <= limit for found, steps in rates.values()), (
            f'The check: {check_count} in {check_steps / 1e9:.1f} billion steps. The screen: {words}.')
    return check


level = lambda found: found['side'] == 'level'
ahead = lambda found: found['side'] == 'ahead'
within = lambda found: abs(found['mean']) <= WITHIN
far_behind = lambda found: found['mean'] < -BEHIND
CASES = (('Medium with reward 1 on arrival', MEDIUM_ARRIVAL), ('Medium with the step cost', MEDIUM_STEP_COST),
         ('Large with reward 1 on arrival', LARGE_ARRIVAL), ('Large with the step cost', LARGE_STEP_COST))

PREDICTIONS = [
    {'title': 'PointMaze, discovery unchanged by the hold',
     'parts': [(f'{name}, second configuration: $M$ equals the fast last layer\'s alone within the interval', [],
                paired(case, SECOND, FAST, 'capped_first_goal_step', level)) for name, case in CASES]},
    {'title': 'PointMaze, the hold gives back part of what the fast last layer costs',
     'parts': [(f'{name}, second configuration: $W$ above the fast last layer\'s alone, interval above zero', [],
                paired(case, SECOND, FAST, 'whole_curve_mean', ahead)) for name, case in CASES]
              + [(f'{name}, second configuration: still behind RND\'s best', [], behind_rnd_best(case))
                 for name, case in CASES[:2]]},
    {'title': "PointMaze, at the hold's own best configurations",
     'parts': [('Large with reward 1 on arrival: the first-goal time is lower than the hold alone\'s, interval below zero', [],
                paired(LARGE_ARRIVAL, FIRST, HOLD, 'capped_first_goal_step', ahead)),
               ('Large with the step cost: the first-goal time is lower than the hold alone\'s, interval below zero', [],
                paired(LARGE_STEP_COST, FIRST, HOLD, 'capped_first_goal_step', ahead)),
               ('Large with reward 1 on arrival: $W$ level with the hold alone (within 0.005)', [],
                paired(LARGE_ARRIVAL, FIRST, HOLD, 'whole_curve_mean', within)),
               ('Large with the step cost: $W$ level with the hold alone (within 0.005)', [],
                paired(LARGE_STEP_COST, FIRST, HOLD, 'whole_curve_mean', within)),
               ('Medium with reward 1 on arrival: $W$ behind the hold alone by more than 0.01', [],
                paired(MEDIUM_ARRIVAL, FIRST, HOLD, 'whole_curve_mean', far_behind)),
               ('Medium with reward 1 on arrival: no configuration ahead of RND\'s best on both metrics', [],
                none_ahead_on_both(MEDIUM_ARRIVAL)),
               ('Medium with the step cost: no configuration ahead of RND\'s best on both metrics', [],
                none_ahead_on_both(MEDIUM_STEP_COST))]},
    {'title': 'ant, the first-arrival hold',
     'parts': [part for count in COUNTS for part in (
         (f'{count} environments per policy: $W$ above RND at the same configuration in at least 6 of 8', [],
          ant_signs('hold', count, 'whole_curve_mean', lambda values: sum(value > 0 for value in values) >= MOST)),
         (f'{count} environments per policy: its best $W$ is at least 0.06', [], ant_best('hold', count, 'whole_curve_mean', ANT_BEST)),
         (f'{count} environments per policy: its final level at that configuration is at least 0.09', [],
          ant_best('hold', count, 'final_level', ANT_FINAL)))]},
    {'title': 'ant, the fast last layer',
     'parts': [part for count in COUNTS for part in (
         (f'{count} environments per policy: $M$ below RND at the same configuration in at least 6 of 8', [],
          ant_signs('fast', count, 'capped_first_goal_step', lambda values: sum(value < 0 for value in values) >= MOST)),
         (f'{count} environments per policy: $W$ not ahead (above RND in fewer than 6 of 8)', [],
          ant_signs('fast', count, 'whole_curve_mean', lambda values: sum(value > 0 for value in values) < MOST)))]},
    {'title': 'ant, both together',
     'parts': [part for count in COUNTS for part in (
         (f'{count} environments per policy: $M$ as the fast last layer\'s (in at least 6 of 8 configurations)', [],
          ant_both(count, 'capped_first_goal_step')),
         (f'{count} environments per policy: $W$ between the two single arms (in at least 6 of 8 configurations)', [],
          ant_both(count, 'whole_curve_mean')))]},
    {'title': 'ant, the physics guard',
     'parts': [(f'{count} environments per policy: respawns as rare as in the check', [], ant_respawns(count)) for count in COUNTS]},
]
