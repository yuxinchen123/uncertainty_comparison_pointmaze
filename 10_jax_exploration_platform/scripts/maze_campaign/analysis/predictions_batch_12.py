"""The written predictions of batch 12 (its ideas.md, section "Predictions"), read on the first validation block.

Batch 12 has no screen: the fast last layer with the first-arrival hold at steps 0.03 and 0.01 is read at one
configuration per case against the hold alone, the step 0.1 of batch 10 and RND (`screen.same_configuration` and
`screen.validation_cases`).
"""

from .prediction_readings import (LARGE_ARRIVAL, LARGE_STEP_COST, MEDIUM_ARRIVAL, MEDIUM_STEP_COST,
                                  against_control_best, at_cell, cell_text, paired_check, thousands, w_text)

STEP_03_TITLE = 'Fast last layer at step 0.03 with the first-arrival hold, first validation block'
STEP_01_TITLE = 'Fast last layer at step 0.01 with the first-arrival hold, first validation block'
STEPS = (('0.1', 'fast-layer-first-arrival-hold'), ('0.03', 'fast-step-0-point-03-with-hold'),
         ('0.01', 'fast-step-0-point-01-with-hold'))
HOLD = 'divisor-held-from-first-arrival'
CONFIGURATION = {LARGE_ARRIVAL: (0.002, 2.0), LARGE_STEP_COST: (0.002, 8.0), MEDIUM_ARRIVAL: (0.006, 4.0),
                 MEDIUM_STEP_COST: (0.006, 2.0)}
# Prediction 2: "at least 60 thousand steps below" and "within 0.005"; prediction 3: "behind by 0.02 to 0.05" and
# "within 0.01".
SOONER, WITHIN, BEHIND, CLOSE = 60_000, 0.005, (0.02, 0.05), 0.01

READING = ('Where a prediction leaves a choice, it is read as follows. Every reading is on the first validation block '
           '(seeds 10000 to 10255) at the batch\'s one configuration per case. "Both rise as the step falls" is read '
           'on the means: $W$ and $M$ at step 0.1 below those at step 0.03, and those below the ones at step 0.01; '
           'the hold alone (step 0) is given beside. "Below with the interval below zero" and "above with the '
           'interval above zero" are read on the 95% interval of the paired difference; "at least 60 thousand '
           'steps", "within 0.005", "behind by 0.02 to 0.05" and "within 0.01" on the mean difference. "RND\'s best" '
           'is RND at its best configuration for $W$ among all those run on the same seeds.')


def cell(case):
    """The batch's configuration of a case as keyword settings."""
    learning_rate, weight = CONFIGURATION[case]
    return {'learning_rate': learning_rate, 'int_coef': weight}


def ordered(case):
    """A part: in a case, W and M both rise from step 0.1 to 0.03 to 0.01."""
    def check(screen):
        """Read each step's configuration, and the hold alone's beside."""
        found = [at_cell(screen, case, slug, **cell(case)) for _, slug in STEPS + (('0 (the hold alone)', HOLD),)]
        if any(entry is None for entry, _ in found):
            return None, ' '.join(words for entry, words in found if entry is None)
        values = [entry[1] for entry, _ in found]
        rising = lambda name: values[0][name] < values[1][name] < values[2][name]
        words = '; '.join(f"step {step}: $W$ {row['whole_curve_mean']:.4f}, $M$ {thousands(row['capped_first_goal_step'])}"
                          for (step, _), row in zip(STEPS + (('0 (the hold alone)', HOLD),), values))
        return rising('whole_curve_mean') and rising('capped_first_goal_step'), f'At {cell_text(found[0][0][0])}: {words}.'
    return check


def step_ahead_of_rnd(screen):
    """Whether step 0.03 or step 0.01 is ahead of RND's best W on Large with reward 1 on arrival."""
    found = [against_control_best(screen, LARGE_ARRIVAL, slug, 'whole', **cell(LARGE_ARRIVAL)) for _, slug in STEPS[1:]]
    if any(entry is None for entry, _ in found):
        return None, ' '.join(words for entry, words in found if entry is None)
    return any(entry['result']['whole_curve_mean']['side'] == 'ahead' for entry, _ in found), ' '.join(
        f"Step {step}: $W$ {entry['measures']['whole_curve_mean']:.4f}, difference from RND's best "
        f"({entry['control_measures']['whole_curve_mean']:.4f}) {w_text(entry['result'])}."
        for (step, _), (entry, _) in zip(STEPS[1:], found))


M, W = 'capped_first_goal_step', 'whole_curve_mean'
CASES = (('Large with reward 1 on arrival', LARGE_ARRIVAL), ('Large with the step cost', LARGE_STEP_COST),
         ('Medium with reward 1 on arrival', MEDIUM_ARRIVAL), ('Medium with the step cost', MEDIUM_STEP_COST))

PREDICTIONS = [
    {'title': 'ordered by the step',
     'parts': [(f'{name}: $W$ and $M$ both rise from step 0.1 to 0.03 to 0.01', [], ordered(case)) for name, case in CASES]},
    {'title': 'PointMaze Large with reward 1 on arrival',
     'parts': [('Step 0.03: $M$ at least 60 thousand steps below the hold alone\'s, interval below zero', [],
                paired_check(STEP_03_TITLE, LARGE_ARRIVAL, HOLD, M, lambda found: found['side'] == 'ahead' and found['mean'] <= -SOONER,
                             **cell(LARGE_ARRIVAL))),
               ('Step 0.03: $W$ within 0.005 of the hold alone\'s', [],
                paired_check(STEP_03_TITLE, LARGE_ARRIVAL, HOLD, W, lambda found: abs(found['mean']) <= WITHIN, **cell(LARGE_ARRIVAL))),
               ("Step 0.03 or 0.01: $W$ above RND's best with the interval above zero", [], step_ahead_of_rnd)]},
    {'title': 'PointMaze Medium with reward 1 on arrival',
     'parts': [('Step 0.03 is behind the hold alone by 0.02 to 0.05', [],
                paired_check(STEP_03_TITLE, MEDIUM_ARRIVAL, HOLD, W, lambda found: -BEHIND[1] <= found['mean'] <= -BEHIND[0],
                             **cell(MEDIUM_ARRIVAL))),
               ('Step 0.01 is within 0.01 of the hold alone', [],
                paired_check(STEP_01_TITLE, MEDIUM_ARRIVAL, HOLD, W, lambda found: abs(found['mean']) <= CLOSE, **cell(MEDIUM_ARRIVAL)))]},
]
