"""Readings shared by the prediction modules of batches 8 to 14, from the validation seeds and the pairs at one configuration.

Each reading returns (a value, None) once every unit it needs is complete, else (None, the words for what it waits
for), so that a prediction part can say "cannot be read yet" instead of reading a partial number.
"""

import json
from pathlib import Path

from . import case_sweeps, measures, report, same_configuration, validation_seeds

MEDIUM_ARRIVAL = ('pointmaze-medium', 'arrival_plus_one')
MEDIUM_STEP_COST = ('pointmaze-medium', 'step_cost')
LARGE_ARRIVAL = ('pointmaze-large', 'arrival_plus_one')
LARGE_STEP_COST = ('pointmaze-large', 'step_cost')
ANT = ('antmaze-umaze', 'arrival_plus_one')
CONTROL = 'control-cleanrl-ppo-rnd'
# The AntMaze baseline folder, beside this study's folder, whose units hold the ant screens and RND's check.
ANT_FOLDER = ('2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-'
              'tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_'
              'confirmation-seeds-from-30000')
RUNS = Path(__file__).resolve().parents[3] / 'runs'


# ---- words --------------------------------------------------------------------------------------------------------

def w_text(result):
    """A difference in W with its interval."""
    return case_sweeps.fine_difference(result['whole_curve_mean'], 4)


def l_text(result):
    """A difference in L with its interval."""
    return case_sweeps.fine_difference(result['final_level'], 4)


def m_text(result):
    """A difference in M with its interval, in thousand steps."""
    return report.discovery_interval_text(result['capped_first_goal_step']) + ' thousand steps'


def cell_text(group):
    """A configuration as (learning rate, weight)."""
    return f"(learning rate {group.setting('learning_rate'):g}, weight {group.setting('int_coef'):g})"


def thousands(value):
    """A step count in thousand steps."""
    return f'{report.thousands(value)} thousand steps'


# ---- the first validation block -----------------------------------------------------------------------------------

def first_of(screen, case):
    """The first block of a case as validation_seeds.read_first gives it."""
    return screen.validation_cases[case]['first']


def first_waiting(screen, case, *slugs):
    """The words for the arms (and the control) of a case not read in full on the first block; None when all are."""
    first = first_of(screen, case)
    waiting = validation_seeds.unfinished(first['sides'], first['control'], *slugs)
    return f"Waiting for complete units in {case_sweeps.case_words(*case)}, first block: {'; '.join(waiting)}." if waiting else None


def best_entry(screen, case, slug, metric='whole'):
    """An arm at its best for a metric on the first block against the control at its own (compare_at_best's entry)."""
    found, waiting = validation_seeds.first_entries(screen.validation_cases, case, slug, metric=metric)
    return (None, waiting) if found is None else (found[slug], None)


def best_against_arm(screen, case, mine, theirs, metric='whole'):
    """One arm at its best for a metric on the first block against another arm at its own best, paired on the seeds.

    The entry of case_sweeps.compare_at_best, with the other arm's measures at its best (`baseline_measures`).
    """
    waiting = first_waiting(screen, case, mine, theirs)
    if waiting:
        return None, waiting
    first = first_of(screen, case)
    found = case_sweeps.compare_at_best(first['sides'][mine], first['sides'][theirs], first['budget'],
                                        'prediction reading, best against best', mine, theirs, *case)[metric]
    return {**found, 'baseline_measures': measures.configuration_measures(found['baseline'].policies, first['budget'])}, None


def group_at(side, **cell):
    """The configuration of a side with the named cell settings; None when it has none."""
    return next((group for group in side.groups if all(group.setting(name) == value for name, value in cell.items())), None)


def at_cell(screen, case, slug, **cell):
    """An arm's configuration on the first block at the named settings, with its measures."""
    waiting = first_waiting(screen, case, slug)
    if waiting:
        return None, waiting
    first = first_of(screen, case)
    group = group_at(first['sides'][slug], **cell)
    if group is None:
        return None, f'{slug} has no configuration {cell} in {case_sweeps.case_words(*case)}.'
    return (group, measures.configuration_measures(group.policies, first['budget'])), None


def against_control_best(screen, case, slug, metric='whole', **cell):
    """An arm's configuration at the named settings against the control at its best for a metric on the first block."""
    found, waiting = at_cell(screen, case, slug, **cell)
    if found is None:
        return None, waiting
    first = first_of(screen, case)
    control = first['sides'][first['control']]
    best = case_sweeps.best_configuration(control.groups, first['budget'], metric)
    return {'group': found[0], 'measures': found[1], 'control': best,
            'control_measures': measures.configuration_measures(best.policies, first['budget']),
            'result': validation_seeds.compared(found[0], best, first['budget'], 'prediction reading, against the best',
                                                slug, metric, *case, *sorted(cell.items()))}, None


# ---- the second validation block ----------------------------------------------------------------------------------

def second_row(screen, case, slug, metric='whole'):
    """An arm's row on the second block (validation_seeds.second_rows), with its comparison under equal rules."""
    rows, waiting = validation_seeds.second_rows(screen.validation_cases, case, slug, metric=metric)
    return (None, waiting) if rows is None else (rows[slug], None)


# ---- pairs at one configuration -----------------------------------------------------------------------------------

def pair(screen, title, case, other, **cell):
    """The pair of a same-configuration group in one case against another side, at the named settings."""
    return same_configuration.pair_at(screen.same_configuration, title, case, other, **cell)


def pairs_of(screen, title, case, other):
    """Every pair of a same-configuration group in one case against another side; None while that side is not read."""
    found = same_configuration.find_group(screen.same_configuration, title)['cases'][case]
    return found['pairs'][other][0] if other in found['pairs'] else None


def group_case(screen, title, case):
    """A same-configuration group in one case as read: the arm, the other sides, the budget and the pairs."""
    return same_configuration.find_group(screen.same_configuration, title)['cases'][case]


def means(pairs, quantity):
    """The mean paired differences of one quantity over pairs, in their order."""
    return [found['result'][quantity]['mean'] for found in pairs]


def signs_words(values, digits=4, unit=''):
    """How many differences are above and below zero, with their range (in `unit` when one is named)."""
    above, below = sum(value > 0 for value in values), sum(value < 0 for value in values)
    return (f'above zero in {above} and below zero in {below} of {len(values)} configurations '
            f'(from {min(values):+,.{digits}f} to {max(values):+,.{digits}f}{unit})')


# ---- the ant's fault guard ------------------------------------------------------------------------------------------

def respawns(side):
    """The fault-guard respawns of a side's units (each shard's completion record) and the steps they trained."""
    # before: two units of 128 policies x 16,777,216 steps with 12 and 5 respawns; after: (17, 4,294,967,296).
    count = steps = 0
    for unit in side.units:
        lines = (RUNS / ANT_FOLDER / 'data' / f"{unit['unit_id']}.jsonl").read_text().splitlines()
        done = [json.loads(line) for line in lines if '"unit_complete"' in line]
        if not done:
            raise ValueError(f"Unit {unit['unit_id']} has no completion record.")
        count += done[-1]['nan_respawned_episodes']
        steps += done[-1]['env_steps']
    return count, steps


def pair_words(found, quantity):
    """A pair's values and difference in one quantity, in words."""
    mine, theirs, result = found['arm_measures'], found['other_measures'], found['result']
    if quantity == 'capped_first_goal_step':
        return (f"At {cell_text(found['arm'])}: $M$ {thousands(mine[quantity])} against {thousands(theirs[quantity])}; "
                f'difference {m_text(result)}.')
    symbol, text_of = {'whole_curve_mean': ('$W$', w_text), 'final_level': ('$L$', l_text)}[quantity]
    return f"At {cell_text(found['arm'])}: {symbol} {mine[quantity]:.4f} against {theirs[quantity]:.4f}; difference {text_of(result)}."


def paired_check(title, case, other, quantity, verdict, **cell):
    """A part: a group's arm against another side at one configuration, the difference in one quantity judged by
    `verdict` (it receives the difference: its mean, interval and side)."""
    def check(screen):
        """Read the pair of the same-configuration group."""
        found, waiting = pair(screen, title, case, other, **cell)
        if found is None:
            return None, waiting
        return verdict(found['result'][quantity]), pair_words(found, quantity)
    return check
