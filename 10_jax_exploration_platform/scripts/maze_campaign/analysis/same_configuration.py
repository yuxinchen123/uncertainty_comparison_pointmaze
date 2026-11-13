"""Arms compared at the same configuration on the same seeds, case by case.

The validation seeds (validation_seeds.py) compare each arm at its best configuration with the control at its own.
Some questions need two sides at one configuration instead: an arm with one part added against the arm without it,
or an arm against RND trained at the same settings. A pair is read at every configuration of the arm that the other
side also trained, matched on the settings of the cells (intrinsic weight, learning rate, entropy coefficient, clip
coefficient; the bonus options of the two sides differ on purpose), and paired on the seeds both have, with the 95%
interval from 10,000 resamples of those seeds.

The block `same_configuration` of a batch's summary_settings.json is a list of groups, each with:

- `title`: the group's heading, in words;
- `arm`: the side read at its configurations: `name` (in words), `method` (as the unit specifications spell it),
  `phases` (shell patterns of the phases whose units hold it) and, optionally, `run_folder` (a run folder beside
  this one whose units hold it; by default this one);
- `against`: a list of sides of the same form, each compared with `arm` at the configurations both have;
- `cases`: a list of `environment` and `reward_setting` as the unit specifications spell them.

Only complete units are read; a unit that is not complete is named and left out.
"""

from pathlib import Path

from . import case_sweeps, measures, report, validation_seeds

GROUP_KEYS = {'title', 'arm', 'against', 'cases'}
SIDE_KEYS = {'name', 'method', 'phases', 'run_folder'}
# The settings of a unit's cells: two configurations of different arms are the same configuration when these agree.
MATCHED = ('int_coef', 'learning_rate', 'ent_coef', 'clip_coef')
# The cell settings a row is named by (the clip coefficient is 0.1 in every unit of the study).
NAMED = ('int_coef', 'learning_rate', 'ent_coef')
SIDE_WORDS = {'whole': {'ahead': 'ahead', 'level': 'level', 'behind': 'behind'},
              'discovery': {'ahead': 'sooner', 'level': 'level', 'behind': 'later'}}


# ---- reading ----------------------------------------------------------------------------------------------------

def cell_key(group):
    """A configuration's cell settings, by which configurations of two sides are matched."""
    # before: settings (('clip_coef', 0.1), ('ent_coef', 0.01), ('fast_last_layer_step', 0.1), ('int_coef', 4.0),
    #         ('learning_rate', 0.006), ...); after: (4.0, 0.006, 0.01, 0.1).
    values = dict(group.settings)
    return tuple(values.get(name) for name in MATCHED)


def read_one_side(run_dir, side, environment, reward_setting, specifications):
    """One side in one case, from the run folder its block names; `specifications` caches each folder's files."""
    folder = Path(run_dir).parent / side['run_folder'] if 'run_folder' in side else Path(run_dir)
    if folder not in specifications:
        specifications[folder] = case_sweeps.read_specifications(folder)
    chosen = case_sweeps.select(specifications[folder], side['phases'], side['method'], environment, reward_setting)
    return case_sweeps.read_side(folder, chosen, side['method'], side['name'])


def pairs(arm, other, budget, name):
    """The arm's configurations that the other side also trained, each with the paired differences; and those it did not."""
    # before: the arm at weights 2 and 4, the other side at weights 4 and 8;
    # after:  one pair at weight 4, and the arm's configuration at weight 2 among the missing.
    theirs = {cell_key(group): group for group in other.groups}
    found, missing = [], []
    for group in sorted(arm.groups, key=cell_key):
        partner = theirs.get(cell_key(group))
        if partner is None:
            missing.append(group)
            continue
        found.append({'arm': group, 'other': partner,
                      'arm_measures': measures.configuration_measures(group.policies, budget),
                      'other_measures': measures.configuration_measures(partner.policies, budget),
                      'result': validation_seeds.compared(group, partner, budget, *name, other.slug, cell_key(group))})
    return found, missing


def read_group(run_dir, group, specifications):
    """One group case by case: the arm, the sides it is compared with, the budget, and the pairs per side."""
    case_sweeps.check_keys(group, GROUP_KEYS, GROUP_KEYS, 'same_configuration')
    for side in [group['arm']] + group['against']:
        case_sweeps.check_keys(side, SIDE_KEYS, SIDE_KEYS - {'run_folder'}, f"same_configuration, {side.get('name')}")
    cases = {}
    for case in group['cases']:
        environment, reward_setting = case['environment'], case['reward_setting']
        arm = read_one_side(run_dir, group['arm'], environment, reward_setting, specifications)
        others = [read_one_side(run_dir, side, environment, reward_setting, specifications) for side in group['against']]
        read = {'arm': arm, 'others': others, 'budget': None, 'pairs': {}}
        if arm.groups:
            read['budget'] = case_sweeps.one_budget([arm] + others, f"{group['title']}, {environment}, {reward_setting}")
            name = ('same configuration', arm.slug, environment, reward_setting)
            read['pairs'] = {other.slug: pairs(arm, other, read['budget'], name) for other in others if other.groups}
        cases[(environment, reward_setting)] = read
    return {'title': group['title'], 'cases': cases}


def read(run_dir, block):
    """Every group of a `same_configuration` block, in its order."""
    specifications = {}
    return [read_group(run_dir, group, specifications) for group in block]


# ---- what the prediction modules read -----------------------------------------------------------------------------

def find_group(groups, title):
    """The group with this title."""
    return next(group for group in groups if group['title'] == title)


def pair_at(groups, title, case, other, **cell):
    """The pair of one group in one case against the side `other` (its method) at the named cell settings.

    Returns (the pair, None), or (None, the words for why it cannot be read yet).
    """
    # before: cell {'learning_rate': 0.006, 'int_coef': 4.0}; after: the pair whose arm configuration has those values.
    found = find_group(groups, title)['cases'][case]
    if other not in found['pairs']:
        return None, f'Waiting for complete units of {other} in {case_sweeps.case_words(*case)}.'
    for pair in found['pairs'][other][0]:
        if all(pair['arm'].setting(name) == value for name, value in cell.items()):
            return pair, None
    return None, f'Waiting for the configuration {cell} of {title} and of {other} in {case_sweeps.case_words(*case)}.'


# ---- the section of a batch summary -------------------------------------------------------------------------------

def counts_words(found, metric):
    """How many pairs stand on each side for one metric, in words."""
    quantity = 'whole_curve_mean' if metric == 'whole' else 'capped_first_goal_step'
    sides = [pair['result'][quantity].get('side') for pair in found if pair['result']['seeds'] >= 2]
    words = SIDE_WORDS[metric]
    return ', '.join(f'{words[side]} at {sides.count(side)}' for side in ('ahead', 'level', 'behind')) + f' of {len(sides)}'


def whole_table(read, names, digits):
    """The W and L of every pair in one case: the arm's value, the other's, and the paired differences."""
    rows = []
    for other in read['others']:
        for pair in read['pairs'].get(other.slug, ([], []))[0]:
            mine, theirs, result = pair['arm_measures'], pair['other_measures'], pair['result']
            rows.append([other.name, case_sweeps.configuration_words(pair['arm'], names),
                         case_sweeps.fine_mean(mine['whole_curve_mean'], mine['whole_curve_mean_error'], digits),
                         case_sweeps.fine_mean(theirs['whole_curve_mean'], theirs['whole_curve_mean_error'], digits),
                         validation_seeds.cell_text(result, lambda found: case_sweeps.fine_difference(found['whole_curve_mean'], digits)),
                         validation_seeds.cell_text(result, lambda found: case_sweeps.fine_difference(found['final_level'], digits)),
                         str(result['seeds'])])
    return report.markdown_table(['compared with', 'configuration', '$W$ of the arm', '$W$ of the other side',
                                  'difference in $W$ [95% interval]', 'difference in $L$ [95% interval]',
                                  'seeds in common'], rows)


def discovery_table(read, names):
    """The M of every pair in one case: the arm's value, the other's, the paired difference and the arrivals."""
    rows = []
    for other in read['others']:
        for pair in read['pairs'].get(other.slug, ([], []))[0]:
            mine, theirs, result = pair['arm_measures'], pair['other_measures'], pair['result']
            reached = lambda row: f"{round(row['reached_goal_share'] * row['seeds'])} of {row['seeds']}"
            rows.append([other.name, case_sweeps.configuration_words(pair['arm'], names),
                         report.thousands_with_error(mine['capped_first_goal_step'], mine['capped_first_goal_step_error']),
                         report.thousands_with_error(theirs['capped_first_goal_step'], theirs['capped_first_goal_step_error']),
                         validation_seeds.cell_text(result, lambda found: report.discovery_interval_text(found['capped_first_goal_step'])),
                         f'{reached(mine)}; {reached(theirs)}', str(result['seeds'])])
    return report.markdown_table(['compared with', 'configuration', '$M$ of the arm, thousand steps',
                                  '$M$ of the other side, thousand steps',
                                  'difference in $M$ [95% interval], thousand steps',
                                  'policies that reach the goal: the arm; the other side', 'seeds in common'], rows)


def best_words(side, budget, names):
    """A side's best configuration for W and for M among all it trained in the case, in words."""
    whole = case_sweeps.best_configuration(side.groups, budget, 'whole')
    discovery = case_sweeps.best_configuration(side.groups, budget, 'discovery')
    of = f'of {len(side.groups)} configuration{"s" if len(side.groups) > 1 else ""}'
    found = lambda group: measures.configuration_measures(group.policies, budget)
    return (f"- {side.name}: best $W$ {found(whole)['whole_curve_mean']:.4f} at "
            f"{case_sweeps.configuration_words(whole, names)}; best $M$ "
            f"{report.thousands(found(discovery)['capped_first_goal_step'])} thousand steps at "
            f"{case_sweeps.configuration_words(discovery, names)} ({of}).\n")


def case_text(group, case, read):
    """One case of a group: the W table, the M table, the counts, each side's best, and what was left out."""
    words = case_sweeps.case_words(*case)
    text = [f'#### {words}\n\n']
    sides = [read['arm']] + read['others']
    waiting = [f'{side.name}: {case_sweeps.read_words(side)}' for side in sides if len(side.groups) < side.planned]
    if not read['arm'].groups or not read['pairs']:
        text.append('Nothing is compared yet: '
                    + ('the arm has no complete unit' if not read['arm'].groups else 'no other side has a complete unit')
                    + ' in this case.\n\n')
    else:
        budget = read['budget']
        groups = [configuration for side in sides for configuration in side.groups]
        names = list(NAMED)
        seeds = sorted({policy.seed for configuration in read['arm'].groups for policy in configuration.policies})
        digits = case_sweeps.decimals(max(len(configuration.policies) for configuration in groups))
        text.append(f"The arm: {read['arm'].name}, on {len(seeds)} seeds from {seeds[0]} to {seeds[-1]}, read at "
                    f'{budget:,} environment steps per policy. Each row is one configuration of the arm and the same '
                    'configuration of the side it is compared with; differences are the arm minus that side.\n\n'
                    f'Whole-curve mean $W$ and final level $L$:\n\n{whole_table(read, names, digits)}'
                    f'Mean capped first-goal time $M$:\n\n{discovery_table(read, names)}'
                    'Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: '
                    'sooner in $M$):\n\n')
        for other in read['others']:
            if other.slug in read['pairs']:
                found, missing = read['pairs'][other.slug]
                text.append(f"- against {other.name}: $W$ {counts_words(found, 'whole')}; $M$ "
                            f"{counts_words(found, 'discovery')}"
                            + (f"; not trained by it: {'; '.join(case_sweeps.configuration_words(configuration, names) for configuration in missing)}"
                               if missing else '') + '.\n')
        text.append('\nEach side at its best among all its configurations read in this case (for $W$ the highest, for '
                    '$M$ the smallest):\n\n' + ''.join(best_words(side, budget, names) for side in sides if side.groups) + '\n')
    text.append(('Not complete, left out: ' + '; '.join(waiting) + '.\n\n') if waiting else '')
    return ''.join(text)


def section(groups):
    """The section of a batch summary: every group, case by case."""
    text = ['## Arms at the same configuration, on the same seeds\n\n'
            'Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic '
            'weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with '
            'the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section '
            'Measures. No configuration is chosen here.\n\n']
    for group in groups:
        text.append(f"### {group['title']}\n\n")
        text += [case_text(group, case, found) for case, found in group['cases'].items()]
    return ''.join(text)
