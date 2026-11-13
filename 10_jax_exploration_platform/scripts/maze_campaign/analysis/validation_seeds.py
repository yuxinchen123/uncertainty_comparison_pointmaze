"""Picked configurations on the validation seeds, case by case, on two blocks of seeds.

A case is one environment with one reward setting. An arm is one method under one name in words; its units carry
one method name on the first block and may carry another on the second (a unit name has a length limit). One arm is
the control.

- **First block** (the phase `first_block`): each unit trains one picked configuration on seeds that are neither
  the development seeds nor the confirmation seeds. Each side stands at its best configuration among those run
  there: a second selection step (text and tables of case_sweeps.validation_text).
- **Second block** (the phase `second_block`, optional): seeds on which nothing was chosen. Per metric an arm
  stands at the configuration that was its best on the first block, when that configuration was run on the second
  block; when it was not, or when the arm has no complete unit on the first block, the arm stands at its best on
  the second block and its row says so. The arm is compared twice with the control, paired on the seeds both have:
  1. equal rules: the control at the configuration that was its best on the first block, so both sides were fixed
     before the second block was read;
  2. favouring the control: the control at its best configuration on the second block.

The block `validation_seeds` of a batch's summary_settings.json (and the settings of the standing page):

- `first_block`: the phase of the first block's units, or a list of phases whose units train on the same seeds
  (for example the units picked for the whole-curve mean and those picked for the first-goal time, so that the
  control stands at its best for each metric among both);
- `second_block` (optional): the phase or phases of the second block's units;
- `arms`: a list, each with `name` (in words) and at least one of `first_block` and `second_block` (the method as
  the unit specifications of that phase spell it); exactly one arm has `"control": true`; `family` (optional) is
  read by the standing page only;
- `cases`: a list of `environment` and `reward_setting` as the unit specifications spell them.
"""

from . import case_sweeps, compare, measures, records, report

BLOCK_KEYS = {'first_block', 'second_block', 'arms', 'cases'}
ARM_KEYS = {'name', 'first_block', 'second_block', 'control', 'family'}
METRICS = case_sweeps.METRICS


# ---- reading ----------------------------------------------------------------------------------------------------

def check_block(block, where='validation_seeds'):
    """Refuse a settings block with unknown or missing keys, an arm without a phase, or not exactly one control."""
    case_sweeps.check_keys(block, BLOCK_KEYS, BLOCK_KEYS - {'second_block'}, where)
    for arm in block['arms']:
        case_sweeps.check_keys(arm, ARM_KEYS, {'name'}, f"{where}, arm {arm.get('name')}")
        if 'first_block' not in arm and 'second_block' not in arm:
            raise ValueError(f"summary_settings.json, {where}: the arm {arm['name']} names no method on either block.")
    controls = [arm['name'] for arm in block['arms'] if arm.get('control')]
    if len(controls) != 1:
        raise ValueError(f'summary_settings.json, {where}: exactly one arm is the control; found {controls}.')
    for case in block['cases']:
        case_sweeps.check_keys(case, {'environment', 'reward_setting'}, {'environment', 'reward_setting'}, f'{where}, cases')


def control_arm(block):
    """The arm that is the control."""
    return next(arm for arm in block['arms'] if arm.get('control'))


def same_configuration(groups, wanted):
    """The configuration among `groups` that has the settings of `wanted`; None when there is none."""
    # before: groups at (learning rate 0.006, weight 2) and (0.006, weight 4), wanted (0.006, weight 4) of another
    #         block; after: the second group. The seeds differ between blocks; the settings do not.
    return next((group for group in groups if group.settings == wanted.settings), None)


def compared(mine, theirs, budget, *name):
    """The paired differences of one configuration from another in W, L and M, on the seeds both have."""
    return {**compare.compare_with_baseline(mine.policies, theirs.policies, budget, *name),
            **compare.compare_discovery(mine.policies, theirs.policies, budget, *name)}


def read_sides(run_dir, specifications, phase, arms, key, environment, reward_setting):
    """Every arm's side on one block in one case, by the method name the arm has on that block."""
    return {arm[key]: case_sweeps.read_side(
        run_dir, case_sweeps.select(specifications, case_sweeps.phase_names(phase), arm[key], environment, reward_setting),
        arm[key], arm['name'])
        for arm in arms if key in arm}


def read_first(sides, control, phase, environment, reward_setting):
    """The first block as read: the sides, the control's method, their budget, and each arm at its best against the
    control at its own."""
    if not any(side.units for side in sides.values()):
        return {'sides': sides, 'control': control, 'budget': None, 'comparisons': {}}
    budget = case_sweeps.one_budget(sides.values(), f'the first validation block in {environment}, {reward_setting}')
    reference = sides[control]
    # The same words as case_sweeps.validation_text names its comparisons by, so both give the same intervals.
    name = case_sweeps.validation_name(phase, environment, reward_setting)
    return {'sides': sides, 'control': control, 'budget': budget,
            'comparisons': {slug: case_sweeps.compare_at_best(side, reference, budget, *name)
                            for slug, side in sides.items() if slug != control}}


def read_second(first, sides, control, arms, phase, environment, reward_setting):
    """The second block as read: per metric the control twice, and each arm at the configuration fixed on the first block.

    Per metric: `wanted` is the control's best configuration on the first block (None when it has none there),
    `fixed` the same configuration on the second block (None when it was not run there or is not complete), `best`
    the control's best on the second block, `between` and `reverse` the differences between those two when they
    are not the same configuration, and `rows` one entry per arm with a complete unit on the second block:
    its configuration here (`group`), how it was fixed (`how`: 'first' when it is the arm's first-block best,
    'absent' when that configuration was not run here, 'none' when the arm has no complete unit on the first
    block), the first-block best (`wanted`), its measures, and the two comparisons (`equal`, `favouring`; None
    when the control has no configuration to compare with). Where the control's two configurations are the same
    one, the two comparisons are one result.
    """
    if not any(side.units for side in sides.values()):
        return None
    budget = case_sweeps.one_budget(sides.values(), f'the second validation block in {environment}, {reward_setting}')
    if first['budget'] not in (None, budget):
        raise ValueError(f"The two validation blocks of {environment}, {reward_setting} train {first['budget']} and "
                         f'{budget} steps per policy; one length is needed.')
    reference, earlier = sides[control], first['sides'].get(first['control'])
    if not reference.groups:
        # Without a complete control unit here nothing is compared; the sides are still listed.
        return {'sides': sides, 'budget': budget, 'control': reference, 'earlier': earlier,
                'metrics': {metric: {'wanted': None, 'fixed': None, 'best': None, 'rows': {}, 'between': None, 'reverse': None}
                            for metric in METRICS}}
    name = ('second validation block', case_sweeps.phase_key(phase), environment, reward_setting)
    found = {}
    for metric in METRICS:
        # The control twice: at the configuration that was its best on the first block, and at its best here.
        wanted = (case_sweeps.best_configuration(earlier.groups, first['budget'], metric)
                  if earlier is not None and earlier.groups else None)
        fixed = same_configuration(reference.groups, wanted) if wanted is not None else None
        best = case_sweeps.best_configuration(reference.groups, budget, metric) if reference.groups else None
        rows = {}
        for arm in arms:
            if arm.get('control') or 'second_block' not in arm or not sides[arm['second_block']].groups:
                continue
            side, before = sides[arm['second_block']], first['sides'].get(arm.get('first_block'))
            # The arm's own best on the first block, looked up among its configurations here.
            chosen = (case_sweeps.best_configuration(before.groups, first['budget'], metric)
                      if before is not None and before.groups else None)
            group = same_configuration(side.groups, chosen) if chosen is not None else None
            how = 'first' if group is not None else 'absent' if chosen is not None else 'none'
            if group is None:
                group = case_sweeps.best_configuration(side.groups, budget, metric)
            equal = compared(group, fixed, budget, *name, side.slug, metric, 'equal rules') if fixed is not None else None
            rows[arm['second_block']] = {
                'side': side, 'group': group, 'how': how, 'wanted': chosen,
                'measures': measures.configuration_measures(group.policies, budget), 'equal': equal,
                'favouring': equal if best is fixed else compared(group, best, budget, *name, side.slug, metric, 'favouring the control')}
        # The control's two configurations against each other, when they differ: what the choice on this block gives it.
        differ = fixed is not None and best is not None and best is not fixed
        between = compared(best, fixed, budget, *name, 'control', metric, 'its best here against its first-block best') if differ else None
        reverse = compared(fixed, best, budget, *name, 'control', metric, 'its first-block best against its best here') if differ else None
        found[metric] = {'wanted': wanted, 'fixed': fixed, 'best': best, 'rows': rows, 'between': between, 'reverse': reverse}
    return {'sides': sides, 'budget': budget, 'control': reference, 'earlier': earlier, 'metrics': found}


def read_case(run_dir, block, environment, reward_setting, specifications):
    """One case on both blocks: `first` (read_first) and `second` (read_second; None without units of that phase)."""
    arms, control = block['arms'], control_arm(block)
    if 'first_block' in control:
        first = read_first(read_sides(run_dir, specifications, block['first_block'], arms, 'first_block', environment, reward_setting),
                           control['first_block'], block['first_block'], environment, reward_setting)
    else:
        first = {'sides': {}, 'control': None, 'budget': None, 'comparisons': {}}
    second = None
    if 'second_block' in block and 'second_block' in control:
        second = read_second(first, read_sides(run_dir, specifications, block['second_block'], arms, 'second_block',
                                               environment, reward_setting),
                             control['second_block'], arms, block['second_block'], environment, reward_setting)
    return {'words': case_sweeps.case_words(environment, reward_setting), 'first': first, 'second': second}


def read_cases(run_dir, block, where='validation_seeds'):
    """Every case of a `validation_seeds` block, in its order: {(environment, reward setting): read_case}."""
    check_block(block, where)
    specifications = case_sweeps.read_specifications(run_dir)
    return {(case['environment'], case['reward_setting']):
            read_case(run_dir, block, case['environment'], case['reward_setting'], specifications)
            for case in block['cases']}


# ---- the second block's tables ----------------------------------------------------------------------------------

def described(read):
    """The settings that name a configuration of the second block: those that differ among all it holds or points to."""
    # The configurations read here, and the first-block bests the rows name (some were not run here).
    groups = [group for side in read['sides'].values() for group in side.groups]
    groups += [found['wanted'] for found in read['metrics'].values() if found['wanted'] is not None]
    groups += [row['wanted'] for found in read['metrics'].values() for row in found['rows'].values() if row['wanted'] is not None]
    return [name for name in records.varying_settings(groups) if all(name in dict(group.settings) for group in groups)]


def fixed_words(row, names):
    """An arm's configuration on the second block with how it was fixed, as a table cell."""
    words = case_sweeps.configuration_words(row['group'], names)
    if row['how'] == 'first':
        return f'{words} (its best on the first block)'
    if row['how'] == 'absent':
        return (f"{words} (its best on this block: its first-block best, "
                f"{case_sweeps.configuration_words(row['wanted'], names)}, was not run here)")
    return f'{words} (its best on this block: it has no complete unit on the first block)'


def no_equal_words(read, found, names):
    """Why an arm has no comparison under equal rules, in words."""
    if found['wanted'] is None:
        return 'the control has no complete unit on the first block'
    return (f"the control's first-block best ({case_sweeps.configuration_words(found['wanted'], names)}) has no "
            'complete unit on this block')


def table_rows(read, metric, names):
    """The rows of one metric's table: the control once or twice, then the arms from the best value on.

    Each row: its name, its configuration cell, whether it is a method (not the control), its measures, and per
    comparison either the word 'control' (the row is that comparison's control), a result (compared) or the words
    for why there is none.
    """
    found, control, budget = read['metrics'][metric], read['control'], read['budget']
    missing = no_equal_words(read, found, names)
    rows = []
    measured = lambda group: measures.configuration_measures(group.policies, budget)
    words = lambda group: case_sweeps.configuration_words(group, names)
    count = len(control.groups)
    if found['fixed'] is not None and found['fixed'] is found['best']:
        rows.append({'name': control.name, 'configuration': f"{words(found['best'])} (its best on the first block and on this block)",
                     'method': False, 'measures': measured(found['best']), 'equal': 'control', 'favouring': 'control'})
    else:
        if found['fixed'] is not None:
            rows.append({'name': f'{control.name}, at its first-block best',
                         'configuration': f"{words(found['fixed'])} (its best on the first block)", 'method': False,
                         'measures': measured(found['fixed']), 'equal': 'control', 'favouring': found['reverse']})
        if found['best'] is not None:
            rows.append({'name': f'{control.name}, at its best on this block',
                         'configuration': f"{words(found['best'])} (best of {count} on this block)" if count > 1
                         else f"{words(found['best'])} (the only configuration read on this block)",
                         'method': False, 'measures': measured(found['best']),
                         'equal': found['between'] if found['between'] is not None else missing, 'favouring': 'control'})
    value = ((lambda row: -row['measures']['whole_curve_mean']) if metric == 'whole'
             else (lambda row: row['measures']['capped_first_goal_step']))
    arms = [{'name': row['side'].name, 'configuration': fixed_words(row, names), 'method': True, 'measures': row['measures'],
             'equal': row['equal'] if row['equal'] is not None else missing,
             'favouring': row['favouring'] if row['favouring'] is not None else 'the control has no complete unit on this block'}
            for row in found['rows'].values()]
    return rows + sorted(arms, key=value)


def result_of(cell):
    """A comparison cell's result when it is one with at least two common seeds, else None."""
    return cell if isinstance(cell, dict) and cell['seeds'] >= 2 else None


def cell_text(cell, text_of):
    """A comparison cell: the result as text, 'control', or the words for why there is no difference."""
    if isinstance(cell, str):
        return cell
    return text_of(cell) if cell['seeds'] >= 2 else 'fewer than two seeds in common'


def common_seeds(row):
    """The seeds a row is compared on: of its first comparison that has a result, else its own seeds."""
    for key in ('equal', 'favouring'):
        if isinstance(row[key], dict):
            return str(row[key]['seeds'])
    return str(row['measures']['seeds'])


def second_table(read, metric, names):
    """One metric's table of the second block: every row with both comparisons."""
    rows = table_rows(read, metric, names)
    digits = case_sweeps.decimals(max(row['measures']['seeds'] for row in rows))
    number, signed = (lambda value: f'{value:.{digits}f}'), (lambda value: format(value, f'+.{digits}f'))
    lead = [[row['name'], row['configuration']] for row in rows]
    tail = [[common_seeds(row)] for row in rows]
    # The differences that are ranked for bold and underline: those of the methods, not the control's own rows.
    results = {key: [result_of(row[key]) if row['method'] else None for row in rows] for key in ('equal', 'favouring')}
    if metric == 'whole':
        whole = case_sweeps.marks([row['measures']['whole_curve_mean'] for row in rows], number, True)
        final = case_sweeps.marks([row['measures']['final_level'] for row in rows], number, True)
        gaps = {(key, quantity): case_sweeps.marks([found[quantity]['mean'] if found else None for found in results[key]], signed, True)
                for key in results for quantity in ('whole_curve_mean', 'final_level')}
        difference = lambda row, index, key, quantity: cell_text(
            row[key], lambda found: case_sweeps.fine_difference(found[quantity], digits, gaps[(key, quantity)][index]))
        body = [[case_sweeps.fine_mean(row['measures']['whole_curve_mean'], row['measures']['whole_curve_mean_error'], digits, whole[index]),
                 difference(row, index, 'equal', 'whole_curve_mean'), difference(row, index, 'favouring', 'whole_curve_mean'),
                 case_sweeps.fine_mean(row['measures']['final_level'], row['measures']['final_level_error'], digits, final[index]),
                 difference(row, index, 'equal', 'final_level')] for index, row in enumerate(rows)]
        header = ['method', 'configuration and how it was fixed', '$W$ ↑ ± standard error',
                  'difference in $W$, equal rules [95% interval]', 'difference in $W$, favouring the control [95% interval]',
                  '$L$ ± standard error', 'difference in $L$, equal rules [95% interval]', 'seeds in common']
    else:
        capped = case_sweeps.marks([row['measures']['capped_first_goal_step'] for row in rows], report.thousands, False)
        gaps = {key: case_sweeps.marks([found['capped_first_goal_step']['mean'] if found else None for found in results[key]],
                                       lambda value: report.thousands(value, True), False) for key in results}
        reached = [round(row['measures']['reached_goal_share'] * row['measures']['seeds']) for row in rows]
        reached_marks = report.column_marks([row['measures']['reached_goal_share'] for row in rows],
                                            [f"{count} of {row['measures']['seeds']}" for count, row in zip(reached, rows)], True)
        difference = lambda row, index, key: cell_text(
            row[key], lambda found: report.discovery_interval_text(found['capped_first_goal_step'], gaps[key][index]))
        body = [[report.thousands_with_error(row['measures']['capped_first_goal_step'],
                                             row['measures']['capped_first_goal_step_error'], capped[index]),
                 difference(row, index, 'equal'), difference(row, index, 'favouring'),
                 report.styled(f"{reached[index]} of {row['measures']['seeds']}", reached_marks[index])]
                for index, row in enumerate(rows)]
        header = ['method', 'configuration and how it was fixed', '$M$ ↓ ± standard error, thousand steps',
                  'difference in $M$, equal rules [95% interval], thousand steps',
                  'difference in $M$, favouring the control [95% interval], thousand steps', 'policies that reach the goal',
                  'seeds in common']
    return report.markdown_table(header, [first + middle + last for first, middle, last in zip(lead, body, tail)])


def second_text(read, phase, environment, reward_setting, heading):
    """The second block of one case: every configuration, then per metric the table with both comparisons."""
    level = heading.split(' ')[0] + '#'
    sides = read['sides']
    units = f'{level} Units\n\n' + case_sweeps.units_table([(side, case_sweeps.case_words(environment, reward_setting))
                                                            for side in sides.values()])
    listed = [read['control']] + [side for side in sides.values() if side is not read['control']]
    complete = [side for side in listed if side.groups]
    if not complete:
        return heading + f'No unit of {case_sweeps.phase_words(phase)} is complete yet.\n\n' + units
    names = described(read)
    seeds = sorted({policy.seed for side in complete for group in side.groups for policy in group.policies})
    waiting = [f'{side.name}: {case_sweeps.read_words(side)}' for side in sides.values() if len(side.groups) < side.planned]
    text = [heading
            + f'The units of {case_sweeps.phase_words(phase)} in {case_sweeps.case_words(environment, reward_setting)}. Each trains one '
            f'configuration on {len(seeds)} seeds, from {seeds[0]} to {seeds[-1]}. Nothing was chosen on these seeds '
            f"before they were read. Read at {read['budget']:,} environment steps per policy. Define $W$, $L$ and $M$ as "
            'in the section Measures. First table: every configuration run on this block, the control first; per '
            'column the best value is bold and the second best underlined (the smallest for $M$).\n\n'
            + case_sweeps.unit_values_table(complete, names, read['budget'], 'whole')
            + 'Then one table per metric. A method stands at the configuration that was its best for that metric on '
            'the first block; a row says so when that configuration was not run here, and the method then stands '
            'at its best on this block, which favours it. Each method is compared twice with the control, paired '
            'on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:\n\n'
            '1. **Equal rules:** against the control at the configuration that was its best on the first block. '
            'Both sides were fixed before this block was read.\n'
            '2. **Favouring the control:** against the control at its best configuration on this block, chosen '
            'among those run here after the block was read.\n\n'
            'The control\'s rows come first, then the methods from the best value on.\n\n']
    if read['control'].groups:
        for metric in METRICS:
            symbol, best = case_sweeps.METRIC_WORDS[metric]
            text.append(f'Second block, {symbol} ({best} is the best):\n\n' + second_table(read, metric, names))
    else:
        text.append('The control has no complete unit on this block, so nothing is compared yet.\n\n')
    text.append((('Not complete, left out: ' + '; '.join(waiting) + '.\n\n') if waiting else '') + units)
    return ''.join(text)


# ---- the section of a batch summary -------------------------------------------------------------------------------

def case_text(block, environment, reward_setting, case, level='###'):
    """One case: its heading, the first block (every configuration, best against best), then the second block."""
    control = control_arm(block)
    text = [f"{level} {case['words']}\n\n"]
    first = None
    if case['first']['sides']:
        first = case_sweeps.validation_text(
            block['first_block'], control['first_block'], case['first']['sides'], environment, reward_setting, METRICS,
            f'{level}# First block: each side at its best configuration there\n\n',
            'the configurations run here were picked from each side\'s sweeps and screens')
    text.append(first or f"{case_sweeps.phase_words(block['first_block']).capitalize()} has no unit of these methods in this case.\n\n")
    if case['second'] is not None:
        text.append(second_text(case['second'], block['second_block'], environment, reward_setting,
                                f'{level}# Second block: seeds that chose nothing\n\n'))
    return ''.join(text)


def section(block, cases):
    """The section of a batch summary: the validation seeds case by case, on the first block and on the second."""
    text = ['## Validation seeds, case by case\n\n'
            'A case is one environment with one reward setting. The validation seeds are neither the development '
            'seeds nor the confirmation seeds. They come in two blocks. On the first block each side stands at its '
            'best configuration among the few run there, so that block is a second selection step. '
            + ('On the second block nothing was chosen: a method stands there at the configuration fixed on the '
               'first block, and is compared with the control under equal rules and with the control favoured. '
               if 'second_block' in block else '')
            + 'Only complete units are read; nothing is estimated for the others.\n\n']
    for (environment, reward_setting), case in cases.items():
        text.append(case_text(block, environment, reward_setting, case))
    return ''.join(text)


# ---- what the prediction modules and the standing page read ---------------------------------------------------------

def unfinished(sides, *slugs):
    """Which of the named methods are not read in full on one block, in words; empty when every planned
    configuration of each is read."""
    # before: a side with 1 of its 2 planned configurations read, and a method with no unit;
    # after:  ['<its name>: 1 of 2 configurations read (1 of 2 units complete)', '<method> (no unit)'].
    waiting = []
    for slug in slugs:
        side = sides.get(slug)
        if side is None or not side.units:
            waiting.append(f"{side.name if side is not None else slug.replace('-', ' ')} (no unit)")
        elif len(side.groups) < side.planned:
            waiting.append(f'{side.name}: {case_sweeps.read_words(side)}')
    return waiting


def first_entries(cases, case, *slugs, metric='whole'):
    """The named arms at their best on the first block against the control at its own (compare_at_best's entries).

    Returns (entries by method, None) once every planned configuration of each arm and of the control is read
    there, so a best configuration can no longer change; else (None, the words that name what is waited for).
    """
    first = cases[case]['first']
    waiting = unfinished(first['sides'], first['control'], *slugs)
    if waiting:
        return None, f"Waiting for complete units in {cases[case]['words']}, first block: {'; '.join(waiting)}."
    return {slug: first['comparisons'][slug][metric] for slug in slugs}, None


def second_rows(cases, case, *slugs, metric='whole'):
    """The named arms' rows on the second block (read_second), each with its comparison under equal rules.

    Returns (rows by method, None) once every planned configuration of each arm and of the control is read there
    and the control's first-block best is among them; else (None, the words that name what is waited for).
    """
    second = cases[case]['second']
    if second is None:
        return None, f"Waiting for units of the second block in {cases[case]['words']}."
    waiting = unfinished(second['sides'], second['control'].slug, *slugs)
    if waiting:
        return None, f"Waiting for complete units in {cases[case]['words']}, second block: {'; '.join(waiting)}."
    rows = {slug: second['metrics'][metric]['rows'][slug] for slug in slugs}
    if any(row['equal'] is None for row in rows.values()):
        return None, (f"No comparison under equal rules in {cases[case]['words']}: the control's first-block best has no "
                      'complete unit on the second block.')
    return rows, None
