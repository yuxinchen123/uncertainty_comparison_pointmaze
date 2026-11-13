"""A method's sweep against the baseline's sweep in one case, each side at its own best configuration per metric.

A case is one environment with one reward setting. A side is one method's configurations in a case: every
configuration of its complete units of the phases named for it. A configuration and seed that two units trained
is read from the unit that finished first; a unit that is not complete is named and not read.

Per metric each side stands at its own best configuration: for the whole-curve mean W the highest W (ties: the
higher final level L), for the mean capped first-goal time M the smallest M (ties: more policies reach the goal).
Differences are the method minus the baseline on the seeds both have, with the 95% interval from resampling those
seeds (compare.py). A best value at an end of the side's grid is flagged: the edge rule of program.md asks for
more values on that side before anything is confirmed.

Two sections of a batch summary are built here, each from one block of the batch's summary_settings.json:

- `development_sweeps`: the development sweeps of the batch's methods in one or more cases, each against the
  baseline's development sweeps in the same case. Keys: `study_description` (the file name, in the batch folder,
  of the study description: its `phase` selects the units, and every sweep names its `method`, `environments`,
  `reward_settings` and `steps_per_policy`), `method_names` (the name in words of every method), `baseline`, and
  optionally `validation` (`phase`, `control`, `method_names`): units that each train one picked configuration
  on the validation seeds; a case that has units of that phase at the case's budget shows them after its grids,
  every configuration and then each method's best against the control's best for both metrics.
- `discovery`: discovery checks in one case. Keys: `environment` and `reward_setting` as the unit specifications
  spell them; `validation` (optional: `phase`, `control`, `method_names`), units that each train one
  configuration on the validation seeds, compared with the control's units of the same phase; `development`
  (optional: `phases` as shell patterns, `method_names`, `baseline`), sweeps on the development seeds.

A third section, from the block `validation_of_configurations` (`environment`, `reward_setting`, `phase`,
`control`, `method_names`), reads every configuration of one phase on the validation seeds in one case and
compares each method's best with the control's best for both metrics (`validation_section`).

A `baseline` block names another run folder's sweeps: `run_folder` (a run folder beside this one), `method` (the
method as its unit specifications spell it), `name` (in words), `phases` (shell patterns) and, optionally,
`confirmation_phases` (shell patterns of the phases whose units train a chosen configuration on the confirmation
seeds; the baseline's best configuration is then also given on those seeds when a complete unit holds it).
"""

from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path

import numpy as np

from . import compare, contents, measures, records, report

METRICS = ('whole', 'discovery')
# A table of one value per intrinsic weight holds at most this many weights, so that it fits a printed page.
MOST_WEIGHT_COLUMNS = 6
DEVELOPMENT_KEYS = {'study_description', 'method_names', 'baseline', 'validation'}
DISCOVERY_KEYS = {'environment', 'reward_setting', 'validation', 'development'}
BASELINE_KEYS = {'run_folder', 'method', 'name', 'phases', 'confirmation_phases'}


@dataclass(frozen=True, eq=False)
class Side:
    """One method's sweep in one case: its units, and the configurations of the complete ones."""
    slug: str        # the method as the unit specifications spell it
    name: str        # the method in words
    units: tuple     # one row per unit (unit_row)
    groups: tuple    # the configurations of the complete units, a repeated configuration and seed read once
    planned: int     # the configurations the units were built with, complete or not
    budget: int      # environment steps per policy; None when the side has no unit


# ---- reading ----------------------------------------------------------------------------------------------------

def check_keys(block, allowed, required, where):
    """Refuse a settings block with a key it does not know or without a key it needs."""
    unknown, missing = set(block) - allowed, required - set(block)
    if unknown or missing:
        raise ValueError(f'summary_settings.json, {where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}.')


def read_specifications(run_dir):
    """Every unit specification of a run folder, in the order of their file names."""
    return [json.loads(path.read_text()) for path in sorted((Path(run_dir) / 'specifications').glob('*.json'))]


def select(specifications, phases, method, environment, reward_setting, budget=None):
    """The specifications of one method in one case whose phase matches one of the shell patterns."""
    # before: phases ['development-at-the-budget*'], method 'cleanrl-ppo-rnd', 'pointmaze-medium', 'step_cost'
    # after:  the units of 'development-at-the-budget' and of 'development-at-the-budget-lower-intrinsic-weights'
    #         of that method in that case; with a budget, only those that train that many steps per policy.
    return [specification for specification in specifications
            if any(fnmatch.fnmatchcase(specification.get('phase') or '', pattern) for pattern in phases)
            and specification['method'] == method and specification['environment'] == environment
            and specification['reward_setting'] == reward_setting
            and (budget is None or specification['steps_per_policy'] == budget)]


def case_words(environment, reward_setting):
    """A case in words, from the names the unit specifications use."""
    # before: ('pointmaze-medium', 'step_cost'); after: 'PointMaze Medium, step cost'.
    family, map_name = environment.split('-', 1)
    return f'{records.ENVIRONMENT_NAMES[(family, map_name)]}, {records.REWARD_SETTING_NAMES[reward_setting]}'


def unit_row(run_dir, specification, name):
    """One unit as (its row for a units table, its policies when it is complete, else an empty list)."""
    shard = Path(run_dir) / 'data' / f"{specification['unit_id']}.jsonl"
    row = {'unit_id': specification['unit_id'], 'name': name, 'cells': len(specification['cells']),
           'planned_windows': specification.get('windows_expected'), 'state': 'not started (no shard)', 'windows': 0,
           'finished': None, 'started_again': 0, 'continued': 0}
    if not shard.exists():
        return row, []
    summary, policies = records.read_unit(shard, [{'name': name, 'unit_id_pattern': shard.stem}])
    row.update(state=summary['state'], windows=summary['windows'], finished=summary['finished'],
               started_again=summary['started_from_zero_again'], continued=summary['continued_from_a_checkpoint'])
    return row, policies if summary['state'] == 'complete' else []


def read_side(run_dir, specifications, slug, name):
    """One method's sweep in one case, from the specifications of its units."""
    budgets = sorted({specification['steps_per_policy'] for specification in specifications})
    if len(budgets) > 1:
        raise ValueError(f'The units of {slug} in one case train {budgets} steps per policy; one length is needed.')
    # Every unit's row; the policies of the complete units, with the time each unit finished.
    rows, policies, finished = [], [], {}
    for specification in specifications:
        row, members = unit_row(run_dir, specification, name)
        rows.append(row)
        policies += members
        finished[row['unit_id']] = row['finished']
    # A configuration and seed trained by two units is read from the unit that finished first.
    # before: unit A (cells 1 to 18) and unit B (cells 1 to 72) both hold seed 3 of one configuration
    # after:  that policy once, from the unit that finished first; the configurations of both units, each once.
    used, _ = records.split_repeats(policies, finished)
    planned = len({(tuple(specification['cell_setting_names']), tuple(cell))
                   for specification in specifications for cell in specification['cells']})
    return Side(slug=slug, name=name, units=tuple(rows), groups=tuple(records.configurations(used)), planned=planned,
                budget=budgets[0] if budgets else None)


def read_baseline(run_dir, block, environment, reward_setting, budget, specifications=None):
    """The baseline's sweep in one case at one budget, from the run folder its settings block names."""
    folder = Path(run_dir).parent / block['run_folder']
    found = read_specifications(folder) if specifications is None else specifications
    return read_side(folder, select(found, block['phases'], block['method'], environment, reward_setting, budget),
                     block['method'], block['name'])


# ---- choosing and comparing -------------------------------------------------------------------------------------

def best_configuration(groups, budget, metric):
    """A side's best configuration for one metric: the highest W (ties: L), or the smallest M (ties: more arrive)."""
    ranked = (compare.rank_configurations(groups, budget) if metric == 'whole'
              else compare.rank_for_discovery(groups, budget))
    return ranked[0][2]


def number_values(groups, name):
    """The values one setting takes among configurations, in order; empty when it is not a number in all of them."""
    values = {dict(group.settings).get(name) for group in groups}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return []
    return sorted(values)


def grid_ends(groups, best):
    """The settings whose value in the best configuration is the smallest or the largest of the side's grid."""
    # before: a grid of the learning rates [0.003, 0.006, 0.01] and the weights [0.25, 0.5, 1], best at (0.01, 0.25)
    # after:  [('int_coef', 'smallest'), ('learning_rate', 'largest')]. A setting with one value is at no end.
    ends = []
    for name in records.varying_settings(groups):
        values = number_values(groups, name)
        if len(values) > 1 and best.setting(name) in (values[0], values[-1]):
            ends.append((name, 'smallest' if best.setting(name) == values[0] else 'largest'))
    return ends


def ends_words(ends):
    """The ends of a grid a best configuration sits at, as a table cell."""
    # before: [('int_coef', 'smallest'), ('learning_rate', 'largest')]
    # after:  'intrinsic weight at the smallest of its grid; learning rate at the largest'.
    if not ends:
        return 'no'
    words = [f"{records.SETTING_WORDS.get(name, name.replace('_', ' '))} at the {end}" for name, end in ends]
    return words[0] + ' of its grid' + ''.join(f'; {word}' for word in words[1:])


def compare_at_best(side, baseline, budget, *name):
    """One method against the baseline in one case, per metric at each side's own best configuration.

    Returns None while the method has no complete configuration. Else, per metric: the method's best
    configuration (`best`) with its measures and the ends of its grid it sits at, the baseline's best
    configuration (`baseline`, None when the baseline has none) and the paired differences (`result`).
    """
    if not side.groups:
        return None
    found = {}
    for metric in METRICS:
        mine = best_configuration(side.groups, budget, metric)
        entry = {'best': mine, 'measures': measures.configuration_measures(mine.policies, budget),
                 'ends': grid_ends(side.groups, mine), 'baseline': None, 'result': None}
        if baseline.groups:
            theirs = best_configuration(baseline.groups, budget, metric)
            entry['baseline'] = theirs
            entry['result'] = {**compare.compare_with_baseline(mine.policies, theirs.policies, budget, *name, side.slug, metric),
                               **compare.compare_discovery(mine.policies, theirs.policies, budget, *name, side.slug, metric)}
        found[metric] = entry
    return found


def described_settings(sides):
    """The settings whose values differ among the configurations of the sides taken together, the main ones first.

    A configuration is named by these. A setting that some configurations do not record at all (an option their
    code did not have yet) names nothing and is left out.
    """
    # before: a method with one configuration (weight 4, learning rate 0.0001) and a control with the learning
    #         rates 0.0001 and 0.0003 at the weight 1; after: ['int_coef', 'learning_rate'].
    groups = [group for side in sides for group in side.groups]
    return [name for name in records.varying_settings(groups) if all(name in dict(group.settings) for group in groups)]


def configuration_words(group, names):
    """A configuration by the settings that vary, in words; 'its only configuration' when none varies."""
    return records.describe_settings(group, names) if names else 'its only configuration'


def best_words(side, group, names):
    """A side's best configuration in words, with the number of configurations it was the best of."""
    # before: a side with 15 configurations; after: 'intrinsic weight 0.5, learning rate 0.006 (best of 15)'.
    if not names:
        return configuration_words(group, names)
    count = len(side.groups)
    return f"{configuration_words(group, names)} ({f'best of {count}' if count > 1 else 'the only configuration read'})"


def confirmed_words(run_dir, block, environment, reward_setting, budget, best, names, metric):
    """The baseline's best configuration on the confirmation seeds, in words; None when the settings name no such phases."""
    if 'confirmation_phases' not in block or not names:
        return None
    folder = Path(run_dir).parent / block['run_folder']
    wanted = {name: best.setting(name) for name in names}
    # The confirmation units that hold this configuration, by their specifications' cells.
    # before: cell names ['learning_rate', 'ent_coef', 'int_coef'], cells [[0.01, 0.01, 64.0]], wanted
    #         {'int_coef': 64.0, 'learning_rate': 0.01, 'ent_coef': 0.01}; after: the unit is one of them.
    holding, others = [], []
    for specification in select(read_specifications(folder), block['confirmation_phases'], block['method'],
                                environment, reward_setting, budget):
        cells = [dict(zip(specification['cell_setting_names'], cell)) for cell in specification['cells']]
        if any(all(cell.get(name) == value for name, value in wanted.items()) for cell in cells):
            holding.append(specification)
        # The other configurations the confirmation units hold, in the words a configuration is named by.
        others += [', '.join(f"{records.SETTING_WORDS.get(name, name.replace('_', ' '))} {cell[name]:g}" for name in names if name in cell)
                   for cell in cells]
    side = read_side(folder, holding, block['method'], block['name'])
    groups = [group for group in side.groups if all(dict(group.settings).get(name) == value for name, value in wanted.items())]
    if not groups:
        held = '; '.join(dict.fromkeys(others))
        return (f"{block['name']}'s best configuration here has no complete unit on the confirmation seeds "
                f"(phases {', '.join(f'`{phase}`' for phase in block['confirmation_phases'])})"
                + (f': the confirmation units of this case hold {held}.' if held else '.'))
    row = measures.configuration_measures(groups[0].policies, budget)
    value = (f"$W$ {report.with_error(row['whole_curve_mean'], row['whole_curve_mean_error'])}" if metric == 'whole' else
             f"$M$ {report.thousands_with_error(row['capped_first_goal_step'], row['capped_first_goal_step_error'])} thousand steps")
    return f"On {row['seeds']:,} confirmation seeds the same configuration of {block['name']} has {value}."


# ---- tables -----------------------------------------------------------------------------------------------------

def decimals(seeds):
    """How many decimals a table of W and L shows: four once the seeds are many, else three."""
    return 4 if seeds >= 100 else 3


def marks(values, text_of, higher_better):
    """The marks of one column ('best', 'second' or None per row); a row without a value takes none."""
    return report.column_marks(values, [None if value is None else text_of(value) for value in values], higher_better)


def fine_mean(value, error, digits, style=None):
    """A mean with its standard error at a chosen number of decimals; the mean alone with one seed."""
    return report.styled(f'{value:.{digits}f}', style) + ('' if error is None else f' ± {error:.{digits}f}')


def fine_difference(found, digits, style=None):
    """A paired difference with its 95% interval at a chosen number of decimals; the mean bold or underlined."""
    low, high = found['interval']
    return f"{report.styled(format(found['mean'], f'+.{digits}f'), style)} [{low:+.{digits}f}, {high:+.{digits}f}]"


def metric_table(metric, baseline, entries, names, budget, against='baseline', ends=True):
    """One metric's table for one case: the baseline at its best configuration first, then each method at its own.

    `entries` holds (side, its comparison from compare_at_best) for the methods with a complete configuration;
    `against` is the word for the side they are compared with ('baseline', or 'control' inside one run folder);
    `ends` is False where the configurations are a few picked ones and not a grid, so no end of a grid is named.
    """
    # The rows: the baseline (when it has a configuration), then the methods from the best value on.
    value = ((lambda row: -row['measures']['whole_curve_mean']) if metric == 'whole'
             else (lambda row: row['measures']['capped_first_goal_step']))
    rows = [{'side': side, 'best': found[metric]['best'], 'measures': found[metric]['measures'],
             'ends': found[metric]['ends'], 'result': found[metric]['result']} for side, found in entries]
    rows.sort(key=value)
    if baseline.groups:
        best = best_configuration(baseline.groups, budget, metric)
        rows.insert(0, {'side': baseline, 'best': best, 'measures': measures.configuration_measures(best.policies, budget),
                        'ends': grid_ends(baseline.groups, best), 'result': 'baseline'})
    compared = [row['result'] if isinstance(row['result'], dict) and row['result']['seeds'] >= 2 else None for row in rows]
    digits = decimals(max(row['measures']['seeds'] for row in rows))
    number, signed = (lambda value: f'{value:.{digits}f}'), (lambda value: format(value, f'+.{digits}f'))

    def difference(row, found, text):
        """A difference cell: the baseline's word, why there is none, or the difference itself."""
        if row['result'] == 'baseline':
            return against
        if row['result'] is None:
            return f'the {against} has no complete configuration'
        return text if found else 'fewer than two seeds in common'

    # The cells shared by both metrics: the method, its best configuration among those read, and the last two.
    lead = [[row['side'].name, best_words(row['side'], row['best'], names)] for row in rows]
    tail = [([ends_words(row['ends'])] if ends else [])
            + [str(row['measures']['seeds']) if row['result'] == 'baseline' else str(found['seeds']) if found else 'not compared']
            for row, found in zip(rows, compared)]
    end_header = ['best configuration at an end of its grid'] if ends else []
    whole = marks([row['measures']['whole_curve_mean'] for row in rows], number, True)
    final = marks([row['measures']['final_level'] for row in rows], number, True)
    if metric == 'whole':
        gaps = {quantity: marks([found[quantity]['mean'] if found else None for found in compared], signed, True)
                for quantity in ('whole_curve_mean', 'final_level')}
        body = [[fine_mean(row['measures']['whole_curve_mean'], row['measures']['whole_curve_mean_error'], digits, whole[index]),
                 difference(row, found, found and fine_difference(found['whole_curve_mean'], digits, gaps['whole_curve_mean'][index])),
                 fine_mean(row['measures']['final_level'], row['measures']['final_level_error'], digits, final[index]),
                 difference(row, found, found and fine_difference(found['final_level'], digits, gaps['final_level'][index]))]
                for index, (row, found) in enumerate(zip(rows, compared))]
        header = ['method', 'best configuration for $W$', '$W$ ↑ ± standard error',
                  f'difference in $W$ from the {against} [95% interval]', '$L$ ± standard error',
                  f'difference in $L$ from the {against} [95% interval]', *end_header, f'seeds in common with the {against}']
    else:
        capped = marks([row['measures']['capped_first_goal_step'] for row in rows], report.thousands, False)
        gaps = marks([found['capped_first_goal_step']['mean'] if found else None for found in compared],
                     lambda value: report.thousands(value, True), False)
        reached = [round(row['measures']['reached_goal_share'] * row['measures']['seeds']) for row in rows]
        reached_marks = report.column_marks([row['measures']['reached_goal_share'] for row in rows],
                                            [f"{count} of {row['measures']['seeds']}" for count, row in zip(reached, rows)], True)
        body = [[report.thousands_with_error(row['measures']['capped_first_goal_step'],
                                             row['measures']['capped_first_goal_step_error'], capped[index]),
                 difference(row, found, found and report.discovery_interval_text(found['capped_first_goal_step'], gaps[index])),
                 report.styled(f"{reached[index]} of {row['measures']['seeds']}", reached_marks[index]),
                 report.styled(number(row['measures']['whole_curve_mean']), whole[index]),
                 report.styled(number(row['measures']['final_level']), final[index])]
                for index, (row, found) in enumerate(zip(rows, compared))]
        header = ['method', 'best configuration for $M$', '$M$ ↓ ± standard error, thousand steps',
                  f'difference in $M$ from the {against} [95% interval], thousand steps', 'policies that reach the goal',
                  '$W$ at that configuration', '$L$ at that configuration', *end_header, f'seeds in common with the {against}']
    return report.markdown_table(header, [first + middle + last for first, middle, last in zip(lead, body, tail)])


def grid_tables(metric, sides, budget):
    """One value per configuration of every method: rows are a method with its other settings, columns the intrinsic weights."""
    # The value of every configuration, and the rows it belongs to.
    # before: a method's grid of the learning rates [0.003, 0.006] and the weights [0.5, 1]
    # after:  two rows ('the method, learning rate 0.003' and '..., learning rate 0.006') and the columns 0.5 and 1.
    measure = ((lambda group: float(np.mean([measures.whole_curve_mean(policy, budget) for policy in group.policies])))
               if metric == 'whole' else
               (lambda group: float(np.mean([measures.capped_first_goal_step(policy, budget) for policy in group.policies]))))
    text_of = (lambda value: f'{value:.3f}') if metric == 'whole' else report.thousands
    weights = sorted({group.setting('int_coef') for side in sides for group in side.groups})
    rows = []
    for side in sides:
        others = [name for name in described_settings([side]) if name != 'int_coef']
        if 'int_coef' not in dict(side.groups[0].settings):
            raise ValueError(f'The configurations of {side.name} record no intrinsic weight; the grid tables need one.')
        values = {group: measure(group) for group in side.groups}
        # Over the method's whole grid, the best value is bold and the second best underlined.
        found = dict(zip(values, report.column_marks(list(values.values()), [text_of(value) for value in values.values()],
                                                     metric == 'whole')))
        lines = report.grouped(side.groups, lambda group: tuple(group.setting(name) for name in others))
        for key in sorted(lines):
            by_weight = {group.setting('int_coef'): group for group in lines[key]}
            label = side.name + (', ' + records.describe_settings(lines[key][0], others) if others else '')
            rows.append((label, {weight: report.styled(text_of(values[group]), found[group]) for weight, group in by_weight.items()}))
    # At most MOST_WEIGHT_COLUMNS weights per table.
    count = -(-len(weights) // MOST_WEIGHT_COLUMNS)
    size = -(-len(weights) // count)
    return ''.join(report.markdown_table(
        ['method and its other settings'] + [f'weight {weight:g}' for weight in weights[first:first + size]],
        [[label] + [cells.get(weight, '') for weight in weights[first:first + size]] for label, cells in rows])
        for first in range(0, len(weights), size))


def units_table(sides):
    """The units of several sides in one table, and the unit names under it."""
    rows, names = [], []
    for side, case in sides:
        for unit in side.units:
            finished = contents.pacific(unit['finished']) if unit['finished'] else (
                'time not recorded' if unit['state'] == 'complete' else 'not finished')
            written = f"{unit['windows']} of {unit['planned_windows']}" if unit['planned_windows'] else str(unit['windows'])
            again = ', '.join(f'{count} {words}' for count, words in ((unit['started_again'], 'started again from zero'),
                                                                      (unit['continued'], 'continued from a checkpoint')) if count)
            rows.append([str(len(rows) + 1), side.name, case, str(unit['cells']), unit['state'], written, finished, again or 'none'])
            names.append(unit['unit_id'])
    return (report.markdown_table(['unit', 'method', 'case', 'configurations', 'state', 'logging windows written',
                                   'finished', 'restarts'], rows)
            + 'Unit names, by the number in the first column:\n\n'
            + ''.join(f'{index}. `{name}`\n' for index, name in enumerate(names, 1)) + '\n')


def read_words(side):
    """How much of a side was read, in words."""
    complete = sum(unit['state'] == 'complete' for unit in side.units)
    return (f'{len(side.groups)} of {side.planned} configurations read ({complete} of {len(side.units)} units complete)')


def edge_sentences(entries, baseline, budget, metrics=METRICS):
    """Which best configurations sit at an end of their grid, one sentence per metric named; says so when none does."""
    sentences = []
    for metric, symbol in ((metric, {'whole': '$W$', 'discovery': '$M$'}[metric]) for metric in metrics):
        found = [f"{side.name} ({ends_words(comparison[metric]['ends'])})" for side, comparison in entries if comparison[metric]['ends']]
        if baseline.groups:
            ends = grid_ends(baseline.groups, best_configuration(baseline.groups, budget, metric))
            found += [f'{baseline.name} ({ends_words(ends)})'] if ends else []
        if found:
            sentences.append(f"Best configuration for {symbol} at an end of its grid: {'; '.join(found)}.")
    if not sentences:
        return 'No best configuration is at an end of its grid.\n\n'
    return (' '.join(sentences) + ' The edge rule of `program.md` asks for more values on that side before anything is '
            'confirmed.\n\n')


# ---- measures, for a batch that has no screen ---------------------------------------------------------------------

def measures_section():
    """The definitions the tables of this module use, each with its formula, for a summary without a screen."""
    return (
        '## Measures\n\n'
        'Two metrics are reported, each in its own table with its own best configuration per method: the whole-curve '
        'mean $W$ measures discovery and optimization together, and the mean capped first-goal time $M$ measures '
        'discovery alone. The two are never mixed in one choice.\n\n'
        'Define $K$ as the number of policies (seeds) of one configuration, $B$ as the budget in environment steps '
        'per policy (named with every table), $R_i(s)$ as the mean return of the training episodes of policy $i$ '
        'that end in the logging window containing step $s$, and $\\Delta s$ as the number of that window\'s steps '
        'inside the range being averaged. Define $H$ as the episode step limit and $t$ as the step of an episode at '
        'which the goal is reached. With reward 1 on arrival an episode returns 1 if it reaches the goal and 0 '
        'otherwise; with the step cost it returns $-(t-1)/H$ if it reaches the goal and $-1$ otherwise. A window in '
        'which no episode ends takes the value of the window before it; before a policy\'s first finished episode '
        'the value is the return of an episode that never reaches the goal.\n\n'
        '1. **Whole-curve mean:** $W(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\dfrac{1}{B}\\sum_{s \\le B} R_i(s) '
        '\\thinspace \\Delta s$, the area under the training curve divided by the budget.\n'
        '2. **Final level:** $L(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\dfrac{1}{0.1 B}\\sum_{0.9 B \\lt s \\le B} R_i(s) '
        '\\thinspace \\Delta s$, the same mean over the last 10% of the budget.\n'
        '3. **Mean capped first-goal time:** define $T_i$ as the training step of policy $i$\'s first arrival at the '
        'goal ($T_i = \\infty$ if it has none by step $B$). $M(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\min(T_i, B)$. '
        'Smaller is better. The tables give $M$ in thousands of steps.\n'
        '4. **Best configuration:** per method and case, for $W$ the configuration with the highest $W$ (ties: the '
        'higher $L$), for $M$ the one with the smallest $M$ (ties: more policies reach the goal). The two can '
        'differ.\n'
        '5. **Difference from the baseline:** define $d_i$ as the value of seed $i$ under the method at its best '
        'configuration minus its value under the baseline at the baseline\'s own best configuration for the same '
        'metric. The tables give $\\bar d = \\dfrac{1}{K}\\sum_{i=1}^{K} d_i$ over the seeds both have, with a 95% '
        'interval: the 2.5% and 97.5% points of the means of 10,000 resamples of those seeds, drawn with '
        'replacement. For $W$ and $L$ a method is ahead when the interval is entirely above zero; for $M$ it is '
        'ahead when the interval is entirely below zero.\n\n'
        'Standard errors are over the $K$ policies.\n\n')


# ---- development sweeps in several cases --------------------------------------------------------------------------

def study_cases(description):
    """The cases of a study description in its order, each with its budget and its methods."""
    # before: sweeps of two methods in ('pointmaze-medium', 'arrival_plus_one') at 1,048,576 steps and of the same
    #         two in ('pointmaze-large', 'step_cost') at 33,554,432
    # after:  {('pointmaze-medium', 'arrival_plus_one'): {'budget': 1048576, 'methods': [first, second]}, ...}.
    cases = {}
    for sweep in description['sweeps']:
        for environment in sweep.get('environments', description.get('environments', ())):
            for reward_setting in sweep.get('reward_settings', description.get('reward_settings', ())):
                budget = sweep.get('steps_per_policy', description.get('steps_per_policy'))
                case = cases.setdefault((environment, reward_setting), {'budget': budget, 'methods': []})
                if case['budget'] != budget:
                    raise ValueError(f'The sweeps of the case {environment}, {reward_setting} train {case["budget"]} and '
                                     f'{budget} steps per policy; one length per case is needed.')
                case['methods'].append(sweep['method'])
    return cases


def read_development(run_dir, batch, folder, block):
    """Every case of a batch's development sweeps: each method's side, the baseline's side, and their comparisons."""
    check_keys(block, DEVELOPMENT_KEYS, DEVELOPMENT_KEYS - {'validation'}, 'development_sweeps')
    if 'validation' in block:
        check_keys(block['validation'], VALIDATION_KEYS, VALIDATION_KEYS, 'development_sweeps, validation')
    check_keys(block['baseline'], BASELINE_KEYS, BASELINE_KEYS - {'confirmation_phases'}, 'development_sweeps, baseline')
    description = json.loads((Path(folder) / block['study_description']).read_text())
    ours = read_specifications(run_dir)
    theirs = read_specifications(Path(run_dir).parent / block['baseline']['run_folder'])
    cases = {}
    for (environment, reward_setting), case in study_cases(description).items():
        unnamed = sorted(set(case['methods']) - set(block['method_names']))
        if unnamed:
            raise ValueError(f'summary_settings.json, development_sweeps: no name in words for the methods {unnamed}.')
        baseline = read_baseline(run_dir, block['baseline'], environment, reward_setting, case['budget'], theirs)
        sides = {slug: read_side(run_dir, select(ours, [description['phase']], slug, environment, reward_setting, case['budget']),
                                 slug, block['method_names'][slug]) for slug in case['methods']}
        cases[(environment, reward_setting)] = {
            'words': case_words(environment, reward_setting), 'budget': case['budget'], 'baseline': baseline, 'sides': sides,
            'comparisons': {slug: compare_at_best(side, baseline, case['budget'], 'development sweeps', batch,
                                                  environment, reward_setting) for slug, side in sides.items()}}
    return cases


def development_section(run_dir, batch, cases, block):
    """The section of the development sweeps: the units, then per case the two tables, the methods' grids, and the
    picked configurations on the validation seeds where the settings name such a phase and the case has units of it."""
    ours = read_specifications(run_dir) if 'validation' in block else None
    baseline_block = block['baseline']
    phases = ', '.join(f'`{phase}`' for phase in baseline_block['phases'])
    text = ['## Development sweeps against the baseline, case by case\n\n'
            'A case is one environment with one reward setting. In each case every method of this batch was trained '
            f"on a grid of configurations on the development seeds, and so was the baseline, {baseline_block['name']} "
            f"(run folder `{baseline_block['run_folder']}`, phases {phases}, the units that train the same number of "
            'steps). Define $W$, $L$, $M$, the best configuration and the difference from the baseline as in the '
            'section Measures. Per case, two tables, one per metric: the baseline at its own best configuration in the '
            'first row, then each method at its own, from the best value on. A best configuration is named by the '
            'settings that vary in the grids, with the number of configurations it was the best of. Per column the '
            'best value is bold and the second best underlined. Every best configuration was chosen on the seeds '
            'the table is read on: these tables are a screen and not a verdict. Only complete units are read; '
            'nothing is estimated for the others.\n\n'
            '### Units\n\n'
            + units_table([(side, case['words']) for case in cases.values() for side in case['sides'].values()])]
    for (environment, reward_setting), case in cases.items():
        baseline, budget = case['baseline'], case['budget']
        entries = [(case['sides'][slug], found) for slug, found in case['comparisons'].items() if found is not None]
        waiting = [side.name for slug, side in case['sides'].items() if case['comparisons'][slug] is None]
        text.append(f"### {case['words']}\n\n"
                    f'Read at {budget:,} environment steps per policy. {baseline.name}: {read_words(baseline)}. '
                    + ' '.join(f'{side.name}: {read_words(side)}.' for side in case['sides'].values())
                    + (f" Not read yet, no complete unit: {'; '.join(waiting)}." if waiting else '') + '\n\n')
        if 'validation' in block:
            # Appended after the case's own tables below; read here so that a case with no complete sweep shows it too.
            checked = validation_part(
                run_dir, batch, block['validation'], environment, reward_setting, ours, METRICS,
                '#### On the validation seeds: picked configurations of this case\n\n',
                'these configurations were picked as the leading ones of each side\'s sweep', budget)
        else:
            checked = None
        if not entries:
            text.append(checked or '')
            continue
        names = described_settings([baseline] + [side for side, _ in entries])
        for metric, title in zip(METRICS, ('Discovery and optimization: whole-curve mean', 'Discovery: mean capped first-goal time')):
            text.append(f'#### {title}\n\n' + metric_table(metric, baseline, entries, names, budget))
            noted = (confirmed_words(run_dir, baseline_block, environment, reward_setting, budget,
                                     best_configuration(baseline.groups, budget, metric), names, metric)
                     if baseline.groups else None)
            text.append(f'{noted}\n\n' if noted else '')
        text.append(edge_sentences(entries, baseline, budget))
        sides = [side for side, _ in entries]
        text.append('#### Whole-curve mean on every configuration of the methods\n\n'
                    'Rows: a method with its settings other than the intrinsic weight; columns: the intrinsic weights. '
                    'Over each method\'s whole grid the highest $W$ is bold and the second highest underlined.\n\n'
                    + grid_tables('whole', sides, budget)
                    + '#### Mean capped first-goal time on every configuration of the methods\n\n'
                    'The same rows and columns, $M$ in thousands of steps; over each method\'s whole grid the smallest '
                    'is bold and the second smallest underlined.\n\n' + grid_tables('discovery', sides, budget)
                    + (checked or ''))
    return ''.join(text)


# ---- discovery ----------------------------------------------------------------------------------------------------

def unit_values_table(sides, names, budget, metric):
    """Every configuration of several sides with its W, L, M and the policies that reach the goal: one row each."""
    # The rows: the sides in their order, each side's configurations from its best on for the metric named.
    rank = compare.rank_configurations if metric == 'whole' else compare.rank_for_discovery
    rows = [(side, group, measures.configuration_measures(group.policies, budget))
            for side in sides for _, _, group in rank(side.groups, budget)]
    digits = decimals(max(row['seeds'] for _, _, row in rows))
    number = lambda value: f'{value:.{digits}f}'
    capped = marks([row['capped_first_goal_step'] for _, _, row in rows], report.thousands, False)
    reached = [round(row['reached_goal_share'] * row['seeds']) for _, _, row in rows]
    reached_marks = report.column_marks([row['reached_goal_share'] for _, _, row in rows],
                                        [f"{count} of {row['seeds']}" for count, (_, _, row) in zip(reached, rows)], True)
    whole = marks([row['whole_curve_mean'] for _, _, row in rows], number, True)
    final = marks([row['final_level'] for _, _, row in rows], number, True)
    return report.markdown_table(
        ['method', 'configuration', '$W$ ± standard error', '$L$ ± standard error', '$M$ ± standard error, thousand steps',
         'policies that reach the goal'],
        [[side.name, configuration_words(group, names),
          fine_mean(row['whole_curve_mean'], row['whole_curve_mean_error'], digits, whole[index]),
          fine_mean(row['final_level'], row['final_level_error'], digits, final[index]),
          report.thousands_with_error(row['capped_first_goal_step'], row['capped_first_goal_step_error'], capped[index]),
          report.styled(f"{reached[index]} of {row['seeds']}", reached_marks[index])]
         for index, (side, group, row) in enumerate(rows)])


def one_budget(sides, where):
    """The one number of steps per policy the units of several sides train; refused when they differ."""
    budgets = sorted({side.budget for side in sides if side.budget is not None})
    if len(budgets) != 1:
        raise ValueError(f'The units of {where} train {budgets} steps per policy; exactly one length is needed.')
    return budgets[0]


VALIDATION_KEYS = {'phase', 'control', 'method_names'}
CASE_VALIDATION_KEYS = VALIDATION_KEYS | {'environment', 'reward_setting'}
METRIC_WORDS = {'whole': ('$W$', 'the highest'), 'discovery': ('$M$', 'the smallest')}


def validation_part(run_dir, batch, block, environment, reward_setting, specifications, metrics, heading, picked, budget=None):
    """Picked configurations on the validation seeds in one case: every unit's measures, then best against best per metric.

    `metrics` names the metrics the units were picked for; `heading` is the part's whole heading line; `picked` says
    in words where the configurations come from. Returns the text, or None when the phase has no unit in the case.
    """
    sides = {slug: read_side(run_dir, select(specifications, [block['phase']], slug, environment, reward_setting, budget), slug, name)
             for slug, name in block['method_names'].items()}
    if not any(side.units for side in sides.values()):
        return None
    control = sides[block['control']]
    others = [side for slug, side in sides.items() if slug != block['control']]
    level = heading.split(' ')[0] + '#'
    units = f'{level} Units\n\n' + units_table([(side, case_words(environment, reward_setting)) for side in sides.values()])
    read = [side for side in [control] + others if side.groups]
    if not read:
        return heading + f"No unit of the phase `{block['phase']}` is complete yet.\n\n" + units
    budget = one_budget(sides.values(), f"the phase {block['phase']}")
    entries = [(side, compare_at_best(side, control, budget, 'picked configurations on the validation seeds', batch,
                                      environment, reward_setting)) for side in others if side.groups]
    names = described_settings(read)
    seeds = sorted({policy.seed for side in read for group in side.groups for policy in group.policies})
    waiting = [f'{side.name}: {read_words(side)}' for side in sides.values() if len(side.groups) < side.planned]
    symbols = ' and '.join(METRIC_WORDS[metric][0] for metric in metrics)
    text = [heading
            + f"The units of the phase `{block['phase']}` in {case_words(environment, reward_setting)}. Each trains one "
            f'configuration on {len(seeds)} seeds, from {seeds[0]} to {seeds[-1]}: neither the development seeds, on '
            f'which {picked}, nor the confirmation seeds. Read at {budget:,} environment steps per policy. Define '
            '$W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each '
            f"method from its best on for {METRIC_WORDS[metrics[0]][0]}; per column the best value is bold and the "
            'second best underlined (the smallest for $M$).\n\n'
            + unit_values_table(read, names, budget, metrics[0])
            + f'Then, for {symbols}: each method at its best configuration on these seeds against the control at its '
            'own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each '
            'best configuration was chosen on these same seeds, among the few run here: a second selection step, '
            'not the comparison on the confirmation seeds.\n\n']
    for metric in metrics:
        text.append(f'Best against best for {METRIC_WORDS[metric][0]} ({METRIC_WORDS[metric][1]} is the best):\n\n'
                    + (metric_table(metric, control, entries, names, budget, 'control', ends=False) if entries else
                       'No method other than the control has a complete unit yet.\n\n'))
    text.append((('Not complete, left out: ' + '; '.join(waiting) + '.\n\n') if waiting else '') + units)
    return ''.join(text)


def validation_section(run_dir, batch, block):
    """A section of its own: every configuration of one phase on the validation seeds in one case, best against best."""
    check_keys(block, CASE_VALIDATION_KEYS, CASE_VALIDATION_KEYS, 'validation_of_configurations')
    heading = ('## Validation seeds: each side at the best of all its configurations\n\n'
               'The section Validation block compares at the screen\'s settings, where only the intrinsic weight '
               'differs. Here every configuration of the phase counts, also one at another learning rate or entropy '
               'coefficient, so each side, the control included, stands at the best of everything it was run at on '
               'these seeds.\n\n')
    found = validation_part(run_dir, batch, {key: block[key] for key in VALIDATION_KEYS}, block['environment'],
                            block['reward_setting'], read_specifications(run_dir), METRICS, heading,
                            'these configurations were picked from each side\'s screens and sweeps')
    return found or heading + f"The phase `{block['phase']}` has no unit in this case.\n\n"


def discovery_validation(run_dir, batch, block, environment, reward_setting, specifications):
    """The discovery check on the validation seeds: every unit's M, then each method's best against the control's best."""
    check_keys(block, VALIDATION_KEYS, VALIDATION_KEYS, 'discovery, validation')
    heading = '### On the validation seeds: the leading configurations for the first-goal time\n\n'
    found = validation_part(run_dir, batch, block, environment, reward_setting, specifications, ('discovery',), heading,
                            "these configurations were picked as the leading ones of each method's discovery sweep")
    return found or heading + f"The phase `{block['phase']}` has no unit in this case.\n\n"


def discovery_development(run_dir, batch, block, environment, reward_setting, specifications):
    """The discovery sweeps on the development seeds: each method's best M against the baseline's best of its own sweep."""
    check_keys(block, {'phases', 'method_names', 'baseline'}, {'phases', 'method_names', 'baseline'}, 'discovery, development')
    check_keys(block['baseline'], BASELINE_KEYS, BASELINE_KEYS - {'confirmation_phases'}, 'discovery, development, baseline')
    sides = [read_side(run_dir, select(specifications, block['phases'], slug, environment, reward_setting), slug, name)
             for slug, name in block['method_names'].items()]
    title = '### On the development seeds: discovery sweeps against the baseline\'s\n\n'
    units = '#### Units\n\n' + units_table([(side, case_words(environment, reward_setting)) for side in sides])
    if not any(side.groups for side in sides):
        return title + 'No unit of these sweeps is complete yet.\n\n' + units
    budget = one_budget(sides, f"the phases {block['phases']}")
    baseline = read_baseline(run_dir, block['baseline'], environment, reward_setting, budget)
    entries = [(side, compare_at_best(side, baseline, budget, 'discovery sweeps on the development seeds', batch))
               for side in sides if side.groups]
    names = described_settings([baseline] + [side for side, _ in entries])
    phases = ', '.join(f'`{phase}`' for phase in block['phases'])
    theirs = ', '.join(f'`{phase}`' for phase in block['baseline']['phases'])
    # The warning: what the baseline's best of many on these seeds becomes on the confirmation seeds.
    noted = (confirmed_words(run_dir, block['baseline'], environment, reward_setting, budget,
                             best_configuration(baseline.groups, budget, 'discovery'), names, 'discovery')
             if baseline.groups else None)
    return (title
            + f"The sweeps of the phases {phases} in {case_words(environment, reward_setting)}, on the development seeds, "
            f"against the sweeps of {baseline.name} (run folder `{block['baseline']['run_folder']}`, phases {theirs}). "
            f'Read at {budget:,} environment steps per policy. One row per method at its configuration with the '
            'smallest $M$ among those read (ties: more policies reach the goal), the baseline first; the cell of the '
            'best configuration says how many configurations it was the best of. Differences are the method minus '
            'the baseline at the baseline\'s own best configuration, on the seeds both have.\n\n'
            + f'{baseline.name}: {read_words(baseline)}. ' + ' '.join(f'{side.name}: {read_words(side)}.' for side in sides) + '\n\n'
            + metric_table('discovery', baseline, entries, names, budget)
            + '**A best of many configurations on few seeds is too low.** The smallest $M$ among many configurations '
            'on the development seeds is partly luck of those seeds, the more so the more configurations there are, '
            'so the numbers of this table are not what the same configurations give on other seeds. '
            + (f'{noted} ' if noted else '')
            + 'The methods\' rows carry the same effect; the validation seeds above and the confirmation seeds are '
            'what decides.\n\n'
            + edge_sentences(entries, baseline, budget, ('discovery',))
            + units)


def discovery_section(run_dir, batch, block):
    """The section of the discovery checks in one case: the validation seeds, then the development sweeps."""
    check_keys(block, DISCOVERY_KEYS, {'environment', 'reward_setting'}, 'discovery')
    specifications = read_specifications(run_dir)
    parts = ['## Discovery checks beyond the screen\n\n'
             'The screen reads each method on one grid of intrinsic weights at the control\'s other settings. For the '
             'secondary metric each method, and the baseline, has its own configuration: the one with the smallest '
             'mean capped first-goal time $M$. This section follows that choice in '
             f"{case_words(block['environment'], block['reward_setting'])}: sweeps of whole configurations on the "
             'development seeds, and the leading configurations of each sweep again on the validation seeds.\n\n']
    if 'validation' in block:
        parts.append(discovery_validation(run_dir, batch, block['validation'], block['environment'],
                                          block['reward_setting'], specifications))
    if 'development' in block:
        parts.append(discovery_development(run_dir, batch, block['development'], block['environment'],
                                           block['reward_setting'], specifications))
    return ''.join(parts)
