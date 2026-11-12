"""Generate one screening batch's summary: <run folder>/batches/<batch>/summary.md, summary.html and plots/.

A screen trains one unit per method: every unit holds the same grid of intrinsic weights (its cells) and the same
development seeds. The summary reads each shard once and holds, for the complete units only: the screen table (each
method at its best intrinsic weight, with its paired difference from the control), the whole-curve mean at every
weight, the property measures of the bonus on the maze grid (`maze_grid.py`), two figures, the batch's written
predictions with what was observed, and a consistency check of the control against a unit of another run folder.
A unit that is not complete is listed and left out; nothing is estimated for it.

The batch folder holds `summary_settings.json`:

- `title`: the summary's title;
- `study_description`: the file name, in the batch folder, of the screen's study description; its `phase` selects
  the batch's units among the run folder's `specifications/`, and its `sweeps` name the methods in order;
- `control`: the `method` of the control's sweep;
- `method_names`: the name in words of every sweep's `method`;
- `consistency_check` (optional): `run_folder` (the name of a run folder beside this one), `unit` (a unit of
  it), `settings` (setting name -> value: the cell of that unit and of the control that are compared) and
  `reported` (what was written about that cell earlier, in words);
- `predictions` (optional): the module of this package that checks the predictions written in the batch's
  `ideas.md` (for example `predictions_batch_01`).

Through the test helper's Slurm job (numpy and plotting never run on a login node): see batch_summary_job.py.
"""

from collections import defaultdict
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import re

import numpy as np

from . import compare, contents, figures, maze_grid, measures, records, report

# The test helper that runs the job, relative to the platform folder; the summary names the whole command.
SUBMITTER = ('benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_'
             'screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh')
SETTING_KEYS = {'title', 'study_description', 'control', 'method_names', 'consistency_check', 'predictions'}
REQUIRED_KEYS = {'title', 'study_description', 'control', 'method_names'}
# Shares of the budget at which the unvisited cells and the size of the intrinsic reward are read.
UNVISITED_SHARES = (0.05, 0.25, 1.0)
REWARD_SHARES = (0.25, 1.0)
# A median over fewer (policy, cell) pairs than this is not printed.
FEWEST_PAIRS = 20
VERDICT_WORDS = {True: 'held', False: 'did not hold', None: 'cannot be read yet'}


@dataclass(frozen=True, eq=False)
class Method:
    """One complete unit of a screen: a method's configurations over the intrinsic weights, and what was measured."""
    slug: str                  # the method as the study description spells it
    name: str                  # the method in words
    unit_id: str
    budget: int                # environment steps per policy
    groups: tuple              # the method's configurations, one per intrinsic weight, in weight order
    scores: dict               # intrinsic weight -> (W, L) at the budget
    best: records.Configuration
    measures: dict             # the best configuration's measures at the budget, with 'median_first_goal_step'
    grid: dict                 # the property measures of the bonus on the maze grid (grid_measures)

    @property
    def best_weight(self):
        """The intrinsic weight of the method's best configuration."""
        return weight_of(self.best)

    @property
    def weights(self):
        """The intrinsic weights of the method's cells, in order."""
        return [weight_of(group) for group in self.groups]


@dataclass(frozen=True, eq=False)
class Screen:
    """Everything read for one batch summary."""
    batch: str
    control: str               # the control's method, as the study description spells it
    order: tuple               # every method of the study description, in its order
    names: dict                # method -> its name in words
    units: tuple               # one row per method for the units table
    methods: dict              # method -> Method, the complete units only
    differences: dict          # idea -> its paired differences from the control (compare.compare_with_baseline)
    budget: int
    seeds: tuple               # the screen's seeds

    def complete(self, *slugs):
        """Whether every named method's unit is complete."""
        return all(slug in self.methods for slug in slugs)


# ---- reading ----------------------------------------------------------------------------------------------------

def read_settings(folder):
    """The batch folder's summary_settings.json, with its keys checked."""
    path = Path(folder) / 'summary_settings.json'
    settings = json.loads(path.read_text())
    unknown, missing = set(settings) - SETTING_KEYS, REQUIRED_KEYS - set(settings)
    if unknown or missing:
        raise ValueError(f'{path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}.')
    return settings


def batch_units(run_dir, description):
    """The unit of every method of the screen, from the run folder's unit specifications; None where none was built."""
    # before: specifications of the phase 'batch-01-screen' with the methods 'control-cleanrl-ppo-rnd' and 'even-start'
    # after:  {'control-cleanrl-ppo-rnd': 'batch-01-screen_..._control-cleanrl-ppo-rnd_cells-...', 'even-start': ...},
    #         in the order of the study description's sweeps.
    # The units of the screen's phase, by method; a method may have one unit only.
    found = defaultdict(list)
    for path in sorted((Path(run_dir) / 'specifications').glob('*.json')):
        specification = json.loads(path.read_text())
        if specification.get('phase') == description['phase']:
            found[specification['method']].append(specification['unit_id'])
    split = sorted(method for method, units in found.items() if len(units) > 1)
    if split:
        raise ValueError(f"The method {split[0]} of the phase {description['phase']} is split over the units "
                         f'{found[split[0]]}; a screen summary reads one unit per method.')
    return {sweep['method']: found[sweep['method']][0] if sweep['method'] in found else None
            for sweep in description['sweeps']}


def queue_place(run_dir, unit_id):
    """The folders of the run's queue that hold the unit's file, as written on disk."""
    queue = Path(run_dir) / 'queue'
    held = sorted(folder.name for folder in queue.iterdir() if (folder / f'{unit_id}.json').exists()) if queue.is_dir() else []
    return ', '.join(f'`queue/{name}`' for name in held) if held else 'in no queue folder'


def weight_of(group):
    """A configuration's intrinsic weight."""
    return group.setting('int_coef')


def score_weights(groups, budget):
    """A method's configurations with their W and L at the budget: (intrinsic weight -> (W, L), the best one)."""
    # before: configurations at the weights 1 and 4 with W 0.5 and 0.5 and L 0.2 and 0.9
    # after:  ({1: (0.5, 0.2), 4: (0.5, 0.9)}, the configuration at weight 4): equal W, so the higher L decides.
    ranked = compare.rank_configurations(groups, budget)
    return {weight_of(group): (whole, final) for whole, final, group in ranked}, ranked[0][2]


def grid_measures(grid, cell_weights, best_weight, budget):
    """The property measures of one unit's maze-grid records: at the best intrinsic weight, and across the weights."""
    # The policies of each weight. before: copy_cell [0, 0, 1, 1], cell weights [1.0, 4.0]; after: {1.0: [0, 1],
    # 4.0: [2, 3]}.
    members = {weight: np.flatnonzero(grid.copy_cell == index) for index, weight in enumerate(cell_weights)}
    best = members[best_weight]
    # Per policy the start ratio, per policy and cell the decay slope; the medians are taken at the end.
    ratios = maze_grid.start_ratio(grid.start_bonus)
    slopes = maze_grid.decay_slopes(grid.bonus, grid.visits)

    # Unvisited cells, read at the last logging window that ends at or before each share of the budget.
    unvisited = {}
    for share in UNVISITED_SHARES:
        window = maze_grid.window_at(grid.steps, share * budget)
        found = maze_grid.unvisited_ratios(grid.bonus[window][best], grid.start_bonus[best], grid.visits[window][best])
        unvisited[share] = {'pairs': len(found), 'step': int(grid.steps[window]),
                            'median': float(np.median(found)) if len(found) >= FEWEST_PAIRS else None}
    # The size of the intrinsic reward and its divisor: at the start, after the first window, and at the shares.
    reward = {'start': maze_grid.weighted_intrinsic_reward(grid.start_reward[best], best_weight),
              'first_window': maze_grid.weighted_intrinsic_reward(grid.reward[0][best], best_weight)}
    divisor = {'start': float(np.median(grid.start_divisor[best])), 'first_window': float(np.median(grid.divisor[0][best]))}
    for share in REWARD_SHARES:
        window = maze_grid.window_at(grid.steps, share * budget)
        reward[share] = maze_grid.weighted_intrinsic_reward(grid.reward[window][best], best_weight)
    return {'start_ratio': finite_median(ratios),
            'start_ratio_by_weight': {weight: finite_median(ratios[rows]) for weight, rows in members.items()},
            'decay_slope': finite_median(slopes[best]), 'decay_slope_pairs': int(np.isfinite(slopes[best]).sum()),
            'decay_slope_by_weight': {weight: finite_median(slopes[rows]) for weight, rows in members.items()},
            'unvisited': unvisited, 'intrinsic_reward': reward, 'divisor': divisor, 'first_window_step': int(grid.steps[0]),
            'bins': maze_grid.visit_bins(grid.bonus[:, best], grid.visits[:, best]),
            'pairs': int(len(best) * len(grid.cells)), 'problems': maze_grid.problems(grid),
            'largest_bonus': {'start': float(np.nanmax(grid.start_bonus)), 'windows': float(np.nanmax(grid.bonus))}}


def finite_median(values):
    """The median of the finite values; None when there is none."""
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def read_method(shard, slug, name, budget):
    """One unit's shard, read once: (its row of the units table, the Method when the unit is complete else None)."""
    # What the units table shows of any unit, complete or not.
    lines = records.read_lines(shard)
    summary, policies = records.unit_from_records(shard, lines, [{'name': name, 'unit_id_pattern': shard.stem}])
    attempt = records.latest_attempt(lines)
    jobs = [record for record in lines if record.get('record') == 'job']
    row = {'slug': slug, 'name': name, 'unit_id': shard.stem, 'state': summary['state'], 'windows': summary['windows'],
           'planned_windows': attempt[0]['iterations'] // attempt[0]['window_iterations'] if attempt else None,
           'card': jobs[-1].get('device_kind', 'not recorded') if jobs else 'not recorded',
           'finished': summary['finished'],
           'started_again': summary['started_from_zero_again'], 'continued': summary['continued_from_a_checkpoint'],
           'respawned': summary['respawned_episodes'], 'problems': {}, 'largest_bonus': None}
    if summary['state'] != 'complete':
        return row, None
    if summary['planned_steps'] != budget:
        raise ValueError(f"Unit {shard.stem} trains {summary['planned_steps']:,} steps per policy; the screen's "
                         f'study description says {budget:,}.')

    # One configuration per intrinsic weight: the unit's cells must differ in nothing else.
    groups = sorted(records.configurations(policies), key=weight_of)
    if records.varying_settings(groups) != ['int_coef']:
        raise ValueError(f'The cells of unit {shard.stem} differ in {records.varying_settings(groups)}; a screen '
                         'summary reads units whose cells differ in the intrinsic weight only.')
    scores, best = score_weights(groups, budget)
    measured = measures.configuration_measures(best.policies, budget)
    measured['median_first_goal_step'] = measures.median_first_goal_step(best.policies, budget)

    # The maze-grid records, reduced to the property measures so that the arrays of one unit are not kept.
    start = attempt[0]
    position = start['cell_setting_names'].index('int_coef')
    grid = grid_measures(maze_grid.read_grid(attempt), [cell[position] for cell in start['cell_settings']],
                         weight_of(best), budget)
    not_finite = sum(int((~np.isfinite(getattr(policy, name_of_sum))).sum())
                     for policy in policies for name_of_sum in ('episodes', 'return_sum'))
    row['problems'] = {**grid['problems'], **({'episode counts or return sums that are not finite': not_finite}
                                              if not_finite else {})}
    row['largest_bonus'] = grid['largest_bonus']
    return row, Method(slug=slug, name=name, unit_id=shard.stem, budget=budget, groups=tuple(groups), scores=scores,
                       best=best, measures=measured, grid=grid)


def read_screen(run_dir, batch, settings):
    """Read every unit of the batch's screen and compare each complete idea with the control."""
    # The screen's methods and their units; every method needs a name in words.
    run_dir = Path(run_dir)
    description = json.loads((run_dir / 'batches' / batch / settings['study_description']).read_text())
    units = batch_units(run_dir, description)
    unnamed = sorted(set(units) - set(settings['method_names']))
    if unnamed or settings['control'] not in units:
        raise ValueError(f"summary_settings.json: no name in words for the methods {unnamed}, or the control "
                         f"{settings['control']} is not a method of the study description.")
    # One row per method for the units table; a unit with no shard is listed without being read.
    budget = description['steps_per_policy']
    rows, methods = [], {}
    for slug, unit_id in units.items():
        name = settings['method_names'][slug]
        shard = run_dir / 'data' / f'{unit_id}.jsonl'
        if unit_id is None or not shard.exists():
            rows.append({'slug': slug, 'name': name, 'unit_id': unit_id,
                         'state': 'no unit was built' if unit_id is None else 'not started (no shard)',
                         'queue': 'no unit' if unit_id is None else queue_place(run_dir, unit_id)})
            continue
        row, method = read_method(shard, slug, name, budget)
        rows.append({**row, 'queue': queue_place(run_dir, unit_id)})
        if method is not None:
            methods[slug] = method

    # Every complete idea against the control at the control's own best weight, on the seeds both have.
    differences = {}
    if settings['control'] in methods:
        control = methods[settings['control']]
        differences = {slug: compare.compare_with_baseline(method.best.policies, control.best.policies, budget,
                                                           'batch screen', batch, slug)
                       for slug, method in methods.items() if slug != settings['control']}
    seeds = tuple(range(description['seeds']['first'], description['seeds']['first'] + description['seeds']['count']))
    return Screen(batch=batch, control=settings['control'], order=tuple(units), names=dict(settings['method_names']),
                  units=tuple(rows), methods=methods, differences=differences, budget=budget, seeds=seeds)


# ---- words and cells ----------------------------------------------------------------------------------------------

def weight_text(weight):
    """An intrinsic weight for a table: the number, or a power of 4 when the number is long."""
    # before: 2.0, 0.0625, 0.000244140625; after: '2', '0.0625', '$4^{-6}$'.
    text = f'{weight:g}'
    power = round(math.log(weight, 4))
    return f'$4^{{{power}}}$' if len(text) > 6 and math.isclose(4.0 ** power, weight) else text


def steps_text(value):
    """A step count for a table cell; 'none' for a median that falls on a policy with no arrival."""
    return 'none' if math.isinf(value) else report.compact_steps(value)


def short(value):
    """A ratio or reward for a table cell: three significant digits, written out without a power of ten."""
    # before: 1.0234, 70.0, 0.00096012, 0.0000123; after: '1.02', '70', '0.00096', '0.0000123'.
    return '0' if value == 0 else np.format_float_positional(value, precision=3, unique=False, fractional=False, trim='-')


def difference_cell(result, quantity, style):
    """A paired difference with its 95% interval; the mean bold or underlined."""
    # before: {'mean': 0.021, 'interval': [0.004, 0.038]}, 'best'; after: '**+0.021** [+0.004, +0.038]'.
    low, high = result[quantity]['interval']
    return f"{report.styled(format(result[quantity]['mean'], '+.3f'), style)} [{low:+.3f}, {high:+.3f}]"


def row_order(screen):
    """The complete methods in table order: the control first, then the ideas from the highest W down (ties: L)."""
    ideas = [slug for slug in screen.order if slug in screen.methods and slug != screen.control]
    ideas.sort(key=lambda slug: (-screen.methods[slug].measures['whole_curve_mean'],
                                 -screen.methods[slug].measures['final_level'], screen.order.index(slug)))
    return ([screen.control] if screen.control in screen.methods else []) + ideas


def end_of_grid(method):
    """'smallest' or 'largest' when the method's best weight is at an end of its grid, else None."""
    weights = method.weights
    return 'smallest' if method.best_weight == weights[0] else 'largest' if method.best_weight == weights[-1] else None


# ---- sections -----------------------------------------------------------------------------------------------------

def units_section(screen):
    """Which units are complete: one row per method, and the unit names under the table."""
    # One row per unit; a unit that was not read (no shard) has no windows, card or finishing time.
    header = ['unit', 'method', 'state', 'logging windows written', 'card', 'finished', 'place in the queue']
    rows = []
    for index, unit in enumerate(screen.units, 1):
        if 'windows' not in unit:
            rows.append([str(index), unit['name'], unit['state'], '0', 'none', 'not finished', unit['queue']])
            continue
        finished = contents.pacific(unit['finished']) if unit['finished'] else (
            'time not recorded' if unit['state'] == 'complete' else 'not finished')
        written = f"{unit['windows']} of {unit['planned_windows']}" if unit['planned_windows'] else '0'
        rows.append([str(index), unit['name'], unit['state'], written, unit['card'], finished, unit['queue']])
    # The methods left out of the tables by name, and the unit ids under the table.
    left_out = [unit['name'] for unit in screen.units if unit['slug'] not in screen.methods]
    names = ''.join(f"{index}. `{unit['unit_id']}`\n" for index, unit in enumerate(screen.units, 1))
    return ('## Units\n\n'
            f'{len(screen.methods)} of {len(screen.units)} units are complete. Only complete units enter the tables and '
            'figures below; nothing is estimated for the others. '
            + (f"Left out: {'; '.join(left_out)}.\n\n" if left_out else 'No unit is left out.\n\n')
            + report.markdown_table(header, rows) + 'Unit names, by the number in the first column:\n\n' + names + '\n')


def measures_section(screen):
    """The definitions the tables use, each with its formula."""
    count, weights = len(screen.seeds), len(next(iter(screen.methods.values())).weights)
    return (
        '## Measures\n\n'
        f'Define $K$ as the number of policies (seeds) of one method at one intrinsic weight ($K = {count}$, the '
        f'development seeds {screen.seeds[0]} to {screen.seeds[-1]}), $B$ as the budget in environment steps per '
        f'policy ($B$ = {screen.budget:,}), $R_i(s)$ as the mean return of the training episodes of policy $i$ that '
        'end in the logging window containing step $s$, and $\\Delta s$ as the number of that window\'s steps inside '
        'the range being averaged. An episode returns 1 if it reaches the goal and 0 otherwise. A window in which '
        'no episode ends takes the value of the window before it; before a policy\'s first finished episode the '
        'value is 0.\n\n'
        '1. **Whole-curve mean:** $W(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\dfrac{1}{B}\\sum_{s \\le B} R_i(s) '
        '\\thinspace \\Delta s$, the area under the training curve divided by the budget.\n'
        '2. **Final level:** $L(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\dfrac{1}{0.1 B}\\sum_{0.9 B \\lt s \\le B} R_i(s) '
        '\\thinspace \\Delta s$, the same mean over the last 10% of the budget.\n'
        f'3. **Best intrinsic weight:** per method, the one of its {weights} intrinsic weights with the highest $W$; '
        'ties go to the higher $L$.\n'
        '4. **Difference from the control:** define $d_i$ as the value of seed $i$ under the method at its best '
        'weight minus its value under the control at the control\'s own best weight. The tables give '
        '$\\bar d = \\dfrac{1}{K}\\sum_{i=1}^{K} d_i$ for $W$ and for $L$, each with a 95% interval: the 2.5% and '
        '97.5% points of the means of 10,000 resamples of the $K$ seeds, drawn with replacement. An idea is ahead '
        'of the control when the interval is entirely above zero, behind when it is entirely below zero, and '
        'within the interval otherwise.\n'
        '5. **Median first-arrival step:** define $T_i$ as the training step of policy $i$\'s first arrival at the '
        'goal ($T_i = \\infty$ if it has none by step $B$). The measure is the median of $T_i$ over the $K$ '
        'policies; it reads "none" when the median falls on a policy with no arrival.\n'
        '6. **Policies that reach the goal:** the number of the $K$ policies with $T_i \\le B$.\n\n'
        'Standard errors are over the $K$ policies.\n\n')


def screen_section(screen, order):
    """The screen table: each method at its best intrinsic weight, the control first, the ideas sorted by W."""
    table = [screen.methods[slug] for slug in order]
    results = [screen.differences.get(slug) for slug in order]
    compared = [result if result and result['seeds'] >= 2 else None for result in results]

    # The marks of every metric column; a row with no value in a column (the control's differences, a median with
    # no arrival) takes no mark there.
    whole = report.mean_marks([method.measures['whole_curve_mean'] for method in table])
    final = report.mean_marks([method.measures['final_level'] for method in table])
    marks = {}
    for quantity in ('whole_curve_mean', 'final_level'):
        values = [result[quantity]['mean'] if result else None for result in compared]
        marks[quantity] = report.column_marks(values, [None if value is None else f'{value:+.3f}' for value in values], True)
    arrivals = [method.measures['median_first_goal_step'] for method in table]
    arrival_marks = report.column_marks([None if math.isinf(value) else value for value in arrivals],
                                        [steps_text(value) for value in arrivals], False)
    reached = [round(method.measures['reached_goal_share'] * method.measures['seeds']) for method in table]
    reached_marks = report.column_marks(reached, [str(value) for value in reached], True)

    # One row per method; the control's two difference cells say that it is the control.
    header = ['method', 'best intrinsic weight', 'whole-curve mean $W$ ↑ ± standard error',
              'difference in $W$ from the control [95% interval]', 'final level $L$',
              'difference in $L$ from the control [95% interval]', 'median first-arrival step',
              'policies that reach the goal']
    rows = []
    for index, (slug, method, result) in enumerate(zip(order, table, compared)):
        row = method.measures
        if slug == screen.control:
            differences = ['control', 'control']
        elif result is None:
            differences = ['not compared'] * 2
        else:
            differences = [difference_cell(result, quantity, marks[quantity][index])
                           for quantity in ('whole_curve_mean', 'final_level')]
        rows.append([method.name, weight_text(method.best_weight),
                     report.marked_mean(row['whole_curve_mean'], row['whole_curve_mean_error'], whole[index]),
                     differences[0], report.styled(f"{row['final_level']:.3f}", final[index]), differences[1],
                     report.styled(steps_text(arrivals[index]), arrival_marks[index]),
                     report.styled(f"{reached[index]} of {row['seeds']}", reached_marks[index])])
    # What the differences are taken against, in words.
    control = screen.methods.get(screen.control)
    against = (f'Differences are the method minus the control at the control\'s own best intrinsic weight '
               f'({weight_text(control.best_weight)}), on the {len(screen.seeds)} seeds both have; the brackets hold '
               'the 95% interval.' if control else
               'The control\'s unit is not complete, so no difference from the control can be given yet.')
    return ('## Screen table\n\n'
            'Define $W$, $L$ and the other columns as in the section Measures. One row per method at its best '
            f'intrinsic weight, read at {screen.budget:,} environment steps per policy. The control is the first '
            'row; the ideas follow, sorted by $W$. Per column, the best value is bold and the second best '
            f'underlined; for the median first-arrival step the smallest value is the best. {against} The seeds are '
            'the development seeds, on which each best weight was also chosen: this table is a screen and not a '
            'verdict.\n\n' + report.markdown_table(header, rows))


def weights_section(screen, order):
    """W of every method at every intrinsic weight, in two tables of half the weights each."""
    # Every method must have been trained on the same grid of weights.
    table = [screen.methods[slug] for slug in order]
    weights = table[0].weights
    different = [method.name for method in table if method.weights != weights]
    if different:
        raise ValueError(f'The methods {different} were trained on another grid of intrinsic weights than {table[0].name}.')
    text = ['## Whole-curve mean at every intrinsic weight\n\n'
            f'Define $W$ as in the section Measures. Rows are in the order of the screen table. The {len(weights)} '
            'weights are split over two tables so that each fits the width of a printed page. Per weight (column), '
            'the highest $W$ among the methods is bold and the second highest underlined. The last column names '
            'each method\'s best weight and says when it is the smallest or the largest weight of the grid.\n\n']
    # One table per half of the grid; per weight, the best and second best method are marked.
    half = (len(weights) + 1) // 2
    for part in (weights[:half], weights[half:]):
        marks = {weight: report.mean_marks([method.scores[weight][0] for method in table]) for weight in part}
        rows = []
        for index, method in enumerate(table):
            end = end_of_grid(method)
            rows.append([method.name]
                        + [report.styled(f'{method.scores[weight][0]:.3f}', marks[weight][index]) for weight in part]
                        + [weight_text(method.best_weight) + (f' ({end} of the grid)' if end else '')])
        header = ['method'] + [f'weight {weight_text(weight)}' for weight in part] + ['best intrinsic weight']
        text.append(report.markdown_table(header, rows))
    # The methods whose best weight is the smallest or the largest of the grid.
    at_end = [f'{method.name} ({weight_text(method.best_weight)}, the {end_of_grid(method)})'
              for method in table if end_of_grid(method)]
    text.append(('Best weight at an end of the grid: ' + '; '.join(at_end) + '. The protocol\'s edge rule '
                 '(`program.md`) asks for three more values on that side before anything is confirmed.\n\n')
                if at_end else 'No method has its best weight at an end of the grid.\n\n')
    return ''.join(text)


def slope_range(by_weight):
    """The smallest and the largest of the per-weight median slopes, each with its weight, as a table cell."""
    # before: {1.0: -0.4, 4.0: -0.9, 16.0: None}; after: '-0.90 (4) to -0.40 (1)'.
    found = sorted((value, weight) for weight, value in by_weight.items() if value is not None)
    if not found:
        return 'no slope'
    return ' to '.join(f'{value:.2f} ({weight_text(weight)})' for value, weight in (found[0], found[-1]))


def unvisited_cell(entry):
    """One share's unvisited-cell reading as a table cell: the median with its number of pairs, or why there is none."""
    if entry['pairs'] == 0:
        return 'no pair without a visit'
    if entry['median'] is None:
        return f"not printed: {entry['pairs']} pairs"
    return f"{short(entry['median'])} ({entry['pairs']} pairs)"


def properties_section(screen, order, visits_figure_name):
    """The property measures of the bonus on the maze grid: the start, the decay, and the size of the intrinsic reward."""
    # The definitions, with the number of open cells and the steps at which the shares of the budget are read.
    table = [screen.methods[slug] for slug in order]
    first = table[0].grid
    cells = first['pairs'] // len(table[0].best.policies)
    steps = {share: first['unvisited'][share]['step'] for share in UNVISITED_SHARES}
    shares = ', '.join(f'{100 * share:g}% is read at step {steps[share]:,}' for share in UNVISITED_SHARES)
    text = [
        '## Properties of the bonus on the maze grid\n\n'
        f'Every unit records, per policy and per open maze cell ({cells} cells), the bonus at one probe observation '
        '(the cell\'s centre, zero velocity) at the end of every logging window, and the visits to the cell. Define '
        '$b_i(c, w)$ as the bonus before the divisor of policy $i$ at the probe of cell $c$ at the end of logging '
        'window $w$, with $w = 0$ for the record written after the random-action rollouts and before the first '
        'training iteration; $r_i(c, w)$ as the same bonus after the divisor (the intrinsic reward); $n_i(c, w)$ as '
        'the number of states of policy $i$\'s rollouts in cell $c$ from the first window to window $w$; $\\beta$ as '
        f'the intrinsic weight; $C$ as the number of open cells ($C = {cells}$); and $w_f$ as the last logging window '
        f'that ends at or before the share $f$ of the budget ({shares}).\n\n'
        '1. **Start ratio:** $E_i = \\dfrac{\\max_c b_i(c, 0)}{\\min_c b_i(c, 0)}$, the largest over the smallest '
        'start bonus on the grid. The table gives the median over all policies of the unit (every weight).\n'
        '2. **Decay slope:** define $x_w = \\log(1 + n_i(c, w))$ and $y_w = \\log b_i(c, w)$, and $\\bar x$ and '
        '$\\bar y$ as their means over the windows $w \\ge 1$ with $n_i(c, w) \\ge 1$. Per policy and cell, '
        '$a_i(c) = \\dfrac{\\sum_w (x_w - \\bar x)(y_w - \\bar y)}{\\sum_w (x_w - \\bar x)^2}$ over those windows: the '
        'least-squares slope of the logarithm of the bonus on the logarithm of one plus the cumulative visits. A '
        'slope of $-1$ means the bonus falls in proportion to $1/(1 + n)$. The table gives the median over the '
        'policies and cells at the method\'s best weight, and the smallest and the largest of the per-weight medians '
        'with the weights they belong to: when the two differ, the slope depends on the weight.\n'
        '3. **Bonus at unvisited cells over its start:** $u_i(c, f) = \\dfrac{b_i(c, w_f)}{b_i(c, 0)}$ for the '
        '(policy, cell) pairs with $n_i(c, w_f) = 0$, at the method\'s best weight. The table gives the median over '
        f'those pairs and their number; a median over fewer than {FEWEST_PAIRS} pairs is not printed. A value '
        'below 1 means that training on other cells lowered the bonus of a cell with no visit.\n'
        '4. **Weighted intrinsic reward:** $q(w) = \\dfrac{\\beta}{K C}\\sum_{i=1}^{K}\\sum_{c=1}^{C} r_i(c, w)$, at '
        'the method\'s best weight: the intrinsic reward a first visit to a cell would pay, times the intrinsic '
        'weight, averaged over cells and policies. It is read at the start ($w = 0$), at the end of the first '
        f"logging window (step {first['first_window_step']:,}) and at "
        + ' and '.join(f'{100 * share:g}%' for share in REWARD_SHARES) + ' of the budget. The task pays 1 on arrival. '
        'The second table also gives the divisor (the median over the policies) in the start record and at the end '
        'of the first logging window. A divisor that follows the running standard deviation is 1 in the start '
        'record, because no rollout has entered its statistics yet, and the trainer updates it with each rollout '
        'before it divides that rollout\'s rewards; for such a method the start column is the bonus itself times '
        'the weight, and the first logging window is the earliest reading with a divisor the learner used.\n\n'
        'Rows are in the order of the screen table. These columns describe the bonus and have no better or worse '
        'direction, so no value is marked.\n\n']

    # First table: the start, the decay and the unvisited cells.
    header = ['method', 'start ratio, median over policies', 'decay slope at the best weight, median',
              'decay slope, smallest and largest median over the weights (at weight)']
    header += [f'bonus at unvisited cells over its start at {100 * share:g}% of the budget' for share in UNVISITED_SHARES]
    rows = []
    for method in table:
        grid = method.grid
        rows.append([method.name,
                     'no usable value' if grid['start_ratio'] is None else f"{grid['start_ratio']:.2f}",
                     'no slope' if grid['decay_slope'] is None else f"{grid['decay_slope']:.2f}",
                     slope_range(grid['decay_slope_by_weight'])]
                    + [unvisited_cell(grid['unvisited'][share]) for share in UNVISITED_SHARES])
    text.append(report.markdown_table(header, rows))

    # Second table: the divisor and the size of the intrinsic reward; then the figure.
    header = (['method', 'best intrinsic weight', 'divisor at the start, and at the end of the first logging window',
               'weighted intrinsic reward at the start', 'weighted intrinsic reward at the end of the first logging window']
              + [f'weighted intrinsic reward at {100 * share:g}% of the budget' for share in REWARD_SHARES])
    rows = [[method.name, weight_text(method.best_weight),
             f"{short(method.grid['divisor']['start'])} and {short(method.grid['divisor']['first_window'])}"]
            + [short(method.grid['intrinsic_reward'][key]) for key in ('start', 'first_window') + REWARD_SHARES]
            for method in table]
    text.append(report.markdown_table(header, rows))
    text.append('### Bonus against cumulative visits\n\n'
                'One panel per method in the order of the screen table, at the method\'s best intrinsic weight. Each '
                'point is the median of $b_i(c, w)$ over the policies, cells and logging windows whose '
                '$1 + n_i(c, w)$ falls in one bin (each bin a factor of two wide; pairs with no visit are left '
                'out); the band holds the middle half of the same points. Both axes are logarithmic, and every '
                'panel shows the same range of visits and the same number of decades of the bonus, so equal slopes '
                'look equal. The dashed line has slope $-1$ and passes through the first bin.\n\n'
                f'![bonus against cumulative visits](plots/{visits_figure_name}.png)\n\n')
    return ''.join(text)


def curves_section(screen, name):
    """The section that shows the training-curve figure; empty when no idea is complete."""
    if name is None:
        return ''
    control = screen.methods.get(screen.control)
    grey = (f'the control at its best intrinsic weight ({weight_text(control.best_weight)}) in grey' if control
            else 'the control is not complete and is not drawn')
    return ('## Training curves\n\n'
            'Define $\\bar R(s) = \\dfrac{1}{K}\\sum_{i=1}^{K} R_i(s)$, with $R_i(s)$ and $K$ as in the section '
            'Measures. One panel per idea in the order of the screen table: the idea at its best intrinsic weight '
            f'in blue, {grey}; bands are one standard error over the {len(screen.seeds)} seeds.\n\n'
            f'![training curves](plots/{name}.png)\n\n')


def sides_section(screen, order):
    """Which ideas are ahead of the control, within the interval, or behind, by the interval of the difference in W."""
    if screen.control not in screen.methods:
        return '## Ideas against the control\n\nThe control\'s unit is not complete, so no idea can be compared yet.\n\n'
    # Every compared idea under the side of its interval, with the difference and the interval.
    lists = {'ahead': [], 'level': [], 'behind': []}
    for slug in order:
        result = screen.differences.get(slug)
        if result and result['seeds'] >= 2:
            lists[result['whole_curve_mean']['side']].append(
                f"{screen.methods[slug].name} ({report.interval_text(result['whole_curve_mean']).rsplit(',', 1)[0]})")
    waiting = [screen.names[slug] for slug in screen.order if slug not in screen.methods]
    rows = [('Ahead of the control', 'interval entirely above zero', lists['ahead']),
            ('Within the interval', 'it contains zero', lists['level']),
            ('Behind the control', 'interval entirely below zero', lists['behind']),
            ('Not compared yet', 'unit not complete', waiting)]
    return ('## Ideas against the control\n\n'
            'By the 95% interval of the difference in $W$ from the control (section Measures, item 4), on the '
            'development seeds. After each name: the difference in $W$ and its interval.\n\n'
            + ''.join(f"- **{label}** ({rule}): {'; '.join(names) if names else 'none'}.\n" for label, rule, names in rows)
            + '\n')


# ---- predictions --------------------------------------------------------------------------------------------------

def written_predictions(ideas_text):
    """The numbered predictions of a batch's ideas.md, each as one line of text."""
    # before: '## Predictions (written before the runs)\n\n1. **Even start alone** changes ...\n   above 2 ...\n2. ...'
    # after:  ['**Even start alone** changes ... above 2 ...', '...'].
    section = re.split(r'(?m)^## Predictions[^\n]*\n', ideas_text)
    if len(section) != 2:
        raise ValueError('ideas.md has no single section whose heading starts with "## Predictions".')
    body = re.split(r'(?m)^## ', section[1])[0]
    items = re.split(r'(?m)^\d+\. ', body)[1:]
    return [' '.join(item.split()) for item in items]


def evaluate(parts, screen):
    """Each part of a prediction as (claim, True / False / None, numbers); None when it cannot be read yet."""
    rows = []
    for claim, needs, check in parts:
        waiting = [screen.names[slug] for slug in needs if slug not in screen.methods]
        if waiting:
            rows.append((claim, None, f"Waiting for: {'; '.join(waiting)}."))
        else:
            # A check may itself say that it cannot be read yet (None), when its answer depends on further units.
            held, numbers = check(screen)
            rows.append((claim, None if held is None else bool(held), numbers))
    return rows


def overall(rows):
    """A prediction's word from its parts: held when all held, did not hold when one did not, else cannot be read yet."""
    # before: verdicts [True, None, False]; after: 'did not hold'. [True, None] -> 'cannot be read yet'.
    verdicts = [verdict for _, verdict, _ in rows]
    return VERDICT_WORDS[False if False in verdicts else None if None in verdicts else True]


def predictions_section(screen, ideas_text, module_name):
    """Every written prediction with what was observed, part by part."""
    module = importlib.import_module(f'{__package__}.{module_name}')
    written = written_predictions(ideas_text)
    if len(written) != len(module.PREDICTIONS):
        raise ValueError(f'ideas.md holds {len(written)} predictions; {module_name} checks {len(module.PREDICTIONS)}.')
    text = ['## Predictions\n\n'
            'The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a '
            'part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not '
            'complete. A prediction held when every part held, and did not hold when at least one part did not. '
            f'{module.READING}\n\n']
    for number, (quoted, prediction) in enumerate(zip(written, module.PREDICTIONS), 1):
        rows = evaluate(prediction['parts'], screen)
        text.append(f"### Prediction {number}: {prediction['title']}\n\n> {quoted}\n\nObserved: **{overall(rows)}**.\n\n")
        text += [f'- {claim}: **{VERDICT_WORDS[verdict]}**. {numbers}\n' for claim, verdict, numbers in rows]
        for needs, words in prediction.get('beside', ()):
            if screen.complete(*needs):
                text.append(f'- Beside: {words(screen)}\n')
        text.append('\n')
    return ''.join(text)


# ---- consistency check ----------------------------------------------------------------------------------------------

def matching(groups, wanted, where):
    """The one configuration whose settings have the wanted values; refused when none or several have."""
    found = [group for group in groups if all(group.setting(name) == value for name, value in wanted.items())]
    if len(found) != 1:
        raise ValueError(f'{len(found)} configurations of {where} have the settings {wanted}; exactly one is needed.')
    return found[0]


def setting_differences(ours, theirs):
    """The settings recorded with different values in two configurations, and those only one of them records."""
    # before: ours {'int_coef': 2.0, 'clip_coef': 0.1, 'permutation_draw': 'x'}, theirs {'int_coef': 2.0, 'clip_coef': 0.2}
    # after:  {'different': {'clip_coef': (0.1, 0.2)}, 'only_first': {'permutation_draw': 'x'}, 'only_second': {}}.
    first, second = dict(ours.settings), dict(theirs.settings)
    return {'different': {name: (first[name], second[name]) for name in sorted(first.keys() & second.keys())
                          if first[name] != second[name]},
            'only_first': {name: first[name] for name in sorted(first.keys() - second.keys())},
            'only_second': {name: second[name] for name in sorted(second.keys() - first.keys())}}


def consistency_section(screen, run_dir, check):
    """The control's cell against the same settings and seeds in a unit of another run folder."""
    title = '## Consistency check against the baseline run\n\n'
    if screen.control not in screen.methods:
        return title + 'The control\'s unit is not complete, so the check cannot be made yet.\n\n'
    shard = Path(run_dir).parent / check['run_folder'] / 'data' / f"{check['unit']}.jsonl"
    lines = records.read_lines(shard)
    summary, policies = records.unit_from_records(shard, lines)
    if summary['state'] != 'complete':
        raise ValueError(f"The unit {check['unit']} of the consistency check is {summary['state']}.")
    jobs = [record for record in lines if record.get('record') == 'job']
    control = screen.methods[screen.control]
    ours = matching(control.groups, check['settings'], f'the control unit {control.unit_id}')
    theirs = matching(records.configurations(policies), check['settings'], f"the unit {check['unit']}")
    budget = screen.budget
    if theirs.planned_steps != budget or theirs.environment != ours.environment or theirs.reward_setting != ours.reward_setting:
        raise ValueError(f"The unit {check['unit']} trains {theirs.planned_steps:,} steps in {theirs.environment} with "
                         f'{theirs.reward_setting}; the control trains {budget:,} in {ours.environment} with {ours.reward_setting}.')

    # The two rows side by side, then the paired difference on the seeds both have.
    card = {unit['slug']: unit['card'] for unit in screen.units if 'card' in unit}[screen.control]
    rows = []
    for words, group, device in ((f'control of this batch ({control.name})', ours, card),
                                 ('the baseline run\'s unit', theirs,
                                  jobs[-1].get('device_kind', 'not recorded') if jobs else 'not recorded')):
        row = measures.configuration_measures(group.policies, budget)
        rows.append([words, report.with_error(row['whole_curve_mean'], row['whole_curve_mean_error']),
                     report.with_error(row['final_level'], row['final_level_error']), str(row['seeds']), device])
    result = compare.compare_with_baseline(ours.policies, theirs.policies, budget, 'batch screen', screen.batch,
                                           'consistency check')
    if result['seeds'] < 2:
        raise ValueError(f"The control and the unit {check['unit']} have {result['seeds']} seeds in common.")
    inside = {'ahead': 'lies entirely above zero', 'behind': 'lies entirely below zero', 'level': 'contains zero'}
    sentences = [f"{words}: {report.interval_text(result[quantity]).rsplit(',', 1)[0]}; the interval "
                 f"{inside[result[quantity]['side']]}"
                 for quantity, words in (('whole_curve_mean', 'in $W$'), ('final_level', 'in $L$'))]

    # What the two start records say differently about the cell.
    found = setting_differences(ours, theirs)
    notes = []
    if found['different']:
        notes.append('Settings recorded with different values (control, baseline unit): '
                     + '; '.join(f'`{name}` {first} and {second}' for name, (first, second) in found['different'].items()) + '.')
    for key, words in (('only_first', 'the control'), ('only_second', 'the baseline unit')):
        if found[key]:
            notes.append(f'Settings that only {words} records (its code has options the other did not have yet): '
                         + ', '.join(f'`{name}` = {value}' for name, value in found[key].items()) + '.')
    if ours.environments_per_policy != theirs.environments_per_policy:
        notes.append(f'Environments per policy differ: {ours.environments_per_policy} and {theirs.environments_per_policy}.')
    if not notes:
        notes.append('The two start records hold the same settings for this cell.')
    described = records.describe_settings(ours, [name for name in records.SETTING_ORDER if name in check['settings']])
    return (title
            + f"The control's cell with {described} has the settings and seeds of one cell of the unit "
            f"`{check['unit']}` in the run folder `{check['run_folder']}`. Reported earlier for that cell: "
            f"{check['reported']}. Define $W$ and $L$ as in the section Measures; both rows are read at {budget:,} "
            'steps per policy.\n\n'
            + report.markdown_table(['run', 'whole-curve mean $W$ ± standard error',
                                     'final level $L$ ± standard error', 'seeds', 'card'], rows)
            + f"Paired difference (this batch minus the baseline run's unit) on the {result['seeds']} seeds both "
            f"have, with its 95% interval from 10,000 resamples of the seeds: {'. '.join(sentences)}.\n\n"
            + ' '.join(notes) + '\n\n')


# ---- records that need a look ---------------------------------------------------------------------------------------

def records_section(screen):
    """Restarts and values the measures could not use, by unit."""
    lines = []
    for unit in screen.units:
        if 'windows' not in unit:
            continue
        found = []
        if unit['started_again']:
            found.append(f"started again from zero {unit['started_again']} time(s); only the records after the last "
                         'start are read')
        if unit['continued']:
            found.append(f"continued from a checkpoint {unit['continued']} time(s)")
        if unit['respawned']:
            found.append(f"{unit['respawned']} episodes started again after a value that was not a number")
        found += [f'{count:,} {kind}' for kind, count in unit['problems'].items()]
        largest = unit['largest_bonus']
        if largest and largest['windows'] > largest['start']:
            found.append(f"the largest bonus at a probe in a logging window is {short(largest['windows'])}, above the "
                         f"largest at the start ({short(largest['start'])}), over all intrinsic weights")
        if found:
            lines.append(f"- {unit['name']} (`{unit['unit_id']}`): {'; '.join(found)}.\n")
    body = ''.join(lines) if lines else ('No unit read so far was started again, no recorded value was unusable, and '
                                         'no bonus rose above the largest of its start record.\n')
    return ('## Records that need a look\n\n'
            'Per unit read: starts after the first one, continuations from a checkpoint, recorded values the '
            'measures cannot use (they are left out of the measure and counted here), and a bonus at a probe that '
            'rose above the largest bonus of the start record. A unit whose records lack a field stops the '
            'generation with an error that names it.\n\n' + body + '\n')


# ---- figures and the whole summary --------------------------------------------------------------------------------

def curves_figure(screen, order, plots):
    """Training curves, one panel per idea with the control in grey; returns the file's stem, None with no idea."""
    control = screen.methods.get(screen.control)
    ideas = [screen.methods[slug] for slug in order if slug != screen.control]
    if not ideas:
        return None
    base = None
    if control:
        grid, mean, error = measures.mean_curve(control.best.policies, screen.budget)
        base = {'steps': grid, 'mean': mean, 'error': error, 'band': True, 'color': figures.BASELINE_COLOR,
                'width': 1.2, 'order': 3}
    panels = []
    for method in ideas:
        grid, mean, error = measures.mean_curve(method.best.policies, screen.budget)
        series = [{'steps': grid, 'mean': mean, 'error': error, 'band': True, 'color': figures.SERIES_COLORS[0],
                   'width': 1.4, 'order': 4}]
        title = figures.wrapped(f'{method.name}, intrinsic weight {weight_text(method.best_weight)}', 28)
        panels.append({'title': title, 'series': ([base] if base else []) + series})
    legend = [{'label': 'the idea named above the panel, at its best intrinsic weight', 'color': figures.SERIES_COLORS[0],
               'width': 1.4}]
    if control:
        legend.append({'label': f'control at its best intrinsic weight ({weight_text(control.best_weight)})',
                       'color': figures.BASELINE_COLOR, 'width': 1.2})
    name = figures.file_slug('training curves', 'control against each idea', 'best intrinsic weights')
    return figures.curve_figure(panels, legend, plots, name, 'mean return of training episodes')


def visits_figure(screen, order, plots):
    """The bonus against cumulative visits, one panel per method in table order; returns the file's stem."""
    panels = []
    for slug in order:
        method = screen.methods[slug]
        slope = method.grid['decay_slope']
        words = figures.wrapped(f'{method.name}, intrinsic weight {weight_text(method.best_weight)}', 28)
        panels.append({'title': words + ('\nno slope' if slope is None else f'\nmedian slope {slope:.2f}'),
                       **method.grid['bins']})
    name = figures.file_slug('bonus at the probes against cumulative visits', 'best intrinsic weights')
    return figures.visits_figure(panels, plots, name, 'bonus before the divisor')


def command(run_dir, batch):
    """The command that regenerates the summary, from the platform folder."""
    platform = Path(run_dir).parents[1]
    return (f'cd {platform}\n'
            f'bash "{SUBMITTER}" -s scripts/maze_campaign/analysis/batch_summary_job.py \\\n'
            f'    --run-dir=runs/{Path(run_dir).name} \\\n    --batch={batch}\n')


def build(screen, run_dir, settings, plots):
    """The whole summary in Markdown, section by section; the figures are written on the way."""
    run_dir, folder = Path(run_dir), Path(run_dir) / 'batches' / screen.batch
    order = row_order(screen)
    parts = [f"# {settings['title']}\n\n"
             f'Generated {contents.pacific(contents.now())} by the command below. This file is generated: regenerate '
             'it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: '
             f'`{run_dir.name}`; batch folder: `batches/{screen.batch}`.\n\n```bash\n{command(run_dir, screen.batch)}```\n\n',
             units_section(screen)]
    if order:
        parts += [measures_section(screen), screen_section(screen, order), weights_section(screen, order),
                  curves_section(screen, curves_figure(screen, order, plots)),
                  properties_section(screen, order, visits_figure(screen, order, plots))]
    if 'predictions' in settings:
        parts.append(predictions_section(screen, (folder / 'ideas.md').read_text(), settings['predictions']))
    if order:
        parts.append(sides_section(screen, order))
    if 'consistency_check' in settings:
        parts.append(consistency_section(screen, run_dir, settings['consistency_check']))
    parts.append(records_section(screen))
    return ''.join(parts)


def generate(run_dir, batch):
    """Write the batch's summary.md, summary.html and plots/ from the run folder's shards."""
    run_dir = Path(run_dir).resolve()
    folder = run_dir / 'batches' / batch
    settings = read_settings(folder)
    screen = read_screen(run_dir, batch, settings)
    plots = folder / 'plots'
    plots.mkdir(exist_ok=True)
    markdown = build(screen, run_dir, settings, plots)
    states = contents.write_report(folder, markdown, settings['title'], name='summary')
    return {'summary': folder / 'summary.md', 'sections': states, 'screen': screen}
