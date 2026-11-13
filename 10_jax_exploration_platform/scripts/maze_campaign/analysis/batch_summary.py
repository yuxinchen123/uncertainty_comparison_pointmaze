"""Generate one screening batch's summary: <run folder>/batches/<batch>/summary.md, summary.html and plots/.

A screen trains one unit per method: every unit holds the same grid of intrinsic weights (its cells) and the same
development seeds. The summary reads each shard once and holds, for the complete units only: the two screen tables,
the property measures of the bonus on the maze grid (`maze_grid.py`), two figures, the batch's written predictions
with what was observed, and a consistency check of the control against a unit of another run folder. A unit that
is not complete is listed and left out; nothing is estimated for it.

Two metrics, each with its own table and its own best intrinsic weight per method (user decision 2026-11-12):

- "Discovery and optimization: whole-curve mean": each method at the weight with the highest whole-curve mean W
  (ties: the higher final level L), with its paired difference from the control, and W at every weight;
- "Discovery: mean capped first-goal time": each method at the weight with the smallest mean capped first-goal
  step M (ties: more policies that reach the goal), with its paired difference in M from the control at the
  control's own best weight for M, W and L at that same weight, and M at every weight.

The batch folder holds `summary_settings.json`:

- `title`: the summary's title;
- `study_description`: the file name, in the batch folder, of the screen's study description; its `phase` selects
  the batch's units among the run folder's `specifications/`, and its `sweeps` name the methods in order;
- `control`: the `method` of the control's sweep;
- `control_study_description` (optional): for a batch that trains no control of its own, the study description
  of another batch (a path relative to the batch folder, for example `../02_<name>/screen_<...>.json`) whose
  phase holds the control's unit; it must have the screen's seeds and budget;
- `method_names`: the name in words of every sweep's `method`;
- `consistency_check` (optional): `run_folder` (the name of a run folder beside this one), `unit` (a unit of
  it), `settings` (setting name -> value: the cell of that unit and of the control that are compared) and
  `reported` (what was written about that cell earlier, in words);
- `predictions` (optional): the module of this package that checks the predictions written in the batch's
  `ideas.md` (for example `predictions_batch_01`). Without it, the predictions of `ideas.md` are quoted and
  marked "not read yet";
- `more_intrinsic_weights` (optional): file names, in the batch folder, of the study descriptions of further units
  that train some of the screen's methods at more intrinsic weights, with the same seeds, budget and other
  settings. Each one's `phase` selects its units, one per method. A method's best weight per metric is chosen
  over every weight of its complete units; a method without such a unit keeps the screen's weights, and the
  tables at every weight leave its cells at the further weights empty. These units need no maze-grid records:
  the property measures are read from the screen's own units, at the best weight among their weights;
- `validation_block` (optional): `phase` (the phase of units that each train one method at one intrinsic weight
  on other seeds, in the screen's environment and reward setting), `control` (the `method` of the control among
  them) and `method_names` (the name in words of each method of this batch among them). The summary then lists,
  per method and weight, W, L and M with the paired differences from the control at the same weight, and after
  that each method at its best weight against the control at the control's best weight, per metric. Units of
  the phase whose method is not in `method_names` belong to another batch: they are named and not read;
- `development_sweeps` (optional): development sweeps of the batch's methods in one or more cases, each against
  the baseline's development sweeps in the same case (keys in `case_sweeps.py`). A batch that only has such
  sweeps is not a screen: it leaves out `study_description`, `control` and `method_names`, and its summary holds
  the measures, this section and the predictions;
- `discovery` (optional): discovery checks in one case, on the validation seeds and on the development seeds
  (keys in `case_sweeps.py`);
- `validation_of_configurations` (optional): every configuration of one phase on the validation seeds in one
  case, each method's best against the control's best for both metrics (keys in `case_sweeps.py`). Unlike
  `validation_block` it also counts units at other settings than the screen's.
- `validation_seeds` (optional): picked configurations on the validation seeds case by case, on a first block of
  seeds (each side at its best there) and, where it was run, on a second block on which nothing was chosen (keys
  in `validation_seeds.py`). A batch whose only runs are such units leaves out the screen's keys and
  `development_sweeps`.
- `same_configuration` (optional): arms compared with other sides at the configurations both trained, on the same
  seeds, also with units of other phases or of another run folder (keys in `same_configuration.py`).
- Predictions: those of the section of `ideas.md` whose heading starts with "## Predictions", then those of each
  section whose heading starts with "## Addendum", in order (see `written_predictions`).

Through the test helper's Slurm job (numpy and plotting never run on a login node): see batch_summary_job.py.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
import importlib
import json
import math
from pathlib import Path
import re

import numpy as np

from . import (case_sweeps, compare, contents, figures, maze_grid, measures, records, report, same_configuration,
               validation_seeds)

# The test helper that runs the job, relative to the platform folder; the summary names the whole command.
SUBMITTER = ('benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_'
             'screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh')
SETTING_KEYS = {'title', 'study_description', 'control', 'control_study_description', 'method_names',
                'consistency_check', 'predictions', 'more_intrinsic_weights', 'validation_block',
                'development_sweeps', 'discovery', 'validation_of_configurations', 'validation_seeds',
                'same_configuration'}
REQUIRED_KEYS = {'title', 'study_description', 'control', 'method_names'}
# What a batch without a screen needs: a title, and its runs in `development_sweeps` or in `validation_seeds`.
REQUIRED_WITHOUT_A_SCREEN = {'title'}
RUNS_WITHOUT_A_SCREEN = {'development_sweeps', 'validation_seeds', 'same_configuration'}
# The keys that only mean something beside a screen.
SCREEN_ONLY_KEYS = {'control', 'control_study_description', 'method_names', 'consistency_check',
                    'more_intrinsic_weights', 'validation_block', 'discovery'}
VALIDATION_KEYS = {'phase', 'control', 'method_names'}
# A table of one value per intrinsic weight holds at most this many weights, so that it fits a printed page.
MOST_WEIGHT_COLUMNS = 6
TABLE_COUNT_WORDS = {1: 'one table', 2: 'two tables', 3: 'three tables', 4: 'four tables'}
# Shares of the budget at which the unvisited cells and the size of the intrinsic reward are read.
UNVISITED_SHARES = (0.05, 0.25, 1.0)
REWARD_SHARES = (0.25, 1.0)
# A median over fewer (policy, cell) pairs than this is not printed.
FEWEST_PAIRS = 20
VERDICT_WORDS = {True: 'held', False: 'did not hold', None: 'cannot be read yet'}


@dataclass(frozen=True, eq=False)
class Method:
    """One method of a screen whose own unit is complete: its configurations over the intrinsic weights, and what was measured."""
    slug: str                  # the method as the study description spells it
    name: str                  # the method in words
    unit_id: str               # its unit of the screen's own study description
    budget: int                # environment steps per policy
    groups: tuple              # its configurations, one per intrinsic weight, in weight order, of all its complete units
    scores: dict               # intrinsic weight -> (W, L) at the budget
    best: records.Configuration             # the configuration with the highest W
    measures: dict             # that configuration's measures at the budget, with 'median_first_goal_step'
    discovery_scores: dict     # intrinsic weight -> (M, share of policies that reach the goal) at the budget
    discovery_best: records.Configuration   # the configuration with the smallest M
    discovery_measures: dict   # that configuration's measures at the budget
    grid: dict                 # the property measures of the bonus on the maze grid (grid_measures), and their 'weight'

    @property
    def best_weight(self):
        """The intrinsic weight of the method's best configuration for W."""
        return weight_of(self.best)

    @property
    def discovery_weight(self):
        """The intrinsic weight of the method's best configuration for M."""
        return weight_of(self.discovery_best)

    @property
    def weights(self):
        """The intrinsic weights the method was trained at, in order."""
        return [weight_of(group) for group in self.groups]


@dataclass(frozen=True, eq=False)
class Screen:
    """Everything read for one batch summary."""
    batch: str
    control: str               # the control's method, as the study description spells it
    order: tuple               # every method of the study description, in its order
    names: dict                # method -> its name in words
    units: tuple               # one row per unit for the units table
    methods: dict              # method -> Method, the methods whose own unit is complete
    differences: dict          # idea -> its paired differences from the control (compare.compare_with_baseline)
    discovery_differences: dict   # idea -> its paired difference in M from the control (compare.compare_discovery)
    budget: int
    seeds: tuple               # the screen's seeds
    validation: dict = None    # the validation block as read (read_validation); None without one
    cases: dict = None         # the development sweeps case by case (case_sweeps.read_development); None without them
    validation_cases: dict = None   # the validation seeds case by case (validation_seeds.read_cases); None without them
    same_configuration: list = None   # arms at the same configuration (same_configuration.read); None without them

    def complete(self, *slugs):
        """Whether every named method's unit is complete."""
        return all(slug in self.methods for slug in slugs)


# ---- reading ----------------------------------------------------------------------------------------------------

def read_settings(folder):
    """The batch folder's summary_settings.json, with its keys checked."""
    path = Path(folder) / 'summary_settings.json'
    settings = json.loads(path.read_text())
    # A batch whose only runs are development sweeps or validation units has no screen and none of the keys that
    # belong to one.
    without_screen = bool(RUNS_WITHOUT_A_SCREEN & set(settings)) and 'study_description' not in settings
    required = REQUIRED_WITHOUT_A_SCREEN if without_screen else REQUIRED_KEYS
    unknown = set(settings) - SETTING_KEYS | (set(settings) & SCREEN_ONLY_KEYS if without_screen else set())
    missing = required - set(settings)
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


def score_weights_for_discovery(groups, budget):
    """A method's configurations with their M and share of arriving policies: (intrinsic weight -> (M, share), the best one)."""
    # before: configurations at the weights 1 and 4 with M 300 and 300 steps, of which 1 and 2 of 2 policies arrive
    # after:  ({1: (300.0, 0.5), 4: (300.0, 1.0)}, the configuration at weight 4): equal M, so the larger share decides.
    ranked = compare.rank_for_discovery(groups, budget)
    return {weight_of(group): (capped, share) for capped, share, group in ranked}, ranked[0][2]


def measured_at(group, budget):
    """Every measure of one configuration at the budget, with the median first-arrival step."""
    return {**measures.configuration_measures(group.policies, budget),
            'median_first_goal_step': measures.median_first_goal_step(group.policies, budget)}


def grid_measures(grid, cell_weights, best_weight, budget):
    """The property measures of one unit's maze-grid records: at the best intrinsic weight, and across the weights."""
    # The policies of each weight. before: copy_cell [0, 0, 1, 1], cell weights [1.0, 4.0]; after: {1.0: [0, 1],
    # 4.0: [2, 3]}.
    members = {weight: np.flatnonzero(grid.copy_cell == index) for index, weight in enumerate(cell_weights)}
    best = members[best_weight]
    # Per policy the start ratio, per policy and cell the decay slope; the medians are taken at the end.
    ratios = maze_grid.start_ratio(grid.start_bonus)
    slopes = maze_grid.decay_slopes(grid.bonus, grid.visits)

    # Unvisited cells, read at the last logging window that ends at or before each share of the budget: their bonus
    # over its start, and their bonus over the bonus of the visited cells at the same window.
    unvisited, apart = {}, {}
    for share in UNVISITED_SHARES:
        window = maze_grid.window_at(grid.steps, share * budget)
        found = maze_grid.unvisited_ratios(grid.bonus[window][best], grid.start_bonus[best], grid.visits[window][best])
        unvisited[share] = {'pairs': len(found), 'step': int(grid.steps[window]),
                            'median': float(np.median(found)) if len(found) >= FEWEST_PAIRS else None}
        ratio, never, seen = maze_grid.unvisited_over_visited(grid.bonus[window][best], grid.visits[window][best], FEWEST_PAIRS)
        apart[share] = {'ratio': ratio, 'unvisited_pairs': never, 'visited_pairs': seen}
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
            'unvisited': unvisited, 'unvisited_over_visited': apart, 'intrinsic_reward': reward, 'divisor': divisor,
            'first_window_step': int(grid.steps[0]),
            'bins': maze_grid.visit_bins(grid.bonus[:, best], grid.visits[:, best]),
            'pairs': int(len(best) * len(grid.cells)), 'problems': maze_grid.problems(grid),
            'largest_bonus': {'start': float(np.nanmax(grid.start_bonus)), 'windows': float(np.nanmax(grid.bonus))}}


def finite_median(values):
    """The median of the finite values; None when there is none."""
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def read_unit(shard, slug, name, budget):
    """One unit's shard, read once: (its row of the units table, its configurations in weight order, its last attempt).

    The configurations are None when the unit is not complete.
    """
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
           'respawned': summary['respawned_episodes'], 'problems': {}, 'largest_bonus': None, 'notes': []}
    if summary['state'] != 'complete':
        return row, None, attempt
    if summary['planned_steps'] != budget:
        raise ValueError(f"Unit {shard.stem} trains {summary['planned_steps']:,} steps per policy; the screen's "
                         f'study description says {budget:,}.')

    # One configuration per intrinsic weight: the unit's cells must differ in nothing else.
    groups = sorted(records.configurations(policies), key=weight_of)
    if records.varying_settings(groups) != ['int_coef']:
        raise ValueError(f'The cells of unit {shard.stem} differ in {records.varying_settings(groups)}; a screen '
                         'summary reads units whose cells differ in the intrinsic weight only.')
    not_finite = sum(int((~np.isfinite(getattr(policy, name_of_sum))).sum())
                     for policy in policies for name_of_sum in ('episodes', 'return_sum'))
    if not_finite:
        row['problems']['episode counts or return sums that are not finite'] = not_finite
    return row, groups, attempt


def listed_unit(run_dir, unit_id, slug, name, budget, row_name=None):
    """One unit for the units table, with its configurations when it is complete; a unit with no shard is not read."""
    shard = run_dir / 'data' / f'{unit_id}.jsonl'
    if unit_id is None or not shard.exists():
        return {'slug': slug, 'name': row_name or name, 'unit_id': unit_id,
                'state': 'no unit was built' if unit_id is None else 'not started (no shard)',
                'queue': 'no unit' if unit_id is None else queue_place(run_dir, unit_id)}, None, None
    row, groups, attempt = read_unit(shard, slug, name, budget)
    return {**row, 'name': row_name or name, 'queue': queue_place(run_dir, unit_id)}, groups, attempt


def join_further_weights(first, further, row):
    """A method's configurations with those of a further unit added, after checking the two differ in the weight only."""
    # before: the screen unit's configurations at the weights [1, 16, 64], a further unit's at [256, 1024]
    # after:  the five configurations in weight order. A setting recorded with two values is refused; a setting
    #         only one of the two units records (an option its code did not have yet) is noted in the unit's row.
    found = setting_differences(first[0], further[0])
    changed = sorted(set(found['different']) - {'int_coef'})
    repeated = sorted({weight_of(group) for group in first} & {weight_of(group) for group in further})
    if changed or repeated:
        raise ValueError(f"Unit {row['unit_id']} cannot extend its method's weights: settings recorded with other "
                         f'values {changed}, intrinsic weights trained twice {repeated}.')
    for key, words in (('only_first', "the method's first unit"), ('only_second', 'this unit')):
        if found[key]:
            row['notes'].append(f'settings that only {words} records: '
                                + ', '.join(f'`{name}` = {value}' for name, value in found[key].items()))
    return sorted(first + further, key=weight_of)


def read_screen(run_dir, batch, settings):
    """Read every unit of the batch's screen and compare each complete idea with the control."""
    # The screen's methods and their units; every method needs a name in words.
    run_dir = Path(run_dir)
    folder = run_dir / 'batches' / batch
    description = json.loads((folder / settings['study_description']).read_text())
    units = batch_units(run_dir, description)
    # A batch that trains no control of its own takes the control's unit from another batch's study description.
    if 'control_study_description' in settings:
        other = json.loads((folder / settings['control_study_description']).read_text())
        found = batch_units(run_dir, other)
        if (settings['control'] in units or settings['control'] not in found
                or (other['steps_per_policy'], other['seeds']) != (description['steps_per_policy'], description['seeds'])):
            raise ValueError(f"summary_settings.json: {settings['control_study_description']} cannot give this batch "
                             f"its control {settings['control']}: the batch trains that method itself, or the other "
                             'study description does not, or it has other seeds or another budget.')
        units = {settings['control']: found[settings['control']], **units}
    unnamed = sorted(set(units) - set(settings['method_names']))
    if unnamed or settings['control'] not in units:
        raise ValueError(f"summary_settings.json: no name in words for the methods {unnamed}, or the control "
                         f"{settings['control']} is not a method of the study description.")
    # One row per unit of the screen's own study description. The maze-grid records of a complete unit are reduced
    # to the property measures at once, at the best weight for W among the unit's own weights, so that the
    # arrays of one unit are not kept.
    budget = description['steps_per_policy']
    rows, found, grids = [], {}, {}
    for slug, unit_id in units.items():
        row, groups, attempt = listed_unit(run_dir, unit_id, slug, settings['method_names'][slug], budget)
        rows.append(row)
        if groups is None:
            continue
        weight = weight_of(score_weights(groups, budget)[1])
        position = attempt[0]['cell_setting_names'].index('int_coef')
        grid = grid_measures(maze_grid.read_grid(attempt), [cell[position] for cell in attempt[0]['cell_settings']],
                             weight, budget)
        row['problems'], row['largest_bonus'] = {**grid['problems'], **row['problems']}, grid['largest_bonus']
        found[slug], grids[slug] = groups, {**grid, 'weight': weight}

    # Further units that train some of the methods at more intrinsic weights, with the same seeds and budget.
    for file_name in settings.get('more_intrinsic_weights', ()):
        further = json.loads((folder / file_name).read_text())
        if (further['steps_per_policy'], further['seeds']) != (budget, description['seeds']):
            raise ValueError(f'{file_name} has other seeds or another budget than the screen: its units cannot '
                             "extend the screen's intrinsic weights.")
        weights = ', '.join(weight_text(weight) for weight in further['cells']['values']['intrinsic_weight'])
        for slug, unit_id in batch_units(run_dir, further).items():
            if slug not in units:
                raise ValueError(f'{file_name} names the method {slug}, which is not a method of the screen.')
            name = settings['method_names'][slug]
            row, groups, _ = listed_unit(run_dir, unit_id, slug, name, budget, f'{name}, at the intrinsic weights {weights}')
            rows.append(row)
            if groups is not None and slug in found:
                found[slug] = join_further_weights(found[slug], groups, row)
            elif groups is not None:
                # Nothing is read from a further unit while the method's own unit of the screen is not complete.
                row['left_out'] = True

    # Per method, the best weight for each metric over every weight it was trained at.
    methods = {}
    for slug, groups in found.items():
        scores, best = score_weights(groups, budget)
        discovery_scores, discovery_best = score_weights_for_discovery(groups, budget)
        measured = measured_at(best, budget)
        methods[slug] = Method(
            slug=slug, name=settings['method_names'][slug], unit_id=units[slug], budget=budget, groups=tuple(groups),
            scores=scores, best=best, measures=measured, discovery_scores=discovery_scores,
            discovery_best=discovery_best, grid=grids[slug],
            discovery_measures=measured if discovery_best is best else measured_at(discovery_best, budget))

    # Every complete idea against the control on the seeds both have: for W and L at the two best weights for W,
    # for M at the two best weights for M.
    differences, discovery_differences = {}, {}
    if settings['control'] in methods:
        control = methods[settings['control']]
        for slug, method in methods.items():
            if slug != settings['control']:
                differences[slug] = compare.compare_with_baseline(method.best.policies, control.best.policies, budget,
                                                                  'batch screen', batch, slug)
                discovery_differences[slug] = compare.compare_discovery(
                    method.discovery_best.policies, control.discovery_best.policies, budget, 'batch screen', batch, slug)
    seeds = tuple(range(description['seeds']['first'], description['seeds']['first'] + description['seeds']['count']))
    return Screen(batch=batch, control=settings['control'], order=tuple(units), names=dict(settings['method_names']),
                  units=tuple(rows), methods=methods, differences=differences,
                  discovery_differences=discovery_differences, budget=budget, seeds=seeds)


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


def discovery_order(screen):
    """The complete methods in the order of the discovery table: the control first, then the ideas from the smallest M up."""
    # Ties in M go to the method of which more policies reach the goal, then to the study description's order.
    ideas = [slug for slug in screen.order if slug in screen.methods and slug != screen.control]
    ideas.sort(key=lambda slug: (screen.methods[slug].discovery_measures['capped_first_goal_step'],
                                 -screen.methods[slug].discovery_measures['reached_goal_share'], screen.order.index(slug)))
    return ([screen.control] if screen.control in screen.methods else []) + ideas


def end_of_grid(weights, weight):
    """'smallest' or 'largest' when a weight is at an end of the weights a method was trained at, else None."""
    # before: ([1, 16, 64], 64), ([1, 16, 64], 16); after: 'largest', None.
    return 'smallest' if weight == weights[0] else 'largest' if weight == weights[-1] else None


def weight_cell(method, weight):
    """A method's best weight as a table cell, marked when it is at an end of the weights the method was trained at."""
    # before: weights [1, 16, 64] and the weight 64; after: '64 (largest of the grid)'.
    end = end_of_grid(method.weights, weight)
    return weight_text(weight) + (f' ({end} of the grid)' if end else '')


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
    # The units left out of the tables by name, and the unit ids under the table. A complete unit with further
    # weights is left out too while its method's own unit of the screen is not complete.
    complete = sum(unit['state'] == 'complete' for unit in screen.units)
    left_out = [unit['name'] for unit in screen.units if unit['state'] != 'complete' or unit.get('left_out')]
    names = ''.join(f"{index}. `{unit['unit_id']}`\n" for index, unit in enumerate(screen.units, 1))
    return ('## Units\n\n'
            f'{complete} of {len(screen.units)} units are complete. Only complete units enter the tables and '
            'figures below; nothing is estimated for the others. '
            + (f"Left out: {'; '.join(left_out)}.\n\n" if left_out else 'No unit is left out.\n\n')
            + report.markdown_table(header, rows) + 'Unit names, by the number in the first column:\n\n' + names + '\n')


def measures_section(screen):
    """The definitions the tables use, each with its formula."""
    count = len(screen.seeds)
    return (
        '## Measures\n\n'
        'Two metrics are reported, each in its own table with its own best intrinsic weight per method: the '
        'whole-curve mean $W$ measures discovery and optimization together, and the mean capped first-goal time $M$ '
        'measures discovery alone. The two are never mixed in one choice.\n\n'
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
        '3. **Best intrinsic weight for the whole-curve mean:** per method, the one of the intrinsic weights it was '
        'trained at with the highest $W$; ties go to the higher $L$.\n'
        '4. **Difference from the control:** define $d_i$ as the value of seed $i$ under the method at its best '
        'weight for $W$ minus its value under the control at the control\'s own best weight for $W$. The tables give '
        '$\\bar d = \\dfrac{1}{K}\\sum_{i=1}^{K} d_i$ for $W$ and for $L$, each with a 95% interval: the 2.5% and '
        '97.5% points of the means of 10,000 resamples of the $K$ seeds, drawn with replacement. An idea is ahead '
        'of the control when the interval is entirely above zero, behind when it is entirely below zero, and '
        'within the interval otherwise.\n'
        '5. **Median first-arrival step:** define $T_i$ as the training step of policy $i$\'s first arrival at the '
        'goal ($T_i = \\infty$ if it has none by step $B$). The measure is the median of $T_i$ over the $K$ '
        'policies; it reads "none" when the median falls on a policy with no arrival.\n'
        '6. **Policies that reach the goal:** the number of the $K$ policies with $T_i \\le B$.\n'
        '7. **Mean capped first-goal time, the secondary metric:** $M(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\min(T_i, B)$, '
        'the mean over the policies of the training step of the first arrival at the goal, a policy with no arrival '
        'by step $B$ counting as $B$. Smaller is better. The tables give $M$ in thousands of steps.\n'
        '8. **Best intrinsic weight for the mean capped first-goal time:** per method, the one of the intrinsic '
        'weights it was trained at with the smallest $M$; ties go to the weight at which more policies reach the '
        'goal. It can differ from the best weight of item 3.\n'
        '9. **Difference in the mean capped first-goal time from the control:** as item 4, with $d_i$ the value of '
        '$\\min(T_i, B)$ of seed $i$ under the method at its best weight for $M$ minus the same under the control at '
        'the control\'s own best weight for $M$. A negative $\\bar d$ means the goal was first reached earlier than '
        'under the control. An idea is ahead of the control on discovery when the 95% interval is entirely below '
        'zero, behind when it is entirely above zero, and within the interval otherwise.\n\n'
        'Standard errors are over the $K$ policies.\n\n')


def screen_section(screen, order):
    """The table of the primary metric: each method at its best weight for W, the control first, the ideas sorted by W."""
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
    header = ['method', 'best intrinsic weight for $W$', 'whole-curve mean $W$ ↑ ± standard error',
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
    return ('## Discovery and optimization: whole-curve mean\n\n'
            'Define $W$, $L$ and the other columns as in the section Measures. One row per method at its best '
            f'intrinsic weight for $W$, read at {screen.budget:,} environment steps per policy. The control is the '
            'first row; the ideas follow, sorted by $W$. Per column, the best value is bold and the second best '
            f'underlined; for the median first-arrival step the smallest value is the best. {against} The seeds are '
            'the development seeds, on which each best weight was also chosen: this table is a screen and not a '
            'verdict.\n\n' + report.markdown_table(header, rows))


def weight_tables(table, value_of, text_of, higher_better, best_of, last_header):
    """One value per method and intrinsic weight, in tables that fit a printed page.

    Returns the tables, the number of weights, the number of tables, and the methods whose best weight is the
    smallest or the largest weight they were trained at.
    """
    # The columns are every weight any method was trained at; a method without a weight has an empty cell there.
    # before: methods trained at [1, 16, 64] and at [1, 16, 64, 256], at most 2 weights per table
    # after:  two tables, with the weights [1, 16] and [64, 256]; the first method's cell under 256 is empty.
    weights = sorted({weight for method in table for weight in method.weights})
    count = math.ceil(len(weights) / MOST_WEIGHT_COLUMNS)
    size = math.ceil(len(weights) / count)
    text = []
    for first in range(0, len(weights), size):
        part = weights[first:first + size]
        # Per weight, the best and the second best among the methods trained at it.
        marks = {}
        for weight in part:
            trained = [index for index, method in enumerate(table) if weight in method.weights]
            values = [value_of(table[index], weight) for index in trained]
            found = report.column_marks(values, [text_of(value) for value in values], higher_better)
            marks[weight] = dict(zip(trained, found))
        rows = [[method.name]
                + [report.styled(text_of(value_of(method, weight)), marks[weight][index]) if weight in method.weights
                   else '' for weight in part]
                + [weight_cell(method, best_of(method))] for index, method in enumerate(table)]
        text.append(report.markdown_table(
            ['method'] + [f'weight {weight_text(weight)}' for weight in part] + [last_header], rows))
    at_end = [f'{method.name} ({weight_text(best_of(method))}, the {end_of_grid(method.weights, best_of(method))})'
              for method in table if end_of_grid(method.weights, best_of(method))]
    return ''.join(text), len(weights), count, at_end


def split_words(weights, count):
    """How the weights are spread over the tables, in words."""
    return (f'The {weights} weights are in one table.' if count == 1 else
            f'The {weights} weights are split over {TABLE_COUNT_WORDS.get(count, f"{count} tables")} so that each '
            'fits the width of a printed page.')


def weights_section(screen, order):
    """W of every method at every intrinsic weight it was trained at."""
    table = [screen.methods[slug] for slug in order]
    tables, weights, count, at_end = weight_tables(
        table, lambda method, weight: method.scores[weight][0], lambda value: f'{value:.3f}', True,
        lambda method: method.best_weight, 'best intrinsic weight for $W$')
    return ('## Whole-curve mean at every intrinsic weight\n\n'
            'Define $W$ as in the section Measures. Rows are in the order of the table of the whole-curve mean. '
            f'{split_words(weights, count)} A method that was not trained at a weight has an empty cell there. Per '
            'weight (column), the highest $W$ among the methods is bold and the second highest underlined. The last '
            'column names each method\'s best weight for $W$ and says when it is the smallest or the largest of the '
            'weights the method was trained at.\n\n' + tables
            + (('Best weight at an end of the grid: ' + '; '.join(at_end) + '. The protocol\'s edge rule '
                '(`program.md`) asks for three more values on that side before anything is confirmed.\n\n')
               if at_end else 'No method has its best weight at an end of the grid.\n\n'))


def discovery_section(screen, order):
    """The table of the secondary metric: each method at its best weight for M, the control first, the ideas sorted by M."""
    table = [screen.methods[slug] for slug in order]
    results = [screen.discovery_differences.get(slug) for slug in order]
    compared = [result if result and result['seeds'] >= 2 else None for result in results]

    # The marks of every metric column: the smallest is the best for M and for its difference, the largest for
    # the policies that reach the goal and for W and L; the control has no difference and takes no mark there.
    capped = [method.discovery_measures['capped_first_goal_step'] for method in table]
    capped_marks = report.column_marks(capped, [report.thousands(value) for value in capped], False)
    means = [result['capped_first_goal_step']['mean'] if result else None for result in compared]
    mean_marks = report.column_marks(means, [None if mean is None else report.thousands(mean, True) for mean in means], False)
    reached = [round(method.discovery_measures['reached_goal_share'] * method.discovery_measures['seeds']) for method in table]
    reached_marks = report.column_marks(reached, [str(value) for value in reached], True)
    whole = report.mean_marks([method.discovery_measures['whole_curve_mean'] for method in table])
    final = report.mean_marks([method.discovery_measures['final_level'] for method in table])

    # One row per method; the control's difference cell says that it is the control.
    header = ['method', 'best intrinsic weight for $M$', '$M$ ↓ ± standard error, thousand steps',
              'difference in $M$ from the control [95% interval], thousand steps', 'policies that reach the goal',
              '$W$ at that weight ± standard error', '$L$ at that weight']
    rows = []
    for index, (slug, method, result) in enumerate(zip(order, table, compared)):
        row = method.discovery_measures
        difference = ('control' if slug == screen.control else 'not compared' if result is None else
                      report.discovery_interval_text(result['capped_first_goal_step'], mean_marks[index]))
        rows.append([method.name, weight_cell(method, method.discovery_weight),
                     report.thousands_with_error(row['capped_first_goal_step'], row['capped_first_goal_step_error'],
                                                 capped_marks[index]),
                     difference, report.styled(f"{reached[index]} of {row['seeds']}", reached_marks[index]),
                     report.marked_mean(row['whole_curve_mean'], row['whole_curve_mean_error'], whole[index]),
                     report.styled(f"{row['final_level']:.3f}", final[index])])
    control = screen.methods.get(screen.control)
    against = (f'Differences are the method minus the control at the control\'s own best intrinsic weight for $M$ '
               f'({weight_text(control.discovery_weight)}), on the {len(screen.seeds)} seeds both have; the brackets '
               'hold the 95% interval, and a negative difference means the goal was first reached earlier than under '
               'the control.' if control else
               'The control\'s unit is not complete, so no difference from the control can be given yet.')
    return ('## Discovery: mean capped first-goal time\n\n'
            'Define $M$, $W$ and $L$ as in the section Measures. One row per method at its best intrinsic weight for '
            f'$M$ (the smallest $M$; ties go to the weight at which more policies reach the goal), read at '
            f'{screen.budget:,} environment steps per policy. This weight can differ from the best weight for $W$ in '
            'the table of the whole-curve mean. $M$, its standard error and its differences are in thousands of '
            'steps. The control is the first row; the ideas follow, sorted by $M$ from the smallest. Per column, the '
            'best value is bold and the second best underlined: the smallest for $M$ and for its difference, the '
            'largest for the other columns. The columns $W$ and $L$ are read at the same weight as $M$, so they show '
            f'what the configuration that is best for discovery gives on the primary metric. {against} A best '
            'weight marked "largest of the grid" or "smallest of the grid" is at an end of the weights the method '
            'was trained at: larger or smaller weights must be tried before anything is concluded from that row. '
            'The seeds are the development seeds, on which each best weight was also chosen: this table is a screen '
            'and not a verdict.\n\n' + report.markdown_table(header, rows))


def discovery_weights_section(screen, order):
    """M of every method at every intrinsic weight it was trained at."""
    table = [screen.methods[slug] for slug in order]
    tables, weights, count, at_end = weight_tables(
        table, lambda method, weight: method.discovery_scores[weight][0], report.thousands, False,
        lambda method: method.discovery_weight, 'best intrinsic weight for $M$')
    return ('## Mean capped first-goal time at every intrinsic weight\n\n'
            'Define $M$ as in the section Measures; the cells give it in thousands of steps. Rows are in the order '
            f'of the table of the mean capped first-goal time. {split_words(weights, count)} A method that was not '
            'trained at a weight has an empty cell there. Per weight (column), the smallest $M$ among the methods is '
            'bold and the second smallest underlined. The last column names each method\'s best weight for $M$ and '
            'says when it is the smallest or the largest of the weights the method was trained at.\n\n' + tables
            + (('Best weight for $M$ at an end of the grid: ' + '; '.join(at_end) + '. Larger or smaller weights must '
                'be tried on that side before anything is concluded from these rows (the edge rule of '
                '`program.md`).\n\n')
               if at_end else 'No method has its best weight for $M$ at an end of the grid.\n\n'))


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
    # The maze-grid records belong to the screen's own units, so a best weight of a further unit cannot be read here.
    moved = [method.name for method in table if method.grid['weight'] != method.best_weight]
    grid_note = ('In this section "the best weight" of a method is its best weight for $W$ among the weights of '
                 'its own unit of the screen, which records the bonus on the maze grid; the units with further '
                 'weights have no such records. It differs from the best weight for $W$ over all weights for: '
                 + '; '.join(moved) + '.\n\n' if moved else '')
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
        + grid_note +
        'Rows are in the order of the table of the whole-curve mean. These columns describe the bonus and have no better or worse '
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
    rows = [[method.name, weight_text(method.grid['weight']),
             f"{short(method.grid['divisor']['start'])} and {short(method.grid['divisor']['first_window'])}"]
            + [short(method.grid['intrinsic_reward'][key]) for key in ('start', 'first_window') + REWARD_SHARES]
            for method in table]
    text.append(report.markdown_table(header, rows))
    text.append('### Bonus against cumulative visits\n\n'
                'One panel per method in the order of the table of the whole-curve mean, at the best intrinsic '
                'weight of this section. Each '
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
            'Measures. One panel per idea in the order of the table of the whole-curve mean: the idea at its best '
            f'intrinsic weight for $W$ in blue, {grey}; bands are one standard error over the {len(screen.seeds)} '
            'seeds.\n\n'
            f'![training curves](plots/{name}.png)\n\n')


def side_lists(screen, order, results, quantity, text_of, ahead, behind):
    """The ideas under the side of one difference's interval: ahead of the control, within the interval, behind, waiting."""
    # before: results {'even-start': {quantity: {'side': 'ahead', ...}, 'seeds': 16}, ...}
    # after:  four list items; `ahead` and `behind` are (label, rule) of the two sides, as their sign depends on the metric.
    lists = {'ahead': [], 'level': [], 'behind': []}
    for slug in order:
        result = results.get(slug)
        if result and result['seeds'] >= 2:
            lists[result[quantity]['side']].append(f'{screen.methods[slug].name} ({text_of(result[quantity])})')
    waiting = [screen.names[slug] for slug in screen.order if slug not in screen.methods]
    rows = [(*ahead, lists['ahead']), ('Within the interval', 'it contains zero', lists['level']),
            (*behind, lists['behind']), ('Not compared yet', 'unit not complete', waiting)]
    return ''.join(f"- **{label}** ({rule}): {'; '.join(names) if names else 'none'}.\n" for label, rule, names in rows) + '\n'


def sides_section(screen, order, discovery):
    """Which ideas are ahead of the control, within the interval, or behind: by the difference in W, then by the difference in M."""
    if screen.control not in screen.methods:
        return '## Ideas against the control\n\nThe control\'s unit is not complete, so no idea can be compared yet.\n\n'
    return ('## Ideas against the control\n\n'
            'Two readings, one per metric, each with the methods at their own best weights for that metric. Both '
            'are on the development seeds, on which the best weights were also chosen: a screen, not a verdict.\n\n'
            '### Discovery and optimization: whole-curve mean\n\n'
            'By the 95% interval of the difference in $W$ from the control (section Measures, item 4), on the '
            'development seeds. After each name: the difference in $W$ and its interval.\n\n'
            + side_lists(screen, order, screen.differences, 'whole_curve_mean',
                         lambda result: report.interval_text(result).rsplit(',', 1)[0],
                         ('Ahead of the control', 'interval entirely above zero'),
                         ('Behind the control', 'interval entirely below zero'))
            + '### Discovery: mean capped first-goal time\n\n'
            'By the 95% interval of the difference in $M$ from the control (section Measures, item 9), on the '
            'development seeds. After each name: the difference in $M$ and its interval, in thousands of steps; a '
            'negative difference means the goal was first reached earlier than under the control.\n\n'
            + side_lists(screen, discovery, screen.discovery_differences, 'capped_first_goal_step',
                         report.discovery_interval_text,
                         ('Ahead of the control on discovery', 'interval entirely below zero'),
                         ('Behind the control on discovery', 'interval entirely above zero')))


# ---- predictions --------------------------------------------------------------------------------------------------

def numbered_items(body):
    """The numbered items of a piece of Markdown, each as one line of text."""
    # before: '\n1. **Even start alone** changes ...\n   above 2 ...\n2. ...'; after: ['**Even start alone** changes ... above 2 ...', '...'].
    return [' '.join(item.split()) for item in re.split(r'(?m)^\d+\. ', body)[1:]]


def addendum_predictions(body):
    """The predictions of one addendum of ideas.md: the numbered items after its line that starts "**Predictions",
    or else its paragraph that starts "Prediction:"; none when it has neither."""
    # before: '... 1. At the switch ... **Predictions of the addendum.**\n\n6. Under reading (a) ...\n7. ...'
    # after:  ['Under reading (a) ...', '...'] (the numbered items before that line are findings, not predictions).
    # before: '... Queued: ...\n\nPrediction: the fast last layer is ahead ...\ncost.'; after: ['the fast last layer is ahead ... cost.'].
    listed = re.split(r'(?m)^\*\*Predictions[^\n]*\n', body)
    if len(listed) == 2:
        return numbered_items(listed[1])
    single = re.search(r'(?ms)^Prediction: (.*?)(?:\n\n|\Z)', body)
    return [' '.join(single.group(1).split())] if single else []


def written_predictions(ideas_text):
    """The predictions of a batch's ideas.md, each as one line of text: those of its section "## Predictions", then
    those of each section "## Addendum ..." in order."""
    # before: '## Predictions (written before the runs)\n\n1. ...\n2. ...\n## Addendum 1 ...\n**Predictions of the addendum.**\n\n3. ...'
    # after:  the texts of items 1, 2 and 3.
    section = re.split(r'(?m)^## Predictions[^\n]*\n', ideas_text)
    if len(section) != 2:
        raise ValueError('ideas.md has no single section whose heading starts with "## Predictions".')
    found = numbered_items(re.split(r'(?m)^## ', section[1])[0])
    for addendum in re.split(r'(?m)^## Addendum[^\n]*\n', ideas_text)[1:]:
        found += addendum_predictions(re.split(r'(?m)^## ', addendum)[0])
    return found


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


def predictions_section(screen, ideas_text, module_name=None):
    """Every written prediction with what was observed, part by part; without a module each is quoted and not read yet."""
    written = written_predictions(ideas_text)
    if module_name is None:
        return ('## Predictions\n\n'
                'The predictions were written in `ideas.md` before the runs. Their readings from the records have not '
                'been written yet (this batch has no predictions module), so each is quoted and marked not read yet. '
                'The tables of this summary hold the numbers the predictions speak of.\n\n'
                + ''.join(f'### Prediction {number}\n\n> {quoted}\n\nObserved: **not read yet**.\n\n'
                          for number, quoted in enumerate(written, 1)))
    module = importlib.import_module(f'{__package__}.{module_name}')
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


# ---- validation block ---------------------------------------------------------------------------------------------

def screen_other_settings(specifications, description):
    """The values every setting but the intrinsic weight takes in the screen's units; None when no unit names its cells."""
    # before: a screen unit with the cell names ['learning_rate', 'ent_coef', 'int_coef'] and the cells
    #         [[0.002, 0.01, 1.0], [0.002, 0.01, 2.0]]; after: {'learning_rate': {0.002}, 'ent_coef': {0.01}}.
    for specification in specifications:
        if specification.get('phase') == description['phase'] and 'cells' in specification:
            return {name: {cell[index] for cell in specification['cells']}
                    for index, name in enumerate(specification['cell_setting_names']) if name != 'int_coef'}
    return None


def other_settings_words(specification, fixed):
    """The settings of a unit's cells that the screen does not have, in words; empty when every cell is the screen's."""
    # before: a unit with the cell [0.001, 0.01, 2.0] against a screen whose learning rate is 0.002
    # after:  'learning rate 0.001'.
    found = {}
    for index, name in enumerate(specification['cell_setting_names']):
        if name != 'int_coef' and name in fixed:
            found.update({(name, cell[index]): None for cell in specification['cells'] if cell[index] not in fixed[name]})
    return ', '.join(f"{records.SETTING_WORDS.get(name, name.replace('_', ' '))} {value:g}" for name, value in found)


def validation_units(run_dir, description, block):
    """The validation block's complete units as {(method, intrinsic weight): configuration}, and what is left out.

    Left out: the block's units that are not complete, in words; the methods of the phase that the block does not
    name (they belong to another batch); and the units whose settings other than the intrinsic weight are not the
    screen's (they belong to another reading: the block compares at the screen's settings).
    """
    # before: specifications of the block's phase for the control at the weights 1 and 2 (complete), for an idea
    #         at the weight 2 (still training), for a method of another batch, and for the control at another
    #         learning rate, all in the screen's environment
    # after:  ({('control', 1.0): ..., ('control', 2.0): ...}, ['the idea, intrinsic weight 2: unfinished, ...'],
    #          ['the method of another batch'], ['control, intrinsic weight 2 (learning rate 0.001)']).
    specifications = [json.loads(path.read_text()) for path in sorted((Path(run_dir) / 'specifications').glob('*.json'))]
    fixed = screen_other_settings(specifications, description)
    groups, waiting, others, elsewhere = {}, [], [], []
    for specification in specifications:
        if (specification.get('phase') != block['phase']
                or specification['environment'] not in description['environments']
                or specification['reward_setting'] not in description['reward_settings']):
            continue
        slug = specification['method']
        if slug not in block['method_names']:
            others.append(slug.replace('-', ' '))
            continue
        name = block['method_names'][slug]
        position = specification['cell_setting_names'].index('int_coef')
        words = f"{name}, intrinsic weight {', '.join(weight_text(cell[position]) for cell in specification['cells'])}"
        apart = other_settings_words(specification, fixed) if fixed is not None else ''
        if apart:
            elsewhere.append(f'{words} ({apart})')
            continue
        shard = Path(run_dir) / 'data' / f"{specification['unit_id']}.jsonl"
        if not shard.exists():
            waiting.append(f'{words}: not started (no shard)')
            continue
        summary, policies = records.read_unit(shard, [{'name': name, 'unit_id_pattern': shard.stem}])
        if summary['state'] != 'complete':
            waiting.append(f"{words}: {summary['state']}, {summary['windows']} logging windows written")
            continue
        for group in records.configurations(policies):
            if (slug, weight_of(group)) in groups:
                raise ValueError(f"Two units of the phase {block['phase']} train {slug} at the intrinsic weight "
                                 f'{weight_of(group):g}; a validation block reads one unit per method and weight.')
            groups[(slug, weight_of(group))] = group
    return groups, waiting, sorted(set(others)), elsewhere


def validation_best(groups, block, budget, batch):
    """Each method of the validation block at its best intrinsic weight per metric, against the control at its own.

    Returns {'whole': rows, 'discovery': rows}. One row per method with a complete unit: the control first, then
    the others from the best value on. A row holds the method, its best weight for the metric, the weights it
    was run at, its measures at the best weight, and its paired differences from the control at the control's
    own best weight for the same metric (None for the control, and when the control has no complete unit).
    """
    # The configurations of each method in weight order.
    # before: {('control', 2.0): b, ('control', 1.0): a, ('idea', 2.0): c}; after: {'control': [a, b], 'idea': [c]}.
    by_method = defaultdict(list)
    for (slug, _), group in sorted(groups.items(), key=lambda item: item[0][1]):
        by_method[slug].append(group)
    order = list(block['method_names'])
    tables = {}
    for metric, rank, value in (('whole', compare.rank_configurations, lambda row: -row['measures']['whole_curve_mean']),
                                ('discovery', compare.rank_for_discovery, lambda row: row['measures']['capped_first_goal_step'])):
        # W: the highest whole-curve mean (ties: the higher final level); M: the smallest (ties: more policies arrive).
        best = {slug: rank(found, budget)[0][2] for slug, found in by_method.items()}
        control = best.get(block['control'])
        rows = []
        for slug, group in best.items():
            result = None
            if control is not None and slug != block['control']:
                name = ('batch validation block at the best weights', batch, slug, metric)
                result = {**compare.compare_with_baseline(group.policies, control.policies, budget, *name),
                          **compare.compare_discovery(group.policies, control.policies, budget, *name)}
            rows.append({'slug': slug, 'weight': weight_of(group), 'weights': [weight_of(member) for member in by_method[slug]],
                         'measures': measures.configuration_measures(group.policies, budget), 'result': result})
        tables[metric] = sorted(rows, key=lambda row: (row['slug'] != block['control'], value(row), order.index(row['slug'])))
    return tables


def chosen_weight_cell(row):
    """A best weight with the weights it was chosen among, as a table cell."""
    # before: weight 0.5 of the weights [0.25, 0.5, 1.0]; after: '0.5 (of 0.25, 0.5, 1)'. One weight: '2 (its only weight)'.
    if len(row['weights']) == 1:
        return f"{weight_text(row['weight'])} (its only weight)"
    return f"{weight_text(row['weight'])} (of {', '.join(weight_text(weight) for weight in row['weights'])})"


def fine_difference(found, digits, style=None):
    """A paired difference with its 95% interval at a chosen number of decimals; the mean bold or underlined."""
    # before: ({'mean': 0.0137, 'interval': [0.0072, 0.0203]}, 4, 'best'); after: '**+0.0137** [+0.0072, +0.0203]'.
    low, high = found['interval']
    return f"{report.styled(format(found['mean'], f'+.{digits}f'), style)} [{low:+.{digits}f}, {high:+.{digits}f}]"


def fine_mean(value, error, digits, style=None):
    """A mean with its standard error at a chosen number of decimals; the mean alone with one seed."""
    # before: (0.8917, 0.0031, 4, 'best'); after: '**0.8917** ± 0.0031'.
    return report.styled(f'{value:.{digits}f}', style) + ('' if error is None else f' ± {error:.{digits}f}')


def validation_best_text(tables, block):
    """The two tables of the validation block that compare best weight with best weight, one per metric."""
    names, control = block['method_names'], block['control']
    # Differences of a few thousandths need four decimals once the seeds are many; a small block keeps three.
    digits = 4 if max(row['measures']['seeds'] for row in tables['whole']) >= 100 else 3
    compared = {metric: [row['result'] if row['result'] and row['result']['seeds'] >= 2 else None for row in rows]
                for metric, rows in tables.items()}

    def cell(row, found, text):
        """A difference cell: the control's word, why there is none, or the difference itself."""
        if row['slug'] == control:
            return 'control'
        if row['result'] is None:
            return 'no complete control unit'
        return text if found else 'fewer than two seeds in common'

    def common(row, found):
        """The number of seeds a row shares with the control."""
        return str(row['measures']['seeds']) if row['slug'] == control else str(found['seeds']) if found else 'not compared'

    def marks(values, text_of, higher_better):
        """The marks of one column; a row without a value takes none."""
        return report.column_marks(values, [None if value is None else text_of(value) for value in values], higher_better)

    # The table of the primary metric, at the best weights for W.
    rows, found = tables['whole'], compared['whole']
    number = lambda value: f'{value:.{digits}f}'
    signed = lambda value: format(value, f'+.{digits}f')
    whole = marks([row['measures']['whole_curve_mean'] for row in rows], number, True)
    final = marks([row['measures']['final_level'] for row in rows], number, True)
    differences = {quantity: marks([result[quantity]['mean'] if result else None for result in found], signed, True)
                   for quantity in ('whole_curve_mean', 'final_level')}
    capped = marks([result['capped_first_goal_step']['mean'] if result else None for result in found],
                   lambda value: report.thousands(value, True), False)
    first = []
    for index, (row, result) in enumerate(zip(rows, found)):
        measured = row['measures']
        first.append([
            names[row['slug']], chosen_weight_cell(row),
            fine_mean(measured['whole_curve_mean'], measured['whole_curve_mean_error'], digits, whole[index]),
            cell(row, result, result and fine_difference(result['whole_curve_mean'], digits, differences['whole_curve_mean'][index])),
            fine_mean(measured['final_level'], measured['final_level_error'], digits, final[index]),
            cell(row, result, result and fine_difference(result['final_level'], digits, differences['final_level'][index])),
            cell(row, result, result and report.discovery_interval_text(result['capped_first_goal_step'], capped[index])),
            common(row, result)])

    # The table of the secondary metric, at the best weights for M.
    rows, found = tables['discovery'], compared['discovery']
    capped = marks([row['measures']['capped_first_goal_step'] for row in rows], report.thousands, False)
    differences = marks([result['capped_first_goal_step']['mean'] if result else None for result in found],
                        lambda value: report.thousands(value, True), False)
    reached = [round(row['measures']['reached_goal_share'] * row['measures']['seeds']) for row in rows]
    reached_marks = report.column_marks([row['measures']['reached_goal_share'] for row in rows],
                                        [f"{count} of {row['measures']['seeds']}" for count, row in zip(reached, rows)], True)
    whole = marks([row['measures']['whole_curve_mean'] for row in rows], number, True)
    final = marks([row['measures']['final_level'] for row in rows], number, True)
    second = []
    for index, (row, result) in enumerate(zip(rows, found)):
        measured = row['measures']
        second.append([
            names[row['slug']], chosen_weight_cell(row),
            report.thousands_with_error(measured['capped_first_goal_step'], measured['capped_first_goal_step_error'], capped[index]),
            cell(row, result, result and report.discovery_interval_text(result['capped_first_goal_step'], differences[index])),
            report.styled(f"{reached[index]} of {measured['seeds']}", reached_marks[index]),
            report.styled(number(measured['whole_curve_mean']), whole[index]),
            report.styled(number(measured['final_level']), final[index]), common(row, result)])
    return ('### Each method at its best intrinsic weight against the control at its best weight\n\n'
            'The two tables above compare at equal intrinsic weights. A method and the control need not have the '
            'same best weight, so here each method stands at its own best weight for the metric and the control at '
            'its own: for $W$ the weight with the highest $W$ (ties: the higher $L$), for $M$ the weight with the '
            'smallest $M$ (ties: more policies reach the goal). The weight column names the weights each was chosen '
            'among. Each best weight was chosen on these same seeds, among the weights the method was run at in '
            'this block, and the control\'s was chosen the same way: a side that was run at more weights had more '
            'chances, so these differences lean slightly towards it. They are a second selection step, not the '
            'comparison on the confirmation seeds. Differences are the method minus the control on the seeds both '
            'have, with the 95% interval from 10,000 resamples of those seeds. The control is the first row; the '
            'other methods follow from the best value of the sorted column on. Per column the best value is bold '
            'and the second best underlined.\n\n'
            '**Whole-curve mean, each side at its best weight for it.** The last difference column gives the mean '
            'capped first-goal time at these same weights, in thousands of steps (negative: the goal was first '
            'reached earlier than under the control).\n\n'
            + report.markdown_table(['method', 'best intrinsic weight for $W$', '$W$ ↑ ± standard error',
                                     'difference in $W$ from the control [95% interval]', '$L$ ± standard error',
                                     'difference in $L$ from the control [95% interval]',
                                     'difference in $M$ at these weights [95% interval], thousand steps',
                                     'seeds in common with the control'], first)
            + '**Mean capped first-goal time, each side at its best weight for it.** $M$, its standard error and '
            'its difference are in thousands of steps; $W$ and $L$ are read at the same weight as $M$.\n\n'
            + report.markdown_table(['method', 'best intrinsic weight for $M$', '$M$ ↓ ± standard error, thousand steps',
                                     'difference in $M$ from the control [95% interval], thousand steps',
                                     'policies that reach the goal', '$W$ at that weight', '$L$ at that weight',
                                     'seeds in common with the control'], second))


def read_validation(run_dir, description, block, batch):
    """The validation block as read: its complete configurations, what is left out, its budget, and the best weights."""
    unknown, missing = set(block) - VALIDATION_KEYS, VALIDATION_KEYS - set(block)
    if unknown or missing:
        raise ValueError(f'summary_settings.json, validation_block: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}.')
    groups, waiting, others, elsewhere = validation_units(run_dir, description, block)
    budgets = sorted({group.planned_steps for group in groups.values()})
    if len(budgets) > 1:
        raise ValueError(f"The units of the phase {block['phase']} train {budgets} steps per policy; one length is needed.")
    budget = budgets[0] if budgets else None
    return {'groups': groups, 'waiting': waiting, 'others': others, 'elsewhere': elsewhere, 'budget': budget, 'batch': batch,
            'best': validation_best(groups, block, budget, batch) if groups else None}


def validation_section(read, block):
    """The validation block as read (read_validation): per method and weight W, L and M with the paired differences
    from the control at the same weight, then each method at its best weight against the control at its own."""
    title = '## Validation block\n\n'
    groups, waiting, others, budget = read['groups'], read['waiting'], read['others'], read['budget']
    left_out = ('Not complete, left out:\n\n' + ''.join(f'- {words}.\n' for words in waiting) + '\n') if waiting else ''
    if others:
        left_out += ('The phase also holds units of methods that this batch does not name; they are not read here: '
                     + '; '.join(others) + '.\n\n')
    if read['elsewhere']:
        left_out += ('The phase also holds units of these methods at settings other than the screen\'s, beside the '
                     'intrinsic weight. This block compares at the screen\'s settings, so they are not read here: '
                     + '; '.join(read['elsewhere']) + '.\n\n')
    if not groups:
        return title + f"No unit of the phase `{block['phase']}` is complete yet.\n\n" + left_out
    seeds = sorted({policy.seed for group in groups.values() for policy in group.policies})

    # The control first, then the other methods in the order of the settings; each method by intrinsic weight.
    slugs = [block['control']] + [slug for slug in block['method_names'] if slug != block['control']]
    keys = sorted(groups, key=lambda key: (slugs.index(key[0]), key[1]))
    table = [measures.configuration_measures(groups[key].policies, budget) for key in keys]
    # Each row against the control at the same weight, on the seeds both have: W and L, then M.
    results = []
    for slug, weight in keys:
        control = groups.get((block['control'], weight))
        if slug == block['control']:
            results.append('control')
        elif control is None:
            results.append('no complete control unit at this weight')
        else:
            name = ('batch validation block', read['batch'], slug, weight)
            results.append({**compare.compare_with_baseline(groups[(slug, weight)].policies, control.policies, budget, *name),
                            **compare.compare_discovery(groups[(slug, weight)].policies, control.policies, budget, *name)})
    compared = [result if isinstance(result, dict) and result['seeds'] >= 2 else None for result in results]

    # The marks of every metric column over all rows; a row with no difference takes no mark there.
    marks = {name: report.mean_marks([row[name] for row in table]) for name in ('whole_curve_mean', 'final_level')}
    for quantity in ('whole_curve_mean', 'final_level'):
        values = [result[quantity]['mean'] if result else None for result in compared]
        marks[f'{quantity} difference'] = report.column_marks(
            values, [None if value is None else f'{value:+.3f}' for value in values], True)
    capped = [row['capped_first_goal_step'] for row in table]
    marks['capped'] = report.column_marks(capped, [report.thousands(value) for value in capped], False)
    values = [result['capped_first_goal_step']['mean'] if result else None for result in compared]
    marks['capped difference'] = report.column_marks(
        values, [None if value is None else report.thousands(value, True) for value in values], False)
    reached = [round(row['reached_goal_share'] * row['seeds']) for row in table]
    marks['reached'] = report.column_marks([row['reached_goal_share'] for row in table],
                                           [f"{count} of {row['seeds']}" for count, row in zip(reached, table)], True)

    # Two tables with the same rows: the primary metric with the final level, then the secondary metric.
    first, second = [], []
    for index, ((slug, weight), row, result, found) in enumerate(zip(keys, table, results, compared)):
        lead = [block['method_names'][slug], weight_text(weight)]
        words = result if isinstance(result, str) else 'fewer than two seeds in common'
        common = str(row['seeds']) if slug == block['control'] else str(found['seeds']) if found else 'not compared'
        differences = {quantity: difference_cell(found, quantity, marks[f'{quantity} difference'][index]) if found else words
                       for quantity in ('whole_curve_mean', 'final_level')}
        first.append(lead + [report.marked_mean(row['whole_curve_mean'], row['whole_curve_mean_error'], marks['whole_curve_mean'][index]),
                             differences['whole_curve_mean'],
                             report.marked_mean(row['final_level'], row['final_level_error'], marks['final_level'][index]),
                             differences['final_level'], common])
        second.append(lead + [report.thousands_with_error(capped[index], row['capped_first_goal_step_error'], marks['capped'][index]),
                              report.discovery_interval_text(found['capped_first_goal_step'], marks['capped difference'][index])
                              if found else words,
                              report.styled(f"{reached[index]} of {row['seeds']}", marks['reached'][index]), common])
    total = len(keys) + len(waiting)
    return (title
            + f"The units of the phase `{block['phase']}` in the screen's environment and reward setting. Each trains "
            f'one method at one intrinsic weight on {len(seeds)} seeds, from {seeds[0]} to {seeds[-1]}: these are '
            'neither the development seeds of the screen nor the confirmation seeds. '
            f'{len(keys)} of {total} method and weight pairs have a complete unit; a unit that is not complete is '
            'listed under the tables and left out. Define $W$, $L$ and $M$ as in the section Measures, here with $K$ '
            f'the number of seeds of a row and read at {budget:,} environment steps per policy. Differences are the '
            'method minus the control at the same intrinsic weight, on the seeds both have, each with its 95% '
            'interval from 10,000 resamples of those seeds. Rows: the control first, then the other methods, each '
            'by intrinsic weight. Per column the best value is bold and the second best underlined.\n\n'
            '### Discovery and optimization: whole-curve mean and final level\n\n'
            + report.markdown_table(['method', 'intrinsic weight', '$W$ ± standard error',
                                     'difference in $W$ from the control [95% interval]', '$L$ ± standard error',
                                     'difference in $L$ from the control [95% interval]',
                                     'seeds in common with the control'], first)
            + '### Discovery: mean capped first-goal time\n\n'
            '$M$, its standard error and its differences are in thousands of steps; the smallest value is the best, '
            'and a negative difference means the goal was first reached earlier than under the control.\n\n'
            + report.markdown_table(['method', 'intrinsic weight', '$M$ ± standard error, thousand steps',
                                     'difference in $M$ from the control [95% interval], thousand steps',
                                     'policies that reach the goal', 'seeds in common with the control'], second)
            + validation_best_text(read['best'], block)
            + left_out)


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
        found += unit['notes'] + (["complete, but left out: the method's own unit of the screen is not complete"]
                                  if unit.get('left_out') else [])
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
        words = figures.wrapped(f"{method.name}, intrinsic weight {weight_text(method.grid['weight'])}", 28)
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
    has_screen = 'study_description' in settings
    order, discovery = row_order(screen), discovery_order(screen)
    parts = [f"# {settings['title']}\n\n"
             f'Generated {contents.pacific(contents.now())} by the command below. This file is generated: regenerate '
             'it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: '
             f'`{run_dir.name}`; batch folder: `batches/{screen.batch}`.\n\n```bash\n{command(run_dir, screen.batch)}```\n\n']
    if has_screen:
        parts.append(units_section(screen))
    else:
        parts.append(case_sweeps.measures_section())
    if order:
        parts += [measures_section(screen), screen_section(screen, order), weights_section(screen, order),
                  discovery_section(screen, discovery), discovery_weights_section(screen, discovery),
                  curves_section(screen, curves_figure(screen, order, plots)),
                  properties_section(screen, order, visits_figure(screen, order, plots))]
    if screen.cases is not None:
        parts.append(case_sweeps.development_section(run_dir, screen.cases, settings['development_sweeps']))
    if screen.validation_cases is not None:
        parts.append(validation_seeds.section(settings['validation_seeds'], screen.validation_cases))
    if screen.same_configuration is not None:
        parts.append(same_configuration.section(screen.same_configuration))
    # The predictions: read by the batch's module when it has one, else quoted when `ideas.md` holds any.
    ideas = (folder / 'ideas.md').read_text() if (folder / 'ideas.md').exists() else ''
    if 'predictions' in settings or re.search(r'(?m)^## Predictions', ideas):
        parts.append(predictions_section(screen, ideas, settings.get('predictions')))
    if order:
        parts.append(sides_section(screen, order, discovery))
    if screen.validation is not None:
        parts.append(validation_section(screen.validation, settings['validation_block']))
    if 'validation_of_configurations' in settings:
        parts.append(case_sweeps.validation_section(run_dir, settings['validation_of_configurations']))
    if 'discovery' in settings:
        parts.append(case_sweeps.discovery_section(run_dir, screen.batch, settings['discovery']))
    if 'consistency_check' in settings:
        parts.append(consistency_section(screen, run_dir, settings['consistency_check']))
    if has_screen:
        parts.append(records_section(screen))
    return ''.join(parts)


def read_batch(run_dir, batch, settings):
    """Everything a batch summary reads: the screen (empty for a batch without one), the validation block, the sweeps."""
    run_dir = Path(run_dir)
    folder = run_dir / 'batches' / batch
    if 'study_description' in settings:
        screen = read_screen(run_dir, batch, settings)
    else:
        screen = Screen(batch=batch, control=None, order=(), names={}, units=(), methods={}, differences={},
                        discovery_differences={}, budget=None, seeds=())
    validation = cases = validation_cases = same = None
    if 'validation_seeds' in settings:
        validation_cases = validation_seeds.read_cases(run_dir, settings['validation_seeds'])
    if 'same_configuration' in settings:
        same = same_configuration.read(run_dir, settings['same_configuration'])
    if 'validation_block' in settings:
        description = json.loads((folder / settings['study_description']).read_text())
        validation = read_validation(run_dir, description, settings['validation_block'], batch)
    if 'development_sweeps' in settings:
        cases = case_sweeps.read_development(run_dir, batch, folder, settings['development_sweeps'])
    return replace(screen, validation=validation, cases=cases, validation_cases=validation_cases, same_configuration=same)


def generate(run_dir, batch):
    """Write the batch's summary.md, summary.html and plots/ from the run folder's shards."""
    run_dir = Path(run_dir).resolve()
    folder = run_dir / 'batches' / batch
    settings = read_settings(folder)
    screen = read_batch(run_dir, batch, settings)
    plots = folder / 'plots'
    plots.mkdir(exist_ok=True)
    markdown = build(screen, run_dir, settings, plots)
    states = contents.write_report(folder, markdown, settings['title'], name='summary')
    return {'summary': folder / 'summary.md', 'sections': states, 'screen': screen}
