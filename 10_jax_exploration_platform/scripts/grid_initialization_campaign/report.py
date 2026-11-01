"""Generate initial-bonus results and behavior figures from completed run records."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from common import PLATFORM, pacific, read_json, seed_for, write_json
from campaign import result_records

sys.path.insert(0, str(PLATFORM / 'src'))
from exploration_platform.grid_initialization.fields import ADVERSARIAL_FIELD_KINDS


WAVE_ORDER = ('calibration', 'field-search', 'fine-tuning', 'confirmation', 'long-confirmation')
LABELS = {'uniform': 'Uniform initial bonuses', 'none': 'No intrinsic bonus',
          'random_log_uniform': 'Random initial bonuses', 'random_network': 'Frozen coordinate networks'}
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')
LINE_STYLES = ('solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1)))


def capped_first_goal_steps(first_goal_steps, budget):
    """Count failures at the full common budget instead of dropping them."""
    # Before: [12,-1,30] at budget 40; after: [12,40,30].
    values = np.asarray(first_goal_steps, dtype=np.int64)
    if values.ndim != 1 or np.any(values < -1) or np.any(values > budget):
        raise ValueError('First-goal steps must be a vector of -1 or steps within the budget.')
    return np.where(values < 0, budget, values).astype(np.float64)


def grouped_mean_interval(values, group_ids, seed, repetitions=2000):
    """Resample independent field seeds while retaining all their policy seeds together."""
    # Before: policies [1,3] for one field and [8,10] for another; after: two field groups.
    values = np.asarray(values, dtype=np.float64)
    identities = np.asarray(group_ids)
    if values.ndim != 1 or identities.shape != values.shape or not len(values):
        raise ValueError('Uncertainty requires matching nonempty value and group vectors.')
    unique, inverse = np.unique(identities, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    if len(unique) < 2:
        return [None, None]

    # Whole groups are sampled with replacement; unequal group sizes keep their original weights.
    generator = np.random.default_rng(seed)
    chosen = generator.integers(0, len(unique), size=(repetitions, len(unique)))
    means = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def field_label(specification):
    """Name a field by its literal initialization rule and relevant severity."""
    # Numerical exponents stay in words so table entries remain ordinary wrappable text.
    kind = specification['field_kind']
    label = LABELS.get(kind, kind.replace('_', ' ').capitalize())
    if kind in ADVERSARIAL_FIELD_KINDS or kind == 'random_log_uniform':
        label += f"; severity exponent {specification['severity_exponent']}"
    return label


def geometry_label(specification):
    """Describe a geometry without hiding its width or number of turns."""
    # A snake's width is its horizontal corridor width, not its total number of states.
    geometry = specification['geometry']
    return f"{geometry['kind'].capitalize()} grid, width {geometry['width']}, turns {geometry['turns']}"


def collect_rows(run_dir):
    """Combine seed batches only when their geometry, field, settings, and budget match."""
    # Completed summaries supply selection and metrics; behavior archives are loaded only for plots.
    grouped = {}
    for record in result_records(run_dir):
        specification = record['specification']
        for setting in record['settings']:
            configuration = setting['configuration']
            identity = json.dumps([specification['wave'], specification['geometry'],
                specification['policy_kind'], specification['field_kind'],
                specification['severity_exponent'], specification['total_timesteps'], configuration],
                sort_keys=True)
            if identity not in grouped:
                grouped[identity] = {'identity': identity, 'specification': specification,
                    'configuration': configuration, 'first_goal_steps': [], 'policy_seeds': [],
                    'field_seeds': [], 'coverage_sum': 0.0, 'sources': []}
            row = grouped[identity]
            count = len(setting['first_goal_steps'])
            for key in ('first_goal_steps', 'policy_seeds', 'field_seeds'):
                if len(setting[key]) != count:
                    raise ValueError(f"Seed metric lengths disagree for {specification['name']}.")
                row[key].extend(setting[key])
            row['coverage_sum'] += float(setting['coverage_mean']) * count
            row['sources'].append({'task': specification['name'], 'setting_index': setting['setting_index']})

    # Recompute the capped mean from every completed seed rather than averaging batch means.
    rows = []
    for row in grouped.values():
        specification = row['specification']
        budget = specification['total_timesteps']
        capped = capped_first_goal_steps(row['first_goal_steps'], budget)
        if len(set(zip(row['policy_seeds'], row['field_seeds']))) != len(capped):
            raise ValueError(f"Repeated policy/field seed pairs in {specification['name']}.")
        groups = row['field_seeds'] if specification['field_kind'] in (
            'random_network', 'random_log_uniform') else row['policy_seeds']
        successes = np.asarray(row['first_goal_steps']) >= 0
        row.update(seed_count=len(capped), independent_group_count=len(set(groups)),
                   capped_mean_first_goal_steps=float(capped.mean()),
                   capped_mean_interval=grouped_mean_interval(capped, groups,
                       seed_for('report', row['identity'], 'mean_steps')),
                   goal_fraction=float(successes.mean()),
                   goal_fraction_interval=grouped_mean_interval(successes, groups,
                       seed_for('report', row['identity'], 'goal_fraction')),
                   coverage_mean=row['coverage_sum'] / len(capped))
        rows.append(row)
    return rows


def select_reporting_rows(rows):
    """Use development minima but preserve every frozen confirmation comparison."""
    # Fresh confirmation data never choose a winning learning rate or bonus coefficient.
    selected = {}
    for row in rows:
        specification = row['specification']
        confirmation = specification['wave'] in ('confirmation', 'long-confirmation')
        comparison = row['configuration'].get('comparison', 'development best')
        key = json.dumps([specification['wave'], specification['geometry'], specification['policy_kind'],
                          specification['field_kind'], specification['severity_exponent'],
                          specification['total_timesteps'], comparison], sort_keys=True)
        if key in selected and confirmation:
            raise ValueError('A frozen confirmation comparison contains multiple different configurations.')
        if key not in selected or row['capped_mean_first_goal_steps'] < selected[key]['capped_mean_first_goal_steps']:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (
        WAVE_ORDER.index(row['specification']['wave']), geometry_label(row['specification']),
        row['specification']['policy_kind'], row['configuration'].get('comparison', ''),
        field_label(row['specification'])))


def _load_behavior(run_dir, row):
    """Gather one selected setting's complete seed arrays and sampled movement history."""
    # Before: archives containing all settings; after: matching copies concatenated along seed axis.
    gathered = defaultdict(list)
    trajectories, trajectory_steps, maximum_history = [], [], []
    positions = grid = None
    for source in row['sources']:
        path = Path(run_dir) / 'data/research' / source['task'] / 'behavior.npz'
        with np.load(path, allow_pickle=False) as archive:
            selected = archive['setting_indices'] == source['setting_index']
            if positions is not None and (not np.array_equal(positions, archive['positions']) or
                                          not np.array_equal(grid, archive['grid'])):
                raise ValueError('A reporting group contains incompatible physical grids.')
            positions, grid = archive['positions'].copy(), archive['grid'].copy()
            for key in ('initial_fields', 'visit_counts', 'late_visit_counts', 'policy_probabilities',
                        'first_goal_steps', 'seed_ids', 'field_seed_ids'):
                gathered[key].append(archive[key][selected].copy())
            for copy in np.flatnonzero(selected):
                trajectories.append(archive['trajectory_states'][:, copy].copy())
                trajectory_steps.append(archive['trajectory_steps'].copy())
                maximum_history.append(archive['maximum_progress_history'][:, copy].copy())
    arrays = {key: np.concatenate(values, axis=0) for key, values in gathered.items()}
    arrays.update(positions=positions, grid=grid, trajectories=trajectories,
                  trajectory_steps=trajectory_steps, maximum_history=maximum_history)
    if len(arrays['seed_ids']) != row['seed_count']:
        raise ValueError('Behavior archive seed count disagrees with completed summary metrics.')
    return arrays


def representative_seed_index(first_goal_steps, budget, seed_ids, field_seed_ids):
    """Choose the seed nearest median capped first-goal time with fixed seed tie breaks."""
    # Failures all contribute the budget; equal candidates use increasing field and policy seeds.
    capped = capped_first_goal_steps(first_goal_steps, budget)
    median = np.median(capped)
    return min(range(len(capped)), key=lambda index: (
        abs(capped[index] - median), int(field_seed_ids[index]), int(seed_ids[index])))


def _draw_map(axis, values, grid, title, normalization=None, label_states=False):
    """Draw true grid cells with walls distinct from unvisited states in vector output."""
    # Before: one value per traversable state; after: a masked physical grid with dark walls.
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm
    displayed = np.asarray(values)
    if isinstance(normalization, LogNorm):
        displayed = np.where(displayed == 0, normalization.vmin / 2, displayed)
    field = np.ma.masked_where(grid < 0, displayed[np.maximum(grid, 0)])
    colors = colormaps['viridis'].copy()
    colors.set_under('white')
    axis.set_facecolor('#505050')
    mesh = axis.pcolormesh(np.arange(grid.shape[1] + 1) - 0.5,
                          np.arange(grid.shape[0] + 1) - 0.5, field,
                          cmap=colors, norm=normalization, edgecolors='none')
    axis.set_xlim(-0.5, grid.shape[1] - 0.5)
    axis.set_ylim(grid.shape[0] - 0.5, -0.5)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel('Column')
    axis.set_ylabel('Row')
    axis.scatter([0, grid.shape[1] - 1], [0, grid.shape[0] - 1], s=25,
                 marker='o', facecolors='none', edgecolors='#D55E00', linewidths=1.5)
    if label_states:
        for row_index, column_index in np.argwhere(grid >= 0):
            axis.text(column_index, row_index, str(grid[row_index, column_index]),
                      ha='center', va='center', fontsize=7)
    return mesh


def _save_figure(figure, directory, name):
    """Keep a publication PDF and a directly readable PNG for each behavior figure."""
    # Both formats are generated from the same figure with the same vector wall geometry.
    figure.savefig(directory / (name + '.pdf'), bbox_inches='tight')
    figure.savefig(directory / (name + '.png'), bbox_inches='tight', dpi=180)


def _plot_comparison(run_dir, rows, plots_dir, figure_name):
    """Compare equal-budget initial fields, visits, movement, and learned action probabilities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    # Select at most one development adversary by the largest tuned capped mean, then freeze that view.
    priority = ('uniform', 'random_log_uniform', 'none', 'random_network')
    selected = [row for kind in priority for row in rows if row['specification']['field_kind'] == kind]
    adversaries = [row for row in rows if row['specification']['field_kind'] in ADVERSARIAL_FIELD_KINDS]
    if adversaries:
        selected.insert(min(2, len(selected)), max(adversaries, key=lambda row: row['capped_mean_first_goal_steps']))
    # Different random severities are tabled; the figure uses the strongest measured random condition.
    unique = {}
    for row in selected:
        kind = row['specification']['field_kind']
        if kind not in unique or row['capped_mean_first_goal_steps'] > unique[kind]['capped_mean_first_goal_steps']:
            unique[kind] = row
    selected = list(unique.values())
    data = [_load_behavior(run_dir, row) for row in selected]
    if not selected:
        return []

    # Every map in a row uses the same color scale and equal transition budgets.
    summaries = [[values[key].mean(axis=0) for values in data]
                 for key in ('initial_fields', 'visit_counts', 'late_visit_counts')]
    figure, axes = plt.subplots(3, len(selected), figsize=(4.5 * len(selected), 7.5), squeeze=False,
                                layout='constrained')
    for metric, fields in enumerate(summaries):
        maximum = max(float(field.max()) for field in fields)
        positive_minima = [float(field[field > 0].min()) for field in fields if np.any(field > 0)]
        minimum = min(positive_minima) if positive_minima else 1
        normalization = Normalize(vmin=0, vmax=max(1, maximum)) if metric == 0 else LogNorm(vmin=minimum, vmax=max(minimum * 2, maximum))
        for column, (row, values, field) in enumerate(zip(selected, data, fields)):
            label = field_label(row['specification']) if metric == 0 else ('Total visits' if metric == 1 else 'Late visits')
            mesh = _draw_map(axes[metric, column], field, values['grid'], label, normalization)
        figure.colorbar(mesh, ax=axes[metric].tolist(), label=('Initial bonus' if metric == 0 else 'Mean visits per seed'))
    figure.suptitle(geometry_label(selected[0]['specification']) + '; ' + selected[0]['specification']['policy_kind'] + ' policy', fontsize=12)
    _save_figure(figure, plots_dir, figure_name + '_maps')
    plt.close(figure)

    # Sampled training positions across all seeds show persistent concentration over actual training time.
    figure, axes = plt.subplots(2, len(selected), figsize=(4.5 * len(selected), 6), squeeze=False,
                                layout='constrained')
    representatives = []
    for column, (row, values) in enumerate(zip(selected, data)):
        steps = values['trajectory_steps'][0]
        if any(not np.array_equal(steps, other) for other in values['trajectory_steps']):
            raise ValueError('Training position samples have inconsistent global step coordinates.')
        trajectories = np.stack(values['trajectories'], axis=1)
        density = np.zeros((len(values['positions']), len(steps)), dtype=np.float64)
        for time_index in range(len(steps)):
            density[:, time_index] = np.bincount(trajectories[time_index], minlength=len(values['positions'])) / row['seed_count']
        axes[0, column].pcolormesh(steps, np.arange(len(values['positions'])), density, shading='nearest',
                                  cmap='viridis', vmin=0, vmax=1)
        index = representative_seed_index(values['first_goal_steps'], row['specification']['total_timesteps'],
                                          values['seed_ids'], values['field_seed_ids'])
        axes[0, column].plot(steps, values['maximum_history'][index], color='#D55E00', linewidth=1,
                             label='Farthest state, representative seed')
        axes[0, column].set_title(field_label(row['specification']), fontsize=10)
        axes[0, column].set_xlabel('Training transitions per policy')
        axes[0, column].set_ylabel('Distance along path')
        axes[0, column].legend(fontsize=7, loc='lower right')
        axes[1, column].plot(steps, values['trajectories'][index], linewidth=.8, color=COLORS[column % len(COLORS)])
        axes[1, column].set_xlabel('Training transitions per policy')
        axes[1, column].set_ylabel('Representative sampled position')
        representatives.append({'condition': field_label(row['specification']),
                                'policy_seed': int(values['seed_ids'][index]),
                                'field_seed': int(values['field_seed_ids'][index])})
    _save_figure(figure, plots_dir, figure_name + '_movement')
    plt.close(figure)

    # Arrows show legal action probabilities; cell color shows probability assigned to blocked moves.
    figure, axes = plt.subplots(1, len(selected), figsize=(4.5 * len(selected), 3.5), squeeze=False,
                                layout='constrained')
    for column, (row, values) in enumerate(zip(selected, data)):
        positions, grid = values['positions'], values['grid']
        probabilities = values['policy_probabilities'].mean(axis=0)
        stay = np.zeros(len(positions))
        arrows = []
        for state, (physical_row, physical_column) in enumerate(positions):
            for action, (row_offset, column_offset) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
                next_row, next_column = physical_row + row_offset, physical_column + column_offset
                legal = 0 <= next_row < grid.shape[0] and 0 <= next_column < grid.shape[1] and grid[next_row, next_column] >= 0
                if legal:
                    arrows.append((physical_column, physical_row, column_offset * probabilities[state, action] * .8,
                                   row_offset * probabilities[state, action] * .8, state))
                else:
                    stay[state] += probabilities[state, action]
        _draw_map(axes[0, column], stay, grid, field_label(row['specification']), Normalize(0, 1))
        center = int(np.argmax(values['late_visit_counts'].mean(axis=0)))
        window = positions[max(0, center - 16):min(len(positions), center + 17)]
        for x, y, dx, dy, state in arrows:
            if abs(state - center) <= 16:
                axes[0, column].quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1,
                                      color='#D55E00', width=.008)
        axes[0, column].set_xlim(window[:, 1].min() - .75, window[:, 1].max() + .75)
        axes[0, column].set_ylim(window[:, 0].max() + .75, window[:, 0].min() - .75)
    figure.suptitle('Policy arrows near the most visited late state; color is blocked-action probability', fontsize=11)
    _save_figure(figure, plots_dir, figure_name + '_policy')
    plt.close(figure)

    # First-goal curves use all seeds, including failures remaining undiscovered at the budget.
    figure, axis = plt.subplots(figsize=(7, 4), layout='constrained')
    for index, row in enumerate(selected):
        budget = row['specification']['total_timesteps']
        times = np.linspace(0, budget, 129)
        goals = np.asarray(row['first_goal_steps'])
        fraction = ((goals[:, None] >= 0) & (goals[:, None] <= times[None, :])).mean(axis=0)
        axis.plot(times, fraction, color=COLORS[index % len(COLORS)], linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                  label=field_label(row['specification']))
    axis.set_xlabel('Training transitions per policy')
    axis.set_ylabel('Fraction of seeds reaching the goal')
    axis.set_ylim(0, 1.02)
    axis.legend(fontsize=8)
    _save_figure(figure, plots_dir, figure_name + '_discovery')
    plt.close(figure)
    return representatives


def _format_interval(value, interval, percentage=False):
    """Render an estimate and its 95-percent interval in a compact plain-text cell."""
    # Single independent groups have no estimable interval and are marked explicitly.
    scale = 100 if percentage else 1
    estimate = f'{value * scale:.1f}'
    if interval[0] is None:
        return estimate + ' (interval N/A)'
    return estimate + f' [{interval[0] * scale:.1f}, {interval[1] * scale:.1f}]'


def _latex_escape(value):
    """Escape ordinary report labels for text inside generated LaTeX tables."""
    # Before: names containing underscores or percent signs; after: literal printable text.
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_\allowbreak ', '{': r'\{', '}': r'\}'}
    return ''.join(replacements.get(character, character) for character in str(value))


def _write_results_tex(analysis_dir, display_rows, figure_groups):
    """Write generated result tables and selected behavior figures for the development document."""
    # Keep results separate across budgets and shared versus independently tuned comparisons.
    pieces = ['% Generated by scripts/grid_initialization_campaign/report.py; edit that generator.\n']
    if not display_rows:
        pieces.append('No completed training records are available yet. No failure case has been established.\n')
    groups = defaultdict(list)
    for row in display_rows:
        spec = row['specification']
        identity = (spec['wave'], geometry_label(spec), spec['policy_kind'],
                    spec['total_timesteps'], row['configuration'].get('comparison', 'development best'))
        groups[identity].append(row)
    for group_index, (identity, rows) in enumerate(groups.items()):
        wave, geometry, policy, budget, comparison = identity
        pieces.append('\n\\begin{table}[H]\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n'
                      '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.0cm}'
                      '>{\\raggedright\\arraybackslash}p{3.7cm}'
                      '>{\\raggedright\\arraybackslash}p{2.8cm}r@{}}\n\\toprule\n'
                      'Initialization & Mean steps, 95\\% interval $\\downarrow$ & Goal found, \\% & Seeds \\\\\n\\midrule\n')
        ranking = sorted(set(row['capped_mean_first_goal_steps'] for row in rows))
        for row in rows:
            value = _format_interval(row['capped_mean_first_goal_steps'], row['capped_mean_interval'])
            if row['capped_mean_first_goal_steps'] == ranking[0]:
                value = r'\textbf{' + value + '}'
            elif len(ranking) > 1 and row['capped_mean_first_goal_steps'] == ranking[1]:
                estimate, interval = value.split(' ', 1)
                value = r'\underline{' + estimate + '} ' + interval
            goal = _format_interval(row['goal_fraction'], row['goal_fraction_interval'], percentage=True)
            pieces.append(_latex_escape(field_label(row['specification'])) + ' & ' + value + ' & ' +
                          _latex_escape(goal) + f" & {row['seed_count']} \\\\\n")
        pieces.append('\\bottomrule\n\\end{tabular}\n\\caption{' + _latex_escape(
            f'{geometry}; {policy} policy; {wave.replace("-", " ")}; {comparison.replace("-", " ")}. '
            f'Every failed seed contributes the full {budget:,}-transition budget. '
            'Intervals resample network or random-field seeds as groups, retaining their policy seeds; '
            'deterministic fields resample policy seeds. Development rows select the lowest development mean; '
            'confirmation rows retain frozen settings.') + '}\n'
            + f'\\label{{tab:grid-initialization-results-{group_index}}}\n\\end{{table}}\n')
    for figure in figure_groups:
        for suffix, caption in (('maps', 'Initial bonus, cumulative visitation, and final-window visitation. '
                'Each row uses a common color scale across initializations. Dark cells are walls; white visitation cells have zero visits.'),
                ('movement', 'Sampled training positions across all seeds, followed by a representative seed selected by median capped first-goal time. '
                'Orange lines show its farthest discovered state. Samples use the recorded training-transition times.'),
                ('policy', 'Final legal-action probabilities near the most visited late state. Cell color is the probability assigned to blocked actions.'),
                ('discovery', 'Fraction of independent policy copies that have reached the goal by each transition budget, including failed copies.')):
            filename = analysis_dir / 'plots' / (figure['name'] + '_' + suffix + '.pdf')
            pieces.append('\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{\\detokenize{' +
                          str(filename) + '}}\n\\caption{' + _latex_escape(caption) +
                          '}\n\\end{figure}\n')
    (analysis_dir / 'results.tex').write_text(''.join(pieces))


def generate_report(run_dir):
    """Regenerate durable metrics and figures without launching or modifying training."""
    # This generator performs numerical analysis and must run on a compute node.
    run_dir = Path(run_dir).resolve()
    analysis_dir = run_dir / 'analysis'
    plots_dir = analysis_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(run_dir)
    display_rows = select_reporting_rows(rows)
    pieces = ['<!-- Generated by scripts/grid_initialization_campaign/report.py; edit that generator. -->\n',
              '# Initial bonus differences and exploration\n\n', f'Updated {pacific()}.\n\n',
              'For policy seed $i$, let $T_i$ be the number of training transitions before its first goal, '
              '$B$ its transition budget, and $K$ the number of completed independent copies. A failure contributes $B$. '
              'The primary metric is the mean capped first-goal time, lower being better.\n\n',
              '$$ M = \\frac{1}{K} \\sum_{i=1}^{K} \\min(T_i,B). $$\n\n',
              'A missing first goal is treated as $T_i$ exceeding $B$. Goal-found percentage counts every seed. '
              'Coverage is the mean fraction of traversable states visited. Brackets give 95% bootstrap intervals. '
              'For random initial fields, entire field seeds are resampled with their policy seeds kept together. '
              'For deterministic fields, policy seeds are resampled. One independent group gives no interval.\n\n',
              'Development rows choose settings using development results. Confirmation comparisons retain settings '
              'chosen before their fresh seeds ran. Shared and independently tuned settings appear separately. '
              'These results concern the recorded finite budget; they do not establish permanent trapping.\n\n']
    if not display_rows:
        pieces.append('No completed training records are available. No failure case has been established.\n')
    previous = None
    for row in display_rows:
        specification = row['specification']
        heading = (specification['wave'], geometry_label(specification), specification['policy_kind'],
                   specification['total_timesteps'], row['configuration'].get('comparison', 'development best'))
        if heading != previous:
            pieces.append('\n## ' + '; '.join(map(str, heading)) + '\n\n'
                          '| Initialization | Mean steps [95% interval] | Goal found, % | Coverage, % | Seeds |\n'
                          '|---|---:|---:|---:|---:|\n')
            previous = heading
        pieces.append(f"| {field_label(specification)} | " +
            _format_interval(row['capped_mean_first_goal_steps'], row['capped_mean_interval']) + ' | ' +
            _format_interval(row['goal_fraction'], row['goal_fraction_interval'], percentage=True) +
            f" | {100 * row['coverage_mean']:.1f} | {row['seed_count']} |\n")

    # Refresh behavior figures for the latest completed phase while preserving older phase files.
    figure_groups = []
    if display_rows:
        latest = max(WAVE_ORDER.index(row['specification']['wave']) for row in display_rows)
        grouped = defaultdict(list)
        for row in display_rows:
            spec = row['specification']
            if WAVE_ORDER.index(spec['wave']) == latest:
                key = json.dumps([spec['geometry'], spec['policy_kind'], spec['total_timesteps'],
                    row['configuration'].get('comparison', 'development best')], sort_keys=True)
                grouped[key].append(row)
        pieces.append('\n## Behavior\n\nMaps show means over every completed seed at the displayed setting. '
                      'Movement examples use the seed nearest the median capped first-goal time, '
                      'with field seed and policy seed as fixed tie breaks. Policy arrows show legal actions; '
                      'blocked actions are represented by the background probability. '
                      'The policy view covers up to 33 states around the most visited late state.\n\n')
        for index, comparison_rows in enumerate(grouped.values()):
            name = f'{WAVE_ORDER[latest]}_comparison-{index}'
            representatives = _plot_comparison(run_dir, comparison_rows, plots_dir, name)
            figure_groups.append({'name': name, 'representatives': representatives})
            pieces.append(f'### {geometry_label(comparison_rows[0]["specification"])}; '
                          f'{comparison_rows[0]["specification"]["policy_kind"]} policy\n\n')
            for suffix in ('maps', 'movement', 'policy', 'discovery'):
                pieces.append(f'![{suffix.capitalize()}](plots/{name}_{suffix}.png)\n\n')
            for example in representatives:
                pieces.append(f"- {example['condition']}: policy seed {example['policy_seed']}, field seed {example['field_seed']}.\n")
            pieces.append('\n')

    # Keep every seed's metric available independently of selection and illustrative figures.
    data_dir = analysis_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    with (data_dir / 'seed_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['wave', 'geometry', 'policy', 'field', 'configuration', 'budget',
                         'policy_seed', 'field_seed', 'first_goal_steps', 'capped_first_goal_steps'])
        for row in rows:
            spec = row['specification']
            capped = capped_first_goal_steps(row['first_goal_steps'], spec['total_timesteps'])
            for index in range(row['seed_count']):
                writer.writerow([spec['wave'], geometry_label(spec), spec['policy_kind'], field_label(spec),
                    json.dumps(row['configuration'], sort_keys=True), spec['total_timesteps'],
                    row['policy_seeds'][index], row['field_seeds'][index], row['first_goal_steps'][index], capped[index]])
    pieces.append('\n## Records and visualization sources\n\n'
                  '[Every seed metric](data/seed_metrics.csv) is retained, including unselected development settings. '
                  'Every task retains its complete seed arrays in `data/research/<task>/behavior.npz`.\n\n'
                  'The figure choices follow [visitation maps and action arrows](https://arxiv.org/pdf/2001.00119), '
                  '[trajectory and coverage comparisons](https://proceedings.neurips.cc/paper_files/paper/2023/file/9c7900fac04a701cbed83256b76dbaa3-Paper-Conference.pdf), '
                  'and [initial random-network bonus maps](https://proceedings.mlr.press/v235/yang24w.html).\n')
    (analysis_dir / 'analysis.md').write_text(''.join(pieces))
    _write_results_tex(analysis_dir, display_rows, figure_groups)
    write_json(analysis_dir / 'data/aggregate_results.json', {'generated_at': pacific(),
        'completed_setting_groups': len(rows), 'reported_groups': display_rows, 'figures': figure_groups})
    return {'completed_setting_groups': len(rows), 'reported_rows': len(display_rows),
            'figure_comparisons': len(figure_groups), 'report': str(analysis_dir / 'analysis.md')}


def main():
    """Generate one run's numerical report from the command line."""
    # No training command is called; numerical plotting still belongs on a compute node.
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(generate_report(arguments.run_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
