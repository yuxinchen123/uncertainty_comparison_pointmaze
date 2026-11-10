"""Generate the online tabular-RND auto research's analysis.md: screen results and the two properties measured in RL."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from common import now, pacific, read_json, write_json
from fixed_settings_campaign import BUDGET
from fixed_settings_report import (CASE_LABELS, CASE_ORDER, POLICY_LABELS, fitted_slopes, grouped_standard_errors,
                                   mark, number, styled)
from report import _write_report_with_history, grouped_mean_interval

TARGET_SLOPE = (-0.6, -0.4)
TARGET_START = 1.5


def unit_rows(run_dir):
    """One row per finished setting of every auto-research unit, with its per-seed records."""
    rows = []
    for path in sorted((run_dir/'data/research').glob('*/summary.json')):
        summary = read_json(path)
        spec = summary['specification']
        if not summary.get('completed') or 'idea' not in spec:
            continue
        for setting in summary['settings']:
            copies = [row for row in summary['copy_records'] if row['setting_index'] == setting['setting_index']]
            rows.append({'stage': spec['wave'].removeprefix('fixed-'), 'batch': spec['batch'], 'idea': spec['idea'],
                         'kind': spec['geometry']['kind'], 'policy': spec['policy_kind'], 'budget': summary['budget'],
                         'configuration': setting['configuration'], 'copies': copies, 'folder': path.parent,
                         'setting_index': setting['setting_index'],
                         'mean': float(np.mean([row['capped_first_goal_steps'] for row in copies])),
                         'discovery': float(np.mean([row['first_goal_step'] >= 0 for row in copies]))})
    return rows


def properties(row):
    """Start evenness, per-cell decay slopes and late count consistency of one setting's copies, from its records."""
    # Before: behavior.npz with [checkpoints, copies, cells] histories; after: medians over this setting's copies.
    data = np.load(row['folder']/'behavior.npz')
    selected = data['setting_indices'] == row['setting_index']
    initial = data['rnd_error_initial'][selected]
    history = data['rnd_error_history'][:, selected]
    visits = data['visit_count_history'][:, selected]
    slopes = fitted_slopes(history / initial[None], visits)
    final, counts = history[-1], visits[-1]
    consistency = [np.std(np.log(final[copy][counts[copy] >= 2] * np.sqrt(counts[copy][counts[copy] >= 2])))
                   for copy in range(final.shape[0]) if (counts[copy] >= 2).sum() >= 2]
    return {'start_max_over_min': float(np.median(initial.max(axis=1) / initial.min(axis=1))),
            'slope_median': float(np.median(slopes)) if len(slopes) else None,
            'slope_quartiles': np.quantile(slopes, [0.25, 0.75]).tolist() if len(slopes) else None,
            'late_consistency': float(np.median(consistency)) if consistency else None}


def best_rows(rows):
    """Each idea's best setting per stage, batch and case: smallest mean capped first-goal time, then discovery."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['stage'], row['batch'], row['idea'], row['kind'], row['policy'])].append(row)
    best = {}
    for key, members in groups.items():
        chosen = min(members, key=lambda row: (row['mean'], -row['discovery']))
        capped = np.asarray([copy['capped_first_goal_steps'] for copy in chosen['copies']], np.float64)
        groups_of_seeds = [copy['field_seed'] for copy in chosen['copies']]
        error, _ = grouped_standard_errors(capped, groups_of_seeds, 0, 7)
        best[key] = {**chosen, 'settings_finished': len(members), 'standard_error': error,
                     'interval': grouped_mean_interval(capped, groups_of_seeds, 7), **properties(chosen)}
    return best


def met(row):
    """Words for which of the two property targets a row meets."""
    even = row['start_max_over_min'] <= TARGET_START
    decay = row['slope_median'] is not None and TARGET_SLOPE[0] <= row['slope_median'] <= TARGET_SLOPE[1]
    return 'both' if even and decay else 'even start' if even else 'decay' if decay else 'neither'


def case_table(rows, ideas):
    """Ideas in rows sorted by the screen score; metrics and the two measured properties in columns."""
    rows = sorted(rows, key=lambda row: (row['mean'], -row['discovery']))
    thousands = min(row['mean'] for row in rows) >= 1000
    means = [row['mean'] for row in rows]
    lines = ['| idea | mean capped first-goal steps ↓ ± standard error [95%] | goal found | start max/min | '
             'decay slope, median [middle half] | late count consistency | properties met | best setting | seeds |\n'
             '|---|---|---|---|---|---|---|---|---|\n']
    for index, row in enumerate(rows):
        low, high = row['interval']
        slope = ('n/a' if row['slope_median'] is None else
                 f"{row['slope_median']:.2f} [{row['slope_quartiles'][0]:.2f}, {row['slope_quartiles'][1]:.2f}]")
        setting = (f"learning rate {row['configuration']['learning_rate']:g}, entropy "
                   f"{row['configuration']['entropy_coefficient']:g}, weight "
                   f"2^{int(round(np.log2(row['configuration']['intrinsic_coefficient'])))}")
        # One field seed gives no resampling groups, so the error and interval are left out.
        spread = ('' if row['standard_error'] is None else
                  f" ± {number(row['standard_error'], thousands)} [{number(low, thousands)}, {number(high, thousands)}]")
        lines.append(f"| {ideas.get(row['idea'], row['idea'])} | {styled(number(row['mean'], thousands), mark(means, index, True))}"
                     f"{spread} | "
                     f"{100*row['discovery']:.1f}% | {row['start_max_over_min']:.2f} | {slope} | "
                     f"{'n/a' if row['late_consistency'] is None else format(row['late_consistency'], '.2f')} | {met(row)} | "
                     f"{setting} | {len(row['copies'])} |\n")
    return ''.join(lines)


def generate(run_dir):
    """Write analysis/analysis.md (with its contents table) and a JSON of every best row."""
    analysis = run_dir/'analysis'
    analysis.mkdir(exist_ok=True)
    rows = unit_rows(run_dir)
    best = best_rows(rows)
    words = {}
    for path in sorted((run_dir/'batches').glob('*/idea_names.json')):
        words.update(read_json(path))
    markdown = [f"# Online tabular-RND auto research\n\nGenerated {pacific()} (times are Pacific, converted from machines "
                f"running Eastern). Finished settings: {len(rows)}. Protocol: `program.md`; log: `STATE.md`.\n\n"]
    markdown.append(
        "## Measures\n\nDefine $b_t(s)$ as the bonus of cell $s$ after $t$ updates (before reward normalization) and "
        "$N_t(s)$ as the visits of cell $s$. Start max/min is $\\max_s b_0(s) / \\min_s b_0(s)$ (median over seeds; "
        f"target at most {TARGET_START}). The decay slope is the least-squares slope of $\\log(b_t(s)/b_0(s))$ on "
        "$\\log N_t(s)$ per cell and seed, over checkpoints with $N_t(s) \\ge 2$ (target "
        f"[{TARGET_SLOPE[0]}, {TARGET_SLOPE[1]}]). Late count consistency is the spread over cells of "
        "$\\log(b_T(s) \\sqrt{N_T(s)})$ at the last checkpoint $T$ (zero when every cell follows the same $c/\\sqrt{N}$). "
        "The screen score is the mean capped first-goal time $M(B)$ at the screen's budget, the best of each idea's "
        "settings (policy learning rate 1e-3, and 1e-4 for the control and the main ideas; entropy 0.001 or 0.01; "
        "weight 2^-6 to 2^6 in steps of 4×) on seeds 0-31; the "
        "standard error and interval resample field seeds (four policies each). Every idea keeps CleanRL's reward "
        "normalization on unless its name says it replaces it.\n\n")
    for stage_batch in sorted({(key[0], key[1]) for key in best}):
        section = [f"## {stage_batch[0].replace('-', ' ')}: results\n\n"]
        budgets = {row['budget'] for key, row in best.items() if (key[0], key[1]) == stage_batch}
        section.append(f"Budget per policy: {', '.join(f'{value:,}' for value in sorted(budgets))} transitions, on the full "
                       f"{BUDGET:,}-transition learning-rate schedule.\n\n")
        for kind, policy in CASE_ORDER:
            selected = [row for key, row in best.items() if (key[0], key[1]) == stage_batch and key[3:] == (kind, policy)]
            section.append(f"### {CASE_LABELS[kind]}, {POLICY_LABELS[policy]}\n\n")
            section.append(case_table(selected, words) + '\n' if selected else 'Not run yet.\n\n')
        markdown.append(''.join(section))
    _write_report_with_history(analysis, ''.join(markdown))
    write_json(analysis/'best_rows.json', {'at': now(), 'rows': [
        {key: value for key, value in row.items() if key not in ('copies', 'folder')} | {'folder': str(row['folder'])}
        for row in best.values()]})


def main():
    """Generate the auto research's report from its records."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    generate(parser.parse_args().run_dir)


if __name__ == '__main__':
    main()
