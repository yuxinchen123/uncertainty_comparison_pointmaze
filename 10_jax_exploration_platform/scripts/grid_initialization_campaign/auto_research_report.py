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
BASELINE_IDEA = 'control'


def unit_rows(run_dir):
    """One row per finished setting of every auto-research unit, with its per-seed records."""
    rows = []
    for path in sorted((run_dir/'data/research').glob('*/summary.json')):
        summary = read_json(path)
        spec = summary['specification']
        # Confirmation units are pooled over all their seeds by confirmation_records(), never ranked by unit.
        if not summary.get('completed') or 'idea' not in spec or spec['wave'].startswith('fixed-confirmation'):
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
    if 'rnd_error_history' not in data.files:
        # A count-rule row has no network error: its start is its field (uniform, or the frozen tabular-RND field).
        fields = data['initial_fields'][selected]
        return {'start_max_over_min': float(np.median(fields.max(axis=1) / fields.min(axis=1))),
                'slope_median': None, 'slope_quartiles': None, 'late_consistency': None}
    initial = data['rnd_error_initial'][selected]
    history = data['rnd_error_history'][:, selected]
    visits = data['visit_count_history'][:, selected]
    slopes = fitted_slopes(history / initial[None], visits)
    final, counts = history[-1], visits[-1]
    # A bonus that reached exactly zero has no logarithm; such cells are left out of the spread.
    consistency = []
    for copy in range(final.shape[0]):
        usable = (counts[copy] >= 2) & (final[copy] > 0)
        if usable.sum() >= 2:
            consistency.append(np.std(np.log(final[copy][usable] * np.sqrt(counts[copy][usable]))))
    slopes = slopes[np.isfinite(slopes)]
    return {'start_max_over_min': float(np.median(initial.max(axis=1) / initial.min(axis=1))),
            'slope_median': float(np.median(slopes)) if len(slopes) else None,
            'slope_quartiles': np.quantile(slopes, [0.25, 0.75]).tolist() if len(slopes) else None,
            'late_consistency': float(np.median(consistency)) if consistency else None}


def nonfinite_copies(folder):
    """Count a unit's copies whose bonus became not-a-number or infinite at any checkpoint, and its copies."""
    data = np.load(folder/'behavior.npz')
    if 'rnd_error_history' not in data.files:
        return 0, int(len(data['seed_ids']))
    history = data['rnd_error_history']
    return int((~np.isfinite(history).all(axis=(0, 2))).sum()), int(history.shape[1])


def best_rows(rows):
    """Each idea's best setting per stage, batch and case: smallest mean capped first-goal time, then discovery."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['stage'], row['batch'], row['idea'], row['kind'], row['policy'])].append(row)
    best = {}
    for key, members in groups.items():
        # An idea whose predictor ran away in some copies is not a fair test of its properties; count them.
        counts = [nonfinite_copies(folder) for folder in sorted({row['folder'] for row in members})]
        chosen = min(members, key=lambda row: (row['mean'], -row['discovery']))
        capped = np.asarray([copy['capped_first_goal_steps'] for copy in chosen['copies']], np.float64)
        groups_of_seeds = [copy['field_seed'] for copy in chosen['copies']]
        error, _ = grouped_standard_errors(capped, groups_of_seeds, 0, 7)
        best[key] = {**chosen, 'settings_finished': len(members), 'standard_error': error,
                     'nonfinite_copies': sum(bad for bad, _ in counts), 'all_copies': sum(total for _, total in counts),
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
             'decay slope, median [middle half] | late count consistency | properties met | best setting | seeds | '
             'copies with a non-finite bonus, all settings |\n|---|---|---|---|---|---|---|---|---|---|\n']
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
                     f"{setting} | {len(row['copies'])} | {row['nonfinite_copies']} of {row['all_copies']} |\n")
    return ''.join(lines)


def confirmation_records(run_dir):
    """Pool every finished confirmation unit's copies by (batch, idea, grid, policy), one record per policy seed."""
    # Before: 8 units of 128 seeds for one idea in one case; after: {policy seed: record} with 1,024 entries.
    pooled, configurations = defaultdict(dict), {}
    for path in sorted((run_dir/'data/research').glob('confirmation-*/summary.json')):
        summary = read_json(path)
        spec = summary['specification']
        if not summary.get('completed'):
            continue
        key = (spec['batch'], spec['idea'], spec['geometry']['kind'], spec['policy_kind'])
        for record in summary['copy_records']:
            pooled[key][record['policy_seed']] = record
        configurations[key] = summary['settings'][0]['configuration']
    return pooled, configurations


def confirmation_row(records, baseline):
    """One idea's confirmed metrics in one case, and its paired difference to the baseline on the seeds both have."""
    seeds = sorted(records)
    capped = np.asarray([records[seed]['capped_first_goal_steps'] for seed in seeds], np.float64)
    groups = [records[seed]['field_seed'] for seed in seeds]
    row = {'mean': float(capped.mean()), 'standard_error': grouped_standard_errors(capped, groups, 0, 7)[0],
           'interval': grouped_mean_interval(capped, groups, 7), 'seeds': len(seeds),
           'discovery': float(np.mean([records[seed]['first_goal_step'] >= 0 for seed in seeds])),
           'coverage': float(np.mean([records[seed]['coverage'] for seed in seeds])),
           'sampled': float(np.mean([records[seed]['evaluation_goal_rate_sampled'] for seed in seeds])),
           'greedy': float(np.mean([records[seed]['evaluation_goal_rate_greedy'] for seed in seeds])), 'paired': None}
    common = [seed for seed in seeds if baseline is not None and seed in baseline]
    if common and baseline is not records:
        # Before: two score lists on the same policy seeds; after: their per-seed differences, grouped by field seed.
        difference = np.asarray([records[seed]['capped_first_goal_steps'] - baseline[seed]['capped_first_goal_steps']
                                 for seed in common], np.float64)
        row['paired'] = {'mean': float(difference.mean()), 'seeds': len(common),
                         'interval': grouped_mean_interval(difference, [records[seed]['field_seed'] for seed in common], 7),
                         'baseline_mean': float(np.mean([baseline[seed]['capped_first_goal_steps'] for seed in common]))}
    return row


def confirmation_table(pooled, configurations, batch, kind, policy, ideas):
    """Ideas in rows sorted by the confirmed score; metrics and the paired difference to the baseline in columns."""
    baseline = pooled.get((batch, BASELINE_IDEA, kind, policy))
    rows = sorted(((idea, confirmation_row(records, baseline), configurations[(group, idea, grid, kind_of_policy)])
                   for (group, idea, grid, kind_of_policy), records in pooled.items()
                   if (group, grid, kind_of_policy) == (batch, kind, policy)), key=lambda item: item[1]['mean'])
    means = [row['mean'] for _, row, _ in rows]
    lines = ['| idea | mean capped first-goal steps ↓ ± standard error [95%] | difference to the baseline on the same '
             'seeds [95%] | goal found | coverage | final-policy goal rate, sampled | greedy | configuration | seeds |\n'
             '|---|---|---|---|---|---|---|---|---|\n']
    for index, (idea, row, setting) in enumerate(rows):
        low, high = row['interval']
        paired = row['paired']
        if idea == BASELINE_IDEA:
            difference = 'baseline'
        elif paired is None:
            difference = 'baseline not finished'
        else:
            difference = (f"{paired['mean'] / 1000:+,.2f}K [{paired['interval'][0] / 1000:+,.2f}K, "
                          f"{paired['interval'][1] / 1000:+,.2f}K], {100 * paired['mean'] / paired['baseline_mean']:+.0f}% "
                          f"({paired['seeds']} seeds)")
        configuration = (f"learning rate {setting['learning_rate']:g}, entropy {setting['entropy_coefficient']:g}, "
                         f"weight 2^{int(round(np.log2(setting['intrinsic_coefficient'])))}")
        lines.append(f"| {ideas.get(idea, idea)} | {styled(number(row['mean'], True), mark(means, index, True))} ± "
                     f"{number(row['standard_error'], True)} [{number(low, True)}, {number(high, True)}] | {difference} | "
                     f"{100 * row['discovery']:.1f}% | {100 * row['coverage']:.1f}% | {100 * row['sampled']:.1f}% | "
                     f"{100 * row['greedy']:.1f}% | {configuration} | {row['seeds']} |\n")
    return ''.join(lines)


def verdict(paired):
    """Say in a word where a row stands against the baseline: its 95% interval below zero, above zero, or across it."""
    low, high = paired['interval']
    return 'ahead' if high < 0 else 'behind' if low > 0 else 'level'


def confirmation_summary(pooled, batch, ideas):
    """One row per idea: confirmed score, percentage difference to the baseline and verdict in each of the four cases."""
    # Before: per-case tables; after: ideas in rows, the four cases in columns, sorted by the mean percentage difference.
    names = sorted({idea for group, idea, _, _ in pooled if group == batch})
    rows = []
    for idea in names:
        cells, percentages, ahead, compared = [], [], 0, 0
        for kind, policy in CASE_ORDER:
            records = pooled.get((batch, idea, kind, policy))
            baseline = pooled.get((batch, BASELINE_IDEA, kind, policy))
            if records is None:
                cells.append('not run yet')
                continue
            row = confirmation_row(records, baseline)
            text = f"{number(row['mean'], True)} ({row['seeds']} seeds)"
            if idea != BASELINE_IDEA and row['paired'] is not None:
                percentage = 100 * row['paired']['mean'] / row['paired']['baseline_mean']
                word = verdict(row['paired'])
                text += f"; {percentage:+.0f}%, {word} ({row['paired']['seeds']} seeds in common)"
                percentages.append(percentage)
                ahead += word == 'ahead'
                compared += 1
            elif idea != BASELINE_IDEA:
                text += '; baseline not finished'
            cells.append(text)
        count = 'baseline' if idea == BASELINE_IDEA else f'{ahead} of {compared} compared'
        rows.append((idea != BASELINE_IDEA, float(np.mean(percentages)) if percentages else 0.0, idea, cells, count))
    lines = ['| idea | ' + ' | '.join(f'{CASE_LABELS[kind]}, {POLICY_LABELS[policy]}' for kind, policy in CASE_ORDER)
             + ' | cases ahead of the baseline |\n|---|---|---|---|---|---|\n']
    for _, _, idea, cells, count in sorted(rows):
        lines.append(f"| {ideas.get(idea, idea)} | " + ' | '.join(cells) + f' | {count} |\n')
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
        "normalization on unless its name says it replaces it. The last column counts, over all of an idea's settings in "
        "the case, the copies whose bonus became not-a-number or infinite: such an idea is unstable, and its score is "
        "not evidence about the two properties.\n\n")
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
    pooled, configurations = confirmation_records(run_dir)
    for batch in sorted({key[0] for key in pooled}):
        section = [f"## confirmation {batch}: chosen configurations on the confirmation seeds\n\n"
                   "Define $K$ as the number of policy seeds, $B$ as the budget, and $T_i$ as the transition at which "
                   "policy $i$ first reaches the goal ($T_i = \\infty$ if never). The score is "
                   "$M(B) = \\dfrac{1}{K} \\sum_{i=1}^{K} \\min(T_i, B)$ at "
                   f"$B$ = {BUDGET:,}. Each row is one idea's configuration chosen on the development seeds, trained on "
                   "the confirmation seeds from 20000 on; every finished seed of a row is pooled. Define $T_i^{a}$ and "
                   "$T_i^{b}$ as policy $i$'s first-goal transitions under the row's idea and under the baseline "
                   f"(`{BASELINE_IDEA}`, CleanRL online tabular-RND with reward normalization on). The difference "
                   "column is $\\dfrac{1}{K} \\sum_{i=1}^{K} \\left(\\min(T_i^{a}, B) - \\min(T_i^{b}, B)\\right)$ over the "
                   "$K$ seeds both rows have (negative means faster than the baseline), followed by the same "
                   "difference as a percentage of the baseline's mean. Standard errors and 95% intervals resample "
                   "field seeds (four policy seeds each).\n\n"]
        section.append("### All four cases\n\nEach cell gives the confirmed score, then the difference to the baseline on "
                       "the seeds both have as a percentage of the baseline's mean, and a verdict: ahead when the whole "
                       "95% interval of the difference is below zero, behind when it is above zero, level when it "
                       "contains zero. Rows are sorted by the mean of those percentages.\n\n"
                       + confirmation_summary(pooled, batch, words) + '\n')
        for kind, policy in CASE_ORDER:
            section.append(f"### {CASE_LABELS[kind]}, {POLICY_LABELS[policy]}\n\n")
            present = any((group, grid, kind_of_policy) == (batch, kind, policy) for group, _, grid, kind_of_policy in pooled)
            section.append(confirmation_table(pooled, configurations, batch, kind, policy, words) + '\n'
                           if present else 'Not run yet.\n\n')
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
