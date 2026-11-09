"""Generate the fixed CleanRL PPO-RND settings study's analysis.md, results.tex and figures."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import now, pacific, read_json, write_json
from fixed_settings_campaign import (BUDGET, DEFAULT_CELL, ENTROPIES, LEARNING_RATES, WEIGHTS,
                                     condition_name, conditions)
from report import _write_report_with_history, grouped_mean_interval

NETWORK_METHODS = {'frozen-network', 'online-rnd'}
METHOD_LABELS = {'no-bonus': 'no bonus', 'uniform': 'uniform field',
                 'alternating-high-low': 'alternating high/low field (64:1)',
                 'frozen-network': 'frozen network field', 'online-rnd': 'online RND'}
CASE_ORDER = [('straight', 'tabular'), ('straight', 'neural'), ('snake', 'tabular'), ('snake', 'neural')]
CASE_LABELS = {'straight': 'straight grid, 32 cells', 'snake': 'snake grid, width 16, 2 turns (50 cells)'}
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')


def condition_label(name):
    """Words for a condition: method, then the two normalizations."""
    condition = next(row for row in conditions() if condition_name(row) == name)
    if condition['method'] == 'no-bonus':
        return 'no bonus'
    parts = [METHOD_LABELS[condition['method']]]
    if condition['method'] in NETWORK_METHODS:
        parts.append('state normalization ' + ('on' if condition['state_normalization'] else 'off'))
    parts.append('reward normalization ' + ('on' if condition['reward_normalization'] else 'off'))
    return ', '.join(parts)


def collect(run_dir):
    """One row per finished setting, with its per-seed records and summary metrics."""
    rows = []
    for path in sorted((run_dir/'data/research').glob('fixed-*/summary.json')):
        summary = read_json(path)
        if not summary.get('completed'):
            continue
        spec = summary['specification']
        for setting in summary['settings']:
            copies = [row for row in summary['copy_records'] if row['setting_index'] == setting['setting_index']]
            capped = np.asarray([row['capped_first_goal_steps'] for row in copies], np.float64)
            groups = [row['field_seed'] if spec['method'] in NETWORK_METHODS else row['policy_seed'] for row in copies]
            rows.append({'wave': spec['wave'].removeprefix('fixed-'), 'protocol': spec.get('protocol'),
                'kind': spec['geometry']['kind'], 'policy': spec['policy_kind'], 'condition': spec['condition'],
                'method': spec['method'], 'configuration': setting['configuration'], 'copies': copies,
                'unit': path.parent.name, 'seeds': len(copies), 'mean': float(capped.mean()),
                'interval': grouped_mean_interval(capped, groups, 7) if len(copies) > 1 else [None, None],
                'discovery': float(np.mean([row['first_goal_step'] >= 0 for row in copies])),
                'coverage': float(np.mean([row['coverage'] for row in copies])),
                'sampled': float(np.mean([row['evaluation_goal_rate_sampled'] for row in copies])),
                'greedy': float(np.mean([row['evaluation_goal_rate_greedy'] for row in copies]))})
    return rows


def number(value):
    """Round large step counts to whole numbers with thousands separators."""
    return f'{value:,.0f}'


def mark(values, index, smaller_better):
    """Bold the best and underline the second best of one column."""
    order = sorted(set(values), reverse=not smaller_better)
    if values[index] == order[0]:
        return 'best'
    return 'second' if len(order) > 1 and values[index] == order[1] else None


def styled(text, style, latex=False):
    """Apply bold or underline in Markdown or LaTeX."""
    if style == 'best':
        return f'\\textbf{{{text}}}' if latex else f'**{text}**'
    if style == 'second':
        return f'\\underline{{{text}}}' if latex else f'<u>{text}</u>'
    return text


def setting_text(configuration):
    """Learning rate, entropy and intrinsic weight of a chosen configuration, in words."""
    exponent = int(round(np.log2(configuration['intrinsic_coefficient'])))
    return (f"learning rate {configuration['learning_rate']:g}, entropy {configuration['entropy_coefficient']:g}, "
            f"weight 2^{exponent}")


def results_table(rows, latex=False):
    """Conditions in rows sorted by mean capped first-goal time; metrics in columns."""
    rows = sorted(rows, key=lambda row: (row['mean'], -row['discovery']))
    columns = [('mean', True), ('discovery', False), ('coverage', False), ('sampled', False), ('greedy', False)]
    lines = []
    for index, row in enumerate(rows):
        low, high = row['interval']
        cells = [condition_label(row['condition'])]
        for name, smaller in columns:
            values = [item[name] for item in rows]
            # Style only the value: an underlined phrase cannot wrap, so the interval stays plain.
            # Before: '\underline{163,714 [70,645, 294,237]}'; after: '\underline{163,714} [70,645, 294,237]'.
            text = number(row[name]) if name == 'mean' else f'{100*row[name]:.1f}%'
            interval = f' [{number(low)}, {number(high)}]' if name == 'mean' and low is not None else ''
            cells.append(styled(text.replace('%', '\\%') if latex else text, mark(values, index, smaller), latex)
                         + interval)
        # The weight is set as math in LaTeX so it never breaks across lines.
        # Before: 'weight 2^-6'; after: 'weight $2^{-6}$'.
        text = setting_text(row['configuration'])
        cells.append(text.replace(text.rsplit(' ', 1)[1], f"$2^{{{text.rsplit('^', 1)[1]}}}$") if latex else text)
        cells.append(str(row['seeds']))
        lines.append(cells)
    return lines


def markdown_table(rows):
    """Render one results table in Markdown with the sorted column's arrow."""
    header = ('| condition | mean capped first-goal steps ↓ [95%] | goal found | coverage | '
              'final-policy goal rate, sampled | greedy | configuration | seeds |\n|---|---|---|---|---|---|---|---|\n')
    return header + ''.join('| ' + ' | '.join(cells) + ' |\n' for cells in results_table(rows))


def latex_table(rows, caption, label):
    """Render one results table in LaTeX for the development document."""
    body = ''.join(' & '.join(cells) + ' \\\\\n' for cells in results_table(rows, latex=True))
    # Fit the text width (13.97 cm): 12.0 cm of columns plus 14 gaps of 3 pt (1.5 cm); the
    # earlier 13.5 cm of columns with 6 pt gaps ran 82.6 pt into the margin.
    return ('\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\begin{tabular}{@{}' + ''.join(f'>{{\\raggedright\\arraybackslash}}p{{{width}cm}}' for width in
                                              (3.3, 2.3, 0.8, 0.9, 1.0, 0.8, 2.3, 0.6)) + '@{}}\\toprule\n'
            'condition & mean capped first-goal steps $\\downarrow$ [95\\%] & goal found & coverage & '
            'final goal rate, sampled & greedy & configuration & seeds \\\\\\midrule\n' + body +
            '\\bottomrule\\end{tabular}\\end{table}\n')


def development_figure(rows, kind, policy, plots):
    """Mean capped first-goal time against the intrinsic weight, one panel per condition."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [condition_name(row) for row in conditions() if row['method'] != 'no-bonus']
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
    baseline = [row for row in rows if row['method'] == 'no-bonus']
    for axis, name in zip(axes.flat, names):
        for color, (rate, entropy) in zip(COLORS, [(r, e) for r in LEARNING_RATES for e in ENTROPIES]):
            points = sorted((row['configuration']['intrinsic_coefficient'], row['mean']) for row in rows
                            if row['condition'] == name and row['configuration']['learning_rate'] == rate
                            and row['configuration']['entropy_coefficient'] == entropy)
            if points:
                axis.plot(*zip(*points), marker='o', markersize=3, color=color,
                          label=f'learning rate {rate:g}, entropy {entropy:g}')
        for color, row in zip(COLORS, sorted(baseline, key=lambda r: (r['configuration']['learning_rate'],
                                                                      r['configuration']['entropy_coefficient']))):
            axis.axhline(row['mean'], color=color, linestyle=':', linewidth=1)
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_title(condition_label(name), fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('intrinsic weight')
    for axis in axes[:, 0]:
        axis.set_ylabel('mean capped first-goal steps')
    axes.flat[0].legend(fontsize=7)
    figure.suptitle(f'Development, {CASE_LABELS[kind]}, {policy} policy, 16 seeds per point '
                    '(dotted: no bonus at each learning rate and entropy)', fontsize=10)
    figure.tight_layout()
    name = f'development_intrinsic-weight_{kind}_{policy}'
    for suffix in ('png', 'pdf'):
        figure.savefig(plots/f'{name}.{suffix}', dpi=130)
    plt.close(figure)
    return name


def decay_records(run_dir, rows):
    """Per online-RND confirmation row: relative error e_t/e_0 and visits N_t at every checkpoint."""
    output = []
    for row in rows:
        if row['method'] != 'online-rnd':
            continue
        data = np.load(run_dir/'data/research'/row['unit']/'behavior.npz')
        ratio = data['rnd_error_history'] / data['rnd_error_initial'][None]
        output.append({**row, 'ratio': ratio, 'visits': data['visit_count_history'],
                       'steps': data['visit_count_history_steps']})
    return output


def fitted_slopes(ratio, visits):
    """Least-squares slope of log(e_t/e_0) on log(N_t) for every cell visited at least twice."""
    # Before: [checkpoints, copies, cells]; after: one slope per (copy, cell) with >= 3 usable points.
    slopes = []
    for copy in range(ratio.shape[1]):
        for cell in range(ratio.shape[2]):
            n, q = visits[:, copy, cell], ratio[:, copy, cell]
            usable = (n >= 2) & (q > 0)
            if usable.sum() >= 3 and np.ptp(np.log(n[usable])) > 0:
                slopes.append(np.polyfit(np.log(n[usable]), np.log(q[usable]), 1)[0])
    return np.asarray(slopes)


def decay_figure(records, kind, policy, plots):
    """Online RND relative error against visits on log-log axes, beside the 1/sqrt(n) rule."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 5))
    for color, record in zip(COLORS, records):
        n, q = record['visits'].ravel(), record['ratio'].ravel()
        keep = (n >= 1) & (q > 0)
        bins = np.unique(np.round(np.logspace(0, np.log10(max(n[keep].max(), 2)), 30)))
        index = np.digitize(n[keep], bins)
        centers = [np.median(n[keep][index == b]) for b in np.unique(index)]
        medians = [np.median(q[keep][index == b]) for b in np.unique(index)]
        axis.plot(centers, medians, color=color, marker='o', markersize=3, label=condition_label(record['condition']))
    grid = np.logspace(0, 6, 50)
    axis.plot(grid, grid ** -0.5, color='black', linestyle='--', label='count rule: 1/sqrt(visits)')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel('visits to the cell, N')
    axis.set_ylabel('online RND error / its value at step 0 (median over cells and seeds)')
    axis.set_title(f'{CASE_LABELS[kind]}, {policy} policy, confirmation seeds', fontsize=10)
    axis.legend(fontsize=7)
    figure.tight_layout()
    name = f'decay_online-rnd_{kind}_{policy}'
    for suffix in ('png', 'pdf'):
        figure.savefig(plots/f'{name}.{suffix}', dpi=130)
    plt.close(figure)
    return name


def generate(run_dir):
    """Write analysis.md, analysis.html, results.tex and the figures from the finished records."""
    analysis, plots = run_dir/'analysis', run_dir/'analysis/plots'
    plots.mkdir(parents=True, exist_ok=True)
    rows = collect(run_dir)
    state = read_json(run_dir/'campaign_state.json') if (run_dir/'campaign_state.json').exists() else {}
    by_case = defaultdict(list)
    for row in rows:
        by_case[(row['kind'], row['policy'], row['wave'], row['protocol'])].append(row)
    markdown = [f"# Fixed CleanRL PPO-RND settings on the straight and snake grids\n\nGenerated {pacific()} "
                "(times are Pacific, converted from machines running Eastern). Phase: "
                f"{state.get('phase', 'not started')}. Finished settings: {len(rows)}.\n\n"]
    markdown.append(
        "## Metrics\n\nDefine $K$ as the number of policies of a configuration, $T_i$ as the training transition at "
        "which policy $i$ first arrives at the goal cell, and $B = 16{,}777{,}216$ as the budget. Configurations are "
        "chosen by the mean capped first-goal time $M(B) = \\dfrac{1}{K} \\sum_{i=1}^{K} \\min(T_i, B)$, with ties "
        "broken by the goal discovery rate $F(B) = \\dfrac{1}{K} \\sum_{i=1}^{K} \\mathbf{1}[T_i \\le B]$. Coverage is "
        "the fraction of path cells reached in training by $B$. The final-policy goal rate is the fraction of 32 "
        "evaluation episodes (actions sampled from the final policy, no learning) that reach the goal; the greedy "
        "column uses one episode with the most likely action. Intervals are 95% bootstrap intervals; network "
        "conditions resample field seeds (four policies share a field), the others resample policy seeds.\n\n")
    markdown.append(
        "## Settings\n\nAll conditions use CleanRL PPO-RND's scalar settings: rollout 128, 4 epochs, 4 minibatches, "
        "extrinsic discount 0.999, intrinsic discount 0.99, GAE 0.95, clip 0.1, value coefficient 0.5, maximum "
        "gradient norm 0.5, Adam epsilon 1e-5, extrinsic advantage weight 2, linear learning-rate annealing. Swept "
        "per condition: learning rate {1e-4, 1e-3} × entropy {0.001, 0.01} × intrinsic weight 2^-6 … 2^6 (13 "
        "values). Count-based fields are scaled to average 1 over cells and divided by √visits; online RND uses the "
        "raw CleanRL prediction error (half the summed squared error of 512 features). State normalization whitens "
        "the RND networks' input coordinates with running statistics, started by 50 random-action rollouts of 128 "
        "steps; reward normalization divides the bonus by the running standard deviation of its discounted sum. "
        "Development: 16 seeds per configuration; confirmation: 128 fresh seeds (20000-20127).\n\n")
    tex = []
    for protocol, title in (('chosen', 'chosen configuration of each condition'),
                            ('default-cell', 'learning rate 1e-4, entropy 0.001, best intrinsic weight')):
        section = [f"## Confirmation, {title}\n\n"]
        for kind, policy in CASE_ORDER:
            selected = by_case.get((kind, policy, 'confirmation', protocol), [])
            if protocol == 'default-cell':
                # A condition whose chosen configuration is already in the default cell reuses that run.
                frozen = read_json(run_dir/'confirmation_frozen.json') if (run_dir/'confirmation_frozen.json').exists() else []
                reused = {row['condition'] for row in frozen if row['same_as_chosen'] and row['geometry']['kind'] == kind
                          and row['policy_kind'] == policy}
                selected = selected + [row for row in by_case.get((kind, policy, 'confirmation', 'chosen'), [])
                                       if row['condition'] in reused]
            section.append(f"### {CASE_LABELS[kind]}, {policy} policy\n\n")
            if not selected:
                section.append('Not run yet.\n\n')
                continue
            section.append(markdown_table(selected) + '\n')
            tex.append(latex_table(selected, f'Confirmation ({title}), {CASE_LABELS[kind]}, {policy} policy, '
                                   '128 seeds per condition. Best in bold, second best underlined, per column.',
                                   f'tab:fixed-confirmation-{protocol}-{kind}-{policy}'))
        markdown.append(''.join(section))
    section = ["## Development\n\n"]
    for kind, policy in CASE_ORDER:
        selected = by_case.get((kind, policy, 'development', None), [])
        section.append(f"### {CASE_LABELS[kind]}, {policy} policy\n\n")
        if not selected:
            section.append('Not run yet.\n\n')
            continue
        best = {}
        for row in selected:
            if row['condition'] not in best or (row['mean'], -row['discovery']) < (best[row['condition']]['mean'], -best[row['condition']]['discovery']):
                best[row['condition']] = row
        section.append(f'Best configuration of each condition ({len(selected)} of 628 settings finished):\n\n'
                       + markdown_table(list(best.values())) + '\n')
        name = development_figure(selected, kind, policy, plots)
        section.append(f'![development, {kind}, {policy}](plots/{name}.png)\n\n')
    markdown.append(''.join(section))
    section = ["## Decay: online RND against the count rule\n\n"
               "Define $e_t(s)$ as the online RND error of cell $s$ after $t$ updates and $N_t(s)$ as its visits. The "
               "figures plot $e_t(s)/e_0(s)$ against $N_t(s)$ for the chosen confirmation configurations, beside "
               "$N^{-1/2}$, the count rule's decay. Slopes are least-squares fits of $\\log(e_t/e_0)$ on $\\log N_t$ "
               "per cell and seed (cells with at least three checkpoints of 2 or more visits); the count rule's slope "
               "is -0.5.\n\n"]
    for kind, policy in CASE_ORDER:
        records = decay_records(run_dir, by_case.get((kind, policy, 'confirmation', 'chosen'), []))
        section.append(f"### {CASE_LABELS[kind]}, {policy} policy\n\n")
        if not records:
            section.append('Not run yet.\n\n')
            continue
        name = decay_figure(records, kind, policy, plots)
        section.append(f'![decay, {kind}, {policy}](plots/{name}.png)\n\n| condition | median slope | '
                       'middle half of slopes | fitted cells |\n|---|---|---|---|\n')
        for record in records:
            slopes = fitted_slopes(record['ratio'], record['visits'])
            if len(slopes):
                low, high = np.quantile(slopes, [0.25, 0.75])
                section.append(f"| {condition_label(record['condition'])} | {np.median(slopes):.2f} | "
                               f"{low:.2f} to {high:.2f} | {len(slopes)} |\n")
        section.append('\n')
        tex.append('\\begin{figure}[H]\\centering\\includegraphics[width=0.7\\linewidth]{\\FixedSettingsRunPath/analysis/plots/'
                   + name + '.pdf}\\caption{Online RND relative error against visits, ' + CASE_LABELS[kind] + ', '
                   + policy + ' policy; dashed: $N^{-1/2}$.}\\end{figure}\n')
    markdown.append(''.join(section))
    markdown.append(f"## Records\n\nUnit records: `data/research/fixed-*/summary.json` and `behavior.npz`. Frozen "
                    f"confirmation choices: `confirmation_frozen.json`. Report generated {pacific()}.\n")
    _write_report_with_history(analysis, ''.join(markdown))
    (analysis/'results.tex').write_text('% Generated by fixed_settings_report.py; do not edit.\n' + ''.join(tex))
    write_json(analysis/'report_record.json', {'at': now(), 'settings': len(rows)})


def main():
    """Generate the report from the run folder's records."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    generate(args.run_dir)


if __name__ == '__main__':
    main()
