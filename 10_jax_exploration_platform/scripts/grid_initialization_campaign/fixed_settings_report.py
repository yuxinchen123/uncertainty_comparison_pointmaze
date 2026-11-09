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
# Seeds removed from EACH end before the trimmed mean, per 128 seeds (16 seeds: 1 per end).
TRIM_PER_128 = 10
TRIM_CHOICES = (0, 2, 4, 6, 8, 10, 13, 16, 20)


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


def trim_count(seeds, per_128=TRIM_PER_128):
    """Seeds removed from each end: per_128 scaled to the number of seeds, rounded down."""
    # Before: 128 seeds -> 10 per end; after scaling: 16 seeds -> 1 per end.
    return seeds * per_128 // 128


def trimmed_mean(values, per_end):
    """Mean after dropping the per_end smallest and per_end largest values."""
    ordered = np.sort(np.asarray(values, np.float64))
    return float(ordered[per_end:len(ordered)-per_end].mean())


def grouped_standard_errors(values, group_ids, per_end, seed, repetitions=2000):
    """Bootstrap standard errors of the mean and of the trimmed mean, resampling whole groups."""
    # Network conditions: 32 field seeds of 4 policies, resampled as groups of 4; others: single policies.
    values = np.asarray(values, np.float64)
    unique, inverse = np.unique(np.asarray(group_ids), return_inverse=True)
    if len(unique) < 2:
        return None, None
    members = [values[inverse == index] for index in range(len(unique))]
    generator = np.random.default_rng(seed)
    chosen = generator.integers(0, len(unique), size=(repetitions, len(unique)))
    if len({len(group) for group in members}) == 1:
        pooled = np.sort(np.stack(members)[chosen].reshape(repetitions, -1), axis=1)
        means = pooled.mean(axis=1)
        trimmed = pooled[:, per_end:pooled.shape[1]-per_end].mean(axis=1)
    else:
        samples = [np.concatenate([members[index] for index in draw]) for draw in chosen]
        means = np.asarray([sample.mean() for sample in samples])
        trimmed = np.asarray([trimmed_mean(sample, trim_count(len(sample))) for sample in samples])
    return float(means.std(ddof=1)), float(trimmed.std(ddof=1))


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
                'trim': trim_count(len(copies)), 'trimmed': trimmed_mean(capped, trim_count(len(copies))),
                'capped_seeds': int(np.sum(capped >= BUDGET)), 'values': capped,
                'discovery': float(np.mean([row['first_goal_step'] >= 0 for row in copies])),
                'coverage': float(np.mean([row['coverage'] for row in copies])),
                'sampled': float(np.mean([row['evaluation_goal_rate_sampled'] for row in copies])),
                'greedy': float(np.mean([row['evaluation_goal_rate_greedy'] for row in copies]))})
            rows[-1]['standard_error'], rows[-1]['trimmed_standard_error'] = grouped_standard_errors(
                capped, groups, rows[-1]['trim'], 7)
    return rows


def number(value, thousands=False):
    """A step count: in thousands with two decimals (8.11K) when the table allows, else a whole number."""
    return f'{value/1000:,.2f}K' if thousands else f'{value:,.0f}'


def in_thousands(rows):
    """Write a table's step counts in thousands when every mean, trimmed mean and interval end is at least 1,000."""
    values = [row[name] for row in rows for name in ('mean', 'trimmed')]
    values += [end for row in rows for end in row['interval'] if end is not None]
    return min(values) >= 1000


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
    columns = [('mean', True), ('trimmed', True), ('discovery', False), ('coverage', False), ('sampled', False),
               ('greedy', False)]
    thousands = in_thousands(rows)
    plus_minus = '$\\pm$' if latex else '±'
    lines = []
    for index, row in enumerate(rows):
        low, high = row['interval']
        cells = [condition_label(row['condition'])]
        for name, smaller in columns:
            values = [item[name] for item in rows]
            # Style only the value: an underlined phrase cannot wrap, so the error and interval stay plain.
            # Before: '\underline{163,714} [70,645, 294,237]'; after: '\underline{163.71K} ± 56.40K [70.65K, 294.24K]'.
            text = number(row[name], thousands) if name in ('mean', 'trimmed') else f'{100*row[name]:.1f}%'
            extra = ''
            error = row.get({'mean': 'standard_error', 'trimmed': 'trimmed_standard_error'}.get(name, ''))
            if error is not None and name in ('mean', 'trimmed'):
                extra += f' {plus_minus} {number(error, thousands)}'
            if name == 'mean' and low is not None:
                extra += f' [{number(low, thousands)}, {number(high, thousands)}]'
            cells.append(styled(text.replace('%', '\\%') if latex else text, mark(values, index, smaller), latex)
                         + extra)
        # The weight is set as math in LaTeX so it never breaks across lines.
        # Before: 'weight 2^-6'; after: 'weight $2^{-6}$'.
        text = setting_text(row['configuration'])
        cells.append(text.replace(text.rsplit(' ', 1)[1], f"$2^{{{text.rsplit('^', 1)[1]}}}$") if latex else text)
        cells.append(str(row['seeds']))
        lines.append(cells)
    return lines


def markdown_table(rows):
    """Render one results table in Markdown with the sorted column's arrow."""
    trim = max(row['trim'] for row in rows)
    header = ('| condition | mean capped first-goal steps ↓ ± standard error [95%] | '
              f'trimmed mean ({trim} lowest and {trim} highest seeds removed) ± standard error | goal found | coverage | '
              'final-policy goal rate, sampled | greedy | configuration | seeds |\n|---|---|---|---|---|---|---|---|---|\n')
    return header + ''.join('| ' + ' | '.join(cells) + ' |\n' for cells in results_table(rows))


def latex_table(rows, caption, label):
    """Render one results table in LaTeX for the development document."""
    body = ''.join(' & '.join(cells) + ' \\\\\n' for cells in results_table(rows, latex=True))
    # Fit the text width (13.97 cm): 12.2 cm of columns plus 16 gaps of 3 pt (1.69 cm); the
    # earlier 13.5 cm of columns with 6 pt gaps ran 82.6 pt into the margin.
    # Before: 8 columns (3.3, 2.3, 0.8, 0.9, 1.0, 0.8, 2.3, 0.6); after: a trimmed-mean column added.
    trim = max(row['trim'] for row in rows)
    return ('\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\begin{tabular}{@{}' + ''.join(f'>{{\\raggedright\\arraybackslash}}p{{{width}cm}}' for width in
                                              (2.8, 2.1, 1.5, 0.8, 0.9, 1.0, 0.8, 1.7, 0.6)) + '@{}}\\toprule\n'
            'condition & mean capped first-goal steps $\\downarrow$ $\\pm$ standard error [95\\%] & '
            f'trimmed mean, {trim} lowest and {trim} highest removed, $\\pm$ standard error & goal found & coverage & '
            'final goal rate, sampled & greedy & configuration & seeds \\\\\\midrule\n' + body +
            '\\bottomrule\\end{tabular}\\end{table}\n')


def best_by_trim(rows, per_end):
    """The two conditions with the smallest trimmed mean when per_end seeds are removed from each end."""
    ranked = sorted(rows, key=lambda row: (trimmed_mean(row['values'], per_end), -row['discovery']))
    return [(row, trimmed_mean(row['values'], per_end)) for row in ranked[:2]]


def tail_counts(values):
    """Seeds more than three interquartile ranges above the third quartile, and below the first."""
    first, third = np.quantile(values, [0.25, 0.75])
    spread = third - first
    return int(np.sum(values > third + 3*spread)), int(np.sum(values < first - 3*spread))


def trimming_section(cases):
    """Markdown and LaTeX discussing how many seeds to trim and whether the best condition changes."""
    # cases: [(kind, policy, rows of the 128-seed chosen confirmation)], in CASE_ORDER.
    tops = [row for _, _, rows in cases for row in sorted(rows, key=lambda row: row['mean'])[:4]]
    high = max(tail_counts(row['values'])[0] for row in tops)
    low = max(tail_counts(row['values'])[1] for row in tops)
    change = lambda row, a, b: abs(trimmed_mean(row['values'], b)/trimmed_mean(row['values'], a) - 1)
    markdown = [
        "## Trimmed means: how many seeds to remove, and whether the results change\n\n"
        "Define $v_{(1)} \\le \\dots \\le v_{(K)}$ as the sorted capped first-goal times $\\min(T_i, B)$ of one "
        "configuration. The trimmed mean removing $k$ seeds from each end is "
        "$M_k(B) = \\dfrac{1}{K-2k} \\sum_{i=k+1}^{K-k} v_{(i)}$; $M_0(B) = M(B)$.\n\n"
        f"**The extreme values are on the high side.** Among the four best conditions of each case (by $M(B)$, 16 "
        f"conditions), up to {high} of 128 seeds lie more than three interquartile ranges above the third quartile, "
        + (f"and at most {low} lie that far below the first quartile. " if low else "and none lie that far below the "
           "first quartile. ") + f"A seed that never reaches the goal counts as "
        f"$B$ = {BUDGET:,}, so one such seed adds {BUDGET/128/1000:,.1f}K to a 128-seed mean: a single failure can "
        "decide the ranking of conditions whose typical seed needs 10K to 100K transitions.\n\n"
        f"**Chosen: {TRIM_PER_128} seeds from each end of 128** (7.8% per end, 108 seeds kept; 1 per end of the 16 "
        "development seeds). Removing 2 or 4 leaves the single failures of several conditions in the mean. Between "
        f"6 and {TRIM_PER_128} removed per end, the trimmed means of these 16 conditions still move by up to "
        f"{100*max(change(row, 6, TRIM_PER_128) for row in tops):.1f}%; between {TRIM_PER_128} and 13 (10% per end) "
        f"by up to {100*max(change(row, TRIM_PER_128, 13) for row in tops):.1f}%. Removing more than about 10% per "
        "end removes every failure of conditions that fail on up to that many seeds, so they look as good as "
        "conditions that never fail. Trimming also removes exactly the never-found seeds that $M(B)$ is meant to "
        "count, so configurations are still chosen by the untrimmed $M(B)$; the trimmed mean is reported beside "
        "it.\n\n"]
    tex_rows = []
    for kind, policy, rows in cases:
        if not rows:
            continue
        markdown.append(f"### {CASE_LABELS[kind]}, {policy} policy\n\n| seeds removed from each end | best condition "
                        "| its trimmed mean | second condition | its trimmed mean |\n|---|---|---|---|---|\n")
        best = {}
        for per_end in TRIM_CHOICES:
            (first, first_value), (second, second_value) = best_by_trim(rows, per_end)
            best[per_end] = first
            markdown.append(f"| {per_end} | {condition_label(first['condition'])} | {number(first_value, True)} | "
                            f"{condition_label(second['condition'])} | {number(second_value, True)} |\n")
        changes = [per_end for per_end in TRIM_CHOICES if best[per_end]['condition'] != best[0]['condition']]
        (chosen, chosen_value), (runner, runner_value) = best_by_trim(rows, TRIM_PER_128)
        if not changes:
            verdict = 'The best condition is the same for every number removed, up to 20 per end.'
        else:
            verdict = f"The best condition differs from the untrimmed one at {', '.join(map(str, changes))} removed per end."
        verdict += (f" With {TRIM_PER_128} removed, {condition_label(chosen['condition'])} leads "
                    f"{condition_label(runner['condition'])} by {number(runner_value - chosen_value, True)}; their "
                    f"standard errors are {number(chosen['trimmed_standard_error'], True)} and "
                    f"{number(runner['trimmed_standard_error'], True)}.")
        if chosen['condition'] != best[0]['condition']:
            verdict += (f" It is not the untrimmed best ({condition_label(best[0]['condition'])}): "
                        f"{chosen['capped_seeds']} of its 128 seeds never reached the goal, against "
                        f"{best[0]['capped_seeds']}; each such seed adds {BUDGET/128/1000:,.1f}K to an untrimmed mean.")
        markdown.append(f"\n{verdict}\n\n")
        tex_rows.append(f"{kind} grid, {policy} & {condition_label(best[0]['condition'])}, "
                        f"{number(best[0]['mean'], True)} & {condition_label(chosen['condition'])}, "
                        f"{number(trimmed_mean(chosen['values'], TRIM_PER_128), True)} & "
                        + (', '.join(map(str, changes)) if changes else 'none') + ' \\\\\n')
    tex = ('\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n\\caption{Best condition of each '
           f'case by the mean capped first-goal time and by the trimmed mean with {TRIM_PER_128} of 128 seeds removed '
           'from each end (chosen configurations, 128 seeds). Extreme values are mostly seeds that need millions of '
           'transitions or never reach the goal; configurations are still chosen by the untrimmed mean.}'
           '\\label{tab:fixed-trimming}\n\\begin{tabular}{@{}' + ''.join(
               f'>{{\\raggedright\\arraybackslash}}p{{{width}cm}}' for width in (2.2, 4.4, 4.4, 2.2)) + '@{}}\\toprule\n'
           f'case & best, untrimmed mean & best, trimmed mean ({TRIM_PER_128} per end) & seeds removed per end (of '
           f"{', '.join(map(str, TRIM_CHOICES[1:]))}) at which the best differs from the untrimmed best "
           '\\\\\\midrule\n' + ''.join(tex_rows) + '\\bottomrule\\end{tabular}\\end{table}\n')
    return ''.join(markdown), tex


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
        "column uses one episode with the most likely action. Intervals are 95% bootstrap intervals and standard "
        "errors are bootstrap standard errors (2,000 resamples); network conditions resample field seeds (four "
        "policies share a field), the others resample policy seeds. The trimmed mean drops the "
        f"{TRIM_PER_128} smallest and {TRIM_PER_128} largest capped first-goal times of 128 seeds (1 of 16 in "
        "development) before averaging; it is defined, and the number removed is discussed, in the section on "
        "trimmed means. Step counts are written in thousands (8.11K = 8,110) when every value in a table is at least "
        "1,000.\n\n")
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
    trimming_markdown, trimming_tex = trimming_section(
        [(kind, policy, by_case.get((kind, policy, 'confirmation', 'chosen'), [])) for kind, policy in CASE_ORDER])
    markdown.append(trimming_markdown)
    tex.append(trimming_tex)
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
