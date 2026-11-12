"""Generate a maze-campaign run folder's report: analysis/analysis.md, analysis/results.tex and analysis/plots/.

Sections of analysis.md (each top-level section has a row in the contents table, marked read / unread / updated):
the measures; the records read; the rollout-shape check (when a folder holds several numbers of environments per
policy); the development sweeps (configurations in rows sorted by the whole-curve mean); the budget the plan's rule
proposes from each finished sweep of the baseline; the training curves; the comparison with the baseline on the
confirmation seeds; the final level against the training length; the decline check (chosen configurations whose
final level is below 90% of the peak of their smoothed curve).

Usage, on a compute node (numpy and plotting never run on a login node):
  PYTHONPATH=<platform>/scripts <python> -m maze_campaign.analysis.report --run-dir <run folder> [--at-steps N]
Through the test helper's Slurm job: see report_job.py beside this file.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from . import compare, contents, figures, measures, records
from .records import ENVIRONMENT_ORDER, REWARD_SETTING_NAMES, REWARD_SETTING_ORDER

DECLINE_SHARE = 0.9
LATEX_SWEEP_ROWS = 10
# The rollout-shape check's table of every configuration has one column per setting; more than this many do not
# fit across a page.
MOST_SETTING_COLUMNS = 6
# The development document's text width, and the gap LaTeX puts between two columns at 3 pt per side.
TEXT_WIDTH_CM, COLUMN_GAP_CM = 13.97, 0.212
MAIN_SETTINGS = ('int_coef', 'learning_rate', 'ent_coef')
# The hue of a number of environments per policy is its place in this list, so adding a number to a check never
# repaints the others: the three of the plan first, then further ones.
WIDTH_HUE_ORDER = (4, 16, 128, 1, 2, 8, 32, 64)
# The measure columns of a sweep or comparison table after W and L: (key, whether higher is better, header).
OTHER_MEASURES = (
    ('final_over_peak', True, 'final level over peak'),
    ('capped_first_goal_step', False, 'mean capped first-goal step'),
    ('reached_goal_share', True, 'policies that reach the goal'),
    ('coverage', True, 'maze cells reached'),
    ('steps_to_goal', False, 'mean steps to the goal, last 10%'),
    ('final_goal_share', True, 'episodes that reach the goal, last 10%'))

MEASURES_TEXT = r"""## Measures

Define $K$ as the number of policies (seeds) of a configuration, $B$ as the budget in environment steps per policy, $R_i(s)$ as the mean return of the training episodes of policy $i$ that end in the logging window containing step $s$, and $\Delta s$ as the number of a logging window's steps that lie inside the range being averaged. The return is of the reward the learner sees. With reward 1 on arrival an episode returns 1 if it reaches the goal and 0 otherwise. With the step cost it returns $-(t-1)/H$ when it reaches the goal at step $t$ of an episode limited to $H$ steps, and $-1$ otherwise. Define $R_0$ as the return of an episode that never reaches the goal ($R_0 = 0$ with reward 1 on arrival, $R_0 = -1$ with the step cost).

A window in which no episode ends takes the value of the window before it; before a policy's first finished episode the value is $R_0$. A window that straddles an end of a range counts in proportion to its steps inside the range.

1. **Training curve:** $\bar R(s) = \dfrac{1}{K}\sum_{i=1}^{K} R_i(s)$. Bands in the figures are one standard error over the $K$ policies.
2. **Whole-curve mean, which chooses configurations:** $W(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{B}\sum_{s \le B} R_i(s) \thinspace \Delta s$, the area under the curve divided by the budget.
3. **Final level:** $L(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{0.1 B}\sum_{0.9 B \lt s \le B} R_i(s) \thinspace \Delta s$, the same mean over the last 10% of the budget. The final level at a share $f$ of the budget is $L(f B)$ read from the same runs: the mean over the steps from $0.9 f B$ to $f B$.
4. **Final level over peak:** define $S(s)$ as the mean of the seed-averaged curve $\bar R$ over the $0.1 B$ steps that end at step $s$, and $P(B) = \max_{0.1 B \le s \le B} S(s)$ as its highest value. The ratio is $\dfrac{L(B) - R_0}{P(B) - R_0}$; both are measured from $R_0$, so the ratio means the same in both reward settings. The smoothing has the width of the final level, so $S(B) = L(B)$ and the ratio is at most 1. The decline check also gives the ratio with the highest single window of $\bar R$ in place of $P(B)$; that ratio is the stricter one, because one window of a few seeds is raised by noise.
5. **Mean capped first-goal step:** define $T_i$ as the training step of policy $i$'s first arrival at the goal ($T_i = \infty$ if it has none). The measure is $M(B) = \dfrac{1}{K}\sum_{i=1}^{K} \min(T_i, B)$.
6. **Policies that reach the goal:** $F(B) = \dfrac{1}{K}\sum_{i=1}^{K} \mathbf{1}[T_i \le B]$.
7. **Maze cells reached:** the share of open maze cells a policy has reached in training by step $B$, averaged over the policies.
8. **Mean steps to the goal, last 10%:** the mean length of the configuration's training episodes that reach the goal in the last 10% of the budget, over all its policies together.
9. **Episodes that reach the goal, last 10%:** per policy, the share of its training episodes ending in the last 10% of the budget that reach the goal; then the mean over the policies.

Standard errors are over the $K$ policies. Measures 4 to 9 are reported beside $W$ and $L$ and never choose a configuration.

**Choice and comparison.** A configuration is chosen on the development seeds (below 30000) by the highest $W(B)$; ties go to the higher $L(B)$. On the confirmation seeds (30000 on), define $d_i$ as the value of seed $i$ under a method minus its value under the baseline, on the $K$ seeds both have. The tables give $\bar d = \dfrac{1}{K}\sum_{i=1}^{K} d_i$ for $W$ and for $L$, each with a 95% interval: the 2.5% and 97.5% points of the means of 10,000 resamples of the seeds, drawn with replacement. A difference is ahead when its interval is entirely above zero, behind when it is entirely below zero, and level otherwise. A method beats the baseline in one environment and reward setting when $W$ is ahead and $L$ is not behind. It succeeds when it beats the baseline in all five environments in one reward setting.

**Unfinished runs.** A table compares its rows at one step count: the planned steps per policy when at least one row has trained that far, else the largest step count every policy of the table has reached. Rows that have not reached it are listed under the table without measures. The learning rate falls to zero over the planned steps, so the measures of an unfinished run are not those of a run trained for fewer steps.

"""


def grouped(items, key):
    """Items collected by a key, the groups in the order their keys first appear."""
    # before: [a1, b1, a2] keyed by letter; after: {'a': [a1, a2], 'b': [b1]}.
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def case_position(environment, reward_setting):
    """Where a case stands in the report: environments in the study's order, reward 1 on arrival first."""
    return ENVIRONMENT_ORDER.index(environment), REWARD_SETTING_ORDER.index(reward_setting)


def compact_steps(value):
    """A step count for a table cell: 387.1K, or 1.48M from one million on."""
    return f'{value / 1e6:.2f}M' if value >= 1e6 else f'{value / 1e3:.1f}K' if value >= 1e3 else f'{value:.0f}'


def with_error(value, error):
    """A mean with its standard error; the mean alone when there is one seed."""
    return f'{value:.3f}' if error is None else f'{value:.3f} ± {error:.3f}'


def format_measure(name, value):
    """One of the other measures as a table cell."""
    if value is None:
        return 'not recorded' if name == 'coverage' else 'none'
    if name == 'capped_first_goal_step':
        return compact_steps(value)
    if name == 'steps_to_goal':
        return f'{value:.1f}'
    return f'{100 * value:.1f}%'


def column_marks(values, texts, higher_better):
    """Per row of one column: 'best', 'second' or None. Rows that show the same text share a mark."""
    # before: values [0.8904, 0.7041, 0.00004, 0.0] shown as ['0.890', '0.704', '0.000', '0.000'], higher is better
    # after:  ['best', 'second', None, None]; a column that shows one text only has no mark.
    shown = {}
    for value, text in zip(values, texts):
        if value is not None:
            shown[text] = (max if higher_better else min)(shown.get(text, value), value)
    order = sorted(shown, key=shown.get, reverse=higher_better)
    if len(order) < 2:
        return [None] * len(values)
    return [None if value is None else 'best' if text == order[0] else 'second' if text == order[1] else None
            for value, text in zip(values, texts)]


def mean_marks(values):
    """The marks of a column of means shown with three decimals, where the highest is the best."""
    return column_marks(values, [f'{value:.3f}' for value in values], True)


def styled(text, style):
    """Bold the best and underline the second best, in Markdown."""
    return f'**{text}**' if style == 'best' else f'<u>{text}</u>' if style == 'second' else text


def marked_mean(value, error, style):
    """A mean with its standard error, the mean bold or underlined; the error stays plain so the cell can wrap."""
    # before: (0.412, 0.031, 'best'); after: '**0.412** ± 0.031'.
    number, _, rest = with_error(value, error).partition(' ')
    return styled(number, style) + (f' {rest}' if rest else '')


def interval_text(result):
    """A paired difference with its 95% interval and its side."""
    # before: {'mean': 0.021, 'interval': [0.004, 0.038], 'side': 'ahead'}; after: '+0.021 [+0.004, +0.038], ahead'.
    low, high = result['interval']
    return f"{result['mean']:+.3f} [{low:+.3f}, {high:+.3f}], {result['side']}"


def markdown_table(header, rows):
    """A Markdown table from a header and rows of cells."""
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(cells) + ' |' for cells in rows]
    return '\n'.join(lines) + '\n\n'


def latex_text(text):
    """A Markdown table cell or sentence as LaTeX; formulas between dollar signs are kept as they are."""
    # before: '**0.412** ± 0.031', '<u>97.5%</u>', '-0.021 [-0.040, +0.002]', '$W$ ↑'
    # after:  '\textbf{0.412} $\pm$ 0.031', '\underline{97.5\%}', '$-$0.021 [$-$0.040, +0.002]', '$W$ $\uparrow$'
    pieces = re.split(r'(\$[^$]*\$)', text)
    for index in range(0, len(pieces), 2):
        piece = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', pieces[index])
        piece = re.sub(r'<u>(.+?)</u>', r'\\underline{\1}', piece)
        piece = re.sub(r'(?<![\w.])-(?=\d)', '$-$', piece)
        for symbol, command in (('%', '\\%'), ('&', '\\&'), ('#', '\\#'), ('_', '\\_'), ('±', '$\\pm$'),
                                ('↑', '$\\uparrow$'), ('×', '$\\times$')):
            piece = piece.replace(symbol, command)
        pieces[index] = piece
    return ''.join(pieces)


def latex_table(header, rows, caption, label, first_width=2.5, shares=None):
    """A LaTeX table that fits the development document's text width; long cells wrap inside their columns."""
    # The first column holds names; the others divide what is left of the text width by their shares (equal
    # shares when none are given). before: 4 columns, first 2.5 cm, shares (2, 1, 1); after: 2.5, 5.39, 2.70, 2.70 cm.
    count = len(header)
    shares = shares or [1.0] * (count - 1)
    left = TEXT_WIDTH_CM - 0.05 - COLUMN_GAP_CM * (count - 1) - first_width
    columns = ''.join(f'>{{\\raggedright\\arraybackslash\\hspace{{0pt}}}}p{{{width:.2f}cm}}'
                      for width in [first_width] + [left * share / sum(shares) for share in shares])
    body = '\\addlinespace[2pt]\n'.join(' & '.join(latex_text(cell) for cell in cells) + ' \\\\\n' for cells in rows)
    return ('\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
            f'\\caption{{{latex_text(caption)}}}\\label{{tab:maze-{label}}}\\vspace{{3pt}}\n'
            f'\\begin{{tabular}}{{@{{}}{columns}@{{}}}}\\toprule\n'
            + ' & '.join(latex_text(cell) for cell in header) + ' \\\\\\midrule\n' + body
            + '\\bottomrule\\end{tabular}\\end{table}\n')


def shown_words(shown, total):
    """How many of a sweep's configurations a LaTeX table holds, in words."""
    if shown < total:
        return f'the best {shown} of {total} configurations by the whole-curve mean $W$'
    return 'its only configuration' if total == 1 else f'all {total} configurations, sorted by the whole-curve mean $W$'


def latex_figure(name, caption):
    """A figure of the run folder's plots for the development document, which defines \\MazeCampaignRunPath."""
    return ('\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth]{\\MazeCampaignRunPath/analysis/plots/'
            f'{name}.pdf}}\\caption{{{latex_text(caption)}}}\\label{{fig:maze-{name}}}\\end{{figure}}\n')


def unfinished_note(steps, planned):
    """A sentence for a table whose rows are compared short of their planned steps; empty otherwise."""
    if steps >= planned:
        return ''
    return (f' **Unfinished runs:** the measures are read at {steps:,} of the {planned:,} planned steps per policy, '
            'and the learning rate has not reached zero there.')


def width_words(count):
    """A number of environments per policy in words."""
    return f"{count} environment{'' if count == 1 else 's'} per policy"


def row_name(group, names, with_method=False, with_width=False):
    """A table row's name in words: the method and rollout width when they differ in the table, then the settings."""
    parts = [group.method] if with_method else []
    if with_width:
        parts.append(width_words(group.environments_per_policy))
    if names:
        parts.append(records.describe_settings(group, names))
    return ', '.join(parts) or records.describe_settings(group, MAIN_SETTINGS)


def seeds_text(groups):
    """How many seeds the rows of a table have: one number, or the smallest and largest."""
    counts = sorted({len(group.policies) for group in groups})
    return f'{counts[0]} seeds' if len(counts) == 1 else f'{counts[0]} to {counts[-1]} seeds'


class Report:
    """One run folder's report while it is built: the policies, the measures computed so far, and the LaTeX."""

    def __init__(self, run, plots, folder_name, at_steps=None):
        """Group the policies of each stage into configurations once; every section reads these groups."""
        self.run, self.plots, self.folder_name, self.at_steps = run, plots, folder_name, at_steps
        self.groups = {stage: sorted(records.configurations([policy for policy in run.policies if policy.stage == stage]),
                                     key=lambda group: (*case_position(group.environment, group.reward_setting),
                                                        group.planned_steps, group.method,
                                                        group.environments_per_policy, compare.settings_order(group.settings)))
                       for stage in ('development', 'confirmation')}
        self.cache = {}
        self.choices = {}      # (environment, reward setting, planned steps, method) -> the development choice
        self.budgets = []      # what the budget rule proposes per finished sweep of the baseline
        self.checked = []      # rows of the decline check: (case words, stage words, group, steps)
        self.tex = ['% Generated by scripts/maze_campaign/analysis/report.py; do not edit.\n'
                    '% The including document defines \\MazeCampaignRunPath as this run folder.\n']
        # A method keeps one hue in every figure: the methods other than the baseline, in name order.
        methods = sorted({policy.method for policy in run.policies} - {run.baseline_method})
        self.method_position = {method: index for index, method in enumerate(methods)}

    def measured(self, group, steps):
        """A configuration's measures at a step count, computed once."""
        key = (id(group), steps)
        if key not in self.cache:
            self.cache[key] = measures.configuration_measures(group.policies, steps)
        return self.cache[key]

    def evaluation(self, groups):
        """The step count a table's rows are compared at, the rows that have reached it, and the rows that have not."""
        planned = min(group.planned_steps for group in groups)
        if self.at_steps is not None:
            steps = self.at_steps
        elif any(measures.common_steps(group.policies) >= planned for group in groups):
            steps = planned
        else:
            steps = measures.common_steps([policy for group in groups for policy in group.policies])
        ready = [group for group in groups if measures.common_steps(group.policies) >= steps]
        return steps, ready, [group for group in groups if group not in ready]

    def naming(self, groups):
        """How the rows of a table are named: the settings that differ, and whether the rollout width differs."""
        return records.varying_settings(groups), len({group.environments_per_policy for group in groups}) > 1

    def waiting_list(self, waiting, names, with_width, with_method=False):
        """Rows that have not reached the table's step count, as a list under the table."""
        if not waiting:
            return ''
        lines = ['Not at this step count yet:\n\n']
        for group in waiting:
            lines.append(f'- {row_name(group, names, with_method, with_width)}: '
                         f'{measures.common_steps(group.policies):,} steps so far.\n')
        return ''.join(lines) + '\n'

    def measure_rows(self, ranked, steps, name_of):
        """Per configuration, already in order: its name, W, L, the other measures and seeds as cells; best and second marked."""
        table = [self.measured(group, steps) for group in ranked]
        marks = {key: mean_marks([row[key] for row in table]) for key in ('whole_curve_mean', 'final_level')}
        for key, higher, _ in OTHER_MEASURES:
            marks[key] = column_marks([row[key] for row in table], [format_measure(key, row[key]) for row in table], higher)
        rows = []
        for index, (group, row) in enumerate(zip(ranked, table)):
            cells = {'name': name_of(group), 'seeds': str(row['seeds'])}
            for key in ('whole_curve_mean', 'final_level'):
                cells[key] = marked_mean(row[key], row[f'{key}_error'], marks[key][index])
            cells['others'] = [styled(format_measure(key, row[key]), marks[key][index]) for key, _, _ in OTHER_MEASURES]
            rows.append(cells)
        return rows

    # ---- sections -------------------------------------------------------------------------------------------

    def records_section(self):
        """What was read: one row per unit, in words, and the unit names in a list under the table."""
        units = self.run.units
        header = ['unit', 'environment', 'reward setting', 'method', 'environments per policy', 'policies',
                  'steps per policy so far', 'planned steps per policy', 'state']
        rows = []
        for index, unit in enumerate(units, start=1):
            state = unit['state']
            if unit['continued_from_a_checkpoint']:
                state += f"; continued from a checkpoint {unit['continued_from_a_checkpoint']} time(s)"
            if unit['started_from_zero_again']:
                state += f"; started from zero again {unit['started_from_zero_again']} time(s)"
            if unit['respawned_episodes']:
                state += f"; {unit['respawned_episodes']:,} episodes restarted after a non-finite physics state"
            planned = unit['planned_steps']
            rows.append([str(index), unit['environment'] or 'not known yet',
                         ' and '.join(REWARD_SETTING_NAMES[name] for name in unit['reward_settings']) or 'not known yet',
                         unit['method'] or 'not known yet', str(unit['environments_per_policy'] or 'not known yet'),
                         str(unit['policies']), f"{unit['steps_reached']:,}",
                         'not known yet' if planned is None else f'{planned:,}', state])
        complete = sum(unit['state'] == 'complete' for unit in units)
        text = [f"## Records read\n\n{len(units)} units, {complete} complete; {len(self.run.policies):,} policies with at "
                'least one logging window. A unit that was continued from a checkpoint keeps one unbroken sequence of '
                'windows; a unit that was started again from zero is read from its last start. Episodes restarted '
                'after a non-finite physics state count as episodes that reach the step limit, and stay in every '
                'sum (the records hold sums per window, not single episodes).\n\n', markdown_table(header, rows),
                'Unit names, by the number in the first column:\n\n']
        # A unit of another run folder (named in report_settings.json) is listed with that folder.
        for index, unit in enumerate(units, start=1):
            other = '' if unit['folder'] == self.folder_name else f" (run folder `{unit['folder']}`)"
            text.append(f"{index}. `{unit['unit_id']}`{other}\n")
        return ''.join(text) + '\n'

    def rollout_shape_section(self):
        """The rollout-shape check: W and L per number of environments per policy at equal environment steps."""
        text = []
        cases = grouped(self.groups['development'], lambda group: (group.environment, group.reward_setting, group.method))
        for (environment, setting, method), groups in cases.items():
            if len({group.environments_per_policy for group in groups}) < 2:
                continue
            steps, ready, waiting = self.evaluation(groups)
            if not ready:
                continue
            planned = min(group.planned_steps for group in groups)
            names = records.varying_settings(groups)
            case = f'{environment}, {REWARD_SETTING_NAMES[setting]}, {method}'
            name = figures.file_slug('rollout shape', environment, REWARD_SETTING_NAMES[setting], method)
            part = [f'### {case}\n\nEvery number of environments per policy is compared at the same {steps:,} '
                    f'environment steps per policy, with {seeds_text(ready)} per configuration.'
                    f'{unfinished_note(steps, planned)} Each row is one number of environments per policy at its own '
                    'best configuration (highest $W$, ties by $L$); rows are sorted by that $W$.\n\n',
                    self.rollout_tables(ready, names, steps, case, name),
                    self.waiting_list(waiting, names, True)]

            # While some numbers of environments per policy are still training, all of them are also compared at
            # the steps every policy has reached.
            common = measures.common_steps([policy for group in groups for policy in group.policies])
            if waiting and self.at_steps is None:
                part += [f'All numbers of environments per policy at the {common:,} environment steps that every '
                         f'policy has reached.{unfinished_note(common, planned)}\n\n',
                         self.rollout_tables(groups, names, common, f'{case}, runs still training included',
                                             name + '_runs-still-training-included')]
            self.rollout_figure(groups, names, steps, name)
            words = ('seed-mean training curves, one panel per setting, one line per number of environments per '
                     'policy; bands are one standard error.' + (' A line that stops early is a run still training.'
                                                                if waiting else ''))
            part.append(f'{words[0].upper()}{words[1:]}\n\n![rollout shape, {environment}](plots/{name}.png)\n\n')
            self.tex.append(latex_figure(name, f'Rollout-shape check, {case}: {words}'))
            text.append(''.join(part))
        if not text:
            return ''
        return ('## Rollout shape: environments per policy\n\nDefine $W$ and $L$ as in the section Measures: the '
                'whole-curve mean and the final level of the training return, read at the step count named in each '
                'subsection.\n\n' + ''.join(text))

    def rollout_tables(self, groups, names, steps, case, label):
        """At one step count: each rollout width at its best configuration, the best width, and every configuration."""
        by_width = grouped(groups, lambda group: group.environments_per_policy)
        best = {width: compare.choose_configuration(members, steps) for width, members in by_width.items()}
        table = {width: self.measured(group, steps) for width, group in best.items()}
        # The widths in the order of their best configuration's W (ties: L, then the smaller width).
        order = sorted(best, key=lambda width: (-table[width]['whole_curve_mean'], -table[width]['final_level'], width))
        mean_whole = {width: float(np.mean([self.measured(group, steps)['whole_curve_mean'] for group in by_width[width]]))
                      for width in order}
        header = ['environments per policy', 'best whole-curve mean $W$ ↑ ± standard error', 'at',
                  'its final level $L$ ± standard error', 'mean $W$ over its configurations',
                  'policies that reach the goal, best configuration', 'steps per policy so far']
        marks = {key: mean_marks([table[width][key] for width in order])
                 for key in ('whole_curve_mean', 'final_level', 'reached_goal_share')}
        marks['mean_whole'] = mean_marks([mean_whole[width] for width in order])
        rows = []
        for index, width in enumerate(order):
            row = table[width]
            rows.append([
                str(width),
                marked_mean(row['whole_curve_mean'], row['whole_curve_mean_error'], marks['whole_curve_mean'][index]),
                records.describe_settings(best[width], names) or 'its only configuration',
                marked_mean(row['final_level'], row['final_level_error'], marks['final_level'][index]),
                styled(f'{mean_whole[width]:.3f}', marks['mean_whole'][index]),
                styled(format_measure('reached_goal_share', row['reached_goal_share']), marks['reached_goal_share'][index]),
                f'{measures.common_steps([policy for group in by_width[width] for policy in group.policies]):,}'])
        winner = compare.best_environments_per_policy(groups, steps)[0]
        unfinished = steps < min(group.planned_steps for group in groups)
        self.tex.append(latex_table(header, rows,
            f'Rollout-shape check, {case}: each number of environments per policy at its best configuration, at '
            f'{steps:,} environment steps per policy ({seeds_text(groups)} per configuration'
            + (', unfinished runs' if unfinished else '') + '). The best $W$ is bold, the second best underlined.', label))
        text = (markdown_table(header, rows)
                + f'Best at these steps: **{width_words(winner)}** ($W$ = {table[winner]["whole_curve_mean"]:.3f}, '
                + (records.describe_settings(best[winner], names) or 'its only configuration') + ').'
                + (' The runs are unfinished, so this is not the check\'s result yet.' if unfinished else '') + '\n\n')

        # Every configuration, widths in rows and settings in columns, while the settings fit across a page; a
        # check over more settings is read from its sweep table.
        cross_header, cross_rows = self.width_by_setting(groups, names, steps)
        if len(cross_header) > MOST_SETTING_COLUMNS + 1:
            return text
        self.tex.append(latex_table(cross_header, cross_rows,
            f'Rollout-shape check, {case}: $W$ and $L$ of every configuration at {steps:,} environment steps per '
            'policy' + (' (unfinished runs)' if unfinished else '') + ', each with its standard error. The best $W$ '
            'of each column is bold, the second best underlined.', label + '_every-configuration', first_width=2.3))
        return (text + '$W$ and $L$ of every configuration at these steps, each with its standard error; the best $W$ '
                'of each column is bold and the second best underlined.\n\n' + markdown_table(cross_header, cross_rows))

    def width_by_setting(self, groups, names, steps):
        """W and L of every configuration: numbers of environments per policy in rows, the settings in columns."""
        # before: nine configurations (3 widths x 3 intrinsic weights);
        # after:  header ['environments per policy', 'intrinsic weight 0.0625', ...] and one row per width with
        #         cells like '$W$ **0.890** ± 0.025; $L$ 0.994 ± 0.002'.
        setting_of = lambda group: records.describe_settings(group, names) or 'the only configuration'
        columns = list(grouped(groups, setting_of))
        widths = sorted({group.environments_per_policy for group in groups})
        cells = {(group.environments_per_policy, setting_of(group)): self.measured(group, steps) for group in groups}
        # The marks of each column, over the widths that were trained at that setting.
        marks = {}
        for column in columns:
            trained = [width for width in widths if (width, column) in cells]
            marks.update(zip(((width, column) for width in trained),
                             mean_marks([cells[(width, column)]['whole_curve_mean'] for width in trained])))
        rows = []
        for width in widths:
            row = [str(width)]
            for column in columns:
                found = cells.get((width, column))
                if found is None:
                    row.append('not trained')
                    continue
                whole = marked_mean(found['whole_curve_mean'], found['whole_curve_mean_error'], marks[(width, column)])
                row.append(f"$W$ {whole}; $L$ {with_error(found['final_level'], found['final_level_error'])}")
            rows.append(row)
        return ['environments per policy'] + columns, rows

    def rollout_figure(self, groups, names, steps, name):
        """One panel per setting, one line per number of environments per policy, each to the steps it has reached."""
        widths = sorted({group.environments_per_policy for group in groups})
        unknown = [width for width in widths if width not in WIDTH_HUE_ORDER]
        if unknown:
            raise ValueError(f'No hue is set for {unknown} environments per policy; add them to WIDTH_HUE_ORDER.')
        color = {width: figures.series_color(WIDTH_HUE_ORDER.index(width)) for width in widths}
        legend = [{'label': width_words(width), 'color': color[width], 'width': 1.4} for width in widths]
        panels = []
        for settings, members in grouped(groups, lambda group: records.describe_settings(group, names)).items():
            series = []
            for group in sorted(members, key=lambda group: group.environments_per_policy):
                grid, mean, error = measures.mean_curve(group.policies, min(steps, measures.common_steps(group.policies)))
                series.append({'steps': grid, 'mean': mean, 'error': error, 'band': True, 'width': 1.4, 'order': 3,
                               'color': color[group.environments_per_policy]})
            panels.append({'title': settings or 'the only configuration', 'series': series})
        figures.curve_figure(panels, legend, self.plots, name, 'mean return of training episodes')

    def development_section(self):
        """One sweep table per environment, reward setting, training length and method, sorted by W."""
        text = []
        cases = grouped(self.groups['development'], lambda group: (group.environment, group.reward_setting,
                                                                   group.planned_steps, group.method))
        lengths = defaultdict(set)
        for environment, setting, planned, _ in cases:
            lengths[(environment, setting)].add(planned)
        for (environment, setting, planned, method), groups in cases.items():
            steps, ready, waiting = self.evaluation(groups)
            names, with_width = self.naming(groups)
            title = f'{environment}, {REWARD_SETTING_NAMES[setting]}, {method}'
            if len(lengths[(environment, setting)]) > 1:
                title += f', trained for {planned:,} steps'
            if not ready:
                text.append(f'### {title}\n\nNo configuration has reached {steps:,} steps yet.\n\n'
                            + self.waiting_list(waiting, names, with_width))
                continue
            ranked = [row[2] for row in compare.rank_configurations(ready, steps)]
            chosen = ranked[0]
            self.choices[(environment, setting, planned, method)] = chosen
            self.checked.append((f'{environment}, {REWARD_SETTING_NAMES[setting]}', 'development', chosen, steps))
            header = (['configuration', 'whole-curve mean $W$ ↑ ± standard error', 'final level $L$ ± standard error']
                      + [words for _, _, words in OTHER_MEASURES] + ['seeds'])
            rows = [[cells['name'], cells['whole_curve_mean'], cells['final_level'], *cells['others'], cells['seeds']]
                    for cells in self.measure_rows(ranked, steps, lambda group: row_name(group, names, False, with_width))]
            chosen_words = row_name(chosen, MAIN_SETTINGS, False, with_width)
            text.append(
                f'### {title}\n\n{len(ready)} configurations with {seeds_text(ready)} each, compared at {steps:,} '
                f'environment steps per policy.{unfinished_note(steps, planned)} Rows are sorted by $W$; the best '
                'value of each column is bold and the second best underlined. For the mean capped first-goal step '
                'and the mean steps to the goal the smallest value is the best.\n\n' + markdown_table(header, rows)
                + f'Chosen (highest $W$, ties by $L$): **{chosen_words}**.{self.edge_sentence(chosen, groups)}\n\n'
                + self.waiting_list(waiting, names, with_width))
            label = figures.file_slug('sweep', environment, REWARD_SETTING_NAMES[setting], method, f'steps {planned}')
            shown = min(len(rows), LATEX_SWEEP_ROWS)
            self.tex.append(latex_table(header, rows[:shown],
                f'Development sweep, {environment}, {REWARD_SETTING_NAMES[setting]}, {method}: '
                f'{shown_words(shown, len(rows))}, at {steps:,} environment steps per policy '
                f'({seeds_text(ready)} each). Best value of each column bold, second best underlined; for the first-goal '
                'step and the steps to the goal the smallest is the best.', label,
                shares=[1.4, 1.4] + [1.0] * len(OTHER_MEASURES) + [0.75]))
        if not text:
            return ''
        return ('## Development sweeps\n\nDefine $W$, $L$ and the other measures as in the section Measures. Each '
                'table holds one method\'s configurations in one environment and reward setting, trained on the '
                'development seeds (below 30000).\n\n' + ''.join(text))

    def edge_sentence(self, chosen, groups):
        """Where the chosen intrinsic weight, learning rate and entropy coefficient stand among their swept values."""
        # before: chosen intrinsic weight 0.25 of the swept 4^-6 to 64, learning rate 0.001 of three, one entropy value
        # after:  ' Place in the sweep: intrinsic weight 0.25 is inside its 10 swept values; learning rate 0.001 is
        #         the largest of its 3 swept values.' A setting with one value in the sweep is left out.
        same_width = [group for group in groups if group.environments_per_policy == chosen.environments_per_policy]
        places, weight_at_an_end = [], False
        for name in MAIN_SETTINGS:
            values = sorted({group.setting(name) for group in same_width})
            if len(values) < 2:
                continue
            value = chosen.setting(name)
            end = 'the smallest of' if value == values[0] else 'the largest of' if value == values[-1] else 'inside'
            places.append(f'{records.SETTING_WORDS[name]} {value:g} is {end} its {len(values)} swept values')
            weight_at_an_end |= name == 'int_coef' and end != 'inside' and len(values) >= 3
        if not places:
            return ''
        rule = (' The intrinsic weight is at an end of its grid, so the edge rule of the protocol asks for three more '
                'values on that side before anything is confirmed.' if weight_at_an_end else '')
        return f" Place in the sweep: {'; '.join(places)}.{rule}"

    def budget_section(self):
        """The plan's budget rule on each finished sweep of the baseline, and every configuration at that budget."""
        baseline_method = self.run.baseline_method
        cases = grouped([group for group in self.groups['development'] if group.method == baseline_method],
                        lambda group: (group.environment, group.reward_setting, group.planned_steps,
                                       group.environments_per_policy))
        # A case is named with its rollout width or its training length only when the folder holds several.
        widths, lengths = defaultdict(set), defaultdict(set)
        for environment, setting, planned, width in cases:
            widths[(environment, setting)].add(width)
            lengths[(environment, setting)].add(planned)
        text = []
        for (environment, setting, planned, width), groups in cases.items():
            title = f'{environment}, {REWARD_SETTING_NAMES[setting]}'
            if len(widths[(environment, setting)]) > 1:
                title += f', {width_words(width)}'
            if len(lengths[(environment, setting)]) > 1:
                title += f', trained for {planned:,} steps'
            finished = [group for group in groups if measures.common_steps(group.policies) >= planned]
            if not finished:
                text.append(f'### {title}\n\nNo configuration has trained its {planned:,} steps yet.\n\n')
                continue
            names = records.varying_settings(groups)
            name_of = lambda group: row_name(group, names)
            proposal = compare.propose_budget(finished, planned)
            self.budgets.append({'case': title, 'run_steps': planned, **proposal})

            # The configurations the rule reads: W and L at the run's length, the level to reach, the first step.
            header = ['configuration', 'whole-curve mean $W$ ± standard error', 'final level $L$',
                      '95% of the final level, measured from $R_0$', 'first step at which the smoothed curve reaches it']
            # A step count equal to the first step with a full smoothing range is a floor of the rule, not a
            # reading of the curve: the rise happened inside that first range.
            earliest = measures.earliest_smoothed_step(finished[0].policies, planned)
            at_floor = [step == earliest for _, step in proposal['configurations']]
            self.budgets[-1]['every_step_is_the_earliest'] = all(at_floor)
            rows = []
            for (group, step), floor_step in zip(proposal['configurations'], at_floor):
                row, floor = self.measured(group, planned), group.policies[0].never_reached_return
                level = floor + measures.BUDGET_LEVEL_SHARE * (row['final_level'] - floor)
                reached = (f'{step:,}' + (' (the earliest step the rule can read)' if floor_step else '')
                           if step is not None else
                           'never: its final level is $R_0$' if row['final_level'] <= floor else 'never')
                rows.append([name_of(group), with_error(row['whole_curve_mean'], row['whole_curve_mean_error']),
                             f"{row['final_level']:.3f}", f'{level:.3f}', reached])
            sentence = self.budget_sentence(proposal, planned, name_of)
            if all(at_floor):
                sentence += (f' Every step count is the earliest the rule can read from this run ({earliest:,}: one '
                             'full range of 5% of its steps), so the curves rise faster than this run can show, and '
                             'a shorter run would give a smaller budget.')
            few = ('' if len(rows) == 3 else f' Only {len(rows)} configuration{"" if len(rows) == 1 else "s"} '
                   f'ha{"s" if len(rows) == 1 else "ve"} trained the whole length, so the rule reads '
                   f'{"it" if len(rows) == 1 else "them"}.')
            part = [f'### {title}\n\nThe sweep was trained for {planned:,} steps per policy with '
                    f'{seeds_text(finished)} per configuration.{few}\n\n', markdown_table(header, rows), sentence + '\n\n']
            label = figures.file_slug('budget', title)
            self.tex.append(latex_table(header, rows,
                f'Budget rule, {title}: the configurations with the highest whole-curve mean $W$ of the sweep '
                f'trained for {planned:,} steps per policy, and the first logged step at which each one\'s '
                'seed-averaged curve, smoothed over 5% of those steps, reaches 95% of its final level. '
                + sentence.replace('**', ''), label, first_width=2.6))

            # Every finished configuration again at the proposed budget; at the run's own length that is the sweep table.
            budget = proposal['budget']
            if budget < planned:
                ranked = [row[2] for row in compare.rank_configurations(finished, budget)]
                sweep_header = (['configuration', 'whole-curve mean $W$ ↑ ± standard error',
                                 'final level $L$ ± standard error'] + [words for _, _, words in OTHER_MEASURES] + ['seeds'])
                sweep_rows = [[cells['name'], cells['whole_curve_mean'], cells['final_level'], *cells['others'], cells['seeds']]
                              for cells in self.measure_rows(ranked, budget, name_of)]
                # Whether the configuration with the highest W at the budget is the one chosen at the run's length.
                at_length, at_budget = compare.choose_configuration(finished, planned), ranked[0]
                whole = [with_error(row['whole_curve_mean'], row['whole_curve_mean_error'])
                         for row in (self.measured(at_budget, budget), self.measured(at_length, budget))]
                change = (f'At the budget the highest $W$ belongs to **{name_of(at_budget)}**, '
                          + ('the configuration that is also the best at the run\'s length, so the choice does not '
                             'change.' if at_budget is at_length else
                             f'while at the run\'s length it belongs to **{name_of(at_length)}**: the choice changes. '
                             f'At the budget their $W$ are {whole[0]} and {whole[1]}.'))
                self.budgets[-1]['choice_changes'] = at_budget is not at_length
                part.append(f'The measures of every configuration that trained the whole length, recomputed at '
                            f'{budget:,} steps and sorted by $W$. The runs were trained for {planned:,} steps, so '
                            'their learning rate has not reached zero at the budget.\n\n'
                            + markdown_table(sweep_header, sweep_rows) + change + '\n\n')
                shown = min(len(sweep_rows), LATEX_SWEEP_ROWS)
                self.tex.append(latex_table(sweep_header, sweep_rows[:shown],
                    f'Budget rule, {title}: {shown_words(shown, len(sweep_rows))}, with every measure '
                    f'recomputed at the proposed budget of {budget:,} steps, from runs trained for '
                    f'{planned:,} steps. Best value of each column bold, second best underlined; for the first-goal '
                    'step and the steps to the goal the smallest is the best.', label + '_measures-at-the-budget',
                    shares=[1.4, 1.4] + [1.0] * len(OTHER_MEASURES) + [0.75]))
            else:
                part.append('The budget is the run\'s own length, so the measures at the budget are those of the '
                            'sweep table in the section Development sweeps.\n\n')
            text.append(''.join(part))
        if not text:
            return ''
        return ('## Budget\n\nDefine $\\bar R$, $W$, $L$ and $R_0$ as in the section Measures, $N$ as the steps per '
                'policy a sweep was trained for, and $S_5(s)$ as the mean of $\\bar R$ over the $0.05 N$ steps that '
                f'end at step $s$. The rule reads the three {baseline_method} configurations with the highest $W(N)$ on '
                'the development seeds. For each, define $t$ as the first logged step $s \\ge 0.05 N$ with '
                '$S_5(s) - R_0 \\ge 0.95 \\thinspace (L(N) - R_0)$. The proposed budget is twice the largest $t$, '
                'rounded up to a power of two, and never above $N$. A configuration that never reaches 95% of its '
                'final level counts as $t = N$.\n\n' + ''.join(text))

    def budget_sentence(self, proposal, planned, name_of):
        """The proposed budget in words: the arithmetic, and whether a configuration never arrives or the cap applies."""
        doubled, rounded, budget = 2 * proposal['largest'], proposal['rounded'], proposal['budget']
        never = ''
        if proposal['never']:
            names = '; '.join(name_of(group) for group in proposal['never'])
            never = (f' Never at 95% of its final level: {names}. Such a configuration counts as the whole run, so '
                     'the largest step count is the run\'s length.')
        if proposal['capped']:
            return (f'Proposed budget: **{budget:,} steps per policy**, the run\'s own length. Twice the largest step '
                    f'count is {doubled:,}, which rounds up to {rounded:,}, above the {planned:,} steps of the run, '
                    f'so the cap applies.{never}')
        return (f'Proposed budget: **{budget:,} steps per policy**: twice the largest step count is {doubled:,}, '
                f'rounded up to a power of two.{never}')

    def confirmed_rows(self, groups):
        """Per method, its confirmation row: its only one, or the one with the settings chosen on the development seeds."""
        rows = {}
        for method, members in grouped(groups, lambda group: group.method).items():
            first = members[0]
            choice = self.choices.get((first.environment, first.reward_setting, first.planned_steps, method))
            matching = [group for group in members if choice is not None and group.settings == choice.settings
                        and group.environments_per_policy == choice.environments_per_policy]
            rows[method] = members[0] if len(members) == 1 else matching[0] if matching else None
        return rows

    def confirmation_cases(self):
        """The confirmation configurations by environment, reward setting and training length, in report order."""
        return grouped(self.groups['confirmation'], lambda group: (group.environment, group.reward_setting,
                                                                   group.planned_steps))

    def comparison_section(self):
        """Per case: every confirmation row with its paired differences to the baseline; then the verdicts together."""
        cases = self.confirmation_cases()
        if not cases:
            return ''
        baseline_method, references = self.run.baseline_method, self.run.reference_methods
        text, verdicts, case_tex = [], {}, []
        shortest = {}
        for environment, setting, planned in cases:
            shortest[(environment, setting)] = min(planned, shortest.get((environment, setting), planned))
        for (environment, setting, planned), groups in cases.items():
            steps, ready, waiting = self.evaluation(groups)
            names, with_width = self.naming(groups)
            many = {method for method, members in grouped(groups, lambda group: group.method).items() if len(members) > 1}
            name_of = lambda group: row_name(group, names if group.method in many else [], True,
                                             with_width and group.method in many)
            confirmed = self.confirmed_rows(groups)
            baseline = confirmed.get(baseline_method)
            title = f'{environment}, {REWARD_SETTING_NAMES[setting]}'
            if planned != shortest[(environment, setting)]:
                title += f', trained for {planned:,} steps'
            if not ready:
                text.append(f'### {title}\n\nNo row has reached {steps:,} steps yet.\n\n'
                            + self.waiting_list(waiting, names, with_width, True))
                continue
            ranked = [row[2] for row in compare.rank_configurations(ready, steps)]
            rows, other_rows = [], []
            for group, cells in zip(ranked, self.measure_rows(ranked, steps, name_of)):
                self.checked.append((f'{environment}, {REWARD_SETTING_NAMES[setting]}', 'confirmation', group, steps))
                # The paired differences and the verdict; the baseline's own row names itself.
                if group is baseline:
                    extra = ['baseline', 'baseline', 'baseline', str(len(group.policies))]
                elif baseline is None or baseline not in ready:
                    extra = ['no baseline row to compare with'] * 3 + ['0']
                else:
                    result = compare.compare_with_baseline(group.policies, baseline.policies, steps, environment,
                                                           setting, group.method, name_of(group), baseline_method)
                    if result['beats'] is None:
                        extra = [result['verdict']] * 3 + [str(result['seeds'])]
                    else:
                        word = result['verdict'] + (' (reference row)' if group.method in references else '')
                        extra = [interval_text(result['whole_curve_mean']), interval_text(result['final_level']), word,
                                 str(result['seeds'])]
                    if group is confirmed.get(group.method) and planned == shortest[(environment, setting)]:
                        verdicts[(group.method, environment, setting)] = {**result, 'unfinished': steps < planned}
                rows.append([cells['name'], cells['whole_curve_mean'], extra[0], cells['final_level'], extra[1],
                             extra[2], extra[3]])
                other_rows.append([cells['name'], *cells['others'], cells['seeds']])
            header = ['row', 'whole-curve mean $W$ ↑ ± standard error', '$W$ minus the baseline [95%]',
                      'final level $L$ ± standard error', '$L$ minus the baseline [95%]', 'verdict',
                      'seeds in common with the baseline']
            other_header = ['row'] + [words for _, _, words in OTHER_MEASURES] + ['seeds']
            text.append(
                f'### {title}\n\nConfirmation seeds, compared at {steps:,} environment steps per policy.'
                f'{unfinished_note(steps, planned)} Rows are sorted by $W$; differences are the row minus the baseline '
                f'({baseline_method}) on the seeds both have.\n\n' + markdown_table(header, rows)
                + 'The measures reported beside $W$ and $L$, same rows:\n\n' + markdown_table(other_header, other_rows)
                + self.waiting_list(waiting, names, with_width, True))
            label = figures.file_slug('comparison', environment, REWARD_SETTING_NAMES[setting], f'steps {planned}')
            case_tex.append(latex_table(header, rows,
                f'Confirmation seeds, {environment}, {REWARD_SETTING_NAMES[setting]}, at {steps:,} environment steps '
                f'per policy. Differences are the row minus the baseline ({baseline_method}) on the seeds both have, '
                'with 95% intervals from 10,000 resamples of the seeds. A row beats the baseline when $W$ is ahead '
                'and $L$ is not behind.', label, first_width=2.2))
            case_tex.append(latex_table(other_header, other_rows,
                f'Confirmation seeds, {environment}, {REWARD_SETTING_NAMES[setting]}, at {steps:,} environment steps '
                'per policy: the measures reported beside $W$ and $L$. Best value of each column bold, second best '
                'underlined; for the first-goal step and the steps to the goal the smallest is the best.',
                label + '_other-measures', first_width=2.2))
        summary = self.verdict_summary(verdicts)
        self.tex += case_tex
        return ('## Comparison with the baseline on the confirmation seeds\n\nDefine $W$ and $L$ as in the section '
                'Measures. Each method\'s configuration was chosen on the development seeds; the rows below are '
                'trained on the confirmation seeds (30000 on).\n\n' + summary + ''.join(text))

    def verdict_summary(self, verdicts):
        """Methods in rows and the five environments in columns, one table per reward setting, then the success rule."""
        methods = sorted({method for method, _, _ in verdicts})
        if not methods:
            return ''
        references = self.run.reference_methods
        text = ['### All environments\n\nEach cell gives the verdict, then the paired differences in $W$ and in $L$ '
                'with their sides, and the seeds in common. A cell of unfinished runs says so.\n\n']
        beats = {method: {} for method in methods}
        for setting in REWARD_SETTING_ORDER:
            if not any(key[2] == setting for key in verdicts):
                continue
            rows = []
            for method in methods:
                cells = [method]
                for environment in ENVIRONMENT_ORDER:
                    result = verdicts.get((method, environment, setting))
                    if result is None:
                        cells.append('not compared yet')
                        continue
                    if result['beats'] is None:
                        cells.append(result['verdict'])
                        continue
                    beats[method][(environment, setting)] = None if result['unfinished'] else result['beats']
                    cells.append(('unfinished runs: ' if result['unfinished'] else '') + f"{result['verdict']}; "
                                 f"$W$ {result['whole_curve_mean']['mean']:+.3f}, {result['whole_curve_mean']['side']}; "
                                 f"$L$ {result['final_level']['mean']:+.3f}, {result['final_level']['side']} "
                                 f"({result['seeds']:,} seeds)")
                won = sum(bool(beats[method].get((environment, setting))) for environment in ENVIRONMENT_ORDER)
                rows.append(cells + ['reference row' if method in references else f'{won} of {len(ENVIRONMENT_ORDER)}'])
            header = ['method'] + list(ENVIRONMENT_ORDER) + ['environments in which it beats the baseline']
            text.append(f'**{REWARD_SETTING_NAMES[setting].capitalize()}.**\n\n' + markdown_table(header, rows))
            self.tex.append(latex_table(header, rows,
                f'Verdicts on the confirmation seeds, {REWARD_SETTING_NAMES[setting]}: per environment the verdict, the '
                'paired differences in $W$ and in $L$ (method minus baseline) with their sides, and the seeds in common.',
                figures.file_slug('verdicts', REWARD_SETTING_NAMES[setting]), first_width=1.8))
        # The success rule, one list item per method that can succeed.
        text.append('Success rule (the method beats the baseline in all five environments in one reward setting):\n\n')
        for method in methods:
            if method in references:
                continue
            outcome = compare.study_success(beats[method])
            parts = [f"{REWARD_SETTING_NAMES[setting]}: beats in {len(outcome[setting]['beaten'])} of "
                     f"{len(ENVIRONMENT_ORDER)}, not compared or unfinished in {len(outcome[setting]['not_compared'])}"
                     for setting in REWARD_SETTING_ORDER]
            text.append(f"- **{method}:** {'met' if outcome['success'] else 'not met'} ({'; '.join(parts)}).\n")
        return ''.join(text) + '\n'

    def curves_section(self):
        """Training-curve figures: each development sweep's chosen configuration, then the confirmation rows."""
        text = []
        # Development: one figure per reward setting and method, one panel per environment (and training length).
        sweeps = grouped(self.groups['development'], lambda group: (group.reward_setting, group.method))
        legend = [{'label': 'chosen configuration (band: one standard error over its seeds)',
                   'color': figures.SERIES_COLORS[0], 'width': 1.5},
                  {'label': 'the other configurations of the sweep', 'color': figures.OTHER_COLOR, 'width': 1.2}]
        for (setting, method), members in sweeps.items():
            panels = []
            panel_groups = grouped(members, lambda group: (group.environment, group.planned_steps))
            environments = [key[0] for key in panel_groups]
            for (environment, planned), groups in panel_groups.items():
                chosen = self.choices.get((environment, setting, planned, method))
                if chosen is None:
                    continue
                steps, ready, _ = self.evaluation(groups)
                series = []
                for group in ready:
                    grid, mean, error = measures.mean_curve(group.policies, steps)
                    mine = group is chosen
                    series.append({'steps': grid, 'mean': mean, 'error': error, 'band': mine,
                                   'color': figures.SERIES_COLORS[0] if mine else figures.OTHER_COLOR,
                                   'width': 1.5 if mine else 0.6, 'order': 4 if mine else 2})
                several = environments.count(environment) > 1
                panels.append({'title': environment + (f', {planned:,} steps' if several else ''), 'series': series})
            if not panels:
                continue
            name = figures.file_slug('training curves', 'development seeds', REWARD_SETTING_NAMES[setting], method)
            figures.curve_figure(panels, legend, self.plots, name, 'mean return of training episodes')
            words = ('seed-mean training curve of the chosen configuration with a band of one standard error, and '
                     'the other configurations of the sweep in grey.')
            text.append(f'### Development seeds, {REWARD_SETTING_NAMES[setting]}, {method}\n\n'
                        f'{words[0].upper()}{words[1:]}\n\n![training curves, development seeds](plots/{name}.png)\n\n')
            self.tex.append(latex_figure(name, f'Development seeds, {REWARD_SETTING_NAMES[setting]}, {method}: {words}'))

        # Confirmation: one figure per reward setting; the baseline in a neutral colour, each method in its hue.
        baseline_method = self.run.baseline_method
        for setting, cases in grouped(self.confirmation_cases().items(), lambda item: item[0][1]).items():
            panels, drawn = [], {}
            several = len({key[0] for key, _ in cases}) < len(cases)
            for (environment, _, planned), groups in cases:
                steps, ready, _ = self.evaluation(groups)
                series = []
                for method, group in self.confirmed_rows(groups).items():
                    if group is None or group not in ready:
                        continue
                    base = method == baseline_method
                    color = figures.BASELINE_COLOR if base else figures.series_color(self.method_position[method])
                    grid, mean, error = measures.mean_curve(group.policies, steps)
                    series.append({'steps': grid, 'mean': mean, 'error': error, 'band': True, 'color': color,
                                   'width': 1.4, 'order': 3 if base else 4})
                    drawn[method] = color
                if series:
                    panels.append({'title': environment + (f', {planned:,} steps' if several else ''), 'series': series})
            if not panels:
                continue
            # The baseline first, then the methods in the fixed order of their hues.
            order = sorted(drawn, key=lambda method: (method != baseline_method, self.method_position.get(method, -1)))
            legend = [{'label': method + (' (baseline)' if method == baseline_method else ''), 'color': drawn[method],
                       'width': 1.4} for method in order]
            name = figures.file_slug('training curves', 'confirmation seeds', REWARD_SETTING_NAMES[setting])
            figures.curve_figure(panels, legend, self.plots, name, 'mean return of training episodes')
            words = (f'seed-mean training curves with bands of one standard error; the baseline ({baseline_method}) '
                     'in grey.')
            text.append(f'### Confirmation seeds, {REWARD_SETTING_NAMES[setting]}\n\n'
                        f'{words[0].upper()}{words[1:]}\n\n![training curves, confirmation seeds](plots/{name}.png)\n\n')
            self.tex.append(latex_figure(name, f'Confirmation seeds, {REWARD_SETTING_NAMES[setting]}: {words}'))
        if not text:
            return ''
        return ('## Training curves\n\nDefine $\\bar R(s)$ as in the section Measures: the mean over a '
                'configuration\'s policies of the mean return of the training episodes that end in the logging '
                'window containing step $s$.\n\n' + ''.join(text))

    def length_rows(self):
        """The rows of the final-level section per case: confirmation rows, or the development choices without any."""
        cases = {}
        for (environment, setting, planned), groups in self.confirmation_cases().items():
            confirmed = self.confirmed_rows(groups)
            cases[(environment, setting, planned)] = ('confirmation', [group for group in confirmed.values() if group is not None])
        confirmed_cases = {key[:2] for key in cases}
        for (environment, setting, planned, method), group in self.choices.items():
            if (environment, setting) not in confirmed_cases:
                cases.setdefault((environment, setting, planned), ('development', []))[1].append(group)
        return dict(sorted(cases.items(), key=lambda item: (*case_position(*item[0][:2]), item[0][2])))

    def length_section(self):
        """The final level at 25%, 50%, 75% and 100% of the planned steps, and where 'not behind' would change."""
        text = []
        baseline_method = self.run.baseline_method
        for (environment, setting, planned), (stage, groups) in self.length_rows().items():
            baseline = next((group for group in groups if group.method == baseline_method), None)
            # A development row is named with its chosen settings; a confirmation row by its method.
            with_width = len({group.environments_per_policy for group in groups}) > 1
            name_of = lambda group: row_name(group, MAIN_SETTINGS if stage == 'development' else [], True,
                                             with_width or stage == 'development')
            header = ['row'] + [f'{100 * fraction:.0f}% of the planned steps' for fraction in measures.LENGTH_FRACTIONS]
            rows, notes = [], []
            for group in sorted(groups, key=lambda group: (group is not baseline, group.method)):
                cells, behind, compared = [name_of(group)], [], 0
                for fraction in measures.LENGTH_FRACTIONS:
                    steps = fraction * planned
                    if measures.common_steps(group.policies) < steps:
                        cells.append('not reached yet')
                        continue
                    level = float(np.mean([measures.final_level(policy, steps) for policy in group.policies]))
                    cell = f'{level:.3f}'
                    if baseline is not None and group is not baseline and measures.common_steps(baseline.policies) >= steps:
                        _, differences = compare.paired_differences(group.policies, baseline.policies,
                                                                    measures.final_level, steps)
                        result = compare.difference_interval(differences, environment, setting, group.method,
                                                             baseline_method, 'final_level', steps)
                        if result['interval'] is not None:
                            result['side'] = compare.side(result['interval'])
                            cell += f'; {interval_text(result)}'
                            compared += 1
                            if result['side'] == 'behind':
                                behind.append(f'{100 * fraction:.0f}%')
                    cells.append(cell)
                rows.append(cells)
                if compared:
                    notes.append(f'- **{name_of(group)}:** ' + (
                        f'behind the baseline in $L$ at {", ".join(behind)} of the planned steps, so "not behind" '
                        'fails at those lengths.' if behind else
                        f'not behind the baseline in $L$ at any of the {compared} lengths compared.') + '\n')
            label = figures.file_slug('final level', environment, REWARD_SETTING_NAMES[setting], f'steps {planned}')
            words = 'confirmation seeds' if stage == 'confirmation' else 'development seeds, chosen configurations'
            text.append(f'### {environment}, {REWARD_SETTING_NAMES[setting]}, {planned:,} planned steps ({words})\n\n'
                        + markdown_table(header, rows) + (''.join(notes) + '\n' if notes else ''))
            self.tex.append(latex_table(header, rows,
                f'Final level $L$ against the training length, {environment}, {REWARD_SETTING_NAMES[setting]}, '
                f'{planned:,} planned steps ({words}): $L$ over the last 10% of each share of the planned steps and, '
                'for a method, its paired difference to the baseline with the 95% interval and its side.',
                label, first_width=2.2))
        if not text:
            return ''
        return ('## Final level against the training length\n\nDefine $L(f B)$ as in the section Measures: the mean '
                'of the training curve over the last 10% of the first share $f$ of the planned steps $B$, read from '
                'the runs trained for $B$ steps. Each cell gives $L$; a method\'s cell adds its paired difference to '
                'the baseline (method minus baseline, same seeds) with the 95% interval and its side. A run trained '
                'for another number of steps is a separate table, because its learning rate falls to zero over its '
                'own length; its curve is in the section Training curves.\n\n' + ''.join(text))

    def decline_section(self):
        """Both peak ratios of every chosen configuration, and which of them are below 90%."""
        if not self.checked:
            return ''
        header = ['environment and reward setting', 'seeds', 'configuration', 'final level $L$',
                  'peak smoothed over 10% of the budget, and the step its range ends at',
                  'final level over the smoothed peak', 'highest single window, and its step',
                  'final level over the highest single window', 'verdict', 'read at step']
        # A configuration is named with its rollout width where its environment and reward setting hold several.
        widths = defaultdict(set)
        for group in self.groups['development'] + self.groups['confirmation']:
            widths[(group.environment, group.reward_setting)].add(group.environments_per_policy)
        rows, flagged, declined = [], [], 0
        for case, stage, group, steps in self.checked:
            row = self.measured(group, steps)
            several = len(widths[(group.environment, group.reward_setting)]) > 1
            smoothed, single = row['final_over_peak'], row['final_over_single_window_peak']
            if smoothed is None:
                # The curve never left the return of an episode that never reaches the goal: there is no peak.
                cells = ['no rise', 'none', 'no rise', 'none', 'no decline']
            else:
                verdict = ('declines: below 90% of the smoothed peak' if smoothed < DECLINE_SHARE else
                           'below 90% of the highest single window only' if single < DECLINE_SHARE else 'no decline')
                cells = [f"{row['peak']:.3f} at {compact_steps(row['peak_step'])}", f'{100 * smoothed:.1f}%',
                         f"{row['single_window_peak']:.3f} at {compact_steps(row['single_window_peak_step'])}",
                         f'{100 * single:.1f}%', verdict]
                declined += smoothed < DECLINE_SHARE
            rows.append([case, stage, row_name(group, MAIN_SETTINGS, True, several), f"{row['final_level']:.3f}", *cells,
                         compact_steps(steps) + (' (unfinished run)' if steps < group.planned_steps else '')])
            if cells[-1] != 'no decline':
                flagged.append(rows[-1])
        count, share = len(self.checked), f'{100 * DECLINE_SHARE:.0f}%'
        text = ('## Decline check\n\nDefine $P(B)$ as the highest value of the seed-averaged curve smoothed over 10% of '
                'the budget and $R_0$ as the return of an episode that never reaches the goal (section Measures). The '
                'final level over the smoothed peak is $\\dfrac{L(B) - R_0}{P(B) - R_0}$; the ratio with the highest '
                'single window puts that window\'s value in place of $P(B)$ and is the stricter of the two. A chosen '
                f'configuration declines when the ratio with the smoothed peak is below {share}. The table gives both '
                f"ratios of each of the {count} chosen configuration{'' if count == 1 else 's'} (the choice of every "
                f'development sweep and every confirmation row): {declined} below {share} of the smoothed peak, '
                f'{len(flagged) - declined} below {share} of the highest single window only.\n\n')
        # The development document gets the rows below 90% by either peak; a table of every row can be taller than a page.
        if flagged:
            self.tex.append(latex_table(header, flagged,
                f'Decline check: the chosen configurations whose final level is below {share} of the peak of their '
                'seed-averaged training curve, by the peak smoothed over 10% of the budget or by the highest single '
                'window. Both ratios are measured from the return of an episode that never reaches the goal.',
                'decline-check', first_width=1.7,
                # The columns that hold the word "smoothed", which TeX cannot break, are the wide ones.
                shares=[1.15, 1.45, 0.7, 1.25, 1.25, 1.1, 1.1, 1.25, 1.25]))
        return text + markdown_table(header, rows)

    def build(self, title):
        """The whole report in Markdown, section by section."""
        parts = [f'# {title}\n\nGenerated {contents.pacific(contents.now())}; times are Pacific, converted from '
                 f'machines running Eastern. Run folder: `{self.folder_name}`.\n\n', MEASURES_TEXT,
                 self.records_section()]
        # The development section runs before the others because it makes the choices they read; its LaTeX
        # tables still follow the rollout-shape tables, as its Markdown does.
        heading, self.tex = self.tex, []
        development = self.development_section()
        development_tex, self.tex = self.tex, heading
        rollout = self.rollout_shape_section()
        self.tex += development_tex
        parts += [rollout, development, self.budget_section(), self.curves_section(), self.comparison_section(),
                  self.length_section(), self.decline_section()]
        return ''.join(parts)


def generate(run_dir, at_steps=None):
    """Write the run folder's analysis/analysis.md, analysis/analysis.html, analysis/results.tex and analysis/plots/."""
    run_dir = Path(run_dir).resolve()
    run = records.load_run(run_dir)
    if not run.policies:
        raise ValueError(f'No unit of {run_dir} has written a logging window yet.')
    analysis, plots = run_dir / 'analysis', run_dir / 'analysis/plots'
    plots.mkdir(parents=True, exist_ok=True)
    report = Report(run, plots, run_dir.name, at_steps)
    title = run.settings.get('title', 'Training-return report')
    markdown = report.build(title)
    (analysis / 'results.tex').write_text(''.join(report.tex))
    states = contents.write_report(analysis, markdown, title)
    return {'analysis': analysis / 'analysis.md', 'sections': states, 'report': report}


def main():
    """Generate one run folder's report from its records."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--at-steps', type=int, default=None,
                        help='compare every table at this many environment steps per policy')
    arguments = parser.parse_args()
    result = generate(arguments.run_dir, arguments.at_steps)
    print(f"wrote {result['analysis']}")


if __name__ == '__main__':
    main()
