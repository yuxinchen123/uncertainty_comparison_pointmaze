"""Check that the fixed-settings results tables fit the development document's text width."""

from pathlib import Path
import re
import sys


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / 'scripts/grid_initialization_campaign'
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import fixed_settings_report as REPORT  # noqa: E402

TEXT_WIDTH_CM = 13.97  # neurips_2026 text width, 5.5 inches
POINT_CM = 2.54 / 72.27


def row(condition, mean, discovery):
    """Build one condition row in the shape results_table reads."""
    return {'condition': condition, 'mean': mean, 'interval': (mean*0.9, mean*1.1), 'discovery': discovery,
            'trimmed': mean*0.8, 'trim': 10, 'standard_error': mean*0.05, 'trimmed_standard_error': mean*0.04,
            'coverage': 1.0, 'sampled': 1.0, 'greedy': 1.0, 'seeds': 128,
            'configuration': {'learning_rate': 0.001, 'entropy_coefficient': 0.01, 'intrinsic_coefficient': 0.125}}


def table_width_cm(tex):
    """Sum the paragraph-column widths plus the gaps between columns of one generated table."""
    # Before: '...\\setlength{\\tabcolsep}{3pt}...{@{}>{...}p{3.3cm}...p{0.6cm}@{}}'; after: 12.0 + 14 * 3 pt.
    widths = [float(value) for value in re.findall(r'p\{([0-9.]+)cm\}', tex)]
    gap = float(re.search(r'\\tabcolsep\}\{([0-9.]+)pt\}', tex).group(1))
    outer = 0 if '\\begin{longtable}{@{}' in tex else 2
    return sum(widths) + (2*(len(widths)-1) + outer)*gap*POINT_CM


def test_results_table_fits_the_text_width():
    """The eight confirmation tables ran 82.6 pt past the margin before the column change."""
    tex = REPORT.latex_table([row('no-bonus_state-normalization-off_reward-normalization-off', 21468, 1.0),
                              row('uniform_state-normalization-off_reward-normalization-on', 8110, 1.0)],
                             'caption', 'tab:test')
    assert table_width_cm(tex) <= TEXT_WIDTH_CM


def test_results_table_keeps_one_row_per_condition_sorted_by_the_primary_metric():
    """Rows stay sorted by mean capped first-goal time, smallest first, with one row each."""
    tex = REPORT.latex_table([row('no-bonus_state-normalization-off_reward-normalization-off', 21468, 1.0),
                              row('uniform_state-normalization-off_reward-normalization-on', 8110, 1.0)],
                             'caption', 'tab:test')
    body = tex.split('\\endfoot\n')[1].split('\\end{longtable}')[0].strip().splitlines()
    assert len(body) == 2 and body[0].startswith('uniform field')
    # The table may continue on the next page: its header is given once for the first page and once for the others.
    assert tex.count('\\toprule') == 2 and '\\endfirsthead' in tex and '\\endhead' in tex
    assert '10 of every 128 seeds removed from each end' in tex


def test_only_the_mean_is_styled_so_the_interval_can_wrap():
    """An underlined mean with its interval inside the underline ran up to 9.2 pt out of its column."""
    tex = REPORT.latex_table([row('no-bonus_state-normalization-off_reward-normalization-off', 163714, 1.0),
                              row('uniform_state-normalization-off_reward-normalization-on', 8110, 1.0)],
                             'caption', 'tab:test')
    assert '\\underline{163.71K} $\\pm$ 8.19K [147.34K, 180.09K]' in tex
    assert '\\textbf{8.11K} $\\pm$ 0.41K [7.30K, 8.92K]' in tex


def test_weight_is_set_as_math_so_it_does_not_break():
    """The weight 2^-3 broke as '2^-' and '3' on two lines when written as text."""
    tex = REPORT.latex_table([row('no-bonus_state-normalization-off_reward-normalization-off', 21468, 1.0)],
                             'caption', 'tab:test')
    assert 'weight $2^{-3}$' in tex and '\\raggedright' in tex


def test_counts_are_whole_numbers_when_a_value_is_below_one_thousand():
    """Thousands are used only when every mean, trimmed mean and interval end is at least 1,000."""
    small = row('uniform_state-normalization-off_reward-normalization-on', 900, 1.0)
    tex = REPORT.latex_table([small], 'caption', 'tab:test')
    assert '\\textbf{900} $\\pm$ 45 [810, 990]' in tex


def test_trimmed_mean_removes_the_given_number_from_each_end():
    """Before: mean of 1..10 plus one never-found seed; after removing 1 per end: the middle values only."""
    values = list(range(1, 11)) + [16777216]
    assert REPORT.trimmed_mean(values, 1) == sum(range(2, 11)) / 9
    assert REPORT.trim_count(128) == 10 and REPORT.trim_count(16) == 1


def test_grouped_standard_error_matches_the_formula_for_single_seed_groups():
    """With one policy per group the bootstrap standard error is close to s / sqrt(K)."""
    import numpy as np
    values = np.random.default_rng(3).exponential(1000.0, 128)
    error, trimmed_error = REPORT.grouped_standard_errors(values, list(range(128)), 10, 7)
    expected = values.std(ddof=1) / np.sqrt(128)
    assert abs(error / expected - 1) < 0.1
    # One never-found seed dominates the error of the mean but not of the trimmed mean.
    values[0] = 16777216
    error, trimmed_error = REPORT.grouped_standard_errors(values, list(range(128)), 10, 7)
    assert trimmed_error < error / 10


def test_labels_name_the_rnd_input_the_policy_and_the_decay_exponent():
    """Tables say coordinate-RND or tabular-RND, and the uniform field's decay exponent, with a math minus in LaTeX."""
    assert REPORT.condition_label('online-rnd_state-normalization-on_reward-normalization-on') == (
        'online coordinate-RND, state normalization on, reward normalization on')
    assert REPORT.condition_label('online-tabular-rnd_state-normalization-off_reward-normalization-on') == (
        'online tabular-RND, reward normalization on')
    assert REPORT.condition_label('uniform-decay-exponent-0.25_state-normalization-off_reward-normalization-off') == (
        'uniform field, decay exponent -0.25, reward normalization off')
    assert REPORT.latex_label('uniform_state-normalization-off_reward-normalization-on') == (
        'uniform field, decay exponent $-0.5$, reward normalization on')
    assert REPORT.POLICY_LABELS == {'tabular': 'tabular-PPO', 'neural': 'coordinate-PPO'}
