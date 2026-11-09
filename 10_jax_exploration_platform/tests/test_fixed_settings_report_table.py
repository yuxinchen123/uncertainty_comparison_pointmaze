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
            'coverage': 1.0, 'sampled': 1.0, 'greedy': 1.0, 'seeds': 128,
            'configuration': {'learning_rate': 0.001, 'entropy_coefficient': 0.01, 'intrinsic_coefficient': 0.125}}


def table_width_cm(tex):
    """Sum the paragraph-column widths plus the gaps between columns of one generated table."""
    # Before: '...\\setlength{\\tabcolsep}{3pt}...{@{}p{3.3cm}...p{0.6cm}@{}}'; after: 12.0 + 14 * 3 pt.
    widths = [float(value) for value in re.findall(r'p\{([0-9.]+)cm\}', tex)]
    gap = float(re.search(r'\\tabcolsep\}\{([0-9.]+)pt\}', tex).group(1))
    outer = 0 if '@{}p' in tex else 2
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
    body = tex.split('\\midrule\n')[1].split('\\bottomrule')[0].strip().splitlines()
    assert len(body) == 2 and body[0].startswith('uniform field')


def test_only_the_mean_is_styled_so_the_interval_can_wrap():
    """An underlined mean with its interval inside the underline ran up to 9.2 pt out of its column."""
    tex = REPORT.latex_table([row('no-bonus_state-normalization-off_reward-normalization-off', 163714, 1.0),
                              row('uniform_state-normalization-off_reward-normalization-on', 8110, 1.0)],
                             'caption', 'tab:test')
    assert '\\underline{163,714} [147,343, 180,085]' in tex and '\\textbf{8,110} [7,299, 8,921]' in tex
