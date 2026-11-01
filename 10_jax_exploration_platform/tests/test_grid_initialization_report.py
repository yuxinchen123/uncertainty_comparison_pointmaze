"""Check first-goal aggregation and generated behavior artifacts on a compute node."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / 'scripts/grid_initialization_campaign'
sys.path.insert(0, str(SCRIPT_DIRECTORY))
SPECIFICATION = importlib.util.spec_from_file_location('grid_initialization_report', SCRIPT_DIRECTORY / 'report.py')
REPORT = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(REPORT)


def test_failures_contribute_the_full_budget_to_the_mean():
    """Prevent selection of only the seeds that reached the goal."""
    # Before: one success at ten and one failure; after: mean twenty-one at budget thirty-two.
    values = REPORT.capped_first_goal_steps([10, -1], 32)
    np.testing.assert_array_equal(values, [10, 32])
    assert values.mean() == 21
    with pytest.raises(ValueError):
        REPORT.capped_first_goal_steps([33], 32)


def test_confidence_resamples_whole_field_seed_groups():
    """Retain dependence between policy copies that use the same initial field."""
    # Two field groups with constant outcomes have bootstrap endpoints zero and ten.
    interval = REPORT.grouped_mean_interval([0, 0, 10, 10], [1, 1, 2, 2], seed=4)
    np.testing.assert_array_equal(interval, [0, 10])
    assert REPORT.grouped_mean_interval([0, 10], [1, 1], seed=4) == [None, None]


def test_representative_seed_selection_uses_fixed_tie_breaks():
    """Choose median outcomes without selecting the most visually striking failure."""
    # All failures tie at the budget, so increasing field then policy seed decides.
    index = REPORT.representative_seed_index([-1, -1, -1], 32, [20, 10, 5], [2, 1, 1])
    assert index == 2


def test_confirmation_settings_are_not_chosen_using_confirmation_outcomes():
    """Reject two configurations masquerading as the same frozen comparison."""
    # Development may select a minimum, but confirmation must already have one fixed setting.
    specification = {'wave': 'confirmation', 'geometry': {'kind': 'straight', 'width': 4, 'turns': 0},
                     'policy_kind': 'tabular', 'field_kind': 'uniform', 'severity_exponent': 0,
                     'total_timesteps': 32}
    rows = [{'specification': specification, 'configuration': {'comparison': 'shared', 'learning_rate': rate},
             'capped_mean_first_goal_steps': mean} for rate, mean in [(0.1, 10), (0.2, 20)]]
    with pytest.raises(ValueError, match='multiple different configurations'):
        REPORT.select_reporting_rows(rows)


def _paired_rows(candidate_times, geometry_kind='straight', variant='primary'):
    """Build matching uniform and adverse confirmation rows without training."""
    # Matching policy seeds retain two policies per independent neural-field seed.
    specification = {'wave': 'confirmation', 'geometry': {'kind': geometry_kind, 'width': 4, 'turns': 0},
                     'policy_kind': 'tabular', 'field_kind': 'uniform', 'severity_exponent': 0,
                     'total_timesteps': 64, 'variant': variant}
    base = {'configuration': {'comparison': 'shared'}, 'first_goal_steps': [10, 10, 10, 10],
            'policy_seeds': [0, 1, 2, 3], 'field_seeds': [0, 0, 1, 1]}
    adverse = {**specification, 'field_kind': 'random_network' if geometry_kind == 'snake' else 'high_start_cell'}
    return [{**base, 'specification': specification},
            {**base, 'specification': adverse, 'first_goal_steps': candidate_times}]


def test_paired_confirmation_records_adjusted_difference_and_ratio():
    """Confirm a threefold delay only when its adjusted interval supports the claim."""
    # Constant paired outcomes make both adjusted interval endpoints exactly equal to the effect.
    results = REPORT.paired_confirmation_comparisons(_paired_rows([30, 30, 30, 30]), repetitions=1000)
    assert len(results) == 1
    assert results[0]['mean_difference'] == 20
    np.testing.assert_array_equal(results[0]['mean_difference_interval'], [20, 20])
    np.testing.assert_array_equal(results[0]['mean_ratio_interval'], [3, 3])
    assert results[0]['interval_confidence'] == .9875
    assert results[0]['family_comparison_count'] == 4
    assert results[0]['effect_label'] == 'At least twice the capped mean first-goal time'


def test_paired_confirmation_keeps_neural_field_groups_and_reports_no_effect():
    """Keep neural-field dependence and report unchanged outcomes without a failure claim."""
    # Identical discovery times have difference zero and ratio one, regardless of grouping.
    results = REPORT.paired_confirmation_comparisons(_paired_rows([10, 10, 10, 10], 'snake'), repetitions=1000)
    assert results[0]['independent_group_count'] == 2
    assert results[0]['effect_label'] == 'No confirmed increase in first-goal time'
    assert REPORT.paired_confirmation_comparisons(_paired_rows([30, 30, 30, 30], variant='field-shuffle')) == []


def _write_completed_fixture(run_dir, kind, first_goal_steps):
    """Create a complete tiny archive without running a training algorithm."""
    # Four copies share two random-field seeds and three recorded training-position samples.
    folder = run_dir / 'data/research' / kind
    folder.mkdir(parents=True)
    specification = {'name': kind, 'wave': 'confirmation', 'geometry': {'kind': 'straight', 'width': 4, 'turns': 0},
                     'policy_kind': 'tabular', 'field_kind': kind, 'severity_exponent': 0,
                     'total_timesteps': 32}
    configuration = {'learning_rate': 2 ** -8, 'intrinsic_coefficient': 1.0,
                     'entropy_coefficient': 2 ** -10, 'comparison': 'shared'}
    setting = {'configuration': configuration, 'setting_index': 0, 'first_goal_steps': first_goal_steps,
               'policy_seeds': [0, 1, 2, 3], 'field_seeds': [0, 0, 1, 1], 'coverage_mean': 1.0}
    (folder / 'summary.json').write_text(json.dumps({'completed': True, 'specification': specification,
                                                   'settings': [setting]}))
    states = np.asarray([[0, 0, 0, 0], [1, 0, 1, 0], [3, 1, 3, 1]], dtype=np.int32)
    visits = np.asarray([[4, 3, 2, 1], [7, 3, 0, 0], [4, 3, 2, 1], [7, 3, 0, 0]], dtype=np.int32)
    np.savez(folder / 'behavior.npz', initial_fields=np.ones((4, 4)), visit_counts=visits,
             late_visit_counts=visits, policy_probabilities=np.full((4, 4, 4), 0.25),
             positions=np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int32),
             grid=np.asarray([[0, 1, 2, 3]], dtype=np.int32), trajectory_states=states,
             trajectory_steps=np.asarray([1, 16, 32]), maximum_progress_history=states,
             first_goal_steps=np.asarray(first_goal_steps), seed_ids=np.arange(4),
             field_seed_ids=np.asarray([0, 0, 1, 1]), setting_indices=np.zeros(4, dtype=np.int32))


def test_report_generates_metrics_seed_records_and_vector_behavior_figures(tmp_path):
    """Exercise the actual summary and behavior schemas before live results arrive."""
    # Complete uniform and random-network fixtures are a schema check, not experimental evidence.
    _write_completed_fixture(tmp_path, 'uniform', [10, 12, 14, 16])
    _write_completed_fixture(tmp_path, 'random_network', [20, -1, 30, -1])
    result = REPORT.generate_report(tmp_path)
    assert result['reported_rows'] == 2
    assert result['figure_comparisons'] == 1
    report = (tmp_path / 'analysis/analysis.md').read_text()
    assert 'A failure contributes $B$' in report
    assert '28.5' in report
    assert (tmp_path / 'analysis/data/seed_metrics.csv').read_text().count('\n') == 9
    for suffix in ('maps', 'movement', 'policy', 'discovery'):
        for extension in ('png', 'pdf'):
            assert (tmp_path / f'analysis/plots/confirmation_comparison-0_{suffix}.{extension}').stat().st_size > 1000
    tex = (tmp_path / 'analysis/results.tex').read_text()
    assert 'Seeds ' + '\\\\' + '\n' in tex
    assert '\\includegraphics' in tex
    animation = (tmp_path / 'analysis/plots/confirmation_comparison-0_sampled_positions.html').read_text()
    assert 'Intermediate actions are not shown' in animation
    assert 'field seed' in animation


def test_empty_report_has_no_claimed_failure_and_no_behavior_figures(tmp_path):
    """Create a truthful launch report before any training result exists."""
    # No records means zero completed groups and zero plots, not a made-up score.
    result = REPORT.generate_report(tmp_path)
    assert result['completed_setting_groups'] == result['figure_comparisons'] == 0
    assert 'No failure case has been established' in (tmp_path / 'analysis/analysis.md').read_text()
    assert not list((tmp_path / 'analysis/plots').iterdir())
