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


def _paired_rows(candidate_times, geometry_kind='straight', variant='primary', policies_per_field=2):
    """Build matching uniform and adverse confirmation rows without training."""
    # Matching policy seeds retain two policies per independent neural-field seed.
    specification = {'wave': 'confirmation', 'geometry': {'kind': geometry_kind, 'width': 4, 'turns': 0},
                     'policy_kind': 'tabular', 'field_kind': 'uniform', 'severity_exponent': 0,
                     'total_timesteps': 64, 'variant': variant}
    count = len(candidate_times)
    base = {'configuration': {'comparison': 'shared'}, 'first_goal_steps': [10] * count,
            'policy_seeds': list(range(count)), 'field_seeds': [index//policies_per_field for index in range(count)]}
    adverse = {**specification, 'field_kind': 'random_network' if geometry_kind == 'snake' else 'high_start_cell'}
    return [{**base, 'specification': specification},
            {**base, 'specification': adverse, 'first_goal_steps': candidate_times}]


def test_paired_confirmation_records_adjusted_difference_and_ratio():
    """Confirm a threefold delay only when its adjusted interval supports the claim."""
    # Constant paired outcomes make both adjusted interval endpoints exactly equal to the effect.
    results = REPORT.paired_confirmation_comparisons(_paired_rows([30, 30, 30, 30]), repetitions=1000,
                                                    planned_policy_seeds=4, planned_field_seeds=2)
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
    results = REPORT.paired_confirmation_comparisons(_paired_rows([10, 10, 10, 10], 'snake'), repetitions=1000,
                                                    planned_policy_seeds=4, planned_field_seeds=2)
    assert results[0]['independent_group_count'] == 2
    assert results[0]['effect_label'] == 'No confirmed increase in first-goal time'
    assert REPORT.paired_confirmation_comparisons(_paired_rows([30, 30, 30, 30], variant='field-shuffle')) == []


def test_matching_partial_confirmation_remains_interim_without_changing_estimates():
    """Matching the currently available seeds cannot stand in for all planned fresh seeds."""
    # The point ratio is three in every case, but four matching policies do not complete 128.
    result = REPORT.paired_confirmation_comparisons(_paired_rows([30]*4, 'snake'), repetitions=1000)[0]
    assert result['mean_ratio'] == 3
    assert result['confirmation_complete'] is False
    assert result['confirmation_status'] == 'Interim'
    assert result['effect_label'] == 'Interim: planned confirmation is incomplete'
    assert result['all_current_seeds_paired'] is True
    assert result['planned_policy_seed_count'] == 128
    assert result['required_field_seed_count'] == 32


def test_confirmation_requires_both_planned_policies_and_neural_field_groups():
    """Require 128 distinct paired policies and 32 independent neural fields by default."""
    # Holding the number of policies fixed while halving field groups leaves confirmation incomplete.
    complete = REPORT.paired_confirmation_comparisons(
        _paired_rows([30]*128, 'snake', policies_per_field=4), repetitions=1000)[0]
    incomplete = REPORT.paired_confirmation_comparisons(
        _paired_rows([30]*128, 'snake', policies_per_field=8), repetitions=1000)[0]
    assert complete['confirmation_complete'] is True
    assert complete['independent_group_count'] == 32
    assert complete['effect_label'] == 'At least twice the capped mean first-goal time'
    assert complete['family_comparison_count'] == 4 and complete['family_protocol'] == 'shared'
    assert incomplete['confirmation_complete'] is False
    assert incomplete['independent_group_count'] == 16


def _write_completed_fixture(run_dir, kind, first_goal_steps, include_visit_history=False):
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
    history = {'visit_count_history': np.stack([visits//4, visits//2, visits]),
               'visit_count_history_steps': np.asarray([8, 16, 32])} if include_visit_history else {}
    np.savez(folder / 'behavior.npz', initial_fields=np.ones((4, 4)), visit_counts=visits,
             late_visit_counts=visits, policy_probabilities=np.full((4, 4, 4), 0.25),
             positions=np.asarray([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.int32),
             grid=np.asarray([[0, 1, 2, 3]], dtype=np.int32), trajectory_states=states,
             trajectory_steps=np.asarray([1, 16, 32]), maximum_progress_history=states,
             first_goal_steps=np.asarray(first_goal_steps), seed_ids=np.arange(4),
             field_seed_ids=np.asarray([0, 0, 1, 1]), setting_indices=np.zeros(4, dtype=np.int32), **history)


def test_report_generates_metrics_seed_records_and_vector_behavior_figures(tmp_path, monkeypatch):
    """Exercise the actual summary and behavior schemas before live results arrive."""
    # Complete uniform and random-network fixtures are a schema check, not experimental evidence.
    _write_completed_fixture(tmp_path, 'uniform', [10, 12, 14, 16])
    _write_completed_fixture(tmp_path, 'random_network', [20, -1, 30, -1])
    captured = {}
    original_save = REPORT._save_figure

    def record_figure_axes(figure, directory, name):
        """Inspect the real axes before saving without changing the underlying figure data."""
        # The uniform median seed reaches the goal at 12, which is absent from position samples.
        captured.setdefault('figure_protocols', []).append(figure._suptitle.get_text())
        if name.endswith('_movement'):
            captured['limits'] = [axis.get_ylim() for axis in figure.axes]
            captured['markers'] = [float(line.get_xdata()[0]) for axis in figure.axes for line in axis.lines
                                   if line.get_label().startswith('Exact first goal:')]
            captured['failures'] = sum(text.get_text() == 'Goal not reached within budget'
                                       for axis in figure.axes for text in axis.texts)
        if name.endswith('_initial_profiles'):
            captured['scales'] = [axis.get_yscale() for axis in figure.axes]
            captured['profile_extent'] = [(float(axis.lines[0].get_xdata()[0]),
                                           float(axis.lines[0].get_xdata()[-1])) for axis in figure.axes]
        original_save(figure, directory, name)

    monkeypatch.setattr(REPORT, '_save_figure', record_figure_axes)
    result = REPORT.generate_report(tmp_path)
    assert result['reported_rows'] == 2
    assert result['figure_comparisons'] == 1
    report = (tmp_path / 'analysis/analysis.md').read_text()
    assert 'A failure contributes $B$' in report
    assert '28.5' in report
    assert (tmp_path / 'analysis/data/seed_metrics.csv').read_text().count('\n') == 9
    for suffix in ('maps', 'initial_profiles', 'movement', 'policy', 'discovery'):
        for extension in ('png', 'pdf'):
            assert (tmp_path / f'analysis/plots/confirmation_comparison-0_{suffix}.{extension}').stat().st_size > 1000
    tex = (tmp_path / 'analysis/results.tex').read_text()
    assert 'Seeds ' + '\\\\' + '\n' in tex
    assert '\\includegraphics' in tex
    animation = (tmp_path / 'analysis/plots/confirmation_comparison-0_sampled_positions.html').read_text()
    assert 'Intermediate actions are not shown' in animation
    assert 'field seed' in animation
    assert captured['limits'] == [(0., 3.)] * 4
    assert captured['markers'] == [12., 12.]
    assert captured['failures'] == 2
    assert captured['scales'] == ['log', 'log']
    assert captured['profile_extent'] == [(0., 1.)] * 2
    assert all('Shared settings' in title for title in captured['figure_protocols'])
    assert '4 policy seeds; 2 field seeds' in report
    assert 'Shared settings' in report


def test_behavior_waits_for_uniform_baseline_but_retains_all_numerical_results(tmp_path, monkeypatch):
    """Do not draw a comparative figure before its uniform reference has completed."""
    # One completed neural-field record remains in the numerical table while its figures wait.
    _write_completed_fixture(tmp_path, 'random_network', [20, -1, 30, -1])

    def reject_premature_figure(*arguments):
        """Fail if the report tries to draw a comparison with no uniform baseline."""
        raise AssertionError('Comparative figure drawn before the baseline completed.')

    monkeypatch.setattr(REPORT, '_plot_comparison', reject_premature_figure)
    result = REPORT.generate_report(tmp_path)
    report = (tmp_path/'analysis/analysis.md').read_text()
    assert result['reported_rows'] == 1 and result['figure_comparisons'] == 0
    assert '28.5' in report
    assert 'Waiting for a completed uniform-initialization baseline' in report
    assert '4 policy seeds; 2 field seeds' in report


def test_behavior_labels_distinguish_protocols_and_actual_field_counts(tmp_path):
    """Name both settings protocols and avoid field-count labels for deterministic bonuses."""
    # The same four completed policies represent two random fields or one deterministic rule.
    _write_completed_fixture(tmp_path, 'uniform', [10, 12, 14, 16])
    _write_completed_fixture(tmp_path, 'random_network', [20, -1, 30, -1])
    rows = REPORT.collect_rows(tmp_path)
    uniform = next(row for row in rows if row['specification']['field_kind'] == 'uniform')
    neural = next(row for row in rows if row['specification']['field_kind'] == 'random_network')
    assert '4 policy seeds' in REPORT.condition_count_label(uniform)
    assert 'field seeds' not in REPORT.condition_count_label(uniform)
    assert '4 policy seeds; 2 field seeds' in REPORT.condition_count_label(neural)
    assert 'Shared settings' in REPORT.behavior_heading(uniform)
    tuned = {**neural, 'configuration': {**neural['configuration'], 'comparison': 'independently-tuned'}}
    assert 'Independently tuned settings' in REPORT.behavior_heading(tuned)


def test_empty_report_has_no_claimed_failure_and_no_behavior_figures(tmp_path):
    """Create a truthful launch report before any training result exists."""
    # No records means zero completed groups and zero plots, not a made-up score.
    result = REPORT.generate_report(tmp_path)
    assert result['completed_setting_groups'] == result['figure_comparisons'] == 0
    assert 'No failure case has been established' in (tmp_path / 'analysis/analysis.md').read_text()
    assert not list((tmp_path / 'analysis/plots').iterdir())


def test_early_visitation_uses_complete_histories_and_recorded_common_steps(tmp_path, monkeypatch):
    """Check checkpoint/copy/state axes and render only fully matched early budgets."""
    # Tiny complete archives use the final common step 32 instead of pretending to contain 65,536.
    _write_completed_fixture(tmp_path, 'uniform', [10, 12, 14, 16], include_visit_history=True)
    _write_completed_fixture(tmp_path, 'random_network', [20, -1, 30, -1], include_visit_history=True)
    row = next(item for item in REPORT.collect_rows(tmp_path) if item['specification']['field_kind'] == 'uniform')
    loaded = REPORT._load_behavior(tmp_path, row)
    assert loaded['visit_count_history'].shape == (3, 4, 4)
    np.testing.assert_array_equal(loaded['visit_count_history_steps'], [8, 16, 32])
    np.testing.assert_array_equal(loaded['visit_count_history'][1, 1], [3, 1, 0, 0])
    captured = {}
    original_save = REPORT._save_figure

    def record_early_titles(figure, directory, name):
        """Retain the labels actually rendered by the early-visitation figure."""
        # The colorbar is an extra axis; both map titles must name the same actual step budget.
        if name.endswith('_early_visitation'):
            captured['titles'] = [axis.get_title() for axis in figure.axes if axis.get_title()]
        original_save(figure, directory, name)

    monkeypatch.setattr(REPORT, '_save_figure', record_early_titles)
    REPORT.generate_report(tmp_path)
    assert len(captured['titles']) == 2
    assert all('After 32 training transitions' in title for title in captured['titles'])
    for extension in ('png', 'pdf'):
        assert (tmp_path/f'analysis/plots/confirmation_comparison-0_early_visitation.{extension}').stat().st_size > 1000
    assert 'Early visitation' in (tmp_path/'analysis/analysis.md').read_text()
    assert 'early_visitation.pdf' in (tmp_path/'analysis/results.tex').read_text()

    # A mixed old/new seed archive keeps ordinary results valid but provides no partial early map.
    path = tmp_path/'data/research/random_network/behavior.npz'
    with np.load(path) as archive:
        old = {key: archive[key].copy() for key in archive.files if not key.startswith('visit_count_history')}
    np.savez(path, **old)
    mixed_row = {**row, 'seed_count': 8, 'sources': row['sources']+[{'task': 'random_network', 'setting_index': 0}]}
    mixed = REPORT._load_behavior(tmp_path, mixed_row)
    assert mixed['visit_count_history_complete'] is False
    assert mixed['visit_count_history'] is None
    assert REPORT.early_visitation_budgets([loaded, mixed]) == []


def test_early_visitation_skips_budgets_missing_from_any_condition():
    """Do not compare one condition at 65,536 steps with another at a later budget."""
    # Only checkpoints shared by every complete condition may appear in a comparison row.
    complete = lambda steps: {'visit_count_history_complete': True, 'visit_count_history_steps': np.asarray(steps)}
    assert REPORT.early_visitation_budgets([complete([2**16, 2**18, 2**20]),
                                          complete([2**16, 2**20])]) == [2**16, 2**20]


def test_completed_repeatability_probe_is_reported_without_claiming_goal_time_difference(tmp_path):
    """Use recorded probe observations in both reports while preserving the probe's stopping limit."""
    # The two cases differ in action time, while both stop before observing a first-goal-time difference.
    stable = {'repetitions': 20, 'different_repetitions': [], 'maximum_absolute_difference': 0.0}
    case = {'initial_parameters_equal': True, 'initial_action_keys_equal': True,
            'initial_minibatch_permutations_equal': True, 'initial_rollout_actions_equal': True,
            'first_rng_key_difference': None,
            'fixed_minibatch_gradient_repeats': {'maximum_absolute_difference': 1.2e-7},
            'simultaneous_repeated_state_accumulation': {'maximum_absolute_difference': 1.2e-7},
            'unique_state_row_gradient_repeats': stable, 'ordered_state_accumulation': stable,
            'first_parameter_difference': {'after_iteration': 1},
            'first_action_difference': {'iteration': 22, 'transition': 5777},
            'first_counter_difference': {'after_iteration': 25},
            'capture_matches_training_counts': True, 'first_goal_time_difference': None}
    second = {**case, 'first_action_difference': {'iteration': 346, 'transition': 88790},
              'first_counter_difference': {'after_iteration': 347}}
    path = tmp_path/'data/repeatability/duplicate_tabular_policy_seeds.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'completed': True, 'cases': [case, second]}))
    REPORT.generate_report(tmp_path)
    markdown = (tmp_path/'analysis/analysis.md').read_text()
    tex = (tmp_path/'analysis/results.tex').read_text()
    for document in (markdown, tex):
        assert 'Repeatability of tabular training' in document
        assert '5,777, 88,790' in document
        assert '20/20' in document
        assert 'counts remain exact for the recorded actions' in document
        assert 'stopped without observing a first-goal-time difference' in document
        assert 'primary training code remains frozen' in document
        assert 'https://docs.jax.dev/en/latest/' in document


def test_missing_or_incomplete_repeatability_probe_produces_no_observation_claims(tmp_path):
    """Never turn a missing or unfinished probe into a reported experimental observation."""
    # No completed marker means the case data cannot enter the report yet.
    assert REPORT.tabular_repeatability_observations(tmp_path) is None
    path = tmp_path/'data/repeatability/duplicate_tabular_policy_seeds.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'completed': False, 'cases': []}))
    assert REPORT.tabular_repeatability_observations(tmp_path) is None
    REPORT.generate_report(tmp_path)
    assert 'Repeatability of tabular training' not in (tmp_path/'analysis/analysis.md').read_text()
    assert 'Repeatability of tabular training' not in (tmp_path/'analysis/results.tex').read_text()
