"""Check research decisions using fabricated records, without initializing a numerical runtime."""

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[2] / "scripts/grid_initialization_campaign"
sys.path.insert(0, str(SCRIPT_DIRECTORY))
specification = importlib.util.spec_from_file_location("grid_campaign_decisions", SCRIPT_DIRECTORY / "campaign.py")
campaign = importlib.util.module_from_spec(specification)
specification.loader.exec_module(campaign)


def record(field, success, steps, policy="tabular", width=32):
    """Build a minimal complete development result for one condition."""
    return {"specification": {"geometry": {"kind": "straight", "width": width, "turns": 0},
        "policy_kind": policy, "field_kind": field,
        "severity_exponent": 0 if field in {"uniform", "none", "random_network"} else 4,
        "total_timesteps": 1024}, "settings": [{"goal_fraction": success,
        "capped_mean_first_goal_steps": steps, "configuration": {}}]}


def test_map_selection_never_uses_uniform_failure():
    """A hard map with failing uniform controls is not eligible for a demonstration."""
    failed = [record("uniform", .5, 900, policy=policy) for policy in ("tabular", "neural")]
    assert campaign.select_maps(failed) == []
    valid = [record("uniform", 1., 300, policy=policy) for policy in ("tabular", "neural")]
    assert campaign.select_maps(valid) == [{"kind": "straight", "width": 32, "turns": 0}]


def test_comparison_requires_matched_successful_uniform():
    """Failure in both initializations cannot qualify as an initialization effect."""
    impaired = record("high_start_cell", 0., 1024)
    assert campaign.choose_comparisons([impaired]) == []
    assert campaign.choose_comparisons([impaired, record("uniform", .5, 900)]) == []
    chosen = campaign.choose_comparisons([impaired, record("uniform", 1., 128)])
    assert len(chosen) == 1 and chosen[0]["field_kind"] == "high_start_cell"


def test_map_selection_accepts_successful_retries_without_counting_them_as_extra_policies():
    """Original and retry records for the same two policies still describe two policy families."""
    original = [record("uniform", .25, 900, policy=policy) for policy in ("tabular", "neural")]
    retried = [record("uniform", 1., 200, policy=policy) for policy in ("tabular", "neural")]
    expected = [{"kind": "straight", "width": 32, "turns": 0}]
    assert campaign.select_maps(original + retried) == expected
    assert campaign.select_maps(retried + original) == expected
    assert campaign.select_maps(original + retried[:1]) == []


def test_comparison_chooses_relative_slowdown():
    """Prefer measured initialization delay rather than general map difficulty."""
    records = [record("uniform", 1., 100), record("high_start_cell", 0., 1000),
               record("uniform", 1., 800, width=64), record("high_start_region", 0., 1024, width=64)]
    assert campaign.choose_comparisons(records)[0]["geometry"]["width"] == 32


def test_sensitivity_contains_entropy_and_effective_horizons():
    """The follow-up batch covers distinct exploration and credit-assignment settings."""
    settings = campaign.sensitivity_settings()
    assert {setting["entropy_coefficient"] for setting in settings} == {0., 2**-14, 2**-10, 2**-6}
    assert {setting["gamma_int"] for setting in settings} == {1-2**-6, 1-2**-8, 1-2**-10, 1-2**-12}


def test_primary_capped_mean_precedes_discovery_fraction():
    """A lower capped mean wins even when its discovery fraction is slightly lower."""
    candidate = record("high_start_cell", 1., 300)
    smaller_mean = {"goal_fraction": .9375, "capped_mean_first_goal_steps": 200,
                    "configuration": {"learning_rate": 2**-10}}
    candidate["settings"].append(smaller_mean)
    assert campaign.best_setting(candidate) is smaller_mean
    candidate["settings"][0]["capped_mean_first_goal_steps"] = 200
    assert campaign.best_setting(candidate)["goal_fraction"] == 1.


def test_reliable_uniform_row_is_used_when_fastest_row_is_unreliable():
    """Eligibility filters reliable uniform settings before minimizing the primary metric."""
    controls = []
    for policy in ("tabular", "neural"):
        uniform = record("uniform", .5, 100, policy=policy)
        uniform["settings"].append({"goal_fraction": .75, "capped_mean_first_goal_steps": 150,
                                     "configuration": {"learning_rate": 2**-10}})
        controls.append(uniform)
    assert campaign.select_maps(controls) == [{"kind": "straight", "width": 32, "turns": 0}]
    adverse = record("high_start_cell", .5, 900)
    assert campaign.choose_comparisons([adverse, *controls])[0]["field_kind"] == "high_start_cell"


def test_confirmation_shared_filters_reliability_and_tuned_minimizes_mean(tmp_path, monkeypatch):
    """Shared uniform and independent tuning apply their separate prescribed selection rules."""
    uniform = record("uniform", .5, 100)
    uniform["settings"][0]["configuration"] = {"learning_rate": 2**-14}
    uniform["settings"].extend([
        {"goal_fraction": .75, "capped_mean_first_goal_steps": 150,
         "configuration": {"learning_rate": 2**-10}},
        {"goal_fraction": 1., "capped_mean_first_goal_steps": 200,
         "configuration": {"learning_rate": 2**-6}}])
    adverse = record("high_start_cell", 1., 500)
    adverse["settings"][0]["configuration"] = {"learning_rate": 2**-6}
    adverse["settings"].append({"goal_fraction": .9375, "capped_mean_first_goal_steps": 400,
                                  "configuration": {"learning_rate": 2**-10}})
    submitted = []
    monkeypatch.setattr(campaign, 'research_task', lambda *args, **kwargs: submitted.append((args, kwargs)))
    campaign.add_confirmation(tmp_path, [adverse['specification']], [uniform, adverse])
    primary = [args for args, kwargs in submitted if args[4] == 'high_start_cell'
               and kwargs.get('variant', '').startswith('rollout-')]
    assert primary
    selected = primary[0][6]
    assert next(row for row in selected if row['comparison'] == 'shared')['learning_rate'] == 2**-10
    assert next(row for row in selected if row['comparison'] == 'independently-tuned')['learning_rate'] == 2**-10


def test_large_parameter_grid_splits_without_changing_scientific_copies(tmp_path, monkeypatch):
    """A 181-setting grid becomes twelve claimable tasks with every setting and seed retained."""
    submitted = []
    monkeypatch.setattr(campaign, 'enqueue', lambda directory, task: submitted.append(task) or True)
    settings = [{'learning_rate': 2.**(-index-1)} for index in range(181)]
    assert campaign.research_task(tmp_path, 'fine-tuning',
        {'kind': 'straight', 'width': 64, 'turns': 0}, 'tabular', 'uniform', 0,
        settings, seed_start=0, seed_count=16)
    specifications = [json.loads(Path(task['command'][-1]).read_text()) for task in submitted]
    assert len(specifications) == 12
    assert len({task['name'] for task in submitted}) == 12
    assert [setting for spec in specifications for setting in spec['settings']] == settings
    assert [spec['setting_offset'] for spec in specifications] == list(range(0, 181, 16))
    assert specifications[-1]['setting_range'] == [176, 180]
    assert all(spec['variant'] == 'primary' and spec['seed_start'] == 0
               and spec['seed_count'] == 16 and spec['total_setting_count'] == 181
               and len(spec['settings'])*spec['seed_count'] <= 256 for spec in specifications)


def test_coarse_task_name_and_structural_variant_are_preserved(tmp_path, monkeypatch):
    """Small live tasks retain their names, while split structural tasks retain their variant."""
    submitted = []
    monkeypatch.setattr(campaign, 'enqueue', lambda directory, task: submitted.append(task) or True)
    geometry = {'kind': 'snake', 'width': 16, 'turns': 2}
    campaign.research_task(tmp_path, 'field-search', geometry, 'neural', 'uniform', 0,
                           campaign.coarse_settings())
    original = json.loads(Path(submitted[0]['command'][-1]).read_text())
    assert len(submitted) == 1 and 'settings-start-' not in original['name']
    assert 'setting_offset' not in original
    submitted.clear()
    campaign.research_task(tmp_path, 'calibration', geometry, 'neural', 'uniform', 0,
                           campaign.sensitivity_settings(), variant='rollout-256')
    split = [json.loads(Path(task['command'][-1]).read_text()) for task in submitted]
    assert len(split) == 4 and all(spec['variant'] == 'rollout-256' for spec in split)
    assert [setting for spec in split for setting in spec['settings']] == campaign.sensitivity_settings()


def test_confirmation_pools_all_split_tuning_records(tmp_path, monkeypatch):
    """The best setting in a later physical task remains eligible for frozen confirmation."""
    first = record('uniform', 1., 200)
    first['settings'][0]['configuration'] = {'learning_rate': 2**-6}
    later = record('uniform', .9375, 150)
    later['settings'][0]['configuration'] = {'learning_rate': 2**-10}
    later['specification'].update(setting_offset=16, setting_range=[16, 31])
    adverse = record('high_start_cell', 1., 900)
    adverse['settings'][0]['configuration'] = {'learning_rate': 2**-6}
    adverse_later = record('high_start_cell', .9375, 500)
    adverse_later['settings'][0]['configuration'] = {'learning_rate': 2**-10}
    adverse_later['specification'].update(setting_offset=16, setting_range=[16, 31])
    submitted = []
    monkeypatch.setattr(campaign, 'research_task', lambda *args, **kwargs: submitted.append((args, kwargs)))
    campaign.add_confirmation(tmp_path, [adverse['specification']], [first, later, adverse, adverse_later])
    selected = next(args[6] for args, kwargs in submitted if args[4] == 'high_start_cell'
                    and kwargs.get('variant', '').startswith('rollout-'))
    assert all(configuration['learning_rate'] == 2**-10 for configuration in selected)


def test_comparison_pools_parameter_slices_and_excludes_confirmation():
    """One poorly performing slice cannot hide its condition's better development settings."""
    control = record('uniform', 1., 100)
    control['specification']['wave'] = 'field-search'
    bad_slice = record('high_start_cell', 0., 1000)
    bad_slice['specification'].update(wave='field-search', setting_offset=0, setting_range=[0, 15])
    good_slice = record('high_start_cell', 1., 200)
    good_slice['specification'].update(wave='diagnosis', setting_offset=16, setting_range=[16, 31])
    slower_condition = record('high_start_region', .75, 500)
    slower_condition['specification']['wave'] = 'diagnosis'
    confirmation = record('high_interior_pair', 0., 1024)
    confirmation['specification']['wave'] = 'confirmation'
    long_confirmation = record('smooth_interior_peak', 0., 1024)
    long_confirmation['specification']['wave'] = 'long-confirmation'
    records = [control, bad_slice, good_slice, slower_condition, confirmation, long_confirmation]
    chosen = campaign.choose_comparisons(records)
    assert len(chosen) == 1 and chosen[0]['field_kind'] == 'high_start_region'
    assert 'setting_offset' not in chosen[0] and 'settings' not in chosen[0]
    assert campaign.choose_comparisons(list(reversed(records))) == chosen
