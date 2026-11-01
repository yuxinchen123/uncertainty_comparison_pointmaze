"""Check research decisions using fabricated records, without initializing a numerical runtime."""

import importlib.util
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
