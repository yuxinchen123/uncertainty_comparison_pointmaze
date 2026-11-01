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
        "policy_kind": policy, "field_kind": field, "severity_exponent": 4,
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
