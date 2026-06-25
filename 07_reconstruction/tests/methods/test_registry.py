"""Tests for the single algorithm registry in rnd_exploration.methods.

These cover the pure-data parts of the registry (REGISTRY / ALGORITHM_NAMES /
ALGORITHMS_NO_ACTION / each AlgorithmSpec field) plus the two light branches of
build_intrinsic_model that need no env (unknown -> ValueError, no_exploration -> None).
The heavy branches that construct RND / VisitCount / EllipticalBonus need an EnvContext
and are intentionally not exercised here.
"""
import dataclasses

import pytest

from rnd_exploration.methods import (
    ALGORITHM_NAMES,
    ALGORITHMS_NO_ACTION,
    REGISTRY,
    AlgorithmSpec,
    build_intrinsic_model,
)

# The ten algorithm names in their canonical registry order (REGISTRY is insertion-ordered).
EXPECTED_NAMES = [
    "no_exploration",
    "gt_position",
    "gt_position_velocity",
    "rnd_next_state",
    "rnd_next_state_position_only",
    "rnd_state",
    "rnd_state_action",
    "rnd_state_action_next_state",
    "rnd_linear_next_state",
    "rnd_elliptical",
]

# The documented spec table, one row per algorithm: copied verbatim from the source
# AlgorithmSpec(...) lines so the test fails loudly if any field is edited out of step.
# columns: name -> (kind, rnd_feature, linear_rnd, uses_action, gt_wrapper_kind, builds_model)
EXPECTED_SPECS = {
    "no_exploration":               ("none",        None,                           False, False, None,                False),
    "gt_position":                  ("visit_count", None,                           False, False, "position",          True),
    "gt_position_velocity":         ("visit_count", None,                           False, False, "position_velocity", True),
    "rnd_next_state":               ("rnd",         "rnd_next_state",               False, False, None,                True),
    "rnd_next_state_position_only": ("rnd",         "rnd_next_state_position_only", False, False, None,                True),
    "rnd_state":                    ("rnd",         "rnd_state",                    False, False, None,                True),
    "rnd_state_action":             ("rnd",         "rnd_state_action",             False, True,  None,                True),
    "rnd_state_action_next_state":  ("rnd",         "rnd_state_action_next_state",  False, True,  None,                True),
    "rnd_linear_next_state":        ("rnd",         "rnd_next_state",               True,  False, None,                True),
    "rnd_elliptical":               ("elliptical",  None,                           False, True,  None,                True),
}


def test_registry_has_expected_names_in_order():
    # Golden path: REGISTRY holds exactly the ten documented names in their documented order.
    assert list(REGISTRY) == EXPECTED_NAMES
    # Edge case: exactly ten entries (no extras / duplicates) and each key equals its spec.name.
    assert len(REGISTRY) == 10
    assert all(name == spec.name for name, spec in REGISTRY.items())


def test_algorithm_names_mirrors_registry():
    # Golden path: ALGORITHM_NAMES is the registry key list, same content and order.
    assert ALGORITHM_NAMES == list(REGISTRY)
    # Edge case: it is a plain list (a fresh object, not the dict itself) usable as a public copy.
    assert isinstance(ALGORITHM_NAMES, list)
    assert ALGORITHM_NAMES is not REGISTRY


def test_algorithms_no_action_exact_list():
    # Golden path: ALGORITHMS_NO_ACTION is exactly the seven action-free algorithms, in order.
    assert ALGORITHMS_NO_ACTION == [
        "no_exploration",
        "gt_position",
        "gt_position_velocity",
        "rnd_next_state",
        "rnd_next_state_position_only",
        "rnd_state",
        "rnd_linear_next_state",
    ]
    # Edge case: it is the complement of uses_action -> the three action-using names are absent.
    action_names = {name for name, spec in REGISTRY.items() if spec.uses_action}
    assert action_names == {"rnd_state_action", "rnd_state_action_next_state", "rnd_elliptical"}
    assert action_names.isdisjoint(ALGORITHMS_NO_ACTION)
    assert ALGORITHMS_NO_ACTION == [n for n, s in REGISTRY.items() if not s.uses_action]


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_each_spec_matches_documented_table(name):
    # Golden path: every AlgorithmSpec field matches the documented spec table, per algorithm.
    spec = REGISTRY[name]
    kind, rnd_feature, linear_rnd, uses_action, gt_wrapper_kind, builds_model = EXPECTED_SPECS[name]
    assert (spec.kind, spec.rnd_feature, spec.linear_rnd, spec.uses_action,
            spec.gt_wrapper_kind, spec.builds_model) == (
        kind, rnd_feature, linear_rnd, uses_action, gt_wrapper_kind, builds_model)
    # Edge case: only no_exploration has builds_model False; everything else builds a model.
    assert spec.builds_model == (name != "no_exploration")


def test_algorithm_spec_is_frozen():
    # Golden path: a spec exposes the documented attribute set as a frozen dataclass row.
    spec = REGISTRY["no_exploration"]
    assert isinstance(spec, AlgorithmSpec)
    # Edge case: frozen means assignment is rejected (registry rows are immutable source-of-truth).
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kind = "rnd"


def test_build_intrinsic_model_unknown_name_raises():
    # Golden path: an unknown algorithm name fails loud with ValueError (no sentinel return).
    with pytest.raises(ValueError, match="not_a_real_algorithm"):
        build_intrinsic_model("not_a_real_algorithm", cfg=None, ctx=None)
    # Edge case: the empty string is also unknown and rejected the same way.
    with pytest.raises(ValueError):
        build_intrinsic_model("", cfg=None, ctx=None)


def test_build_intrinsic_model_no_exploration_returns_none():
    # Golden path: no_exploration builds no intrinsic model (caller forces beta=0).
    assert build_intrinsic_model("no_exploration", cfg=None, ctx=None) is None
    # Edge case: the builds_model=False short-circuit never touches cfg/ctx, so None args are fine.
    assert REGISTRY["no_exploration"].builds_model is False
