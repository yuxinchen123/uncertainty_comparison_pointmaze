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

# The thirteen algorithm names in their canonical registry order (REGISTRY is insertion-ordered).
EXPECTED_NAMES = [
    "no_exploration",
    "gt_position",
    "gt_position_velocity",
    "gt_position_maze_cell",
    "gt_position_1m",
    "rnd_next_state",
    "rnd_next_state_position_only",
    "rnd_state",
    "rnd_state_action",
    "rnd_state_action_next_state",
    "rnd_linear_next_state",
    "rnd_elliptical",
    "rnd_elliptical_global",
]

# The documented spec table, one row per algorithm: copied verbatim from the source
# AlgorithmSpec(...) lines so the test fails loudly if any field is edited out of step.
# columns: name -> (kind, rnd_feature, linear_rnd, uses_action, gt_wrapper_kind, builds_model, elliptical_mode)
EXPECTED_SPECS = {
    "no_exploration":               ("none",        None,                           False, False, None,                False, None),
    "gt_position":                  ("visit_count", None,                           False, False, "position",          True,  None),
    "gt_position_velocity":         ("visit_count", None,                           False, False, "position_velocity", True,  None),
    "gt_position_maze_cell":        ("visit_count", None,                           False, False, "position",          True,  None),
    "gt_position_1m":               ("visit_count", None,                           False, False, "position_1m",       True,  None),
    "rnd_next_state":               ("rnd",         "rnd_next_state",               False, False, None,                True,  None),
    "rnd_next_state_position_only": ("rnd",         "rnd_next_state_position_only", False, False, None,                True,  None),
    "rnd_state":                    ("rnd",         "rnd_state",                    False, False, None,                True,  None),
    "rnd_state_action":             ("rnd",         "rnd_state_action",             False, True,  None,                True,  None),
    "rnd_state_action_next_state":  ("rnd",         "rnd_state_action_next_state",  False, True,  None,                True,  None),
    "rnd_linear_next_state":        ("rnd",         "rnd_next_state",               True,  False, None,                True,  None),
    "rnd_elliptical":               ("elliptical",  None,                           False, True,  None,                True,  "batch"),
    "rnd_elliptical_global":        ("elliptical",  None,                           False, True,  None,                True,  "global"),
}


def test_registry_has_expected_names_in_order():
    # Golden path: REGISTRY holds exactly the thirteen documented names in their documented order.
    assert list(REGISTRY) == EXPECTED_NAMES
    # Edge case: exactly thirteen entries (no extras / duplicates) and each key equals its spec.name.
    assert len(REGISTRY) == 13
    assert all(name == spec.name for name, spec in REGISTRY.items())


def test_algorithm_names_mirrors_registry():
    # Golden path: ALGORITHM_NAMES is the registry key list, same content and order.
    assert ALGORITHM_NAMES == list(REGISTRY)
    # Edge case: it is a plain list (a fresh object, not the dict itself) usable as a public copy.
    assert isinstance(ALGORITHM_NAMES, list)
    assert ALGORITHM_NAMES is not REGISTRY


def test_algorithms_no_action_exact_list():
    # Golden path: ALGORITHMS_NO_ACTION is exactly the nine action-free algorithms, in order.
    assert ALGORITHMS_NO_ACTION == [
        "no_exploration",
        "gt_position",
        "gt_position_velocity",
        "gt_position_maze_cell",
        "gt_position_1m",
        "rnd_next_state",
        "rnd_next_state_position_only",
        "rnd_state",
        "rnd_linear_next_state",
    ]
    # Edge case: it is the complement of uses_action -> the four action-using names are absent.
    action_names = {name for name, spec in REGISTRY.items() if spec.uses_action}
    assert action_names == {
        "rnd_state_action", "rnd_state_action_next_state", "rnd_elliptical", "rnd_elliptical_global",
    }
    assert action_names.isdisjoint(ALGORITHMS_NO_ACTION)
    assert ALGORITHMS_NO_ACTION == [n for n, s in REGISTRY.items() if not s.uses_action]


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_each_spec_matches_documented_table(name):
    # Golden path: every AlgorithmSpec field matches the documented spec table, per algorithm.
    spec = REGISTRY[name]
    kind, rnd_feature, linear_rnd, uses_action, gt_wrapper_kind, builds_model, elliptical_mode = EXPECTED_SPECS[name]
    assert (spec.kind, spec.rnd_feature, spec.linear_rnd, spec.uses_action,
            spec.gt_wrapper_kind, spec.builds_model, spec.elliptical_mode) == (
        kind, rnd_feature, linear_rnd, uses_action, gt_wrapper_kind, builds_model, elliptical_mode)
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


def test_build_elliptical_passes_all_cfg_knobs_through():
    """The factory forwards every elliptical cfg knob to the model — including bonus_clip=inf. Guards the
    silent-masking hazard: the factory reads knobs via getattr with defaults, so a knob missing from cfg
    would silently fall back (e.g. every no-clip run secretly clipping at 5)."""
    import types
    from rnd_exploration.methods import EnvContext

    # cfg stub carrying the run-3.1.2 replica knobs (raw features, ridge 1e-6, clip disabled)
    cfg = types.SimpleNamespace(
        device="cpu",
        elliptical_regularization=1e-6,
        elliptical_bonus_clip=float("inf"),
        elliptical_update_timing="sample",
        elliptical_feature_normalization="none",
        elliptical_feature_input="state_action",
    )
    # the elliptical branch only reads obs_shape / action_dim / device from the context
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                     position_wrapper=None, position_velocity_wrapper=None)
    model = build_intrinsic_model("rnd_elliptical", cfg, ctx)
    # golden path: every knob arrives on the model, clip disabled really is +inf
    assert model.bonus_clip == float("inf")
    assert model.regularization == 1e-6
    assert model.feature_normalization == "none"
    assert model.feature_input == "state_action"
    assert model.update_timing == "sample"
    # edge case: a cfg WITHOUT the clip attribute falls back to the documented default 5.0
    del cfg.elliptical_bonus_clip
    model_default = build_intrinsic_model("rnd_elliptical", cfg, ctx)
    assert model_default.bonus_clip == 5.0


def test_build_rnd_forwards_optimizer_readout_knobs():
    """The factory forwards the four run-3.2.1 RND knobs; a cfg without them falls back to adam/mse. Guards
    the silent-masking hazard: the factory reads the knobs via getattr with defaults, so a knob missing from
    cfg would silently pick the wrong optimizer/readout."""
    import types
    import torch
    from rnd_exploration.methods import EnvContext

    # cfg stub: an O3 sgd1t + l2 configuration (obs-norm off so no warmup rollout is needed)
    cfg = types.SimpleNamespace(
        device="cpu",
        rnd_output_dim=16,
        rnd_obs_norm=False,
        rnd_distance="mse",
        n_predictors=1,
        rnd_optimizer="sgd1t",
        rnd_bonus_readout="l2",
        rnd_sgd_eta0=3e-3,
        rnd_sgd_t0=1e4,
    )
    # the RND branch only reads obs_shape / action_dim / device from the context
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                     position_wrapper=None, position_velocity_wrapper=None)
    model = build_intrinsic_model("rnd_next_state", cfg, ctx)
    # golden path: every knob arrives on the model and the SGD optimizer really is constructed
    assert model.optimizer == "sgd1t"
    assert model.bonus_readout == "l2"
    assert model.sgd_eta0 == 3e-3
    assert model.sgd_t0 == 1e4
    assert isinstance(model.opt, torch.optim.SGD)
    # edge case: a cfg WITHOUT the new attrs falls back to the historical adam/mse defaults
    for attr in ("rnd_optimizer", "rnd_bonus_readout", "rnd_sgd_eta0", "rnd_sgd_t0"):
        delattr(cfg, attr)
    model_default = build_intrinsic_model("rnd_next_state", cfg, ctx)
    assert model_default.optimizer == "adam"
    assert model_default.bonus_readout == "mse"
    assert isinstance(model_default.opt, torch.optim.Adam)


def test_build_visit_count_forwards_decay_exponent():
    """The factory forwards cfg.visit_count_decay to VisitCount (1/sqrt(n) default -0.5, 1/n = -1); a cfg
    without the field falls back to -0.5. Guards the silent-getattr-masking hazard for the gt_* oracles."""
    import types
    from rnd_exploration.methods import EnvContext

    # a minimal visit-count wrapper stub with the one method VisitCount reads
    class _W:
        def observation_to_count(self, obs):
            return 4
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                     position_wrapper=_W(), position_velocity_wrapper=_W())
    # 1/n oracle: cfg carries decay -1 -> forwarded to the model
    cfg = types.SimpleNamespace(device="cpu", visit_count_decay=-1.0)
    model = build_intrinsic_model("gt_position_velocity", cfg, ctx)
    assert model.intrinsic_decay_rate == -1.0
    # edge case: a cfg WITHOUT the field falls back to the historical 1/sqrt(n) default (-0.5)
    del cfg.visit_count_decay
    model_default = build_intrinsic_model("gt_position_velocity", cfg, ctx)
    assert model_default.intrinsic_decay_rate == -0.5


def test_build_rnd_forwards_bias_init_and_reward_norm_knobs():
    """The factory forwards the run-3.2.3 RND knobs (bias scheme keyed by a_seed, reward
    normalization + filter discount); a cfg without them falls back to the historical zero-bias,
    no-normalization behavior. Guards the same silent-getattr-masking hazard as the run-3.2.1 test."""
    import types
    import torch
    from rnd_exploration.methods import EnvContext

    # cfg stub: normal_0.5 biases keyed by a_seed=7, reward normalization on with a non-default gamma
    cfg = types.SimpleNamespace(
        device="cpu",
        rnd_output_dim=16,
        rnd_obs_norm=False,
        rnd_distance="mse",
        n_predictors=1,
        a_seed=7,
        rnd_bias_init="normal_0.5",
        rnd_reward_norm=True,
        rnd_reward_norm_gamma=0.97,
    )
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                     position_wrapper=None, position_velocity_wrapper=None)
    model = build_intrinsic_model("rnd_next_state", cfg, ctx)
    # golden path: every knob arrives on the model, biases are really nonzero, stats state exists
    assert model.bias_init == "normal_0.5"
    assert model.bias_seed == 7
    assert model.reward_norm is True
    assert model.reward_norm_gamma == 0.97
    assert model.reward_rms is not None
    assert torch.count_nonzero(model.target.network[0].bias) > 0
    # edge case: a cfg WITHOUT the new attrs falls back to zero biases and no normalization
    for attr in ("rnd_bias_init", "rnd_reward_norm", "rnd_reward_norm_gamma"):
        delattr(cfg, attr)
    model_default = build_intrinsic_model("rnd_next_state", cfg, ctx)
    assert model_default.bias_init == "zero"
    assert model_default.reward_norm is False
    assert model_default.reward_rms is None
    assert torch.count_nonzero(model_default.target.network[0].bias) == 0
    assert torch.count_nonzero(model_default.predictor.network[2].bias) == 0


def test_build_rnd_forwards_original_rnd_knobs():
    """The factory forwards the train-run-5 RND knobs (Adam lr, activation, deeper predictor,
    keep-mask); a cfg without them falls back to the historical 1e-3 / relu / symmetric / full-batch
    behavior. Guards the same silent-getattr-masking hazard as the earlier forwarding tests."""
    import types
    import torch
    import torch.nn as nn
    from rnd_exploration.methods import EnvContext

    # cfg stub: the run-5 original-small knob values (obs-norm off so no warmup env is needed)
    cfg = types.SimpleNamespace(
        device="cpu", rnd_output_dim=16, rnd_obs_norm=False, rnd_distance="mse", n_predictors=1,
        rnd_lr=1e-4, rnd_activation="leaky_relu", rnd_predictor_extra_layers=1,
        rnd_update_proportion=1.0, rnd_bonus_readout="mse_mean",
    )
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                     position_wrapper=None, position_velocity_wrapper=None)
    model = build_intrinsic_model("rnd_next_state", cfg, ctx)
    # golden path: every knob arrives on the model (lr reaches the Adam group, predictor is deeper)
    assert model.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert model.activation == "leaky_relu"
    assert model.predictor_extra_layers == 1
    assert model.bonus_readout == "mse_mean"
    assert len([m for m in model.predictor.network if isinstance(m, nn.Linear)]) == 3
    assert len([m for m in model.target.network if isinstance(m, nn.Linear)]) == 2
    # edge case: a cfg WITHOUT the new attrs falls back to the historical defaults
    for attr in ("rnd_lr", "rnd_activation", "rnd_predictor_extra_layers", "rnd_update_proportion",
                 "rnd_bonus_readout"):
        delattr(cfg, attr)
    model_default = build_intrinsic_model("rnd_next_state", cfg, ctx)
    assert model_default.opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert model_default.activation == "relu"
    assert model_default.predictor_extra_layers == 0
    assert model_default.update_proportion == 1.0
