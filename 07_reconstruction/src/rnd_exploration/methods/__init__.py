"""Intrinsic-reward methods plus the single algorithm registry.

The registry (AlgorithmSpec / REGISTRY / build_intrinsic_model) is the one source of truth that
replaces the four parallel structures the old code hand-maintained in lock-step: 04's
``_algorithm_to_config``, 04's model-instantiation if/elif, and the ``ALGORITHM_NAMES`` /
``ALGORITHMS_NO_ACTION`` lists that lived in ``distance_to_GT/algorithm_vector.py``.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import IntrinsicRewardModel
from .elliptical_bonus import EllipticalBonus
from .rnd import RND, EnsembleObservationEncoder, ObservationEncoder
from .visit_count import VisitCount


@dataclass(frozen=True)
class AlgorithmSpec:
    """One row of the algorithm registry: how an ``--algorithm`` name maps to an intrinsic model and
    how its distance-to-ground-truth vector is computed."""
    name: str
    kind: str                       # "none" | "rnd" | "visit_count" | "elliptical"
    rnd_feature: Optional[str]      # an RND feature mode, or None for non-RND methods
    linear_rnd: bool
    uses_action: bool               # True => needs actions => excluded from distance-to-ground-truth
    gt_wrapper_kind: Optional[str]  # "position" | "position_velocity" | None (visit-count only)
    builds_model: bool              # False only for no_exploration (which forces beta = 0)


# Single source of truth for the ten algorithms. Values copied verbatim from 04's
# _algorithm_to_config (kind / rnd_feature / linear_rnd), 04's VisitCount instantiation
# (gt_wrapper_kind), and the old ALGORITHMS_NO_ACTION list (uses_action is its complement).
REGISTRY: "dict[str, AlgorithmSpec]" = {
    spec.name: spec for spec in [
        AlgorithmSpec("no_exploration",               "none",        None,                           False, False, None,                False),
        AlgorithmSpec("gt_position",                  "visit_count", None,                           False, False, "position",          True),
        AlgorithmSpec("gt_position_velocity",         "visit_count", None,                           False, False, "position_velocity", True),
        AlgorithmSpec("rnd_next_state",               "rnd",         "rnd_next_state",               False, False, None,                True),
        AlgorithmSpec("rnd_next_state_position_only", "rnd",         "rnd_next_state_position_only", False, False, None,                True),
        AlgorithmSpec("rnd_state",                    "rnd",         "rnd_state",                    False, False, None,                True),
        AlgorithmSpec("rnd_state_action",             "rnd",         "rnd_state_action",             False, True,  None,                True),
        AlgorithmSpec("rnd_state_action_next_state",  "rnd",         "rnd_state_action_next_state",  False, True,  None,                True),
        AlgorithmSpec("rnd_linear_next_state",        "rnd",         "rnd_next_state",               True,  False, None,                True),
        AlgorithmSpec("rnd_elliptical",               "elliptical",  None,                           False, True,  None,                True),
    ]
}

# Derived lists (replace the hand-copies that lived in distance_to_GT and in the test suite).
ALGORITHM_NAMES = list(REGISTRY)
ALGORITHMS_NO_ACTION = [name for name, spec in REGISTRY.items() if not spec.uses_action]


@dataclass
class EnvContext:
    """Runtime env handles the factory needs; built by train.py and passed in so methods/ never
    imports train.py or envs/ (keeps the import graph acyclic)."""
    obs_shape: tuple
    action_dim: int
    observation_space: Any
    action_space: Any
    position_wrapper: Any            # PositionVisitCountWrapper (for gt_position)
    position_velocity_wrapper: Any   # PositionVelocityVisitCountWrapper (for gt_position_velocity)


def _warmup_obs_rms(model: "RND", ctx: EnvContext) -> None:
    """Seed an RND model's observation running-mean/std from 200 sampled transitions (exactly 04's
    inline warmup). No-op if the model keeps no obs_rms (use_obs_norm=False)."""
    # only an RND with use_obs_norm has obs_rms; nothing to seed otherwise
    if getattr(model, "obs_rms", None) is None:
        return
    # sample 200 (obs, next_obs, action) triples from the env spaces
    # before: empty buffers; after: each is a list of 200 float32 arrays
    obs_buf, next_buf, act_buf = [], [], []
    for _ in range(200):
        obs_buf.append(np.asarray(ctx.observation_space.sample(), dtype=np.float32))
        next_buf.append(np.asarray(ctx.observation_space.sample(), dtype=np.float32))
        act_buf.append(np.asarray(ctx.action_space.sample(), dtype=np.float32))
    samples = {
        "observations": np.stack(obs_buf, axis=0),       # (200, obs_dim)
        "next_observations": np.stack(next_buf, axis=0), # (200, obs_dim)
        "actions": np.stack(act_buf, axis=0),            # (200, action_dim)
    }
    # update obs_rms with whatever feature slice this RND mode consumes (obs / obs+act / +next_obs)
    x = model._get_feature_tensor(samples)
    model.obs_rms.update(x.detach().cpu().numpy())


def build_intrinsic_model(name: str, cfg: Any, ctx: EnvContext) -> Optional[IntrinsicRewardModel]:
    """Build the intrinsic model for algorithm `name`, or None for no_exploration (the caller then
    forces beta=0). `cfg` is duck-typed (reads cfg.rnd_output_dim / rnd_obs_norm / rnd_distance /
    n_predictors / device) so methods/ never imports train.py. Raises on an unknown algorithm."""
    # one registry lookup; an unknown name fails loud (no sentinel)
    if name not in REGISTRY:
        raise ValueError(f"algorithm must be one of {ALGORITHM_NAMES}; got {name!r}")
    spec = REGISTRY[name]
    # no_exploration: no intrinsic model (beta forced to 0 by the caller)
    if not spec.builds_model:
        return None
    # RND family: feature mode + linear flag come from the spec; obs-RMS warmed up like 04
    if spec.kind == "rnd":
        model = RND(
            obs_shape=ctx.obs_shape,
            output_dim=cfg.rnd_output_dim,
            lr=0.001,
            batch_size=256,
            device=cfg.device,
            use_obs_norm=cfg.rnd_obs_norm,
            distance=cfg.rnd_distance,
            n_predictors=cfg.n_predictors,
            beta_std=0.0,
            linear_rnd=spec.linear_rnd,
            feature=spec.rnd_feature,
            action_dim=ctx.action_dim,
        )
        if cfg.rnd_obs_norm:
            _warmup_obs_rms(model, ctx)
        return model
    # VisitCount (oracle bonus): pick the visit-count wrapper the spec names
    if spec.kind == "visit_count":
        wrapper = (ctx.position_velocity_wrapper if spec.gt_wrapper_kind == "position_velocity"
                   else ctx.position_wrapper)
        return VisitCount(wrapper)
    # EllipticalBonus (Mahalanobis / UCB bonus)
    if spec.kind == "elliptical":
        return EllipticalBonus(
            obs_shape=ctx.obs_shape,
            action_dim=ctx.action_dim,
            feature_dim=128,
            device=cfg.device,
            regularization=1e-6,
        )
    raise ValueError(f"unhandled kind {spec.kind!r} for algorithm {name!r}")


__all__ = [
    "IntrinsicRewardModel",
    "EllipticalBonus",
    "RND",
    "ObservationEncoder",
    "EnsembleObservationEncoder",
    "VisitCount",
    "AlgorithmSpec",
    "REGISTRY",
    "ALGORITHM_NAMES",
    "ALGORITHMS_NO_ACTION",
    "EnvContext",
    "build_intrinsic_model",
]
