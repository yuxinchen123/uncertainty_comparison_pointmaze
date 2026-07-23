"""Intrinsic-reward methods plus the single algorithm registry.

The registry (AlgorithmSpec / REGISTRY / build_intrinsic_model) is the one source of truth that
replaces the four parallel structures the old code hand-maintained in lock-step: 04's
``_algorithm_to_config``, 04's model-instantiation if/elif, and the ``ALGORITHM_NAMES`` /
``ALGORITHMS_NO_ACTION`` lists that lived in ``distance_to_GT/algorithm_vector.py``.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .base import IntrinsicRewardModel
from .elliptical_bonus import EllipticalBonus, GlobalEllipticalBonus
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
    elliptical_mode: Optional[str] = None  # "batch" | "global" (elliptical only)


# Single source of truth for the algorithms. Values copied verbatim from 04's
# _algorithm_to_config (kind / rnd_feature / linear_rnd), 04's VisitCount instantiation
# (gt_wrapper_kind), and the old ALGORITHMS_NO_ACTION list (uses_action is its complement). The
# elliptical family carries elliptical_mode (batch / global); both are action-conditioned
# (uses_action=True), so neither gets a distance-to-ground-truth vector.
REGISTRY: "dict[str, AlgorithmSpec]" = {
    spec.name: spec for spec in [
        AlgorithmSpec("no_exploration",               "none",        None,                           False, False, None,                False),
        AlgorithmSpec("gt_position",                  "visit_count", None,                           False, False, "position",          True),
        AlgorithmSpec("gt_position_velocity",         "visit_count", None,                           False, False, "position_velocity", True),
        # section-8 AntMaze ground-truth variants (2026-07): position-only counts on the maze-cell
        # grid (4 m cells) vs on the 1 m sub-grid — two separately named algorithms so both race.
        AlgorithmSpec("gt_position_maze_cell",        "visit_count", None,                           False, False, "position",          True),
        AlgorithmSpec("gt_position_1m",               "visit_count", None,                           False, False, "position_1m",       True),
        AlgorithmSpec("rnd_next_state",               "rnd",         "rnd_next_state",               False, False, None,                True),
        AlgorithmSpec("rnd_next_state_position_only", "rnd",         "rnd_next_state_position_only", False, False, None,                True),
        AlgorithmSpec("rnd_state",                    "rnd",         "rnd_state",                    False, False, None,                True),
        AlgorithmSpec("rnd_state_action",             "rnd",         "rnd_state_action",             False, True,  None,                True),
        AlgorithmSpec("rnd_state_action_next_state",  "rnd",         "rnd_state_action_next_state",  False, True,  None,                True),
        AlgorithmSpec("rnd_linear_next_state",        "rnd",         "rnd_next_state",               True,  False, None,                True),
        AlgorithmSpec("rnd_elliptical",               "elliptical",  None,                           False, True,  None,                True, "batch"),
        AlgorithmSpec("rnd_elliptical_global",        "elliptical",  None,                           False, True,  None,                True, "global"),
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
    position_wrapper: Any            # PositionVisitCountWrapper (for gt_position / gt_position_maze_cell)
    position_velocity_wrapper: Any   # PositionVelocityVisitCountWrapper (for gt_position_velocity)
    env: Any = None                  # a fresh flat env for env-steps obs-RMS warmup (None otherwise)
    position_1m_wrapper: Any = None  # 1 m sub-grid PositionVisitCountWrapper (for gt_position_1m)


def _warmup_obs_rms(model: "RND", cfg: Any, ctx: EnvContext) -> None:
    """Seed an RND model's observation running-mean/std before training. Two modes (cfg, duck-typed):
    'space_sample' (default, exactly 04's inline warmup) draws obs/action from the env SPACES;
    'env_steps' (original RND) rolls a random agent on a fresh env for rnd_obs_warmup_steps steps.
    rnd_obs_warmup_steps defaults to 200. No-op if the model keeps no obs_rms (use_obs_norm=False)."""
    # only an RND with use_obs_norm has obs_rms; nothing to seed otherwise
    if getattr(model, "obs_rms", None) is None:
        return
    mode = getattr(cfg, "rnd_obs_warmup_mode", "space_sample")
    steps = int(getattr(cfg, "rnd_obs_warmup_steps", 200))
    if mode == "space_sample":
        # sample `steps` (obs, next_obs, action) triples from the env spaces (default 200 = 04's warmup)
        # before: empty buffers; after: each is a list of `steps` float32 arrays
        obs_buf, next_buf, act_buf = [], [], []
        for _ in range(steps):
            obs_buf.append(np.asarray(ctx.observation_space.sample(), dtype=np.float32))
            next_buf.append(np.asarray(ctx.observation_space.sample(), dtype=np.float32))
            act_buf.append(np.asarray(ctx.action_space.sample(), dtype=np.float32))
        samples = {
            "observations": np.stack(obs_buf, axis=0),       # (steps, obs_dim)
            "next_observations": np.stack(next_buf, axis=0), # (steps, obs_dim)
            "actions": np.stack(act_buf, axis=0),            # (steps, action_dim)
        }
    elif mode == "env_steps":
        # original RND: roll a RANDOM agent on a FRESH env (ctx.env, its own visit-count maps so the
        # training counts are untouched) for `steps` steps; collect (obs, next_obs, action) in time order.
        if ctx.env is None:
            raise ValueError("rnd_obs_warmup_mode='env_steps' needs ctx.env (a fresh flat env)")
        # keyed warmup seed (rng-seeding rule): its own stream, so it never collides with the run seed
        seed = int(hashlib.sha256(f"{getattr(cfg, 'a_seed', 0)}::rnd_obs_warmup".encode()).hexdigest(),
                   16) & 0x7FFFFFFF
        obs, _ = ctx.env.reset(seed=seed)
        ctx.env.action_space.seed(seed)
        obs_buf, next_buf, act_buf = [], [], []
        for _ in range(steps):
            action = ctx.env.action_space.sample()
            next_obs, _, terminated, truncated, _ = ctx.env.step(action)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            next_buf.append(np.asarray(next_obs, dtype=np.float32))
            act_buf.append(np.asarray(action, dtype=np.float32))
            obs = next_obs
            # reset on episode end (env_max_episode=400 => ~steps/400 episodes)
            if terminated or truncated:
                obs, _ = ctx.env.reset()
        samples = {
            "observations": np.stack(obs_buf, axis=0),
            "next_observations": np.stack(next_buf, axis=0),
            "actions": np.stack(act_buf, axis=0),
        }
    else:
        raise ValueError(f"rnd_obs_warmup_mode must be 'space_sample' or 'env_steps'; got {mode!r}")
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
            # Adam/Adagrad predictor lr, duck-typed (default 0.001 = the historical hardcoded value;
            # ignored by sgd1t, which uses sgd_eta0). Original RND uses 1e-4.
            lr=getattr(cfg, "rnd_lr", 0.001),
            batch_size=256,
            device=cfg.device,
            use_obs_norm=cfg.rnd_obs_norm,
            distance=cfg.rnd_distance,
            n_predictors=cfg.n_predictors,
            beta_std=0.0,
            linear_rnd=spec.linear_rnd,
            feature=spec.rnd_feature,
            action_dim=ctx.action_dim,
            # run-3.2.1 optimizer/readout knobs, duck-typed like the elliptical block below: the
            # getattr defaults MUST equal the Config defaults so a cfg without the fields (older
            # configs, test stubs) reproduces the historical adam/mse behavior exactly
            optimizer=getattr(cfg, "rnd_optimizer", "adam"),
            bonus_readout=getattr(cfg, "rnd_bonus_readout", "mse"),
            sgd_eta0=getattr(cfg, "rnd_sgd_eta0", 1e-2),
            sgd_t0=getattr(cfg, "rnd_sgd_t0", 1e3),
            # run-3.2.3 / convergence-run-1 knobs, same duck-typed pattern: weight + bias
            # initialization for both nets (keyed by the run seed so the bias draws match the
            # 2026-07-10 bias-ablation fields) and the classic-RND intrinsic reward normalization
            # (off by default = historical behavior)
            weight_init=getattr(cfg, "rnd_weight_init", "orthogonal"),
            bias_init=getattr(cfg, "rnd_bias_init", "zero"),
            bias_seed=getattr(cfg, "a_seed", 0),
            reward_norm=getattr(cfg, "rnd_reward_norm", False),
            reward_norm_gamma=getattr(cfg, "rnd_reward_norm_gamma", 0.99),
            # train-run-5 original-RND knobs, same duck-typed pattern (defaults reproduce the historical
            # relu / symmetric / full-batch behavior): hidden activation, deeper predictor, keep-mask.
            activation=getattr(cfg, "rnd_activation", "relu"),
            predictor_extra_layers=getattr(cfg, "rnd_predictor_extra_layers", 0),
            update_proportion=getattr(cfg, "rnd_update_proportion", 1.0),
        )
        if cfg.rnd_obs_norm:
            _warmup_obs_rms(model, cfg, ctx)
        return model
    # VisitCount (oracle bonus): pick the visit-count wrapper the spec names; the count->bonus decay
    # exponent is read from cfg (duck-typed via getattr; default -0.5 => 1/sqrt(n), -1 => 1/n) so
    # methods/ never imports train.py and a cfg without the field reproduces the historical 1/sqrt(n).
    if spec.kind == "visit_count":
        if spec.gt_wrapper_kind == "position_velocity":
            wrapper = ctx.position_velocity_wrapper
        elif spec.gt_wrapper_kind == "position_1m":
            wrapper = ctx.position_1m_wrapper
            if wrapper is None:
                raise ValueError(f"algorithm {name!r} needs ctx.position_1m_wrapper (the 1 m sub-grid counts)")
        else:
            wrapper = ctx.position_wrapper
        return VisitCount(wrapper, intrinsic_decay_rate=getattr(cfg, "visit_count_decay", -0.5))
    # Elliptical family (Mahalanobis / UCB bonus): one shared config; the class is picked by
    # elliptical_mode. The ridge λ, feature normalization, update timing, and encoder input are read
    # from cfg (duck-typed via getattr so methods/ never imports train.py); they default to the
    # family's unit-norm, sample-time, (s,a)-input behavior when cfg omits them.
    if spec.kind == "elliptical":
        common = dict(
            obs_shape=ctx.obs_shape,
            action_dim=ctx.action_dim,
            feature_dim=128,
            device=cfg.device,
            regularization=getattr(cfg, "elliptical_regularization", 1e-2),
            bonus_clip=getattr(cfg, "elliptical_bonus_clip", 5.0),
            update_timing=getattr(cfg, "elliptical_update_timing", "sample"),
            feature_normalization=getattr(cfg, "elliptical_feature_normalization", "unit"),
            feature_input=getattr(cfg, "elliptical_feature_input", "state_action"),
        )
        if spec.elliptical_mode == "batch":
            return EllipticalBonus(**common)
        if spec.elliptical_mode == "global":
            return GlobalEllipticalBonus(**common)
        raise ValueError(f"unknown elliptical_mode {spec.elliptical_mode!r} for algorithm {name!r}")
    raise ValueError(f"unhandled kind {spec.kind!r} for algorithm {name!r}")


__all__ = [
    "IntrinsicRewardModel",
    "EllipticalBonus",
    "GlobalEllipticalBonus",
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
