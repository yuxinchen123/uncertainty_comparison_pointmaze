"""Resolve the old platform's named environment configurations without sharing algorithm defaults.

Pixel-based CleanRL Atari has its own configuration and trainer. These presets reproduce
PointMaze, the historical AntMaze and object-observation Montezuma implementations, and the
current AntMaze rules of the study that started 2026-11-11.
"""
from dataclasses import replace


# The recorded environment class selects a named configuration, never an Atari pixel default.
ENVIRONMENT_PRESETS = {
    "EnvConfig": "pointmaze_current",
    "AntMazeConfig": "antmaze_historical",
    "MontezumaConfig": "montezuma_object_historical",
}
PRESETS = {
    "pointmaze_current": ("EnvConfig", "reset_observation", "task_only"),
    "pointmaze_historical": ("EnvConfig", "final_observation", "weighted_intrinsic"),
    "antmaze_historical": ("AntMazeConfig", "final_observation", "weighted_intrinsic"),
    # the PointMaze rules on the ant: the intrinsic return continues into the reset state, and a
    # no-bonus policy learns from task advantages only. Chosen by name; never the default.
    "antmaze_current": ("AntMazeConfig", "reset_observation", "task_only"),
    "montezuma_object_historical": ("MontezumaConfig", "final_observation", "weighted_intrinsic"),
}


def resolve_configuration(config, environment_config):
    """Return explicit return choices for one supported environment and reject mixed presets."""
    # Resolve the environment-specific name once, then retain it in the frozen configuration.
    # Before: environment_default + AntMazeConfig; after: antmaze_historical + explicit choices.
    environment_type = type(environment_config).__name__
    if environment_type not in ENVIRONMENT_PRESETS:
        raise ValueError(f"no old-platform configuration for environment {environment_type!r}")
    name = (ENVIRONMENT_PRESETS[environment_type]
            if config.configuration_preset == "environment_default" else config.configuration_preset)
    if name not in PRESETS:
        raise ValueError(f"unknown configuration preset {name!r}; choose from {sorted(PRESETS)}")
    expected_environment, intrinsic_observation, no_bonus_stream = PRESETS[name]
    if expected_environment != environment_type:
        raise ValueError(f"configuration {name!r} requires {expected_environment}, "
                         f"not {environment_type}; pixel Atari uses its own CleanRL configuration")

    # A saved explicit choice must agree with its name; changing semantics requires another name.
    for field, expected in (("intrinsic_bootstrap_observation", intrinsic_observation),
                            ("no_bonus_advantage_stream", no_bonus_stream)):
        value = getattr(config, field)
        if value and value != expected:
            raise ValueError(f"{field}={value!r} conflicts with configuration {name!r} "
                             f"which requires {expected!r}")
    if environment_type != "EnvConfig":
        # the diagnostics bundle reads PointMaze's velocity out of the observation, so it stays
        # refused for every other environment; the other study options are refused only under a
        # historical name, which must keep reproducing what it recorded
        pointmaze_study = (config.success_gate_rate > 0
                           or config.normalize_agent_obs or config.normalize_ext_reward
                           or config.clip_per_network or config.logstd_floor > float("-inf")
                           or config.critic_lr_scale != 1.0
                           or config.int_coef_schedule != "constant")
        if config.diagnostics or (pointmaze_study and name.endswith("_historical")):
            raise ValueError("PointMaze study options require a PointMaze configuration; "
                             "historical environments do not inherit them, and the diagnostics "
                             "read PointMaze's velocity (use episode_log for episode counts)")
    return replace(config, configuration_preset=name,
                   intrinsic_bootstrap_observation=intrinsic_observation,
                   no_bonus_advantage_stream=no_bonus_stream)
