"""Read complete old-platform experiment files without importing numerical libraries."""

import argparse
import ast
from functools import cache
import hashlib
from pathlib import Path
import sys
import tomllib

SOURCE = Path(__file__).resolve().parents[1] / "src/exploration_platform"
ENVIRONMENT_FILES = {
    "pointmaze": ("envs/pointmaze/pm_common.py", "EnvConfig"),
    "antmaze": ("envs/antmaze/am_common.py", "AntMazeConfig"),
    "montezuma_object": ("envs/atari_montezuma/jax_montezuma.py", "MontezumaConfig"),
}
PRESET_FAMILIES = {
    "pointmaze_current": "pointmaze", "pointmaze_historical": "pointmaze",
    "antmaze_historical": "antmaze", "montezuma_object_historical": "montezuma_object",
}
TRAINER_ARGUMENTS = {
    "configuration_preset": "configuration_preset", "n_copies": "copies",
    "n_envs": "envs_per_copy", "num_steps": "rollout_steps", "update_style": "update_style",
    "learning_rate": "learning_rate", "int_coef": "intrinsic_coefficient",
    "copies_per_group": "copies_per_cell", "flat_params": "flat_params",
    "scan_unroll": "scan_unroll", "base_seed": "base_seed",
    "copy_seed_offset": "copy_seed_offset", "track_coverage": "track_coverage",
    "ent_coef": "ent_coef", "gamma_ext": "gamma_ext", "ext_coef": "ext_coef",
    "max_grad_norm": "max_grad_norm", "update_epochs": "update_epochs",
    "num_minibatches": "num_minibatches", "diagnostics": "diagnostics",
    "int_coef_schedule": "intrinsic_weight_schedule", "success_gate_rate": "success_gate_rate",
    "normalize_ext_reward": "normalize_extrinsic_reward", "clip_per_network": "clip_per_network",
    "logstd_floor": "logstd_floor", "critic_lr_scale": "critic_lr_scale", "vf_coef": "vf_coef",
    "normalize_agent_obs": "normalize_agent_obs",
}
RUNTIME_FIELDS = {"environment", "bonus", "iterations", "window_iterations", "run_seed",
                  "lr_anneal_iterations"}
TUPLE_FIELDS = {"learning_rates", "betas", "parameter_settings"}
SPECIAL_TRAINER_FIELDS = {"n_copies", "learning_rates", "betas", "parameter_settings",
                          "copies_per_group", "learning_rate", "int_coef", "total_iterations",
                          "bootstrap_on_truncation"}


@cache
def configuration_fields(relative_path, class_name):
    """Read dataclass field names from source without executing package initialization."""
    # Importing exploration_platform would initialize JAX; AST inspection reads only declarations.
    module = ast.parse((SOURCE / relative_path).read_text())
    declaration = next(node for node in module.body
                       if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {node.target.id for node in declaration.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}


def read_experiment_configuration(path):
    """Require complete trainer/environment fields and an explicit matching historical identity."""
    path = Path(path).resolve()
    content = path.read_bytes()
    document = tomllib.loads(content.decode())
    if set(document) != {"metadata", "trainer", "environment", "runtime"}:
        raise ValueError("old-platform TOML requires metadata, trainer, environment and runtime sections; "
                         "pixel CleanRL files use their separate command")
    metadata, trainer, environment, runtime = (document[name] for name in
                                               ("metadata", "trainer", "environment", "runtime"))
    if metadata.get("implementation") != "jax_exploration_platform":
        raise ValueError("this command accepts jax_exploration_platform configurations only")
    family = metadata.get("environment_family")
    if family not in ENVIRONMENT_FILES:
        raise ValueError(f"unknown configuration environment family {family!r}")
    if not metadata.get("name") or not metadata.get("source_files"):
        raise ValueError("configuration metadata must name the experiment and its source files")
    expected = configuration_fields("agents/ppo/config.py", "PPOConfig")
    if set(trainer) != expected:
        raise ValueError(f"trainer fields differ: missing={sorted(expected-set(trainer))}, "
                         f"unknown={sorted(set(trainer)-expected)}")
    expected_environment = configuration_fields(*ENVIRONMENT_FILES[family])
    if set(environment) != expected_environment:
        raise ValueError(f"environment fields differ: missing={sorted(expected_environment-set(environment))}, "
                         f"unknown={sorted(set(environment)-expected_environment)}")
    if set(runtime) != RUNTIME_FIELDS:
        raise ValueError(f"runtime must specify exactly {sorted(RUNTIME_FIELDS)}")
    if PRESET_FAMILIES.get(trainer["configuration_preset"]) != family:
        raise ValueError("configuration preset and environment family disagree")
    selected_environment = runtime["environment"]
    expected_selector = {"pointmaze": "pointmaze-large", "montezuma_object": "montezuma"}
    if family in expected_selector and selected_environment != expected_selector[family]:
        raise ValueError("runtime environment and configuration family disagree")
    if family == "antmaze" and selected_environment != "antmaze-" + environment["map_name"]:
        raise ValueError("runtime AntMaze selector and recorded maze map disagree")

    # Before: TOML arrays; after: immutable tuple settings and row/column cells for dataclasses.
    trainer = dict(trainer)
    for name in TUPLE_FIELDS:
        trainer[name] = tuple(tuple(pair) for pair in trainer[name]) if name == "parameter_settings" else tuple(trainer[name])
    environment = {name: tuple(value) if name in ("start_cell", "goal_cell") else value
                   for name, value in environment.items()}
    return {"metadata": metadata, "trainer": trainer, "environment": environment,
            "runtime": runtime, "path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def argument_defaults(document):
    """Translate file values into existing CLI defaults while retaining unexposed fields."""
    trainer, environment, runtime = (document[name] for name in ("trainer", "environment", "runtime"))
    defaults = {destination: trainer[name] for name, destination in TRAINER_ARGUMENTS.items()}
    defaults.update({name: runtime[name] for name in RUNTIME_FIELDS - {"environment"}})
    defaults.update(env=runtime["environment"], description=document["metadata"]["name"],
                    learning_rates=",".join(str(value) for value in trainer["learning_rates"]),
                    intrinsic_weights=",".join(str(value) for value in trainer["betas"]),
                    no_bootstrap_on_truncation=not trainer["bootstrap_on_truncation"],
                    episode_steps=environment["max_episode_steps"], reward_shift=environment["reward_shift"])
    if document["metadata"]["environment_family"] == "pointmaze":
        defaults.update(map=environment["map_name"], start_cell=",".join(map(str, environment["start_cell"])),
                        goal_cell=",".join(map(str, environment["goal_cell"])),
                        terminate_at_goal=not environment["continuing_task"],
                        goal_reward=environment["goal_reward"])
    return defaults


def parse_with_configuration_file(parser, argv=None):
    """Apply a named file first, then honor explicitly supplied ordinary CLI options."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--configuration-file", type=Path)
    known, _ = preliminary.parse_known_args(arguments)
    if known.configuration_file is None:
        return parser.parse_args(arguments)
    document = read_experiment_configuration(known.configuration_file)
    parser.set_defaults(**argument_defaults(document))
    parsed = parser.parse_args(arguments)
    if parsed.env != document["runtime"]["environment"]:
        raise ValueError("--env differs from the configuration file; select that environment's own file")
    explicit = {parser._option_string_actions[token.split("=", 1)[0]].dest
                for token in arguments if token.split("=", 1)[0] in parser._option_string_actions}

    # Scalar overrides change every supplied cell; explicit list/JSON overrides replace the file grid.
    settings = document["trainer"]["parameter_settings"]
    if parsed.parameter_settings:
        settings = ()
        if "learning_rates" not in explicit:
            parsed.learning_rates = ""
        if "intrinsic_weights" not in explicit:
            parsed.intrinsic_weights = ""
    elif explicit & {"learning_rates", "intrinsic_weights"}:
        if settings:
            if "learning_rates" not in explicit:
                parsed.learning_rates = ",".join(str(value) for value in dict.fromkeys(pair[0] for pair in settings))
            if "intrinsic_weights" not in explicit:
                parsed.intrinsic_weights = ",".join(str(value) for value in dict.fromkeys(pair[1] for pair in settings))
        settings = ()
    elif settings:
        settings = tuple(dict.fromkeys((parsed.learning_rate if "learning_rate" in explicit else rate,
                                       parsed.intrinsic_coefficient if "intrinsic_coefficient" in explicit else weight)
                                      for rate, weight in settings))
    if not settings:
        if "learning_rate" in explicit and "learning_rates" not in explicit and parsed.learning_rates:
            parsed.learning_rates = str(parsed.learning_rate)
        if "intrinsic_coefficient" in explicit and "intrinsic_weights" not in explicit and parsed.intrinsic_weights:
            parsed.intrinsic_weights = str(parsed.intrinsic_coefficient)
    parsed.experiment_configuration_data = document
    parsed.experiment_parameter_settings = settings
    return parsed


def additional_trainer_fields(arguments):
    """Retain file knobs not exposed by the existing CLI or its sweep-construction logic."""
    document = getattr(arguments, "experiment_configuration_data", None)
    if document is None:
        return {}
    exposed = set(TRAINER_ARGUMENTS) | SPECIAL_TRAINER_FIELDS
    return {name: value for name, value in document["trainer"].items() if name not in exposed}
