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
    "antmaze_historical": "antmaze", "antmaze_current": "antmaze",
    "montezuma_object_historical": "montezuma_object",
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
    "clip_coef": "clip_coef", "prime_iterations": "prime_iterations",
    "episode_log": "episode_log",
    "gather_minibatches_per_epoch": "gather_minibatches_per_epoch",
}
RUNTIME_FIELDS = {"environment", "bonus", "iterations", "window_iterations", "run_seed",
                  "lr_anneal_iterations", "reward_setting"}
TUPLE_FIELDS = {"learning_rates", "betas", "parameter_settings", "setting_names"}
SPECIAL_TRAINER_FIELDS = {"n_copies", "learning_rates", "betas", "parameter_settings",
                          "copies_per_group", "learning_rate", "int_coef", "total_iterations",
                          "bootstrap_on_truncation", "setting_names", "ext_reward_scale",
                          "ext_reward_offset"}
DEFAULT_SETTING_NAMES = ("learning_rate", "int_coef")
REWARD_SETTING_NAMES = ("ext_reward_scale", "ext_reward_offset")
# the command-line option that replaces each per-cell value when it is given explicitly
SETTING_ARGUMENTS = {"learning_rate": "learning_rate", "int_coef": "intrinsic_coefficient",
                     "ent_coef": "ent_coef", "clip_coef": "clip_coef", "vf_coef": "vf_coef"}
REWARD_SETTINGS = ("environment", "arrival_plus_one", "step_cost",
                   "arrival_plus_one_and_step_cost")


def reward_mappings(reward_setting, step_limit):
    """The (scale, offset) pairs a named reward setting puts on the environment's reward.

    The environment pays 1 on arrival at the goal and 0 on every other step, and the episode ends
    on arrival. `arrival_plus_one` keeps that reward. `step_cost` maps a reward r to
    (r - 1) / step_limit: every step costs 1 / step_limit and arrival pays 0.

    before: "environment", 400                    -> () (the trainer's own scale and offset)
    before: "step_cost", 400                      -> ((0.0025, -0.0025),)
    before: "arrival_plus_one_and_step_cost", 400 -> ((1.0, 0.0), (0.0025, -0.0025))
    """
    if reward_setting not in REWARD_SETTINGS:
        raise ValueError(f"unknown reward setting {reward_setting!r}; choose from "
                         f"{list(REWARD_SETTINGS)}")
    arrival, cost = (1.0, 0.0), (1.0 / step_limit, -1.0 / step_limit)
    return {"environment": (), "arrival_plus_one": (arrival,), "step_cost": (cost,),
            "arrival_plus_one_and_step_cost": (arrival, cost)}[reward_setting]


def check_reward_setting(trainer, environment, reward_setting):
    """Require a file's reward scale and offset to be the ones its named reward setting means."""
    pairs = reward_mappings(reward_setting, environment["max_episode_steps"])
    names = tuple(trainer["setting_names"])
    per_cell = all(name in names for name in REWARD_SETTING_NAMES)
    if not pairs:
        return
    if (environment.get("continuing_task", True) or environment["reward_shift"] != 0
            or environment.get("goal_reward", "per_step") != "per_step"):
        raise ValueError(f"reward setting {reward_setting!r} needs an environment that ends the "
                         "episode at the goal and pays 1 there: continuing_task false, "
                         "reward_shift 0, goal_reward per_step")
    if len(pairs) == 1:
        recorded = (trainer["ext_reward_scale"], trainer["ext_reward_offset"])
        if per_cell or recorded != pairs[0]:
            raise ValueError(f"reward setting {reward_setting!r} needs ext_reward_scale and "
                             f"ext_reward_offset {pairs[0]}, not {recorded} or per-cell values")
        return
    # both settings in one program: every cell names its own scale and offset
    columns = [names.index(name) for name in REWARD_SETTING_NAMES] if per_cell else []
    recorded = {tuple(row[column] for column in columns)
                for row in trainer["parameter_settings"]}
    if not per_cell or recorded != set(pairs):
        raise ValueError(f"reward setting {reward_setting!r} needs cells that hold "
                         f"ext_reward_scale and ext_reward_offset with exactly the pairs {pairs}")


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
    check_reward_setting(trainer, environment, runtime["reward_setting"])

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
    if document["metadata"]["environment_family"] == "antmaze":
        defaults.update(terminate_at_goal=not environment["continuing_task"])
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
    names = tuple(document["trainer"]["setting_names"])
    if settings and "reward_setting" in explicit and set(REWARD_SETTING_NAMES) & set(names):
        raise ValueError("this file's cells hold the reward scale and offset; --reward-setting "
                         "cannot replace them, so edit the file's cells")
    if parsed.parameter_settings:
        settings = ()
        if "learning_rates" not in explicit:
            parsed.learning_rates = ""
        if "intrinsic_weights" not in explicit:
            parsed.intrinsic_weights = ""
    elif explicit & {"learning_rates", "intrinsic_weights"}:
        if settings:
            if names != DEFAULT_SETTING_NAMES:
                raise ValueError("this file's cells hold more than a rate and a weight; sweep "
                                 "lists on the command line cannot replace them, so edit the "
                                 "file's cells")
            if "learning_rates" not in explicit:
                parsed.learning_rates = ",".join(str(value) for value in dict.fromkeys(pair[0] for pair in settings))
            if "intrinsic_weights" not in explicit:
                parsed.intrinsic_weights = ",".join(str(value) for value in dict.fromkeys(pair[1] for pair in settings))
        settings = ()
    elif settings:
        # Before: cells ((0.001, 1.0), (0.001, 10.0)) and an explicit --learning-rate 0.0002;
        # after:  ((0.0002, 1.0), (0.0002, 10.0)); a column with no explicit option is kept.
        given = {name: getattr(parsed, argument) for name, argument in SETTING_ARGUMENTS.items()
                 if argument in explicit and hasattr(parsed, argument)}
        settings = tuple(dict.fromkeys(tuple(given.get(name, value) for name, value in zip(names, row))
                                      for row in settings))
    if not settings:
        if "learning_rate" in explicit and "learning_rates" not in explicit and parsed.learning_rates:
            parsed.learning_rates = str(parsed.learning_rate)
        if "intrinsic_coefficient" in explicit and "intrinsic_weights" not in explicit and parsed.intrinsic_weights:
            parsed.intrinsic_weights = str(parsed.intrinsic_coefficient)
    parsed.experiment_configuration_data = document
    parsed.experiment_parameter_settings = settings
    parsed.experiment_setting_names = names if settings else DEFAULT_SETTING_NAMES
    return parsed


def additional_trainer_fields(arguments):
    """Retain file knobs not exposed by the existing CLI or its sweep-construction logic."""
    document = getattr(arguments, "experiment_configuration_data", None)
    if document is None:
        return {}
    exposed = set(TRAINER_ARGUMENTS) | SPECIAL_TRAINER_FIELDS
    return {name: value for name, value in document["trainer"].items() if name not in exposed}
