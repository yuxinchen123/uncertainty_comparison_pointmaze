"""Turn a study description (JSON) into queue units for the PointMaze and AntMaze study.

A unit is one process on one card: one environment, one or both reward settings, a list of cells
(a cell is one value for each per-copy setting, such as learning rate, entropy coefficient and
intrinsic weight) and a range of seeds. Its policies are cells x reward settings x seeds.

    python build_units.py --run-dir RUN --study description.json [--dry-run]

Standard library only, so it runs on a login node. Per unit it writes
  RUN/specifications/<unit id>.json            the unit file: everything the worker needs
  RUN/slurm/unit_configurations/<unit id>.toml the configuration file run_training.py reads (the cells are in it)
  RUN/queue/pending/<unit id>.json             the queue record that moves between the queue folders
and it keeps a copy of the description in RUN/study_descriptions/. A unit that is already in any
queue folder is left as it is, so building the same description twice adds nothing.

A unit's configuration file lists every field of the trainer, so units are built against the code
copy that will run them: the run folder's frozen copy (freeze_code.py) when it exists. After a new
frozen copy, `--replace-pending` rebuilds the pending units that never started.

The description's keys, the three kinds of cells and one example are in README.md of this folder.
"""
import argparse
import ast
from datetime import datetime
from hashlib import sha256
from itertools import product
import json
from math import ceil
from pathlib import Path
import re
import tempfile
import tomllib
from zoneinfo import ZoneInfo

from grid_campaign import common, load_file
import unit_queue

ESTIMATES = common.PROJECT / "10_jax_exploration_platform/device/estimates"
# the five environments of the study (plan section 1.1): the map, its fixed start and goal cell
# (row, column), its step limit, and the name run_training.py selects the environment by
ENVIRONMENTS = {
    "pointmaze-medium": {"family": "pointmaze", "selector": "pointmaze-large", "map_name": "medium",
                         "start_cell": [6, 1], "goal_cell": [1, 6], "max_episode_steps": 400},
    "pointmaze-large": {"family": "pointmaze", "selector": "pointmaze-large", "map_name": "large",
                        "start_cell": [7, 1], "goal_cell": [1, 10], "max_episode_steps": 400},
    "antmaze-umaze": {"family": "antmaze", "selector": "antmaze-umaze", "map_name": "umaze",
                      "start_cell": [3, 1], "goal_cell": [1, 1], "max_episode_steps": 700},
    "antmaze-medium": {"family": "antmaze", "selector": "antmaze-medium", "map_name": "medium",
                       "start_cell": [6, 1], "goal_cell": [1, 6], "max_episode_steps": 1000},
    "antmaze-large": {"family": "antmaze", "selector": "antmaze-large", "map_name": "large",
                      "start_cell": [7, 1], "goal_cell": [1, 10], "max_episode_steps": 1000},
}
REWARD_SETTINGS = ("arrival_plus_one", "step_cost")
BOTH_REWARD_SETTINGS = "arrival_plus_one_and_step_cost"
# the words a description and a unit id use for a per-copy setting -> the trainer's own name
SETTING_NAMES = {"learning_rate": "learning_rate", "entropy_coefficient": "ent_coef",
                 "intrinsic_weight": "int_coef", "clip_coefficient": "clip_coef",
                 "value_coefficient": "vf_coef"}
SETTING_WORDS = {name: words for words, name in SETTING_NAMES.items()}
WORDS = re.compile(r"[a-z0-9][a-z0-9.-]*")
MEMORY_MARGIN = 1.25   # the copies-per-card skill's margin on the largest memory peak seen
# the interpreter a unit needs when its AntMaze physics is not MJX's JAX implementation (the
# `physics` value of its configuration file's [environment]); every other unit runs with the
# interpreter of the worker that starts it (platform_jax)
PHYSICS_INTERPRETERS = {"warp": "/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python"}


def per_copy_setting_names(platform):
    """Read the names of the settings the trainer takes per copy, from its source, without importing it."""
    # Importing the trainer would start JAX; the names are the keys of SETTING_RULES in training/sweep.py.
    module = ast.parse((platform / "src/exploration_platform/training/sweep.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "SETTING_RULES"
                                                for target in node.targets):
            return tuple(key.value for key in node.value.keys)
    raise ValueError("training/sweep.py no longer defines SETTING_RULES, which names the per-copy settings")


def first_pass_grid():
    """The plan's first-pass grid: 3 learning rates x 2 entropy coefficients x 10 intrinsic weights."""
    # the intrinsic weights go from 4^-6 to 4^3, a factor 4 between neighbours
    return {"learning_rate": [1e-4, 3e-4, 1e-3], "entropy_coefficient": [0.001, 0.01],
            "intrinsic_weight": [4.0 ** power for power in range(-6, 4)]}


def expand_cells(cells, allowed):
    """Turn a description's cells into the trainer's column names and one row of numbers per cell.

    before: {"kind": "grid", "values": {"learning_rate": [1e-4, 3e-4], "intrinsic_weight": [1, 4]}}
    after:  (("learning_rate", "int_coef"), [(0.0001, 1.0), (0.0001, 4.0), (0.0003, 1.0), (0.0003, 4.0)])
    before: {"kind": "list", "cells": [{"learning_rate": 3e-4, "entropy_coefficient": 0.01, "intrinsic_weight": 0.25}]}
    after:  (("learning_rate", "ent_coef", "int_coef"), [(0.0003, 0.01, 0.25)])
    """
    # every kind becomes a list of {words: value} rows; a grid is the cross product, first name slowest
    if cells["kind"] in ("first_pass_grid", "grid"):
        values = first_pass_grid() if cells["kind"] == "first_pass_grid" else cells["values"]
        words = list(values)
        rows = [dict(zip(words, combination)) for combination in product(*(values[word] for word in words))]
    elif cells["kind"] == "list":
        rows, words = cells["cells"], list(cells["cells"][0])
        if any(set(row) != set(words) for row in rows):
            raise ValueError("every cell of a list must name the same settings")
    else:
        raise ValueError(f"unknown kind of cells {cells['kind']!r}; choose first_pass_grid, grid or list")
    # the names must be per-copy settings of the trainer; the reward columns belong to reward_settings
    names = tuple(SETTING_NAMES.get(word, word) for word in words)
    refused = [name for name in names if name not in allowed or name.startswith("ext_reward_")]
    if refused or len(set(names)) != len(names):
        raise ValueError(f"cells may name each of {sorted(set(allowed))} once, without the reward "
                         f"scale and offset; got {list(names)}")
    numbers = [tuple(float(row[word]) for word in words) for row in rows]
    if len(set(numbers)) != len(numbers):
        raise ValueError("the cells hold the same cell twice; the trainer refuses duplicate cells")
    return names, numbers


def pack_units(cell_count, seed_count, policies_per_seed, policies_per_unit, half_size_units=0):
    """Cut cells x seeds into units of at most policies_per_unit policies; a cell's seeds stay together when they fit.

    policies_per_seed is 2 when both reward settings share one process, else 1. The first
    half_size_units units get half the size, so that a card class sees two sizes (step 4 of the
    copies-per-card skill). Returns (first cell, cell count, seed offset, seed count) per unit.

    before: 60 cells, 16 seeds, 2 policies per seed, 512 policies per unit
    after:  [(0, 16, 0, 16), (16, 16, 0, 16), (32, 16, 0, 16), (48, 12, 0, 16)]
    before: 1 cell, 1,024 seeds, 1 policy per seed, 512 policies per unit
    after:  [(0, 1, 0, 512), (0, 1, 512, 512)]   (the second unit starts at seed offset 512)
    """
    if policies_per_unit < policies_per_seed:
        raise ValueError(f"a unit of {policies_per_unit} policies cannot hold one seed ({policies_per_seed} policies)")
    units = []

    def room(index):
        """Policies the unit with this index may hold."""
        return policies_per_unit // 2 if index < half_size_units else policies_per_unit

    policies_per_cell = seed_count * policies_per_seed
    if policies_per_cell <= policies_per_unit:
        # whole cells: a unit holds every seed of each of its cells
        first = 0
        while first < cell_count:
            cells = min(max(1, room(len(units)) // policies_per_cell), cell_count - first)
            units.append((first, cells, 0, seed_count))
            first += cells
        return units
    # one cell is larger than a unit: each cell is cut along its seeds
    for cell in range(cell_count):
        offset = 0
        while offset < seed_count:
            seeds = min(max(1, room(len(units)) // policies_per_seed), seed_count - offset)
            units.append((cell, 1, offset, seeds))
            offset += seeds
    return units


def policies_per_unit(setting, workload, environments_per_policy, estimates_dir):
    """The most policies one unit may hold, where the number comes from, and the memory per policy if known."""
    # after: {"policies": 512, "source": "<workload>.json, card RTX A4500, status measured", "peak_memory_mb_per_copy": 3.3}
    if isinstance(setting, int):
        return {"policies": setting, "source": "the study description", "peak_memory_mb_per_copy": None}
    path = Path(estimates_dir) / f"{workload}.json"
    fallback = {"policies": setting["default"], "peak_memory_mb_per_copy": None}
    if not path.exists():
        return dict(fallback, source=f"the description's default (no estimate file {path.name})")
    estimate = common.read_json(path)
    if estimate["environments_per_copy"] != environments_per_policy:
        raise ValueError(f"{path} is for {estimate['environments_per_copy']} environments per policy, "
                         f"the sweep uses {environments_per_policy}")
    rows = {row["card"]: row for row in estimate["rows"]}
    if setting["card"] not in rows:
        return dict(fallback, source=f"the description's default ({path.name} has no row for {setting['card']})")
    # any card with enough memory may take the unit, so the memory per policy is the largest of all rows
    peaks = [row["peak_memory_mb_per_copy"] for row in estimate["rows"] if row.get("peak_memory_mb_per_copy")]
    row = rows[setting["card"]]
    return {"policies": row["copies"], "peak_memory_mb_per_copy": max(peaks) if peaks else None,
            "source": f"{path.name}, card {setting['card']}, status {row['status']}"}


def toml_value(value):
    """Write one Python value the way a TOML file holds it."""
    # before: [[0.0001, 0.001, 0.25]] ; after: "[[0.0001, 0.001, 0.25]]"
    # before: True ; after: "true"          before: "medium" ; after: '"medium"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"no TOML form for {value!r}")


def replace_values(text, changes):
    """Replace single-line `key = value` lines of a TOML text, section by section."""
    # before: "[trainer]\nn_envs = 4\nparameter_settings = []\n" with {"trainer": {"n_envs": 16}}
    # after:  "[trainer]\nn_envs = 16\nparameter_settings = []\n"
    wanted = {(section, key): value for section, keys in changes.items() for key, value in keys.items()}
    lines, section, found = text.splitlines(), None, set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            continue
        key = line.split("=", 1)[0].strip() if "=" in line and not stripped.startswith("#") else None
        if (section, key) in wanted:
            if (section, key) in found:
                raise ValueError(f"[{section}] holds {key} twice")
            lines[index] = f"{key} = {toml_value(wanted[(section, key)])}"
            found.add((section, key))
    if set(wanted) - found:
        raise ValueError(f"the configuration file has no line for {sorted(set(wanted) - found)}")
    return "\n".join(lines) + "\n"


def configuration_reader(platform):
    """The trainer's own reader of configuration files, taken from the code copy the units are built against."""
    # that copy's experiment_configuration.py checks a file against that copy's trainer fields
    name = "maze_experiment_configuration_" + sha256(str(platform).encode()).hexdigest()[:12]
    return load_file(name, platform / "scripts/experiment_configuration.py").read_experiment_configuration


def unit_configuration(base_text, changes, read_experiment_configuration):
    """Write a unit's configuration file from a base file, and check that only the named values changed."""
    text = replace_values(base_text, changes)
    # read both back: the new file must be the base file with exactly these values replaced
    expected = tomllib.loads(base_text)
    for section, keys in changes.items():
        expected[section].update(keys)
    if tomllib.loads(text) != expected:
        raise ValueError("the unit's configuration file did not come out as the base file plus the "
                         "changes; a changed key probably spans several lines in the base file")
    # the trainer's own reader must accept it: complete field sets, family and environment in agreement
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "unit_configuration.toml"
        path.write_text(text)
        read_experiment_configuration(path)
    return text


def pick(value, environment, family, field, default=None):
    """Take a description value given once for all, per family or per environment."""
    # before: {"pointmaze": 16, "antmaze-umaze": 4} for "antmaze-umaze" ; after: 4
    # before: 16 ; after: 16         before: None (the key is absent) ; after: the default
    if value is None:
        return default
    if isinstance(value, dict) and set(value) & (set(ENVIRONMENTS) | {"pointmaze", "antmaze"}):
        if environment not in value and family not in value:
            raise ValueError(f"{field} has no entry for {environment} or {family}")
        return value[environment] if environment in value else value[family]
    return value


def unit_id(phase, environment, reward_setting, method, names, cells, positions, first_seed, seed_count):
    """Spell a unit's content in words: phase, environment, reward, method, cells and seeds."""
    # one cell is spelled out; several cells are named by their places in the sweep's cell list
    # after: "confirmation_pointmaze-large_reward-step-cost_cleanrl-ppo-rnd_learning-rate-0.0003_
    #         entropy-coefficient-0.01_intrinsic-weight-0.25_seeds-30000-to-30511"
    # after: "development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_
    #         cells-01-to-16-of-60_seeds-0-to-15"
    if len(cells) == 1:
        cell_words = "_".join(f"{SETTING_WORDS.get(name, name).replace('_', '-')}-{value!r}"
                              for name, value in zip(names, cells[0]))
    else:
        width = len(str(positions["of"]))
        cell_words = f"cells-{positions['first']:0{width}d}-to-{positions['last']:0{width}d}-of-{positions['of']}"
    return (f"{phase}_{environment}_reward-{reward_setting.replace('_', '-')}_{method}_{cell_words}"
            f"_seeds-{first_seed}-to-{first_seed + seed_count - 1}")


def sweep_units(sweep, estimates_dir, platform):
    """Build the unit specifications of one sweep: its environments x reward settings x cells x seeds."""
    for key in ("phase", "method"):
        if not WORDS.fullmatch(sweep[key]):
            raise ValueError(f"{key} {sweep[key]!r} must be lower-case words joined by hyphens")
    rewards = list(sweep["reward_settings"])
    if not rewards or len(set(rewards)) != len(rewards) or set(rewards) - set(REWARD_SETTINGS):
        raise ValueError(f"reward_settings must name one or both of {list(REWARD_SETTINGS)}")
    # both reward settings in one process: every cell once per setting, on the same seeds
    together = len(rewards) == 2 and sweep.get("reward_settings_in_one_process", True)
    reward_groups = [BOTH_REWARD_SETTINGS] if together else rewards
    names, cells = expand_cells(sweep["cells"], per_copy_setting_names(platform))
    reader = configuration_reader(platform)
    marker = platform / "SOURCE_COMMIT"
    built_against = {"platform": str(platform),
                     "source_commit": marker.read_text().strip() if marker.exists() else "working tree"}
    specifications = []
    for environment in sweep["environments"]:
        facts = ENVIRONMENTS[environment]
        family = facts["family"]
        base_file = pick(sweep["configuration_files"], environment, family, "configuration_files")
        base_text = (platform / base_file).read_text()
        base = tomllib.loads(base_text)
        trainer_changes = dict(sweep.get("trainer_changes", {}))
        # the implementation of the AntMaze physics is the base file's; a file written before
        # the field existed, and every PointMaze file, means MJX's JAX implementation
        physics = base["environment"].get("physics", "jax")
        if physics != "jax" and physics not in PHYSICS_INTERPRETERS:
            raise ValueError(f"{base_file} names the physics {physics!r}, for which no interpreter is recorded")
        environments_per_policy = pick(sweep.get("environments_per_policy"), environment, family,
                                       "environments_per_policy", base["trainer"]["n_envs"])
        rollout_steps = trainer_changes.get("num_steps", base["trainer"]["num_steps"])
        update_style = trainer_changes.get("update_style", base["trainer"]["update_style"])

        # the training length: given in iterations, or in environment steps per policy (rounded up)
        if ("iterations" in sweep) == ("steps_per_policy" in sweep):
            raise ValueError("give exactly one of iterations and steps_per_policy")
        if "iterations" in sweep:
            iterations = pick(sweep["iterations"], environment, family, "iterations")
        else:
            steps = pick(sweep["steps_per_policy"], environment, family, "steps_per_policy")
            iterations = ceil(steps / (rollout_steps * environments_per_policy))
        # the logging windows: given as iterations per window, or as windows per unit
        if ("window_iterations" in sweep) == ("windows_per_unit" in sweep):
            raise ValueError("give exactly one of window_iterations and windows_per_unit")
        window_iterations = (pick(sweep["window_iterations"], environment, family, "window_iterations")
                             if "window_iterations" in sweep else
                             ceil(iterations / pick(sweep["windows_per_unit"], environment, family, "windows_per_unit")))

        # how many policies a unit may hold comes from the copies-per-card estimate of this workload
        # (another physics is another workload: its speed and memory per policy differ)
        workload = pick(sweep.get("workload"), environment, family, "workload",
                        f"{environment}_environments-per-policy-{environments_per_policy}"
                        f"_update-{update_style.replace('_', '-')}_bonus-{sweep['bonus'].replace('_', '-')}"
                        + ("" if physics == "jax" else f"_physics-{physics}"))
        limit = policies_per_unit(pick(sweep["policies_per_unit"], environment, family, "policies_per_unit"),
                                  workload, environments_per_policy, estimates_dir)
        seeds = pick(sweep["seeds"], environment, family, "seeds")
        stated_memory = pick(sweep.get("minimum_card_memory_mb"), environment, family, "minimum_card_memory_mb")
        for reward in reward_groups:
            packing = pack_units(len(cells), seeds["count"], 2 if together else 1, limit["policies"],
                                 sweep.get("half_size_units", 0))
            for first_cell, cell_count, seed_offset, seed_count in packing:
                unit_cells = [list(cell) for cell in cells[first_cell:first_cell + cell_count]]
                positions = {"first": first_cell + 1, "last": first_cell + cell_count, "of": len(cells)}
                first_seed = seeds["first"] + seed_offset
                identifier = unit_id(sweep["phase"], environment, reward, sweep["method"], names, unit_cells,
                                     positions, first_seed, seed_count)
                policies = cell_count * (2 if together else 1) * seed_count
                # the run seed keys the action noise and the minibatch order of the whole process;
                # units with different seed ranges get different ones, so policies in different
                # units never share those draws place by place (rng-seeding rule: a keyed hash)
                run_seed = sweep.get("run_seed", common.seed_for(
                    "maze_campaign", "run_seed", environment, first_seed, seed_count) % (2 ** 31 - 1))
                if stated_memory is None and limit["peak_memory_mb_per_copy"] is None:
                    raise ValueError(f"no memory estimate for workload {workload}: give minimum_card_memory_mb")
                memory = (int(stated_memory) if stated_memory is not None
                          else ceil(MEMORY_MARGIN * limit["peak_memory_mb_per_copy"] * policies))
                description = (f"{sweep['study']}, {sweep['phase']}: {sweep['method']} on {environment}, reward "
                               f"setting {reward.replace('_', ' ')}, cells {positions['first']} to "
                               f"{positions['last']} of {positions['of']}, seeds {first_seed} to "
                               f"{first_seed + seed_count - 1}")
                # everything of the unit goes into its own configuration file; the reward setting
                # alone is passed as an option, because it is a name and not a field of the file
                changes = {
                    "metadata": {"name": description},
                    "trainer": {**trainer_changes, "n_envs": environments_per_policy,
                                "setting_names": list(names), "parameter_settings": unit_cells,
                                "copies_per_group": seed_count, "n_copies": cell_count * seed_count,
                                "copy_seed_offset": first_seed},
                    "environment": {key: facts[key] for key in
                                    ("map_name", "start_cell", "goal_cell", "max_episode_steps")},
                    "runtime": {"environment": facts["selector"], "bonus": sweep["bonus"],
                                "iterations": iterations, "window_iterations": window_iterations,
                                "lr_anneal_iterations": iterations, "run_seed": run_seed},
                }
                header = (f"# Unit {identifier}\n# Written by scripts/maze_campaign/build_units.py from "
                          f"{base_file}; the changed values are listed in the unit's specification file.\n")
                specifications.append({
                    "unit_id": identifier, "study": sweep["study"], "phase": sweep["phase"],
                    "method": sweep["method"], "bonus": sweep["bonus"], "environment": environment,
                    "family": family, "reward_setting": reward, "workload": workload,
                    "physics": physics,
                    **({} if physics == "jax" else {"interpreter": PHYSICS_INTERPRETERS[physics]}),
                    "cell_setting_names": list(names), "cells": unit_cells, "cell_positions": positions,
                    "first_seed": first_seed, "seed_count": seed_count, "policies": policies,
                    "environments_per_policy": environments_per_policy, "rollout_steps": rollout_steps,
                    "iterations": iterations, "window_iterations": window_iterations,
                    "windows_expected": ceil(iterations / window_iterations),
                    "steps_per_policy": iterations * rollout_steps * environments_per_policy,
                    "run_seed": run_seed, "policies_per_unit": limit["policies"],
                    "policies_per_unit_source": limit["source"], "minimum_card_memory_mb": memory,
                    "base_configuration_file": str(base_file),
                    "base_configuration_sha256": sha256(base_text.encode()).hexdigest(),
                    "built_against": built_against,
                    "configuration_changes": changes,
                    "configuration_toml": header + unit_configuration(base_text, changes, reader),
                    "options": ["--reward-setting", reward, "--description", description,
                                *sweep.get("extra_options", [])],
                })
    return specifications


def study_units(description, estimates_dir=ESTIMATES, platform=common.PLATFORM):
    """Build the specifications of every sweep of a study description, in the order given."""
    # before: {"study": "s", "method": "m", "sweeps": [{"phase": "a", ...}, {"phase": "b", ...}]}
    # after:  the units of sweep a (with the top level's keys filled in), then those of sweep b
    shared = {key: value for key, value in description.items() if key != "sweeps"}
    sweeps = [dict(shared, **sweep) for sweep in description["sweeps"]] if "sweeps" in description else [shared]
    specifications = [unit for sweep in sweeps for unit in sweep_units(sweep, estimates_dir, platform)]
    names = [specification["unit_id"] for specification in specifications]
    if len(set(names)) != len(names):
        raise ValueError("two units of this description have the same id; give the sweeps different phases")
    # a unit's files are named by its id; 255 bytes is the file system's limit for one name
    if max(len(name) for name in names) > 200:
        raise ValueError(f"a unit id is longer than 200 characters: {max(names, key=len)}")
    return specifications


def queue_record(specification, order):
    """The small record of a unit that moves between the queue folders."""
    record = {key: specification[key] for key in (
        "phase", "environment", "reward_setting", "method", "workload", "policies", "iterations",
        "windows_expected")}
    return {"name": specification["unit_id"], "order": order,
            "minimum_device_memory_mb": specification["minimum_card_memory_mb"],
            "specification": f"specifications/{specification['unit_id']}.json", **record,
            "attempts": 0, "failures": 0, "recoveries": 0, "history": []}


def next_order(run_dir):
    """The first free place in the queue's order: one after the largest order any unit holds."""
    folders = [folder for folder in (run_dir / "queue").iterdir() if folder.is_dir()] if (run_dir / "queue").exists() else []
    orders = [common.read_json(path)["order"] for folder in folders for path in folder.glob("*.json")]
    return max(orders, default=0) + 1


def main(argv=None):
    """Read a study description, print its units, and queue them unless this is a dry run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True, help="the study description (JSON)")
    parser.add_argument("--estimates-dir", type=Path, default=ESTIMATES,
                        help="the copies-per-card estimates, one <workload>.json per workload")
    parser.add_argument("--platform", type=Path,
                        help="the platform folder whose configuration files and trainer source the units are "
                             "built against; default: the run folder's frozen code copy (RUN/source.json), "
                             "or this file's own tree when the run folder has none")
    parser.add_argument("--replace-pending", action="store_true",
                        help="give pending units that never started the new specification (after a new code copy)")
    parser.add_argument("--dry-run", action="store_true", help="print the units and write nothing")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    # a unit's configuration file lists the trainer's exact fields, so it is built against the
    # code copy that will run it: the run folder's frozen copy when there is one
    platform = args.platform or common.PLATFORM
    if not args.platform and (run_dir / "source.json").exists():
        platform = Path(common.read_json(run_dir / "source.json")["snapshot"]) / "10_jax_exploration_platform"
    print(f"units are built against {platform}")
    specifications = study_units(json.loads(args.study.read_text()), args.estimates_dir, platform)
    for specification in specifications:
        print(f"{specification['unit_id']}\n    {specification['policies']} policies ("
              f"{len(specification['cells'])} cells x {specification['seed_count']} seeds, reward setting "
              f"{specification['reward_setting']}), {specification['iterations']} iterations, "
              f"{specification['windows_expected']} windows, card memory at least "
              f"{specification['minimum_card_memory_mb']} MB, policies per unit from "
              f"{specification['policies_per_unit_source']}")
    if args.dry_run:
        print(f"dry run: {len(specifications)} units, nothing written")
        return
    # later builds queue behind earlier ones; a unit that already exists keeps its place and its state
    first = next_order(run_dir)
    added = [unit_queue.enqueue(run_dir, specification, queue_record(specification, first + index),
                                args.replace_pending)
             for index, specification in enumerate(specifications)]
    stamp = datetime.now().astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d-%H-%M")
    copy = run_dir / "study_descriptions" / f"{stamp}_{args.study.name}"
    copy.parent.mkdir(exist_ok=True)
    copy.write_text(args.study.read_text())
    print(f"{added.count('added')} units added to {run_dir / 'queue/pending'}, {added.count('replaced')} pending "
          f"units replaced, {added.count('')} left as they are; description copied to {copy}")


if __name__ == "__main__":
    main()
