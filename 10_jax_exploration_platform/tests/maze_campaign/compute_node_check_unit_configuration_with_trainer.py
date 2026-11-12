"""Compute-node check: the trainer's own command reading accepts units built by scripts/maze_campaign/build_units.py.

This file imports scripts/run_training.py and the trainer's sweep module, which import JAX, so it
runs on a compute node only, through the test job (it is not among the `test_maze_*.py` files,
which import no numerical library):

  bash <benchmark run folder>/infrastructure/submit_tests.sh \
      tests/maze_campaign/compute_node_check_unit_configuration_with_trainer.py

No training is started: each test reads a unit's command the way run_training.py does and looks
at the trainer configuration, the environment configuration and the per-copy seeds that come out.
"""
from pathlib import Path
import sys
import tempfile

from maze_test_support import PLATFORM, description, units_of
from grid_campaign import own

sys.path.insert(0, str(PLATFORM / "src"))
import run_training  # noqa: E402
from exploration_platform.training.sweep import build_sweep  # noqa: E402

worker = own("worker")
CELL = {"learning_rate": 3e-4, "entropy_coefficient": 0.01, "intrinsic_weight": 0.25}


def built(unit, folder):
    """Read one unit's worker command as run_training.py does; returns (arguments, trainer and environment configuration)."""
    path = Path(folder) / f"{unit['unit_id']}.toml"
    path.write_text(unit["configuration_toml"])
    command = worker.unit_command(Path(folder), unit, path, 3000, PLATFORM / "scripts/run_training.py")
    arguments = run_training.parse_args(command[2:])
    return arguments, run_training.build_config(arguments), run_training.build_env_config(arguments)


def confirmation(**changes):
    """A confirmation of one cell on 1,024 seeds from 30000 in the step-cost reward setting."""
    return description(**{"phase": "confirmation", "reward_settings": ["step_cost"],
                          "cells": {"kind": "list", "cells": [CELL]}, "seeds": {"first": 30000, "count": 1024},
                          **changes})


def test_development_unit_with_both_reward_settings():
    """Every cell of the unit once per reward setting, on seeds 0 to 15, with the unit's length and run seed."""
    with tempfile.TemporaryDirectory() as folder:
        unit = units_of(description(), Path(folder) / "estimates")[3]
        arguments, config, environment = built(unit, folder)
    assert config.n_copies == unit["policies"] == 12 * 2 * 16
    assert config.setting_names == ("learning_rate", "ent_coef", "int_coef", "ext_reward_scale", "ext_reward_offset")
    assert sorted({row[:3] for row in config.parameter_settings}) == sorted(tuple(cell) for cell in unit["cells"])
    assert {row[3:] for row in config.parameter_settings} == {(1.0, 0.0), (1 / 400, -1 / 400)}
    assert len(config.parameter_settings) == 24 and config.copies_per_group == 16
    assert (config.update_style, config.clip_coef, config.prime_iterations, config.episode_log) == (
        "epoch_minibatch", 0.1, 50, True)
    assert (arguments.iterations, arguments.lr_anneal_iterations, arguments.window_iterations) == (
        unit["iterations"], unit["iterations"], unit["window_iterations"])
    assert (arguments.run_seed, arguments.bonus, arguments.no_aggregate) == (unit["run_seed"], "rnd_next_state", True)
    assert (arguments.checkpoint_seconds, arguments.time_limit_seconds) == (1200.0, 3000.0)
    assert not environment.continuing_task and environment.max_episode_steps == 400
    sweep = build_sweep(config)
    assert sorted(set(sweep.copy_seed_index)) == list(range(16)) and len(sweep.copy_seed_index) == 384
    assert run_training.env_spec(environment, arguments.reward_setting).endswith(
        "_reward-arrival-plus-one-and-step-cost@1")


def test_second_unit_of_a_split_cell_holds_its_own_seeds():
    """The second unit of a cell cut along its seeds trains seeds 30512 to 31023, in the step-cost setting."""
    with tempfile.TemporaryDirectory() as folder:
        first, second = units_of(confirmation(), Path(folder) / "estimates")
        arguments, config, _ = built(second, folder)
    assert config.n_copies == 512 and config.copy_seed_offset == 30512
    assert (config.ext_reward_scale, config.ext_reward_offset) == (1 / 400, -1 / 400)
    assert config.setting_names == ("learning_rate", "ent_coef", "int_coef")
    assert config.parameter_settings == ((3e-4, 0.01, 0.25),)
    assert list(build_sweep(config).copy_seed_index) == list(range(30512, 31024))
    assert arguments.run_seed == second["run_seed"] != first["run_seed"]


def test_other_maps_reach_the_environment():
    """PointMaze Medium and the three AntMaze maps come out with the plan's start, goal and step limit."""
    expected = {"pointmaze-medium": ("medium", (6, 1), (1, 6), 400), "antmaze-umaze": ("umaze", (3, 1), (1, 1), 700),
                "antmaze-medium": ("medium", (6, 1), (1, 6), 1000), "antmaze-large": ("large", (7, 1), (1, 10), 1000)}
    with tempfile.TemporaryDirectory() as folder:
        units = units_of(confirmation(environments=list(expected), seeds={"first": 30000, "count": 256}),
                         Path(folder) / "estimates")
        for unit in units:
            arguments, config, environment = built(unit, folder)
            assert (environment.map_name, tuple(environment.start_cell), tuple(environment.goal_cell),
                    environment.max_episode_steps) == expected[unit["environment"]]
            assert not environment.continuing_task
            assert config.n_copies == 256
            assert config.configuration_preset == ("antmaze_current" if unit["family"] == "antmaze"
                                                   else "pointmaze_current")
            assert (config.ext_reward_scale, config.ext_reward_offset) == (
                1 / environment.max_episode_steps, -1 / environment.max_episode_steps)


def test_environments_per_policy_and_trainer_changes():
    """16 environments per policy with minibatches gathered per epoch reach the trainer configuration."""
    with tempfile.TemporaryDirectory() as folder:
        unit = units_of(description(environments_per_policy=16, policies_per_unit=2048,
                                    trainer_changes={"gather_minibatches_per_epoch": True}),
                        Path(folder) / "estimates")[0]
        arguments, config, _ = built(unit, folder)
    assert (config.n_envs, config.gather_minibatches_per_epoch, config.n_copies) == (16, True, 1920)
    assert arguments.iterations * config.num_steps * config.n_envs == unit["steps_per_policy"]
