"""Unit building of the maze campaign: counts, ids, seeds, the seed split, and the configuration files."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

from maze_test_support import ANTMAZE, ANTMAZE_WARP, PLATFORM, description, units_of
from grid_campaign import common
import build_units
import unit_queue


class PackUnits(unittest.TestCase):
    """How cells x seeds are cut into units."""

    def test_whole_cells_fill_units_and_the_last_unit_takes_the_rest(self):
        """60 cells x 16 seeds x 2 rewards at 512 policies: 16 cells per unit, the last unit 12."""
        self.assertEqual(build_units.pack_units(60, 16, 2, 512),
                         [(0, 16, 0, 16), (16, 16, 0, 16), (32, 16, 0, 16), (48, 12, 0, 16)])
        self.assertEqual(build_units.pack_units(60, 16, 2, 1024), [(0, 32, 0, 16), (32, 28, 0, 16)])
        self.assertEqual(build_units.pack_units(60, 16, 2, 2048), [(0, 60, 0, 16)])

    def test_a_cell_larger_than_a_unit_is_cut_along_its_seeds(self):
        """1 cell x 1,024 seeds at 512 policies: two units, the second at seed offset 512."""
        self.assertEqual(build_units.pack_units(1, 1024, 1, 512), [(0, 1, 0, 512), (0, 1, 512, 512)])
        # both rewards in one process: a seed costs two policies, and 1,000 seeds leave a smaller last unit
        self.assertEqual(build_units.pack_units(1, 1000, 2, 512),
                         [(0, 1, 0, 256), (0, 1, 256, 256), (0, 1, 512, 256), (0, 1, 768, 232)])

    def test_every_policy_is_in_exactly_one_unit(self):
        """Whatever the sizes, the units cover each (cell, seed) once."""
        for cells, seeds, per_seed, size, halves in ((60, 16, 2, 512, 0), (7, 256, 1, 100, 0), (5, 16, 2, 512, 2),
                                                     (3, 1024, 2, 512, 1), (1, 1, 1, 1, 0)):
            covered = [(cell, offset + seed) for first, count, offset, seed_count
                       in build_units.pack_units(cells, seeds, per_seed, size, halves)
                       for cell in range(first, first + count) for seed in range(seed_count)]
            self.assertEqual(sorted(covered), [(cell, seed) for cell in range(cells) for seed in range(seeds)])

    def test_half_size_units_come_first(self):
        """The first units get half the policies, so that a card class sees two sizes."""
        self.assertEqual(build_units.pack_units(60, 16, 2, 512, half_size_units=1)[:2], [(0, 8, 0, 16), (8, 16, 0, 16)])

    def test_a_unit_too_small_for_one_seed_is_refused(self):
        """Both rewards need two policies per seed."""
        with self.assertRaises(ValueError):
            build_units.pack_units(1, 4, 2, 1)


class Cells(unittest.TestCase):
    """How a description's cells become the trainer's columns."""

    ALLOWED = build_units.per_copy_setting_names(PLATFORM)

    def test_the_trainer_names_its_per_copy_settings(self):
        """The names are read from the trainer's source without importing it."""
        self.assertLessEqual({"learning_rate", "int_coef", "ent_coef", "clip_coef", "vf_coef"}, set(self.ALLOWED))

    def test_first_pass_grid_is_the_plans(self):
        """3 learning rates x 2 entropy coefficients x 10 intrinsic weights from 4^-6 to 4^3."""
        names, cells = build_units.expand_cells({"kind": "first_pass_grid"}, self.ALLOWED)
        self.assertEqual(names, ("learning_rate", "ent_coef", "int_coef"))
        self.assertEqual(len(cells), 60)
        self.assertEqual(sorted({cell[2] for cell in cells}), [4.0 ** power for power in range(-6, 4)])
        self.assertEqual(cells[0], (1e-4, 0.001, 4.0 ** -6))

    def test_grid_and_list(self):
        """A grid is a cross product, first name slowest; a list is taken as written, by name."""
        names, cells = build_units.expand_cells(
            {"kind": "grid", "values": {"learning_rate": [1e-4, 3e-4], "clip_coefficient": [0.1, 0.2]}}, self.ALLOWED)
        self.assertEqual((names, cells), (("learning_rate", "clip_coef"),
                                          [(1e-4, 0.1), (1e-4, 0.2), (3e-4, 0.1), (3e-4, 0.2)]))
        names, cells = build_units.expand_cells({"kind": "list", "cells": [
            {"learning_rate": 3e-4, "intrinsic_weight": 1}, {"intrinsic_weight": 4, "learning_rate": 1e-4}]}, self.ALLOWED)
        self.assertEqual((names, cells), (("learning_rate", "int_coef"), [(3e-4, 1.0), (1e-4, 4.0)]))

    def test_refusals(self):
        """Unknown names, the reward columns, duplicate cells and unequal lists are refused."""
        for cells in ({"kind": "grid", "values": {"momentum": [0.9]}},
                      {"kind": "grid", "values": {"ext_reward_scale": [1.0]}},
                      {"kind": "list", "cells": [{"learning_rate": 1e-4}, {"learning_rate": 1e-4}]},
                      {"kind": "list", "cells": [{"learning_rate": 1e-4}, {"intrinsic_weight": 1}]},
                      {"kind": "other"}):
            with self.assertRaises(ValueError):
                build_units.expand_cells(cells, self.ALLOWED)


class DevelopmentSweep(unittest.TestCase):
    """The plan's first-pass sweep on PointMaze Large in both reward settings."""

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.estimates = Path(self.folder.name) / "estimates"

    def test_units_at_three_sizes(self):
        """4, 2 and 1 units at 512, 1,024 and 2,048 policies per unit; every cell and seed once per reward."""
        for size, expected in ((512, 4), (1024, 2), (2048, 1)):
            units = units_of(description(policies_per_unit=size), self.estimates)
            self.assertEqual(len(units), expected)
            self.assertEqual(sum(unit["policies"] for unit in units), 60 * 16 * 2)
            self.assertTrue(all(unit["policies"] <= size for unit in units))
            cells = [tuple(cell) for unit in units for cell in unit["cells"]]
            self.assertEqual(len(set(cells)), 60)
            self.assertTrue(all((unit["first_seed"], unit["seed_count"]) == (0, 16) for unit in units))

    def test_reward_settings_in_separate_processes(self):
        """One unit list per reward setting when they do not share a process."""
        units = units_of(description(reward_settings_in_one_process=False), self.estimates)
        self.assertEqual([unit["reward_setting"] for unit in units], ["arrival_plus_one"] * 2 + ["step_cost"] * 2)
        self.assertEqual([unit["policies"] for unit in units], [512, 448, 512, 448])

    def test_ids_spell_the_content(self):
        """Phase, environment, reward, method, the cells' places and the seeds are in the id."""
        units = units_of(description(), self.estimates)
        self.assertEqual(units[0]["unit_id"],
                         "development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_"
                         "cleanrl-ppo-rnd_cells-01-to-16-of-60_seeds-0-to-15")
        self.assertEqual(units[3]["unit_id"].split("_")[4], "cells-49-to-60-of-60")
        self.assertEqual(len({unit["unit_id"] for unit in units}), 4)

    def test_the_unit_file_holds_what_the_trainer_needs(self):
        """Iterations, windows, the reward option, and a configuration file with the unit's cells."""
        unit = units_of(description(), self.estimates)[1]
        self.assertEqual((unit["iterations"], unit["window_iterations"], unit["windows_expected"]), (98304, 492, 200))
        self.assertEqual(unit["steps_per_policy"], 50331648)
        self.assertEqual(unit["options"][:2], ["--reward-setting", "arrival_plus_one_and_step_cost"])
        self.assertEqual(unit["workload"],
                         "pointmaze-large_environments-per-policy-4_update-epoch-minibatch_bonus-rnd-next-state")
        self.assertEqual(unit["policies_per_unit_source"].split(" (")[0], "the description's default")
        document = tomllib.loads(unit["configuration_toml"])
        self.assertEqual(document["trainer"]["setting_names"], ["learning_rate", "ent_coef", "int_coef"])
        self.assertEqual(document["trainer"]["parameter_settings"], unit["cells"])
        self.assertEqual((document["trainer"]["copies_per_group"], document["trainer"]["copy_seed_offset"]), (16, 0))
        self.assertEqual(document["runtime"]["iterations"], document["runtime"]["lr_anneal_iterations"])
        self.assertEqual(document["runtime"]["run_seed"], unit["run_seed"])
        # everything else is the base file's
        base = tomllib.loads((PLATFORM / unit["base_configuration_file"]).read_text())
        self.assertEqual(document["trainer"]["clip_coef"], base["trainer"]["clip_coef"])
        self.assertEqual(document["environment"], base["environment"])

    def test_trainer_changes_and_environments_per_policy(self):
        """Other trainer values and the environments per policy go into the unit's file and its length."""
        unit = units_of(description(environments_per_policy={"pointmaze": 16},
                                    trainer_changes={"gather_minibatches_per_epoch": True}), self.estimates)[0]
        document = tomllib.loads(unit["configuration_toml"])
        self.assertEqual((document["trainer"]["n_envs"], document["trainer"]["gather_minibatches_per_epoch"]), (16, True))
        self.assertEqual(unit["iterations"], 50331648 // (128 * 16))
        self.assertIn("environments-per-policy-16", unit["workload"])


class ConfirmationAndOtherMaps(unittest.TestCase):
    """One chosen cell on many seeds from 30000, and the maps built from another map's file."""

    CELL = {"learning_rate": 3e-4, "entropy_coefficient": 0.01, "intrinsic_weight": 0.25}

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.estimates = Path(self.folder.name) / "estimates"

    def confirmation(self, **changes):
        """A confirmation of one cell in one reward setting."""
        return description(**{"phase": "confirmation", "reward_settings": ["step_cost"],
                              "cells": {"kind": "list", "cells": [self.CELL]},
                              "seeds": {"first": 30000, "count": 1024}, **changes})

    def test_a_split_cell_uses_seed_offsets(self):
        """Two units of 512 seeds; the second starts at seed 30512 and has its own run seed."""
        first, second = units_of(self.confirmation(), self.estimates)
        self.assertEqual([(unit["first_seed"], unit["seed_count"], unit["policies"]) for unit in (first, second)],
                         [(30000, 512, 512), (30512, 512, 512)])
        self.assertEqual(tomllib.loads(second["configuration_toml"])["trainer"]["copy_seed_offset"], 30512)
        self.assertEqual(first["unit_id"], "confirmation_pointmaze-large_reward-step-cost_cleanrl-ppo-rnd_"
                         "learning-rate-0.0003_entropy-coefficient-0.01_intrinsic-weight-0.25_seeds-30000-to-30511")
        self.assertNotEqual(first["run_seed"], second["run_seed"])

    def test_the_run_seed_is_keyed_by_environment_and_seeds_only(self):
        """Another method on the same seeds gets the same run seed; a fixed one can be given."""
        other = units_of(self.confirmation(method="another-method"), self.estimates)
        mine = units_of(self.confirmation(), self.estimates)
        self.assertEqual([unit["run_seed"] for unit in other], [unit["run_seed"] for unit in mine])
        self.assertEqual({unit["run_seed"] for unit in units_of(self.confirmation(run_seed=7), self.estimates)}, {7})
        self.assertTrue(all(0 <= unit["run_seed"] < 2 ** 31 for unit in mine))

    def test_other_maps_come_from_the_family_file(self):
        """PointMaze Medium and AntMaze Medium take the plan's start, goal and step limit."""
        study = self.confirmation(environments=["pointmaze-medium", "antmaze-medium"],
                                  seeds={"pointmaze": {"first": 30000, "count": 1024},
                                         "antmaze": {"first": 30000, "count": 256}})
        units = units_of(study, self.estimates)
        self.assertEqual([unit["environment"] for unit in units], ["pointmaze-medium"] * 2 + ["antmaze-medium"])
        point, ant = tomllib.loads(units[0]["configuration_toml"]), tomllib.loads(units[2]["configuration_toml"])
        self.assertEqual((point["environment"]["map_name"], point["environment"]["start_cell"],
                          point["environment"]["goal_cell"], point["environment"]["max_episode_steps"],
                          point["runtime"]["environment"]), ("medium", [6, 1], [1, 6], 400, "pointmaze-large"))
        self.assertEqual((ant["environment"]["map_name"], ant["environment"]["max_episode_steps"],
                          ant["runtime"]["environment"]), ("medium", 1000, "antmaze-medium"))
        self.assertEqual(units[2]["base_configuration_file"], ANTMAZE)
        self.assertEqual(units[2]["seed_count"], 256)

    def test_a_warp_physics_unit_names_its_interpreter_and_its_own_workload(self):
        """A unit built from a file with physics "warp" names the platform_jax_warp interpreter; others name none."""
        import sys
        from grid_campaign import own
        study = self.confirmation(environments=["antmaze-umaze", "pointmaze-large"],
                                  seeds={"first": 30000, "count": 256})
        jax_ant, point = units_of(study, self.estimates)
        warp_ant, _ = units_of(dict(study, configuration_files=dict(study["configuration_files"],
                                                                    antmaze=ANTMAZE_WARP)), self.estimates)
        self.assertEqual((warp_ant["physics"], warp_ant["interpreter"]),
                         ("warp", "/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python"))
        self.assertEqual(tomllib.loads(warp_ant["configuration_toml"])["environment"]["physics"], "warp")
        self.assertEqual(warp_ant["workload"], jax_ant["workload"] + "_physics-warp")
        for unit in (jax_ant, point):
            self.assertEqual(unit["physics"], "jax")
            self.assertNotIn("interpreter", unit)
        # the worker starts the unit with the named interpreter when there is one, else with its own
        command = lambda unit: own("worker").unit_command(Path("/run"), unit, Path("/unit.toml"), 3000, "run_training.py")
        self.assertEqual(command(jax_ant)[0], sys.executable)
        self.assertEqual(command(dict(warp_ant, interpreter=__file__))[:2], [__file__, "run_training.py"])
        with self.assertRaisesRegex(FileNotFoundError, "needs the interpreter /no/such/python"):
            command(dict(warp_ant, interpreter="/no/such/python"))
        # on a remote server the named path is UVA's: the worker's own interpreter runs the unit there
        self.assertEqual(own("worker").unit_interpreter(dict(warp_ant, interpreter="/no/such/python"), remote=True),
                         sys.executable)

    def test_several_sweeps_in_one_description(self):
        """A description may list sweeps that share its top-level keys; equal ids are refused."""
        study = dict(self.confirmation(), sweeps=[{"reward_settings": ["arrival_plus_one"]},
                                                 {"reward_settings": ["step_cost"]}])
        self.assertEqual([unit["reward_setting"] for unit in units_of(study, self.estimates)],
                         ["arrival_plus_one"] * 2 + ["step_cost"] * 2)
        with self.assertRaises(ValueError):
            units_of(dict(self.confirmation(), sweeps=[{}, {}]), self.estimates)


class Estimates(unittest.TestCase):
    """The policies per unit and the card memory come from the copies-per-card estimate when one exists."""

    WORKLOAD = "pointmaze-large_environments-per-policy-4_update-epoch-minibatch_bonus-rnd-next-state"

    def setUp(self):
        """Make the temporary folder the tests of this class work in."""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.estimates = Path(self.folder.name)
        common.write_json(self.estimates / f"{self.WORKLOAD}.json", {
            "workload": self.WORKLOAD, "environments_per_copy": 4, "rows": [
                {"card": "RTX A4500", "status": "measured", "copies": 1024, "peak_memory_mb_per_copy": 3.0},
                {"card": "GTX 1080", "status": "interpolated", "copies": 256, "peak_memory_mb_per_copy": 4.0}]})

    def test_the_cards_row_sets_the_size_and_the_memory(self):
        """1,024 policies per unit from the row; memory = 1.25 x the largest peak per policy x the unit's policies."""
        study = description()
        del study["minimum_card_memory_mb"]
        units = units_of(study, self.estimates)
        self.assertEqual([unit["policies"] for unit in units], [1024, 896])
        self.assertEqual(units[0]["policies_per_unit_source"], f"{self.WORKLOAD}.json, card RTX A4500, status measured")
        self.assertEqual([unit["minimum_card_memory_mb"] for unit in units], [5120, 4480])

    def test_default_without_a_row_and_refusals(self):
        """A card without a row takes the default; no memory figure at all, or another shape, is refused."""
        study = description(policies_per_unit={"card": "H200", "default": 2048})
        self.assertEqual(len(units_of(study, self.estimates)), 1)
        del study["minimum_card_memory_mb"]
        with self.assertRaises(ValueError):
            units_of(study, self.estimates)
        with self.assertRaises(ValueError):
            units_of(description(environments_per_policy=16, workload=self.WORKLOAD), self.estimates)


class ConfigurationText(unittest.TestCase):
    """The line replacement that writes a unit's configuration file."""

    def test_values_and_replacement(self):
        """Booleans, numbers, strings and nested lists; one line per key, within its section."""
        self.assertEqual(build_units.toml_value([[1e-4, 0.001], [True, "medium"]]), '[[0.0001, 0.001], [true, "medium"]]')
        text = "[trainer]\nn_envs = 4\n# n_envs = 9\n[runtime]\nn_envs = 1\n"
        self.assertEqual(build_units.replace_values(text, {"trainer": {"n_envs": 16}}),
                         "[trainer]\nn_envs = 16\n# n_envs = 9\n[runtime]\nn_envs = 1\n")
        with self.assertRaises(ValueError):
            build_units.replace_values(text, {"trainer": {"missing_key": 1}})


class TrainerCommandLine(unittest.TestCase):
    """The standard-library half of the trainer's command parsing accepts a unit's file and options."""

    def test_the_units_cells_reach_the_trainer_as_named_cells(self):
        """run_training.py's reader of configuration files hands on the unit's cells, seeds, length and run seed."""
        import argparse
        from experiment_configuration import parse_with_configuration_file
        from grid_campaign import own
        with tempfile.TemporaryDirectory() as folder:
            unit = units_of(description(environments=["pointmaze-medium"]), Path(folder) / "estimates")[1]
            path = Path(folder) / "unit.toml"
            path.write_text(unit["configuration_toml"])
            command = own("worker").unit_command(Path(folder), unit, path, 3000, "run_training.py")[2:]
            # the options the worker passes, declared as run_training.py declares them
            parser = argparse.ArgumentParser()
            for option in ("--run-dir", "--unit-id", "--description", "--reward-setting", "--env",
                           "--parameter-settings"):
                parser.add_argument(option)
            parser.add_argument("--configuration-file", type=Path)
            parser.add_argument("--checkpoint-seconds", type=float)
            parser.add_argument("--time-limit-seconds", type=float)
            parser.add_argument("--no-aggregate", action="store_true")
            parsed = parse_with_configuration_file(parser, command)
        self.assertEqual(parsed.experiment_setting_names, ("learning_rate", "ent_coef", "int_coef"))
        self.assertEqual([list(cell) for cell in parsed.experiment_parameter_settings], unit["cells"])
        self.assertEqual((parsed.copies_per_cell, parsed.copy_seed_offset, parsed.iterations,
                          parsed.lr_anneal_iterations, parsed.window_iterations, parsed.run_seed, parsed.bonus),
                         (16, 0, unit["iterations"], unit["iterations"], unit["window_iterations"],
                          unit["run_seed"], "rnd_next_state"))
        self.assertEqual((parsed.env, parsed.map, parsed.start_cell, parsed.goal_cell, parsed.episode_steps),
                         ("pointmaze-large", "medium", "6,1", "1,6", 400))
        self.assertEqual((parsed.reward_setting, parsed.checkpoint_seconds, parsed.time_limit_seconds,
                          parsed.no_aggregate), ("arrival_plus_one_and_step_cost", 1200.0, 3000.0, True))


class QueueFiles(unittest.TestCase):
    """What the builder writes into a run folder."""

    def test_main_writes_units_once(self):
        """Specification, configuration file and pending record per unit; a second build adds nothing."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, study = Path(folder) / "run", Path(folder) / "development.json"
            study.write_text(json.dumps(description()))
            arguments = ["--run-dir", str(run_dir), "--study", str(study), "--estimates-dir", folder]
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                build_units.main(arguments + ["--dry-run"])
            self.assertFalse(run_dir.exists())
            self.assertIn("dry run: 4 units", printed.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                build_units.main(arguments)
                build_units.main(arguments)
            self.assertEqual(unit_queue.counts(run_dir), {"done": 0, "failed": 0, "pending": 4, "running": 0})
            records = unit_queue.read_units(run_dir, "pending")
            self.assertEqual(sorted(record["order"] for record in records), [1, 2, 3, 4])
            for record in records:
                specification = common.read_json(run_dir / record["specification"])
                self.assertEqual(unit_queue.configuration_path(run_dir, record["name"]).read_text(),
                                 specification["configuration_toml"])
                self.assertEqual((record["minimum_device_memory_mb"], record["windows_expected"]), (7000, 200))
            self.assertTrue(list((run_dir / "study_descriptions").glob("*_development.json")))
            self.assertEqual(build_units.next_order(run_dir), 5)

    def test_units_are_built_against_the_frozen_copy_and_pending_units_can_be_replaced(self):
        """With RUN/source.json the frozen copy's files are read; only units that never started are replaced."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, study = Path(folder) / "run", Path(folder) / "development.json"
            # a stand-in for a frozen copy: a folder whose platform is the working tree
            (Path(folder) / "copy").mkdir()
            (Path(folder) / "copy/10_jax_exploration_platform").symlink_to(PLATFORM)
            run_dir.mkdir()
            common.write_json(run_dir / "source.json", {"commit": "abc", "snapshot": str(Path(folder) / "copy")})
            arguments = ["--run-dir", str(run_dir), "--study", str(study), "--estimates-dir", folder]
            study.write_text(json.dumps(description()))
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                build_units.main(arguments)
            self.assertIn(f"units are built against {Path(folder) / 'copy/10_jax_exploration_platform'}",
                          printed.getvalue())
            first, second = sorted(record["name"] for record in unit_queue.read_units(run_dir, "pending"))[:2]
            self.assertEqual(common.read_json(run_dir / "specifications" / f"{first}.json")["built_against"],
                             {"platform": str(Path(folder) / "copy/10_jax_exploration_platform"),
                              "source_commit": "working tree"})
            # the first unit starts; then the description changes its windows
            started = unit_queue.claim(run_dir, 20470, {"job_id": "local"})
            self.assertEqual(started["name"], first)
            study.write_text(json.dumps(description(windows_per_unit=100)))
            with contextlib.redirect_stdout(io.StringIO()):
                build_units.main(arguments)
            self.assertEqual(common.read_json(run_dir / "specifications" / f"{second}.json")["windows_expected"], 200)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                build_units.main(arguments + ["--replace-pending"])
            self.assertIn("0 units added", printed.getvalue())
            self.assertIn("3 pending units replaced, 1 left as they are", printed.getvalue())
            windows = {name: common.read_json(run_dir / "specifications" / f"{name}.json")["windows_expected"]
                       for name in (first, second)}
            self.assertEqual(windows, {first: 200, second: 100})
            record = common.read_json(unit_queue.unit_path(run_dir, "pending", second))
            self.assertEqual((record["order"], record["windows_expected"]), (2, 100))
            self.assertIn("window_iterations = 984", unit_queue.configuration_path(run_dir, second).read_text())


class CodeCopyNumber(unittest.TestCase):
    """The lowest code copy number a unit's queue record names."""

    def test_the_record_names_the_copy_the_unit_was_built_against(self):
        """queue_record writes the number it is given (0 by default); main reads it from the platform it builds against."""
        with tempfile.TemporaryDirectory() as folder:
            run_dir, study = Path(folder) / "run", Path(folder) / "development.json"
            study.write_text(json.dumps(description()))
            arguments = ["--run-dir", str(run_dir), "--study", str(study), "--estimates-dir", folder]
            # the working tree has no number file: its units need no particular copy
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                build_units.main(arguments)
            self.assertIn("(code copy number 0)", printed.getvalue())
            self.assertEqual({record["minimum_copy_number"] for record in unit_queue.read_units(run_dir, "pending")}, {0})
            # after a freeze the copy holds number 4: units that never started are rebuilt for it
            with mock.patch.object(unit_queue, "copy_number", return_value=4) as read, \
                    contextlib.redirect_stdout(io.StringIO()) as printed:
                build_units.main(arguments + ["--replace-pending"])
            read.assert_called_once_with(build_units.common.PLATFORM)
            self.assertIn("(code copy number 4)", printed.getvalue())
            self.assertEqual({record["minimum_copy_number"] for record in unit_queue.read_units(run_dir, "pending")}, {4})
        specification = {"unit_id": "u", "minimum_card_memory_mb": 7000, **{key: 0 for key in (
            "phase", "environment", "reward_setting", "method", "workload", "policies", "iterations",
            "windows_expected")}}
        self.assertEqual((build_units.queue_record(specification, 1)["minimum_copy_number"],
                          build_units.queue_record(specification, 1, 5)["minimum_copy_number"]), (0, 5))


if __name__ == "__main__":
    unittest.main()
