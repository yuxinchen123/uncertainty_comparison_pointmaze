"""Check recorded experiment files and CLI merging using the standard library only."""

import argparse
import ast
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_configuration import (additional_trainer_fields, argument_defaults,
                                      parse_with_configuration_file, read_experiment_configuration,
                                      reward_mappings)

CONFIGURATIONS = ROOT / "configs/experiments"
CURRENT = CONFIGURATIONS / "pointmaze_large_current_returns_random_network_distillation_steps-100-million.toml"
HISTORICAL = CONFIGURATIONS / "pointmaze_large_historical_random_network_distillation_steps-100-million.toml"
PART_A = CONFIGURATIONS / "pointmaze_large_entropy-0.01_learning-rate-0.001_within-run-anneal_steps-100-million.toml"
PART_B = CONFIGURATIONS / "pointmaze_large_entropy-0.01_learning-rate-0.0001_within-run-anneal_steps-400-million.toml"
# The study that started 2026-11-11 adds one example per family; they record no earlier run.
POINTMAZE_EXAMPLE = CONFIGURATIONS / "pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml"
ANTMAZE_EXAMPLE = CONFIGURATIONS / "antmaze_umaze_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml"
# The same AntMaze example with the physics stepped by MuJoCo Warp.
ANTMAZE_WARP_EXAMPLE = CONFIGURATIONS / "antmaze_umaze_warp-physics_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml"
# The Warp example with the physics the AntMaze baseline runs with (time step 0.005 s, 10 physics
# steps per environment step, the fault guard at 30 m/s and 100 rad/s).
ANTMAZE_STABLE_PHYSICS_EXAMPLE = CONFIGURATIONS / (
    "antmaze_umaze_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_"
    "cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml")
# The same file on the Medium and Large maps (added 2026-11-13): the map, its start and goal cells, its
# step limit and the runtime selector differ from the U-maze file, and nothing else.
ANTMAZE_STABLE_PHYSICS_MAPS = {
    name: CONFIGURATIONS / (
        f"antmaze_{name}_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_"
        "cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml")
    for name in ("medium", "large")}
STABLE_PHYSICS_FILES = (ANTMAZE_STABLE_PHYSICS_EXAMPLE, *ANTMAZE_STABLE_PHYSICS_MAPS.values())
# The physics of the AntMaze work from 2026-11-13 (user decision): each stable-physics file with the RK4
# integrator at a time step of 0.01 s and 5 physics steps per environment step, and nothing else changed.
RK4_PHYSICS_MAPS = {
    name: CONFIGURATIONS / (
        f"antmaze_{name}_warp-physics_rk4_time-step-0.01_frame-skip-5_fault-guard-30-metres-and-100-radians-per-second_"
        "cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml")
    for name in ("umaze", "medium", "large")}
EXAMPLES = (POINTMAZE_EXAMPLE, ANTMAZE_EXAMPLE, ANTMAZE_WARP_EXAMPLE, *STABLE_PHYSICS_FILES, *RK4_PHYSICS_MAPS.values())
# The number of trainer fields every recorded file must hold (the fields of PPOConfig).
FIELDS = 58


def antmaze_presets():
    """The maps' presets of the environment code, read from am_common.py without importing it.

    before: presets = {"umaze": AntMazeConfig("umaze", (3, 1), (1, 1), 700), ...} in `preset`
    after:  {"umaze": {"map_name": "umaze", "start_cell": (3, 1), "goal_cell": (1, 1), "max_episode_steps": 700}, ...}
    """
    source = (ROOT / "src/exploration_platform/envs/antmaze/am_common.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "preset")
    table = next(node.value for node in function.body if isinstance(node, ast.Assign))
    names = ("map_name", "start_cell", "goal_cell", "max_episode_steps")
    return {ast.literal_eval(key): dict(zip(names, map(ast.literal_eval, call.args)))
            for key, call in zip(table.keys, table.values)}


def small_parser():
    """Expose only the ordinary options needed to check merging without importing training code."""
    # The file supplies every remaining existing CLI destination through parser defaults.
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration-file", type=Path)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--parameter-settings", type=Path)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--intrinsic-coefficient", type=float, default=1.0)
    parser.add_argument("--learning-rates", default="")
    parser.add_argument("--intrinsic-weights", default="")
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--env", default="pointmaze-large")
    parser.add_argument("--reward-setting", default="environment")
    return parser


class RecordedExperimentFiles(unittest.TestCase):
    """Keep source-specific experiments separate and reject incomplete or mixed configuration files."""

    def test_complete_files_preserve_environment_and_return_choices(self):
        """Current and historical returns remain explicit in six complete recorded experiments."""
        # Pixel CleanRL uses another schema and command, so old-platform loading must reject it.
        paths = [path for path in CONFIGURATIONS.glob("*.toml")
                 if path.name != "montezuma_pixels_original_cleanrl.toml" and path not in EXAMPLES]
        self.assertEqual(len(paths), 6)
        documents = [read_experiment_configuration(path) for path in paths]
        for document in documents:
            self.assertEqual(len(document["trainer"]), FIELDS)
            # The fields added on 2026-11-12 stay at the values that change nothing.
            self.assertEqual(document["trainer"]["setting_names"], ("learning_rate", "int_coef"))
            self.assertEqual((document["trainer"]["ext_reward_scale"],
                              document["trainer"]["ext_reward_offset"]), (1.0, 0.0))
            self.assertFalse(document["trainer"]["episode_log"])
            self.assertFalse(document["trainer"]["gather_minibatches_per_epoch"])
            # The speed keys added on 2026-11-12 stay off in every recorded experiment.
            self.assertFalse(document["trainer"]["target_features_once_per_iteration"])
            self.assertEqual(document["trainer"]["permutation_draw"], "argsort_uniform")
            self.assertFalse(document["trainer"]["next_values_from_shifted_values"])
            self.assertFalse(document["trainer"]["skip_reset_draws_at_zero_noise"])
            self.assertFalse(document["trainer"]["intrinsic_sums_by_matrix_product"])
            self.assertEqual(document["trainer"]["advantage_scan_unroll"], 1)
            self.assertEqual(document["trainer"]["action_noise_generator"], "threefry")
            self.assertFalse(document["trainer"]["actor_critic_joint_trunk"])
            self.assertEqual(document["runtime"]["reward_setting"], "environment")
            self.assertTrue(document["metadata"]["source_files"])
            self.assertEqual(len(document["sha256"]), 64)
            # The physics field added on 2026-11-12 stays at MJX's JAX implementation.
            if document["metadata"]["environment_family"] == "antmaze":
                self.assertEqual(document["environment"]["physics"], "jax")
            if document["metadata"]["environment_family"] != "pointmaze":
                self.assertEqual(document["trainer"]["ent_coef"], 0.0)
                self.assertFalse(document["trainer"]["diagnostics"])
                self.assertFalse(document["trainer"]["normalize_agent_obs"])
        current, historical = map(read_experiment_configuration, (CURRENT, HISTORICAL))
        self.assertEqual(current["trainer"]["intrinsic_bootstrap_observation"], "reset_observation")
        self.assertEqual(current["trainer"]["no_bonus_advantage_stream"], "task_only")
        self.assertEqual(historical["trainer"]["intrinsic_bootstrap_observation"], "final_observation")
        self.assertEqual(historical["trainer"]["no_bonus_advantage_stream"], "weighted_intrinsic")

    def test_study_examples_hold_the_cleanrl_values_and_goal_termination(self):
        """One example per family: CleanRL learner values, the episode ends at the goal."""
        for path, preset, limit in ((POINTMAZE_EXAMPLE, "pointmaze_current", 400),
                                    (ANTMAZE_EXAMPLE, "antmaze_current", 700),
                                    (ANTMAZE_WARP_EXAMPLE, "antmaze_current", 700),
                                    (ANTMAZE_STABLE_PHYSICS_EXAMPLE, "antmaze_current", 700),
                                    (ANTMAZE_STABLE_PHYSICS_MAPS["medium"], "antmaze_current", 1000),
                                    (ANTMAZE_STABLE_PHYSICS_MAPS["large"], "antmaze_current", 1000)):
            document = read_experiment_configuration(path)
            trainer, environment = document["trainer"], document["environment"]
            self.assertEqual(len(trainer), FIELDS)
            self.assertEqual(trainer["configuration_preset"], preset)
            self.assertEqual((trainer["update_style"], trainer["update_epochs"],
                              trainer["num_minibatches"]), ("epoch_minibatch", 4, 4))
            self.assertEqual((trainer["clip_coef"], trainer["learning_rate"], trainer["ent_coef"],
                              trainer["prime_iterations"]), (0.1, 0.0001, 0.001, 50))
            self.assertTrue(trainer["anneal_lr"] and trainer["episode_log"])
            self.assertEqual(document["runtime"]["lr_anneal_iterations"],
                             document["runtime"]["iterations"])
            self.assertFalse(environment["continuing_task"])
            self.assertEqual((environment["max_episode_steps"], environment["reward_shift"]),
                             (limit, 0.0))
            self.assertEqual(document["runtime"]["reward_setting"], "arrival_plus_one")
        # The Warp example is the AntMaze example with the physics field changed and nothing else;
        # a file's physics becomes the command's default, and PointMaze files carry none.
        jax_example, warp_example, pointmaze = map(read_experiment_configuration, (
            ANTMAZE_EXAMPLE, ANTMAZE_WARP_EXAMPLE, POINTMAZE_EXAMPLE))
        self.assertEqual(warp_example["environment"], {**jax_example["environment"], "physics": "warp"})
        self.assertEqual((warp_example["trainer"], warp_example["runtime"]),
                         (jax_example["trainer"], jax_example["runtime"]))
        self.assertEqual((jax_example["environment"]["warp_contact_slots_per_environment"],
                          jax_example["environment"]["warp_constraint_rows_per_environment"]), (48, 512))
        self.assertEqual((argument_defaults(jax_example)["physics"],
                          argument_defaults(warp_example)["physics"]), ("jax", "warp"))
        self.assertNotIn("physics", argument_defaults(pointmaze))

    def test_the_physics_time_step_is_0_01_everywhere_but_in_the_stable_physics_example(self):
        """Every AntMaze file keeps 0.01 s, 5 physics steps and the guard off, except the three files the
        AntMaze baseline uses (one per map); the U-maze one differs from the Warp example in exactly four
        environment values."""
        ant_files = [path for path in CONFIGURATIONS.glob("antmaze_*.toml")]
        self.assertEqual(len(ant_files), 9)
        for path in ant_files:
            environment = read_experiment_configuration(path)["environment"]
            if path in STABLE_PHYSICS_FILES or path in RK4_PHYSICS_MAPS.values():
                continue
            self.assertEqual((environment["physics_time_step"], environment["frame_skip"],
                              environment["fault_torso_speed"], environment["fault_torso_angular_speed"]),
                             (0.01, 5, 0.0, 0.0), path.name)
        warp_example, stable = map(read_experiment_configuration, (ANTMAZE_WARP_EXAMPLE,
                                                                   ANTMAZE_STABLE_PHYSICS_EXAMPLE))
        self.assertEqual(stable["environment"], {**warp_example["environment"], "physics_time_step": 0.005,
                                                 "frame_skip": 10, "fault_torso_speed": 30.0,
                                                 "fault_torso_angular_speed": 100.0})
        self.assertEqual((stable["trainer"], stable["runtime"]), (warp_example["trainer"], warp_example["runtime"]))
        # one environment step lasts 0.05 s in both
        for document in (warp_example, stable):
            environment = document["environment"]
            self.assertAlmostEqual(environment["physics_time_step"] * environment["frame_skip"], 0.05, places=12)
        self.assertEqual(argument_defaults(stable)["physics"], "warp")

    def test_the_medium_and_large_files_are_the_u_maze_file_on_their_own_map(self):
        """The Medium and Large stable-physics files hold the U-maze file's learner, physics and runtime
        values, and the map, start and goal cells and step limit of the environment's own preset."""
        umaze = read_experiment_configuration(ANTMAZE_STABLE_PHYSICS_EXAMPLE)
        presets = antmaze_presets()
        self.assertEqual(presets["umaze"], {key: umaze["environment"][key] for key in presets["umaze"]})
        for name, path in ANTMAZE_STABLE_PHYSICS_MAPS.items():
            document = read_experiment_configuration(path)
            # before: the U-maze file's [environment]; after: the same with the map's four values
            self.assertEqual(document["environment"], {**umaze["environment"], **presets[name]})
            self.assertEqual(document["trainer"], umaze["trainer"])
            self.assertEqual(document["runtime"], {**umaze["runtime"], "environment": f"antmaze-{name}"})
            self.assertEqual(document["bonus_options"], umaze["bonus_options"])
            self.assertEqual(document["metadata"]["source_files"], umaze["metadata"]["source_files"])
            self.assertEqual(argument_defaults(document)["episode_steps"], 1000)
            self.assertEqual(argument_defaults(document)["physics"], "warp")
        self.assertEqual((presets["medium"]["start_cell"], presets["medium"]["goal_cell"]), ((6, 1), (1, 6)))
        self.assertEqual((presets["large"]["start_cell"], presets["large"]["goal_cell"]), ((7, 1), (1, 10)))
        # a file whose map and runtime selector disagree is refused
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration.toml"
            path.write_text(ANTMAZE_STABLE_PHYSICS_MAPS["large"].read_text().replace(
                'environment = "antmaze-large"', 'environment = "antmaze-medium"'))
            with self.assertRaisesRegex(ValueError, "recorded maze map disagree"):
                read_experiment_configuration(path)

    def test_the_rk4_files_are_the_stable_physics_files_with_three_physics_values_changed(self):
        """Each RK4 file holds its stable-physics file's learner, runtime, bonus and map values, and in [environment]
        only the integrator (RK4), the time step (0.01 s) and the physics steps per environment step (5) differ."""
        for name, path in RK4_PHYSICS_MAPS.items():
            stable_path = ANTMAZE_STABLE_PHYSICS_EXAMPLE if name == "umaze" else ANTMAZE_STABLE_PHYSICS_MAPS[name]
            stable, rk4 = map(read_experiment_configuration, (stable_path, path))
            # before: implicitfast, 0.005 s x 10 ; after: RK4, 0.01 s x 5
            self.assertEqual(rk4["environment"], {**stable["environment"], "integrator": "RK4",
                                                  "physics_time_step": 0.01, "frame_skip": 5})
            self.assertEqual((rk4["trainer"], rk4["runtime"], rk4["bonus_options"]),
                             (stable["trainer"], stable["runtime"], stable["bonus_options"]))
            # one environment step still lasts 0.05 s, and the guard stays on
            self.assertAlmostEqual(rk4["environment"]["physics_time_step"] * rk4["environment"]["frame_skip"], 0.05,
                                   places=12)
            self.assertEqual((rk4["environment"]["fault_torso_speed"], rk4["environment"]["fault_torso_angular_speed"]),
                             (30.0, 100.0))
            self.assertEqual(argument_defaults(rk4)["physics"], "warp")

    def test_named_reward_settings_fix_the_scale_and_offset(self):
        """A file's reward scale and offset must be the ones its named reward setting means."""
        # Before: reward 1 on arrival, 0 elsewhere; after step_cost at limit 400: 0 and -1/400.
        self.assertEqual(reward_mappings("environment", 400), ())
        self.assertEqual(reward_mappings("arrival_plus_one", 400), ((1.0, 0.0),))
        self.assertEqual(reward_mappings("step_cost", 400), ((0.0025, -0.0025),))
        self.assertEqual(reward_mappings("arrival_plus_one_and_step_cost", 700),
                         ((1.0, 0.0), (1 / 700, -1 / 700)))
        with self.assertRaisesRegex(ValueError, "unknown reward setting"):
            reward_mappings("per_step", 400)
        text = POINTMAZE_EXAMPLE.read_text()
        step_cost = text.replace('reward_setting = "arrival_plus_one"', 'reward_setting = "step_cost"')
        both = (text.replace('reward_setting = "arrival_plus_one"',
                             'reward_setting = "arrival_plus_one_and_step_cost"')
                .replace('setting_names = ["learning_rate", "int_coef"]',
                         'setting_names = ["int_coef", "ext_reward_scale", "ext_reward_offset"]')
                .replace("parameter_settings = []",
                         "parameter_settings = [[1.0, 1.0, 0.0], [1.0, 0.0025, -0.0025]]")
                .replace("copies_per_group = 0", "copies_per_group = 8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration.toml"
            # step_cost with the scale and offset of arrival_plus_one is refused
            path.write_text(step_cost)
            with self.assertRaisesRegex(ValueError, "ext_reward_scale"):
                read_experiment_configuration(path)
            path.write_text(step_cost.replace("ext_reward_scale = 1.0", "ext_reward_scale = 0.0025")
                            .replace("ext_reward_offset = 0.0", "ext_reward_offset = -0.0025"))
            self.assertEqual(read_experiment_configuration(path)["trainer"]["ext_reward_offset"], -0.0025)
            # a named setting on an episode that does not end at the goal is refused
            path.write_text(text.replace("continuing_task = false", "continuing_task = true"))
            with self.assertRaisesRegex(ValueError, "ends the episode at the goal"):
                read_experiment_configuration(path)
            # both settings need cells that hold exactly the two pairs
            path.write_text(both)
            document = read_experiment_configuration(path)
            self.assertEqual(document["trainer"]["parameter_settings"],
                             ((1.0, 1.0, 0.0), (1.0, 0.0025, -0.0025)))
            parsed = parse_with_configuration_file(small_parser(), [
                "--run-dir", "/tmp/test", "--unit-id", "test", "--configuration-file", str(path),
                "--intrinsic-coefficient", "4"])
            self.assertEqual(parsed.experiment_setting_names,
                             ("int_coef", "ext_reward_scale", "ext_reward_offset"))
            self.assertEqual(parsed.experiment_parameter_settings,
                             ((4.0, 1.0, 0.0), (4.0, 0.0025, -0.0025)))
            for refused in (["--reward-setting", "step_cost"], ["--intrinsic-weights", "1,2"]):
                with self.assertRaisesRegex(ValueError, "edit the file's cells"):
                    parse_with_configuration_file(small_parser(), [
                        "--run-dir", "/tmp/test", "--unit-id", "test",
                        "--configuration-file", str(path), *refused])
            path.write_text(both.replace("[1.0, 0.0025, -0.0025]", "[1.0, 0.5, -0.5]"))
            with self.assertRaisesRegex(ValueError, "exactly the pairs"):
                read_experiment_configuration(path)

    def test_confirmed_studies_use_exact_unit_headers_and_schedule_horizons(self):
        """The two confirmed entropy experiments retain their actual cells, copy counts and budgets."""
        # The old run-level YAML describes another unit; these values came from each unit_start row.
        first, second = map(read_experiment_configuration, (PART_A, PART_B))
        self.assertEqual(first["trainer"]["parameter_settings"],
                         ((0.001, 1.0), (0.001, 10.0), (0.001, 100.0), (0.001, 1000.0)))
        self.assertEqual(first["trainer"]["n_copies"], 512)
        self.assertEqual(first["runtime"]["iterations"], 195313)
        self.assertEqual(first["runtime"]["lr_anneal_iterations"], 195313)
        self.assertEqual(second["trainer"]["parameter_settings"], ((0.0001, 100.0), (0.0001, 1000.0)))
        self.assertEqual(second["trainer"]["n_copies"], 256)
        self.assertEqual(second["runtime"]["iterations"], 781250)
        self.assertEqual(second["runtime"]["lr_anneal_iterations"], 781250)
        for document in (first, second):
            self.assertEqual(document["trainer"]["ent_coef"], 0.01)
            self.assertFalse(document["trainer"]["clip_per_network"])
            self.assertTrue(document["environment"]["continuing_task"])
            self.assertEqual(document["environment"]["reward_shift"], 0.0)

    def test_explicit_cli_overrides_file_values_without_mutating_the_file(self):
        """An explicit scalar or list changes its requested cells and preserves unrelated settings."""
        # Before: four cells at rate .001 and entropy .01; after: .0002 and entropy zero.
        arguments = ["--run-dir", "/tmp/recorded-config-test", "--unit-id", "test",
                     "--configuration-file", str(PART_A), "--learning-rate", "0.0002",
                     "--ent-coef", "0"]
        parsed = parse_with_configuration_file(small_parser(), arguments)
        self.assertEqual(parsed.ent_coef, 0.0)
        self.assertEqual(parsed.experiment_parameter_settings,
                         ((0.0002, 1.0), (0.0002, 10.0), (0.0002, 100.0), (0.0002, 1000.0)))
        self.assertEqual(parsed.experiment_setting_names, ("learning_rate", "int_coef"))
        self.assertEqual(read_experiment_configuration(PART_A)["trainer"]["ent_coef"], 0.01)
        parsed = parse_with_configuration_file(small_parser(), arguments + ["--intrinsic-weights", "2,3"])
        self.assertEqual(parsed.experiment_parameter_settings, ())
        self.assertEqual(parsed.intrinsic_weights, "2,3")
        self.assertEqual(parsed.learning_rates, "0.0002")
        self.assertEqual(additional_trainer_fields(parsed)["gamma_int"], 0.99)
        parsed = parse_with_configuration_file(small_parser(), [
            "--run-dir", "/tmp/test", "--unit-id", "test", "--configuration-file", str(HISTORICAL),
            "--parameter-settings", "/tmp/explicit-settings.json"])
        self.assertEqual(parsed.learning_rates, "")
        self.assertEqual(parsed.intrinsic_weights, "")
        self.assertEqual(parsed.experiment_parameter_settings, ())

    def test_no_file_retains_the_existing_parser_defaults(self):
        """No new file flag means no experiment metadata or changed entropy/rate defaults."""
        parsed = parse_with_configuration_file(small_parser(), ["--run-dir", "/tmp/test", "--unit-id", "test"])
        self.assertFalse(hasattr(parsed, "experiment_configuration_data"))
        self.assertEqual(parsed.learning_rate, 0.0003)
        self.assertEqual(parsed.ent_coef, 0.0)
        self.assertEqual(additional_trainer_fields(parsed), {})

    def test_incomplete_mixed_and_pixel_files_are_rejected(self):
        """Loading refuses hidden defaults, mixed environment identities and the pixel schema."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration.toml"
            path.write_text(CURRENT.read_text().replace("n_envs = 4\n", ""))
            with self.assertRaisesRegex(ValueError, "missing"):
                read_experiment_configuration(path)
            path.write_text(CURRENT.read_text().replace('configuration_preset = "pointmaze_current"',
                                                       'configuration_preset = "antmaze_historical"'))
            with self.assertRaisesRegex(ValueError, "family disagree"):
                read_experiment_configuration(path)
        with self.assertRaisesRegex(ValueError, "separate command"):
            read_experiment_configuration(CONFIGURATIONS / "montezuma_pixels_original_cleanrl.toml")
        with self.assertRaisesRegex(ValueError, "own file"):
            parse_with_configuration_file(small_parser(), ["--run-dir", "/tmp/test", "--unit-id", "test",
                "--configuration-file", str(CURRENT), "--env", "antmaze-umaze"])


if __name__ == "__main__":
    unittest.main()
