"""Check recorded experiment files and CLI merging using the standard library only."""

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_configuration import (additional_trainer_fields, argument_defaults,
                                      parse_with_configuration_file, read_experiment_configuration)

CONFIGURATIONS = ROOT / "configs/experiments"
CURRENT = CONFIGURATIONS / "pointmaze_large_current_returns_random_network_distillation_steps-100-million.toml"
HISTORICAL = CONFIGURATIONS / "pointmaze_large_historical_random_network_distillation_steps-100-million.toml"
PART_A = CONFIGURATIONS / "pointmaze_large_entropy-0.01_learning-rate-0.001_within-run-anneal_steps-100-million.toml"
PART_B = CONFIGURATIONS / "pointmaze_large_entropy-0.01_learning-rate-0.0001_within-run-anneal_steps-400-million.toml"


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
    return parser


class RecordedExperimentFiles(unittest.TestCase):
    """Keep source-specific experiments separate and reject incomplete or mixed configuration files."""

    def test_complete_files_preserve_environment_and_return_choices(self):
        """Current and historical returns remain explicit in six complete recorded experiments."""
        # Pixel CleanRL uses another schema and command, so old-platform loading must reject it.
        paths = [path for path in CONFIGURATIONS.glob("*.toml")
                 if path.name != "montezuma_pixels_original_cleanrl.toml"]
        self.assertEqual(len(paths), 6)
        documents = [read_experiment_configuration(path) for path in paths]
        for document in documents:
            self.assertEqual(len(document["trainer"]), 45)
            self.assertTrue(document["metadata"]["source_files"])
            self.assertEqual(len(document["sha256"]), 64)
            if document["metadata"]["environment_family"] != "pointmaze":
                self.assertEqual(document["trainer"]["ent_coef"], 0.0)
                self.assertFalse(document["trainer"]["diagnostics"])
                self.assertFalse(document["trainer"]["normalize_agent_obs"])
        current, historical = map(read_experiment_configuration, (CURRENT, HISTORICAL))
        self.assertEqual(current["trainer"]["intrinsic_bootstrap_observation"], "reset_observation")
        self.assertEqual(current["trainer"]["no_bonus_advantage_stream"], "task_only")
        self.assertEqual(historical["trainer"]["intrinsic_bootstrap_observation"], "final_observation")
        self.assertEqual(historical["trainer"]["no_bonus_advantage_stream"], "weighted_intrinsic")

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
