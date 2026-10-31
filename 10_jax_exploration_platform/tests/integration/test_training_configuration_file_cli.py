"""Check old training CLI configuration files on a compute node without training policies."""

from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from experiment_configuration import read_experiment_configuration
from run_training import build_config, build_env_config, parse_args


def test_all_recorded_files_reconstruct_the_complete_trainer_and_environment():
    """File-based construction must preserve every recorded field instead of using global defaults."""
    # Build dataclasses only; this test never initializes a game, network or output directory.
    for path in (ROOT / "configs/experiments").glob("*.toml"):
        if path.name == "montezuma_pixels_original_cleanrl.toml":
            continue
        document = read_experiment_configuration(path)
        arguments = parse_args(["--run-dir", "/tmp/configuration-cli-test", "--unit-id", "test",
                                "--configuration-file", str(path)])
        actual = asdict(build_config(arguments))
        assert actual == document["trainer"], (path.name, {
            key: (actual[key], value) for key, value in document["trainer"].items() if actual[key] != value})
        assert asdict(build_env_config(arguments)) == document["environment"], path.name


def test_no_file_keeps_existing_pointmaze_arguments_and_no_bonus_behavior():
    """The optional file loader changes neither ordinary CLI defaults nor the no-bonus correction."""
    arguments = parse_args(["--run-dir", "/tmp/configuration-cli-test", "--unit-id", "test"])
    configuration = build_config(arguments)
    assert configuration.configuration_preset == "pointmaze_current"
    assert configuration.ent_coef == 0.0 and configuration.learning_rate == 0.0003
    no_bonus = parse_args(["--run-dir", "/tmp/configuration-cli-test", "--unit-id", "test", "--bonus", "none"])
    assert build_config(no_bonus).int_coef == 0.0
