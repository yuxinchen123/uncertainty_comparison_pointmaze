"""Keep PointMaze corrections and historical environment choices in distinct named configurations."""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig
from exploration_platform.agents.ppo.presets import resolve_configuration
from exploration_platform.envs.pointmaze.pm_common import EnvConfig
from exploration_platform.envs.antmaze.am_common import AntMazeConfig


def test_environment_specific_choices():
    """Resolve each environment to explicit choices without modifying the input configuration."""
    original = PPOConfig()
    current = resolve_configuration(original, EnvConfig())
    historical = resolve_configuration(original, AntMazeConfig())
    assert current.configuration_preset == "pointmaze_current"
    assert current.intrinsic_bootstrap_observation == "reset_observation"
    assert current.no_bonus_advantage_stream == "task_only"
    assert historical.configuration_preset == "antmaze_historical"
    assert historical.intrinsic_bootstrap_observation == "final_observation"
    assert historical.no_bonus_advantage_stream == "weighted_intrinsic"
    assert original.configuration_preset == "environment_default"
    assert resolve_configuration(current, EnvConfig()) == current
    print("PASS environment-specific configurations resolve explicitly")


def test_historical_pointmaze_and_rejections():
    """Select historical PointMaze deliberately and refuse mixed environment or return choices."""
    historical = resolve_configuration(PPOConfig(configuration_preset="pointmaze_historical"),
                                       EnvConfig())
    assert historical.intrinsic_bootstrap_observation == "final_observation"
    for config, environment in (
            (historical, AntMazeConfig()),
            (replace(historical, intrinsic_bootstrap_observation="reset_observation"), EnvConfig()),
            (PPOConfig(diagnostics=True), AntMazeConfig())):
        try:
            resolve_configuration(config, environment)
        except ValueError:
            continue
        raise AssertionError("a mixed environment configuration was accepted")
    print("PASS historical PointMaze is explicit and incompatible choices fail")


if __name__ == "__main__":
    test_environment_specific_choices()
    test_historical_pointmaze_and_rejections()
