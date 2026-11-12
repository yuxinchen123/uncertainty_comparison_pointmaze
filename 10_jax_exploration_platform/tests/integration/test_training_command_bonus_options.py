"""The training command's bonus options (2026-11-12): `--bonus-options`, the optional
`[bonus_options]` table of a configuration file, the `unit_start` record's `bonus_options`, the
maze-grid fields of the window records, and stopping and continuing a unit that uses them.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_training  # noqa: E402
from experiment_configuration import read_experiment_configuration  # noqa: E402
from run_training import bonus_options_of, parse_args  # noqa: E402

BASE = ["--run-dir", "/tmp/bonus-options-test", "--unit-id", "test"]
POINTMAZE_EXAMPLE = ROOT / "configs/experiments" / (
    "pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml")
# open cells of the large PointMaze map
OPEN_CELLS = 46
SHORT = ["--unit-id", "unit", "--copies", "2", "--envs-per-copy", "2", "--rollout-steps", "8",
         "--iterations", "5", "--window-iterations", "2", "--episode-steps", "7",
         "--update-style", "epoch_minibatch", "--clip-coef", "0.1", "--prime-iterations", "1",
         "--episode-log", "--terminate-at-goal"]


def run_command(*options) -> None:
    """Call the training command in this process with the given command line."""
    command_line = sys.argv
    sys.argv = ["run_training.py", *options]
    try:
        run_training.main()
    finally:
        sys.argv = command_line
        run_training.STOP["requested"] = False


def records_of(run_dir: Path, unit: str) -> list:
    """Every record of one unit's shard, in file order."""
    return [json.loads(line) for line in (run_dir / "data" / f"{unit}.jsonl").read_text()
            .splitlines() if line.strip()]


def test_options_from_the_command_line_and_from_a_file():
    """Pairs on the command line are read by name; a file's table comes first and a pair on the
    command line replaces the same name; a file without the table gives no options."""
    assert bonus_options_of(parse_args(BASE)) == {}
    arguments = parse_args([*BASE, "--bonus-options", "even_start=true, reward_divisor=none"])
    assert bonus_options_of(arguments) == {"even_start": "true", "reward_divisor": "none"}
    for wrong in ("even_start", "=true", "even_start="):
        try:
            bonus_options_of(parse_args([*BASE, "--bonus-options", wrong]))
            raise AssertionError(f"accepted: {wrong}")
        except ValueError as error:
            assert "name=value" in str(error)

    assert read_experiment_configuration(POINTMAZE_EXAMPLE)["bonus_options"] == {}
    assert bonus_options_of(parse_args([*BASE, "--configuration-file",
                                        str(POINTMAZE_EXAMPLE)])) == {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "with_bonus_options.toml"
        path.write_text(POINTMAZE_EXAMPLE.read_text()
                        + "\n[bonus_options]\neven_start = true\nfast_last_layer_step = 0.1\n")
        assert read_experiment_configuration(path)["bonus_options"] == {
            "even_start": True, "fast_last_layer_step": 0.1}
        arguments = parse_args([*BASE, "--configuration-file", str(path),
                                "--bonus-options", "fast_last_layer_step=0.3,bonus_form=error_length"])
        assert bonus_options_of(arguments) == {"even_start": True, "fast_last_layer_step": "0.3",
                                               "bonus_form": "error_length"}
        # a table that holds anything but single values is refused
        path.write_text(POINTMAZE_EXAMPLE.read_text() + "\n[bonus_options]\neven_start = [true]\n")
        try:
            read_experiment_configuration(path)
            raise AssertionError("a list was accepted as an option's value")
        except ValueError as error:
            assert "bonus_options" in str(error)
    print("PASS bonus options from the command line, from a file, and both")


def test_unit_records_hold_the_options_and_the_maze_grid():
    """A unit with the scheduled last layer recorded on the maze grid, stopped once and
    continued, equals the same unit run in one go; its records hold the options and the grid."""
    options = [*SHORT, "--bonus-options",
               "scheduled_last_layer_decay=0.5,record_bonus_on_maze_grid=true"]
    with tempfile.TemporaryDirectory() as directory:
        whole, parts = Path(directory) / "whole", Path(directory) / "parts"
        run_command("--run-dir", str(whole), *options)
        try:
            run_command("--run-dir", str(parts), *options, "--time-limit-seconds", "1e-9")
            raise AssertionError("the time limit did not stop the unit")
        except SystemExit as stop:
            assert "run the same command again" in str(stop)

        # the checkpoint names the options; a unit with other options may not continue from it
        with np.load(parts / "checkpoints" / "unit.npz") as saved:
            identity = json.loads(bytes(saved["record"]).decode())["identity"]
            assert any(name.startswith("bonus_constant/predictor_body/") for name in saved.files)
        assert identity["bonus_options"]["scheduled_last_layer_decay"] == 0.5
        try:
            run_command("--run-dir", str(parts), *SHORT, "--checkpoint-seconds", "1e-9",
                        "--bonus-options", "scheduled_last_layer_decay=1,record_bonus_on_maze_grid=true")
            raise AssertionError("a checkpoint of other bonus options was continued")
        except ValueError as error:
            assert "bonus_options" in str(error)
        run_command("--run-dir", str(parts), *options, "--checkpoint-seconds", "1e-9")

        expected, actual = records_of(whole, "unit"), records_of(parts, "unit")
        windows = lambda records: [r for r in records if r["record"] == "episode_window"]
        timing = {"written_at", "seconds_since_first_iteration"}
        assert len(windows(expected)) == len(windows(actual)) == 3
        for one, other in zip(windows(expected), windows(actual)):
            assert {k: v for k, v in one.items() if k not in timing} == {
                k: v for k, v in other.items() if k not in timing}, one["window_index"]

        # the identity card: every option of the family, with what the schedule implies
        start = expected[1]
        assert start["record"] == "unit_start" and start["bonus"] == "rnd_next_state"
        assert start["bonus_options"] == {
            "feature_dim": 128, "hidden": 256, "even_start": True, "reward_divisor": "none",
            "predictor_learning_rate_multiple": 0.0, "bonus_form": "half_squared_error",
            "fast_last_layer_step": 0.0, "scheduled_last_layer_decay": 0.5,
            "record_bonus_on_maze_grid": True}
        cells = start["maze_grid_probe_cells"]
        assert len(cells) == OPEN_CELLS and [7, 1] in cells and [1, 10] in cells
        assert "scheduled_last_layer_decay: 0.5" in (whole / "config_resolved.yaml").read_text()

        # before the first training iteration: squared error 1 at every probe, no visit yet;
        # the continued unit keeps this record from its first attempt
        for records in (expected, actual):
            at_start = [r for r in records if r["record"] == "maze_grid_at_start"]
            assert len(at_start) == 1
            np.testing.assert_allclose(at_start[0]["probe_bonus_per_copy"], 1.0, rtol=1e-5)
            assert not np.asarray(at_start[0]["cell_visits_per_copy"]).any()

        # the window records: one number per copy and probe; no divisor, so the reward is the
        # bonus; every state of the window's rollouts is counted in one open cell
        for window in windows(expected):
            bonus = np.asarray(window["probe_bonus_per_copy"])
            assert bonus.shape == (2, OPEN_CELLS) and (bonus > 0).all() and (bonus < 1.5).all()
            assert window["probe_intrinsic_reward_per_copy"] == window["probe_bonus_per_copy"]
            assert window["intrinsic_reward_divisor_per_copy"] == [1.0, 1.0]
            visits = np.asarray(window["cell_visits_per_copy"])
            assert visits.shape == (2, OPEN_CELLS)
            assert (visits.sum(axis=1) == window["iterations_in_window"] * 8 * 2).all()
        # the start cell is visited in every window, and its bonus has fallen below the start's 1
        start_cell = cells.index([7, 1])
        last = windows(expected)[-1]
        assert all(row[start_cell] > 0 for row in last["cell_visits_per_copy"])
        assert all(row[start_cell] < 0.9 for row in last["probe_bonus_per_copy"])
    print("PASS a unit with bonus options: records hold the options and the maze grid; "
          "stopped and continued equals the whole unit")


def test_a_unit_without_options_is_recorded_as_before_plus_the_option_record():
    """No option given: the unit's record lists the family's defaults, its windows carry no
    maze-grid field, and its checkpoint's identity has no entry for bonus options."""
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"
        try:
            run_command("--run-dir", str(run_dir), *SHORT, "--time-limit-seconds", "1e-9")
            raise AssertionError("the time limit did not stop the unit")
        except SystemExit:
            pass
        with np.load(run_dir / "checkpoints" / "unit.npz") as saved:
            identity = json.loads(bytes(saved["record"]).decode())["identity"]
        assert "bonus_options" not in identity
        run_command("--run-dir", str(run_dir), *SHORT, "--checkpoint-seconds", "1e-9")
        records = records_of(run_dir, "unit")
        start = next(record for record in records if record["record"] == "unit_start")
        assert start["bonus_options"]["even_start"] is False
        assert start["bonus_options"]["reward_divisor"] == "running_std"
        assert "maze_grid_probe_cells" not in start
        window = next(record for record in records if record["record"] == "episode_window")
        assert not any(name.startswith(("probe_", "cell_visits", "intrinsic_reward_divisor"))
                       for name in window)
        assert "bonus_options" in (run_dir / "config_resolved.yaml").read_text()

        # the no-bonus family has no options: an empty record, and options are refused
        none_dir = Path(directory) / "none"
        run_command("--run-dir", str(none_dir), *SHORT, "--bonus", "none",
                    "--intrinsic-coefficient", "0")
        start = next(record for record in records_of(none_dir, "unit")
                     if record["record"] == "unit_start")
        assert start["bonus_options"] == {}
        try:
            run_command("--run-dir", str(Path(directory) / "refused"), *SHORT, "--bonus", "none",
                        "--bonus-options", "even_start=true")
            raise AssertionError("options were accepted for the no-bonus family")
        except ValueError as error:
            assert "has no options" in str(error)
    print("PASS a unit without options: defaults recorded, no maze-grid field, identity as before")


if __name__ == "__main__":
    test_options_from_the_command_line_and_from_a_file()
    test_unit_records_hold_the_options_and_the_maze_grid()
    test_a_unit_without_options_is_recorded_as_before_plus_the_option_record()
