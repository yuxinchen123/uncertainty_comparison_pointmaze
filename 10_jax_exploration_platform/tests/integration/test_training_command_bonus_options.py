"""The training command's bonus options (2026-11-12): `--bonus-options`, the optional
`[bonus_options]` table of a configuration file, the `unit_start` record's `bonus_options`, the
maze-grid fields of the window records, and stopping and continuing a unit that uses them. The
option string of every unit of batch 2 of the PointMaze and AntMaze study is run through the
command (`BATCH_2_UNITS`).

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
from run_training import bonus_options_of, build_config, parse_args  # noqa: E402

BASE = ["--run-dir", "/tmp/bonus-options-test", "--unit-id", "test"]
POINTMAZE_EXAMPLE = ROOT / "configs/experiments" / (
    "pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml")
# open cells of the large PointMaze map
OPEN_CELLS = 46
SHORT = ["--unit-id", "unit", "--copies", "2", "--envs-per-copy", "2", "--rollout-steps", "8",
         "--iterations", "5", "--window-iterations", "2", "--episode-steps", "7",
         "--update-style", "epoch_minibatch", "--clip-coef", "0.1", "--prime-iterations", "1",
         "--episode-log", "--terminate-at-goal"]
# the same unit planned for 16 iterations in 4 windows of 4
SIXTEEN = [*SHORT[:SHORT.index("--iterations")], "--iterations", "16",
           "--window-iterations", "4", *SHORT[SHORT.index("--episode-steps"):]]
# every option of the distillation bonus at its default, as a unit's start record holds them
DEFAULT_RECORD = {
    "feature_dim": 128, "hidden": 256, "even_start": False, "reward_divisor": "running_std",
    "predictor_learning_rate_multiple": 0.0, "bonus_form": "half_squared_error",
    "fast_last_layer_step": 0.0, "scheduled_last_layer_decay": 0.0,
    "record_bonus_on_maze_grid": False, "input_clip": 5.0, "input_statistics": "running",
    "input_features": "raw", "fourier_feature_count": 256, "fourier_length_scale": 1.0,
    "scheduled_last_layer_features": "body", "divisor_hold_fraction": 0.0}

# The 18 units of batch 2 in the order of its ideas table: (name, the unit's --bonus-options,
# what its start record must hold beside the defaults and the maze-grid record). Random Fourier
# features bring input statistics frozen after priming and no clip; the scheduled last layer
# also brings the even start and no divisor.
GRID = "record_bonus_on_maze_grid=true"
KEPT = {"input_statistics": "frozen_after_priming"}
LOCAL_INPUT = {"input_features": "random_fourier", **KEPT, "input_clip": 0.0}
LOCAL_LAYER = {"scheduled_last_layer_features": "random_fourier", "fourier_feature_count": 512,
               **KEPT, "input_clip": 0.0, "even_start": True, "reward_divisor": "none"}
HELD = {"reward_divisor": "held_after_fraction"}
BATCH_2_UNITS = (
    ("control", GRID, {}),
    ("wider input clip, 10", f"input_clip=10,{GRID}", {"input_clip": 10.0}),
    ("wider input clip, 20", f"input_clip=20,{GRID}", {"input_clip": 20.0}),
    ("wider input clip, no clip", f"input_clip=0,{GRID}", {"input_clip": 0.0}),
    ("input statistics kept from the random-action rollouts, no clip",
     f"input_statistics=frozen_after_priming,input_clip=0,{GRID}", {**KEPT, "input_clip": 0.0}),
    ("input statistics kept from the random-action rollouts, clip 20",
     f"input_statistics=frozen_after_priming,input_clip=20,{GRID}",
     {**KEPT, "input_clip": 20.0}),
    ("slower predictor, 0.0625 x 1e-4", f"predictor_learning_rate_multiple=0.0625,{GRID}",
     {"predictor_learning_rate_multiple": 0.0625}),
    ("slower predictor, 0.125 x 1e-4", f"predictor_learning_rate_multiple=0.125,{GRID}",
     {"predictor_learning_rate_multiple": 0.125}),
    ("local features as the input of RND, length 0.5",
     f"input_features=random_fourier,fourier_feature_count=256,fourier_length_scale=0.5,{GRID}",
     {**LOCAL_INPUT, "fourier_length_scale": 0.5}),
    ("local features as the input of RND, length 1",
     f"input_features=random_fourier,fourier_feature_count=256,fourier_length_scale=1,{GRID}",
     LOCAL_INPUT),
    ("local features as the input of RND, length 2",
     f"input_features=random_fourier,fourier_feature_count=256,fourier_length_scale=2,{GRID}",
     {**LOCAL_INPUT, "fourier_length_scale": 2.0}),
    ("scheduled last layer on local features, decay 0.5, length 0.5",
     "scheduled_last_layer_decay=0.5,scheduled_last_layer_features=random_fourier,"
     f"fourier_feature_count=512,fourier_length_scale=0.5,{GRID}",
     {**LOCAL_LAYER, "scheduled_last_layer_decay": 0.5, "fourier_length_scale": 0.5}),
    ("scheduled last layer on local features, decay 0.5, length 1",
     "scheduled_last_layer_decay=0.5,scheduled_last_layer_features=random_fourier,"
     f"fourier_feature_count=512,fourier_length_scale=1,{GRID}",
     {**LOCAL_LAYER, "scheduled_last_layer_decay": 0.5}),
    ("scheduled last layer on local features, decay 0.5, length 2",
     "scheduled_last_layer_decay=0.5,scheduled_last_layer_features=random_fourier,"
     f"fourier_feature_count=512,fourier_length_scale=2,{GRID}",
     {**LOCAL_LAYER, "scheduled_last_layer_decay": 0.5, "fourier_length_scale": 2.0}),
    ("scheduled last layer on local features, decay 1, length 1",
     "scheduled_last_layer_decay=1,scheduled_last_layer_features=random_fourier,"
     f"fourier_feature_count=512,fourier_length_scale=1,{GRID}",
     {**LOCAL_LAYER, "scheduled_last_layer_decay": 1.0}),
    ("divisor held after 1/16 of the run",
     f"reward_divisor=held_after_fraction,divisor_hold_fraction=0.0625,{GRID}",
     {**HELD, "divisor_hold_fraction": 0.0625}),
    ("divisor held after 1/8 of the run",
     f"reward_divisor=held_after_fraction,divisor_hold_fraction=0.125,{GRID}",
     {**HELD, "divisor_hold_fraction": 0.125}),
    ("divisor held after 1/4 of the run",
     f"reward_divisor=held_after_fraction,divisor_hold_fraction=0.25,{GRID}",
     {**HELD, "divisor_hold_fraction": 0.25}),
)
# fields of a window record that differ between two runs of one unit
TIMING = {"written_at", "seconds_since_first_iteration"}


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
            **DEFAULT_RECORD, "even_start": True, "reward_divisor": "none",
            "scheduled_last_layer_decay": 0.5, "record_bonus_on_maze_grid": True}
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


def windows_of(records: list) -> list:
    """A unit's window records without the fields that differ between two runs of it."""
    return [{name: value for name, value in record.items() if name not in TIMING}
            for record in records if record["record"] == "episode_window"]


def stopped_and_continued(run_dir: Path, options: list) -> tuple:
    """Run a unit that its time limit stops once, then the same command again to its end;
    returns (the names of the arrays in the checkpoint of the stop, the unit's records)."""
    try:
        run_command("--run-dir", str(run_dir), *options, "--time-limit-seconds", "1e-9")
        raise AssertionError("the time limit did not stop the unit")
    except SystemExit as stop:
        assert "run the same command again" in str(stop)
    with np.load(run_dir / "checkpoints" / "unit.npz") as saved:
        names = list(saved.files)
    run_command("--run-dir", str(run_dir), *options, "--checkpoint-seconds", "1e-9")
    return names, records_of(run_dir, "unit")


def test_the_planned_length_reaches_the_trainer_with_a_held_divisor():
    """A divisor held after a fraction of the run counts the iterations the learning rate
    anneals over; a unit without it carries no planned length, as before."""
    held = "reward_divisor=held_after_fraction,divisor_hold_fraction=0.25"
    assert build_config(parse_args([*BASE, "--iterations", "16"])).total_iterations == 0
    assert build_config(parse_args([*BASE, "--iterations", "16",
                                    "--bonus-options", held])).total_iterations == 16
    assert build_config(parse_args([*BASE, "--iterations", "16", "--lr-anneal-iterations", "64",
                                    "--bonus-options", held])).total_iterations == 64
    # a configuration file: the 2,000 iterations its learning rate anneals over
    from_file = [*BASE, "--configuration-file", str(POINTMAZE_EXAMPLE)]
    assert build_config(parse_args(from_file)).total_iterations == 0
    assert build_config(parse_args([*from_file, "--bonus-options", held])).total_iterations == 2000
    # the intrinsic weight's schedule counts the run's own iterations: with both, one length
    scheduled = ["--intrinsic-weight-schedule", "linear_decay"]
    assert build_config(parse_args([*BASE, "--iterations", "16", *scheduled,
                                    "--bonus-options", held])).total_iterations == 16
    try:
        build_config(parse_args([*BASE, "--iterations", "16", "--lr-anneal-iterations", "64",
                                 *scheduled, "--bonus-options", held]))
        raise AssertionError("two planned lengths were accepted")
    except ValueError as error:
        assert "one planned length" in str(error)
    print("PASS the planned length reaches the trainer with a held divisor, and only then")


def test_the_command_runs_every_unit_of_batch_2():
    """Each of the 18 option strings of batch 2 is accepted by the training command; the unit's
    start record holds the options with what they imply; the maze-grid record is written. A
    held divisor shows one value from its hold on, and the unit held after a quarter of the run
    equals the control up to there. Units with random Fourier features and with a held divisor,
    stopped once and continued, equal the same units run in one go."""
    assert len(BATCH_2_UNITS) == 18 and len({options for _, options, _ in BATCH_2_UNITS}) == 18
    with tempfile.TemporaryDirectory() as directory:
        windows = {}
        for number, (name, options, changes) in enumerate(BATCH_2_UNITS):
            run_dir = Path(directory) / f"unit_{number}"
            run_command("--run-dir", str(run_dir), *SIXTEEN, "--bonus-options", options)
            records = records_of(run_dir, "unit")
            start = next(record for record in records if record["record"] == "unit_start")
            assert start["bonus_options"] == {
                **DEFAULT_RECORD, "record_bonus_on_maze_grid": True, **changes}, name
            assert len(start["maze_grid_probe_cells"]) == OPEN_CELLS, name

            # the bonus on the maze grid: before training, and in each of the four windows
            at_start = [record for record in records
                        if record["record"] == "maze_grid_at_start"]
            windows[name] = windows_of(records)
            assert len(at_start) == 1 and len(windows[name]) == 4, name
            for record in (*at_start, *windows[name]):
                bonus = np.asarray(record["probe_bonus_per_copy"])
                assert bonus.shape == (2, OPEN_CELLS) and np.isfinite(bonus).all(), name
                assert (bonus > 0).all(), name
                divisor = np.asarray(record["intrinsic_reward_divisor_per_copy"])
                assert divisor.shape == (2,) and (divisor > 0).all(), name
            if "scheduled_last_layer_decay" in changes:
                # the schedule starts at a squared error of 1 at every probe
                np.testing.assert_allclose(at_start[0]["probe_bonus_per_copy"], 1.0, rtol=1e-5)

        # the divisor: moving from window to window in the control; one value in every window
        # of a unit that holds it from iteration number 1, 2 or 4 (the first window ends there)
        divisors = lambda name: [window["intrinsic_reward_divisor_per_copy"]
                                 for window in windows[name]]
        control = divisors("control")
        assert all(control[index] != control[index + 1] for index in range(3))
        for name in ("divisor held after 1/16 of the run", "divisor held after 1/8 of the run",
                     "divisor held after 1/4 of the run"):
            assert all(one == divisors(name)[0] for one in divisors(name)), name
        # held from iteration number 4: the first window (iterations 0 to 3) is the control's
        quarter = windows["divisor held after 1/4 of the run"]
        assert quarter[0] == windows["control"][0]
        assert quarter[1] != windows["control"][1]
        assert divisors("divisor held after 1/16 of the run")[0] != control[0]

        # stopped once and continued: the checkpoint holds the frequencies and phases (and no
        # predictor body), and both units end with the records of the unit run in one go
        for number, (name, expected_names) in enumerate((
                ("scheduled last layer on local features, decay 1, length 1",
                 ("bonus_constant/fourier/frequencies", "bonus_constant/fourier/phases",
                  "bonus_constant/target/W0")),
                ("local features as the input of RND, length 0.5",
                 ("bonus_constant/fourier/frequencies", "bonus_constant/fourier/phases",
                  "bonus_constant/target/W0")),
                ("divisor held after 1/4 of the run", ("bonus_constant/target/W0",)))):
            options = next(options for unit, options, _ in BATCH_2_UNITS if unit == name)
            names, records = stopped_and_continued(
                Path(directory) / f"parts_{number}", [*SIXTEEN, "--bonus-options", options])
            assert all(expected in names for expected in expected_names), (name, names)
            assert not any(one.startswith("bonus_constant/predictor_body/") for one in names)
            assert windows_of(records) == windows[name], name
    print("PASS the training command runs the 18 units of batch 2:\n  "
          + "\n  ".join(f"{name}: {options}" for name, options, _ in BATCH_2_UNITS))


if __name__ == "__main__":
    test_options_from_the_command_line_and_from_a_file()
    test_unit_records_hold_the_options_and_the_maze_grid()
    test_a_unit_without_options_is_recorded_as_before_plus_the_option_record()
    test_the_planned_length_reaches_the_trainer_with_a_held_divisor()
    test_the_command_runs_every_unit_of_batch_2()
