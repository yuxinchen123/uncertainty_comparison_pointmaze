"""The training command's options for the PointMaze and AntMaze study (2026-11-12): CleanRL
values, sweeps over more settings, the two reward settings, goal termination for AntMaze, the
episode log's records, and stopping and continuing a unit from its checkpoint.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_training  # noqa: E402
from aggregate_run import aggregate  # noqa: E402
from run_training import build_config, build_env_config, cross_product_cells, env_spec, parse_args  # noqa: E402
from summarize_episode_records import summarize_run  # noqa: E402

BASE = ["--run-dir", "/tmp/study-options-test", "--unit-id", "test"]
POINTMAZE_EXAMPLE = ROOT / "configs/experiments" / (
    "pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml")
ANTMAZE_EXAMPLE = ROOT / "configs/experiments" / (
    "antmaze_umaze_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml")


def built(*options):
    """The arguments, trainer configuration and environment configuration of one command line."""
    arguments = parse_args([*BASE, *options])
    return arguments, build_config(arguments), build_env_config(arguments)


def refused(*options) -> str:
    """The message with which one command line is refused; fails if it is accepted."""
    try:
        built(*options)
    except ValueError as error:
        return str(error)
    raise AssertionError(f"accepted: {options}")


def test_cleanrl_values_reach_the_configuration():
    """The clip coefficient and the number of priming rollouts are options; defaults are kept."""
    _, config, _ = built()
    assert (config.clip_coef, config.prime_iterations) == (0.2, 10)
    assert not config.episode_log and not config.gather_minibatches_per_epoch
    assert (config.ext_reward_scale, config.ext_reward_offset) == (1.0, 0.0)
    assert config.setting_names == ("learning_rate", "int_coef") and not config.parameter_settings
    _, config, _ = built("--clip-coef", "0.1", "--prime-iterations", "50",
                         "--gather-minibatches-per-epoch", "--update-style", "epoch_minibatch")
    assert (config.clip_coef, config.prime_iterations) == (0.1, 50)
    assert config.gather_minibatches_per_epoch
    print("PASS clip coefficient and priming rollouts reach the configuration")


def test_sweeps_over_more_settings_become_named_cells():
    """Lists of entropy, clip and value coefficients cross with rates and weights, by name."""
    names, cells = cross_product_cells({"learning_rate": [1e-4], "int_coef": [],
                                        "ent_coef": [0.001, 0.01]}, ((1.0, 0.0),))
    assert names == ("learning_rate", "ent_coef") and cells == ((1e-4, 0.001), (1e-4, 0.01))

    _, config, _ = built("--learning-rates", "1e-4,3e-4", "--entropy-coefficients", "0.001,0.01",
                         "--copies-per-cell", "2")
    assert config.setting_names == ("learning_rate", "ent_coef")
    assert config.parameter_settings == ((1e-4, 0.001), (1e-4, 0.01), (3e-4, 0.001), (3e-4, 0.01))
    assert config.n_copies == 8 and config.copies_per_group == 2

    # a list of one name only leaves the learning rate and the intrinsic weight as scalars
    _, config, _ = built("--clip-coefficients", "0.1,0.2", "--value-coefficients", "0.5,1",
                         "--copies-per-cell", "1", "--learning-rate", "1e-4",
                         "--intrinsic-coefficient", "4")
    assert config.setting_names == ("clip_coef", "vf_coef") and config.n_copies == 4
    assert (config.learning_rate, config.int_coef) == (1e-4, 4.0)

    # the earlier forms build exactly what they did
    _, config, _ = built("--learning-rates", "1e-4,3e-4", "--intrinsic-weights", "1,10",
                         "--copies-per-cell", "2")
    assert config.learning_rates == (1e-4, 3e-4) and config.betas == (1.0, 10.0)
    assert not config.parameter_settings
    assert "copies-per-cell" in refused("--entropy-coefficients", "0.001,0.01")
    print("PASS sweeps over entropy, clip and value coefficients become named cells")


def test_reward_settings_and_goal_termination():
    """A named reward setting ends the episode at the goal and fixes the scale and offset."""
    # PointMaze, each setting alone: the same environment, a scalar scale and offset
    _, config, env = built("--reward-setting", "arrival_plus_one", "--episode-log")
    assert not env.continuing_task and (env.reward_shift, env.goal_reward) == (0.0, "per_step")
    assert (config.ext_reward_scale, config.ext_reward_offset) == (1.0, 0.0)
    assert env_spec(env, "arrival_plus_one") == (
        "pointmaze_large_term400_nonoise_reward-arrival-plus-one@1")
    arguments, config, env = built("--reward-setting", "step_cost", "--episode-log")
    assert (config.ext_reward_scale, config.ext_reward_offset) == (1 / 400, -1 / 400)
    assert env_spec(env, arguments.reward_setting) == (
        "pointmaze_large_term400_nonoise_reward-step-cost@1")
    assert run_training.specs_of("rnd_next_state", config, env, "step_cost")["env"].endswith(
        "_reward-step-cost@1")

    # both settings in one process: every cell once per reward setting, on paired seeds
    _, config, _ = built("--reward-setting", "arrival_plus_one_and_step_cost", "--episode-log",
                         "--intrinsic-weights", "1,4", "--copies-per-cell", "3",
                         "--episode-steps", "200")
    assert config.setting_names == ("int_coef", "ext_reward_scale", "ext_reward_offset")
    assert config.parameter_settings == ((1.0, 1.0, 0.0), (1.0, 1 / 200, -1 / 200),
                                         (4.0, 1.0, 0.0), (4.0, 1 / 200, -1 / 200))
    assert config.n_copies == 12 and config.sweep_seed_mode == "paired"

    # AntMaze: the episode may end at the goal, under the current rules or the historical ones
    _, config, env = built("--env", "antmaze-umaze", "--terminate-at-goal", "--episode-log")
    assert not env.continuing_task and config.configuration_preset == "antmaze_historical"
    assert env_spec(env) == "antmaze_umaze_term700_nonoise@1"
    arguments, config, env = built("--env", "antmaze-medium", "--reward-setting", "step_cost",
                                   "--configuration-preset", "antmaze_current", "--episode-log")
    assert not env.continuing_task and env.max_episode_steps == 1000
    assert (config.ext_reward_scale, config.ext_reward_offset) == (1 / 1000, -1 / 1000)
    assert config.intrinsic_bootstrap_observation == "reset_observation"
    assert env_spec(env, arguments.reward_setting) == (
        "antmaze_medium_term1000_nonoise_reward-step-cost@1")
    # the environments of earlier runs keep their names
    _, _, env = built("--env", "antmaze-umaze")
    assert env.continuing_task and env_spec(env) == "antmaze_umaze_cont700_nonoise@1"

    # refused: an episode that ends at the goal with nothing counting its episodes, a named
    # setting on a shifted reward, and goal termination where there is no goal
    assert "--episode-log" in refused("--reward-setting", "step_cost")
    assert "--episode-log" in refused("--env", "antmaze-umaze", "--terminate-at-goal")
    assert "--reward-shift" in refused("--reward-setting", "step_cost", "--episode-log",
                                       "--reward-shift", "-1")
    assert "PointMaze and AntMaze" in refused("--env", "montezuma", "--reward-setting",
                                              "arrival_plus_one", "--episode-log")
    print("PASS reward settings, goal termination on both families, environment names")


def test_example_files_start_both_reward_settings():
    """One example file per family; `--reward-setting step_cost` turns it into the other one."""
    for path, limit, preset in ((POINTMAZE_EXAMPLE, 400, "pointmaze_current"),
                                (ANTMAZE_EXAMPLE, 700, "antmaze_current")):
        arguments, config, env = built("--configuration-file", str(path))
        assert arguments.reward_setting == "arrival_plus_one" and not env.continuing_task
        assert (config.configuration_preset, config.update_style) == (preset, "epoch_minibatch")
        assert (config.clip_coef, config.learning_rate, config.ent_coef,
                config.prime_iterations, config.episode_log) == (0.1, 1e-4, 0.001, 50, True)
        assert (config.ext_reward_scale, config.ext_reward_offset) == (1.0, 0.0)
        arguments, config, env = built("--configuration-file", str(path),
                                       "--reward-setting", "step_cost")
        assert (config.ext_reward_scale, config.ext_reward_offset) == (1 / limit, -1 / limit)
        assert env_spec(env, arguments.reward_setting).endswith("_reward-step-cost@1")
        # both settings at once: the file's single configuration becomes two paired cells
        _, config, _ = built("--configuration-file", str(path), "--reward-setting",
                             "arrival_plus_one_and_step_cost", "--copies-per-cell", "16")
        assert config.setting_names == ("ext_reward_scale", "ext_reward_offset")
        assert config.parameter_settings == ((1.0, 0.0), (1 / limit, -1 / limit))
        assert config.n_copies == 32 and config.learning_rate == 1e-4
    print("PASS the example files start reward setting A, B, and both")


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


def test_episode_log_records_and_resume_from_a_checkpoint():
    """A unit stopped twice and continued equals the same unit run in one go, record by record."""
    options = ["--unit-id", "unit", "--copies-per-cell", "2", "--envs-per-copy", "2",
               "--rollout-steps", "8", "--iterations", "7", "--window-iterations", "3",
               "--episode-steps", "7", "--update-style", "epoch_minibatch", "--clip-coef", "0.1",
               "--prime-iterations", "1", "--episode-log", "--track-coverage",
               "--reward-setting", "arrival_plus_one_and_step_cost"]
    with tempfile.TemporaryDirectory() as directory:
        whole, parts = Path(directory) / "whole", Path(directory) / "parts"
        run_command("--run-dir", str(whole), *options)

        # stopped after its first iteration, then after its second, then run to the end
        for attempt in range(2):
            try:
                run_command("--run-dir", str(parts), *options, "--time-limit-seconds", "1e-9")
                raise AssertionError("the time limit did not stop the unit")
            except SystemExit as stop:
                assert "run the same command again" in str(stop)
            assert (parts / "checkpoints" / "unit.npz").exists()
        run_command("--run-dir", str(parts), *options, "--checkpoint-seconds", "1e-9")
        assert not (parts / "checkpoints" / "unit.npz").exists(), "a complete unit kept its save"

        expected, actual = records_of(whole, "unit"), records_of(parts, "unit")
        kinds = lambda records: [record["record"] for record in records]
        assert kinds(expected) == ["job", "unit_start", *["episode_window"] * 3, "unit_complete"]
        assert kinds(actual) == ["job", "unit_start", "job", "unit_resume", "job", "unit_resume",
                                 *["episode_window"] * 3, "unit_complete"]
        resumes = [record for record in actual if record["record"] == "unit_resume"]
        assert [record["resumed_from_iteration"] for record in resumes] == [1, 2]
        assert actual[-1]["resumed_from_iteration"] == 2 and actual[-1]["iterations_this_attempt"] == 5

        # the unit's identity card names the cells' columns and the reward setting
        start = expected[1]
        assert start["environment"] == (
            "pointmaze_large_term7_nonoise_reward-arrival-plus-one-and-step-cost@1")
        assert start["cell_setting_names"] == ["ext_reward_scale", "ext_reward_offset"]
        assert start["cell_settings"] == [[1.0, 0.0], [1 / 7, -1 / 7]]
        assert start["copy_seed_index"] == [0, 1, 0, 1] and start["episode_log"] is True
        assert start["episodes_per_copy_per_iteration"] is None

        # every number of every window is the same in the two runs
        windows = lambda records: [r for r in records if r["record"] == "episode_window"]
        timing = {"written_at", "seconds_since_first_iteration"}
        for one, other in zip(windows(expected), windows(actual)):
            assert {k: v for k, v in one.items() if k not in timing} == {
                k: v for k, v in other.items() if k not in timing}, one["window_index"]
        assert [w["last_iteration"] for w in windows(actual)] == [3, 6, 7]

        # the log's fields: an episode ends after 7 steps (no copy reaches the goal this soon),
        # so 7 iterations x 8 steps hold 8 episodes per environment, 16 per copy; the copies paid
        # 1 on arrival have return 0, and those with the step cost have -1 an episode
        last = windows(actual)[-1]
        assert last["phase_blocked"] is None and last["episodes_per_copy_in_window"] is None
        total = lambda name: [sum(w[name][c] for w in windows(actual)) for c in range(4)]
        assert total("episodes_per_copy") == [16, 16, 16, 16], total("episodes_per_copy")
        assert total("length_sum_per_copy") == [112, 112, 112, 112]
        assert total("arrivals_per_copy") == [0, 0, 0, 0]
        assert last["episodes_per_copy"] == [4, 4, 4, 4]
        assert last["first_goal_step_per_copy"] == [-1, -1, -1, -1]
        assert total("return_sum_per_copy")[:2] == [0.0, 0.0]
        assert all(abs(value + 16) < 1e-9 for value in total("return_sum_per_copy")[2:])
        assert total("environment_return_sum_per_copy") == [0.0, 0.0, 0.0, 0.0]

        # the run-level files and the episode summary read these records
        summary = aggregate(parts)
        assert summary["units_complete"] == 1 and summary["records"] == 3
        assert summary["first_goal"] == {"copies": 4, "copies_that_reached_the_goal": 0,
                                         "mean_capped_first_goal_step": 7 * 8 * 2}
        returns = summary["last_window_episode_return_per_copy"]
        assert returns["max"] == 0.0 and abs(returns["min"] + 1) < 1e-9
        cells = summarize_run(parts)
        assert len(cells) == 2 and all(cell["complete"] for cell in cells.values())
        step_cost = next(cell for key, cell in cells.items() if "ext_reward_offset=-" in key)
        assert abs(step_cost["final"] + 1) < 1e-9 and step_cost["goal_discovery_rate"] == 0.0
        assert step_cost["episode_length_final"] == 7.0
    print("PASS episode-log records; a unit stopped twice and continued equals the whole unit")


def test_example_files_run_through_the_command():
    """Each family's example file starts a unit: PointMaze with the step cost, AntMaze with
    reward 1 on arrival, cut down to two policies and two iterations."""
    short = ["--copies", "2", "--iterations", "2", "--window-iterations", "1",
             "--prime-iterations", "1", "--rollout-steps", "16"]
    cases = ((POINTMAZE_EXAMPLE, ["--reward-setting", "step_cost"], "pointmaze_current",
              "pointmaze_large_term400_nonoise_reward-step-cost@1", (1 / 400, -1 / 400)),
             (ANTMAZE_EXAMPLE, [], "antmaze_current",
              "antmaze_umaze_term700_nonoise_reward-arrival-plus-one@1", (1.0, 0.0)))
    for path, reward, preset, environment, mapping in cases:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_command("--run-dir", str(run_dir), "--unit-id", "unit",
                        "--configuration-file", str(path), *reward, *short)
            records = records_of(run_dir, "unit")
            assert [r["record"] for r in records] == [
                "job", "unit_start", "episode_window", "episode_window", "unit_complete"]
            start, last = records[1], records[-2]
            trainer = start["trainer_config"]
            assert start["environment"] == environment and start["copies"] == 2
            assert start["configuration_file"]["path"] == str(path)
            assert (trainer["configuration_preset"], trainer["update_style"]) == (
                preset, "epoch_minibatch")
            assert (trainer["clip_coef"], trainer["learning_rate"], trainer["ent_coef"],
                    trainer["prime_iterations"], trainer["episode_log"]) == (
                0.1, 1e-4, 0.001, 1, True)
            assert (trainer["ext_reward_scale"], trainer["ext_reward_offset"]) == mapping
            # 2 iterations x 16 steps x 4 environments: no episode has ended yet
            assert last["env_steps_per_copy"] == 2 * 16 * 4
            assert last["episodes_per_copy"] == [0, 0] and last["arrivals_per_copy"] == [0, 0]
            assert len(last["coverage_per_copy"]) == 2
            manifest = (run_dir / "manifest.yaml").read_text()
            assert f"env: {environment}" in manifest and f"configuration_preset: {preset}" in manifest
    print("PASS both example files start a unit through the command")


if __name__ == "__main__":
    test_cleanrl_values_reach_the_configuration()
    test_sweeps_over_more_settings_become_named_cells()
    test_reward_settings_and_goal_termination()
    test_example_files_start_both_reward_settings()
    test_episode_log_records_and_resume_from_a_checkpoint()
    test_example_files_run_through_the_command()
