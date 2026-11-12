"""The trainer and the training command with the Warp AntMaze physics, MuJoCo Warp on processors.

Checked here: a small run through `Runner` and through `scripts/run_training.py` with
`physics = "warp"` (goal termination, the episode log, both reward settings), the unit's records
of the shared contact pool and the kernel folder in its log, a unit that stops when MuJoCo Warp
drops a contact, and a unit that is stopped and continued from its checkpoint.

On processors MuJoCo Warp is deterministic, so the stopped and continued unit is compared with
the uninterrupted one bit for bit: every number of every window record, and every array of the
training state except the 14 arrays of constraint-island bookkeeping that a physics step of this
model returns unwritten (`warp_antmaze.ISLAND_ARRAYS`; they differ between any two runs, also
without a stop, and nothing reads them). On a card the contacts enter the shared pool in an order
that differs from run to run, so two runs of the same unit differ there even without a stop; that
is not tested here.

Needs an interpreter with warp-lang (`/p/rlprojects/RND/.venvs/platform_jax_warp`); the canonical
`platform_jax` has none, so this file is skipped there.

Run on a compute node, never on a login node:
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax_warp python> -m pytest -q \
      tests/envs/test_antmaze_warp_training.py
"""
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("warp")
ROOT = Path(__file__).resolve().parents[2]
for folder in ("src", "scripts", "tests/agents"):
    sys.path.insert(0, str(ROOT / folder))
import numpy as np  # noqa: E402
import run_training  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import ISLAND_ARRAYS  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from run_training import build_env_config, env_spec, parse_args  # noqa: E402
from test_checkpoint_resume import check_resume  # noqa: E402

WARP_EXAMPLE = ROOT / "configs/experiments" / (
    "antmaze_umaze_warp-physics_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one"
    "_example_steps-204800.toml")
# the study's learner at a size a processor runs in seconds: current rules, clipped update
STUDY = dict(configuration_preset="antmaze_current", n_copies=2, n_envs=2, num_steps=8,
             prime_iterations=1, update_style="epoch_minibatch", clip_coef=0.1, ent_coef=0.001,
             episode_log=True, gather_minibatches_per_epoch=True, copies_per_group=1,
             setting_names=("ext_reward_scale", "ext_reward_offset"))
SHORT = ["--copies", "2", "--iterations", "2", "--window-iterations", "1",
         "--prime-iterations", "1", "--rollout-steps", "16"]


def warp_umaze(**changes):
    """The U-maze with the Warp physics and the episode ending at the goal."""
    return replace(preset("umaze"), physics="warp", continuing_task=False, **changes)


def trained(env_cfg, step_limit, iterations=2):
    """Prime and train the small study program; returns (runner, state, last metrics, episode log)."""
    cfg = PPOConfig(**STUDY, parameter_settings=((1.0, 0.0), (1 / step_limit, -1 / step_limit)))
    runner = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
    state = runner.prime(runner.init_state(run_seed=0))
    for k in range(1, iterations + 1):
        state, metrics = runner.iterate(state, runner.lr_argument(k, iterations))
    log = {name: np.asarray(value) for name, value in state.agent_state["episode_log"].items()}
    return runner, state, metrics, log


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


def test_runner_trains_with_the_step_limit_the_episode_log_and_both_reward_settings():
    """Two iterations of the study's program on the Warp environment: one copy paid 1 on
    arrival, its paired copy the step cost; episodes end at the 6-step limit."""
    runner, state, metrics, log = trained(warp_umaze(max_episode_steps=6), step_limit=6)
    assert type(runner.env).__name__ == "WarpAntMaze" and runner.env.cfg.physics == "warp"
    assert np.isfinite(np.asarray(metrics["loss"]))
    assert np.isfinite(np.asarray(state.obs)).all()
    # 2 iterations x 8 steps x 2 environments, an episode at most 6 steps: 2 episodes ended per
    # environment, none of them at the goal, each of 6 steps
    np.testing.assert_array_equal(log["episodes"], [4, 4])
    np.testing.assert_array_equal(log["length_sum"], [24, 24])
    np.testing.assert_array_equal(log["arrivals"], [0, 0])
    np.testing.assert_array_equal(log["first_goal_step"], [-1, -1])
    # the pool is sized from the batch (4 environments x 48 slots) and nothing was dropped
    report = runner.check_environment(state)
    assert (report["naconmax"], report["njmax"], report["dropped"]) == (4 * 48, 160, False)
    assert int(np.asarray(state.env_state.nan_count).sum()) == 0


def test_runner_ends_the_episode_at_the_goal():
    """With the goal on the start cell every step arrives: each episode is one step long, ends
    by termination and pays the environment's reward of 1."""
    runner, state, metrics, log = trained(warp_umaze(goal_cell=(3, 1)), step_limit=700)
    assert np.isfinite(np.asarray(metrics["loss"]))
    # 2 iterations x 8 steps x 2 environments = 32 one-step episodes per copy, all arrivals
    np.testing.assert_array_equal(log["episodes"], [32, 32])
    np.testing.assert_array_equal(log["arrivals"], [32, 32])
    np.testing.assert_array_equal(log["length_sum"], [32, 32])
    np.testing.assert_array_equal(log["return_sum"], [32.0, 32.0])
    assert (log["first_goal_step"] >= 0).all()
    assert not runner.check_environment(state)["dropped"]


def test_a_pool_below_the_need_stops_the_run_with_a_message():
    """One contact slot per environment is too few once the ants stand on the floor: the check
    the trainer makes after priming and at every window end raises and says what to change."""
    env_cfg = warp_umaze(warp_contact_slots_per_environment=1)
    cfg = PPOConfig(**{**STUDY, "num_steps": 16},
                    parameter_settings=((1.0, 0.0), (1 / 700, -1 / 700)))
    runner = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
    assert runner.env.naconmax == 4
    state = runner.prime(runner.init_state(run_seed=0))
    state, _ = runner.iterate(state, runner.lr_argument(1, 1))
    with pytest.raises(RuntimeError, match="dropped part of the physics") as stop:
        runner.check_environment(state)
    assert "warp_contact_slots_per_environment" in str(stop.value)
    assert "shared pool of 4 slots (1 per environment x 4 environments)" in str(stop.value)
    with pytest.raises(RuntimeError, match="dropped part of the physics"):
        runner.train(1, log_every_seconds=0)


def test_the_option_and_the_file_choose_the_physics_and_name_it():
    """`--physics warp`, or a file whose [environment] says so, gives the Warp configuration and
    a specification name that says which implementation ran; `--physics jax` takes it back."""
    base = ["--run-dir", "/tmp/warp-option-test", "--unit-id", "test"]
    env = build_env_config(parse_args([*base, "--env", "antmaze-umaze", "--physics", "warp",
                                       "--reward-setting", "step_cost", "--episode-log"]))
    assert env.physics == "warp" and not env.continuing_task
    assert env_spec(env, "step_cost") == "antmaze_umaze_term700_nonoise_physics-warp_reward-step-cost@1"
    arguments = parse_args([*base, "--configuration-file", str(WARP_EXAMPLE)])
    env = build_env_config(arguments)
    assert (arguments.physics, env.physics) == ("warp", "warp")
    assert env_spec(env, arguments.reward_setting) == (
        "antmaze_umaze_term700_nonoise_physics-warp_reward-arrival-plus-one@1")
    env = build_env_config(parse_args([*base, "--configuration-file", str(WARP_EXAMPLE),
                                       "--physics", "jax"]))
    assert env.physics == "jax"
    assert env_spec(env, "arrival_plus_one") == "antmaze_umaze_term700_nonoise_reward-arrival-plus-one@1"


def test_the_command_runs_the_warp_example_in_both_reward_settings():
    """The Warp example file starts a unit with reward 1 on arrival, and with the step cost when
    the option is added; the records name the physics and hold the pool use, the log the kernel
    folder."""
    cases = (([], "antmaze_umaze_term700_nonoise_physics-warp_reward-arrival-plus-one@1", (1.0, 0.0)),
             (["--reward-setting", "step_cost"],
              "antmaze_umaze_term700_nonoise_physics-warp_reward-step-cost@1", (1 / 700, -1 / 700)))
    for reward, environment, mapping in cases:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_command("--run-dir", str(run_dir), "--unit-id", "unit",
                        "--configuration-file", str(WARP_EXAMPLE), *reward, *SHORT)
            records = records_of(run_dir, "unit")
            assert [r["record"] for r in records] == [
                "job", "unit_start", "episode_window", "episode_window", "unit_complete"]
            start, last, complete = records[1], records[-2], records[-1]
            assert start["environment"] == environment
            assert start["environment_config"]["physics"] == "warp"
            assert start["configuration_file"]["path"] == str(WARP_EXAMPLE)
            trainer = start["trainer_config"]
            assert (trainer["configuration_preset"], trainer["update_style"],
                    trainer["episode_log"]) == ("antmaze_current", "epoch_minibatch", True)
            assert (trainer["ext_reward_scale"], trainer["ext_reward_offset"]) == mapping
            # 2 iterations x 16 steps x 4 environments: no episode has ended yet
            assert last["env_steps_per_copy"] == 2 * 16 * 4
            assert last["episodes_per_copy"] == [0, 0] and last["arrivals_per_copy"] == [0, 0]
            # the pool: 2 policies x 4 environments x 48 slots; its largest use grows or stays
            pools = [r["physics_pool"] for r in records[2:]]
            assert all(pool["pool_slots"] == 2 * 4 * 48 and pool["constraint_row_limit"] == 160
                       for pool in pools)
            assert 0 < pools[0]["candidate_pairs_largest"] <= pools[1]["candidate_pairs_largest"]
            assert complete["physics_pool"] == pools[1]
            assert 0 < complete["physics_pool"]["pool_fraction_largest"] < 1
            assert complete["nan_respawned_episodes"] == 0
            log = (run_dir / "logs" / "unit.log").read_text()
            assert "MuJoCo Warp physics: kernel folder " in log
            assert "shared contact pool 384 slots (48 per environment x 8 environments)" in log
            assert f"env: {environment}" in (run_dir / "manifest.yaml").read_text()


def test_the_command_stops_a_unit_whose_pool_is_too_small():
    """A unit in which MuJoCo Warp dropped a contact ends with an error, says why in its own
    log, and leaves no window record, no completion record and no checkpoint."""
    with tempfile.TemporaryDirectory() as directory:
        small = Path(directory) / "small_pool.toml"
        small.write_text(WARP_EXAMPLE.read_text().replace(
            "warp_contact_slots_per_environment = 48", "warp_contact_slots_per_environment = 1"))
        run_dir = Path(directory) / "run"
        with pytest.raises(RuntimeError, match="dropped part of the physics"):
            run_command("--run-dir", str(run_dir), "--unit-id", "unit", "--configuration-file",
                        str(small), *SHORT, "--checkpoint-seconds", "1e-9")
        kinds = [record["record"] for record in records_of(run_dir, "unit")]
        assert "episode_window" not in kinds and "unit_complete" not in kinds
        assert not (run_dir / "checkpoints" / "unit.npz").exists()
        assert "STOPPED: MuJoCo Warp dropped part of the physics" in (
            run_dir / "logs" / "unit.log").read_text()


def test_state_saved_and_restored_continues_bit_for_bit():
    """The whole training state with the Warp data tree in it: saved after 1 of 3 iterations and
    restored into a newly built trainer. The restored state equals the saved one in all 257
    arrays; after two more iterations 243 arrays equal the uninterrupted unit's bit for bit, and
    the 14 left out are the island arrays the physics step never writes."""
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=1, num_steps=4,
                    prime_iterations=1, update_style="epoch_minibatch", num_minibatches=2,
                    episode_log=True)
    check_resume(cfg, warp_umaze(max_episode_steps=6), "rnd_next_state", total=3, stop_after=1,
                 unwritten=ISLAND_ARRAYS)


def test_a_unit_stopped_twice_and_continued_equals_the_whole_unit():
    """Through the command: stopped after its first and after its second iteration, then run to
    its end; every number of every window record equals the unit run in one go."""
    options = ["--unit-id", "unit", "--env", "antmaze-umaze", "--physics", "warp",
               "--configuration-preset", "antmaze_current", "--copies-per-cell", "1",
               "--envs-per-copy", "2", "--rollout-steps", "8", "--iterations", "5",
               "--window-iterations", "3", "--episode-steps", "7", "--update-style",
               "epoch_minibatch", "--clip-coef", "0.1", "--prime-iterations", "1",
               "--episode-log", "--track-coverage", "--reward-setting",
               "arrival_plus_one_and_step_cost"]
    with tempfile.TemporaryDirectory() as directory:
        whole, parts = Path(directory) / "whole", Path(directory) / "parts"
        run_command("--run-dir", str(whole), *options)
        for attempt in range(2):
            with pytest.raises(SystemExit, match="run the same command again"):
                run_command("--run-dir", str(parts), *options, "--time-limit-seconds", "1e-9")
            assert (parts / "checkpoints" / "unit.npz").exists()
        run_command("--run-dir", str(parts), *options, "--checkpoint-seconds", "1e-9")
        assert not (parts / "checkpoints" / "unit.npz").exists(), "a complete unit kept its save"

        expected, actual = records_of(whole, "unit"), records_of(parts, "unit")
        kinds = lambda records: [record["record"] for record in records]
        assert kinds(expected) == ["job", "unit_start", *["episode_window"] * 2, "unit_complete"]
        assert kinds(actual) == ["job", "unit_start", "job", "unit_resume", "job", "unit_resume",
                                 *["episode_window"] * 2, "unit_complete"]
        assert expected[1]["environment"] == (
            "antmaze_umaze_term7_nonoise_physics-warp_reward-arrival-plus-one-and-step-cost@1")
        windows = lambda records: [r for r in records if r["record"] == "episode_window"]
        timing = {"written_at", "seconds_since_first_iteration"}
        for one, other in zip(windows(expected), windows(actual)):
            assert {k: v for k, v in one.items() if k not in timing} == {
                k: v for k, v in other.items() if k not in timing}, one["window_index"]
        assert [w["last_iteration"] for w in windows(actual)] == [3, 5]
        # 5 iterations x 8 steps, an episode of 7 steps: 5 episodes ended per environment
        total = lambda name: [sum(w[name][c] for w in windows(actual)) for c in range(2)]
        assert total("episodes_per_copy") == [10, 10] and total("length_sum_per_copy") == [70, 70]
        # the largest pool use is part of the state, so it continues across the stops as well
        assert actual[-1]["physics_pool"] == expected[-1]["physics_pool"]
        assert actual[-1]["resumed_from_iteration"] == 2
