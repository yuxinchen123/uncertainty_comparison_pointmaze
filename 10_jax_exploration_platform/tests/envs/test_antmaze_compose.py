"""AntMaze through the whole fused trainer: compose, prime, iterate, and refuse what must fail.

Checks that the platform's one-program iteration — rollout, bonus, advantages, update — compiles
and runs with the MJX environment inside it, for the bonuses that are defined on it, and that
the visit-count family (a PointMaze-only bonus) is refused at composition time rather than
producing wrong numbers.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_antmaze_compose.py
"""
import importlib.util
import sys
from pathlib import Path

from dataclasses import replace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401
import numpy as np  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

SMALL = dict(n_copies=2, n_envs=2, num_steps=8, prime_iterations=1, track_coverage=True)


def test_rnd_trains_end_to_end():
    """Two full iterations with the distillation bonus: finite loss, finite rewards, coverage."""
    runner = Runner(PPOConfig(**SMALL), bonus="rnd_next_state", env_cfg=preset("umaze"))
    state, stats = runner.train(2, log_every_seconds=0, history_every=1)
    assert np.isfinite(stats["history"][-1]["reward_ext_sum_per_copy"]).all()
    assert all(np.isfinite(r["rint_mean_per_copy"]).all() for r in stats["history"])
    cov = np.asarray(stats["history"][-1]["coverage_per_copy"])
    # 2 iterations x 8 steps never leave the start cell's neighbourhood, but the start cell
    # itself must be counted
    assert (cov > 0).all(), cov
    print("ok test_rnd_trains_end_to_end")


def test_none_trains_end_to_end():
    """The no-bonus arm composes and runs on the MJX environment too."""
    runner = Runner(PPOConfig(**SMALL), bonus="none", env_cfg=preset("umaze"))
    state, stats = runner.train(2, log_every_seconds=0, history_every=2)
    assert np.isfinite(stats["history"][-1]["reward_ext_sum_per_copy"]).all()
    assert all(v == 0.0 for v in stats["history"][-1]["rint_mean_per_copy"])
    print("ok test_none_trains_end_to_end")


def test_visit_count_is_refused():
    """The PointMaze-only visit-count bonus fails at composition, with a plain message."""
    try:
        Runner(PPOConfig(**SMALL), bonus="gt_position_velocity_sqrt", env_cfg=preset("umaze"))
    except ValueError as e:
        assert "not defined for this environment" in str(e), e
        print("ok test_visit_count_is_refused")
        return
    raise AssertionError("composing visit-count with AntMaze did not raise")


def test_the_physics_field_selects_the_environment_class():
    """`physics` picks the stepper: "jax" (the default) builds JaxAntMaze as before, "warp"
    builds WarpAntMaze where warp-lang is installed and is refused with the environment's name
    where it is not; any other value is refused."""
    from exploration_platform.training.compose import environment_class
    assert preset("umaze").physics == "jax"
    assert environment_class(preset("umaze")).__name__ == "JaxAntMaze"
    assert environment_class(replace(preset("umaze"), physics="warp")).__name__ == "WarpAntMaze"
    try:
        environment_class(replace(preset("umaze"), physics="mjx"))
        raise AssertionError("an unknown physics was accepted")
    except ValueError as error:
        assert "unknown AntMaze physics" in str(error), error
    if importlib.util.find_spec("warp") is None:
        try:
            Runner(PPOConfig(**SMALL), bonus="none", env_cfg=replace(preset("umaze"), physics="warp"))
            raise AssertionError("the Warp physics was built without warp-lang")
        except RuntimeError as error:
            assert "platform_jax_warp" in str(error), error
    print("ok test_the_physics_field_selects_the_environment_class")


def test_current_rules_train_with_goal_termination_and_the_episode_log():
    """The study's AntMaze program: current rules, clipped update, the episode ends at the goal
    or after 6 steps, the step cost for one of two paired copies, and the episode log."""
    env_cfg = replace(preset("umaze"), continuing_task=False, max_episode_steps=6)
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=2, n_envs=2, num_steps=8,
                    prime_iterations=1, update_style="epoch_minibatch", clip_coef=0.1,
                    ent_coef=0.001, episode_log=True, gather_minibatches_per_epoch=True,
                    copies_per_group=1, setting_names=("ext_reward_scale", "ext_reward_offset"),
                    parameter_settings=((1.0, 0.0), (1 / 6, -1 / 6)))
    runner = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
    assert runner.cfg.intrinsic_bootstrap_observation == "reset_observation"
    assert runner.cfg.no_bonus_advantage_stream == "task_only"
    state = runner.prime(runner.init_state(run_seed=0))
    for k in range(1, 3):
        state, metrics = runner.iterate(state, runner.lr_argument(k, 2))
    assert np.isfinite(np.asarray(metrics["loss"]))
    log = {name: np.asarray(value) for name, value in state.agent_state["episode_log"].items()}
    # 2 iterations x 8 steps x 2 environments, an episode at most 6 steps: 2 episodes ended per
    # environment, none of them at the goal, each of 6 steps
    np.testing.assert_array_equal(log["episodes"], [4, 4])
    np.testing.assert_array_equal(log["length_sum"], [24, 24])
    np.testing.assert_array_equal(log["arrivals"], [0, 0])
    np.testing.assert_array_equal(log["first_goal_step"], [-1, -1])
    # the two copies share a seed and differ in the reward, so their policies have parted
    actor = np.asarray(state.agent_params["actor"]["W0"])
    assert np.isfinite(actor).all() and (actor[0] != actor[1]).any()
    print("ok test_current_rules_train_with_goal_termination_and_the_episode_log")


if __name__ == "__main__":
    test_rnd_trains_end_to_end()
    test_none_trains_end_to_end()
    test_visit_count_is_refused()
    test_current_rules_train_with_goal_termination_and_the_episode_log()
