"""Arrival ends the episode with no bootstrap, the step limit bootstraps, the intrinsic return runs
across both, and the two reward settings pay what they say: hand-built rollouts in both families.

One program holds two copies on the same seed. Copy 0 sees the environment's reward (1 on arrival,
0 elsewhere); copy 1 sees it mapped by (r - 1) / limit (0 on arrival, -1 / limit elsewhere). In
each copy, environment 0 is placed on the goal, so its first step arrives, and environment 1 is
two steps short of the step limit, so its second step is a truncation. The update batch the
trainer builds is compared with an explicit reverse-time loop over a replay of the same actions.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.networks import critic_values  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

COPIES, ENVS, STEPS = 2, 2, 4


def reference_returns(rewards, values, next_values, gamma, gae_lambda, stop):
    """Returns by an explicit reverse-time loop, independent of the trainer's scan.

    rewards, values, next_values, stop: [T, C, N]. `stop` is 1 where the advantage sum must not
    continue past the step (an episode end, for the extrinsic stream).
    """
    advantage = np.zeros_like(values[0])
    returns = np.empty_like(values)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] - values[step]
        advantage = delta + gamma * gae_lambda * (1.0 - stop[step]) * advantage
        returns[step] = advantage + values[step]
    return returns


def hand_built_state(runner, state, limit: int):
    """Put environment 0 of every copy on the goal and environment 1 two steps before the limit.

    Returns (state with that environment state and its observation, the observation of a fresh
    episode's first step [obs]).
    """
    env = runner.env
    step_count = np.zeros((COPIES, ENVS), np.int32)
    step_count[:, 1] = limit - 2
    if hasattr(state.env_state, "pos"):
        # PointMaze: positions [C, N, 2]; the ball rests on the goal's centre
        pos = np.array(state.env_state.pos)
        spawn_obs = np.concatenate([pos[0, 0], np.zeros(2, np.float32)])
        pos[:, 0] = env.goal_center
        env_state = state.env_state._replace(pos=jnp.asarray(pos),
                                             step_count=jnp.asarray(step_count))
        obs = jnp.concatenate([env_state.pos, env_state.vel], -1)
    else:
        # AntMaze: joint positions [C * N, 15], row c * N + n; the torso's xy moves to the goal
        qpos = np.array(state.env_state.data.qpos)
        spawn_obs = np.concatenate([np.asarray(env.init_qpos), np.asarray(env.init_qvel)])
        qpos[0::ENVS, 0:2] = np.asarray(env.goal)
        data = state.env_state.data.replace(qpos=jnp.asarray(qpos))
        env_state = state.env_state._replace(data=data, step_count=jnp.asarray(step_count))
        obs = env._obs(data)
    return state._replace(env_state=env_state, obs=obs), spawn_obs


def check_family(configuration_preset: str, env_cfg) -> None:
    """Build the two-copy program for one family and check its batch and its episode log."""
    limit = env_cfg.max_episode_steps
    scale, offset = np.array([1.0, 1 / limit]), np.array([0.0, -1 / limit])
    cfg = PPOConfig(configuration_preset=configuration_preset, n_copies=COPIES, n_envs=ENVS,
                    num_steps=STEPS, prime_iterations=0, update_style="full_batch",
                    copies_per_group=1, episode_log=True,
                    setting_names=("ext_reward_scale", "ext_reward_offset"),
                    parameter_settings=tuple(zip(scale.tolist(), offset.tolist())))
    runner = Runner(cfg, env_cfg=env_cfg)
    assert runner.cfg.intrinsic_bootstrap_observation == "reset_observation"
    assert runner.sweep.copy_seed_index == [0, 0], "the two reward settings must share a seed"
    state, spawn_obs = hand_built_state(runner, runner.prime(runner.init_state(run_seed=17)),
                                        limit)
    params = state.agent_params
    after, _, batch = jax.jit(runner.composition.iteration_capturing_batch())(
        state, jnp.asarray(1e-4, jnp.float32))

    # before: batch fields [C, T * N, ...]; after: [T, C, N, ...], one entry per rollout step
    by_step = lambda x: np.asarray(x).reshape(COPIES, STEPS, ENVS, *x.shape[2:]).transpose(
        1, 0, 2, *range(3, x.ndim + 1))
    actions, observations = by_step(batch["actions"]), by_step(batch["obs"])

    # replay the recorded actions through the environment, one step at a time
    step = jax.jit(runner.env.step)
    current = state.env_state
    rewards, rewards_int, terms, dones = [], [], [], []
    value_ext_final, value_int_final, value_int_next = [], [], []
    for action in actions:
        current, actual_next, reward, term, trunc, final_obs = step(current, jnp.asarray(action))
        final_ext, final_int = critic_values(params["critic"], final_obs)
        _, next_int = critic_values(params["critic"], actual_next)
        rewards.append(np.asarray(reward))
        rewards_int.append(np.asarray(runner.bonus.rollout_step(
            state.bonus_params, state.bonus_state, final_obs)))
        terms.append(np.asarray(term))
        dones.append(np.asarray(term | trunc))
        value_ext_final.append(np.asarray(final_ext))
        value_int_final.append(np.asarray(final_int))
        value_int_next.append(np.asarray(next_int))
    rewards, terms, dones = np.asarray(rewards), np.asarray(terms), np.asarray(dones)
    value_ext_final = np.asarray(value_ext_final)

    # the events are the ones the state was built for: an arrival at the first step of
    # environment 0, a truncation at the second step of environment 1, and nothing else
    assert terms[0, :, 0].all() and terms.sum() == COPIES, terms
    assert dones[1, :, 1].all() and not terms[1, :, 1].any() and dones.sum() == 2 * COPIES, dones
    assert (rewards[0, :, 0] == 1.0).all() and rewards.sum() == COPIES, rewards
    # after the arrival the environment is back at the start, exactly
    np.testing.assert_array_equal(observations[1][:, 0], np.tile(spawn_obs, (COPIES, 1)))

    # values of the stored observations, and the reward each copy's learner sees
    values_ext, values_int = (np.asarray(v) for v in zip(*(
        critic_values(params["critic"], jnp.asarray(obs)) for obs in observations)))
    seen = rewards * scale[None, :, None] + offset[None, :, None]

    # extrinsic: no value after an arrival, the last state's value after a truncation, and the
    # advantage sum stops at every episode end
    expected_ext = reference_returns(seen, values_ext, value_ext_final * (1.0 - terms),
                                     cfg.gamma_ext, cfg.gae_lambda, dones.astype(float))
    actual_ext = by_step(batch["ret_ext"])
    np.testing.assert_allclose(actual_ext, expected_ext, rtol=2e-5, atol=2e-5)
    # at the arrival the return is the reward alone: 1 for copy 0, 0 for copy 1
    np.testing.assert_allclose(actual_ext[0, :, 0], [1.0, 0.0], atol=2e-6)
    # at the truncation the return is the reward plus the discounted value of the last state
    bootstrap = cfg.gamma_ext * value_ext_final[1, :, 1]
    assert (np.abs(bootstrap) > 1e-6).all(), "the last state's value is zero; nothing to check"
    np.testing.assert_allclose(actual_ext[1, :, 1], seen[1, :, 1] + bootstrap, rtol=2e-5,
                               atol=2e-6)
    # every step before the arrival's reset costs 1 / limit under the step cost
    np.testing.assert_allclose(seen[:, 1][~terms[:, 1]], -1 / limit)

    # intrinsic: the return runs across both episode ends, into the value of the reset state
    std = np.sqrt(np.asarray(after.agent_state["int_rms"].var) + 1e-8).astype(np.float32)
    normalized = np.asarray(rewards_int) / std[None]
    no_stop = np.zeros_like(dones, dtype=float)
    expected_int = reference_returns(normalized, values_int, np.asarray(value_int_next),
                                     cfg.gamma_int, cfg.gae_lambda, no_stop)
    np.testing.assert_allclose(by_step(batch["ret_int"]), expected_int, rtol=2e-5, atol=2e-5)
    from_final = reference_returns(normalized, values_int, np.asarray(value_int_final),
                                   cfg.gamma_int, cfg.gae_lambda, no_stop)
    assert np.max(np.abs(from_final - expected_int)) > 1e-4, \
        "the final and the reset observation have the same value; the check proves nothing"

    # the episode log: per copy one arrival after 1 step and one truncated episode of `limit`
    # steps; the first arrival falls on rollout index 0, which ends the copy's training step N
    log = {name: np.asarray(value) for name, value in after.agent_state["episode_log"].items()}
    np.testing.assert_array_equal(log["episodes"], [2, 2])
    np.testing.assert_array_equal(log["arrivals"], [1, 1])
    np.testing.assert_array_equal(log["return_sum"], [1.0, 1.0])
    np.testing.assert_array_equal(log["length_sum"], [1 + limit, 1 + limit])
    np.testing.assert_array_equal(log["steps_to_goal_sum"], [1, 1])
    np.testing.assert_array_equal(log["first_goal_step"], [ENVS, ENVS])
    # the returns of the reward the learner sees: 1 + 0 for copy 0, 0 + (-1) for copy 1
    np.testing.assert_allclose(scale * log["return_sum"] + offset * log["length_sum"],
                               [1.0, -1.0], atol=1e-12)
    print(f"PASS {configuration_preset}: arrival pays {actual_ext[0, :, 0]} with no bootstrap, "
          f"truncation bootstraps {bootstrap}, intrinsic return continues, episode log agrees")


def test_pointmaze_termination_truncation_and_both_reward_settings():
    """PointMaze Large with the episode ending at the goal, step limit 400."""
    check_family("pointmaze_current", EnvConfig(continuing_task=False))


def test_antmaze_termination_truncation_and_both_reward_settings():
    """AntMaze U-maze with the episode ending at the goal, step limit 700, current rules."""
    check_family("antmaze_current", replace(preset("umaze"), continuing_task=False))


if __name__ == "__main__":
    test_pointmaze_termination_truncation_and_both_reward_settings()
    test_antmaze_termination_truncation_and_both_reward_settings()
