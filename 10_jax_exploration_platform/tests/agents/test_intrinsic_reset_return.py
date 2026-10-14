"""Check continuing intrinsic returns against an independent recurrence across real maze resets."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig
from exploration_platform.agents.ppo.networks import critic_values
from exploration_platform.training.runner import Runner

import jax
import jax.numpy as jnp


def reference_returns(rewards, values, next_values, gamma, gae_lambda, stop):
    """Compute returns with an explicit reverse-time loop, independent of the trainer scan."""
    advantage = np.zeros_like(values[0])
    returns = np.empty_like(values)
    # Before: separate rewards and value predictions; after: one return target per transition.
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] - values[step]
        advantage = delta + gamma * gae_lambda * (1.0 - stop[step]) * advantage
        returns[step] = advantage + values[step]
    return returns


def check_reset_case(hoist, initial_steps):
    """Compare captured intrinsic and task targets at an internal reset, final reset, or no reset."""
    cfg = PPOConfig(n_copies=2, n_envs=2, num_steps=4, prime_iterations=0,
                    update_style="full_batch", hoist_rollout=hoist)
    runner = Runner(cfg)
    state = runner.prime(runner.init_state(run_seed=17))
    # Start away from the reset location so using the wrong state changes the value prediction.
    position = jnp.broadcast_to(jnp.asarray([0.5, 1.0], jnp.float32), state.env_state.pos.shape)
    env_state = state.env_state._replace(
        pos=position, step_count=jnp.full_like(state.env_state.step_count, initial_steps))
    state = state._replace(env_state=env_state, obs=jnp.concatenate([position, env_state.vel], -1))
    params = state.agent_params
    fn = jax.jit(runner.composition.iteration_capturing_batch())
    after, _, batch = fn(state, jnp.asarray(1e-4, jnp.float32))

    # Replay the recorded actions to distinguish final observations from actual next observations.
    actions = np.asarray(batch["actions"]).reshape(2, 4, 2, 2).transpose(1, 0, 2, 3)
    observations = np.asarray(batch["obs"]).reshape(2, 4, 2, 4).transpose(1, 0, 2, 3)
    current = env_state
    rewards_ext, rewards_int, value_ext_next, value_int_next, old_int_next, dones = [], [], [], [], [], []
    for action in actions:
        current, actual_next, reward, term, trunc, final_obs = runner.env.step(current, jnp.asarray(action))
        final_ext, final_int = critic_values(params["critic"], final_obs)
        _, actual_int = critic_values(params["critic"], actual_next)
        rewards_ext.append(np.asarray(reward))
        rewards_int.append(np.asarray(runner.bonus.rollout_step(
            state.bonus_params, state.bonus_state, final_obs)))
        value_ext_next.append(np.asarray(final_ext) * (1.0 - np.asarray(term)))
        value_int_next.append(np.asarray(actual_int))
        old_int_next.append(np.asarray(final_int))
        dones.append(np.asarray(term | trunc, dtype=float))
    dones = np.asarray(dones)

    # Use the recorded normalization denominator; independently calculate the return recurrence.
    values_ext, values_int = [], []
    for obs in observations:
        ext, intrinsic = critic_values(params["critic"], jnp.asarray(obs))
        values_ext.append(np.asarray(ext))
        values_int.append(np.asarray(intrinsic))
    std = np.sqrt(np.asarray(after.agent_state["int_rms"].var) + 1e-8).astype(np.float32)
    normalized = np.asarray(rewards_int) / std[None]
    expected_int = reference_returns(normalized, np.asarray(values_int), np.asarray(value_int_next),
                                     cfg.gamma_int, cfg.gae_lambda, np.zeros_like(dones))
    expected_ext = reference_returns(np.asarray(rewards_ext), np.asarray(values_ext),
                                     np.asarray(value_ext_next), cfg.gamma_ext, cfg.gae_lambda, dones)
    actual_int = np.asarray(batch["ret_int"]).reshape(2, 4, 2).transpose(1, 0, 2)
    actual_ext = np.asarray(batch["ret_ext"]).reshape(2, 4, 2).transpose(1, 0, 2)
    np.testing.assert_allclose(actual_int, expected_int, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_ext, expected_ext, rtol=2e-5, atol=2e-5)

    # Make the regression meaningful: the old final-state bootstrap must fail in reset cases.
    old_return = reference_returns(normalized, np.asarray(values_int), np.asarray(old_int_next),
                                   cfg.gamma_int, cfg.gae_lambda, np.zeros_like(dones))
    if initial_steps:
        assert dones.sum() == 4, "each of the four environments must reset exactly once"
        assert np.max(np.abs(old_return - expected_int)) > 1e-3
    else:
        assert dones.sum() == 0
        np.testing.assert_allclose(old_return, expected_int, rtol=2e-5, atol=2e-5)
    print(f"PASS hoist={hoist} initial_episode_steps={initial_steps} "
          f"intrinsic_target_error={np.max(np.abs(actual_int - expected_int)):.3g}", flush=True)


def test_continuing_intrinsic_return():
    """Cover both rollout implementations with a reset first, a reset last, and no reset."""
    # A reset at the last step requires the carried observation, beyond the stored rollout rows.
    for hoist in (False, True):
        for initial_steps in (399, 396, 0):
            check_reset_case(hoist, initial_steps)


if __name__ == "__main__":
    test_continuing_intrinsic_return()
