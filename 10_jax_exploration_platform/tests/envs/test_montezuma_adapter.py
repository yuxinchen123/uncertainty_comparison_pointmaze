"""The Montezuma adapter: contract, keyed determinism, semantics, isolation, x64 boundary.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_montezuma_adapter.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401  (sets jax_enable_x64=True, the platform default)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.atari_montezuma.jax_montezuma import (  # noqa: E402
    FRAME_FEATURES, N_ROOMS, JaxMontezuma, MontezumaConfig)


def rollout(env, state, steps, key, n_actions=18):
    """Step keyed random actions under one jit; returns (state, stacked obs, rewards)."""
    step = jax.jit(env.step)
    obs_all, rew_all = [], []
    for k in range(steps):
        act = jax.random.randint(jax.random.fold_in(key, k), (env.C, env.N), 0, n_actions,
                                 jnp.int32)
        state, obs, rew, term, trunc, final = step(state, act)
        obs_all.append(obs)
        rew_all.append(rew)
    return state, jnp.stack(obs_all), jnp.stack(rew_all)


def test_contract_shapes_and_room_feature():
    """The six outputs have platform shapes, and the room feature names the game's room."""
    env = JaxMontezuma(MontezumaConfig(), 2, 2)
    s = env.reset()
    assert env.obs_dim == 4 * FRAME_FEATURES + 6
    step = jax.jit(env.step)
    s, obs, rew, term, trunc, final = step(s, jnp.zeros((2, 2), jnp.int32))
    assert obs.shape == (2, 2, env.obs_dim) and obs.dtype == jnp.float32
    assert rew.shape == (2, 2) and rew.dtype == jnp.float32
    assert term.dtype == jnp.bool_ and trunc.dtype == jnp.bool_
    # the appended room feature equals the game's mapped room index (start room id 4 -> 1)
    from jaxatari.games.montezuma_revenge.core import get_room_idx
    want = np.asarray(get_room_idx(s.atari.env_state.room_id)).reshape(2, 2)
    np.testing.assert_array_equal(np.asarray(obs[..., 4 * FRAME_FEATURES]), want)
    print("ok test_contract_shapes_and_room_feature")


def test_keyed_determinism_and_env_diversity():
    """Same base seed -> bitwise-equal runs; different env indices -> different sticky draws."""
    key = jax.random.PRNGKey(9)
    env = JaxMontezuma(MontezumaConfig(), 1, 2, base_seed=3)
    s1, obs1, _ = rollout(env, env.reset(), 12, key)
    env2 = JaxMontezuma(MontezumaConfig(), 1, 2, base_seed=3)
    s2, obs2, _ = rollout(env2, env2.reset(), 12, key)
    np.testing.assert_array_equal(np.asarray(obs1), np.asarray(obs2))
    # the two environments of one copy hold different keys, so their sticky-action draws —
    # and after enough steps their states — differ under identical requested actions
    assert (np.asarray(obs1[-1, 0, 0]) != np.asarray(obs1[-1, 0, 1])).any()
    print("ok test_keyed_determinism_and_env_diversity")


def test_copy_isolation():
    """Changing what other copies do never changes copy 0's numbers, bit for bit."""
    env = JaxMontezuma(MontezumaConfig(), 2, 1)
    key = jax.random.PRNGKey(4)
    step = jax.jit(env.step)

    def run(other_action):
        """8 steps: copy 0 gets keyed actions, copy 1 a fixed action."""
        s = env.reset()
        for k in range(8):
            a0 = jax.random.randint(jax.random.fold_in(key, k), (1, 1), 0, 18, jnp.int32)
            act = jnp.concatenate([a0, jnp.full((1, 1), other_action, jnp.int32)], axis=0)
            s, obs, *_ = step(s, act)
        return np.asarray(obs[0])

    np.testing.assert_array_equal(run(2), run(5))
    print("ok test_copy_isolation")


def test_truncation_and_autoreset():
    """At the cap the env truncates and the fresh episode has 5 lives, room 4, step 0."""
    cfg = MontezumaConfig(max_episode_steps=6)
    env = JaxMontezuma(cfg, 1, 1)
    s = env.reset()
    step = jax.jit(env.step)
    truncated_seen = False
    for k in range(6):
        s, obs, rew, term, trunc, final = step(s, jnp.zeros((1, 1), jnp.int32))
        truncated_seen = truncated_seen or bool(trunc.any())
    assert truncated_seen
    assert int(s.step_count[0, 0]) == 0 and int(s.reset_count[0, 0]) >= 1
    assert int(s.atari.env_state.lives[0]) == 5
    assert int(s.atari.step[0]) == 0
    print("ok test_truncation_and_autoreset")


def test_x64_boundary_and_coverage():
    """Under global x64 the observation stays float32, and cell_index reads the room."""
    assert jax.config.jax_enable_x64
    env = JaxMontezuma(MontezumaConfig(), 1, 1)
    s = env.reset()
    s, obs, *_ = jax.jit(env.step)(s, jnp.zeros((1, 1), jnp.int32))
    assert obs.dtype == jnp.float32
    assert env.n_cells == N_ROOMS and bool(env.open_cells.all())
    idx = np.asarray(env.cell_index(obs))
    assert idx.shape == (1, 1) and 0 <= idx[0, 0] < N_ROOMS
    print("ok test_x64_boundary_and_coverage")


if __name__ == "__main__":
    test_contract_shapes_and_room_feature()
    test_keyed_determinism_and_env_diversity()
    test_copy_isolation()
    test_truncation_and_autoreset()
    test_x64_boundary_and_coverage()
