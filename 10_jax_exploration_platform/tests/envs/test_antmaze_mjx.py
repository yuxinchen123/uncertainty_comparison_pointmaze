"""The MJX AntMaze stepper: shapes, determinism, semantics, isolation, and the x64 boundary.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_antmaze_mjx.py
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401  (sets jax_enable_x64=True, the platform default)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze.am_common import cell_center, preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import MAPS  # noqa: E402


def rollout(env, state, key, steps):
    """Step `steps` random torques under one jit; returns (state, stacked obs, rewards)."""
    step = jax.jit(env.step)
    obs_all, rew_all = [], []
    for k in range(steps):
        act = jax.random.uniform(jax.random.fold_in(key, k), (env.C, env.N, env.act_dim),
                                 jnp.float32, -1.0, 1.0)
        state, obs, rew, term, trunc, final = step(state, act)
        obs_all.append(obs)
        rew_all.append(rew)
    return state, jnp.stack(obs_all), jnp.stack(rew_all)


def test_reset_shapes_and_spawn():
    """Reset puts every env exactly at the start cell's centre, zero velocity, float32."""
    env = JaxAntMaze(preset("umaze"), 2, 3)
    s = env.reset()
    assert s.data.qpos.shape == (6, env.nq) and s.data.qpos.dtype == jnp.float32
    start = cell_center((3, 1), env.rows, env.cols)
    np.testing.assert_array_equal(np.asarray(s.data.qpos[:, 0:2]),
                                  np.tile(np.float32(start), (6, 1)))
    assert float(s.data.qpos[:, 2].min()) == 0.75  # the ant.xml torso height
    assert float(jnp.abs(s.data.qvel).max()) == 0.0
    _, obs = env.respawn(s)
    assert obs.shape == (2, 3, 29) and obs.dtype == jnp.float32
    print("ok test_reset_shapes_and_spawn")


def test_step_contract_and_determinism():
    """The six outputs have the platform shapes, and the same actions give bitwise-equal states."""
    env = JaxAntMaze(preset("umaze"), 2, 2)
    key = jax.random.PRNGKey(7)
    s1, obs1, rew1 = rollout(env, env.reset(), key, 5)
    s2, obs2, rew2 = rollout(env, env.reset(), key, 5)
    assert obs1.shape == (5, 2, 2, 29) and rew1.shape == (5, 2, 2)
    np.testing.assert_array_equal(np.asarray(obs1), np.asarray(obs2))
    np.testing.assert_array_equal(np.asarray(s1.data.qpos), np.asarray(s2.data.qpos))
    assert bool(jnp.isfinite(obs1).all())
    print("ok test_step_contract_and_determinism")


def test_copy_isolation():
    """Changing what other copies do never changes copy 0's numbers, bit for bit."""
    env = JaxAntMaze(preset("umaze"), 2, 1)
    key = jax.random.PRNGKey(3)
    step = jax.jit(env.step)

    def run(other_scale):
        """5 steps: copy 0 gets fixed torques, copy 1 gets torques scaled by other_scale."""
        s = env.reset()
        for k in range(5):
            a0 = jax.random.uniform(jax.random.fold_in(key, k), (1, 1, 8),
                                    jnp.float32, -1.0, 1.0)
            act = jnp.concatenate([a0, a0 * other_scale], axis=0)
            s, obs, *_ = step(s, act)
        return np.asarray(s.data.qpos)

    qa, qb = run(1.0), run(-0.5)
    np.testing.assert_array_equal(qa[0], qb[0])
    assert (qa[1] != qb[1]).any()
    print("ok test_copy_isolation")


def test_goal_reward_semantics():
    """Reward is exactly 1 inside the 0.45 m radius and 0 outside, and never terminates."""
    env = JaxAntMaze(preset("umaze"), 1, 2)
    s = env.reset()
    goal = np.asarray(env.goal)
    qpos = np.asarray(s.data.qpos).copy()
    qpos[0, 0:2] = goal + np.array([0.2, 0.0], np.float32)   # env 0: inside the radius
    qpos[1, 0:2] = goal + np.array([0.6, 0.0], np.float32)   # env 1: outside
    s = s._replace(data=s.data.replace(qpos=jnp.asarray(qpos)))
    s, obs, rew, term, trunc, final = jax.jit(env.step)(s, jnp.zeros((1, 2, 8), jnp.float32))
    # 0.2 m from the goal minus at most ~0.05 m of drift in one step stays well inside 0.45
    assert float(rew[0, 0]) == 1.0 and float(rew[0, 1]) == 0.0, np.asarray(rew)
    assert not bool(term.any())
    print("ok test_goal_reward_semantics")


def test_truncation_and_autoreset():
    """At the cap the env truncates, respawns exactly, and counts the episode."""
    cfg = preset("umaze")
    env = JaxAntMaze(cfg, 1, 1)
    s = env.reset()
    s = s._replace(step_count=jnp.full((1, 1), cfg.max_episode_steps - 1, jnp.int32))
    s, obs, rew, term, trunc, final = jax.jit(env.step)(s, jnp.ones((1, 1, 8), jnp.float32))
    assert bool(trunc.all()) and not bool(term.any())
    np.testing.assert_array_equal(np.asarray(s.data.qpos[0]), np.asarray(env.init_qpos))
    assert float(jnp.abs(s.data.qvel).max()) == 0.0
    assert int(s.step_count[0, 0]) == 0 and int(s.reset_count[0, 0]) == 1
    # final_obs carries the pre-reset observation, obs the post-reset one
    assert (np.asarray(final[0, 0, :15]) != np.asarray(env.init_qpos)).any()
    np.testing.assert_array_equal(np.asarray(obs[0, 0, :15]), np.asarray(env.init_qpos))
    print("ok test_truncation_and_autoreset")


def test_x64_boundary():
    """Under the platform's global x64, the physics state and observations stay float32."""
    assert jax.config.jax_enable_x64
    env = JaxAntMaze(preset("medium"), 1, 1)
    s = env.reset()
    assert s.data.qpos.dtype == jnp.float32 and s.data.qvel.dtype == jnp.float32
    s, obs, rew, *_ = jax.jit(env.step)(s, jnp.zeros((1, 1, 8), jnp.float32))
    assert obs.dtype == jnp.float32 and rew.dtype == jnp.float32
    assert s.data.qpos.dtype == jnp.float32
    print("ok test_x64_boundary")


def test_cell_index_and_open_cells():
    """The coverage indexer maps world metres to the right cells, and the mask counts opens."""
    env = JaxAntMaze(preset("large"), 1, 1)
    start, goal = cell_center((7, 1), 9, 12), cell_center((1, 10), 9, 12)
    obs = jnp.zeros((1, 2, 29), jnp.float32)
    obs = obs.at[0, 0, 0:2].set(jnp.asarray(start))
    obs = obs.at[0, 1, 0:2].set(jnp.asarray(goal))
    idx = np.asarray(env.cell_index(obs))
    assert idx.tolist() == [[7 * 12 + 1, 1 * 12 + 10]], idx
    for name in ("umaze", "medium", "large"):
        e = JaxAntMaze(preset(name), 1, 1)
        n_open = int(np.sum(np.asarray(MAPS[name]) == 0))
        assert int(e.open_cells.sum()) == n_open
    print("ok test_cell_index_and_open_cells")


def test_nan_guard():
    """An environment whose physics state goes non-finite is respawned, counted, pays 0."""
    env = JaxAntMaze(preset("umaze"), 1, 2)
    s = env.reset()
    qpos = np.asarray(s.data.qpos).copy()
    qpos[0, 3] = np.nan   # env 0: poison one coordinate; env 1 stays healthy
    s = s._replace(data=s.data.replace(qpos=jnp.asarray(qpos)))
    s, obs, rew, term, trunc, final = jax.jit(env.step)(s, jnp.zeros((1, 2, 8), jnp.float32))
    assert bool(trunc[0, 0]) and not bool(term[0, 0])
    assert float(rew[0, 0]) == 0.0
    assert bool(jnp.isfinite(final).all()) and bool(jnp.isfinite(obs).all())
    np.testing.assert_array_equal(np.asarray(s.data.qpos[0]), np.asarray(env.init_qpos))
    assert int(s.nan_count[0, 0]) == 1 and int(s.nan_count[0, 1]) == 0
    assert bool(jnp.isfinite(s.data.qpos[1]).all())  # the healthy env kept its state
    print("ok test_nan_guard")


def test_preset_connectivity():
    """In every preset the goal is reachable from the start through open cells."""
    for name in ("umaze", "medium", "large"):
        cfg = preset(name)
        grid = MAPS[name]
        rows, cols = len(grid), len(grid[0])
        seen, q = {cfg.start_cell}, deque([cfg.start_cell])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if (0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] == 0
                        and (ii, jj) not in seen):
                    seen.add((ii, jj))
                    q.append((ii, jj))
        assert cfg.goal_cell in seen, f"{name}: goal {cfg.goal_cell} unreachable"
    print("ok test_preset_connectivity")


if __name__ == "__main__":
    test_reset_shapes_and_spawn()
    test_step_contract_and_determinism()
    test_copy_isolation()
    test_goal_reward_semantics()
    test_truncation_and_autoreset()
    test_x64_boundary()
    test_cell_index_and_open_cells()
    test_nan_guard()
    test_preset_connectivity()
