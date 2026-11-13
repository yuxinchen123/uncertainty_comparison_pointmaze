"""The AntMaze physics fault guard (`fault_torso_speed`, `fault_torso_angular_speed`,
`faults.beyond_bounds`), in both implementations.

With the guard on, an environment whose torso moves or turns faster than its bound after any physics step,
or whose torso centre is outside the maze's volume, is respawned like one whose state is not finite: no
reward, no arrival, a truncation that shows the spawn, and one more in the window's count of physics-fault
respawns. With the guard off (the default, both bounds 0) the same states go on as in every run before
2026-11-12.

The JAX cases run on processors with the platform_jax interpreter; the Warp cases need the
platform_jax_warp interpreter and are skipped without warp-lang.
"""
import hashlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import exploration_platform  # noqa: E402,F401  (64-bit statistics on, as in training)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze import faults  # noqa: E402
from exploration_platform.envs.antmaze.am_common import (AntMazeConfig, cell_center, maze_half_extent,  # noqa: E402
                                                         preset)

HAS_WARP = importlib.util.find_spec("warp") is not None
IMPLEMENTATIONS = ["jax", pytest.param("warp", marks=pytest.mark.skipif(
    not HAS_WARP, reason="this interpreter has no warp-lang (platform_jax_warp holds it)"))]
BOUND = 30.0        # metres per second, the speed bound of these tests
SPIN_BOUND = 100.0  # radians per second, the angular-speed bound of these tests
# the six ants of the forced-state test, one policy with one environment each
ANTS = ("at the spawn, at rest", "at the spawn, moving at 60 m/s along the corridor",
        "at the goal, at rest", "at the goal, moving upward at 40 m/s",
        "2 m outside the outer wall, at rest", "at the spawn, turning at 150 rad/s about the vertical")
BROKEN = (1, 3, 4, 5)


def make_environment(implementation, cfg, n):
    """One environment per ant (n policies x 1 environment) with the named physics."""
    if implementation == "warp":
        from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze
        return WarpAntMaze(replace(cfg, physics="warp"), n, 1)
    from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze
    return JaxAntMaze(cfg, n, 1)


def forced_state(env):
    """The environment's reset state with the six ants of `ANTS` put in by hand."""
    state = env.reset()
    qpos = np.tile(np.asarray(env.init_qpos), (len(ANTS), 1))
    qvel = np.zeros((len(ANTS), 14), np.float32)
    goal = cell_center(env.cfg.goal_cell, env.rows, env.cols)
    qvel[1, 0] = 60.0
    qpos[2, 0:2] = goal
    qpos[3, 0:2], qvel[3, 2] = goal, 40.0
    qpos[4, 0:2] = (maze_half_extent(env.cfg.map_name)[0] + 2.0, 0.0)
    qvel[5, 5] = 150.0
    with jax.enable_x64(False):
        data = state.data.replace(qpos=jnp.asarray(qpos, jnp.float32), qvel=jnp.asarray(qvel, jnp.float32))
    return state._replace(data=data)


def one_step(implementation, bound, spin_bound):
    """The six forced ants after one environment step with zero torques, as host arrays."""
    cfg = replace(preset("umaze"), continuing_task=False, fault_torso_speed=bound,
                  fault_torso_angular_speed=spin_bound)
    env = make_environment(implementation, cfg, len(ANTS))
    state, obs, reward, terminated, truncated, final_obs = jax.jit(env.step)(
        forced_state(env), jnp.zeros((len(ANTS), 1, 8), jnp.float32))
    spawn = np.concatenate([np.asarray(env.init_qpos), np.zeros(14, np.float32)])
    return {"env": env, "state": state, "spawn": spawn, "obs": np.asarray(obs)[:, 0],
            "final_obs": np.asarray(final_obs)[:, 0], "reward": np.asarray(reward)[:, 0],
            "terminated": np.asarray(terminated)[:, 0], "truncated": np.asarray(truncated)[:, 0]}


def test_the_default_configuration_has_the_guard_off():
    """Every configuration built without the field keeps the earlier rule."""
    assert AntMazeConfig().fault_torso_speed == 0.0 and AntMazeConfig().fault_torso_angular_speed == 0.0
    assert all(preset(name).fault_torso_speed == 0.0 and preset(name).fault_torso_angular_speed == 0.0
               for name in ("umaze", "medium", "large"))
    assert maze_half_extent("umaze") == (10.0, 10.0) and maze_half_extent("large") == (24.0, 18.0)


def test_beyond_bounds_on_a_hand_made_case():
    """Five environments, two physics steps: too fast once, outside sideways, above the ceiling, fine,
    turning too fast once."""
    heights = jnp.asarray([[0.6, 0.7, 2.1, 0.6, 0.6], [0.7, 0.7, 1.0, 0.5, 0.6]], jnp.float32)
    speeds = jnp.asarray([[12.0, 3.0, 3.0, 29.9, 2.0], [31.0, 3.0, 3.0, 30.0, 2.0]], jnp.float32)
    spins = jnp.asarray([[5.0, 5.0, 5.0, 100.0, 40.0], [5.0, 5.0, 5.0, 99.0, 140.0]], jnp.float32)
    xy = jnp.asarray([[[0, 0], [10.5, 0], [0, 0], [9.9, -9.9], [0, 0]],
                      [[0, 1], [9.0, 0], [0, 0], [10.0, -10.0], [0, 0]]], jnp.float32)
    flags = faults.beyond_bounds(heights, speeds, xy, spins, 30.0, 100.0, (10.0, 10.0))
    np.testing.assert_array_equal(np.asarray(flags), [True, True, True, False, True])
    # one bound at 0 is not tested, the other still is, and so is the maze's volume
    only_speed = faults.beyond_bounds(heights, speeds, xy, spins, 30.0, 0.0, (10.0, 10.0))
    np.testing.assert_array_equal(np.asarray(only_speed), [True, True, True, False, False])
    only_spin = faults.beyond_bounds(heights, speeds, xy, spins, 0.0, 100.0, (10.0, 10.0))
    np.testing.assert_array_equal(np.asarray(only_spin), [False, True, True, False, True])
    # below the floor, and beyond the far side in y
    low = faults.beyond_bounds(jnp.asarray([[-0.01, 0.5]], jnp.float32), jnp.zeros((1, 2), jnp.float32),
                               jnp.asarray([[[0, 0], [0, -10.01]]], jnp.float32),
                               jnp.zeros((1, 2), jnp.float32), 30.0, 100.0, (10.0, 10.0))
    np.testing.assert_array_equal(np.asarray(low), [True, True])
    # the guard off (both bounds 0): nothing is flagged, whatever the state
    off = faults.beyond_bounds(heights, speeds * 1e6, xy * 1e6, spins * 1e6, 0.0, 0.0, (10.0, 10.0))
    np.testing.assert_array_equal(np.asarray(off), [False] * 5)
    # a value that is not a number is left to the non-finite rule
    nan = faults.beyond_bounds(jnp.asarray([[jnp.nan]], jnp.float32), jnp.asarray([[jnp.nan]], jnp.float32),
                               jnp.asarray([[[jnp.nan, 0.0]]], jnp.float32),
                               jnp.asarray([[jnp.nan]], jnp.float32), 30.0, 100.0, (10.0, 10.0))
    np.testing.assert_array_equal(np.asarray(nan), [False])


def test_the_respawn_count_and_the_window_record():
    """`add_respawns` adds one per broken environment to its policy, `add_spin` keeps the largest finite
    angular speed, and the window record names both."""
    respawns = faults.add_respawns(jnp.asarray([4, 0], jnp.int32),
                                   jnp.asarray([[False, True], [False, False]]), 2, 2)
    np.testing.assert_array_equal(np.asarray(respawns), [5, 0])
    assert respawns.dtype == jnp.int32
    # one policy with two environments, two physics steps: 31 rad/s replaces 20, the NaN counts as 0
    spin = faults.add_spin(jnp.asarray([20.0], jnp.float32),
                           jnp.asarray([[5.0, 31.0], [jnp.nan, 8.0]], jnp.float32), 1, 2)
    np.testing.assert_array_equal(np.asarray(spin), [31.0])
    assert spin.dtype == jnp.float32
    qvel = jnp.zeros((1, 14), jnp.float32).at[0, 3:6].set(jnp.asarray([3.0, 4.0, 12.0], jnp.float32))
    assert float(faults.torso_angular_speed(qvel)[0]) == 13.0
    assert faults.RECORD_NAMES["physics_fault_respawns"] == "physics_fault_respawns_per_copy"
    assert faults.RECORD_NAMES["largest_torso_angular_speed"] == "largest_torso_angular_speed_per_copy"
    zone, speed, count, largest_spin = faults.window_init(3)
    assert count.shape == (3,) and count.dtype == jnp.int32 and not np.asarray(count).any()
    assert largest_spin.shape == (3,) and largest_spin.dtype == jnp.float32


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_a_state_beyond_the_bounds_is_respawned_counted_and_gives_no_arrival(implementation):
    """The fast ant, the ant thrown upward at the goal, the ant outside the maze and the ant turning
    too fast are respawned; the ant at rest at the goal still arrives."""
    run = one_step(implementation, BOUND, SPIN_BOUND)
    broken = np.isin(np.arange(len(ANTS)), BROKEN)
    np.testing.assert_array_equal(run["reward"], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(run["terminated"], [False, False, True, False, False, False])
    np.testing.assert_array_equal(run["truncated"], broken)
    # a broken environment shows the spawn as its final observation, and starts again there
    for ant in BROKEN:
        np.testing.assert_array_equal(run["final_obs"][ant], run["spawn"])
    for ant in (1, 2, 3, 4, 5):
        np.testing.assert_array_equal(run["obs"][ant], run["spawn"])
    state = run["state"]
    np.testing.assert_array_equal(np.asarray(state.nan_count)[:, 0], broken.astype(np.int32))
    np.testing.assert_array_equal(np.asarray(state.step_count)[:, 0], [1, 0, 0, 0, 0, 0])
    record, cleared = run["env"].fault_window(state)
    assert record["physics_fault_respawns_per_copy"] == broken.astype(int).tolist()
    # the speeds that broke the bounds stay in the fault record, so a window shows them
    assert record["largest_torso_speed_per_copy"][1] > BOUND
    assert record["largest_torso_angular_speed_per_copy"][5] > SPIN_BOUND
    assert not np.asarray(cleared.physics_fault_respawns).any()
    assert not np.asarray(cleared.largest_torso_angular_speed).any()
    respawned, _ = run["env"].respawn(state)
    assert not np.asarray(respawned.physics_fault_respawns).any()
    assert not np.asarray(respawned.largest_torso_angular_speed).any()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_each_bound_works_alone(implementation):
    """Only the speed bound: the turning ant goes on. Only the angular bound: the fast ant and the ant
    thrown upward go on, and that ant arrives. The ant outside the maze is respawned in both."""
    only_speed = one_step(implementation, BOUND, 0.0)
    np.testing.assert_array_equal(only_speed["truncated"], [False, True, False, True, True, False])
    only_spin = one_step(implementation, 0.0, SPIN_BOUND)
    np.testing.assert_array_equal(only_spin["truncated"], [False, False, False, False, True, True])
    np.testing.assert_array_equal(only_spin["terminated"], [False, False, True, True, False, False])


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_with_the_guard_off_the_same_states_go_on(implementation):
    """Both bounds 0: nothing is respawned, the ant thrown upward at the goal counts as an arrival (the
    rule of every run before the guard), and the respawn count stays at zero."""
    run = one_step(implementation, 0.0, 0.0)
    np.testing.assert_array_equal(run["reward"], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(run["terminated"], [False, False, True, True, False, False])
    assert not run["truncated"].any()
    assert not np.asarray(run["state"].nan_count).any()
    record, _ = run["env"].fault_window(run["state"])
    assert record["physics_fault_respawns_per_copy"] == [0] * len(ANTS)
    # the fast ant went on from where the physics put it, 60 m/s for 0.05 s further along the corridor
    assert run["obs"][1, 0] - run["spawn"][0] > 2.0


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_ordinary_steps_are_the_same_with_the_guard_on_and_off(implementation):
    """Random torques from the spawn for 20 environment steps: no ant comes near the bound, and every
    output is equal bit for bit with the guard on and off."""
    # before: a generator seeded with the low 32 bits of sha256("0::fault guard test::torques")
    seed = int(hashlib.sha256(b"0::fault guard test::torques").hexdigest(), 16) & 0xFFFFFFFF
    torques = jnp.asarray(np.clip(np.random.default_rng(seed).standard_normal((20, 4, 1, 8)), -1.0, 1.0),
                          jnp.float32)
    outputs = {}
    for bound, spin_bound in ((0.0, 0.0), (BOUND, SPIN_BOUND)):
        env = make_environment(implementation, replace(preset("umaze"), continuing_task=False,
                                                       fault_torso_speed=bound,
                                                       fault_torso_angular_speed=spin_bound), 4)

        def rollout(state, torque_steps, env=env):
            """All 20 steps; the per-step observations, rewards and ends."""
            def body(state, torque):
                state, obs, reward, terminated, truncated, final_obs = env.step(state, torque)
                return state, (obs, reward, terminated, truncated, final_obs)
            return jax.lax.scan(body, state, torque_steps)
        state, outs = jax.jit(rollout)(env.reset(), torques)
        record, _ = env.fault_window(state)
        assert record["physics_fault_respawns_per_copy"] == [0] * 4
        assert max(record["largest_torso_speed_per_copy"]) < BOUND
        assert 0.0 < max(record["largest_torso_angular_speed_per_copy"]) < SPIN_BOUND
        outputs[bound] = [np.asarray(part) for part in outs]
    for off, on in zip(outputs[0.0], outputs[BOUND]):
        np.testing.assert_array_equal(off, on)
