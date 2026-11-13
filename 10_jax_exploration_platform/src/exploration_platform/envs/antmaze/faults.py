"""The AntMaze fault record and the physics fault guard.

The fault record: per policy, how often the torso was in the ceiling zone, how fast it moved and
turned at most, and how many environments were respawned for a broken physics state, since the
driver last read them.

The model has a ceiling on the wall tops that only the torso meets (`am_common`), so an ant
cannot cross a wall. An ant that walks stays far below it. A policy that learns to get itself
thrown upward would still reach the ceiling, and the first two numbers make that visible:

  ceiling_zone_steps      environment steps in which the torso centre was above
                          `CEILING_ZONE_HEIGHT` (1.5 m) after any of the step's physics steps
  largest_torso_speed     the largest speed of the torso centre, in metres per second, after any
                          physics step
  physics_fault_respawns  environment steps that ended in a respawn because the physics state
                          was not finite or, with the guard on, beyond the guard's bounds
  largest_torso_angular_speed  the largest angular speed of the torso, in radians per second,
                          after any physics step

All four live in the environment state as one number per policy and grow inside the compiled
rollout. The training command reads them at the end of each logging window and sets them back
to zero (`read_window`, `clear_window`); `respawn` also sets them to zero, so the first window
holds no priming step. Both physics implementations use the same functions.

The guard (`beyond_bounds`; spec.md, deviation 9) has two bounds in the configuration:
`fault_torso_speed` (metres per second) and `fault_torso_angular_speed` (radians per second). It
is on when either is above zero. An environment whose torso moved or turned faster than a bound
that is above zero after any physics step, or whose torso centre was outside the maze's volume
after any physics step, is then respawned exactly like one whose state is not finite: no reward,
no arrival, a truncation whose final observation is the spawn. With both bounds at 0 (the
default) only a non-finite state is respawned, as in every run before 2026-11-12.
"""
import jax.numpy as jnp
import numpy as np

from .am_common import CEILING_HEIGHT, CEILING_ZONE_HEIGHT

F32 = jnp.float32
# the names of the four numbers in the training command's window record
RECORD_NAMES = {"ceiling_zone_steps": "ceiling_zone_steps_per_copy",
                "largest_torso_speed": "largest_torso_speed_per_copy",
                "physics_fault_respawns": "physics_fault_respawns_per_copy",
                "largest_torso_angular_speed": "largest_torso_angular_speed_per_copy"}


def window_init(n_copies: int):
    """An empty window: (ceiling-zone steps [C] int32, largest torso speed [C] float32,
    physics-fault respawns [C] int32, largest torso angular speed [C] float32), zeros."""
    return (jnp.zeros((n_copies,), jnp.int32), jnp.zeros((n_copies,), F32),
            jnp.zeros((n_copies,), jnp.int32), jnp.zeros((n_copies,), F32))


def torso_angular_speed(qvel):
    """Torso angular speed [B] in radians per second of a batch of physics states.

    before: qvel [B, 14] (torso linear velocity, then torso angular velocity, then 8 joints)
    after:  the length of qvel[:, 3:6]
    """
    return jnp.sqrt((qvel[:, 3:6] * qvel[:, 3:6]).sum(-1))


def torso_height_and_speed(qpos, qvel):
    """Torso-centre height [B] and torso-centre speed [B] of a batch of physics states.

    before: qpos [B, 15] (torso x, y, z first), qvel [B, 14] (torso linear velocity first)
    after:  (qpos[:, 2], the length of qvel[:, 0:3])
    """
    return qpos[:, 2], jnp.sqrt((qvel[:, 0:3] * qvel[:, 0:3]).sum(-1))


def add_step(zone_steps, largest_speed, heights, speeds, n_copies: int, n_envs: int):
    """The window's two numbers after one more environment step.

    heights, speeds: [frame_skip, B], the torso after each physics step of the environment step,
    B = n_copies x n_envs with a policy's environments next to each other. A height or speed
    that is not a finite number counts as outside the zone and as speed 0: such an environment
    is respawned and counted by the non-finite rule.

    before: one policy with two environments, two physics steps; heights [[0.6, 1.4], [0.7, 1.6]],
            speeds [[1.0, 3.0], [2.0, nan]], zone_steps [5], largest_speed [2.5]
    after:  zone_steps [6] (the second environment was above 1.5 m once), largest_speed [3.0]
    """
    in_zone = (heights > CEILING_ZONE_HEIGHT).any(0).reshape(n_copies, n_envs)
    speed = jnp.where(jnp.isfinite(speeds), speeds, 0.0).max(0).reshape(n_copies, n_envs)
    return (zone_steps + in_zone.sum(-1, dtype=jnp.int32),
            jnp.maximum(largest_speed, speed.max(-1)))


def beyond_bounds(heights, speeds, xy, spins, speed_bound: float, spin_bound: float, half_extent):
    """[B] bool: the torso moved or turned faster than its bound, or its centre was outside the
    maze's volume, after any physics step of the environment step. A bound of 0 is not tested;
    with both bounds at 0 the guard is off and nothing is flagged.

    heights, speeds, spins: [frame_skip, B]; xy: [frame_skip, B, 2], the torso centre after each
    physics step. half_extent: (half the maze's width, half its depth) in metres, the outer faces
    of the outer walls. The volume is those faces, the floor (z = 0) and the ceiling (z = 2 m). A
    value that is not a number compares as inside: the non-finite rule counts that environment.

    before: bounds 30 m/s and 100 rad/s, half_extent (10, 10); one environment, two physics steps:
            heights [[0.6], [0.7]], speeds [[12.0], [14.0]], spins [[40.0], [140.0]],
            xy [[[0, 0]], [[0, 1]]]
    after:  [True] (140 rad/s after the second physics step)
    """
    if speed_bound <= 0.0 and spin_bound <= 0.0:
        return jnp.zeros(speeds.shape[1:], bool)
    flagged = ((jnp.abs(xy[..., 0]) > half_extent[0]) | (jnp.abs(xy[..., 1]) > half_extent[1])
               | (heights < 0.0) | (heights > CEILING_HEIGHT))
    if speed_bound > 0.0:
        flagged = flagged | (speeds > speed_bound)
    if spin_bound > 0.0:
        flagged = flagged | (spins > spin_bound)
    return flagged.any(0)


def add_spin(largest_spin, spins, n_copies: int, n_envs: int):
    """The window's largest torso angular speed after one more environment step.

    before: largest_spin [20.0]; one policy with two environments, spins [[5.0, 31.0], [nan, 8.0]]
    after:  [31.0] (a value that is not a finite number counts as 0)
    """
    spin = jnp.where(jnp.isfinite(spins), spins, 0.0).max(0).reshape(n_copies, n_envs)
    return jnp.maximum(largest_spin, spin.max(-1))


def add_respawns(respawns, bad, n_copies: int, n_envs: int):
    """The window's respawn count after one more environment step.

    before: respawns [4, 0]; bad [C=2, N=2] = [[False, True], [False, False]]
    after:  [5, 0]
    """
    return respawns + bad.reshape(n_copies, n_envs).sum(-1, dtype=jnp.int32)


def read_window(env_state) -> dict:
    """The window's four numbers on the host, under their names in the window record.

    before: state with ceiling_zone_steps [0, 3], largest_torso_speed [1.25, 9.5],
            physics_fault_respawns [0, 1], largest_torso_angular_speed [12.0, 48.0]
    after:  {"ceiling_zone_steps_per_copy": [0, 3], "largest_torso_speed_per_copy": [1.25, 9.5],
             "physics_fault_respawns_per_copy": [0, 1],
             "largest_torso_angular_speed_per_copy": [12.0, 48.0]}
    """
    return {RECORD_NAMES[name]: np.asarray(getattr(env_state, name)).tolist()
            for name in RECORD_NAMES}


def clear_window(env_state):
    """The environment state with the window's four numbers back at zero."""
    zone_steps, largest_speed, respawns, largest_spin = window_init(
        env_state.ceiling_zone_steps.shape[0])
    return env_state._replace(ceiling_zone_steps=zone_steps, largest_torso_speed=largest_speed,
                              physics_fault_respawns=respawns,
                              largest_torso_angular_speed=largest_spin)
