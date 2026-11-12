"""The AntMaze fault record: per policy, how often the torso was in the ceiling zone and how
fast it moved at most, since the driver last read them.

The model has a ceiling on the wall tops that only the torso meets (`am_common`), so an ant
cannot cross a wall. An ant that walks stays far below it. A policy that learns to get itself
thrown upward would still reach the ceiling, and these two numbers make that visible:

  ceiling_zone_steps   environment steps in which the torso centre was above
                       `CEILING_ZONE_HEIGHT` (1.5 m) after any of the step's physics steps
  largest_torso_speed  the largest speed of the torso centre, in metres per second, after any
                       physics step

Both live in the environment state as one number per policy and grow inside the compiled
rollout. The training command reads them at the end of each logging window and sets them back
to zero (`read_window`, `clear_window`); `respawn` also sets them to zero, so the first window
holds no priming step. Both physics implementations use the same functions.
"""
import jax.numpy as jnp
import numpy as np

from .am_common import CEILING_ZONE_HEIGHT

F32 = jnp.float32
# the names of the two numbers in the training command's window record
RECORD_NAMES = {"ceiling_zone_steps": "ceiling_zone_steps_per_copy",
                "largest_torso_speed": "largest_torso_speed_per_copy"}


def window_init(n_copies: int):
    """An empty window: (ceiling-zone steps [C] int32, largest torso speed [C] float32), zeros."""
    return jnp.zeros((n_copies,), jnp.int32), jnp.zeros((n_copies,), F32)


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


def read_window(env_state) -> dict:
    """The window's two numbers on the host, under their names in the window record.

    before: state with ceiling_zone_steps [0, 3], largest_torso_speed [1.25, 9.5]
    after:  {"ceiling_zone_steps_per_copy": [0, 3], "largest_torso_speed_per_copy": [1.25, 9.5]}
    """
    return {RECORD_NAMES[name]: np.asarray(getattr(env_state, name)).tolist()
            for name in RECORD_NAMES}


def clear_window(env_state):
    """The environment state with the window's two numbers back at zero."""
    zone_steps, largest_speed = window_init(env_state.ceiling_zone_steps.shape[0])
    return env_state._replace(ceiling_zone_steps=zone_steps, largest_torso_speed=largest_speed)
