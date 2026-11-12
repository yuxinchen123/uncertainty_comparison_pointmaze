"""The bonus recorded on the maze grid, once per logging window (no change to training).

Two things are recorded per policy:

1. The bonus at a fixed set of probe observations, one per open maze cell: the cell's centre with
   everything else at rest. The probes are scored with the bonus's parameters and state as they
   stand at the end of the window, each on its own, so the numbers say what a first visit to
   that cell would pay.
2. The number of states the window's rollouts reached in each cell. The compiled iteration adds
   them up in the training state (`maze_grid_visits`, [C, rows x cols]); the driver reads them at
   the window's end and sets them back to zero.

The driver also writes the same fields once before the first training iteration (record
`maze_grid_at_start`), which is the bonus's start on the grid.

A maze family takes part by giving a probe builder in `PROBE_BUILDERS`. PointMaze has one. An
AntMaze builder has to place the torso at the cell's centre in a resting pose; the visit counts
already work for it, because they use the environment's own cell index.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .. import F32
from ..envs.pointmaze.pm_common import cell_center
from .state import named_params, stored_params
from .train_step import intrinsic_divisor

# the fields one window record gains, in the order they are written
WINDOW_FIELDS = ("probe_bonus_per_copy", "probe_intrinsic_reward_per_copy",
                 "intrinsic_reward_divisor_per_copy", "cell_visits_per_copy")


def open_cell_list(env):
    """The open cells of the environment's maze as (row, column) pairs, row by row.

    before: a 3 x 3 map whose only open cells are the middle row's last two
    after:  [(1, 1), (1, 2)]
    """
    open_cells = np.asarray(env.open_cells).reshape(env.rows, env.cols)
    return [(int(row), int(col)) for row, col in zip(*np.nonzero(open_cells))]


def pointmaze_probe_observations(env):
    """One PointMaze observation per open cell: the cell's centre, zero velocity.

    before: large map, open cell (7, 1)
    after:  observation (-4.5, -3.0, 0.0, 0.0)
    Returns float32 [P, 4] in the order of `open_cell_list`.
    """
    centres = [cell_center(cell, env.rows, env.cols) for cell in open_cell_list(env)]
    return np.asarray([(x, y, 0.0, 0.0) for x, y in centres], np.float32)


# the probe builder of each environment class that has one
PROBE_BUILDERS = {"JaxPointMaze": pointmaze_probe_observations}


def probe_observations(env):
    """The probe observations of an environment, [P, obs] float32; refused where none is built."""
    builder = PROBE_BUILDERS.get(type(env).__name__)
    if builder is None:
        raise ValueError(f"no maze-grid probe observations are built for {type(env).__name__}; "
                         f"the environments with one are {sorted(PROBE_BUILDERS)}")
    return builder(env)


def build_probe(cfg, env, bonus, layout):
    """The program that scores the probes of every policy from a training state.

    Returns probe(state) -> {"bonus": [C, P] the bonus before the divisor, "divisor": [C] what
    the trainer divides each policy's bonus by, "visits": [C, P] the window's visits to each
    probe's cell}. It reads the state and changes nothing.
    """
    if bonus.score_alone is None:
        raise ValueError(f"the {bonus.name} bonus cannot score an observation on its own, so "
                         "it cannot be recorded on the maze grid")
    probes = probe_observations(env)
    # before: probes [P, obs]; after: [C, P, obs], the same probes for every policy
    observations = jnp.broadcast_to(jnp.asarray(probes, F32)[None],
                                    (cfg.n_copies,) + probes.shape)
    open_index = jnp.asarray(np.flatnonzero(np.asarray(env.open_cells)))

    def probe(state):
        """Score the probes with the bonus as it stands; read the divisor and the visits."""
        params = named_params(stored_params(state, layout), layout)["bonus"]
        return {"bonus": bonus.score_alone(params, state.bonus_state, observations),
                "divisor": intrinsic_divisor(bonus.reward_divisor, state.agent_state["int_rms"],
                                             cfg.n_copies),
                "visits": state.agent_state["maze_grid_visits"][:, open_index]}

    return jax.jit(probe)


def read_window(state, probe) -> dict:
    """The window record's maze-grid fields, as lists; numbers keep six significant digits.

    before: probe(state) = bonus [[0.5, 0.25]], divisor [2.0], visits [[3, 0]]
    after:  {"probe_bonus_per_copy": [[0.5, 0.25]],
             "probe_intrinsic_reward_per_copy": [[0.25, 0.125]],
             "intrinsic_reward_divisor_per_copy": [2.0], "cell_visits_per_copy": [[3, 0]]}
    """
    read = jax.device_get(probe(state))
    bonus = np.asarray(read["bonus"], np.float64)
    divisor = np.asarray(read["divisor"], np.float64)
    rounded = lambda values: [float(f"{x:.6g}") for x in values]
    return {"probe_bonus_per_copy": [rounded(row) for row in bonus],
            "probe_intrinsic_reward_per_copy": [rounded(row) for row in bonus / divisor[:, None]],
            "intrinsic_reward_divisor_per_copy": rounded(divisor),
            "cell_visits_per_copy": np.asarray(read["visits"]).tolist()}


def clear_window(agent_state: dict) -> dict:
    """The agent's state with the window's visit counts back at zero."""
    return {**agent_state,
            "maze_grid_visits": jnp.zeros_like(agent_state["maze_grid_visits"])}
