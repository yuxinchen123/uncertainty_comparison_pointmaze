"""Oracle visit counts over discretised position and velocity — the fused form.

The bonus of a state is `min(1, n**decay)` where n is how often that discrete state has been
reached, counted per copy. It is an oracle because nothing is learned: the count table IS the
knowledge, and a state the agent has never reached scores the maximum, 1.0.

Two array operations per iteration and no more: one scatter that adds the whole rollout's visits
into the table, and one gather that reads the updated counts back for the rollout's rows. See
`spec.md` beside this file for the discretisation, the update schedule, and why the schedule is
the one it is.
"""
import jax.numpy as jnp
import numpy as np

from ... import F32
from ...envs.pointmaze.pm_common import MAPS
from ..protocol import BonusFunctions
from .config import VisitCountConfig


def build(cfg, env_cfg, vc_cfg: VisitCountConfig, name: str, n_copies: int, base_seed: int,
          copy_seed_index):
    """Bind this bonus to one run's maze, copy count and decay; `name` is the preset it came from."""
    # the maze decides which cells exist and which are walls; the table covers every cell,
    # including walls, so the index arithmetic stays a plain multiply-add
    wall = np.asarray(MAPS[env_cfg.map_name])
    rows, cols = wall.shape
    bins, clip = vc_cfg.velocity_bins, vc_cfg.velocity_clip
    decay = vc_cfg.decay
    per_cell = bins * bins
    table_size = rows * cols * per_cell
    open_cell = jnp.asarray(wall.reshape(-1) == 0)
    copy_rows = jnp.arange(n_copies)[:, None]

    def init():
        """No parameters; the state is one integer table per copy."""
        return {}, {"counts": jnp.zeros((n_copies, table_size), jnp.int32)}

    def table_index(next_obs):
        """[C, M, 4] observations -> (flat table index [C, M], open-cell flag [C, M]).

        before: one observation (x, y, vx, vy) = (-4.5, -3.0, 0.0, 0.0) on the large maze
        after:  column 1, row 7 (row 0 is the top), velocity bins 5 and 5, so the cell is
                7*12 + 1 = 85 and the index is 85*100 + 5*10 + 5 = 8555
        """
        # position: the world is centred on the origin, one cell is one metre, row 0 is the top
        jj = jnp.clip((next_obs[..., 0] + cols / 2.0).astype(jnp.int32), 0, cols - 1)
        ii = jnp.clip((rows / 2.0 - next_obs[..., 1]).astype(jnp.int32), 0, rows - 1)
        cell = ii * cols + jj
        # velocity: clipped to the environment's own limit, then cut into equal bins per axis
        vx = jnp.clip(next_obs[..., 2], -clip, clip)
        vy = jnp.clip(next_obs[..., 3], -clip, clip)
        bx = jnp.clip(((vx + clip) / (2 * clip) * bins).astype(jnp.int32), 0, bins - 1)
        by = jnp.clip(((vy + clip) / (2 * clip) * bins).astype(jnp.int32), 0, bins - 1)
        return cell * per_cell + bx * bins + by, open_cell[cell]

    def post_rollout(params, state, next_obs_flat, scored):
        """Count the whole rollout, then read every row's bonus out of the updated table."""
        if scored is not None:
            raise ValueError("the visit-count bonus scores a rollout only after it has counted "
                             "it, so it cannot be given per-step scores from the rollout scan")
        idx, is_open = table_index(next_obs_flat)

        # one scatter: every open-cell visit adds one to its own entry, a wall visit adds zero.
        # before: counts [C, rows*cols*100], idx [C, T*N] with repeats where a state recurred
        # after:  counts with each entry raised by how many rows of this rollout landed on it
        counts = state["counts"].at[copy_rows, idx].add(is_open.astype(jnp.int32))

        # one gather: the bonus of a row is read from the counts AFTER the rollout was counted,
        # so a state reached once in this rollout has count 1 and scores min(1, 1**decay) = 1.0.
        # A wall cell was never counted, so it scores the maximum by the same rule that an
        # unvisited cell does.
        n = jnp.take_along_axis(counts, idx, axis=1)
        bonus = jnp.where(is_open,
                          jnp.minimum(1.0, jnp.maximum(n, 1).astype(F32) ** decay), 1.0)
        return {"counts": counts}, bonus.astype(F32), {}

    def loss(params, batch):
        """Nothing is trained: the table is the model."""
        return 0.0

    def metrics(params, state):
        """Nothing beyond the intrinsic reward the composer already reports."""
        return {}

    return BonusFunctions(name=name, init=init, prime=None, rollout_step=None,
                          post_rollout=post_rollout, loss=loss, metrics=metrics)
