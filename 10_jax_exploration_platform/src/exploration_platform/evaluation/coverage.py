"""How much of the maze each copy has seen.

The visited map itself is maintained inside the compiled iteration (a fixed-shape boolean array
per copy); this module only says which cells count and turns the map into a fraction.
"""
import jax.numpy as jnp
import numpy as np

from ..envs.pointmaze.pm_common import MAPS


def open_cell_mask(map_name: str):
    """Which of the maze's cells are open, as a flat boolean array over rows*cols.

    before: the large map, 9 rows x 12 columns, 1 = wall
    after:  [108] booleans, True on the 45 cells a copy can actually reach
    """
    wall = np.asarray(MAPS[map_name]).reshape(-1) == 1
    return jnp.asarray(~wall), int(wall.size)


def coverage_fraction(visited, open_cells):
    """Fraction of the open cells each copy has visited, [C] — one device-to-host copy."""
    return np.asarray(visited[:, open_cells].mean(axis=1))
