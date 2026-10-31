"""How much of the maze each copy has seen.

The visited map itself is maintained inside the compiled iteration (a fixed-shape boolean array
per copy); the environment says which cells count (its `open_cells` mask), and this module turns
the map into a fraction.
"""
import numpy as np


def coverage_fraction(visited, open_cells):
    """Fraction of the open cells each copy has visited, [C] — one device-to-host copy."""
    return np.asarray(visited[:, open_cells].mean(axis=1))
