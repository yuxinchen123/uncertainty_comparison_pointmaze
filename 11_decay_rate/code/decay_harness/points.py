"""Fixed point sets and the checkpoint grid for the decay-rate experiments.

Ported from 07_reconstruction/convergence_train.py so every geometric convention (maze size,
cell centers, sub-cell midpoints) matches the prior convergence runs exactly. This module is
part of the FIXED harness: the experiment loop never edits it (program.md).
"""
import numpy as np

# PointMaze_Large-v3 geometry (verified 2026-07-19 against the live environment): 9 cells high,
# 12 wide, cell size 1x1, x in [-6, 6], y in [-4.5, 4.5]; cell (row, col) center =
# (col - 5.5, 4 - row), row 0 at the top and y up. 46 of the 108 cells are free.
MAZE_ROWS = 9
MAZE_COLS = 12

POINT_SET_CHOICES = ("center_square", "top_right_cell", "cell_midpoints")


def checkpoint_steps(n_steps: int) -> list:
    """Log-spaced checkpoint steps: unique values of round(2^(j/8)) up to n_steps (79 for 4096)."""
    # quarter splits of the half-exponent 2^{k/2} grid; small-end duplicates collapse.
    # before (j=0..12): 1.00, 1.09, 1.19, 1.30, 1.41, 1.54, 1.68, 1.83, 2.00, 2.18, 2.38, ...
    # after round + dedup: 1, 2, 3, ... (for n_steps=4096 the list has 79 entries, ending 4096)
    steps = []
    j = 0
    while round(2 ** (j / 8.0)) <= n_steps:
        s = round(2 ** (j / 8.0))
        if not steps or s != steps[-1]:
            steps.append(s)
        j += 1
    return steps


def point_set(name: str) -> np.ndarray:
    """Build one fixed evaluation point set as a (P, 4) float32 array of [x, y, vx=0, vy=0] rows."""
    # 10x10 sub-cell midpoints of a 1x1 square: offsets -0.45, -0.35, ..., +0.45 (spacing 0.1)
    offs = -0.5 + (np.arange(10) + 0.5) * 0.1
    if name == "center_square":
        # the 1x1 square centered at (0, 0); its left half overlaps wall cell (row 4, col 5) —
        # kept deliberately (the networks are defined at wall points too), as in the prior runs
        xs, ys = np.meshgrid(offs, offs, indexing="ij")
    elif name == "top_right_cell":
        # the top-right free cell (row 1, col 10) = the fixed top_right goal cell, center (4.5, 3)
        xs, ys = np.meshgrid(4.5 + offs, 3.0 + offs, indexing="ij")
    elif name == "cell_midpoints":
        # the midpoint of EVERY maze cell of the 9x12 grid, walls included (46 free + 62 wall).
        # before: (row, col) = (0, 0) .. (8, 11) -> after: x = col - 5.5 in {-5.5 .. 5.5},
        # y = 4 - row in {4 .. -4}; 108 points
        cols, rows = np.meshgrid(np.arange(MAZE_COLS), np.arange(MAZE_ROWS), indexing="ij")
        xs, ys = cols - 5.5, 4.0 - rows
    else:
        raise ValueError(f"point_set must be one of {POINT_SET_CHOICES}; got {name!r}")
    zeros = np.zeros(xs.size)
    return np.stack([xs.ravel(), ys.ravel(), zeros, zeros], axis=1).astype(np.float32)


def nonuniform_probabilities(points: np.ndarray) -> np.ndarray:
    """Fixed sampling distribution for the nonuniform regime: one decade of probability across
    the x extent, so left-edge positions are trained on ~10x as often as right-edge ones."""
    # before: points[:, 0] = x in [-5.5, 5.5] (cell_midpoints); after: unnormalized weight
    # 10^(-(x - x_min) / (x_max - x_min)) in [0.1, 1], then normalized to sum 1
    x = points[:, 0].astype(np.float64)
    span = x.max() - x.min()
    w = 10.0 ** (-(x - x.min()) / span)
    return w / w.sum()
