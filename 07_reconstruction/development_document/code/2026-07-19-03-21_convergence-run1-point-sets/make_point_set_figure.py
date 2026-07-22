"""Point-set figure for convergence run 1: the PointMaze_Large-v3 maze on its true 9 x 12 cell
grid with the three fixed evaluation point sets marked. Walls are drawn as Rectangle patches
(never imshow — imshow walls vanish when the figure is embedded as a vector PDF) in the world
convention used throughout the document: row 0 at top, x right, y up; cell (row, col) center =
(col - 5.5, 4 - row); x in [-6, 6], y in [-4.5, 4.5].

Run with the canonical env python:
  /p/rlprojects/RND/.venvs/exploration/bin/python make_point_set_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# import the EXACT point sets the trainer uses (single source of truth: convergence_train.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from convergence_train import point_set  # noqa: E402

# the maze map: gymnasium-robotics LARGE_MAZE, verified byte-identical to the map inside the
# project's env (gym.spec('PointMaze_Large-v3').kwargs['maze_map'] == LARGE_MAZE, probes 2026-07-19)
from gymnasium_robotics.envs.maze.maps import LARGE_MAZE  # noqa: E402

LEFT, RIGHT, BOTTOM, TOP = -6.0, 6.0, -4.5, 4.5


def main() -> None:
    """Draw the maze grid + walls and overlay the three point sets; save point_sets.pdf."""
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    # walls as filled rectangles; cell (row, col) spans x in [LEFT+col, LEFT+col+1],
    # y in [TOP-row-1, TOP-row] (row 0 at top, y up)
    for r, row in enumerate(LARGE_MAZE):
        for c, v in enumerate(row):
            if v == 1:
                ax.add_patch(Rectangle((LEFT + c, TOP - (r + 1)), 1.0, 1.0,
                                       facecolor="0.82", edgecolor="none", zorder=0))
    # the true 9 x 12 cell grid (the discretization this run uses, unlike Figure 12's 10x10 boxes)
    for gx in range(int(LEFT), int(RIGHT) + 1):
        ax.axvline(gx, color="0.65", lw=0.5, zorder=1)
    for k in range(10):
        ax.axhline(BOTTOM + k, color="0.65", lw=0.5, zorder=1)
    # the three point sets, imported from the trainer so the figure shows exactly what is trained
    cells = point_set("cell_midpoints")
    center = point_set("center_square")
    topright = point_set("top_right_cell")
    ax.scatter(cells[:, 0], cells[:, 1], s=16, c="black", marker="o", zorder=3,
               label="env 2: all 108 cell midpoints")
    ax.scatter(center[:, 0], center[:, 1], s=5, c="tab:blue", marker="o", zorder=3,
               label="env 1.1: 10x10 points, 1x1 square at (0, 0)")
    ax.scatter(topright[:, 0], topright[:, 1], s=5, c="tab:orange", marker="o", zorder=3,
               label="env 1.2: 10x10 points, goal cell (row 1, col 10)")
    # outline the two env-1 squares so their 1x1 extents are visible
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="none",
                           edgecolor="tab:blue", lw=1.8, zorder=4))
    ax.add_patch(Rectangle((4.0, 2.5), 1.0, 1.0, facecolor="none",
                           edgecolor="tab:orange", lw=1.8, zorder=4))
    ax.set_xlim(LEFT - 0.3, RIGHT + 0.3)
    ax.set_ylim(BOTTOM - 0.3, TOP + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False, fontsize=9)
    fig.tight_layout()
    out = Path(__file__).with_name("point_sets.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
