"""
Plot the PointMaze_Large-v3 maze and its discretization for the development document.

Two panels:
  (a) Position discretization: the 9x12 maze grid in world coordinates
      (x in [-6,6], y in [-4.5,4.5]), walls shaded, the 1.0x1.0 cell grid drawn,
      and the sweep's fixed start / goal cells marked.
  (b) Velocity discretization: the 10x10 bins over (vx, vy) in [-5,5]^2, each bin 1.0x1.0.

Together (a) and (b) define the joint gt_position_velocity cell of shape (9,12,10,10).

The maze map and the cell->world mapping are read from the live environment so the
figure stays authoritative (no hard-coded maze).

Walls (and the velocity-bin checkerboard) are drawn as filled Rectangle patches, NOT via
imshow. An imshow raster is re-sampled at the output resolution when saved to a vector PDF,
which drops most wall cells -- the walls render in a PNG but vanish in the embedded PDF that
the document uses. Rectangle patches are true vector objects, so the walls render identically
in the PNG and the PDF. Run with the project env:
    conda run -n exploration python plot_pointmaze_discretization.py
"""
import os
import sys

# make the 07_reconstruction package root importable regardless of the working directory
# (this script lives at 07_reconstruction/development_document/code/<date>/ -> three levels up)
_RECON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _RECON_ROOT not in sys.path:
    sys.path.insert(0, _RECON_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: render to file, no display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers PointMaze envs)

from env_wrapper.point_maze_utils import (
    get_maze_map,
    select_fixed_goal_bottom_left,
    select_fixed_goal_top_right,
)

# World extent of the position grid (from env_wrapper/point_maze_utils.observation_to_grid).
LEFT_EDGE, RIGHT_EDGE = -6.0, 6.0
BOTTOM_EDGE, TOP_EDGE = -4.5, 4.5
# Velocity extent and binning (from PositionVelocityVisitCountWrapper / velocity_to_grid).
V_MIN, V_MAX = -5.0, 5.0
N_VEL_BINS = 10


def cell_to_world_rect(row, col, grid_rows, grid_cols):
    """Return the world-coordinate rectangle (x_left, y_bottom, width, height) of grid cell (row, col).

    Row 0 is the top of the world (high y); the last row is the bottom (low y), matching
    observation_to_grid: y near TOP_EDGE -> row 0, y near BOTTOM_EDGE -> grid_rows-1.
    """
    # cell size in world units (each cell is 1.0 x 1.0 for the Large maze)
    cell_w = (RIGHT_EDGE - LEFT_EDGE) / grid_cols
    cell_h = (TOP_EDGE - BOTTOM_EDGE) / grid_rows
    # x grows with column from the left edge; y grows downward from the top edge as row grows
    x_left = LEFT_EDGE + col * cell_w
    y_bottom = TOP_EDGE - (row + 1) * cell_h
    return x_left, y_bottom, cell_w, cell_h


def verify_rect_against_env(env, grid_rows, grid_cols):
    """Self-check: the center of each cell_to_world_rect must match the env's own cell_rowcol_to_xy."""
    # compare rectangle centers to the environment's authoritative cell->xy mapping
    maze = env.unwrapped.maze
    for (row, col) in [(0, 0), (0, grid_cols - 1), (grid_rows - 1, 0),
                       (grid_rows - 1, grid_cols - 1), (4, 6)]:
        x_left, y_bottom, w, h = cell_to_world_rect(row, col, grid_rows, grid_cols)
        cx, cy = x_left + w / 2.0, y_bottom + h / 2.0
        ex, ey = maze.cell_rowcol_to_xy(np.array([row, col]))
        assert abs(cx - ex) < 1e-9 and abs(cy - ey) < 1e-9, \
            f"cell ({row},{col}) center ({cx},{cy}) != env ({ex},{ey})"


def draw_position_panel(ax, maze_map, goal_cell, start_cell):
    """Draw the maze walls, the 9x12 cell discretization, and the fixed start/goal markers in world coordinates.

    Walls are filled Rectangle patches (one per wall cell), placed with the same env-verified
    cell_to_world_rect mapping as the start/goal markers -- so walls and markers cannot disagree,
    and (unlike an imshow raster) the walls survive saving to the vector PDF the document uses.
    """
    grid_rows, grid_cols = maze_map.shape

    # one gray rectangle per wall cell (maze_map == 1); open cells (== 0) stay white background.
    # cell_to_world_rect maps array (row, col) to its world rectangle, with row 0 at the top
    # (high y) -- the same mapping verify_rect_against_env checks against the env, and the same
    # mapping the goal/start markers below use, so the maze and the markers share one orientation.
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row, col] == 1:
                wx, wy, ww, wh = cell_to_world_rect(row, col, grid_rows, grid_cols)
                ax.add_patch(Rectangle((wx, wy), ww, wh,
                                       facecolor="#5a5a5a", edgecolor="none", zorder=1))

    # discretization grid lines: one line at every cell boundary (the 1.0-spaced cuts)
    x_edges = np.linspace(LEFT_EDGE, RIGHT_EDGE, grid_cols + 1)
    y_edges = np.linspace(BOTTOM_EDGE, TOP_EDGE, grid_rows + 1)
    for x in x_edges:
        ax.axvline(x, color="#9a9a9a", linewidth=0.6, zorder=2)
    for y in y_edges:
        ax.axhline(y, color="#9a9a9a", linewidth=0.6, zorder=2)

    # mark the fixed goal and start at their cell centers (world coordinates)
    gx, gy, gw, gh = cell_to_world_rect(goal_cell[0], goal_cell[1], grid_rows, grid_cols)
    sx, sy, sw, sh = cell_to_world_rect(start_cell[0], start_cell[1], grid_rows, grid_cols)
    ax.scatter([gx + gw / 2], [gy + gh / 2], marker="*", s=320,
               color="#1b9e77", edgecolor="black", linewidth=0.6, zorder=4)
    ax.scatter([sx + sw / 2], [sy + sh / 2], marker="o", s=130,
               color="#377eb8", edgecolor="black", linewidth=0.6, zorder=4)

    # axes: world coordinate ticks at the cell boundaries, labeled axes, title
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.set_xlabel(r"position $x$")
    ax.set_ylabel(r"position $y$")
    ax.set_xlim(LEFT_EDGE, RIGHT_EDGE)
    ax.set_ylim(BOTTOM_EDGE, TOP_EDGE)
    ax.set_aspect("equal")  # square cells (previously supplied by imshow aspect="equal")
    ax.tick_params(labelsize=7)
    ax.set_title(r"(a) Position grid: $9\times12$ cells, each $1.0\times1.0$", fontsize=9)

    # legend tying markers to the start/goal cells (with their (row,col) indices)
    legend_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#1b9e77",
               markeredgecolor="black", markersize=15,
               label=f"goal cell (row {goal_cell[0]}, col {goal_cell[1]})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#377eb8",
               markeredgecolor="black", markersize=10,
               label=f"start cell (row {start_cell[0]}, col {start_cell[1]})"),
        Rectangle((0, 0), 1, 1, facecolor="#5a5a5a", edgecolor="none", label="wall"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7,
              frameon=False, handletextpad=0.4, columnspacing=1.0)


def draw_velocity_panel(ax):
    """Draw the 10x10 velocity binning over (vx, vy) in [-5,5]^2 as a faint checkerboard grid.

    The checkerboard is drawn as Rectangle patches (not imshow) for the same vector-PDF reason
    as the maze walls: an imshow checkerboard is decimated when saved to the embedded PDF.
    """
    edges = np.linspace(V_MIN, V_MAX, N_VEL_BINS + 1)  # bin boundaries: [-5, -4, ..., 5]

    # faint checkerboard: shade every other (vx_bin, vy_bin) cell so the 10x10 binning is visible
    # at a glance (velocity space has no walls). bin i spans vx in [edges[i], edges[i+1]] (width 1.0),
    # bin j spans vy in [edges[j], edges[j+1]]; shade when i + j is odd.
    for i in range(N_VEL_BINS):
        for j in range(N_VEL_BINS):
            if (i + j) % 2 == 1:
                ax.add_patch(Rectangle((edges[i], edges[j]), 1.0, 1.0,
                                       facecolor="#e8eef5", edgecolor="none", zorder=1))

    # bin-boundary grid lines at every 1.0 step from -5 to 5
    for e in edges:
        ax.axvline(e, color="#9a9a9a", linewidth=0.6, zorder=2)
        ax.axhline(e, color="#9a9a9a", linewidth=0.6, zorder=2)

    # axes: ticks on bin boundaries, labeled velocity axes, title
    ax.set_xticks(edges)
    ax.set_yticks(edges)
    ax.set_xlabel(r"velocity $v_x$")
    ax.set_ylabel(r"velocity $v_y$")
    ax.set_xlim(V_MIN, V_MAX)
    ax.set_ylim(V_MIN, V_MAX)
    ax.set_aspect("equal")  # square bins (previously supplied by imshow aspect="equal")
    ax.tick_params(labelsize=7)
    ax.set_title(r"(b) Velocity bins: $10\times10$, each $1.0\times1.0$", fontsize=9)


def build_figure(maze_map, goal_cell, start_cell, out_path_pdf, out_path_png):
    """Assemble the two-panel discretization figure and save it as PDF and PNG."""
    # two side-by-side panels; width ratio ~ world widths (12 vs 10) for visual balance
    fig, (ax_pos, ax_vel) = plt.subplots(
        1, 2, figsize=(11.0, 4.4),
        gridspec_kw={"width_ratios": [12.0, 10.0]},
    )
    draw_position_panel(ax_pos, maze_map, goal_cell, start_cell)
    draw_velocity_panel(ax_vel)
    fig.tight_layout()

    # save vector PDF (for LaTeX) and a raster PNG (for quick viewing)
    fig.savefig(out_path_pdf, bbox_inches="tight")
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    """Build the PointMaze + discretization figure from the live PointMaze_Large-v3 environment."""
    # build the base env with the sweep's task flags (fixed, continuing goal) to read its maze map
    env = gym.make("PointMaze_Large-v3", continuing_task=True, reset_target=False)
    maze_map = get_maze_map(env)
    if maze_map is None:
        raise ValueError("could not read maze_map from PointMaze_Large-v3")
    grid_rows, grid_cols = maze_map.shape

    # the sweep uses goal_position=top_right; start is the opposite corner (see 04_many_exploration_method.py)
    goal_cell = select_fixed_goal_top_right(env)
    start_cell = select_fixed_goal_bottom_left(env)

    # check our rectangle math matches the env's own cell->world mapping before plotting
    verify_rect_against_env(env, grid_rows, grid_cols)

    # the goal and start markers must sit on open cells, not walls (catches any orientation flip:
    # a flipped maze would place a marker on a wall cell)
    assert maze_map[goal_cell[0], goal_cell[1]] == 0, f"goal cell {goal_cell} is a wall"
    assert maze_map[start_cell[0], start_cell[1]] == 0, f"start cell {start_cell} is a wall"

    # write outputs next to this script
    here = os.path.dirname(os.path.abspath(__file__))
    out_pdf = os.path.join(here, "pointmaze_discretization.pdf")
    out_png = os.path.join(here, "pointmaze_discretization.png")
    build_figure(maze_map, goal_cell, start_cell, out_pdf, out_png)
    print(f"maze_map shape: {maze_map.shape}")
    print(f"goal cell (row,col): {goal_cell}; start cell (row,col): {start_cell}")
    print(f"wrote: {out_pdf}")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
