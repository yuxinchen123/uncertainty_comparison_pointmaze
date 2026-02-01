"""Visit count heatmap visualization (07_reconstruction, same format as 06rl_integration)."""
from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_visit_count_heatmap(
    visit_counts: np.ndarray,
    maze_map: Optional[np.ndarray] = None,
    title: str = "Visit Count Heatmap",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    goal_cell: Optional[Tuple[int, int]] = None,
    start_cell: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create a heatmap of visit counts (same format as 06rl_integration).
    Walls masked, goal (green star), start (blue square), colorbar with integer ticks, cell annotations.
    """
    fig, ax = plt.subplots(figsize=figsize)
    vis_counts = visit_counts.copy().astype(float)

    if maze_map is not None:
        maze_map = np.array(maze_map)
        wall_mask = maze_map == 1
        vis_counts[wall_mask] = np.nan

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        valid_max = np.nanmax(vis_counts) if not np.isnan(vis_counts).all() else 1.0
        vmax = max(valid_max, 1.0)
    if vmax < 1.0:
        vmax = 1.0

    im = ax.imshow(
        vis_counts,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Visit Count", rotation=270, labelpad=20)

    # Colorbar ticks: integer values based on vmax (same as 06rl_integration)
    if vmax <= 10:
        ticks = np.arange(0, int(vmax) + 1, 1)
    elif vmax <= 50:
        ticks = np.arange(0, int(vmax) + 5, 5)
    elif vmax <= 100:
        ticks = np.arange(0, int(vmax) + 10, 10)
    elif vmax <= 500:
        ticks = np.arange(0, int(vmax) + 50, 50)
    else:
        ticks = np.arange(0, int(vmax) + 100, 100)
    if len(ticks) > 8:
        ticks = np.linspace(0, vmax, 6).astype(int)
    if len(ticks) < 2:
        ticks = np.array([0, int(vmax)])
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])

    if goal_cell is not None:
        row, col = goal_cell
        if 0 <= row < visit_counts.shape[0] and 0 <= col < visit_counts.shape[1]:
            ax.scatter(col, row, c="green", marker="*", s=500, edgecolors="black", linewidths=2, label="Goal", zorder=10)
    if start_cell is not None:
        row, col = start_cell
        if 0 <= row < visit_counts.shape[0] and 0 <= col < visit_counts.shape[1]:
            ax.scatter(col, row, c="blue", marker="s", s=300, edgecolors="black", linewidths=2, label="Start", zorder=10)

    # Text annotations for visit counts (grids <= 200 cells, same as 06)
    if visit_counts.shape[0] * visit_counts.shape[1] <= 200:
        for i in range(visit_counts.shape[0]):
            for j in range(visit_counts.shape[1]):
                if maze_map is None or maze_map[i, j] == 0:
                    count = visit_counts[i, j]
                    if count > 0:
                        ax.text(j, i, str(count), ha="center", va="center",
                               color="black" if vis_counts[i, j] < vmax * 0.5 else "white",
                               fontsize=8, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.invert_yaxis()
    if goal_cell is not None or start_cell is not None:
        ax.legend(loc="upper right")
    plt.tight_layout()
    return fig
