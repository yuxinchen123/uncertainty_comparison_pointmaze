"""Visit count heatmap visualization (07_reconstruction, same format as 06rl_integration)."""
import textwrap
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
    Create a heatmap of visit counts.
    Walls masked, G/S text labels with visit counts in cells, colorbar with integer ticks.
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

    # Text annotations: G/S labels, visit ratio, and visit counts (grids <= 200 cells)
    total_visits = int(visit_counts.sum())
    is_goal = (goal_cell[0], goal_cell[1]) if goal_cell else (None, None)
    is_start = (start_cell[0], start_cell[1]) if start_cell else (None, None)
    if visit_counts.shape[0] * visit_counts.shape[1] <= 200:
        for i in range(visit_counts.shape[0]):
            for j in range(visit_counts.shape[1]):
                if maze_map is not None and maze_map[i, j] == 1:
                    continue
                count = visit_counts[i, j]
                label_parts = []
                if (i, j) == is_goal:
                    label_parts.append("G")
                if (i, j) == is_start:
                    label_parts.append("S")
                if count > 0 and total_visits > 0:
                    ratio = count / total_visits
                    label_parts.append(f"{ratio:.2f}")
                if count > 0:
                    label_parts.append(str(count))
                if label_parts:
                    label = "\n".join(label_parts)
                    txt_color = "black" if vis_counts[i, j] < vmax * 0.5 else "white"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, fontweight="bold", color=txt_color)

    wrapped_title = "\n".join(textwrap.wrap(title, width=50))
    ax.set_title(wrapped_title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig
