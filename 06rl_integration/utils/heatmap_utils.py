"""
Utilities for creating visit count heatmaps for visualization.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Optional, Tuple


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
    Create a heatmap visualization of visit counts.
    
    Args:
        visit_counts: 2D array of visit counts (grid_rows x grid_cols)
        maze_map: Optional maze map to mask walls (1 = wall, 0 = open)
        title: Title for the plot
        cmap: Colormap name (default: "YlOrRd" - yellow to red)
        figsize: Figure size (width, height)
        vmin: Minimum value for colormap (None = auto)
        vmax: Maximum value for colormap (None = auto)
        goal_cell: Optional (row, col) tuple to mark goal location
        start_cell: Optional (row, col) tuple to mark start location
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a copy for visualization (we'll mask walls)
    vis_counts = visit_counts.copy().astype(float)
    
    # Mask walls if maze_map is provided
    if maze_map is not None:
        # Ensure maze_map is a numpy array
        if not isinstance(maze_map, np.ndarray):
            maze_map = np.array(maze_map)
        # Set walls to NaN so they appear as white/empty in the heatmap
        vis_counts[maze_map == 1] = np.nan
    
    # Create heatmap
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        # Use max of non-NaN values, or 1 if all are NaN
        valid_max = np.nanmax(vis_counts) if not np.isnan(vis_counts).all() else 1.0
        vmax = max(valid_max, 1.0)
    
    im = ax.imshow(
        vis_counts,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        aspect='auto'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count', rotation=270, labelpad=20)
    
    # Mark goal location if provided
    if goal_cell is not None:
        row, col = goal_cell
        if 0 <= row < visit_counts.shape[0] and 0 <= col < visit_counts.shape[1]:
            ax.scatter(col, row, c='green', marker='*', s=500, 
                      edgecolors='black', linewidths=2, label='Goal', zorder=10)
    
    # Mark start location if provided
    if start_cell is not None:
        row, col = start_cell
        if 0 <= row < visit_counts.shape[0] and 0 <= col < visit_counts.shape[1]:
            ax.scatter(col, row, c='blue', marker='s', s=300,
                      edgecolors='black', linewidths=2, label='Start', zorder=10)
    
    # Add text annotations for visit counts (optional, can be disabled for large grids)
    if visit_counts.shape[0] * visit_counts.shape[1] <= 200:  # Only for small grids
        for i in range(visit_counts.shape[0]):
            for j in range(visit_counts.shape[1]):
                if maze_map is None or maze_map[i, j] == 0:  # Only annotate open cells
                    count = visit_counts[i, j]
                    if count > 0:
                        ax.text(j, i, str(count), ha='center', va='center',
                               color='black' if vis_counts[i, j] < vmax * 0.5 else 'white',
                               fontsize=8, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Invert y-axis to match matrix indexing (row 0 at top)
    ax.invert_yaxis()
    
    # Add legend if we have goal/start markers
    if goal_cell is not None or start_cell is not None:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig


def create_multi_goal_heatmap(
    visit_counts: np.ndarray,
    maze_map: Optional[np.ndarray] = None,
    goal_cells: Optional[list] = None,
    title_prefix: str = "Visit Count Heatmap",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (10, 8),
) -> list:
    """
    Create heatmaps for each goal in multi-goal mode.
    
    Args:
        visit_counts: 3D array of visit counts (num_goals x grid_rows x grid_cols)
        maze_map: Optional maze map to mask walls
        goal_cells: Optional list of (row, col) tuples for each goal
        title_prefix: Prefix for plot titles
        cmap: Colormap name
        figsize: Figure size
        
    Returns:
        List of matplotlib Figure objects (one per goal)
    """
    num_goals = visit_counts.shape[0]
    figures = []
    
    for goal_idx in range(num_goals):
        goal_visit_counts = visit_counts[goal_idx]
        # Handle goal_cells - it should be a list of tuples or None
        if goal_cells is not None and isinstance(goal_cells, list) and goal_idx < len(goal_cells):
            goal_cell = goal_cells[goal_idx]
        else:
            goal_cell = None
        
        title = f"{title_prefix} - Goal {goal_idx + 1}"
        fig = create_visit_count_heatmap(
            goal_visit_counts,
            maze_map=maze_map,
            title=title,
            cmap=cmap,
            figsize=figsize,
            goal_cell=goal_cell
        )
        figures.append(fig)
    
    return figures
