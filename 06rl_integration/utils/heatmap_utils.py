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
    
    # Debug: Check input visit counts
    # print(f"DEBUG: Input visit_counts shape: {visit_counts.shape}, dtype: {visit_counts.dtype}")
    # print(f"DEBUG: Input visit_counts min: {np.min(visit_counts)}, max: {np.max(visit_counts)}, sum: {np.sum(visit_counts)}")
    
    # Create a copy for visualization (we'll mask walls)
    vis_counts = visit_counts.copy().astype(float)
    
    # Mask walls if maze_map is provided
    if maze_map is not None:
        # Ensure maze_map is a numpy array
        if not isinstance(maze_map, np.ndarray):
            maze_map = np.array(maze_map)
        # Set walls to NaN so they appear as white/empty in the heatmap
        # Only mask walls, keep visit counts for open cells
        wall_mask = (maze_map == 1)
        vis_counts[wall_mask] = np.nan
        # Debug: Check after masking
        # print(f"DEBUG: After masking - non-NaN cells: {np.sum(~np.isnan(vis_counts))}, max: {np.nanmax(vis_counts) if not np.isnan(vis_counts).all() else 0}")
    
    # Create heatmap
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        # Use max of non-NaN values, or 1 if all are NaN
        valid_max = np.nanmax(vis_counts) if not np.isnan(vis_counts).all() else 1.0
        vmax = max(valid_max, 1.0)
    
    # Ensure vmax is at least 1 to avoid normalization issues
    if vmax < 1.0:
        vmax = 1.0
    
    im = ax.imshow(
        vis_counts,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        aspect='auto'
    )
    
    # Add colorbar with proper formatting - ensure it shows actual data range
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count', rotation=270, labelpad=20)
    
    # Format colorbar to show actual integer values (not normalized 0-1)
    # The colorbar should reflect vmin to vmax range
    if vmax <= 10:
        # For small ranges, show all integer ticks
        ticks = np.arange(0, int(vmax) + 1, 1)
    elif vmax <= 50:
        # For medium ranges, show every 5
        step = 5
        ticks = np.arange(0, int(vmax) + step, step)
    elif vmax <= 100:
        # For larger ranges, show every 10
        step = 10
        ticks = np.arange(0, int(vmax) + step, step)
    elif vmax <= 500:
        # For large ranges, show every 50
        step = 50
        ticks = np.arange(0, int(vmax) + step, step)
    else:
        # For very large ranges, show every 100
        step = 100
        ticks = np.arange(0, int(vmax) + step, step)
    
    # Ensure we have at least 2 ticks and at most 8
    if len(ticks) > 8:
        # Reduce to 6 ticks
        ticks = np.linspace(0, vmax, 6).astype(int)
    if len(ticks) < 2:
        ticks = np.array([0, int(vmax)])
    
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])
    
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
