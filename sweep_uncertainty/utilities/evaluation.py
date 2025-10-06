import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

def get_maze_map():
    """Extract the ACTUAL maze map from the PointMaze environment"""
    import gymnasium as gym
    import gymnasium_robotics
    
    # Register and create the environment (same as notebook)
    gym.register_envs(gymnasium_robotics)
    env = gym.make("PointMaze_Large-v3")
    
    # Get the actual maze map
    unwrapped_env = env.unwrapped
    maze = unwrapped_env.maze
    maze_map = maze.maze_map
    
    # Close environment
    env.close()
    
    print(f"✓ Extracted maze map: {len(maze_map)}×{len(maze_map[0])}")
    return maze_map

def calculate_ground_truth(dataset, maze_map=None, grid_rows=9, grid_cols=12):
    """
    Calculate ground truth uncertainty using EXACT notebook method.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Initialize visit count matrix
    visit_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    
    # Count visits to each grid cell using notebook's exact method
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        if isinstance(observations, dict) and 'achieved_goal' in observations:
            positions = observations['achieved_goal']
        else:
            positions = observations
            
        for pos in positions:
            # Use the exact notebook mapping
            obs = {'achieved_goal': pos}
            _, (row, col) = observation_to_grid_notebook_exact(obs, grid_rows, grid_cols)
            
            # Convert to 0-based indexing for numpy array
            visit_counts[row-1, col-1] += 1
    
    # Calculate uncertainty matrix exactly as notebook
    uncertainty_matrix = np.full((grid_rows, grid_cols), np.nan)
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:  # Open cell
                count = visit_counts[row, col]
                if count > 0:
                    uncertainty_matrix[row, col] = 1.0 / np.sqrt(count)
                else:
                    uncertainty_matrix[row, col] = np.inf
            # Wall cells remain NaN
    
    return uncertainty_matrix

def observation_to_grid_notebook_exact(observation, grid_rows=9, grid_cols=12):
    """
    EXACT implementation from notebook: Map observation to grid cell using MATRIX-STYLE indexing.
    Coordinate system: X[-6.0, 6.0], Y[-4.5, 4.5] -> 9×12 grid
    """
    # Extract (x, y) coordinates from observation
    if isinstance(observation, dict) and 'achieved_goal' in observation:
        x, y = observation['achieved_goal'][0], observation['achieved_goal'][1]
    elif isinstance(observation, (list, tuple, np.ndarray)) and len(observation) >= 2:
        x, y = observation[0], observation[1]
    else:
        raise ValueError(f"Invalid observation format: {type(observation)}")
    
    # Environment boundaries (from notebook)
    left_edge = -6.0
    right_edge = 6.0
    bottom_edge = -4.5
    top_edge = 4.5
    total_width = 12.0   # right_edge - left_edge
    total_height = 9.0   # top_edge - bottom_edge
    
    # Calculate cell dimensions
    cell_width = total_width / grid_cols    # 12.0 / 12 = 1.0m
    cell_height = total_height / grid_rows  # 9.0 / 9 = 1.0m
    
    # X direction (columns): Left to right, 1-based indexing
    if x < left_edge:  # x < -6.0 edge case
        col = 1
    elif x >= right_edge:  # x >= 6.0 edge case  
        col = grid_cols
    else:
        # Standard case: find which column
        col = int((x - left_edge) / cell_width) + 1
        
        # Handle boundary case: boundaries count toward "previous range"
        if abs(x - (left_edge + (col-1) * cell_width)) < 1e-10 and col > 1:
            col = col - 1
            
        col = max(1, min(col, grid_cols))
    
    # Y direction (rows): Top to bottom, 1-based matrix indexing  
    if y <= bottom_edge:  # y <= -4.5 edge case
        row = grid_rows
    elif y > top_edge:   # y > 4.5 edge case
        row = 1
    else:
        # Calculate row based on Y value (top to bottom)
        y_offset_from_top = top_edge - y  # How far down from top edge
        row = int(y_offset_from_top / cell_height) + 1
        
        # Boundary handling: Y boundaries go to "next" row
        if abs(y % 1.0 - 0.5) < 1e-10:  # Y ends in .5 (boundary)
            if row > 1:  # Don't go above Row 1
                row = row - 1
            
        row = max(1, min(row, grid_rows))
    
    grid_name = f"g_({row},{col})"
    return grid_name, (row, col)

def calculate_ground_truth_from_positions(positions, maze_map=None, grid_rows=9, grid_cols=12):
    """
    Calculate ground truth uncertainty using EXACT notebook coordinate mapping.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Initialize visit count matrix
    visit_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    
    # DEBUG: Check position distribution
    print(f"Position analysis:")
    print(f"  Total positions: {len(positions)}")
    print(f"  X range: [{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}]")
    print(f"  Y range: [{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}]")
    
    # Count visits using EXACT notebook mapping
    grid_assignment_counts = {}
    
    for pos in positions:
        # Use the exact notebook mapping function
        obs = {'achieved_goal': pos}
        grid_name, (row, col) = observation_to_grid_notebook_exact(obs, grid_rows, grid_cols)
        
        # Debug: Track which grid cells are being assigned
        grid_assignment_counts[grid_name] = grid_assignment_counts.get(grid_name, 0) + 1
        
        # Convert to 0-based indexing for numpy array
        row_idx = row - 1
        col_idx = col - 1
        
        # Only count visits to open cells
        if maze_map[row_idx][col_idx] == 0:  # Open cell
            visit_counts[row_idx, col_idx] += 1
    
    # DEBUG: Show grid assignment distribution
    print(f"Grid assignment statistics:")
    print(f"  Unique grid cells visited: {len(grid_assignment_counts)}")
    print(f"  Most visited cells: {sorted(grid_assignment_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    # Calculate uncertainty: 1/√N(i) exactly as in notebook
    uncertainty_matrix = np.full((grid_rows, grid_cols), np.nan)
    
    visited_open_cells = 0
    unvisited_open_cells = 0
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:  # Open cell
                count = visit_counts[row, col]
                if count > 0:
                    uncertainty_matrix[row, col] = 1.0 / np.sqrt(count)
                    visited_open_cells += 1
                else:
                    uncertainty_matrix[row, col] = np.inf  # Unvisited open cells
                    unvisited_open_cells += 1
            # Wall cells remain NaN
    
    print(f"✓ Ground truth calculated from {len(positions)} positions using EXACT notebook method")
    print(f"  Visit counts range: {np.min(visit_counts[visit_counts > 0])}-{np.max(visit_counts)}")
    print(f"  Visited open cells: {visited_open_cells}")
    print(f"  Unvisited open cells: {unvisited_open_cells}")
    print(f"  Uncertainty range: {np.nanmin(uncertainty_matrix[np.isfinite(uncertainty_matrix)]):.6f}-{np.nanmax(uncertainty_matrix[np.isfinite(uncertainty_matrix)]):.6f}")
    
    return uncertainty_matrix

def evaluate_uncertainty_method(method, maze_map=None, grid_rows=9, grid_cols=12, device='cpu'):
    """
    Evaluate uncertainty method on grid centers using EXACT notebook coordinate mapping.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    uncertainty_matrix = np.full((grid_rows, grid_cols), np.nan)
    
    # Create coordinates for all open grid centers using EXACT notebook method
    coordinates = []
    valid_indices = []
    
    # Environment boundaries (same as notebook)
    left_edge = -6.0
    top_edge = 4.5
    cell_width = 12.0 / grid_cols    # 1.0m
    cell_height = 9.0 / grid_rows    # 1.0m
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:  # Open cell
                # Calculate grid center coordinates using EXACT notebook formula
                # Convert 0-based matrix indices to 1-based grid coordinates first
                grid_row = row + 1  # Convert to 1-based
                grid_col = col + 1  # Convert to 1-based
                
                center_x = left_edge + (grid_col - 1 + 0.5) * cell_width
                center_y = top_edge - (grid_row - 1 + 0.5) * cell_height
                
                coordinates.append([center_x, center_y])
                valid_indices.append((row, col))
    
    if len(coordinates) > 0:
        print(f"✓ Evaluating {len(coordinates)} open grid centers using exact notebook coordinates")
        # Convert to tensor and get uncertainties for all valid cells at once
        coords_tensor = torch.tensor(coordinates, dtype=torch.float32).to(device)
        
        # Use get_uncertainty method (matching our method implementations)
        uncertainties = method.get_uncertainty(coords_tensor)
        
        # Handle both single values and arrays
        if hasattr(uncertainties, '__len__'):
            uncertainty_values = uncertainties
        else:
            uncertainty_values = [uncertainties] * len(coordinates)
        
        # Fill uncertainty matrix
        for i, (row, col) in enumerate(valid_indices):
            uncertainty_matrix[row, col] = uncertainty_values[i]
    
    return uncertainty_matrix

def normalize_uncertainty_matrix(uncertainty_matrix, maze_map=None):
    """
    CRITICAL: Normalize uncertainty matrix by dividing by maximum value among open cells.
    This implements the normalization step we discussed before L2 computation.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Extract values from open cells only
    open_values = []
    for row in range(uncertainty_matrix.shape[0]):
        for col in range(uncertainty_matrix.shape[1]):
            if maze_map[row][col] == 0:  # Open cell
                val = uncertainty_matrix[row, col]
                if np.isfinite(val):
                    open_values.append(val)
    
    if len(open_values) == 0:
        return uncertainty_matrix
    
    max_val = np.max(open_values)
    if max_val == 0:
        return uncertainty_matrix
    
    # Create normalized matrix
    normalized = uncertainty_matrix.copy()
    for row in range(uncertainty_matrix.shape[0]):
        for col in range(uncertainty_matrix.shape[1]):
            if maze_map[row][col] == 0 and np.isfinite(uncertainty_matrix[row, col]):
                normalized[row, col] = uncertainty_matrix[row, col] / max_val
    
    return normalized

def compute_l2_distance(gt_matrix, pred_matrix, maze_map=None):
    """Compute L2 distance excluding walls"""
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Extract only open cell values
    open_mask = np.array([[maze_map[i][j] == 0 for j in range(12)] for i in range(9)])
    
    gt_open = gt_matrix[open_mask]
    pred_open = pred_matrix[open_mask]
    
    # Only use finite values
    finite_mask = np.isfinite(gt_open) & np.isfinite(pred_open)
    gt_clean = gt_open[finite_mask]
    pred_clean = pred_open[finite_mask]
    
    if len(gt_clean) == 0:
        return np.nan
    
    # Normalize each by its own max
    gt_norm = gt_clean / np.max(gt_clean) if np.max(gt_clean) > 0 else gt_clean
    pred_norm = pred_clean / np.max(pred_clean) if np.max(pred_clean) > 0 else pred_clean
    
    # Compute L2 distance
    return np.linalg.norm(gt_norm - pred_norm)

    # # unnormalized version to test Gaussian noise
    # return np.linalg.norm(gt_clean - pred_clean)

def compute_correlation(gt_matrix, pred_matrix, maze_map=None):
    """
    Compute Pearson correlation between uncertainty matrices (on open cells).
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Extract vectors for open cells only (no normalization needed for correlation)
    gt_vector = []
    pred_vector = []
    
    for row in range(len(maze_map)):
        for col in range(len(maze_map[0])):
            if maze_map[row][col] == 0:  # Open cell
                gt_val = gt_matrix[row, col]
                pred_val = pred_matrix[row, col]
                
                if np.isfinite(gt_val) and np.isfinite(pred_val):
                    gt_vector.append(gt_val)
                    pred_vector.append(pred_val)
    
    if len(gt_vector) < 2:
        return np.nan
    
    gt_vector = np.array(gt_vector)
    pred_vector = np.array(pred_vector)
    
    # Compute correlation
    correlation = np.corrcoef(gt_vector, pred_vector)[0, 1]
    
    return correlation if not np.isnan(correlation) else 0.0

def save_heatmap_to_wandb(uncertainty_matrix, title, wandb_switch=True, maze_map=None):
    """Create heatmap with WHITE walls and proper handling of infinite values"""
    if maze_map is None:
        maze_map = get_maze_map()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create uncertainty matrix excluding walls
    uncertainty_open_only = uncertainty_matrix.copy()
    
    # Set wall cells to NaN (they'll appear white)
    for row in range(len(maze_map)):
        for col in range(len(maze_map[0])):
            if maze_map[row][col] == 1:  # Wall
                uncertainty_open_only[row, col] = np.nan
    
    # CRITICAL FIX: Handle infinite values for visualization
    # Replace inf with a large finite value for proper colormap display
    finite_mask = np.isfinite(uncertainty_open_only)
    if np.any(finite_mask):
        max_finite = np.nanmax(uncertainty_open_only[finite_mask])
        # Set infinite values to be 2x the maximum finite value
        inf_replacement = max_finite * 2 if max_finite > 0 else 1.0
        uncertainty_open_only[np.isinf(uncertainty_open_only)] = inf_replacement
        
        print(f"Debug: Replaced {np.sum(np.isinf(uncertainty_matrix))} infinite values with {inf_replacement:.3f}")
    
    # Create heatmap - now inf values will show as highest color
    im = ax.imshow(uncertainty_open_only, cmap='viridis', aspect='equal', 
                   origin='upper', interpolation='nearest')
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Column (1-12)', fontsize=12)
    ax.set_ylabel('Row (1-9)', fontsize=12)
    
    # Set proper ticks (1-based indexing to match notebook)
    ax.set_xticks(range(12))
    ax.set_yticks(range(9))
    ax.set_xticklabels(range(1, 13))
    ax.set_yticklabels(range(1, 10))
    
    # Add grid
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add colorbar with better labeling
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Uncertainty', rotation=270, labelpad=20, fontsize=12)
    
    # Add text showing what highest values represent
    if np.any(np.isinf(uncertainty_matrix)):
        ax.text(0.02, 0.02, 'Brightest = Unvisited cells (∞)', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Log to WandB
    if wandb_switch:
        wandb.log({f"heatmap_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}": wandb.Image(fig)})

    plt.close(fig)
    return fig