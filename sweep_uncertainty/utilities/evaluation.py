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
    Calculate ground truth uncertainty using 1/√N(i) where N(i) is visit count.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Initialize visit count matrix
    visit_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    
    # Count visits to each grid cell
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        if isinstance(observations, dict) and 'achieved_goal' in observations:
            positions = observations['achieved_goal']
        else:
            positions = observations
            
        for pos in positions:
            # Convert observation to grid indices (0-based)
            x, y = pos[:2]
            grid_x = int(np.clip(x * grid_cols, 0, grid_cols - 1))
            grid_y = int(np.clip(y * grid_rows, 0, grid_rows - 1))
            
            # Only count visits to open cells (maze_map[i][j] == 0 means open)
            if maze_map[grid_y][grid_x] == 0:  # Open cell
                visit_counts[grid_y, grid_x] += 1
    
    # Calculate uncertainty: 1/√N(i)
    uncertainty_matrix = np.full((grid_rows, grid_cols), np.nan)
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:  # Open cell
                count = visit_counts[row, col]
                if count > 0:
                    uncertainty_matrix[row, col] = 1.0 / np.sqrt(count)
                else:
                    uncertainty_matrix[row, col] = np.inf  # Unvisited open cells
            # Wall cells remain NaN
    
    return uncertainty_matrix

def calculate_ground_truth_from_positions(positions, maze_map=None, grid_rows=9, grid_cols=12):
    """
    Calculate ground truth uncertainty using 1/√N(i) where N(i) is visit count.
    This version works directly with position arrays instead of full dataset.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Initialize visit count matrix
    visit_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    
    # Count visits to each grid cell from the position array
    for pos in positions:
        # Convert position to grid indices (0-based)
        x, y = pos[:2]
        grid_x = int(np.clip(x * grid_cols, 0, grid_cols - 1))
        grid_y = int(np.clip(y * grid_rows, 0, grid_rows - 1))
        
        # Only count visits to open cells (maze_map[i][j] == 0 means open)
        if maze_map[grid_y][grid_x] == 0:  # Open cell
            visit_counts[grid_y, grid_x] += 1
    
    # Calculate uncertainty: 1/√N(i)
    uncertainty_matrix = np.full((grid_rows, grid_cols), np.nan)
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:  # Open cell
                count = visit_counts[row, col]
                if count > 0:
                    uncertainty_matrix[row, col] = 1.0 / np.sqrt(count)
                else:
                    uncertainty_matrix[row, col] = np.inf  # Unvisited open cells
            # Wall cells remain NaN
    
    print(f"✓ Ground truth calculated from {len(positions)} positions")
    print(f"  Visit counts range: {np.min(visit_counts[visit_counts > 0])}-{np.max(visit_counts)}")
    print(f"  Uncertainty range: {np.nanmin(uncertainty_matrix[np.isfinite(uncertainty_matrix)]):.6f}-{np.nanmax(uncertainty_matrix[np.isfinite(uncertainty_matrix)]):.6f}")
    
    return uncertainty_matrix

def evaluate_uncertainty_method(method, maze_map=None, grid_rows=9, grid_cols=12, device='cpu'):
    """
    Evaluate uncertainty method by computing uncertainty for ALL grid cell centers at once.
    This matches the notebook's batch processing approach.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    uncertainty_matrix = np.full((grid_rows, grid_cols), np.nan)
    
    # Create coordinates for all open grid centers
    coordinates = []
    valid_indices = []
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:  # Open cell
                # Grid center coordinates (normalized to [0,1])
                center_x = (col + 0.5) / grid_cols
                center_y = (row + 0.5) / grid_rows
                coordinates.append([center_x, center_y])
                valid_indices.append((row, col))
    
    if len(coordinates) > 0:
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
    """Create heatmap with WHITE walls and proper exclusion"""
    if maze_map is None:
        maze_map = get_maze_map()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create uncertainty matrix excluding walls
    uncertainty_open_only = uncertainty_matrix.copy()
    
    # Set wall cells to NaN (they'll appear white and be excluded)
    for row in range(len(maze_map)):
        for col in range(len(maze_map[0])):
            if maze_map[row][col] == 1:  # Wall
                uncertainty_open_only[row, col] = np.nan
    
    # Create heatmap - NaN values will be white automatically
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
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Uncertainty', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # Log to WandB
    if wandb_switch:
        wandb.log({f"heatmap_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}": wandb.Image(fig)})

    plt.close(fig)
    return fig