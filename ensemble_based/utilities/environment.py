import numpy as np
import torch
from pathlib import Path
import minari

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

def load_pointmaze_dataset():
    """Load PointMaze dataset and return (dataset, maze_map) tuple as expected by main script"""
    
    try:
        # Try to load real data first
        dataset = minari.load_dataset('D4RL/pointmaze/large-v2', download=True)
        maze_map = get_maze_map()
        print("✓ Successfully loaded D4RL PointMaze dataset")
        return dataset, maze_map
        
    except Exception as e:
        print(f"Warning: Real data not found ({e}), generating synthetic data")
        return generate_synthetic_dataset(), get_maze_map()

def generate_synthetic_dataset(n_samples=100000):
    """Generate synthetic PointMaze-like data for testing"""
    maze_map = get_maze_map()
    
    # Get all navigable cell centers
    navigable_coords = []
    for i in range(9):
        for j in range(12):
            if maze_map[i, j] == 0:  # Open cell
                x = (j + 0.5) / 12
                y = (i + 0.5) / 9
                navigable_coords.append([x, y])
    
    navigable_coords = np.array(navigable_coords)
    
    # Sample from navigable areas with some noise
    samples = []
    for _ in range(n_samples):
        # Pick random navigable cell
        idx = np.random.randint(len(navigable_coords))
        base_coord = navigable_coords[idx]
        
        # Add small noise within cell
        noise = np.random.normal(0, 0.02, 2)  # Small noise
        coord = base_coord + noise
        
        # Clip to valid range
        coord = np.clip(coord, 0, 1)
        samples.append(coord)
    
    observations = np.array(samples)
    
    # Create simple dataset object
    class SimpleDataset:
        def __init__(self, observations):
            self.observations = observations
        
        def iterate_episodes(self):
            # Yield single episode containing all observations
            class Episode:
                def __init__(self, obs):
                    self.observations = obs
            
            yield Episode(self.observations)
    
    return SimpleDataset(observations)

def extract_positions_from_dataset(dataset, num_samples=None):
    """Extract position coordinates from dataset with proper random sampling"""
    all_positions = []
    
    # Extract all positions from the dataset
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        if isinstance(observations, dict) and 'achieved_goal' in observations:
            positions = observations['achieved_goal']
        else:
            positions = observations
        
        # Add all positions from this episode
        for pos in positions:
            all_positions.append(pos[:2])  # Take only x, y coordinates
    
    all_positions = np.array(all_positions)
    print(f"✓ Extracted {len(all_positions)} total positions from dataset")
    
    # Show position range before sampling
    print(f"  Position range: X=[{np.min(all_positions[:, 0]):.3f}, {np.max(all_positions[:, 0]):.3f}], Y=[{np.min(all_positions[:, 1]):.3f}, {np.max(all_positions[:, 1]):.3f}]")
    
    # Handle sampling - only sample if num_samples is specified and less than total
    if num_samples is not None and len(all_positions) > num_samples:
        # Use random sampling with current random state (controlled externally)
        indices = np.random.choice(len(all_positions), num_samples, replace=False)
        sampled_positions = all_positions[indices]
        
        print(f"✓ Randomly sampled {num_samples} positions from {len(all_positions)} total")
        
        return sampled_positions
    else:
        # Return all positions if num_samples is None or >= total positions
        if num_samples is None:
            print("✓ Using all available positions (no sampling)")
        else:
            print(f"✓ Using all {len(all_positions)} positions (requested {num_samples} >= available)")
        return all_positions

# Add these functions for compatibility with exact notebook mapping
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

def observation_to_grid(observation, grid_rows=9, grid_cols=12):
    """Convert observation coordinates to grid indices using EXACT notebook method"""
    _, (row, col) = observation_to_grid_notebook_exact(observation, grid_rows, grid_cols)
    # Convert to 0-based for backward compatibility
    return col-1, row-1

def grid_to_center_observation(row, col, grid_rows=9, grid_cols=12):
    """Convert grid indices to center coordinates using EXACT notebook method"""
    # Environment boundaries (same as notebook)
    left_edge = -6.0
    top_edge = 4.5
    cell_width = 12.0 / grid_cols    # 1.0m
    cell_height = 9.0 / grid_rows    # 1.0m
    
    # Convert 0-based indices to 1-based grid coordinates
    grid_row = row + 1
    grid_col = col + 1
    
    # Calculate center coordinates using exact notebook formula
    center_x = left_edge + (grid_col - 1 + 0.5) * cell_width
    center_y = top_edge - (grid_row - 1 + 0.5) * cell_height
    
    return center_x, center_y