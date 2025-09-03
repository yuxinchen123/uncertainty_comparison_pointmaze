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
    """Extract positions from dataset with optional sampling"""
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
    
    # Handle sampling - only sample if num_samples is specified and less than total
    if num_samples is not None and len(all_positions) > num_samples:
        # Randomly sample the requested number of positions
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

# Add these functions for compatibility
def observation_to_grid(observation, grid_rows=9, grid_cols=12):
    """Convert observation coordinates to grid indices"""
    x, y = observation[:2]
    grid_x = int(np.clip(x * grid_cols, 0, grid_cols - 1))
    grid_y = int(np.clip(y * grid_rows, 0, grid_rows - 1))
    return grid_x, grid_y

def grid_to_center_observation(row, col, grid_rows=9, grid_cols=12):
    """Convert grid indices to center coordinates of the cell"""
    center_x = (col + 0.5) / grid_cols
    center_y = (row + 0.5) / grid_rows
    return center_x, center_y