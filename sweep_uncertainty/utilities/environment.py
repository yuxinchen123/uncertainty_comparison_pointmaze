import numpy as np
import torch
from pathlib import Path

def get_maze_map():
    """Return the actual PointMaze wall structure"""
    try:
        import gymnasium as gym
        import gymnasium_robotics
        
        # Register and create environment to get maze structure
        gym.register_envs(gymnasium_robotics)
        env = gym.make("PointMaze_Large-v3")
        
        # Extract maze map from environment
        unwrapped_env = env.unwrapped
        maze = unwrapped_env.maze
        maze_map = maze.maze_map
        
        env.close()
        return maze_map
        
    except Exception as e:
        print(f"Warning: Could not load maze from environment: {e}")
        # Fallback to hardcoded maze map
        maze_map = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ])
        return maze_map

def load_pointmaze_dataset():
    """Load PointMaze dataset using Minari (same as notebook)"""
    
    try:
        print("Loading PointMaze dataset using Minari...")
        import minari
        
        # Load the same dataset as in your notebook
        dataset = minari.load_dataset('D4RL/pointmaze/large-v2', download=True)
        maze_map = get_maze_map()
        
        print(f"✓ Successfully loaded Minari dataset")
        print(f"  Dataset type: {type(dataset)}")
        print(f"  Total episodes: {dataset.total_episodes}")
        print(f"  Total steps: {dataset.total_steps}")
        
        return dataset, maze_map
        
    except ImportError as e:
        print(f"❌ Minari not available: {e}")
        print("Please install minari: pip install minari")
        return generate_synthetic_dataset(), get_maze_map()
        
    except Exception as e:
        print(f"❌ Failed to load Minari dataset: {e}")
        print("Falling back to synthetic data...")
        return generate_synthetic_dataset(), get_maze_map()

def generate_synthetic_dataset(n_samples=100000):
    """Generate synthetic PointMaze-like data for testing"""
    print("Generating synthetic PointMaze data...")
    maze_map = get_maze_map()
    
    # Create realistic PointMaze coordinate ranges
    # Based on your notebook: X[-6.0, 6.0], Y[-4.5, 4.5]
    navigable_coords = []
    
    for i in range(9):
        for j in range(12):
            if maze_map[i, j] == 0:  # Open cell
                # Convert grid to actual coordinates
                x = -6.0 + (j + 0.5) * 1.0  # Each cell is 1m wide
                y = 4.5 - (i + 0.5) * 1.0   # Each cell is 1m tall
                navigable_coords.append([x, y])
    
    navigable_coords = np.array(navigable_coords)
    
    # Generate samples with some concentration in certain areas
    samples = []
    episode_data = []
    
    # Create multiple episodes
    n_episodes = 100
    for episode_idx in range(n_episodes):
        episode_length = np.random.randint(800, 1200)  # Variable episode lengths
        episode_positions = []
        
        for step in range(episode_length):
            # Pick random navigable cell with some bias toward certain areas
            if np.random.random() < 0.3:  # 30% chance for concentrated areas
                # Bias toward central areas (like your heatmaps show)
                central_cells = navigable_coords[
                    (np.abs(navigable_coords[:, 0]) < 2.0) & 
                    (np.abs(navigable_coords[:, 1]) < 2.0)
                ]
                if len(central_cells) > 0:
                    idx = np.random.randint(len(central_cells))
                    base_coord = central_cells[idx]
                else:
                    idx = np.random.randint(len(navigable_coords))
                    base_coord = navigable_coords[idx]
            else:
                idx = np.random.randint(len(navigable_coords))
                base_coord = navigable_coords[idx]
            
            # Add small noise within cell
            noise = np.random.normal(0, 0.1, 2)
            coord = base_coord + noise
            
            # Clip to maze bounds
            coord[0] = np.clip(coord[0], -6.0, 6.0)
            coord[1] = np.clip(coord[1], -4.5, 4.5)
            
            episode_positions.append(coord)
        
        episode_data.append(np.array(episode_positions))
    
    print(f"Generated {n_episodes} episodes with {sum(len(ep) for ep in episode_data)} total steps")
    
    # Create dataset object compatible with your code
    class SyntheticDataset:
        def __init__(self, episodes):
            self.episodes = episodes
            self.total_episodes = len(episodes)
            self.total_steps = sum(len(ep) for ep in episodes)
        
        def iterate_episodes(self):
            for episode_positions in self.episodes:
                class Episode:
                    def __init__(self, positions):
                        # Create observations dict like real dataset
                        self.observations = {
                            'achieved_goal': positions,
                            'desired_goal': positions,  # Dummy
                            'observation': positions    # Dummy
                        }
                
                yield Episode(episode_positions)
    
    return SyntheticDataset(episode_data)

def extract_positions_from_dataset(dataset, num_samples=None):
    """Extract position coordinates from dataset (compatible with Minari)"""
    all_positions = []
    
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        
        # Handle both Minari format and synthetic format
        if isinstance(observations, dict) and 'achieved_goal' in observations:
            positions = observations['achieved_goal']
        else:
            positions = observations
        
        # Ensure numpy array
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        # Ensure we have at least 2D coordinates
        if len(positions.shape) == 1:
            positions = positions.reshape(1, -1)
        
        if positions.shape[1] < 2:
            raise ValueError(f"Expected at least 2D coordinates, got {positions.shape[1]}D")
        
        all_positions.append(positions[:, :2])  # Take only x,y coordinates
    
    # Concatenate all positions
    all_positions = np.concatenate(all_positions, axis=0)
    
    # Sample the requested number (if specified)
    if num_samples is not None and len(all_positions) > num_samples:
        indices = np.random.choice(len(all_positions), num_samples, replace=False)
        all_positions = all_positions[indices]
    
    return all_positions

def observation_to_grid(observation, grid_rows=9, grid_cols=12):
    """Convert observation coordinates to grid indices (using notebook's coordinate system)"""
    
    # Extract coordinates
    if isinstance(observation, dict) and 'achieved_goal' in observation:
        x, y = observation['achieved_goal'][0], observation['achieved_goal'][1]
    elif isinstance(observation, (list, tuple, np.ndarray)) and len(observation) >= 2:
        x, y = observation[0], observation[1]
    else:
        raise ValueError(f"Invalid observation format: {type(observation)}")
    
    # Use coordinate system from your notebook
    # Total environment: X[-6.0, 6.0], Y[-4.5, 4.5]
    total_width = 12.0   # 6.0 - (-6.0)
    total_height = 9.0   # 4.5 - (-4.5)
    left_edge = -6.0
    top_edge = 4.5
    
    cell_width = total_width / grid_cols    # 1.0m
    cell_height = total_height / grid_rows  # 1.0m
    
    # X direction (columns): Left to right, 1-based indexing
    if x < left_edge:
        col = 1
    elif x >= (left_edge + total_width):
        col = grid_cols
    else:
        col = int((x - left_edge) / cell_width) + 1
        col = max(1, min(col, grid_cols))
    
    # Y direction (rows): Top to bottom, 1-based indexing
    if y > top_edge:
        row = 1
    elif y <= (top_edge - total_height):
        row = grid_rows
    else:
        y_offset_from_top = top_edge - y
        row = int(y_offset_from_top / cell_height) + 1
        row = max(1, min(row, grid_rows))
    
    grid_name = f"g_({row},{col})"
    return grid_name, (row, col)

def grid_to_center_observation(row, col, grid_rows=9, grid_cols=12):
    """Convert grid indices to center coordinates (using notebook's coordinate system)"""
    
    # Use coordinate system from your notebook
    total_width = 12.0
    total_height = 9.0
    left_edge = -6.0
    top_edge = 4.5
    
    cell_width = total_width / grid_cols
    cell_height = total_height / grid_rows
    
    # Calculate center coordinates
    center_x = left_edge + (col - 1 + 0.5) * cell_width
    center_y = top_edge - (row - 1 + 0.5) * cell_height
    
    return center_x, center_y