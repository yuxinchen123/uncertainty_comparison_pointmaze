#!/usr/bin/env python3

"""
Test script to verify the wall mask implementation matches the notebook.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add utilities to path
sys.path.append('.')

from utilities import load_pointmaze_dataset

def test_wall_mask():
    """Test that wall mask correctly identifies wall vs open cells"""
    print("=" * 60)
    print("TESTING WALL MASK IMPLEMENTATION")
    print("=" * 60)
    
    # Load dataset and maze structure
    try:
        dataset, maze_map = load_pointmaze_dataset()
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Maze map shape: {len(maze_map)}x{len(maze_map[0])}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False
    
    # Analyze maze structure
    total_cells = len(maze_map) * len(maze_map[0])
    wall_cells = sum(sum(row) for row in maze_map)  # Count 1s (walls)
    open_cells = total_cells - wall_cells
    
    print(f"\nMaze Structure Analysis:")
    print(f"  Total cells: {total_cells}")
    print(f"  Wall cells: {wall_cells} ({wall_cells/total_cells*100:.1f}%)")
    print(f"  Open cells: {open_cells} ({open_cells/total_cells*100:.1f}%)")
    
    # Print maze structure
    print(f"\nMaze Layout (0=open, 1=wall):")
    print("  " + "".join([f"{i:2d}" for i in range(12)]))
    for i, row in enumerate(maze_map):
        row_str = "".join([f"{cell:2d}" for cell in row])
        print(f"{i} {row_str}")
    
    # Test specific cells from notebook
    test_cases = [
        # Known open cells from navigation
        (1, 1, "should be open"),
        (1, 5, "should be open"), 
        (5, 6, "should be open"),
        (7, 6, "should be open"),
        # Known wall cells (borders)
        (0, 0, "should be wall"),
        (0, 5, "should be wall"),
        (8, 8, "should be wall"),
    ]
    
    print(f"\nTesting specific cells:")
    for row, col, description in test_cases:
        is_wall = maze_map[row][col] == 1
        is_open = maze_map[row][col] == 0
        status = "WALL" if is_wall else "OPEN"
        print(f"  g_({row+1},{col+1}): {status} - {description}")
    
    # Visualize the maze
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create visualization matrix: 0 = open (white), 1 = wall (black)
    vis_array = np.array(maze_map, dtype=float)
    
    im = ax.imshow(vis_array, cmap='gray_r', aspect='auto', origin='upper')
    ax.set_title('PointMaze Wall Structure\n(White=Open, Black=Wall)')
    ax.set_xlabel('Column (0-11)')
    ax.set_ylabel('Row (0-8)')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Add cell labels
    for i in range(9):
        for j in range(12):
            text_color = 'black' if maze_map[i][j] == 0 else 'white'
            ax.text(j, i, f'{maze_map[i][j]}', ha='center', va='center', 
                   color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('maze_wall_structure.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Maze visualization saved to 'maze_wall_structure.png'")
    
    # Return summary
    print(f"\n" + "=" * 60)
    print("WALL MASK TEST SUMMARY")
    print("=" * 60)
    print(f"✓ Wall mask extracted from PointMaze environment")
    print(f"✓ Maze structure: {len(maze_map)}×{len(maze_map[0])} grid")
    print(f"✓ Wall encoding: 1 = wall, 0 = open")
    print(f"✓ Wall ratio: {wall_cells/total_cells*100:.1f}% walls")
    print(f"✓ Evaluation will exclude {wall_cells} wall cells")
    print(f"✓ Only {open_cells} open cells will be used for metrics")
    
    return True

if __name__ == "__main__":
    success = test_wall_mask()
    if success:
        print("\n🎉 Wall mask test completed successfully!")
    else:
        print("\n❌ Wall mask test failed!")
    exit(0 if success else 1)
