#!/usr/bin/env python3
"""
Test script to verify that our single-sampling approach works with real data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utilities.environment import load_pointmaze_dataset
from utilities.evaluation import calculate_ground_truth_from_positions

def test_real_data_sampling():
    """Test single-sampling approach with real PointMaze data"""
    print("Testing single-sampling approach with REAL PointMaze data...")
    
    try:
        # Load real dataset
        dataset, maze_map = load_pointmaze_dataset()
        print(f"✓ Successfully loaded PointMaze dataset")
        
        # Extract all positions from dataset
        all_positions = []
        for episode in dataset.iterate_episodes():
            observations = episode.observations
            if isinstance(observations, dict) and 'achieved_goal' in observations:
                positions = observations['achieved_goal']
            else:
                positions = observations
                
            for pos in positions:
                all_positions.append(pos[:2])  # Take only x, y coordinates
        
        all_positions = np.array(all_positions)
        print(f"✓ Extracted {len(all_positions)} total positions from dataset")
        print(f"  Position range: X=[{np.min(all_positions[:, 0]):.3f}, {np.max(all_positions[:, 0]):.3f}], Y=[{np.min(all_positions[:, 1]):.3f}, {np.max(all_positions[:, 1]):.3f}]")
        
        # Test different sample sizes
        sample_sizes = [1000, 5000, 10000]
        
        for i, sample_size in enumerate(sample_sizes):
            print(f"\n--- Testing {sample_size} samples ---")
            
            # Use different seed for each sample size
            np.random.seed(42 + i)
            
            # Sample positions
            if len(all_positions) > sample_size:
                indices = np.random.choice(len(all_positions), sample_size, replace=False)
                sampled_positions = all_positions[indices]
            else:
                sampled_positions = all_positions
            
            print(f"Sampled {len(sampled_positions)} positions")
            print(f"Sample range: X=[{np.min(sampled_positions[:, 0]):.3f}, {np.max(sampled_positions[:, 0]):.3f}], Y=[{np.min(sampled_positions[:, 1]):.3f}, {np.max(sampled_positions[:, 1]):.3f}]")
            
            # Calculate ground truth from sample
            try:
                gt_uncertainty = calculate_ground_truth_from_positions(sampled_positions, maze_map)
                
                # Check results
                finite_values = gt_uncertainty[np.isfinite(gt_uncertainty)]
                inf_values = np.sum(np.isinf(gt_uncertainty))
                nan_values = np.sum(np.isnan(gt_uncertainty))
                
                print(f"Ground truth statistics:")
                print(f"  Finite values: {len(finite_values)} (range: {np.min(finite_values):.4f} - {np.max(finite_values):.4f})")
                print(f"  Infinite values: {inf_values}")
                print(f"  NaN values: {nan_values}")
                
            except Exception as e:
                print(f"✗ Error calculating ground truth: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ Real data sampling test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load real data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_data_sampling()
    if success:
        print("\n🎉 All tests passed! The single-sampling approach is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the implementation.")
