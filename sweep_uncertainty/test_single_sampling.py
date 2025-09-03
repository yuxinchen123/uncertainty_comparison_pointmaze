#!/usr/bin/env python3
"""
Test script to verify single-sampling approach produces different results for different sample sizes
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities.environment import extract_positions_from_dataset, observation_to_grid_notebook_exact
from utilities.evaluation import calculate_ground_truth_from_positions
from utilities.uncertainty_methods import RNDMethod, RNDLinearMethod, EllipticalBonusMethod
import torch

def test_sampling_consistency():
    """Test that different sample sizes produce different results"""
    
    # Parameters
    env_name = "PointMaze_UMazeDense-v3"
    feature_dim = 64
    device = 'cpu'
    seed = 42
    
    print("Testing single-sampling approach...")
    print(f"Environment: {env_name}")
    print(f"Base seed: {seed}")
    
    # Test different sample sizes
    sample_sizes = [1000, 5000, 10000]
    all_results = {}
    
    for sample_size in sample_sizes:
        print(f"\n--- Testing {sample_size} samples ---")
        
        # Extract positions with deterministic seeding
        np.random.seed(seed + sample_size)  # Different seed for each sample size
        positions = extract_positions_from_dataset(env_name, n_samples=sample_size)
        
        print(f"Position range: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}], "
              f"Y[{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        
        # Calculate ground truth from these exact positions
        grid_x = np.linspace(-6.0, 6.0, 9)
        grid_y = np.linspace(-4.5, 4.5, 12)
        ground_truth = calculate_ground_truth_from_positions(positions, grid_x, grid_y)
        
        print(f"Ground truth: min={ground_truth.min():.6f}, max={ground_truth.max():.6f}, "
              f"mean={ground_truth.mean():.6f}")
        
        # Test RND method
        rnd_method = RNDMethod(feature_dim=feature_dim, device=device)
        losses = rnd_method.train_on_positions(positions, num_epochs=10)
        
        # Get uncertainty on a small test grid
        test_coords = np.array([[-3.0, -2.0], [0.0, 0.0], [3.0, 2.0]])
        uncertainty = rnd_method.get_uncertainty(test_coords)
        
        print(f"RND uncertainty at test points: {uncertainty}")
        
        all_results[sample_size] = {
            'positions': positions,
            'ground_truth': ground_truth,
            'uncertainty': uncertainty,
            'num_positions': len(positions)
        }
    
    # Verify results are different
    print("\n--- Consistency Check ---")
    for i, size1 in enumerate(sample_sizes[:-1]):
        size2 = sample_sizes[i + 1]
        
        gt1 = all_results[size1]['ground_truth']
        gt2 = all_results[size2]['ground_truth']
        
        # Check if ground truths are different
        gt_diff = np.abs(gt1 - gt2).mean()
        print(f"Ground truth difference ({size1} vs {size2}): {gt_diff:.6f}")
        
        # Check if uncertainties are different
        unc1 = all_results[size1]['uncertainty']
        unc2 = all_results[size2]['uncertainty']
        unc_diff = np.abs(unc1 - unc2).mean()
        print(f"Uncertainty difference ({size1} vs {size2}): {unc_diff:.6f}")
        
        if gt_diff < 1e-6:
            print(f"WARNING: Ground truths are nearly identical for {size1} and {size2} samples!")
        else:
            print(f"✓ Ground truths are different for {size1} and {size2} samples")
    
    print("\nTest completed!")
    return all_results

if __name__ == "__main__":
    results = test_sampling_consistency()
