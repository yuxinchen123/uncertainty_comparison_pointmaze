#!/usr/bin/env python3

"""
Quick test script to verify the uncertainty comparison setup works correctly.
Tests each method with minimal parameters.
"""

import sys
import os

# Add utilities to path
sys.path.append('.')

from utilities import (
    load_pointmaze_dataset, extract_positions_from_dataset,
    RNDMethod, RNDLinearMethod, EllipticalBonusMethod,
    calculate_ground_truth, evaluate_uncertainty_method,
    normalize_uncertainty_matrix, compute_l2_distance
)

import torch
import numpy as np

def test_setup():
    """Test basic setup and data loading"""
    print("=" * 60)
    print("TESTING SETUP")
    print("=" * 60)
    
    # Test device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")
    
    # Test dataset loading
    try:
        dataset, maze_map = load_pointmaze_dataset()
        print(f"✓ Dataset loaded: {dataset.total_episodes} episodes")
        print(f"✓ Maze map shape: {len(maze_map)}x{len(maze_map[0])}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    # Test position extraction
    try:
        positions = extract_positions_from_dataset(dataset, num_samples=1000)
        print(f"✓ Extracted {len(positions)} positions")
    except Exception as e:
        print(f"✗ Position extraction failed: {e}")
        return False
    
    return True, dataset, maze_map, positions, device

def test_ground_truth(dataset, maze_map):
    """Test ground truth calculation"""
    print("\n" + "=" * 60)
    print("TESTING GROUND TRUTH CALCULATION")
    print("=" * 60)
    
    try:
        gt_uncertainty = calculate_ground_truth(dataset, maze_map)
        gt_normalized = normalize_uncertainty_matrix(gt_uncertainty)
        
        # Count open cells with finite values
        finite_count = np.sum(np.isfinite(gt_normalized))
        print(f"✓ Ground truth computed: {finite_count} open cells")
        print(f"✓ Max uncertainty: {np.nanmax(gt_uncertainty):.6f}")
        
        return True, gt_uncertainty, gt_normalized
    except Exception as e:
        print(f"✗ Ground truth calculation failed: {e}")
        return False, None, None

def test_rnd_method(positions, gt_normalized, maze_map, device):
    """Test RND method"""
    print("\n" + "=" * 60)
    print("TESTING RND METHOD")
    print("=" * 60)
    
    try:
        # Initialize RND with small architecture for testing
        rnd = RNDMethod(hidden_dims=[64, 64], output_dim=32, device=device)
        print("✓ RND initialized")
        
        # Train briefly
        losses = rnd.train_on_positions(positions[:500], num_epochs=3)
        print(f"✓ RND trained: final loss = {losses[-1]:.6f}")
        
        # Evaluate
        pred_uncertainty = evaluate_uncertainty_method(rnd, maze_map, device=device)
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics
        l2_dist = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        print(f"✓ RND evaluated: L2 distance = {l2_dist:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ RND method failed: {e}")
        return False

def test_rnd_linear_method(positions, gt_normalized, maze_map, device):
    """Test RND-Linear method"""
    print("\n" + "=" * 60)
    print("TESTING RND-LINEAR METHOD")
    print("=" * 60)
    
    try:
        # Create φ(s) weights
        torch.manual_seed(42)
        temp_layer = torch.nn.Linear(2, 32)
        phi_weights = {
            'weight': temp_layer.weight.clone(),
            'bias': temp_layer.bias.clone()
        }
        
        # Initialize RND-Linear
        rnd_linear = RNDLinearMethod(feature_dim=32, device=device, phi_weights=phi_weights)
        print("✓ RND-Linear initialized")
        
        # Train with full dataset (as per new protocol)
        losses = rnd_linear.train_on_positions(positions[:500], num_epochs=3, subset_ratio=1.0)
        print(f"✓ RND-Linear trained: final loss = {losses[-1]:.6f}")
        
        # Evaluate
        pred_uncertainty = evaluate_uncertainty_method(rnd_linear, maze_map, device=device)
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics
        l2_dist = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        print(f"✓ RND-Linear evaluated: L2 distance = {l2_dist:.6f}")
        
        return True, phi_weights
    except Exception as e:
        print(f"✗ RND-Linear method failed: {e}")
        return False, None

def test_elliptical_method(positions, gt_normalized, maze_map, device, phi_weights):
    """Test Elliptical Bonus method"""
    print("\n" + "=" * 60)
    print("TESTING ELLIPTICAL BONUS METHOD")
    print("=" * 60)
    
    try:
        # Initialize Elliptical with shared φ(s) weights
        elliptical = EllipticalBonusMethod(feature_dim=32, device=device, phi_weights=phi_weights)
        print("✓ Elliptical initialized")
        
        # Update covariance
        elliptical.update_covariance_from_positions(positions[:500])
        print("✓ Elliptical covariance updated")
        
        # Evaluate
        pred_uncertainty = evaluate_uncertainty_method(elliptical, maze_map, device=device)
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics
        l2_dist = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        print(f"✓ Elliptical evaluated: L2 distance = {l2_dist:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Elliptical method failed: {e}")
        return False

def main():
    """Run all tests"""
    print("UNCERTAINTY COMPARISON SYSTEM TEST")
    print("=" * 60)
    
    # Test setup
    result = test_setup()
    if not result[0]:
        print("\n❌ SETUP FAILED - Cannot continue")
        return False
    
    _, dataset, maze_map, positions, device = result
    
    # Test ground truth
    result = test_ground_truth(dataset, maze_map)
    if not result[0]:
        print("\n❌ GROUND TRUTH FAILED")
        return False
    
    _, gt_uncertainty, gt_normalized = result
    
    # Test each method
    success_count = 0
    
    # Test RND
    if test_rnd_method(positions, gt_normalized, maze_map, device):
        success_count += 1
        print("✅ RND method passed")
    else:
        print("❌ RND method failed")
    
    # Test RND-Linear
    result = test_rnd_linear_method(positions, gt_normalized, maze_map, device)
    if result[0]:
        success_count += 1
        print("✅ RND-Linear method passed")
        phi_weights = result[1]
    else:
        print("❌ RND-Linear method failed")
        phi_weights = None
    
    # Test Elliptical (only if RND-Linear provided phi_weights)
    if phi_weights is not None:
        if test_elliptical_method(positions, gt_normalized, maze_map, device, phi_weights):
            success_count += 1
            print("✅ Elliptical method passed")
        else:
            print("❌ Elliptical method failed")
    else:
        print("⚠️  Elliptical method skipped (no phi_weights)")
    
    # Final report
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {success_count}/3 methods")
    
    if success_count == 3:
        print("🎉 ALL TESTS PASSED - System ready for sweep!")
        return True
    else:
        print("⚠️  Some tests failed - Check implementation")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
