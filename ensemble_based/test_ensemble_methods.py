#!/usr/bin/env python3
"""
Quick test script for ensemble methods.
Tests all three ensemble methods with minimal parameters.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "utilities"))

from ensemble_uncertainty_methods import (
    EnsembleRNDMethod, 
    EnsembleRNDLinearSGDMethod, 
    EnsembleRNDLinearLSMethod
)
from evaluation import get_phi_weights

def test_ensemble_rnd():
    """Test Ensemble RND method"""
    print("Testing Ensemble RND...")
    
    # Create method
    method = EnsembleRNDMethod(
        hidden_dims=[64, 64],
        output_dim=64,
        K=3,  # Small K for testing
        device='cpu'
    )
    
    # Generate dummy data
    positions = np.random.randn(100, 2) * 2  # 100 random positions
    
    # Train
    losses = method.train_on_positions(positions, num_epochs=5)
    print(f"  Training completed. Loss history lengths: {[len(l) for l in losses]}")
    
    # Test uncertainty calculation
    test_coords = np.random.randn(10, 2) * 2
    uncertainty = method.get_uncertainty(test_coords)
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    
    return True

def test_ensemble_rnd_linear_sgd():
    """Test Ensemble RND-Linear (SGD) method"""
    print("Testing Ensemble RND-Linear (SGD)...")
    
    # Get shared φ(s) weights
    phi_weights = get_phi_weights(64, 42)
    
    # Create method
    method = EnsembleRNDLinearSGDMethod(
        feature_dim=64,
        K=3,  # Small K for testing
        device='cpu',
        phi_weights=phi_weights
    )
    
    # Generate dummy data
    positions = np.random.randn(100, 2) * 2  # 100 random positions
    
    # Train
    losses = method.train_on_positions(positions, num_epochs=5)
    print(f"  Training completed. Loss history lengths: {[len(l) for l in losses]}")
    
    # Test uncertainty calculation
    test_coords = np.random.randn(10, 2) * 2
    uncertainty = method.get_uncertainty(test_coords)
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    
    return True

def test_ensemble_rnd_linear_ls():
    """Test Ensemble RND-Linear (LS) method"""
    print("Testing Ensemble RND-Linear (LS)...")
    
    # Get shared φ(s) weights
    phi_weights = get_phi_weights(64, 42)
    
    # Create method
    method = EnsembleRNDLinearLSMethod(
        feature_dim=64,
        K=3,  # Small K for testing
        device='cpu',
        phi_weights=phi_weights,
        regularization=1e-6
    )
    
    # Generate dummy data
    positions = np.random.randn(100, 2) * 2  # 100 random positions
    
    # Train
    losses = method.train_on_positions(positions, num_epochs=5)
    print(f"  Training completed. Loss history lengths: {[len(l) for l in losses]}")
    
    # Test uncertainty calculation
    test_coords = np.random.randn(10, 2) * 2
    uncertainty = method.get_uncertainty(test_coords)
    print(f"  Uncertainty shape: {uncertainty.shape}")
    print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    
    return True

def test_bootstrap_sampling():
    """Test that bootstrap sampling works correctly"""
    print("Testing bootstrap sampling...")
    
    # Create method
    method = EnsembleRNDMethod(
        hidden_dims=[32],
        output_dim=32,
        K=3,
        device='cpu'
    )
    
    # Generate test data
    positions = np.random.randn(50, 2)
    
    # Test bootstrap sampling
    sample1 = method._bootstrap_sample(positions, sample_size=30)
    sample2 = method._bootstrap_sample(positions, sample_size=30)
    
    print(f"  Original data shape: {positions.shape}")
    print(f"  Bootstrap sample 1 shape: {sample1.shape}")
    print(f"  Bootstrap sample 2 shape: {sample2.shape}")
    print(f"  Samples are different: {not np.array_equal(sample1, sample2)}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Ensemble-Based RND Methods")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Test bootstrap sampling
        test_bootstrap_sampling()
        print("✓ Bootstrap sampling test passed")
        
        # Test Ensemble RND
        test_ensemble_rnd()
        print("✓ Ensemble RND test passed")
        
        # Test Ensemble RND-Linear (SGD)
        test_ensemble_rnd_linear_sgd()
        print("✓ Ensemble RND-Linear (SGD) test passed")
        
        # Test Ensemble RND-Linear (LS)
        test_ensemble_rnd_linear_ls()
        print("✓ Ensemble RND-Linear (LS) test passed")
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




