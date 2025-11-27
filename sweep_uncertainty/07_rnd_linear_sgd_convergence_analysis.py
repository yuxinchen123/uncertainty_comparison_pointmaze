#!/usr/bin/env python3

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset
from utilities.uncertainty_methods import RNDLinearSGDMethod

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_phi_weights_deterministic(feature_dim, seed):
    """Create deterministic φ(s) weights for sharing between methods"""
    # Set temporary seed for weight generation
    torch.manual_seed(seed)
    
    # Create temporary linear layer to generate weights
    temp_layer = torch.nn.Linear(2, feature_dim)
    phi_weights = temp_layer.state_dict()  # This returns proper state dict
    
    return phi_weights

def main():
    parser = argparse.ArgumentParser(description='RND Linear SGD Convergence Analysis on PointMaze')
    
    # Architecture parameters
    parser.add_argument('--phi_dim', type=int, default=64,
                       help='Feature dimension for RND Linear (φ(s) dimension)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=500,
                       help='Number of training epochs (default: 500 to see full convergence)')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to extract from dataset')
    parser.add_argument('--run_index', type=int, default=0,
                       help='Run index for different data subsets (0-9)')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for experiment reproducibility')
    
    # Output parameters
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output filename for the plot (default: results/rnd_linear_sgd_convergence_analysis.png)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset and maze structure
    print("Loading PointMaze dataset...")
    dataset, maze_map = load_pointmaze_dataset()
    
    # Extract all available positions from dataset
    print("Extracting all available positions...")
    all_positions = extract_positions_from_dataset(dataset, num_samples=None)  # Get all positions
    total_positions = len(all_positions)
    print(f"Total available positions in dataset: {total_positions}")
    
    # Single sampling step - use same positions for both noise conditions
    print(f"Creating random subset with {args.num_samples} samples...")
    current_seed = args.seed + args.run_index  # Different seed for each run_index
    np.random.seed(current_seed)
    
    if args.num_samples > total_positions:
        print(f"Warning: Requested {args.num_samples} samples but only {total_positions} available")
        positions = all_positions
    else:
        # Randomly sample indices (deterministic for this run_index) 
        subset_indices = np.random.choice(total_positions, args.num_samples, replace=False)
        subset_indices = np.sort(subset_indices)  # Sort for consistent iteration
        positions = all_positions[subset_indices]
    
    print(f"✓ Selected {len(positions)} positions using seed {current_seed}")
    print(f"Position range: X=[{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}], Y=[{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}]")
    
    # Setup output directory (results/)
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = os.path.join(results_dir, 'rnd_linear_sgd_convergence_analysis.png')
    else:
        # If user specifies a filename without path, put it in results/
        if os.path.dirname(args.output_file) == '':
            args.output_file = os.path.join(results_dir, args.output_file)
    
    # Create shared φ(s) weights (deterministic)
    phi_weights = create_phi_weights_deterministic(args.phi_dim, args.seed)
    
    print(f"\n{'='*80}")
    print("Training RND Linear SGD with Gaussian Noise = 0.0")
    print(f"{'='*80}")
    
    # Train RND Linear SGD with gaussian_noise = 0.0
    set_seed(args.seed)  # Reset seed for consistent initialization
    method_no_noise = RNDLinearSGDMethod(
        feature_dim=args.phi_dim,
        device=device,
        phi_weights=phi_weights
    )
    
    print("Training RND Linear SGD (no noise)...")
    losses_no_noise = method_no_noise.train_on_positions(
        positions, 
        num_epochs=args.num_epochs,
        subset_ratio=1.0,  # Use full dataset
        gaussian_noise=0.0
    )
    
    print(f"\n{'='*80}")
    print("Training RND Linear SGD with Gaussian Noise = 0.5")
    print(f"{'='*80}")
    
    # Train RND Linear SGD with gaussian_noise = 0.5
    set_seed(args.seed)  # Reset seed for consistent initialization
    method_with_noise = RNDLinearSGDMethod(
        feature_dim=args.phi_dim,
        device=device,
        phi_weights=phi_weights  # Use same φ(s) weights
    )
    
    print("Training RND Linear SGD (with noise)...")
    losses_with_noise = method_with_noise.train_on_positions(
        positions, 
        num_epochs=args.num_epochs,
        subset_ratio=1.0,  # Use full dataset
        gaussian_noise=0.5
    )
    
    # Create visualization - dual plot like the final RND plot
    print(f"\n{'='*80}")
    print("Creating convergence plot...")
    print(f"{'='*80}")
    
    epochs = np.arange(1, args.num_epochs + 1)
    
    # Create dual plot: linear and log scale
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Linear scale
    ax1.plot(epochs, losses_no_noise, 'b-', linewidth=2, alpha=0.8, label='Gaussian Noise = 0.0')
    ax1.plot(epochs, losses_with_noise, 'orange', linewidth=2, alpha=0.8, label='Gaussian Noise = 0.5')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Training Loss (MSE)', fontsize=14)
    ax1.set_title(f'RND Linear SGD Training Loss: Noise = 0.0 vs 0.5 (Linear Scale)\n(φ_dim={args.phi_dim}, {args.num_epochs} epochs, {args.num_samples} samples)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale (semilog y)
    ax2.semilogy(epochs, losses_no_noise, 'b-', linewidth=2, alpha=0.8, label='Gaussian Noise = 0.0')
    ax2.semilogy(epochs, losses_with_noise, 'orange', linewidth=2, alpha=0.8, label='Gaussian Noise = 0.5')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Training Loss (MSE) - Log Scale', fontsize=14)
    ax2.set_title(f'RND Linear SGD Training Loss: Noise = 0.0 vs 0.5 (Log Scale)', 
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(args.output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {args.output_file}")
    
    # Display plot
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    print(f"Gaussian Noise = 0.0:")
    print(f"  Initial Loss: {losses_no_noise[0]:.6f}")
    print(f"  Final Loss:   {losses_no_noise[-1]:.6f}")
    print(f"  Improvement: {losses_no_noise[0] - losses_no_noise[-1]:.6f}")
    print(f"Gaussian Noise = 0.5:")
    print(f"  Initial Loss: {losses_with_noise[0]:.6f}")
    print(f"  Final Loss:   {losses_with_noise[-1]:.6f}")
    print(f"  Improvement: {losses_with_noise[0] - losses_with_noise[-1]:.6f}")
    print(f"  Expected Final Loss ≈ {0.5**2:.4f} (noise²)")
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()

