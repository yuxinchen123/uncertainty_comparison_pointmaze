#!/usr/bin/env python3

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset
from utilities.uncertainty_methods import RNDMethod

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='RND Convergence Analysis on PointMaze')
    
    # Architecture parameters
    parser.add_argument('--output_dim', type=int, default=64,
                       help='Output dimension for RND networks')
    parser.add_argument('--hidden_dims', type=str, default='128,128',
                       help='Hidden dimensions for RND networks (comma-separated)')
    
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
                       help='Output filename for the plot (default: results/rnd_convergence_analysis.png)')
    
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
    
    # Parse hidden dimensions for RND
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # Setup output directory (results/)
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = os.path.join(results_dir, 'rnd_convergence_analysis.png')
    else:
        # If user specifies a filename without path, put it in results/
        if os.path.dirname(args.output_file) == '':
            args.output_file = os.path.join(results_dir, args.output_file)
    
    print(f"\n{'='*80}")
    print("Training RND with Gaussian Noise = 0.0")
    print(f"{'='*80}")
    
    # Train RND with gaussian_noise = 0.0
    set_seed(args.seed)  # Reset seed for consistent initialization
    method_no_noise = RNDMethod(
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        device=device
    )
    
    print("Training RND (no noise)...")
    losses_no_noise = method_no_noise.train_on_positions(
        positions, 
        num_epochs=args.num_epochs,
        subset_ratio=1.0,  # Use full dataset
        gaussian_noise=0.0
    )
    
    print(f"\n{'='*80}")
    print("Training RND with Gaussian Noise = 1.0")
    print(f"{'='*80}")
    
    # Train RND with gaussian_noise = 1.0
    set_seed(args.seed)  # Reset seed for consistent initialization (will have different target network though)
    method_with_noise = RNDMethod(
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        device=device
    )
    
    print("Training RND (with noise)...")
    losses_with_noise = method_with_noise.train_on_positions(
        positions, 
        num_epochs=args.num_epochs,
        subset_ratio=1.0,  # Use full dataset
        gaussian_noise=1.0
    )
    
    # Create visualization
    print(f"\n{'='*80}")
    print("Creating convergence plot...")
    print(f"{'='*80}")
    
    epochs = np.arange(1, args.num_epochs + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, losses_no_noise, label='Gaussian Noise = 0.0', linewidth=2, alpha=0.8)
    plt.plot(epochs, losses_with_noise, label='Gaussian Noise = 1.0', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Training Loss (MSE)', fontsize=14)
    plt.title(f'RND Training Convergence Analysis\n(hidden_dims={args.hidden_dims}, output_dim={args.output_dim}, {args.num_samples} samples)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
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
    print(f"Gaussian Noise = 1.0:")
    print(f"  Initial Loss: {losses_with_noise[0]:.6f}")
    print(f"  Final Loss:   {losses_with_noise[-1]:.6f}")
    print(f"  Improvement: {losses_with_noise[0] - losses_with_noise[-1]:.6f}")
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()

