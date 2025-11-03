#!/usr/bin/env python3
"""
Scalar Ensemble Noise-Based RND-Linear (SGD) Method

Ensemble RND-Linear with:
- Same sub-dataset for all K heads (no bootstrap sampling)
- Different Gaussian noise per head: w_t^i ~ N(0, noise_sigma^2)
- Output already scalar, use directly
- Uncertainty = STD of scalar predictions across K heads
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import os
import sys
from pathlib import Path

# Add utilities to path
sys.path.append(str(Path(__file__).parent / "utilities"))

from scalar_ensemble_methods import ScalarEnsembleRNDLinearSGDMethod
from evaluation import (
    get_maze_map, 
    calculate_ground_truth_from_positions,
    compute_l2_distance,
    compute_min_c_l1_norm_diff, compute_min_c_l2_norm_diff,
    compute_min_c_l1_norm_inv, compute_min_c_l2_norm_inv,
    save_heatmap_to_wandb,
    normalize_uncertainty_matrix,
    get_phi_weights
)
from environment import load_pointmaze_dataset, extract_positions_from_dataset

def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Scalar Ensemble Noise-Based RND-Linear (SGD) Testing")
    
    # Ensemble parameters
    parser.add_argument("--K", type=int, default=10, help="Number of predictors in ensemble")
    parser.add_argument("--num_samples", type=int, default=10000, help="Dataset size (10k)")
    parser.add_argument("--num_averaging_runs", type=int, default=10, help="Number of averaging runs")
    
    # RND-Linear parameters
    parser.add_argument("--phi_dim", type=int, default=128, help="Feature dimension for RND-Linear")
    parser.add_argument("--phi_seed", type=int, default=42, help="Seed for shared φ(s) weights")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--noise_sigma", type=float, default=0.0, help="Gaussian noise sigma (noise_sigma^2 variance)")
    
    # Evaluation parameters
    parser.add_argument("--grid_rows", type=int, default=9, help="Grid rows for evaluation")
    parser.add_argument("--grid_cols", type=int, default=12, help="Grid columns for evaluation")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--a_seed", type=int, default=42, help="Random seed")
    
    # WandB parameters
    parser.add_argument("--wandb_switch", type=str, default="False", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="scalar-ensemble-rnd-linear-sgd", help="WandB project name")
    
    return parser.parse_args()

def run_single_experiment(args, device, run_idx):
    """Run a single experiment with the given parameters"""
    print(f"\n=== Run {run_idx + 1}/{args.num_averaging_runs} ===")
    
    # Load dataset and sample positions
    dataset, maze_map = load_pointmaze_dataset()
    
    # Set seed for reproducible sampling
    current_seed = args.a_seed * 1000 + run_idx
    set_seed(current_seed)
    positions = extract_positions_from_dataset(dataset, args.num_samples)
    print(f"Sampled {len(positions)} positions from dataset (seed={current_seed})")
    
    # Calculate ground truth on the same subset
    ground_truth = calculate_ground_truth_from_positions(
        positions, maze_map, args.grid_rows, args.grid_cols
    )
    print(f"Ground truth calculated: {ground_truth.shape}")
    
    # Get shared φ(s) weights for consistency
    phi_weights = get_phi_weights(args.phi_dim, args.phi_seed)
    
    # Create scalar ensemble RND-Linear (SGD) method
    method = ScalarEnsembleRNDLinearSGDMethod(
        feature_dim=args.phi_dim,
        K=args.K,
        device=device,
        phi_weights=phi_weights
    )
    
    # Train ensemble
    print(f"Training Scalar Ensemble RND-Linear (SGD) with K={args.K} predictors...")
    losses = method.train_on_positions(
        positions, 
        num_epochs=args.num_epochs,
        noise_sigma=args.noise_sigma
    )
    
    # Evaluate on grid
    print("Evaluating on grid...")
    grid_coords = []
    for row in range(args.grid_rows):
        for col in range(args.grid_cols):
            # Convert grid coordinates to maze coordinates
            x = -6 + (col + 0.5) * (12 / args.grid_cols)
            y = -4.5 + (row + 0.5) * (9 / args.grid_rows)
            grid_coords.append([x, y])
    
    grid_coords = np.array(grid_coords)
    uncertainty_values = method.get_uncertainty(grid_coords)
    
    # Reshape to grid
    uncertainty_matrix = uncertainty_values.reshape(args.grid_rows, args.grid_cols)
    
    # Apply wall mask
    wall_mask = np.array(maze_map) == 1
    uncertainty_matrix[wall_mask] = 0
    
    # Keep original (unnormalized) for the 4 new metrics
    uncertainty_matrix_original = uncertainty_matrix.copy()
    
    # Normalize for L2 distance (original metric)
    uncertainty_matrix_normalized = normalize_uncertainty_matrix(uncertainty_matrix, maze_map)
    ground_truth_normalized = normalize_uncertainty_matrix(ground_truth, maze_map)
    
    # Calculate metrics
    maze_map_array = np.array(maze_map)
    # L2 distance: use normalized GT and normalized predictions
    l2_distance = compute_l2_distance(ground_truth_normalized, uncertainty_matrix_normalized, maze_map_array)
    # The 4 new metrics: use original (unnormalized) GT and predictions
    min_c_l1_norm_diff = compute_min_c_l1_norm_diff(ground_truth, uncertainty_matrix_original, maze_map_array)
    min_c_l2_norm_diff = compute_min_c_l2_norm_diff(ground_truth, uncertainty_matrix_original, maze_map_array)
    min_c_l1_norm_inv = compute_min_c_l1_norm_inv(ground_truth, uncertainty_matrix_original, maze_map_array)
    min_c_l2_norm_inv = compute_min_c_l2_norm_inv(ground_truth, uncertainty_matrix_original, maze_map_array)
    
    print(f"L2 distance: {l2_distance:.6f}, L1_diff: {min_c_l1_norm_diff:.6f}, L2_diff: {min_c_l2_norm_diff:.6f}, L1_inv: {min_c_l1_norm_inv:.6f}, L2_inv: {min_c_l2_norm_inv:.6f}")
    
    return {
        'l2_distance': l2_distance,
        'min_c_l1_norm_diff': min_c_l1_norm_diff,
        'min_c_l2_norm_diff': min_c_l2_norm_diff,
        'min_c_l1_norm_inv': min_c_l1_norm_inv,
        'min_c_l2_norm_inv': min_c_l2_norm_inv,
        'uncertainty_matrix': uncertainty_matrix,
        'ground_truth': ground_truth_normalized,
        'losses': losses
    }

def main():
    args = parse_args()
    
    # Handle WandB sweep
    if wandb.run is not None:
        config = wandb.config
        args.K = config.get('K', args.K)
        args.num_samples = config.get('num_samples', args.num_samples)
        args.num_averaging_runs = config.get('num_averaging_runs', args.num_averaging_runs)
        args.phi_dim = config.get('phi_dim', args.phi_dim)
        args.phi_seed = config.get('phi_seed', args.phi_seed)
        args.num_epochs = config.get('num_epochs', args.num_epochs)
        args.noise_sigma = config.get('noise_sigma', args.noise_sigma)
        args.a_seed = config.get('a_seed', args.a_seed)
        args.wandb_switch = config.get('wandb_switch', args.wandb_switch)
        args.grid_rows = config.get('grid_rows', args.grid_rows)
        args.grid_cols = config.get('grid_cols', args.grid_cols)
        print("🔄 Running in WandB sweep mode - Scalar Ensemble RND-Linear (SGD)")
    else:
        if args.wandb_switch.lower() == "true":
            wandb.init(
                project=args.project_name,
                config=vars(args),
                name=f"scalar_ensemble_rnd_linear_sgd_K{args.K}_noise{args.noise_sigma}"
            )
    
    # Setup environment
    set_seed(args.a_seed)
    device = 'cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Starting Scalar Ensemble Noise-Based RND-Linear (SGD) testing")
    print(f"K (number of predictors): {args.K}")
    print(f"Dataset size: {args.num_samples}")
    print(f"Noise sigma: {args.noise_sigma}")
    print(f"Number of averaging runs: {args.num_averaging_runs}")
    
    # Run experiments
    results = []
    for run_idx in range(args.num_averaging_runs):
        result = run_single_experiment(args, device, run_idx)
        results.append(result)
    
    # Calculate statistics
    l2_distances = [r['l2_distance'] for r in results]
    min_c_l1_norm_diff_list = [r['min_c_l1_norm_diff'] for r in results]
    min_c_l2_norm_diff_list = [r['min_c_l2_norm_diff'] for r in results]
    min_c_l1_norm_inv_list = [r['min_c_l1_norm_inv'] for r in results]
    min_c_l2_norm_inv_list = [r['min_c_l2_norm_inv'] for r in results]
    
    mean_l2 = np.nanmean(l2_distances)
    std_l2 = np.nanstd(l2_distances)
    mean_min_c_l1_norm_diff = np.nanmean(min_c_l1_norm_diff_list)
    std_min_c_l1_norm_diff = np.nanstd(min_c_l1_norm_diff_list)
    mean_min_c_l2_norm_diff = np.nanmean(min_c_l2_norm_diff_list)
    std_min_c_l2_norm_diff = np.nanstd(min_c_l2_norm_diff_list)
    mean_min_c_l1_norm_inv = np.nanmean(min_c_l1_norm_inv_list)
    std_min_c_l1_norm_inv = np.nanstd(min_c_l1_norm_inv_list)
    mean_min_c_l2_norm_inv = np.nanmean(min_c_l2_norm_inv_list)
    std_min_c_l2_norm_inv = np.nanstd(min_c_l2_norm_inv_list)
    
    print(f"\n=== Final Results ===")
    print(f"L2 Distance:           {mean_l2:.6f} ± {std_l2:.6f}")
    print(f"min_c L1 diff:         {mean_min_c_l1_norm_diff:.6f} ± {std_min_c_l1_norm_diff:.6f}")
    print(f"min_c L2 diff:         {mean_min_c_l2_norm_diff:.6f} ± {std_min_c_l2_norm_diff:.6f}")
    print(f"min_c L1 inv:          {mean_min_c_l1_norm_inv:.6f} ± {std_min_c_l1_norm_inv:.6f}")
    print(f"min_c L2 inv:          {mean_min_c_l2_norm_inv:.6f} ± {std_min_c_l2_norm_inv:.6f}")
    
    # Plot final uncertainty heatmap
    final_result = results[-1]  # Use last run for visualization
    
    # Log ground truth heatmap
    save_heatmap_to_wandb(
        final_result['ground_truth'],
        title=f"Ground Truth Uncertainty - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    # Log method heatmap
    save_heatmap_to_wandb(
        final_result['uncertainty_matrix'],
        title=f"Scalar Ensemble RND-Linear (SGD) (K={args.K}, noise_sigma={args.noise_sigma}) - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    # Log final results to WandB
    if args.wandb_switch.lower() == "true":
        wandb.log({
            "final_mean_l2": mean_l2,
            "final_mean_min_c_l1_norm_diff": mean_min_c_l1_norm_diff,
            "final_mean_min_c_l2_norm_diff": mean_min_c_l2_norm_diff,
            "final_mean_min_c_l1_norm_inv": mean_min_c_l1_norm_inv,
            "final_mean_min_c_l2_norm_inv": mean_min_c_l2_norm_inv
        })
        wandb.finish()
    
    print(f"Experiment completed successfully!")

if __name__ == "__main__":
    main()

