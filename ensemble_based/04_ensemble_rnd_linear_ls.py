#!/usr/bin/env python3
"""
Ensemble RND-Linear (LS) Method Testing

Focused testing of Ensemble RND-Linear with Least Squares fitting.
K=10 linear predictors, each fitted using least squares on bootstrap samples.
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

from ensemble_uncertainty_methods import EnsembleRNDLinearLSMethod
from evaluation import (
    get_maze_map, 
    calculate_ground_truth, 
    calculate_ground_truth_from_positions,
    compute_l2_distance,
    compute_min_c_l1_norm_diff, compute_min_c_l2_norm_diff,
    compute_min_c_l1_norm_inv, compute_min_c_l2_norm_inv,
    save_heatmap_to_wandb,
    get_phi_weights
)
from environment import load_pointmaze_dataset, extract_positions_from_dataset
from debug import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble RND-Linear (LS) Testing")
    
    # Ensemble parameters
    parser.add_argument("--K", type=int, default=10, help="Number of predictors in ensemble")
    parser.add_argument("--num_samples", type=int, default=10000, help="Dataset size (10k)")
    parser.add_argument("--num_averaging_runs", type=int, default=10, help="Number of averaging runs")
    
    # RND-Linear parameters
    parser.add_argument("--phi_dim", type=int, default=128, help="Feature dimension for RND-Linear")
    parser.add_argument("--phi_seed", type=int, default=42, help="Seed for shared φ(s) weights")
    parser.add_argument("--regularization", type=float, default=1e-6, help="Regularization for least squares")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs (not used for LS)")
    parser.add_argument("--gaussian_noise", type=float, default=0.0, help="Gaussian noise level")
    
    # Evaluation parameters
    parser.add_argument("--grid_rows", type=int, default=9, help="Grid rows for evaluation")
    parser.add_argument("--grid_cols", type=int, default=12, help="Grid columns for evaluation")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--a_seed", type=int, default=42, help="Random seed")
    
    # WandB parameters
    parser.add_argument("--wandb_switch", type=str, default="False", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="ensemble-rnd-linear-ls", help="WandB project name")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup random seeds and device"""
    np.random.seed(args.a_seed)
    torch.manual_seed(args.a_seed)
    import random
    random.seed(args.a_seed)
    
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    return device

def run_single_experiment(args, device, run_idx):
    """Run a single experiment with the given parameters"""
    print(f"\n=== Run {run_idx + 1}/{args.num_averaging_runs} ===")
    
    # Load dataset and sample 10k positions (with seed control)
    dataset, maze_map = load_pointmaze_dataset()
    
    # Set seed for reproducible 10k sampling (same for all K heads in this run)
    np.random.seed(args.a_seed)
    positions = extract_positions_from_dataset(dataset, args.num_samples)
    print(f"Sampled {len(positions)} positions from dataset (seed={args.a_seed})")
    
    # Calculate ground truth on the same 10k subset
    ground_truth = calculate_ground_truth_from_positions(
        positions, maze_map, args.grid_rows, args.grid_cols
    )
    print(f"Ground truth calculated: {ground_truth.shape}")
    
    # Get shared φ(s) weights for consistency
    phi_weights = get_phi_weights(args.phi_dim, args.phi_seed)
    
    # Create ensemble RND-Linear (LS) method
    method = EnsembleRNDLinearLSMethod(
        feature_dim=args.phi_dim,
        K=args.K,
        device=device,
        phi_weights=phi_weights,
        regularization=args.regularization
    )
    
    # Train ensemble
    print(f"Training Ensemble RND-Linear (LS) with K={args.K} predictors...")
    losses = method.train_on_positions(
        positions, 
        num_epochs=args.num_epochs,
        gaussian_noise=args.gaussian_noise
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
    
    # Normalize uncertainty matrix
    max_uncertainty = np.max(uncertainty_matrix[~wall_mask])
    if max_uncertainty > 0:
        uncertainty_matrix = uncertainty_matrix / max_uncertainty
    
    # Calculate metrics (use maze_map instead of wall_mask)
    maze_map_array = np.array(maze_map)
    l2_distance = compute_l2_distance(ground_truth, uncertainty_matrix, maze_map_array)
    min_c_l1_norm_diff = compute_min_c_l1_norm_diff(ground_truth, uncertainty_matrix, maze_map_array)
    min_c_l2_norm_diff = compute_min_c_l2_norm_diff(ground_truth, uncertainty_matrix, maze_map_array)
    min_c_l1_norm_inv = compute_min_c_l1_norm_inv(ground_truth, uncertainty_matrix, maze_map_array)
    min_c_l2_norm_inv = compute_min_c_l2_norm_inv(ground_truth, uncertainty_matrix, maze_map_array)
    
    print(f"L2 distance: {l2_distance:.6f}, L1_diff: {min_c_l1_norm_diff:.6f}, L2_diff: {min_c_l2_norm_diff:.6f}, L1_inv: {min_c_l1_norm_inv:.6f}, L2_inv: {min_c_l2_norm_inv:.6f}")
    
    return {
        'l2_distance': l2_distance,
        'min_c_l1_norm_diff': min_c_l1_norm_diff,
        'min_c_l2_norm_diff': min_c_l2_norm_diff,
        'min_c_l1_norm_inv': min_c_l1_norm_inv,
        'min_c_l2_norm_inv': min_c_l2_norm_inv,
        'uncertainty_matrix': uncertainty_matrix,
        'ground_truth': ground_truth,
        'wall_mask': wall_mask,
        'losses': losses
    }

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Setup environment
    device = setup_environment(args)
    
    # Setup WandB if enabled
    if args.wandb_switch.lower() == "true":
        wandb.init(
            project=args.project_name,
            config=vars(args),
            name=f"ensemble_rnd_linear_ls_K{args.K}_samples{args.num_samples}"
        )
    
    print(f"Starting Ensemble RND-Linear (LS) testing")
    print(f"K (number of predictors): {args.K}")
    print(f"Dataset size: {args.num_samples}")
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
        title=f"Ensemble RND-Linear (LS) (K={args.K}) - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    # Log final results to WandB
    if args.wandb_switch.lower() == "true":
        wandb.log({
            "final_mean_l2": mean_l2,
            "final_std_l2": std_l2,
            "final_mean_min_c_l1_norm_diff": mean_min_c_l1_norm_diff,
            "final_std_min_c_l1_norm_diff": std_min_c_l1_norm_diff,
            "final_mean_min_c_l2_norm_diff": mean_min_c_l2_norm_diff,
            "final_std_min_c_l2_norm_diff": std_min_c_l2_norm_diff,
            "final_mean_min_c_l1_norm_inv": mean_min_c_l1_norm_inv,
            "final_std_min_c_l1_norm_inv": std_min_c_l1_norm_inv,
            "final_mean_min_c_l2_norm_inv": mean_min_c_l2_norm_inv,
            "final_std_min_c_l2_norm_inv": std_min_c_l2_norm_inv
        })
        wandb.finish()
    
    print(f"Experiment completed successfully!")

if __name__ == "__main__":
    main()

