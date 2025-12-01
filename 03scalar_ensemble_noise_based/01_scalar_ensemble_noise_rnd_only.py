#!/usr/bin/env python3
"""
Scalar Ensemble Noise-Based RND Method

Ensemble RND with:
- Same sub-dataset for all K heads (no bootstrap sampling)
- Different Gaussian noise per head: w_t^i ~ N(0, noise_sigma^2)
- Scalar projection layer: output_dim -> 1 (trained)
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

from scalar_ensemble_methods import ScalarEnsembleRNDMethod
from evaluation import (
    get_maze_map, 
    calculate_ground_truth_from_positions,
    compute_l2_distance,
    compute_min_c_l1_norm_diff, compute_min_c_l2_norm_diff,
    compute_min_c_l1_norm_inv, compute_min_c_l2_norm_inv,
    save_heatmap_to_wandb,
    normalize_uncertainty_matrix
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
    parser = argparse.ArgumentParser(description="Scalar Ensemble Noise-Based RND Testing")
    
    # Ensemble parameters
    parser.add_argument("--num_heads", type=int, default=10, help="Number of predictors (heads) in ensemble")
    parser.add_argument("--num_samples", type=int, default=10000, help="Dataset size (10k)")
    parser.add_argument("--num_averaging_runs", type=int, default=10, help="Number of averaging runs")
    
    # RND parameters
    parser.add_argument("--output_dim", type=int, default=128, help="RND output dimension")
    parser.add_argument("--hidden_dims", type=str, default="128,128", help="RND hidden dimensions (comma-separated)")
    
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
    parser.add_argument("--project_name", type=str, default="scalar-ensemble-rnd", help="WandB project name")
    
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
    
    # Create scalar ensemble RND method
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]
    method = ScalarEnsembleRNDMethod(
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        device=device
    )
    
    # Train ensemble
    print(f"Training Scalar Ensemble RND with {args.num_heads} predictors...")
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
    
    # Get both uncertainty types
    uncertainty_values_errors = method.get_uncertainty_errors(grid_coords)
    uncertainty_values_predictions = method.get_uncertainty_predictions(grid_coords)
    
    # Reshape to grid
    uncertainty_matrix_errors = uncertainty_values_errors.reshape(args.grid_rows, args.grid_cols)
    uncertainty_matrix_predictions = uncertainty_values_predictions.reshape(args.grid_rows, args.grid_cols)
    
    # Apply wall mask
    wall_mask = np.array(maze_map) == 1
    uncertainty_matrix_errors[wall_mask] = 0
    uncertainty_matrix_predictions[wall_mask] = 0
    
    # Keep original (unnormalized) for the 4 new metrics
    uncertainty_matrix_errors_original = uncertainty_matrix_errors.copy()
    uncertainty_matrix_predictions_original = uncertainty_matrix_predictions.copy()
    
    # Normalize for L2 distance (original metric)
    uncertainty_matrix_errors_normalized = normalize_uncertainty_matrix(uncertainty_matrix_errors, maze_map)
    uncertainty_matrix_predictions_normalized = normalize_uncertainty_matrix(uncertainty_matrix_predictions, maze_map)
    ground_truth_normalized = normalize_uncertainty_matrix(ground_truth, maze_map)
    
    # Calculate metrics
    maze_map_array = np.array(maze_map)
    
    # Error-based metrics (5 metrics)
    l2_distance_errors = compute_l2_distance(ground_truth_normalized, uncertainty_matrix_errors_normalized, maze_map_array)
    min_c_l1_norm_diff_errors = compute_min_c_l1_norm_diff(ground_truth, uncertainty_matrix_errors_original, maze_map_array)
    min_c_l2_norm_diff_errors = compute_min_c_l2_norm_diff(ground_truth, uncertainty_matrix_errors_original, maze_map_array)
    min_c_l1_norm_inv_errors = compute_min_c_l1_norm_inv(ground_truth, uncertainty_matrix_errors_original, maze_map_array)
    min_c_l2_norm_inv_errors = compute_min_c_l2_norm_inv(ground_truth, uncertainty_matrix_errors_original, maze_map_array)
    
    # Prediction-based metrics (5 metrics)
    l2_distance_predictions = compute_l2_distance(ground_truth_normalized, uncertainty_matrix_predictions_normalized, maze_map_array)
    min_c_l1_norm_diff_predictions = compute_min_c_l1_norm_diff(ground_truth, uncertainty_matrix_predictions_original, maze_map_array)
    min_c_l2_norm_diff_predictions = compute_min_c_l2_norm_diff(ground_truth, uncertainty_matrix_predictions_original, maze_map_array)
    min_c_l1_norm_inv_predictions = compute_min_c_l1_norm_inv(ground_truth, uncertainty_matrix_predictions_original, maze_map_array)
    min_c_l2_norm_inv_predictions = compute_min_c_l2_norm_inv(ground_truth, uncertainty_matrix_predictions_original, maze_map_array)
    
    print(f"Errors - L2: {l2_distance_errors:.6f}, L1_diff: {min_c_l1_norm_diff_errors:.6f}, L2_diff: {min_c_l2_norm_diff_errors:.6f}, L1_inv: {min_c_l1_norm_inv_errors:.6f}, L2_inv: {min_c_l2_norm_inv_errors:.6f}")
    print(f"Predictions - L2: {l2_distance_predictions:.6f}, L1_diff: {min_c_l1_norm_diff_predictions:.6f}, L2_diff: {min_c_l2_norm_diff_predictions:.6f}, L1_inv: {min_c_l1_norm_inv_predictions:.6f}, L2_inv: {min_c_l2_norm_inv_predictions:.6f}")
    
    return {
        # Error-based metrics
        'l2_distance_errors': l2_distance_errors,
        'min_c_l1_norm_diff_errors': min_c_l1_norm_diff_errors,
        'min_c_l2_norm_diff_errors': min_c_l2_norm_diff_errors,
        'min_c_l1_norm_inv_errors': min_c_l1_norm_inv_errors,
        'min_c_l2_norm_inv_errors': min_c_l2_norm_inv_errors,
        # Prediction-based metrics
        'l2_distance_predictions': l2_distance_predictions,
        'min_c_l1_norm_diff_predictions': min_c_l1_norm_diff_predictions,
        'min_c_l2_norm_diff_predictions': min_c_l2_norm_diff_predictions,
        'min_c_l1_norm_inv_predictions': min_c_l1_norm_inv_predictions,
        'min_c_l2_norm_inv_predictions': min_c_l2_norm_inv_predictions,
        # Uncertainty matrices
        'uncertainty_matrix_errors': uncertainty_matrix_errors,
        'uncertainty_matrix_predictions': uncertainty_matrix_predictions,
        'ground_truth': ground_truth_normalized,
        'losses': losses
    }

def main():
    args = parse_args()
    
    # Handle WandB sweep
    if wandb.run is not None:
        config = wandb.config
        args.num_heads = config.get('num_heads', args.num_heads)
        args.num_samples = config.get('num_samples', args.num_samples)
        args.num_averaging_runs = config.get('num_averaging_runs', args.num_averaging_runs)
        args.output_dim = config.get('output_dim', args.output_dim)
        args.hidden_dims = config.get('hidden_dims', args.hidden_dims)
        args.num_epochs = config.get('num_epochs', args.num_epochs)
        args.noise_sigma = config.get('noise_sigma', args.noise_sigma)
        args.a_seed = config.get('a_seed', args.a_seed)
        args.wandb_switch = config.get('wandb_switch', args.wandb_switch)
        args.grid_rows = config.get('grid_rows', args.grid_rows)
        args.grid_cols = config.get('grid_cols', args.grid_cols)
        print("🔄 Running in WandB sweep mode - Scalar Ensemble RND")
    else:
        if args.wandb_switch.lower() == "true":
            wandb.init(
                project=args.project_name,
                config=vars(args),
                name=f"scalar_ensemble_rnd_num_heads{args.num_heads}_noise{args.noise_sigma}"
            )
    
    # Setup environment
    set_seed(args.a_seed)
    device = 'cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Starting Scalar Ensemble Noise-Based RND testing")
    print(f"num_heads (number of predictors): {args.num_heads}")
    print(f"Dataset size: {args.num_samples}")
    print(f"Noise sigma: {args.noise_sigma}")
    print(f"Number of averaging runs: {args.num_averaging_runs}")
    
    # Run experiments
    results = []
    for run_idx in range(args.num_averaging_runs):
        result = run_single_experiment(args, device, run_idx)
        results.append(result)
    
    # Calculate statistics for all 10 metrics
    # Error-based metrics
    l2_distances_errors = [r['l2_distance_errors'] for r in results]
    min_c_l1_norm_diff_errors_list = [r['min_c_l1_norm_diff_errors'] for r in results]
    min_c_l2_norm_diff_errors_list = [r['min_c_l2_norm_diff_errors'] for r in results]
    min_c_l1_norm_inv_errors_list = [r['min_c_l1_norm_inv_errors'] for r in results]
    min_c_l2_norm_inv_errors_list = [r['min_c_l2_norm_inv_errors'] for r in results]
    
    # Prediction-based metrics
    l2_distances_predictions = [r['l2_distance_predictions'] for r in results]
    min_c_l1_norm_diff_predictions_list = [r['min_c_l1_norm_diff_predictions'] for r in results]
    min_c_l2_norm_diff_predictions_list = [r['min_c_l2_norm_diff_predictions'] for r in results]
    min_c_l1_norm_inv_predictions_list = [r['min_c_l1_norm_inv_predictions'] for r in results]
    min_c_l2_norm_inv_predictions_list = [r['min_c_l2_norm_inv_predictions'] for r in results]
    
    # Compute means and stds
    mean_l2_errors = np.nanmean(l2_distances_errors)
    std_l2_errors = np.nanstd(l2_distances_errors)
    mean_min_c_l1_norm_diff_errors = np.nanmean(min_c_l1_norm_diff_errors_list)
    std_min_c_l1_norm_diff_errors = np.nanstd(min_c_l1_norm_diff_errors_list)
    mean_min_c_l2_norm_diff_errors = np.nanmean(min_c_l2_norm_diff_errors_list)
    std_min_c_l2_norm_diff_errors = np.nanstd(min_c_l2_norm_diff_errors_list)
    mean_min_c_l1_norm_inv_errors = np.nanmean(min_c_l1_norm_inv_errors_list)
    std_min_c_l1_norm_inv_errors = np.nanstd(min_c_l1_norm_inv_errors_list)
    mean_min_c_l2_norm_inv_errors = np.nanmean(min_c_l2_norm_inv_errors_list)
    std_min_c_l2_norm_inv_errors = np.nanstd(min_c_l2_norm_inv_errors_list)
    
    mean_l2_predictions = np.nanmean(l2_distances_predictions)
    std_l2_predictions = np.nanstd(l2_distances_predictions)
    mean_min_c_l1_norm_diff_predictions = np.nanmean(min_c_l1_norm_diff_predictions_list)
    std_min_c_l1_norm_diff_predictions = np.nanstd(min_c_l1_norm_diff_predictions_list)
    mean_min_c_l2_norm_diff_predictions = np.nanmean(min_c_l2_norm_diff_predictions_list)
    std_min_c_l2_norm_diff_predictions = np.nanstd(min_c_l2_norm_diff_predictions_list)
    mean_min_c_l1_norm_inv_predictions = np.nanmean(min_c_l1_norm_inv_predictions_list)
    std_min_c_l1_norm_inv_predictions = np.nanstd(min_c_l1_norm_inv_predictions_list)
    mean_min_c_l2_norm_inv_predictions = np.nanmean(min_c_l2_norm_inv_predictions_list)
    std_min_c_l2_norm_inv_predictions = np.nanstd(min_c_l2_norm_inv_predictions_list)
    
    print(f"\n=== Final Results (Error-based) ===")
    print(f"L2 Distance:           {mean_l2_errors:.6f} ± {std_l2_errors:.6f}")
    print(f"min_c L1 diff:         {mean_min_c_l1_norm_diff_errors:.6f} ± {std_min_c_l1_norm_diff_errors:.6f}")
    print(f"min_c L2 diff:         {mean_min_c_l2_norm_diff_errors:.6f} ± {std_min_c_l2_norm_diff_errors:.6f}")
    print(f"min_c L1 inv:          {mean_min_c_l1_norm_inv_errors:.6f} ± {std_min_c_l1_norm_inv_errors:.6f}")
    print(f"min_c L2 inv:          {mean_min_c_l2_norm_inv_errors:.6f} ± {std_min_c_l2_norm_inv_errors:.6f}")
    
    print(f"\n=== Final Results (Prediction-based) ===")
    print(f"L2 Distance:           {mean_l2_predictions:.6f} ± {std_l2_predictions:.6f}")
    print(f"min_c L1 diff:         {mean_min_c_l1_norm_diff_predictions:.6f} ± {std_min_c_l1_norm_diff_predictions:.6f}")
    print(f"min_c L2 diff:         {mean_min_c_l2_norm_diff_predictions:.6f} ± {std_min_c_l2_norm_diff_predictions:.6f}")
    print(f"min_c L1 inv:          {mean_min_c_l1_norm_inv_predictions:.6f} ± {std_min_c_l1_norm_inv_predictions:.6f}")
    print(f"min_c L2 inv:          {mean_min_c_l2_norm_inv_predictions:.6f} ± {std_min_c_l2_norm_inv_predictions:.6f}")
    
    # Plot final uncertainty heatmap
    final_result = results[-1]  # Use last run for visualization
    
    # Log ground truth heatmap
    save_heatmap_to_wandb(
        final_result['ground_truth'],
        title=f"Ground Truth Uncertainty - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    # Log method heatmaps
    save_heatmap_to_wandb(
        final_result['uncertainty_matrix_errors'],
        title=f"Scalar Ensemble RND (Errors) (num_heads={args.num_heads}, noise_sigma={args.noise_sigma}) - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    save_heatmap_to_wandb(
        final_result['uncertainty_matrix_predictions'],
        title=f"Scalar Ensemble RND (Predictions) (num_heads={args.num_heads}, noise_sigma={args.noise_sigma}) - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    # Log final results to WandB
    if args.wandb_switch.lower() == "true":
        wandb.log({
            # Error-based metrics
            "final_mean_l2_errors": mean_l2_errors,
            "final_mean_min_c_l1_norm_diff_errors": mean_min_c_l1_norm_diff_errors,
            "final_mean_min_c_l2_norm_diff_errors": mean_min_c_l2_norm_diff_errors,
            "final_mean_min_c_l1_norm_inv_errors": mean_min_c_l1_norm_inv_errors,
            "final_mean_min_c_l2_norm_inv_errors": mean_min_c_l2_norm_inv_errors,
            # Prediction-based metrics
            "final_mean_l2_predictions": mean_l2_predictions,
            "final_mean_min_c_l1_norm_diff_predictions": mean_min_c_l1_norm_diff_predictions,
            "final_mean_min_c_l2_norm_diff_predictions": mean_min_c_l2_norm_diff_predictions,
            "final_mean_min_c_l1_norm_inv_predictions": mean_min_c_l1_norm_inv_predictions,
            "final_mean_min_c_l2_norm_inv_predictions": mean_min_c_l2_norm_inv_predictions
        })
        wandb.finish()
    
    print(f"Experiment completed successfully!")

if __name__ == "__main__":
    main()

