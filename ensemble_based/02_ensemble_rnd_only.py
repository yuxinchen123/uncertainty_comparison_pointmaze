#!/usr/bin/env python3
"""
Ensemble RND Method Testing

Focused testing of Ensemble RND with K=10 neural network predictors.
Each predictor samples its own bootstrap dataset with replacement from the 10k data pool.
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

from ensemble_uncertainty_methods import EnsembleRNDMethod
from evaluation import (
    get_maze_map, 
    calculate_ground_truth, 
    calculate_ground_truth_from_positions,
    compute_l2_distance, 
    save_heatmap_to_wandb
)
from environment import load_pointmaze_dataset, extract_positions_from_dataset
from debug import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble RND Testing")
    
    # Ensemble parameters
    parser.add_argument("--K", type=int, default=10, help="Number of predictors in ensemble")
    parser.add_argument("--num_samples", type=int, default=10000, help="Dataset size (10k)")
    parser.add_argument("--num_averaging_runs", type=int, default=10, help="Number of averaging runs")
    
    # RND parameters
    parser.add_argument("--output_dim", type=int, default=128, help="RND output dimension")
    parser.add_argument("--hidden_dims", type=str, default="128,128", help="RND hidden dimensions (comma-separated)")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--gaussian_noise", type=float, default=0.0, help="Gaussian noise level")
    
    # Evaluation parameters
    parser.add_argument("--grid_rows", type=int, default=9, help="Grid rows for evaluation")
    parser.add_argument("--grid_cols", type=int, default=12, help="Grid columns for evaluation")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # WandB parameters
    parser.add_argument("--wandb_switch", type=str, default="False", help="Enable WandB logging")
    parser.add_argument("--project_name", type=str, default="ensemble-rnd", help="WandB project name")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup random seeds and device"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    
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
    np.random.seed(args.seed)
    positions = extract_positions_from_dataset(dataset, args.num_samples)
    print(f"Sampled {len(positions)} positions from dataset (seed={args.seed})")
    
    # Calculate ground truth on the same 10k subset
    ground_truth = calculate_ground_truth_from_positions(
        positions, maze_map, args.grid_rows, args.grid_cols
    )
    print(f"Ground truth calculated: {ground_truth.shape}")
    
    # Create ensemble RND method
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]
    method = EnsembleRNDMethod(
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        K=args.K,
        device=device
    )
    
    # Train ensemble
    print(f"Training Ensemble RND with K={args.K} predictors...")
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
    
    # Calculate L2 distance
    l2_distance = compute_l2_distance(ground_truth, uncertainty_matrix, wall_mask)
    
    print(f"L2 distance: {l2_distance:.6f}")
    
    return {
        'l2_distance': l2_distance,
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
            name=f"ensemble_rnd_K{args.K}_samples{args.num_samples}"
        )
    
    print(f"Starting Ensemble RND testing")
    print(f"K (number of predictors): {args.K}")
    print(f"Dataset size: {args.num_samples}")
    print(f"Number of averaging runs: {args.num_averaging_runs}")
    
    # Run experiments
    results = []
    for run_idx in range(args.num_averaging_runs):
        result = run_single_experiment(args, device, run_idx)
        results.append(result)
        
        # Log to WandB if enabled
        if args.wandb_switch.lower() == "true":
            wandb.log({
                "run": run_idx,
                "l2_distance": result['l2_distance']
            })
    
    # Calculate statistics
    l2_distances = [r['l2_distance'] for r in results]
    mean_l2 = np.mean(l2_distances)
    std_l2 = np.std(l2_distances)
    
    print(f"\n=== Final Results ===")
    print(f"L2 Distance: {mean_l2:.6f} ± {std_l2:.6f}")
    print(f"Individual L2 distances: {[f'{d:.6f}' for d in l2_distances]}")
    
    # Plot final uncertainty heatmap
    final_result = results[-1]  # Use last run for visualization
    save_heatmap_to_wandb(
        final_result['uncertainty_matrix'],
        title=f"Ensemble RND (K={args.K}) - Final Run",
        wandb_switch=(args.wandb_switch.lower() == "true")
    )
    
    # Log final results to WandB
    if args.wandb_switch.lower() == "true":
        wandb.log({
            "final_mean_l2": mean_l2,
            "final_std_l2": std_l2,
            "final_l2_distances": l2_distances
        })
        wandb.finish()
    
    print(f"Experiment completed successfully!")

if __name__ == "__main__":
    main()

