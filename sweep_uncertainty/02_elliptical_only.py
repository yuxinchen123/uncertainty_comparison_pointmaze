#!/usr/bin/env python3

import argparse
import random
import numpy as np
import torch
import wandb
import os

# Fix these imports:
from utilities.debug import print_or_wandb_log
from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset
from utilities.uncertainty_methods import EllipticalBonusMethod
from utilities.evaluation import (
    calculate_ground_truth_from_positions, evaluate_uncertainty_method, 
    normalize_uncertainty_matrix, compute_l2_distance,
    compute_min_c_l1_norm_diff, compute_min_c_l2_norm_diff,
    compute_min_c_l1_norm_inv, compute_min_c_l2_norm_inv,
    save_heatmap_to_wandb
)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_phi_weights_deterministic(feature_dim, seed):
    """Create deterministic φ(s) weights for sharing between runs"""
    torch.manual_seed(seed)
    temp_layer = torch.nn.Linear(2, feature_dim)
    phi_weights = temp_layer.state_dict()
    return phi_weights

def main():
    parser = argparse.ArgumentParser(description='Elliptical Bonus Method Sweep on PointMaze')
    
    # Method is fixed
    parser.add_argument('--method', type=str, default='elliptical',
                       help='Method (fixed to elliptical)')
    
    # Core Elliptical Bonus parameters
    parser.add_argument('--phi_dim', type=int, default=64,
                       help='Feature dimension for φ(s) mapping')
    parser.add_argument('--phi_seed', type=int, default=0,
                       help='Seed for generating φ(s) weights')
    parser.add_argument('--regularization', type=float, default=1e-6,
                       help='Regularization parameter for covariance matrix')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to extract from dataset')
    parser.add_argument('--num_averaging_runs', type=int, default=10,
                       help='Number of runs to average over')
    
    # Experiment parameters
    parser.add_argument('--a_seed', type=int, default=0,
                       help='Random seed for experiment reproducibility')
    parser.add_argument('--wandb_switch', type=bool, default=True,
                       help='Whether to log to WandB')
    
    # Grid parameters
    parser.add_argument('--grid_rows', type=int, default=9)
    parser.add_argument('--grid_cols', type=int, default=12)
    
    # Compatibility parameters (ignored)
    parser.add_argument('--gaussian_noise', type=float, default=0.0,
                       help='Ignored for elliptical method')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Ignored for elliptical method')
    
    args = parser.parse_args()
    
    # Handle WandB sweep
    if wandb.run is not None:
        config = wandb.config
        args.phi_dim = config.get('phi_dim', args.phi_dim)
        args.phi_seed = config.get('phi_seed', args.phi_seed)
        args.regularization = config.get('regularization', args.regularization)
        args.num_samples = config.get('num_samples', args.num_samples)
        args.num_averaging_runs = config.get('num_averaging_runs', args.num_averaging_runs)
        args.a_seed = config.get('a_seed', args.a_seed)
        args.wandb_switch = config.get('wandb_switch', args.wandb_switch)
        args.grid_rows = config.get('grid_rows', args.grid_rows)
        args.grid_cols = config.get('grid_cols', args.grid_cols)
        print("🔄 Running in WandB sweep mode - Elliptical Bonus")
    else:
        if args.wandb_switch:
            wandb.init(
                project="elliptical_bonus_pointmaze",
                config=vars(args)
            )
    
    # Set seed
    set_seed(args.a_seed)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading PointMaze dataset...")
    dataset, maze_map = load_pointmaze_dataset()
    
    # Extract positions
    all_positions = extract_positions_from_dataset(dataset, num_samples=None)
    total_positions = len(all_positions)
    print(f"Total available positions: {total_positions}")
    
    # Sample positions
    np.random.seed(args.a_seed)
    if args.num_samples > total_positions:
        positions = all_positions
    else:
        subset_indices = np.random.choice(total_positions, args.num_samples, replace=False)
        positions = all_positions[subset_indices]
    
    print(f"Selected {len(positions)} positions")
    
    # Calculate ground truth
    gt_uncertainty = calculate_ground_truth_from_positions(
        positions, maze_map, args.grid_rows, args.grid_cols
    )
    gt_normalized = normalize_uncertainty_matrix(gt_uncertainty)
    
    # Store results for averaging
    all_l2_distances = []
    all_min_c_l1_norm_diff = []
    all_min_c_l2_norm_diff = []
    all_min_c_l1_norm_inv = []
    all_min_c_l2_norm_inv = []
    all_condition_numbers = []
    all_covariance_traces = []
    
    print(f"\nStarting {args.num_averaging_runs} averaging runs...")
    print(f"Elliptical Bonus: φ_dim={args.phi_dim}, φ_seed={args.phi_seed}, reg={args.regularization}")
    print("=" * 80)
    
    for avg_run in range(args.num_averaging_runs):
        print(f"\n🔄 RUN {avg_run + 1}/{args.num_averaging_runs}")
        
        # Set seed for this run
        current_seed = args.a_seed * 1000 + avg_run
        set_seed(current_seed)
        
        # Create shared φ(s) weights (same across all averaging runs)
        phi_weights = create_phi_weights_deterministic(args.phi_dim, args.phi_seed)
        
        # Initialize Elliptical Bonus
        method = EllipticalBonusMethod(
            feature_dim=args.phi_dim,
            device=device,
            phi_weights=phi_weights,
            regularization=args.regularization
        )
        
        # Update covariance matrix
        print("Updating covariance matrix...")
        method.update_covariance_from_positions(positions)
        
        # Compute diagnostics
        try:
            eigenvals = torch.linalg.eigvals(method.covariance).real
            condition_number = (torch.max(eigenvals) / torch.min(eigenvals)).item()
            covariance_trace = torch.trace(method.covariance).item()
            all_condition_numbers.append(condition_number)
            all_covariance_traces.append(covariance_trace)
        except:
            condition_number = float('inf')
            covariance_trace = float('nan')
            all_condition_numbers.append(condition_number)
            all_covariance_traces.append(covariance_trace)
        
        # Evaluate
        pred_uncertainty = evaluate_uncertainty_method(
            method, maze_map, args.grid_rows, args.grid_cols, device
        )
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics
        # L2 distance: use normalized GT and normalized predictions
        l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        # The 4 new metrics: use original (unnormalized) GT and predictions
        min_c_l1_norm_diff = compute_min_c_l1_norm_diff(gt_uncertainty, pred_uncertainty, maze_map)
        min_c_l2_norm_diff = compute_min_c_l2_norm_diff(gt_uncertainty, pred_uncertainty, maze_map)
        min_c_l1_norm_inv = compute_min_c_l1_norm_inv(gt_uncertainty, pred_uncertainty, maze_map)
        min_c_l2_norm_inv = compute_min_c_l2_norm_inv(gt_uncertainty, pred_uncertainty, maze_map)
        
        all_l2_distances.append(l2_distance)
        all_min_c_l1_norm_diff.append(min_c_l1_norm_diff)
        all_min_c_l2_norm_diff.append(min_c_l2_norm_diff)
        all_min_c_l1_norm_inv.append(min_c_l1_norm_inv)
        all_min_c_l2_norm_inv.append(min_c_l2_norm_inv)
        
        print(f"  L2={l2_distance:.6f}, L1_diff={min_c_l1_norm_diff:.6f}, L2_diff={min_c_l2_norm_diff:.6f}, L1_inv={min_c_l1_norm_inv:.6f}, L2_inv={min_c_l2_norm_inv:.6f}, Cond={condition_number:.2e}")
    
    # Calculate averaged results
    avg_l2_distance = np.nanmean(all_l2_distances)
    std_l2_distance = np.nanstd(all_l2_distances)
    avg_min_c_l1_norm_diff = np.nanmean(all_min_c_l1_norm_diff)
    std_min_c_l1_norm_diff = np.nanstd(all_min_c_l1_norm_diff)
    avg_min_c_l2_norm_diff = np.nanmean(all_min_c_l2_norm_diff)
    std_min_c_l2_norm_diff = np.nanstd(all_min_c_l2_norm_diff)
    avg_min_c_l1_norm_inv = np.nanmean(all_min_c_l1_norm_inv)
    std_min_c_l1_norm_inv = np.nanstd(all_min_c_l1_norm_inv)
    avg_min_c_l2_norm_inv = np.nanmean(all_min_c_l2_norm_inv)
    std_min_c_l2_norm_inv = np.nanstd(all_min_c_l2_norm_inv)
    avg_condition_number = np.mean([x for x in all_condition_numbers if np.isfinite(x)])
    avg_covariance_trace = np.mean([x for x in all_covariance_traces if np.isfinite(x)])
    
    print(f"\n📊 AVERAGED RESULTS ({args.num_averaging_runs} runs)")
    print("=" * 80)
    print(f"L2 Distance:           {avg_l2_distance:.6f} ± {std_l2_distance:.6f}")
    print(f"min_c L1 diff:         {avg_min_c_l1_norm_diff:.6f} ± {std_min_c_l1_norm_diff:.6f}")
    print(f"min_c L2 diff:         {avg_min_c_l2_norm_diff:.6f} ± {std_min_c_l2_norm_diff:.6f}")
    print(f"min_c L1 inv:          {avg_min_c_l1_norm_inv:.6f} ± {std_min_c_l1_norm_inv:.6f}")
    print(f"min_c L2 inv:          {avg_min_c_l2_norm_inv:.6f} ± {std_min_c_l2_norm_inv:.6f}")
    print(f"Avg Condition Number:  {avg_condition_number:.2e}")
    print(f"Avg Covariance Trace: {avg_covariance_trace:.4f}")
    
    # Save heatmaps
    save_heatmap_to_wandb(gt_normalized, "Ground Truth (1/√N)", args.wandb_switch, maze_map)
    save_heatmap_to_wandb(pred_normalized, f"Elliptical-{args.phi_dim}dim", args.wandb_switch, maze_map)
    
    # Final results
    results = {
        'method': 'elliptical',
        'phi_dim': args.phi_dim,
        'phi_seed': args.phi_seed,
        'regularization': args.regularization,
        'avg_l2_distance': avg_l2_distance,
        'avg_min_c_l1_norm_diff': avg_min_c_l1_norm_diff,
        'avg_min_c_l2_norm_diff': avg_min_c_l2_norm_diff,
        'avg_min_c_l1_norm_inv': avg_min_c_l1_norm_inv,
        'avg_min_c_l2_norm_inv': avg_min_c_l2_norm_inv,
        'avg_condition_number': avg_condition_number,
        'avg_covariance_trace': avg_covariance_trace,
        'num_samples': args.num_samples,
        'num_averaging_runs': args.num_averaging_runs,
        'a_seed': args.a_seed,
    }
    
    print_or_wandb_log(args.wandb_switch, results, "ELLIPTICAL BONUS RESULTS")
    
    print(f"\n✓ Elliptical Bonus sweep completed!")
    print(f"📊 Final: L2={avg_l2_distance:.6f}±{std_l2_distance:.6f}")
    print(f"🔧 Config: φ_dim={args.phi_dim}, reg={args.regularization}, samples={args.num_samples}")
    
    if args.wandb_switch:
        wandb.finish()

if __name__ == "__main__":
    main()