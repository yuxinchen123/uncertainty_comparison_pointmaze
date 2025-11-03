#!/usr/bin/env python3

import argparse
import random
import numpy as np
import torch
import wandb

from utilities.debug import print_or_wandb_log
from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset
from utilities.uncertainty_methods import RNDMethod
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

def main():
    parser = argparse.ArgumentParser(description='RND Method Sweep on PointMaze')
    
    # Method is fixed
    parser.add_argument('--method', type=str, default='rnd',
                       help='Method (fixed to rnd)')
    
    # Core RND parameters
    parser.add_argument('--output_dim', type=int, default=64,
                       help='Output dimension for RND networks')
    parser.add_argument('--hidden_dims', type=str, default='128,128',
                       help='Hidden dimensions for RND networks (comma-separated)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--gaussian_noise', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise to add to targets')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to extract from dataset')
    parser.add_argument('--num_averaging_runs', type=int, default=10,
                       help='Number of runs to average over')
    
    # Experiment parameters
    parser.add_argument('--a_seed', type=int, default=0,
                       help='Random seed for data sampling and model initialization')
    parser.add_argument('--wandb_switch', type=bool, default=True,
                       help='Whether to log to WandB')
    
    # Grid parameters
    parser.add_argument('--grid_rows', type=int, default=9)
    parser.add_argument('--grid_cols', type=int, default=12)
    
    args = parser.parse_args()
    
    # Handle WandB sweep
    if wandb.run is not None:
        config = wandb.config
        args.output_dim = config.get('output_dim', args.output_dim)
        args.hidden_dims = config.get('hidden_dims', args.hidden_dims)
        args.num_epochs = config.get('num_epochs', args.num_epochs)
        args.gaussian_noise = config.get('gaussian_noise', args.gaussian_noise)
        args.num_samples = config.get('num_samples', args.num_samples)
        args.num_averaging_runs = config.get('num_averaging_runs', args.num_averaging_runs)
        args.a_seed = config.get('a_seed', args.a_seed)
        args.wandb_switch = config.get('wandb_switch', args.wandb_switch)
        args.grid_rows = config.get('grid_rows', args.grid_rows)
        args.grid_cols = config.get('grid_cols', args.grid_cols)
        print("🔄 Running in WandB sweep mode - RND")
    else:
        if args.wandb_switch:
            wandb.init(
                project="rnd_pointmaze",
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
    
    # Sample positions once for all averaging runs
    np.random.seed(args.a_seed)
    if args.num_samples > total_positions:
        positions = all_positions
    else:
        subset_indices = np.random.choice(total_positions, args.num_samples, replace=False)
        positions = all_positions[subset_indices]
    
    print(f"Selected {len(positions)} positions")
    
    # Calculate ground truth once
    gt_uncertainty = calculate_ground_truth_from_positions(
        positions, maze_map, args.grid_rows, args.grid_cols
    )
    gt_normalized = normalize_uncertainty_matrix(gt_uncertainty)
    
    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # Store results for averaging
    all_l2_distances = []
    all_min_c_l1_norm_diff = []
    all_min_c_l2_norm_diff = []
    all_min_c_l1_norm_inv = []
    all_min_c_l2_norm_inv = []
    all_final_losses = []
    
    print(f"\nStarting {args.num_averaging_runs} averaging runs...")
    print(f"RND: output_dim={args.output_dim}, hidden_dims={args.hidden_dims}, epochs={args.num_epochs}, noise={args.gaussian_noise}")
    print("=" * 80)
    
    for avg_run in range(args.num_averaging_runs):
        print(f"\n🔄 RUN {avg_run + 1}/{args.num_averaging_runs}")
        
        # Set seed for this run
        current_seed = args.a_seed * 1000 + avg_run
        set_seed(current_seed)
        
        # Use same positions for all averaging runs for fair comparison
        current_positions = positions
        
        # Initialize RND method
        method = RNDMethod(
            hidden_dims=hidden_dims,
            output_dim=args.output_dim,
            device=device
        )
        
        # Train RND
        print(f"Training RND (noise={args.gaussian_noise}, epochs={args.num_epochs})...")
        training_losses = method.train_on_positions(
            current_positions,
            num_epochs=args.num_epochs,
            subset_ratio=1.0,
            gaussian_noise=args.gaussian_noise
        )
        
        final_loss = training_losses[-1] if training_losses else float('nan')
        
        # Evaluate
        pred_uncertainty = evaluate_uncertainty_method(
            method, maze_map, args.grid_rows, args.grid_cols, device
        )
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics
        l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        min_c_l1_norm_diff = compute_min_c_l1_norm_diff(gt_normalized, pred_normalized, maze_map)
        min_c_l2_norm_diff = compute_min_c_l2_norm_diff(gt_normalized, pred_normalized, maze_map)
        min_c_l1_norm_inv = compute_min_c_l1_norm_inv(gt_normalized, pred_normalized, maze_map)
        min_c_l2_norm_inv = compute_min_c_l2_norm_inv(gt_normalized, pred_normalized, maze_map)
        
        all_l2_distances.append(l2_distance)
        all_min_c_l1_norm_diff.append(min_c_l1_norm_diff)
        all_min_c_l2_norm_diff.append(min_c_l2_norm_diff)
        all_min_c_l1_norm_inv.append(min_c_l1_norm_inv)
        all_min_c_l2_norm_inv.append(min_c_l2_norm_inv)
        all_final_losses.append(final_loss)
        
        print(f"  L2={l2_distance:.6f}, L1_diff={min_c_l1_norm_diff:.6f}, L2_diff={min_c_l2_norm_diff:.6f}, L1_inv={min_c_l1_norm_inv:.6f}, L2_inv={min_c_l2_norm_inv:.6f}, Loss={final_loss:.6f}")
    
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
    avg_final_loss = np.nanmean(all_final_losses)
    std_final_loss = np.nanstd(all_final_losses)
    
    print(f"\n📊 AVERAGED RESULTS ({args.num_averaging_runs} runs)")
    print("=" * 80)
    print(f"L2 Distance:           {avg_l2_distance:.6f} ± {std_l2_distance:.6f}")
    print(f"min_c L1 diff:         {avg_min_c_l1_norm_diff:.6f} ± {std_min_c_l1_norm_diff:.6f}")
    print(f"min_c L2 diff:         {avg_min_c_l2_norm_diff:.6f} ± {std_min_c_l2_norm_diff:.6f}")
    print(f"min_c L1 inv:          {avg_min_c_l1_norm_inv:.6f} ± {std_min_c_l1_norm_inv:.6f}")
    print(f"min_c L2 inv:          {avg_min_c_l2_norm_inv:.6f} ± {std_min_c_l2_norm_inv:.6f}")
    print(f"Final Loss:            {avg_final_loss:.6f} ± {std_final_loss:.6f}")
    
    # Save heatmaps
    save_heatmap_to_wandb(gt_normalized, "Ground Truth (1/√N)", args.wandb_switch, maze_map)
    save_heatmap_to_wandb(pred_normalized, f"RND-{args.output_dim}dim-{args.hidden_dims}", args.wandb_switch, maze_map)
    
    # Final results
    results = {
        'method': 'rnd',
        'output_dim': args.output_dim,
        'hidden_dims': args.hidden_dims,
        'num_epochs': args.num_epochs,
        'gaussian_noise': args.gaussian_noise,
        'avg_l2_distance': avg_l2_distance,
        'std_l2_distance': std_l2_distance,
        'avg_min_c_l1_norm_diff': avg_min_c_l1_norm_diff,
        'std_min_c_l1_norm_diff': std_min_c_l1_norm_diff,
        'avg_min_c_l2_norm_diff': avg_min_c_l2_norm_diff,
        'std_min_c_l2_norm_diff': std_min_c_l2_norm_diff,
        'avg_min_c_l1_norm_inv': avg_min_c_l1_norm_inv,
        'std_min_c_l1_norm_inv': std_min_c_l1_norm_inv,
        'avg_min_c_l2_norm_inv': avg_min_c_l2_norm_inv,
        'std_min_c_l2_norm_inv': std_min_c_l2_norm_inv,
        'avg_final_loss': avg_final_loss,
        'std_final_loss': std_final_loss,
        'num_samples': args.num_samples,
        'num_averaging_runs': args.num_averaging_runs,
        'a_seed': args.a_seed,
    }
    
    print_or_wandb_log(args.wandb_switch, results, "RND RESULTS")
    
    print(f"\n✓ RND sweep completed!")
    print(f"📊 Final: L2={avg_l2_distance:.6f}±{std_l2_distance:.6f}")
    print(f"🔧 Config: output_dim={args.output_dim}, hidden_dims={args.hidden_dims}, epochs={args.num_epochs}, noise={args.gaussian_noise}")
    
    if args.wandb_switch:
        wandb.finish()

if __name__ == "__main__":
    main()

