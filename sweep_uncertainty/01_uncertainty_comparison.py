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
from utilities.uncertainty_methods import RNDMethod, RNDLinearMethod, EllipticalBonusMethod
from utilities.evaluation import (
    calculate_ground_truth, calculate_ground_truth_from_positions, evaluate_uncertainty_method, 
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
    """Create deterministic φ(s) weights for sharing between methods"""
    # Set temporary seed for weight generation
    torch.manual_seed(seed)
    
    # Create temporary linear layer to generate weights
    temp_layer = torch.nn.Linear(2, feature_dim)
    phi_weights = temp_layer.state_dict()  # This returns proper state dict
    
    return phi_weights

def main():
    parser = argparse.ArgumentParser(description='Uncertainty Method Comparison on PointMaze')
    
    # Method selection
    parser.add_argument('--method', type=str, required=False, default='rnd',
                       choices=['rnd', 'rnd_linear', 'elliptical'],
                       help='Uncertainty method to use (required if not using WandB sweep)')
    
    # Architecture parameters
    parser.add_argument('--output_dim', type=int, default=64,
                       help='Output dimension for RND networks or feature dimension for linear methods')
    parser.add_argument('--hidden_dims', type=str, default='128,128',
                       help='Hidden dimensions for RND networks (comma-separated)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--gaussian_noise', type=float, default=0.0,
                        help='Standard deviation of Gaussian noise to add to targets')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to extract from dataset')
    parser.add_argument('--run_index', type=int, default=0,
                       help='Run index for different data subsets (0-9)')
    parser.add_argument('--num_averaging_runs', type=int, default=10,
                       help='Number of runs to average over for each sample subset')
    
    # φ(s) sharing parameters
    parser.add_argument('--phi_seed', type=int, default=42,
                       help='Seed for generating shared φ(s) weights')
    parser.add_argument('--phi_dim', type=int, default=64,
                       help='Dimension of shared φ(s) features (for RND_Linear and Elliptical)')
    
    # Experiment parameters
    parser.add_argument('--a_seed', type=int, default=0,
                       help='Random seed for data sampling and model initialization')
    parser.add_argument('--wandb_switch', type=bool, default=True,
                       help='Whether to log to WandB')
    
    # Grid parameters
    parser.add_argument('--grid_rows', type=int, default=9,
                       help='Number of grid rows')
    parser.add_argument('--grid_cols', type=int, default=12,
                       help='Number of grid columns')
    
    args = parser.parse_args()
    
    # Check if we're running in a WandB sweep
    if wandb.run is not None:
        # We're in a WandB sweep, use config values and override args
        config = wandb.config
        
        # Override arguments with WandB config
        args.method = config.get('method', args.method)
        args.output_dim = config.get('output_dim', args.output_dim)
        args.hidden_dims = config.get('hidden_dims', args.hidden_dims)
        args.num_epochs = config.get('num_epochs', args.num_epochs)
        args.gaussian_noise = config.get('gaussian_noise', args.gaussian_noise)
        args.num_samples = config.get('num_samples', args.num_samples)
        args.num_averaging_runs = config.get('num_averaging_runs', args.num_averaging_runs)
        args.phi_seed = config.get('phi_seed', args.phi_seed)
        args.phi_dim = config.get('phi_dim', args.phi_dim)
        args.a_seed = config.get('a_seed', args.a_seed)
        args.wandb_switch = config.get('wandb_switch', args.wandb_switch)
        args.grid_rows = config.get('grid_rows', args.grid_rows)
        args.grid_cols = config.get('grid_cols', args.grid_cols)
        
        print("🔄 Running in WandB sweep mode")
    else:
        # Initialize WandB if not already initialized
        if args.wandb_switch:
            wandb.init(
                project="uncertainty_comparison_pointmaze",
                config=vars(args)
            )
        
        # Validate that method is provided when not in sweep mode
        if not args.method:
            parser.error("--method is required when not running in WandB sweep mode")
    
    # Set seed for reproducibility
    set_seed(args.a_seed)
    
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
    
    # Single sampling step - use same positions for both ground truth and training
    print(f"Creating random subset with {args.num_samples} samples...")
    np.random.seed(args.a_seed)  # Use a_seed for data sampling
    
    if args.num_samples > total_positions:
        print(f"Warning: Requested {args.num_samples} samples but only {total_positions} available")
        positions = all_positions
    else:
        # Randomly sample indices (deterministic for this run_index) 
        subset_indices = np.random.choice(total_positions, args.num_samples, replace=False)
        subset_indices = np.sort(subset_indices)  # Sort for consistent iteration
        positions = all_positions[subset_indices]
    
    print(f"✓ Selected {len(positions)} positions using seed {args.a_seed}")
    print(f"Position range: X=[{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}], Y=[{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}]")
    print(f"Will average over {args.num_averaging_runs} training runs with these same {args.num_samples} samples")
    
    # Calculate ground truth uncertainty FROM THE SAME SUBSET
    print("Calculating ground truth uncertainty from the same data subset...")
    
    # Calculate ground truth from subset positions directly
    gt_uncertainty = calculate_ground_truth_from_positions(
        positions, maze_map, args.grid_rows, args.grid_cols
    )
    gt_normalized = normalize_uncertainty_matrix(gt_uncertainty)
    
    # Parse hidden dimensions for RND
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # Store results from multiple runs for averaging
    all_l2_distances = []
    all_min_c_l1_norm_diff = []
    all_min_c_l2_norm_diff = []
    all_min_c_l1_norm_inv = []
    all_min_c_l2_norm_inv = []
    all_training_losses = []
    
    print(f"\nStarting {args.num_averaging_runs} averaging runs...")
    print("=" * 80)
    
    for avg_run in range(args.num_averaging_runs):
        print(f"\n🔄 AVERAGING RUN {avg_run + 1}/{args.num_averaging_runs}")
        print("-" * 60)
        
        # Set different seed for each averaging run
        current_seed = args.a_seed * 1000 + avg_run
        set_seed(current_seed)
        
        # Initialize method based on type
        # Set method-specific epochs: RND: [10, 30, 50], RND Linear (LS): N/A, Elliptical: N/A
        if args.method == 'rnd':
            # RND epoch sweep: [10, 30, 50] - use from config if in sweep, otherwise args
            rnd_epochs = args.num_epochs
            if wandb.run is not None:
                rnd_epochs = config.get('num_epochs', args.num_epochs)
            print(f"Initializing RND method (hidden_dims={hidden_dims}, output_dim={args.output_dim}, epochs={rnd_epochs})...")
            method = RNDMethod(
                hidden_dims=hidden_dims,
                output_dim=args.output_dim,
                device=device
            )
            
            # Train RND with full 10K dataset
            print("Training RND...")
            training_losses = method.train_on_positions(
                positions, 
                num_epochs=rnd_epochs,
                subset_ratio=1.0,  # Use full dataset for fairness
                gaussian_noise=args.gaussian_noise
            )
            
            method_info = f"RND-{args.output_dim}dim-{args.hidden_dims}"
            
        elif args.method == 'rnd_linear':
            print(f"Initializing RND-Linear method (feature_dim={args.phi_dim})...")
            
            # Create shared φ(s) weights (deterministic across averaging runs)
            phi_weights = create_phi_weights_deterministic(args.phi_dim, args.phi_seed)
            
            method = RNDLinearMethod(
                feature_dim=args.phi_dim,
                device=device,
                phi_weights=phi_weights
            )
            
            # Train RND-Linear with full 10K dataset
            print("Training RND-Linear...")
            training_losses = method.train_on_positions(
                positions,
                num_epochs=args.num_epochs,
                subset_ratio=1.0,  # Use full dataset for fairness
                gaussian_noise=args.gaussian_noise
            )
            
            method_info = f"RND-Linear-{args.phi_dim}dim"
            
        elif args.method == 'elliptical':
            print(f"Initializing Elliptical Bonus method (feature_dim={args.phi_dim})...")
            
            # Note: Elliptical method doesn't use gaussian_noise (parameter ignored if provided)
            if args.gaussian_noise > 0:
                print(f"  Note: Gaussian noise ({args.gaussian_noise}) ignored for Elliptical Bonus method")
            
            # Create shared φ(s) weights (deterministic across averaging runs)
            phi_weights = create_phi_weights_deterministic(args.phi_dim, args.phi_seed)
            
            method = EllipticalBonusMethod(
                feature_dim=args.phi_dim,
                device=device,
                phi_weights=phi_weights
            )
            
            # Update covariance from full 10K dataset
            print("Updating covariance matrix...")
            method.update_covariance_from_positions(positions)
            
            training_losses = None  # No training for elliptical method
            method_info = f"Elliptical-{args.phi_dim}dim"
        
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # Evaluate uncertainty method
        print("Evaluating uncertainty method...")
        pred_uncertainty = evaluate_uncertainty_method(
            method, maze_map, args.grid_rows, args.grid_cols, device
        )
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics for this run
        l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        min_c_l1_norm_diff = compute_min_c_l1_norm_diff(gt_normalized, pred_normalized, maze_map)
        min_c_l2_norm_diff = compute_min_c_l2_norm_diff(gt_normalized, pred_normalized, maze_map)
        min_c_l1_norm_inv = compute_min_c_l1_norm_inv(gt_normalized, pred_normalized, maze_map)
        min_c_l2_norm_inv = compute_min_c_l2_norm_inv(gt_normalized, pred_normalized, maze_map)
        
        # Store results
        all_l2_distances.append(l2_distance)
        all_min_c_l1_norm_diff.append(min_c_l1_norm_diff)
        all_min_c_l2_norm_diff.append(min_c_l2_norm_diff)
        all_min_c_l1_norm_inv.append(min_c_l1_norm_inv)
        all_min_c_l2_norm_inv.append(min_c_l2_norm_inv)
        if training_losses:
            all_training_losses.append(training_losses[-1])
        
        print(f"  Run {avg_run + 1} results: L2={l2_distance:.6f}, L1_diff={min_c_l1_norm_diff:.6f}, L2_diff={min_c_l2_norm_diff:.6f}, L1_inv={min_c_l1_norm_inv:.6f}, L2_inv={min_c_l2_norm_inv:.6f}")
    
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
    avg_final_loss = np.nanmean(all_training_losses) if all_training_losses else None
    std_final_loss = np.nanstd(all_training_losses) if all_training_losses else None
    
    print(f"\n📊 AVERAGED RESULTS ({args.num_averaging_runs} runs)")
    print("=" * 80)
    print(f"L2 Distance:           {avg_l2_distance:.6f} ± {std_l2_distance:.6f}")
    print(f"min_c L1 diff:         {avg_min_c_l1_norm_diff:.6f} ± {std_min_c_l1_norm_diff:.6f}")
    print(f"min_c L2 diff:         {avg_min_c_l2_norm_diff:.6f} ± {std_min_c_l2_norm_diff:.6f}")
    print(f"min_c L1 inv:          {avg_min_c_l1_norm_inv:.6f} ± {std_min_c_l1_norm_inv:.6f}")
    print(f"min_c L2 inv:          {avg_min_c_l2_norm_inv:.6f} ± {std_min_c_l2_norm_inv:.6f}")
    if avg_final_loss:
        print(f"Final Loss:            {avg_final_loss:.6f} ± {std_final_loss:.6f}")
    
    # Create final heatmap using the last method instance
    print("Creating final heatmaps...")
    save_heatmap_to_wandb(gt_normalized, "Ground Truth (1/√N)", args.wandb_switch, maze_map)
    save_heatmap_to_wandb(pred_normalized, f"{method_info} Uncertainty (Final)", args.wandb_switch, maze_map)
    
    # Log results
    phi_info = "N/A (full neural network)" if args.method == 'rnd' else f"φ(s) seed={args.phi_seed}, dim={args.phi_dim}"
    
    results = {
        'method': args.method,
        'method_info': method_info,
        'phi_info': phi_info,
        'avg_l2_distance': avg_l2_distance,
        'avg_min_c_l1_norm_diff': avg_min_c_l1_norm_diff,
        'avg_min_c_l2_norm_diff': avg_min_c_l2_norm_diff,
        'avg_min_c_l1_norm_inv': avg_min_c_l1_norm_inv,
        'avg_min_c_l2_norm_inv': avg_min_c_l2_norm_inv,
        'avg_final_loss': avg_final_loss,
        'output_dim': args.output_dim,
        'hidden_dims': args.hidden_dims,
        'phi_dim': args.phi_dim if args.method in ['rnd_linear', 'elliptical'] else None,
        'num_epochs': args.num_epochs,
        'gaussian_noise': args.gaussian_noise,
        'num_samples': args.num_samples,
        'run_index': args.run_index,
        'num_averaging_runs': args.num_averaging_runs,
        'phi_seed': args.phi_seed,
        'a_seed': args.a_seed,
        'max_gt_uncertainty': np.nanmax(gt_uncertainty),
        'max_pred_uncertainty': np.nanmax(pred_uncertainty),
    }
    
    print_or_wandb_log(args.wandb_switch, results, "FINAL AVERAGED RESULTS")
    
    print("Experiment completed!")
    print(f"📊 Results: L2={avg_l2_distance:.6f}±{std_l2_distance:.6f}, min_c L1 diff={avg_min_c_l1_norm_diff:.6f}±{std_min_c_l1_norm_diff:.6f}")
    print(f"🔧 Method: {method_info}")
    print(f"🎯 φ(s) Info: {phi_info}")
    print(f"📈 Averaged over {args.num_averaging_runs} runs with same {args.num_samples} samples")
    
    if args.wandb_switch:
        wandb.finish()

if __name__ == "__main__":
    main()
