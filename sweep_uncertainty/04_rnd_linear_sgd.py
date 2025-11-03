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
from utilities.uncertainty_methods import RNDLinearSGDMethod
from utilities.evaluation import (
    calculate_ground_truth_from_positions, evaluate_uncertainty_method, 
    normalize_uncertainty_matrix, compute_l2_distance, 
    compute_correlation, save_heatmap_to_wandb
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
    parser = argparse.ArgumentParser(description='RND Linear SGD Hyperparameter Sweep on PointMaze')
    
    # Method is fixed
    parser.add_argument('--method', type=str, default='rnd_linear',
                       help='Method (fixed to rnd_linear)')
    
    # Core RND Linear parameters
    parser.add_argument('--phi_dim', type=int, default=64,
                       help='Feature dimension for φ(s) mapping')
    parser.add_argument('--phi_seed', type=int, default=0,
                       help='Seed for generating φ(s) weights')
    parser.add_argument('--gaussian_noise', type=float, default=0.0,
                       help='Gaussian noise level for training targets')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    
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
        args.phi_dim = config.get('phi_dim', args.phi_dim)
        args.phi_seed = config.get('phi_seed', args.phi_seed)
        args.gaussian_noise = config.get('gaussian_noise', args.gaussian_noise)
        args.num_epochs = config.get('num_epochs', args.num_epochs)
        args.num_samples = config.get('num_samples', args.num_samples)
        args.num_averaging_runs = config.get('num_averaging_runs', args.num_averaging_runs)
        args.a_seed = config.get('a_seed', args.a_seed)
        args.wandb_switch = config.get('wandb_switch', args.wandb_switch)
        args.grid_rows = config.get('grid_rows', args.grid_rows)
        args.grid_cols = config.get('grid_cols', args.grid_cols)
        print("🔄 Running in WandB sweep mode - RND Linear SGD Hyperparameter Sweep")
    else:
        if args.wandb_switch:
            wandb.init(
                project="rnd_linear_sgd_pointmaze",
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
    
    # Store results for averaging
    all_l2_distances = []
    all_correlations = []
    all_final_losses = []
    all_convergence_epochs = []
    all_min_losses = []
    
    print(f"\nStarting {args.num_averaging_runs} averaging runs...")
    print(f"RND Linear SGD: φ_dim={args.phi_dim}, φ_seed={args.phi_seed}, noise={args.gaussian_noise}, epochs={args.num_epochs}")
    print("=" * 80)
    
    for avg_run in range(args.num_averaging_runs):
        print(f"\n🔄 RUN {avg_run + 1}/{args.num_averaging_runs}")
        
        # Set seed for this run
        current_seed = args.a_seed * 1000 + avg_run
        set_seed(current_seed)
        
        # Use same positions for all averaging runs for fair comparison
        current_positions = positions
        
        # Create shared φ(s) weights (same φ_seed across all averaging runs)
        phi_weights = create_phi_weights_deterministic(args.phi_dim, args.phi_seed)
        
        # Initialize RND Linear (SGD)
        method = RNDLinearSGDMethod(
            feature_dim=args.phi_dim,
            device=device,
            phi_weights=phi_weights
        )
        
        # Train the method using SGD
        print(f"Training RND Linear SGD (noise={args.gaussian_noise}, epochs={args.num_epochs})...")
        try:
            training_losses = method.train_on_positions(
                current_positions,
                num_epochs=args.num_epochs,
                subset_ratio=1.0,
                gaussian_noise=args.gaussian_noise
            )
            
            final_loss = training_losses[-1] if training_losses else float('nan')
            min_loss = min(training_losses) if training_losses else float('nan')
            
            # Find convergence epoch (when loss stops decreasing significantly)
            convergence_epoch = args.num_epochs
            if len(training_losses) > 10:
                for i in range(10, len(training_losses)):
                    if abs(training_losses[i] - training_losses[i-10]) < 1e-6:
                        convergence_epoch = i
                        break
            
            all_final_losses.append(final_loss)
            all_min_losses.append(min_loss)
            all_convergence_epochs.append(convergence_epoch)
            
        except Exception as e:
            print(f"Training failed: {e}")
            all_final_losses.append(float('nan'))
            all_min_losses.append(float('nan'))
            all_convergence_epochs.append(args.num_epochs)
            continue
        
        # Evaluate
        pred_uncertainty = evaluate_uncertainty_method(
            method, maze_map, args.grid_rows, args.grid_cols, device
        )
        pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
        
        # Compute metrics
        l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
        correlation = compute_correlation(gt_normalized, pred_normalized, maze_map)
        
        all_l2_distances.append(l2_distance)
        all_correlations.append(correlation)
        
        print(f"  L2={l2_distance:.6f}, Corr={correlation:.6f}, Final Loss={final_loss:.6f}, Min Loss={min_loss:.6f}")
        print(f"  Converged@epoch {convergence_epoch}")
        
        # Log individual run
        if args.wandb_switch:
            wandb.log({
                f'run_{avg_run}_l2_distance': l2_distance,
                f'run_{avg_run}_correlation': correlation,
                f'run_{avg_run}_final_loss': final_loss,
                f'run_{avg_run}_min_loss': min_loss,
                f'run_{avg_run}_convergence_epoch': convergence_epoch,
                'averaging_run': avg_run
            })
    
    # Calculate averaged results
    avg_l2_distance = np.nanmean(all_l2_distances)
    std_l2_distance = np.nanstd(all_l2_distances)
    avg_correlation = np.nanmean(all_correlations)
    std_correlation = np.nanstd(all_correlations)
    avg_final_loss = np.nanmean(all_final_losses)
    std_final_loss = np.nanstd(all_final_losses)
    avg_min_loss = np.nanmean(all_min_losses)
    avg_convergence_epoch = np.nanmean(all_convergence_epochs)
    
    print(f"\n📊 AVERAGED RESULTS ({args.num_averaging_runs} runs)")
    print("=" * 80)
    print(f"L2 Distance: {avg_l2_distance:.6f} ± {std_l2_distance:.6f}")
    print(f"Correlation: {avg_correlation:.6f} ± {std_correlation:.6f}")
    print(f"Final Loss:  {avg_final_loss:.6f} ± {std_final_loss:.6f}")
    print(f"Min Loss:    {avg_min_loss:.6f}")
    print(f"Avg Convergence Epoch: {avg_convergence_epoch:.1f}")
    
    # Save heatmaps
    save_heatmap_to_wandb(gt_normalized, "Ground Truth (1/√N)", args.wandb_switch, maze_map)
    save_heatmap_to_wandb(pred_normalized, f"RND-Linear-SGD-{args.phi_dim}dim", args.wandb_switch, maze_map)
    
    # Final results
    results = {
        'method': 'rnd_linear_sgd',
        'phi_dim': args.phi_dim,
        'phi_seed': args.phi_seed,
        'gaussian_noise': args.gaussian_noise,
        'num_epochs': args.num_epochs,
        'avg_l2_distance': avg_l2_distance,
        'std_l2_distance': std_l2_distance,
        'avg_correlation': avg_correlation,
        'std_correlation': std_correlation,
        'avg_final_loss': avg_final_loss,
        'std_final_loss': std_final_loss,
        'avg_min_loss': avg_min_loss,
        'avg_convergence_epoch': avg_convergence_epoch,
        'num_samples': args.num_samples,
        'num_averaging_runs': args.num_averaging_runs,
        'a_seed': args.a_seed,
    }
    
    print_or_wandb_log(args.wandb_switch, results, "RND LINEAR SGD HYPERPARAMETER RESULTS")
    
    print(f"\n✓ RND Linear SGD hyperparameter sweep completed!")
    print(f"📊 Final: L2={avg_l2_distance:.6f}±{std_l2_distance:.6f}")
    print(f"🔧 Config: φ_dim={args.phi_dim}, φ_seed={args.phi_seed}, noise={args.gaussian_noise}, epochs={args.num_epochs}")
    
    if args.wandb_switch:
        wandb.finish()

if __name__ == "__main__":
    main()