#!/usr/bin/env python3
"""
RND Method Epoch Analysis
Tests RND method with different epoch counts and analyzes training loss vs L2 distance
"""

import argparse
import random
import numpy as np
import torch

from utilities.debug import print_or_wandb_log
from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset
from utilities.uncertainty_methods import RNDMethod
from utilities.evaluation import (
    calculate_ground_truth_from_positions, evaluate_uncertainty_method, 
    normalize_uncertainty_matrix, compute_l2_distance,
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
    parser = argparse.ArgumentParser(description='RND Method Epoch Analysis')
    
    # Core RND parameters
    parser.add_argument('--output_dim', type=int, default=128,
                       help='Output dimension for RND networks')
    parser.add_argument('--hidden_dims', type=str, default='128,128',
                       help='Hidden dimensions for RND networks (comma-separated)')
    parser.add_argument('--gaussian_noise', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise to add to targets')
    
    # Epoch values to test
    parser.add_argument('--epoch_values', type=str, default='10,20,30,50,100,200,500,1000',
                       help='Comma-separated list of epoch values to test')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to extract from dataset')
    parser.add_argument('--num_averaging_runs', type=int, default=10,
                       help='Number of runs to average over')
    
    # Experiment parameters
    parser.add_argument('--a_seed', type=int, default=42,
                       help='Random seed for data sampling and model initialization')
    
    # Grid parameters
    parser.add_argument('--grid_rows', type=int, default=9)
    parser.add_argument('--grid_cols', type=int, default=12)
    
    args = parser.parse_args()
    
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
    
    # Sample positions once for all runs
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
    
    # Parse hidden dimensions and epoch values
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    epoch_values = [int(x) for x in args.epoch_values.split(',')]
    
    print(f"\n{'='*100}")
    print(f"RND EPOCH ANALYSIS")
    print(f"{'='*100}")
    print(f"Architecture: hidden_dims={hidden_dims}, output_dim={args.output_dim}")
    print(f"Epoch values to test: {epoch_values}")
    print(f"Gaussian noise: {args.gaussian_noise}")
    print(f"Number of averaging runs: {args.num_averaging_runs}")
    print(f"{'='*100}\n")
    
    # Store results for each epoch value
    results = []
    
    for num_epochs in epoch_values:
        print(f"\n{'='*80}")
        print(f"Testing with {num_epochs} epochs")
        print(f"{'='*80}")
        
        # Store results for this epoch count
        epoch_l2_distances = []
        epoch_final_losses = []
        epoch_min_losses = []
        
        for avg_run in range(args.num_averaging_runs):
            print(f"  Run {avg_run + 1}/{args.num_averaging_runs}", end=' ... ')
            
            # Set seed for this run
            current_seed = args.a_seed * 1000 + avg_run
            set_seed(current_seed)
            
            # Initialize RND method
            method = RNDMethod(
                hidden_dims=hidden_dims,
                output_dim=args.output_dim,
                device=device
            )
            
            # Train RND
            training_losses = method.train_on_positions(
                positions,
                num_epochs=num_epochs,
                subset_ratio=1.0,
                gaussian_noise=args.gaussian_noise
            )
            
            final_loss = training_losses[-1] if training_losses else float('nan')
            min_loss = min(training_losses) if training_losses else float('nan')
            
            # Evaluate
            pred_uncertainty = evaluate_uncertainty_method(
                method, maze_map, args.grid_rows, args.grid_cols, device
            )
            pred_normalized = normalize_uncertainty_matrix(pred_uncertainty)
            
            # Compute L2 distance
            l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
            
            epoch_l2_distances.append(l2_distance)
            epoch_final_losses.append(final_loss)
            epoch_min_losses.append(min_loss)
            
            print(f"L2={l2_distance:.4f}, Loss={final_loss:.6f}")
        
        # Calculate statistics for this epoch count
        avg_l2 = np.nanmean(epoch_l2_distances)
        std_l2 = np.nanstd(epoch_l2_distances)
        min_l2 = np.nanmin(epoch_l2_distances)
        max_l2 = np.nanmax(epoch_l2_distances)
        
        avg_final_loss = np.nanmean(epoch_final_losses)
        std_final_loss = np.nanstd(epoch_final_losses)
        avg_min_loss = np.nanmean(epoch_min_losses)
        
        results.append({
            'epochs': num_epochs,
            'avg_l2': avg_l2,
            'std_l2': std_l2,
            'min_l2': min_l2,
            'max_l2': max_l2,
            'avg_final_loss': avg_final_loss,
            'std_final_loss': std_final_loss,
            'avg_min_loss': avg_min_loss,
        })
        
        print(f"\n  Summary: L2={avg_l2:.4f}±{std_l2:.4f} (range: {min_l2:.4f}-{max_l2:.4f}), "
              f"Final Loss={avg_final_loss:.6f}±{std_final_loss:.6f}, Min Loss={avg_min_loss:.6f}")
    
    # Print comprehensive results table
    print(f"\n\n{'='*100}")
    print("COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*100}")
    print(f"\n{'Epochs':<10} {'Avg L2':<15} {'Std L2':<15} {'Min L2':<15} {'Max L2':<15} {'Avg Final Loss':<20} {'Avg Min Loss':<20}")
    print("-" * 100)
    for r in results:
        print(f"{r['epochs']:<10} "
              f"{r['avg_l2']:<15.6f} "
              f"{r['std_l2']:<15.6f} "
              f"{r['min_l2']:<15.6f} "
              f"{r['max_l2']:<15.6f} "
              f"{r['avg_final_loss']:<20.6f} "
              f"{r['avg_min_loss']:<20.6f}")
    
    # Find best epoch (lowest L2)
    best_idx = min(range(len(results)), key=lambda i: results[i]['avg_l2'])
    best_epoch = results[best_idx]['epochs']
    best_l2 = results[best_idx]['avg_l2']
    
    print(f"\n{'='*100}")
    print("ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*100}")
    print(f"\nBest L2 Distance: {best_l2:.6f} at {best_epoch} epochs")
    
    # Analyze convergence
    print(f"\nL2 Distance Progression:")
    for i, r in enumerate(results):
        improvement = ""
        if i > 0:
            prev_r = results[i-1]
            diff = prev_r['avg_l2'] - r['avg_l2']
            pct = (diff / prev_r['avg_l2']) * 100 if prev_r['avg_l2'] > 0 else 0
            if diff > 0:
                improvement = f" (↓{diff:.4f}, {pct:.1f}% improvement)"
            elif diff < 0:
                improvement = f" (↑{abs(diff):.4f}, {abs(pct):.1f}% worse)"
            else:
                improvement = " (no change)"
        print(f"  {r['epochs']:4d} epochs: L2={r['avg_l2']:.6f}±{r['std_l2']:.6f}{improvement}")
    
    # Loss convergence analysis
    print(f"\nTraining Loss Progression:")
    for r in results:
        print(f"  {r['epochs']:4d} epochs: Final={r['avg_final_loss']:.6f}±{r['std_final_loss']:.6f}, "
              f"Min={r['avg_min_loss']:.6f}")
    
    # Recommendations
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS FOR SWEEP YAML")
    print(f"{'='*100}")
    
    # Find epochs where L2 plateaus (improvement < 1%)
    print("\nEpoch values where L2 distance plateaus (<1% improvement):")
    plateau_epochs = []
    for i in range(1, len(results)):
        prev_l2 = results[i-1]['avg_l2']
        curr_l2 = results[i]['avg_l2']
        improvement = ((prev_l2 - curr_l2) / prev_l2) * 100 if prev_l2 > 0 else 0
        if improvement < 1.0 and improvement > -1.0:  # Less than 1% improvement or worse
            plateau_epochs.append(results[i]['epochs'])
            print(f"  {results[i]['epochs']} epochs: {improvement:.2f}% change from {results[i-1]['epochs']} epochs")
    
    # Suggest optimal range
    if plateau_epochs:
        optimal_start = min(plateau_epochs)
        optimal_end = max(plateau_epochs)
        print(f"\nSuggested epoch range: [{optimal_start}, {optimal_end}]")
        print(f"  (L2 distance plateaus in this range)")
    else:
        # Use best and nearby values
        if best_idx > 0:
            prev_epoch = results[best_idx-1]['epochs']
        else:
            prev_epoch = best_epoch
        if best_idx < len(results) - 1:
            next_epoch = results[best_idx+1]['epochs']
        else:
            next_epoch = best_epoch
        
        suggested = sorted(set([prev_epoch, best_epoch, next_epoch]))
        print(f"\nSuggested epoch values: {suggested}")
        print(f"  (Best: {best_epoch} epochs with L2={best_l2:.6f})")
    
    # Save results to text file
    output_file = 'rnd_epoch_analysis_results.txt'
    with open(output_file, 'w') as f:
        f.write("RND Epoch Analysis Results\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Epochs':<10} {'Avg L2':<15} {'Std L2':<15} {'Min L2':<15} {'Max L2':<15} {'Avg Final Loss':<20} {'Avg Min Loss':<20}\n")
        f.write("-"*100 + "\n")
        for r in results:
            f.write(f"{r['epochs']:<10} {r['avg_l2']:<15.6f} {r['std_l2']:<15.6f} {r['min_l2']:<15.6f} "
                   f"{r['max_l2']:<15.6f} {r['avg_final_loss']:<20.6f} {r['avg_min_loss']:<20.6f}\n")
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    results_df = main()

