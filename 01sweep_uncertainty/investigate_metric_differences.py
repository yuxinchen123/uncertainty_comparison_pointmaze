#!/usr/bin/env python3
"""
Investigation script to analyze where L2 Diff and L2 Inv metrics start to differ.
Uses 01 RND as an example case.
"""

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

# Add utilities to path
sys.path.insert(0, os.path.dirname(__file__))

from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset, get_maze_map

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
from utilities.uncertainty_methods import RNDMethod
from utilities.evaluation import (
    calculate_ground_truth_from_positions, evaluate_uncertainty_method,
    compute_min_c_l2_norm_diff, compute_min_c_l2_norm_inv
)

def analyze_metric_differences(gt_matrix, pred_matrix, maze_map=None):
    """
    Analyze step-by-step where L2 Diff and L2 Inv metrics differ.
    Returns detailed analysis dictionary.
    """
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Extract open cells
    open_mask = np.array([[maze_map[i][j] == 0 for j in range(12)] for i in range(9)])
    gt_open = gt_matrix[open_mask]
    pred_open = pred_matrix[open_mask]
    
    # Filter finite values
    finite_mask = np.isfinite(gt_open) & np.isfinite(pred_open)
    gt_clean = gt_open[finite_mask]
    pred_clean = pred_open[finite_mask]
    
    # For Inv metrics, also filter GT != 0
    inv_finite_mask = finite_mask & (gt_open != 0)
    gt_inv_clean = gt_open[inv_finite_mask]
    pred_inv_clean = pred_open[inv_finite_mask]
    
    analysis = {}
    
    # ===== L2 Diff Analysis =====
    print("\n" + "="*80)
    print("L2 DIFF METRIC ANALYSIS")
    print("="*80)
    
    # Compute optimal c for Diff
    pred_norm_sq = np.dot(pred_clean, pred_clean)
    if pred_norm_sq == 0:
        c_diff = 0.0
    else:
        c_diff = np.dot(pred_clean, gt_clean) / pred_norm_sq
    
    print(f"Optimal scaling constant c* (Diff): {c_diff:.6f}")
    print(f"  Formula: <Pred, GT> / ||Pred||²")
    print(f"  <Pred, GT> = {np.dot(pred_clean, gt_clean):.6f}")
    print(f"  ||Pred||² = {pred_norm_sq:.6f}")
    
    # Compute scaled predictions
    pred_scaled = c_diff * pred_clean
    
    # Compute errors
    errors_diff = gt_clean - pred_scaled
    squared_errors_diff = errors_diff ** 2
    l2_diff_value = np.sqrt(np.sum(squared_errors_diff))
    
    print(f"\nL2 Diff metric value: {l2_diff_value:.6f}")
    print(f"  Sum of squared errors: {np.sum(squared_errors_diff):.6f}")
    print(f"  Mean squared error: {np.mean(squared_errors_diff):.6f}")
    print(f"  Max squared error: {np.max(squared_errors_diff):.6f}")
    print(f"  Min squared error: {np.min(squared_errors_diff):.6f}")
    
    analysis['l2_diff'] = {
        'c_opt': c_diff,
        'metric_value': l2_diff_value,
        'errors': errors_diff,
        'squared_errors': squared_errors_diff,
        'gt': gt_clean,
        'pred': pred_clean,
        'pred_scaled': pred_scaled,
        'n_cells': len(gt_clean)
    }
    
    # ===== L2 Inv Analysis =====
    print("\n" + "="*80)
    print("L2 INV METRIC ANALYSIS")
    print("="*80)
    
    # Compute ratios
    ratio = pred_inv_clean / gt_inv_clean
    
    # Compute optimal c for Inv (mean of ratios)
    c_inv = np.mean(ratio)
    
    print(f"Optimal scaling constant c* (Inv): {c_inv:.6f}")
    print(f"  Formula: mean(Pred / GT)")
    print(f"  Number of cells (GT != 0): {len(gt_inv_clean)}")
    print(f"  Ratio statistics:")
    print(f"    Mean: {np.mean(ratio):.6f}")
    print(f"    Median: {np.median(ratio):.6f}")
    print(f"    Std: {np.std(ratio):.6f}")
    print(f"    Min: {np.min(ratio):.6f}")
    print(f"    Max: {np.max(ratio):.6f}")
    
    # Compute errors (deviation from constant ratio)
    errors_inv = c_inv - ratio
    squared_errors_inv = errors_inv ** 2
    l2_inv_value = np.sqrt(np.sum(squared_errors_inv))
    
    print(f"\nL2 Inv metric value: {l2_inv_value:.6f}")
    print(f"  Sum of squared errors: {np.sum(squared_errors_inv):.6f}")
    print(f"  Mean squared error: {np.mean(squared_errors_inv):.6f}")
    print(f"  Max squared error: {np.max(squared_errors_inv):.6f}")
    print(f"  Min squared error: {np.min(squared_errors_inv):.6f}")
    
    analysis['l2_inv'] = {
        'c_opt': c_inv,
        'metric_value': l2_inv_value,
        'errors': errors_inv,
        'squared_errors': squared_errors_inv,
        'ratio': ratio,
        'gt': gt_inv_clean,
        'pred': pred_inv_clean,
        'n_cells': len(gt_inv_clean)
    }
    
    # ===== Comparison Analysis =====
    print("\n" + "="*80)
    print("COMPARISON: WHERE DO THEY DIFFER?")
    print("="*80)
    
    print(f"\nOptimal scaling constants:")
    print(f"  L2 Diff c*: {c_diff:.6f}")
    print(f"  L2 Inv c*:  {c_inv:.6f}")
    print(f"  Difference: {abs(c_diff - c_inv):.6f}")
    print(f"  Ratio: {c_inv / c_diff if c_diff != 0 else 'N/A':.6f}")
    
    # Find cells that contribute most to differences
    # For cells in common (GT != 0), compare contributions
    
    # Map back to original indices
    common_mask = inv_finite_mask  # Cells used in both metrics
    common_gt = gt_inv_clean
    common_pred = pred_inv_clean
    
    # Compute Diff contribution for common cells
    common_pred_scaled_diff = c_diff * common_pred
    common_errors_diff = common_gt - common_pred_scaled_diff
    common_squared_errors_diff = common_errors_diff ** 2
    
    # Compute Inv contribution
    common_ratio = common_pred / common_gt
    common_errors_inv = c_inv - common_ratio
    common_squared_errors_inv = common_errors_inv ** 2
    
    # Find cells with largest difference in contributions
    diff_contributions = common_squared_errors_diff
    inv_contributions = common_squared_errors_inv
    
    # Normalize contributions to see relative importance
    diff_contrib_norm = diff_contributions / np.sum(diff_contributions) if np.sum(diff_contributions) > 0 else diff_contributions
    inv_contrib_norm = inv_contributions / np.sum(inv_contributions) if np.sum(inv_contributions) > 0 else inv_contributions
    
    # Find top contributors
    top_n = min(10, len(common_gt))
    top_diff_indices = np.argsort(diff_contributions)[-top_n:][::-1]
    top_inv_indices = np.argsort(inv_contributions)[-top_n:][::-1]
    
    print(f"\nTop {top_n} cells contributing to L2 Diff:")
    print(f"{'Rank':<6} {'GT':<12} {'Pred':<12} {'Pred*':<12} {'Error²':<12} {'% Contrib':<10}")
    for i, idx in enumerate(top_diff_indices):
        contrib_pct = diff_contrib_norm[idx] * 100
        print(f"{i+1:<6} {common_gt[idx]:<12.6f} {common_pred[idx]:<12.6f} "
              f"{common_pred_scaled_diff[idx]:<12.6f} {diff_contributions[idx]:<12.6f} {contrib_pct:<10.2f}%")
    
    print(f"\nTop {top_n} cells contributing to L2 Inv:")
    print(f"{'Rank':<6} {'GT':<12} {'Pred':<12} {'Ratio':<12} {'Error²':<12} {'% Contrib':<10}")
    for i, idx in enumerate(top_inv_indices):
        contrib_pct = inv_contrib_norm[idx] * 100
        print(f"{i+1:<6} {common_gt[idx]:<12.6f} {common_pred[idx]:<12.6f} "
              f"{common_ratio[idx]:<12.6f} {inv_contributions[idx]:<12.6f} {contrib_pct:<10.2f}%")
    
    # Find cells where metrics disagree most
    contribution_diff = np.abs(diff_contrib_norm - inv_contrib_norm)
    top_disagreement_indices = np.argsort(contribution_diff)[-top_n:][::-1]
    
    print(f"\nTop {top_n} cells where metrics disagree most (different relative contributions):")
    print(f"{'Rank':<6} {'GT':<12} {'Pred':<12} {'Diff %':<10} {'Inv %':<10} {'Diff':<10}")
    for i, idx in enumerate(top_disagreement_indices):
        diff_pct = diff_contrib_norm[idx] * 100
        inv_pct = inv_contrib_norm[idx] * 100
        disagreement = contribution_diff[idx] * 100
        print(f"{i+1:<6} {common_gt[idx]:<12.6f} {common_pred[idx]:<12.6f} "
              f"{diff_pct:<10.2f}% {inv_pct:<10.2f}% {disagreement:<10.2f}%")
    
    analysis['comparison'] = {
        'c_diff': c_diff,
        'c_inv': c_inv,
        'common_gt': common_gt,
        'common_pred': common_pred,
        'common_ratio': common_ratio,
        'diff_contributions': diff_contributions,
        'inv_contributions': inv_contributions,
        'top_diff_indices': top_diff_indices,
        'top_inv_indices': top_inv_indices,
        'top_disagreement_indices': top_disagreement_indices
    }
    
    return analysis

def visualize_analysis(analysis, output_file=None):
    """Create visualization plots for the analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('L2 Diff vs L2 Inv Metric Analysis', fontsize=16, fontweight='bold')
    
    l2_diff = analysis['l2_diff']
    l2_inv = analysis['l2_inv']
    comparison = analysis['comparison']
    
    # Plot 1: Error distributions
    ax1 = axes[0, 0]
    ax1.hist(l2_diff['errors'], bins=50, alpha=0.7, label='L2 Diff errors', color='blue')
    ax1.hist(l2_inv['errors'], bins=50, alpha=0.7, label='L2 Inv errors', color='red')
    ax1.set_xlabel('Error Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ratio distribution
    ax2 = axes[0, 1]
    ax2.hist(comparison['common_ratio'], bins=50, alpha=0.7, color='green')
    ax2.axvline(comparison['c_inv'], color='red', linestyle='--', linewidth=2, 
                label=f'Optimal c* = {comparison["c_inv"]:.4f}')
    ax2.set_xlabel('Pred/GT Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Ratio Distribution (Pred/GT)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GT vs Pred scatter
    ax3 = axes[1, 0]
    ax3.scatter(comparison['common_gt'], comparison['common_pred'], 
                alpha=0.5, s=10, color='blue', label='Data points')
    
    # Plot optimal scaling lines
    x_range = np.linspace(comparison['common_gt'].min(), comparison['common_gt'].max(), 100)
    ax3.plot(x_range, comparison['c_diff'] * x_range, 'r--', linewidth=2, 
             label=f'Diff: Pred = {comparison["c_diff"]:.4f} * GT')
    ax3.plot(x_range, x_range / comparison['c_inv'] if comparison['c_inv'] != 0 else x_range, 
             'g--', linewidth=2, label=f'Inv: Pred = {comparison["c_inv"]:.4f} * GT')
    
    ax3.set_xlabel('Ground Truth')
    ax3.set_ylabel('Prediction')
    ax3.set_title('GT vs Pred with Optimal Scaling Lines')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Contribution comparison
    ax4 = axes[1, 1]
    top_n = min(20, len(comparison['diff_contributions']))
    top_indices = np.argsort(comparison['diff_contributions'])[-top_n:][::-1]
    
    diff_contrib_norm = comparison['diff_contributions'] / np.sum(comparison['diff_contributions'])
    inv_contrib_norm = comparison['inv_contributions'] / np.sum(comparison['inv_contributions'])
    
    x_pos = np.arange(top_n)
    width = 0.35
    ax4.bar(x_pos - width/2, diff_contrib_norm[top_indices] * 100, width, 
            label='L2 Diff contribution', color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, inv_contrib_norm[top_indices] * 100, width,
            label='L2 Inv contribution', color='red', alpha=0.7)
    ax4.set_xlabel('Cell Rank (by Diff contribution)')
    ax4.set_ylabel('Contribution (%)')
    ax4.set_title(f'Top {top_n} Cells: Relative Contributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Investigate where L2 Diff and L2 Inv metrics differ')
    parser.add_argument('--output_dim', type=int, default=128, help='RND output dimension')
    parser.add_argument('--hidden_dims', type=str, default='64,64', help='RND hidden dimensions')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--gaussian_noise', type=float, default=0.0, help='Gaussian noise level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_file', type=str, default=None, help='Output file for visualization')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading PointMaze dataset...")
    dataset, maze_map = load_pointmaze_dataset()
    
    # Extract all available positions
    print("Extracting all available positions...")
    all_positions = extract_positions_from_dataset(dataset, num_samples=None)
    total_positions = len(all_positions)
    print(f"Total available positions in dataset: {total_positions}")
    
    # Sample positions
    print(f"Creating random subset with {args.num_samples} samples...")
    np.random.seed(args.seed)
    if args.num_samples > total_positions:
        print(f"Warning: Requested {args.num_samples} samples but only {total_positions} available")
        positions = all_positions
    else:
        indices = np.random.choice(total_positions, args.num_samples, replace=False)
        positions = all_positions[indices]
    
    print(f"✓ Selected {len(positions)} positions using seed {args.seed}")
    print(f"Position range: X=[{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}], Y=[{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}]")
    
    # Calculate ground truth from the same subset
    print("\nCalculating ground truth from the same data subset...")
    gt_uncertainty = calculate_ground_truth_from_positions(
        positions, maze_map, grid_rows=9, grid_cols=12
    )
    
    # Train RND
    print("\nTraining RND method...")
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    method = RNDMethod(
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        device=device
    )
    
    training_losses = method.train_on_positions(
        positions,
        num_epochs=args.num_epochs,
        subset_ratio=1.0,
        gaussian_noise=args.gaussian_noise
    )
    
    print(f"Final training loss: {training_losses[-1]:.6f}")
    
    # Evaluate uncertainty
    print("\nEvaluating uncertainty method...")
    pred_uncertainty = evaluate_uncertainty_method(
        method, maze_map, grid_rows=9, grid_cols=12, device=device
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    l2_diff_value = compute_min_c_l2_norm_diff(gt_uncertainty, pred_uncertainty, maze_map)
    l2_inv_value = compute_min_c_l2_norm_inv(gt_uncertainty, pred_uncertainty, maze_map)
    
    print(f"\n{'='*80}")
    print("METRIC VALUES")
    print(f"{'='*80}")
    print(f"L2 Diff: {l2_diff_value:.6f}")
    print(f"L2 Inv:  {l2_inv_value:.6f}")
    print(f"Ratio:   {l2_inv_value / l2_diff_value if l2_diff_value > 0 else 'N/A':.6f}")
    
    # Detailed analysis
    analysis = analyze_metric_differences(gt_uncertainty, pred_uncertainty, maze_map)
    
    # Visualization
    if args.output_file is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        args.output_file = os.path.join(results_dir, 'metric_differences_analysis.png')
    
    visualize_analysis(analysis, args.output_file)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

