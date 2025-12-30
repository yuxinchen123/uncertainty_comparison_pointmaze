#!/usr/bin/env python3
"""
Compare two hyperparameter configurations and analyze when L2 Diff and L2 Inv metrics
produce different rankings. Uses different sample sizes (100 vs 10k) as an example.
"""

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

# Add utilities to path
sys.path.insert(0, os.path.dirname(__file__))

from utilities.environment import load_pointmaze_dataset, extract_positions_from_dataset, get_maze_map
from utilities.uncertainty_methods import RNDMethod
from utilities.evaluation import (
    calculate_ground_truth_from_positions, evaluate_uncertainty_method,
    compute_min_c_l2_norm_diff, compute_min_c_l2_norm_inv
)

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_and_evaluate(config_name, positions, gt_uncertainty, maze_map, 
                       output_dim, hidden_dims, num_epochs, gaussian_noise, 
                       device, seed):
    """Train RND and evaluate both metrics"""
    print(f"\n{'='*80}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*80}")
    
    # Set seed for this configuration
    set_seed(seed)
    
    # Train RND
    print(f"Training RND with {len(positions)} samples...")
    method = RNDMethod(
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        device=device
    )
    
    training_losses = method.train_on_positions(
        positions,
        num_epochs=num_epochs,
        subset_ratio=1.0,
        gaussian_noise=gaussian_noise
    )
    
    print(f"Final training loss: {training_losses[-1]:.6f}")
    
    # Evaluate uncertainty
    print("Evaluating uncertainty method...")
    pred_uncertainty = evaluate_uncertainty_method(
        method, maze_map, grid_rows=9, grid_cols=12, device=device
    )
    
    # Compute both metrics
    l2_diff = compute_min_c_l2_norm_diff(gt_uncertainty, pred_uncertainty, maze_map)
    l2_inv = compute_min_c_l2_norm_inv(gt_uncertainty, pred_uncertainty, maze_map)
    
    print(f"L2 Diff: {l2_diff:.6f}")
    print(f"L2 Inv:  {l2_inv:.6f}")
    
    return {
        'name': config_name,
        'l2_diff': l2_diff,
        'l2_inv': l2_inv,
        'pred_uncertainty': pred_uncertainty,
        'final_loss': training_losses[-1]
    }

def analyze_cell_contributions(gt_matrix, pred_matrix, maze_map=None):
    """Analyze cell-by-cell contributions to both metrics"""
    if maze_map is None:
        maze_map = get_maze_map()
    
    # Extract open cells
    open_mask = np.array([[maze_map[i][j] == 0 for j in range(12)] for i in range(9)])
    gt_open = gt_matrix[open_mask]
    pred_open = pred_matrix[open_mask]
    
    # Filter finite values and GT != 0 (for Inv)
    finite_mask = np.isfinite(gt_open) & np.isfinite(pred_open)
    inv_finite_mask = finite_mask & (gt_open != 0)
    
    gt_clean = gt_open[finite_mask]
    pred_clean = pred_open[finite_mask]
    gt_inv_clean = gt_open[inv_finite_mask]
    pred_inv_clean = pred_open[inv_finite_mask]
    
    # L2 Diff analysis
    pred_norm_sq = np.dot(pred_clean, pred_clean)
    if pred_norm_sq == 0:
        c_diff = 0.0
    else:
        c_diff = np.dot(pred_clean, gt_clean) / pred_norm_sq
    
    pred_scaled = c_diff * pred_clean
    errors_diff = gt_clean - pred_scaled
    squared_errors_diff = errors_diff ** 2
    
    # L2 Inv analysis
    ratio = pred_inv_clean / gt_inv_clean
    c_inv = np.mean(ratio)
    errors_inv = c_inv - ratio
    squared_errors_inv = errors_inv ** 2
    
    # Map back to common cells (GT != 0)
    common_gt = gt_inv_clean
    common_pred = pred_inv_clean
    
    # Get Diff contributions for common cells
    common_pred_scaled = c_diff * common_pred
    common_errors_diff = common_gt - common_pred_scaled
    common_squared_errors_diff = common_errors_diff ** 2
    
    # Normalize contributions
    diff_contrib_norm = common_squared_errors_diff / np.sum(common_squared_errors_diff) if np.sum(common_squared_errors_diff) > 0 else common_squared_errors_diff
    inv_contrib_norm = squared_errors_inv / np.sum(squared_errors_inv) if np.sum(squared_errors_inv) > 0 else squared_errors_inv
    
    return {
        'gt': common_gt,
        'pred': common_pred,
        'ratio': ratio,
        'c_diff': c_diff,
        'c_inv': c_inv,
        'diff_contributions': common_squared_errors_diff,
        'inv_contributions': squared_errors_inv,
        'diff_contrib_norm': diff_contrib_norm,
        'inv_contrib_norm': inv_contrib_norm
    }

def compare_configurations(results, gt_uncertainty, maze_map, output_file=None):
    """Compare two configurations and analyze ranking differences"""
    print(f"\n{'='*80}")
    print("RANKING COMPARISON")
    print(f"{'='*80}")
    
    config1, config2 = results[0], results[1]
    
    # Rankings
    if config1['l2_diff'] < config2['l2_diff']:
        diff_winner = config1['name']
        diff_rank = {config1['name']: 1, config2['name']: 2}
    else:
        diff_winner = config2['name']
        diff_rank = {config1['name']: 2, config2['name']: 1}
    
    if config1['l2_inv'] < config2['l2_inv']:
        inv_winner = config1['name']
        inv_rank = {config1['name']: 1, config2['name']: 2}
    else:
        inv_winner = config2['name']
        inv_rank = {config1['name']: 2, config2['name']: 1}
    
    print(f"\nL2 Diff Rankings:")
    print(f"  1st: {diff_winner} (value: {min(config1['l2_diff'], config2['l2_diff']):.6f})")
    print(f"  2nd: {[c['name'] for c in results if c['name'] != diff_winner][0]} (value: {max(config1['l2_diff'], config2['l2_diff']):.6f})")
    
    print(f"\nL2 Inv Rankings:")
    print(f"  1st: {inv_winner} (value: {min(config1['l2_inv'], config2['l2_inv']):.6f})")
    print(f"  2nd: {[c['name'] for c in results if c['name'] != inv_winner][0]} (value: {max(config1['l2_inv'], config2['l2_inv']):.6f})")
    
    ranking_agreement = (diff_winner == inv_winner)
    print(f"\n{'='*80}")
    if ranking_agreement:
        print("✓ RANKINGS AGREE: Both metrics rank the same configuration as best")
    else:
        print("✗ RANKINGS DISAGREE: Metrics rank different configurations as best")
        print(f"  L2 Diff prefers: {diff_winner}")
        print(f"  L2 Inv prefers:  {inv_winner}")
    print(f"{'='*80}")
    
    # Analyze cell-by-cell differences
    print(f"\n{'='*80}")
    print("CELL-BY-CELL ANALYSIS: Where do rankings differ?")
    print(f"{'='*80}")
    
    analysis1 = analyze_cell_contributions(gt_uncertainty, config1['pred_uncertainty'], maze_map)
    analysis2 = analyze_cell_contributions(gt_uncertainty, config2['pred_uncertainty'], maze_map)
    
    # Compare contributions cell-by-cell
    n_cells = len(analysis1['gt'])
    
    # For each cell, determine which config contributes more to each metric
    diff_preference = np.zeros(n_cells)  # +1 if config1 better, -1 if config2 better
    inv_preference = np.zeros(n_cells)
    
    for i in range(n_cells):
        # L2 Diff: lower error is better
        if analysis1['diff_contributions'][i] < analysis2['diff_contributions'][i]:
            diff_preference[i] = 1  # Config1 better
        elif analysis1['diff_contributions'][i] > analysis2['diff_contributions'][i]:
            diff_preference[i] = -1  # Config2 better
        
        # L2 Inv: lower error is better
        if analysis1['inv_contributions'][i] < analysis2['inv_contributions'][i]:
            inv_preference[i] = 1  # Config1 better
        elif analysis1['inv_contributions'][i] > analysis2['inv_contributions'][i]:
            inv_preference[i] = -1  # Config2 better
    
    # Find cells where metrics disagree
    disagreement_mask = (diff_preference != inv_preference) & (diff_preference != 0) & (inv_preference != 0)
    disagreement_indices = np.where(disagreement_mask)[0]
    
    print(f"\nCells where metrics disagree on which config is better: {len(disagreement_indices)} / {n_cells}")
    
    if len(disagreement_indices) > 0:
        # Sort by magnitude of disagreement
        disagreement_magnitudes = []
        for idx in disagreement_indices:
            diff_diff = abs(analysis1['diff_contributions'][idx] - analysis2['diff_contributions'][idx])
            inv_diff = abs(analysis1['inv_contributions'][idx] - analysis2['inv_contributions'][idx])
            # Weighted by relative importance
            diff_norm1 = analysis1['diff_contrib_norm'][idx]
            diff_norm2 = analysis2['diff_contrib_norm'][idx]
            inv_norm1 = analysis1['inv_contrib_norm'][idx]
            inv_norm2 = analysis2['inv_contrib_norm'][idx]
            magnitude = abs(diff_norm1 - diff_norm2) + abs(inv_norm1 - inv_norm2)
            disagreement_magnitudes.append((magnitude, idx))
        
        disagreement_magnitudes.sort(reverse=True)
        top_n = min(10, len(disagreement_magnitudes))
        
        print(f"\nTop {top_n} cells with largest disagreement:")
        print(f"{'Rank':<6} {'GT':<12} {'Pred1':<12} {'Pred2':<12} {'Diff Pref':<12} {'Inv Pref':<12} {'Magnitude':<12}")
        for rank, (magnitude, idx) in enumerate(disagreement_magnitudes[:top_n]):
            gt_val = analysis1['gt'][idx]
            pred1_val = analysis1['pred'][idx]
            pred2_val = analysis2['pred'][idx]
            diff_pref = "Config1" if diff_preference[idx] > 0 else "Config2"
            inv_pref = "Config1" if inv_preference[idx] > 0 else "Config2"
            print(f"{rank+1:<6} {gt_val:<12.6f} {pred1_val:<12.6f} {pred2_val:<12.6f} "
                  f"{diff_pref:<12} {inv_pref:<12} {magnitude:<12.6f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nConfiguration 1 ({config1['name']}):")
    print(f"  L2 Diff: {config1['l2_diff']:.6f}")
    print(f"  L2 Inv:  {config1['l2_inv']:.6f}")
    print(f"  Optimal c* (Diff): {analysis1['c_diff']:.6f}")
    print(f"  Optimal c* (Inv):  {analysis1['c_inv']:.6f}")
    print(f"  Ratio std: {np.std(analysis1['ratio']):.6f}")
    
    print(f"\nConfiguration 2 ({config2['name']}):")
    print(f"  L2 Diff: {config2['l2_diff']:.6f}")
    print(f"  L2 Inv:  {config2['l2_inv']:.6f}")
    print(f"  Optimal c* (Diff): {analysis2['c_diff']:.6f}")
    print(f"  Optimal c* (Inv):  {analysis2['c_inv']:.6f}")
    print(f"  Ratio std: {np.std(analysis2['ratio']):.6f}")
    
    # Create visualization
    if output_file:
        visualize_comparison(results, analysis1, analysis2, disagreement_indices, 
                           diff_preference, inv_preference, output_file)

def visualize_comparison(results, analysis1, analysis2, disagreement_indices,
                        diff_preference, inv_preference, output_file):
    """Create visualization comparing the two configurations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Metric Ranking Comparison: Configuration 1 vs Configuration 2', 
                 fontsize=16, fontweight='bold')
    
    config1, config2 = results[0], results[1]
    
    # Plot 1: Metric values comparison
    ax1 = axes[0, 0]
    metrics = ['L2 Diff', 'L2 Inv']
    config1_vals = [config1['l2_diff'], config1['l2_inv']]
    config2_vals = [config2['l2_diff'], config2['l2_inv']]
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, config1_vals, width, label=config1['name'], alpha=0.7, color='blue')
    ax1.bar(x + width/2, config2_vals, width, label=config2['name'], alpha=0.7, color='red')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Metric Values Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Ratio distributions
    ax2 = axes[0, 1]
    ax2.hist(analysis1['ratio'], bins=30, alpha=0.6, label=config1['name'], color='blue', density=True)
    ax2.hist(analysis2['ratio'], bins=30, alpha=0.6, label=config2['name'], color='red', density=True)
    ax2.axvline(analysis1['c_inv'], color='blue', linestyle='--', linewidth=2, 
                label=f"{config1['name']} c* = {analysis1['c_inv']:.3f}")
    ax2.axvline(analysis2['c_inv'], color='red', linestyle='--', linewidth=2,
                label=f"{config2['name']} c* = {analysis2['c_inv']:.3f}")
    ax2.set_xlabel('Pred/GT Ratio')
    ax2.set_ylabel('Density')
    ax2.set_title('Ratio Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GT vs Pred scatter
    ax3 = axes[0, 2]
    ax3.scatter(analysis1['gt'], analysis1['pred'], alpha=0.5, s=20, 
                label=config1['name'], color='blue')
    ax3.scatter(analysis2['gt'], analysis2['pred'], alpha=0.5, s=20,
                label=config2['name'], color='red', marker='^')
    x_range = np.linspace(analysis1['gt'].min(), analysis1['gt'].max(), 100)
    ax3.plot(x_range, analysis1['c_diff'] * x_range, 'b--', linewidth=2, alpha=0.7)
    ax3.plot(x_range, analysis2['c_diff'] * x_range, 'r--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Ground Truth')
    ax3.set_ylabel('Prediction')
    ax3.set_title('GT vs Pred with Optimal Scaling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cell contributions - Diff
    ax4 = axes[1, 0]
    n_cells = len(analysis1['gt'])
    top_n = min(20, n_cells)
    top_indices = np.argsort(analysis1['diff_contributions'])[-top_n:][::-1]
    x_pos = np.arange(top_n)
    width = 0.35
    ax4.bar(x_pos - width/2, analysis1['diff_contrib_norm'][top_indices] * 100, width,
            label=config1['name'], color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, analysis2['diff_contrib_norm'][top_indices] * 100, width,
            label=config2['name'], color='red', alpha=0.7)
    ax4.set_xlabel('Cell Rank (by Config1 Diff contribution)')
    ax4.set_ylabel('Contribution (%)')
    ax4.set_title('Top Cells: L2 Diff Contributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Cell contributions - Inv
    ax5 = axes[1, 1]
    top_indices_inv = np.argsort(analysis1['inv_contributions'])[-top_n:][::-1]
    ax5.bar(x_pos - width/2, analysis1['inv_contrib_norm'][top_indices_inv] * 100, width,
            label=config1['name'], color='blue', alpha=0.7)
    ax5.bar(x_pos + width/2, analysis2['inv_contrib_norm'][top_indices_inv] * 100, width,
            label=config2['name'], color='red', alpha=0.7)
    ax5.set_xlabel('Cell Rank (by Config1 Inv contribution)')
    ax5.set_ylabel('Contribution (%)')
    ax5.set_title('Top Cells: L2 Inv Contributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Disagreement cells
    ax6 = axes[1, 2]
    if len(disagreement_indices) > 0:
        # Show cells where metrics disagree
        agree_mask = ~np.isin(np.arange(n_cells), disagreement_indices)
        ax6.scatter(analysis1['gt'][agree_mask], analysis1['pred'][agree_mask], 
                   alpha=0.3, s=10, color='gray', label='Agree')
        ax6.scatter(analysis1['gt'][disagreement_indices], 
                   analysis1['pred'][disagreement_indices],
                   alpha=0.7, s=30, color='orange', label=f'Disagree ({len(disagreement_indices)})', 
                   marker='x', linewidths=2)
        ax6.scatter(analysis2['gt'][disagreement_indices],
                   analysis2['pred'][disagreement_indices],
                   alpha=0.7, s=30, color='purple', marker='+', linewidths=2)
    else:
        ax6.text(0.5, 0.5, 'No disagreement\nbetween metrics', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
    ax6.set_xlabel('Ground Truth')
    ax6.set_ylabel('Prediction')
    ax6.set_title('Cells Where Metrics Disagree')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Compare metric rankings for two configurations')
    parser.add_argument('--output_dim', type=int, default=128, help='RND output dimension')
    parser.add_argument('--hidden_dims', type=str, default='64,64', help='RND hidden dimensions')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--gaussian_noise', type=float, default=0.0, help='Gaussian noise level')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_file', type=str, default=None, help='Output file for visualization')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading PointMaze dataset...")
    dataset, maze_map = load_pointmaze_dataset()
    
    # Extract all positions
    print("Extracting all available positions...")
    all_positions = extract_positions_from_dataset(dataset, num_samples=None)
    total_positions = len(all_positions)
    print(f"Total available positions: {total_positions}")
    
    # Define two configurations with different sample sizes
    configs = [
        {
            'name': '100_samples',
            'num_samples': 100,
            'seed_offset': 0
        },
        {
            'name': '10000_samples',
            'num_samples': 10000,
            'seed_offset': 1000
        }
    ]
    
    results = []
    gt_uncertainties = []
    
    for config in configs:
        # Sample positions for this configuration
        np.random.seed(args.seed + config['seed_offset'])
        if config['num_samples'] > total_positions:
            positions = all_positions
        else:
            indices = np.random.choice(total_positions, config['num_samples'], replace=False)
            positions = all_positions[indices]
        
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']} ({len(positions)} samples)")
        print(f"{'='*80}")
        
        # Calculate ground truth from this subset
        gt_uncertainty = calculate_ground_truth_from_positions(
            positions, maze_map, grid_rows=9, grid_cols=12
        )
        gt_uncertainties.append(gt_uncertainty)
        
        # Train and evaluate
        result = train_and_evaluate(
            config['name'],
            positions,
            gt_uncertainty,
            maze_map,
            args.output_dim,
            [int(x) for x in args.hidden_dims.split(',')],
            args.num_epochs,
            args.gaussian_noise,
            device,
            args.seed + config['seed_offset']
        )
        results.append(result)
    
    # Compare configurations
    if args.output_file is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        args.output_file = os.path.join(results_dir, 'metric_ranking_comparison.png')
    
    compare_configurations(results, gt_uncertainties[0], maze_map, args.output_file)
    
    print(f"\n{'='*80}")
    print("Comparison complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()








