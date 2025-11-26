import torch
import numpy as np
import sys
sys.path.append('sweep_uncertainty')
from utilities.uncertainty_methods import RNDLinearSGDMethod, RNDLinearLSMethod
from utilities.evaluation import get_maze_map, normalize_uncertainty_matrix, compute_l2_distance, calculate_ground_truth_from_positions

def create_phi_weights_deterministic(feature_dim, seed):
    """Create deterministic phi weights"""
    torch.manual_seed(seed)
    phi_net = torch.nn.Linear(2, feature_dim)
    return phi_net.state_dict()

def ridge_least_squares_fit(X, y, lambda_reg=1e-6, add_intercept=True):
    """Ridge Regression: min ||X_aug w - y||^2 + λ * ||w||^2"""
    if y.ndim == 1:
        y = y[:, None]
    if add_intercept:
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        X_aug = X
    
    # Ridge Regression: (X^T X + λI) w = X^T y
    XTX = X_aug.T @ X_aug
    XTy = X_aug.T @ y
    
    # Add regularization: XTX + λI
    n_features = XTX.shape[0]
    XTX_reg = XTX + lambda_reg * np.eye(n_features)
    
    # Solve: (XTX + λI) w = XTy
    try:
        w = np.linalg.solve(XTX_reg, XTy)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        w = np.linalg.pinv(XTX_reg) @ XTy
    
    w = w.squeeze()
    intercept = w[0] if add_intercept else 0.0
    coefs = w[1:] if add_intercept else w
    
    # Compute residuals
    pred = X_aug @ w
    residuals = np.sum((pred - y.squeeze()) ** 2)
    
    return intercept, coefs, residuals

def run_comparison(num_runs=50, feature_dim=256, phi_seed=42, theta_seed=42, num_samples=100, 
                   regularization=0.0, lambda_reg=1e-6):
    """
    Run comparison averaging over num_runs
    
    Args:
        regularization: 0.0 for no regularization, >0.0 to enable
        lambda_reg: regularization strength (only used if regularization > 0)
    """
    
    device = 'cpu'
    epochs_list = [50, 100, 500, 1000, 2000]
    
    reg_str = f"regularized (λ={lambda_reg})" if regularization > 0 else "unregularized"
    
    # Create shared phi weights
    phi_weights = create_phi_weights_deterministic(feature_dim, phi_seed)
    
    # Get grid coordinates
    maze_map = get_maze_map()
    grid_rows, grid_cols = 9, 12
    left_edge = -6.0
    top_edge = 4.5
    cell_width = 12.0 / grid_cols
    cell_height = 9.0 / grid_rows
    
    grid_coords = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if maze_map[row][col] == 0:
                grid_row = row + 1
                grid_col = col + 1
                center_x = left_edge + (grid_col - 1 + 0.5) * cell_width
                center_y = top_edge - (grid_row - 1 + 0.5) * cell_height
                grid_coords.append([center_x, center_y])
    grid_coords = np.array(grid_coords)
    
    # Storage for all runs
    all_results = {}
    for num_epochs in epochs_list:
        all_results[f'SGD_{num_epochs}'] = {
            'training_loss': [],
            'target_mean': [], 'target_min': [], 'target_max': [],
            'pred_mean': [], 'pred_min': [], 'pred_max': [],
            'uncertainty_mean': [], 'uncertainty_min': [], 'uncertainty_max': [],
            'norm_unc_mean': [], 'norm_unc_min': [], 'norm_unc_max': [],
            'gt_mean': [], 'gt_min': [], 'gt_max': [],
            'l2_distance': []
        }
    all_results['OLS'] = {
        'training_loss': [],
        'target_mean': [], 'target_min': [], 'target_max': [],
        'pred_mean': [], 'pred_min': [], 'pred_max': [],
        'uncertainty_mean': [], 'uncertainty_min': [], 'uncertainty_max': [],
        'norm_unc_mean': [], 'norm_unc_min': [], 'norm_unc_max': [],
        'gt_mean': [], 'gt_min': [], 'gt_max': [],
        'l2_distance': []
    }
    
    print("=" * 80)
    print(f"COMPREHENSIVE COMPARISON: SGD vs OLS (averaging over {num_runs} runs, {reg_str})")
    print("=" * 80)
    
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*80}")
        
        # Generate positions for this run
        np.random.seed(42 + run_idx)
        positions = np.random.randn(num_samples, 2) * 3
        
        # Calculate ground truth for this run
        gt_uncertainty = calculate_ground_truth_from_positions(
            positions, maze_map, grid_rows, grid_cols
        )
        gt_normalized = normalize_uncertainty_matrix(gt_uncertainty, maze_map)
        
        # Extract GT values
        gt_values = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if maze_map[row][col] == 0:
                    if np.isfinite(gt_normalized[row, col]):
                        gt_values.append(gt_normalized[row, col])
        gt_values = np.array(gt_values)
        
        # Train SGD for each epoch count
        for num_epochs in epochs_list:
            torch.manual_seed(42 + run_idx)  # Different seed per run for predictor init
            sgd_method = RNDLinearSGDMethod(
                feature_dim, device=device, 
                phi_weights=phi_weights, 
                theta_seed=theta_seed,
                predictor_seed=42 + run_idx
            )
            
            # Add weight decay if regularization is enabled
            if regularization > 0:
                sgd_method.optimizer = torch.optim.Adam(
                    sgd_method.predictor.parameters(), 
                    lr=0.001, 
                    weight_decay=lambda_reg
                )
            
            # Train
            losses = sgd_method.train_on_positions(
                positions, num_epochs=num_epochs, gaussian_noise=0.0
            )
            final_loss = losses[-1] if losses else float('nan')
            
            # Evaluate on grid
            with torch.no_grad():
                grid_tensor = torch.FloatTensor(grid_coords).to(device)
                grid_phi = sgd_method.phi(grid_tensor)
                target = torch.matmul(grid_phi, sgd_method.theta).cpu().numpy()
                pred = sgd_method.predictor(grid_phi).squeeze().cpu().numpy()
                uncertainty = np.abs(target - pred)
                
                # Create uncertainty matrix
                unc_matrix = np.full((grid_rows, grid_cols), np.nan)
                idx = 0
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if maze_map[row][col] == 0:
                            unc_matrix[row, col] = uncertainty[idx]
                            idx += 1
                
                # Normalize
                pred_normalized = normalize_uncertainty_matrix(unc_matrix, maze_map)
                
                # Extract normalized prediction vector
                pred_vector = []
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if maze_map[row][col] == 0:
                            if np.isfinite(pred_normalized[row, col]):
                                pred_vector.append(pred_normalized[row, col])
                pred_vector = np.array(pred_vector)
                
                # Compute L2 distance
                l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
                
                # Store results
                all_results[f'SGD_{num_epochs}']['training_loss'].append(final_loss)
                all_results[f'SGD_{num_epochs}']['target_mean'].append(np.mean(target))
                all_results[f'SGD_{num_epochs}']['target_min'].append(np.min(target))
                all_results[f'SGD_{num_epochs}']['target_max'].append(np.max(target))
                all_results[f'SGD_{num_epochs}']['pred_mean'].append(np.mean(pred))
                all_results[f'SGD_{num_epochs}']['pred_min'].append(np.min(pred))
                all_results[f'SGD_{num_epochs}']['pred_max'].append(np.max(pred))
                all_results[f'SGD_{num_epochs}']['uncertainty_mean'].append(np.mean(uncertainty))
                all_results[f'SGD_{num_epochs}']['uncertainty_min'].append(np.min(uncertainty))
                all_results[f'SGD_{num_epochs}']['uncertainty_max'].append(np.max(uncertainty))
                all_results[f'SGD_{num_epochs}']['norm_unc_mean'].append(np.mean(pred_vector))
                all_results[f'SGD_{num_epochs}']['norm_unc_min'].append(np.min(pred_vector))
                all_results[f'SGD_{num_epochs}']['norm_unc_max'].append(np.max(pred_vector))
                all_results[f'SGD_{num_epochs}']['gt_mean'].append(np.mean(gt_values))
                all_results[f'SGD_{num_epochs}']['gt_min'].append(np.min(gt_values))
                all_results[f'SGD_{num_epochs}']['gt_max'].append(np.max(gt_values))
                all_results[f'SGD_{num_epochs}']['l2_distance'].append(l2_distance)
        
        # Train OLS
        ols_method = RNDLinearLSMethod(
            feature_dim, device=device,
            phi_weights=phi_weights,
            theta_seed=theta_seed
        )
        
        # If regularization is enabled, use Ridge Regression instead of OLS
        if regularization > 0:
            # Compute features and targets
            coords_tensor = torch.FloatTensor(positions[:, :2]).to(device)
            with torch.no_grad():
                phi_output = ols_method.phi(coords_tensor)  # [N, feature_dim]
                target_output = torch.matmul(phi_output, ols_method.theta)  # [N]
            
            # Convert to numpy for Ridge Regression
            X = phi_output.cpu().numpy()  # [N, feature_dim]
            y = target_output.cpu().numpy()  # [N]
            
            # Fit using Ridge Regression
            intercept, coefs, residuals = ridge_least_squares_fit(X, y, lambda_reg=lambda_reg, add_intercept=True)
            
            # Store results
            ols_method.predictor_intercept = torch.FloatTensor([intercept]).to(device)
            ols_method.predictor_weights = torch.FloatTensor(coefs).to(device)
            
            # Compute final loss for logging
            pred_output = torch.matmul(phi_output, ols_method.predictor_weights) + ols_method.predictor_intercept
            final_loss = torch.mean((pred_output - target_output) ** 2).item()
            losses = [final_loss] * 1
        else:
            # Regular OLS (no regularization)
            losses = ols_method.train_on_positions(positions, num_epochs=1, gaussian_noise=0.0)
        final_loss = losses[-1] if losses else float('nan')
        
        # Evaluate on grid
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid_coords).to(device)
            grid_phi = ols_method.phi(grid_tensor)
            target = torch.matmul(grid_phi, ols_method.theta).cpu().numpy()
            pred = torch.matmul(grid_phi, ols_method.predictor_weights) + ols_method.predictor_intercept
            pred = pred.squeeze().cpu().numpy()
            uncertainty = np.abs(target - pred)
            
            # Create uncertainty matrix
            unc_matrix = np.full((grid_rows, grid_cols), np.nan)
            idx = 0
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if maze_map[row][col] == 0:
                        unc_matrix[row, col] = uncertainty[idx]
                        idx += 1
            
            # Normalize
            pred_normalized = normalize_uncertainty_matrix(unc_matrix, maze_map)
            
            # Extract normalized prediction vector
            pred_vector = []
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if maze_map[row][col] == 0:
                        if np.isfinite(pred_normalized[row, col]):
                            pred_vector.append(pred_normalized[row, col])
            pred_vector = np.array(pred_vector)
            
            # Compute L2 distance
            l2_distance = compute_l2_distance(gt_normalized, pred_normalized, maze_map)
            
            # Store results
            all_results['OLS']['training_loss'].append(final_loss)
            all_results['OLS']['target_mean'].append(np.mean(target))
            all_results['OLS']['target_min'].append(np.min(target))
            all_results['OLS']['target_max'].append(np.max(target))
            all_results['OLS']['pred_mean'].append(np.mean(pred))
            all_results['OLS']['pred_min'].append(np.min(pred))
            all_results['OLS']['pred_max'].append(np.max(pred))
            all_results['OLS']['uncertainty_mean'].append(np.mean(uncertainty))
            all_results['OLS']['uncertainty_min'].append(np.min(uncertainty))
            all_results['OLS']['uncertainty_max'].append(np.max(uncertainty))
            all_results['OLS']['norm_unc_mean'].append(np.mean(pred_vector))
            all_results['OLS']['norm_unc_min'].append(np.min(pred_vector))
            all_results['OLS']['norm_unc_max'].append(np.max(pred_vector))
            all_results['OLS']['gt_mean'].append(np.mean(gt_values))
            all_results['OLS']['gt_min'].append(np.min(gt_values))
            all_results['OLS']['gt_max'].append(np.max(gt_values))
            all_results['OLS']['l2_distance'].append(l2_distance)
        
        # Print progress
        if (run_idx + 1) % 10 == 0:
            print(f"\nCompleted {run_idx + 1}/{num_runs} runs...")
    
    # Compute averages and print results
    print("\n" + "=" * 100)
    print("RESULTS TABLE - SGD vs OLS (Averaged over {} runs) {} samples; {}".format(num_runs, num_samples, reg_str))
    print("=" * 100)
    
    print("\n" + "=" * 100)
    print("1. TRAINING LOSS (Final Epoch)")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        losses = all_results[method_name]['training_loss']
        print(f"{method_name:<12} {np.mean(losses):<15.10f} {np.std(losses):<15.10f} {np.min(losses):<15.10f} {np.max(losses):<15.10f}")
    
    print("\n" + "=" * 100)
    print("2. TARGET VALUES (should be same for all methods)")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        means = all_results[method_name]['target_mean']
        mins = all_results[method_name]['target_min']
        maxs = all_results[method_name]['target_max']
        print(f"{method_name:<12} {np.mean(means):<15.6f} {np.mean(mins):<15.6f} {np.mean(maxs):<15.6f}")
    
    print("\n" + "=" * 100)
    print("3. PREDICTION VALUES")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        means = all_results[method_name]['pred_mean']
        mins = all_results[method_name]['pred_min']
        maxs = all_results[method_name]['pred_max']
        print(f"{method_name:<12} {np.mean(means):<15.6f} {np.std(means):<15.6f} {np.mean(mins):<15.6f} {np.mean(maxs):<15.6f}")
    
    print("\n" + "=" * 100)
    print("4. RAW UNCERTAINTY (|target - prediction|)")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        means = all_results[method_name]['uncertainty_mean']
        mins = all_results[method_name]['uncertainty_min']
        maxs = all_results[method_name]['uncertainty_max']
        print(f"{method_name:<12} {np.mean(means):<15.10f} {np.std(means):<15.10f} {np.mean(mins):<15.10f} {np.mean(maxs):<15.10f}")
    
    print("\n" + "=" * 100)
    print("5. NORMALIZED UNCERTAINTY (divided by the max value) – starts to differ between converged SGD and OLS")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        means = all_results[method_name]['norm_unc_mean']
        mins = all_results[method_name]['norm_unc_min']
        maxs = all_results[method_name]['norm_unc_max']
        print(f"{method_name:<12} {np.mean(means):<15.6f} {np.std(means):<15.6f} {np.mean(mins):<15.6f} {np.mean(maxs):<15.6f}")
    
    print("\n" + "=" * 100)
    print("6. GROUND TRUTH VALUES (normalized)")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        means = all_results[method_name]['gt_mean']
        mins = all_results[method_name]['gt_min']
        maxs = all_results[method_name]['gt_max']
        print(f"{method_name:<12} {np.mean(means):<15.6f} {np.mean(mins):<15.6f} {np.mean(maxs):<15.6f}")
    
    print("\n" + "=" * 100)
    print("7. L2 DISTANCE (normalized prediction vs normalized GT) – converged SGD are closer to OLS although not the same")
    print("=" * 100)
    print(f"{'Method':<12} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        l2s = all_results[method_name]['l2_distance']
        print(f"{method_name:<12} {np.mean(l2s):<15.6f} {np.std(l2s):<15.6f} {np.min(l2s):<15.6f} {np.max(l2s):<15.6f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Find best L2 for each method
    print("\nL2 Distance Summary:")
    l2_means = {}
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        l2_means[method_name] = np.mean(all_results[method_name]['l2_distance'])
        print(f"  {method_name}: {l2_means[method_name]:.6f}")
    
    best_method = min(l2_means, key=l2_means.get)
    print(f"\nBest L2: {best_method} with L2 = {l2_means[best_method]:.6f}")
    
    # Compare converged SGD vs OLS
    sgd_converged_l2 = l2_means['SGD_2000']
    ols_l2 = l2_means['OLS']
    print(f"\nConverged SGD (2000 epochs) L2: {sgd_converged_l2:.6f}")
    print(f"OLS L2: {ols_l2:.6f}")
    print(f"Difference: {abs(sgd_converged_l2 - ols_l2):.6f}")
    
    # Check why unconverged might be better
    print("\n" + "=" * 80)
    print("WHY UNCONVERGED SGD SOMETIMES HAS LOWER L2")
    print("=" * 80)
    
    # Compare training loss vs L2
    print("\nTraining Loss vs L2 Distance:")
    print(f"{'Method':<12} {'Train Loss':<15} {'L2 Distance':<15} {'Ratio':<15}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        train_loss = np.mean(all_results[method_name]['training_loss'])
        l2 = l2_means[method_name]
        ratio = train_loss / l2 if l2 > 0 else float('inf')
        print(f"{method_name:<12} {train_loss:<15.10f} {l2:<15.6f} {ratio:<15.2f}")
    
    # Compare raw uncertainty vs normalized uncertainty
    print("\nRaw Uncertainty Max vs Normalized Uncertainty Max:")
    print(f"{'Method':<12} {'Raw Unc Max':<20} {'Norm Unc Max':<20} {'Amplification':<20}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'SGD_2000', 'OLS']:
        raw_max = np.mean(all_results[method_name]['uncertainty_max'])
        norm_max = np.mean(all_results[method_name]['norm_unc_max'])
        amplification = norm_max / raw_max if raw_max > 0 else float('inf')
        print(f"{method_name:<12} {raw_max:<20.10f} {norm_max:<20.6f} {amplification:<20.2f}")
    
    # Compare prediction differences
    print("\nPrediction Mean Differences (relative to SGD_2000):")
    sgd_2000_pred_mean = np.mean(all_results['SGD_2000']['pred_mean'])
    print(f"{'Method':<12} {'Pred Mean':<15} {'Diff from SGD_2000':<20}")
    for method_name in ['SGD_50', 'SGD_100', 'SGD_500', 'SGD_1000', 'OLS']:
        pred_mean = np.mean(all_results[method_name]['pred_mean'])
        diff = pred_mean - sgd_2000_pred_mean
        print(f"{method_name:<12} {pred_mean:<15.6f} {diff:<20.6f}")
    
    return all_results

if __name__ == '__main__':
    # Set regularization: 0.0 = no regularization, >0.0 = enable regularization
    USE_REGULARIZATION = True  # Set to False to disable regularization
    LAMBDA_REG = 1e-2  # Regularization strength (only used if USE_REGULARIZATION=True) - CHANGED TO 1e-2
    
    regularization = LAMBDA_REG if USE_REGULARIZATION else 0.0
    
    results = run_comparison(
        num_runs=50, 
        feature_dim=256, 
        phi_seed=42, 
        theta_seed=42, 
        num_samples=100,
        regularization=regularization,
        lambda_reg=LAMBDA_REG
    )

