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
    
    XTX = X_aug.T @ X_aug
    XTy = X_aug.T @ y
    n_features = XTX.shape[0]
    XTX_reg = XTX + lambda_reg * np.eye(n_features)
    
    try:
        w = np.linalg.solve(XTX_reg, XTy)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(XTX_reg) @ XTy
    
    w = w.squeeze()
    intercept = w[0] if add_intercept else 0.0
    coefs = w[1:] if add_intercept else w
    
    pred = X_aug @ w
    residuals = np.sum((pred - y.squeeze()) ** 2)
    
    return intercept, coefs, residuals

def run_comprehensive_comparison(num_runs=20, num_samples=100, noise_levels=[0.0, 0.1, 0.5], 
                                  regularization_levels=[0.0, 1e-4], lambda_reg=1e-4):
    """
    Comprehensive comparison with noise and regularization
    """
    device = 'cpu'
    feature_dim = 256
    phi_seed = 42
    theta_seed = 42
    
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
    
    # Storage for all results: [noise_level][regularization][method]
    all_results = {}
    weights_biases = {}  # Store weights and biases for final analysis
    
    for noise_std in noise_levels:
        all_results[noise_std] = {}
        weights_biases[noise_std] = {}
        for reg in regularization_levels:
            all_results[noise_std][reg] = {
                'SGD_1000': {'training_loss': [], 'pred_mean': [], 'pred_std': [], 
                            'uncertainty_mean': [], 'uncertainty_max': [],
                            'norm_unc_mean': [], 'l2_distance': [],
                            'weights_l2': [], 'bias': []},
                'SGD_2000': {'training_loss': [], 'pred_mean': [], 'pred_std': [],
                            'uncertainty_mean': [], 'uncertainty_max': [],
                            'norm_unc_mean': [], 'l2_distance': [],
                            'weights_l2': [], 'bias': []},
                'OLS': {'training_loss': [], 'pred_mean': [], 'pred_std': [],
                       'uncertainty_mean': [], 'uncertainty_max': [],
                       'norm_unc_mean': [], 'l2_distance': [],
                       'weights_l2': [], 'bias': []}
            }
            weights_biases[noise_std][reg] = {
                'SGD_1000': {'weights': None, 'bias': None},
                'SGD_2000': {'weights': None, 'bias': None},
                'OLS': {'weights': None, 'bias': None}
            }
    
    print("=" * 100)
    print(f"COMPREHENSIVE COMPARISON: Noise × Regularization × Methods")
    print(f"Runs: {num_runs}, Samples: {num_samples}")
    print(f"Noise levels: {noise_levels}")
    print(f"Regularization: {regularization_levels}")
    print("=" * 100)
    
    for run_idx in range(num_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx + 1}/{num_runs}")
        print(f"{'='*80}")
        
        # Generate positions for this run
        np.random.seed(42 + run_idx)
        positions = np.random.randn(num_samples, 2) * 3
        
        # Calculate ground truth once per run
        gt_matrix = calculate_ground_truth_from_positions(positions, maze_map)
        gt_normalized = normalize_uncertainty_matrix(gt_matrix, maze_map)
        # Extract GT values for reporting
        gt_values = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if maze_map[row][col] == 0 and np.isfinite(gt_normalized[row, col]):
                    gt_values.append(gt_normalized[row, col])
        gt_values = np.array(gt_values)
        
        for noise_std in noise_levels:
            for reg in regularization_levels:
                reg_str = f"λ={reg}" if reg > 0 else "unreg"
                print(f"\n  Noise={noise_std:.1f}, {reg_str}")
                
                # Train SGD_1000
                torch.manual_seed(42 + run_idx)
                sgd_1000 = RNDLinearSGDMethod(
                    feature_dim, device=device,
                    phi_weights=phi_weights,
                    theta_seed=theta_seed,
                    predictor_seed=42 + run_idx
                )
                if reg > 0:
                    sgd_1000.optimizer = torch.optim.Adam(
                        sgd_1000.predictor.parameters(), lr=0.001, weight_decay=lambda_reg if reg > 0 else 0.0
                    )
                losses = sgd_1000.train_on_positions(positions, num_epochs=1000, gaussian_noise=noise_std)
                
                # Train SGD_2000 (continue from 1000, train for 1000 more epochs to reach 2000 total)
                sgd_2000 = RNDLinearSGDMethod(
                    feature_dim, device=device,
                    phi_weights=phi_weights,
                    theta_seed=theta_seed,
                    predictor_seed=42 + run_idx
                )
                sgd_2000.predictor.weight.data = sgd_1000.predictor.weight.data.clone()
                sgd_2000.predictor.bias.data = sgd_1000.predictor.bias.data.clone()
                if reg > 0:
                    sgd_2000.optimizer = torch.optim.Adam(
                        sgd_2000.predictor.parameters(), lr=0.001, weight_decay=lambda_reg
                    )
                # Continue training for 1000 more epochs (total 2000)
                losses_2000 = sgd_2000.train_on_positions(positions, num_epochs=1000, gaussian_noise=noise_std)
                
                # Train OLS
                ols = RNDLinearLSMethod(
                    feature_dim, device=device,
                    phi_weights=phi_weights,
                    theta_seed=theta_seed
                )
                
                if reg > 0:
                    # Add noise to targets for regularized OLS (FIX BUG)
                    coords_tensor = torch.FloatTensor(positions[:, :2]).to(device)
                    
                    # Create noise map (same as in train_on_positions)
                    noise_map = {}
                    if noise_std > 0:
                        torch.manual_seed(42 + run_idx)  # Deterministic noise per run
                        unique_positions_list = []
                        seen = set()
                        for i, pos in enumerate(positions[:, :2]):
                            pos_tuple = tuple(pos.tolist())
                            if pos_tuple not in seen:
                                unique_positions_list.append(pos_tuple)
                                seen.add(pos_tuple)
                        unique_positions_list.sort()
                        for pos_tuple in unique_positions_list:
                            noise_map[pos_tuple] = torch.randn(1).item() * noise_std
                    
                    with torch.no_grad():
                        phi_output = ols.phi(coords_tensor)
                        target_output = torch.matmul(phi_output, ols.theta)
                        
                        # Add noise to targets
                        if noise_std > 0:
                            noise_values = torch.zeros(len(positions), device=device)
                            for i, pos in enumerate(positions[:, :2]):
                                pos_tuple = tuple(pos.tolist())
                                noise_values[i] = noise_map[pos_tuple]
                            target_output += noise_values
                    
                    X = phi_output.cpu().numpy()
                    y = target_output.cpu().numpy()
                    intercept, coefs, residuals = ridge_least_squares_fit(X, y, lambda_reg=lambda_reg, add_intercept=True)
                    ols.predictor_intercept = torch.FloatTensor([intercept]).to(device)
                    ols.predictor_weights = torch.FloatTensor(coefs).to(device)
                    final_loss = residuals / len(y)
                else:
                    # Get actual loss from train_on_positions
                    losses_ols = ols.train_on_positions(positions, num_epochs=1, gaussian_noise=noise_std)
                    final_loss = losses_ols[0] if losses_ols else 0.0
                
                # Evaluate all methods
                for method_name, method in [('SGD_1000', sgd_1000), ('SGD_2000', sgd_2000), ('OLS', ols)]:
                    with torch.no_grad():
                        grid_tensor = torch.FloatTensor(grid_coords).to(device)
                        grid_phi = method.phi(grid_tensor)
                        target = torch.matmul(grid_phi, method.theta).cpu().numpy()
                        
                        if method_name == 'OLS':
                            pred = torch.matmul(grid_phi, method.predictor_weights) + method.predictor_intercept
                            pred = pred.squeeze().cpu().numpy()
                        else:
                            pred = method.predictor(grid_phi).squeeze().cpu().numpy()
                        
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
                        if method_name == 'SGD_1000':
                            final_loss_val = losses[-1] if losses else float('nan')
                        elif method_name == 'SGD_2000':
                            final_loss_val = losses_2000[-1] if losses_2000 else float('nan')
                        else:
                            final_loss_val = final_loss
                        
                        all_results[noise_std][reg][method_name]['training_loss'].append(final_loss_val)
                        all_results[noise_std][reg][method_name]['pred_mean'].append(np.mean(pred))
                        all_results[noise_std][reg][method_name]['pred_std'].append(np.std(pred))
                        all_results[noise_std][reg][method_name]['uncertainty_mean'].append(np.mean(uncertainty))
                        all_results[noise_std][reg][method_name]['uncertainty_max'].append(np.max(uncertainty))
                        all_results[noise_std][reg][method_name]['norm_unc_mean'].append(np.mean(pred_vector))
                        all_results[noise_std][reg][method_name]['l2_distance'].append(l2_distance)
                        
                        # Store weights and biases (only for last run)
                        if run_idx == num_runs - 1:
                            if method_name == 'OLS':
                                w = method.predictor_weights.cpu().numpy()
                                b = method.predictor_intercept.item()
                            else:
                                w = method.predictor.weight.data.squeeze().cpu().numpy()
                                b = method.predictor.bias.data.item()
                            
                            weights_biases[noise_std][reg][method_name]['weights'] = w
                            weights_biases[noise_std][reg][method_name]['bias'] = b
                            all_results[noise_std][reg][method_name]['weights_l2'].append(np.linalg.norm(w))
                            all_results[noise_std][reg][method_name]['bias'].append(b)
    
    # Print comprehensive results
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS")
    print("=" * 100)
    
    for noise_std in noise_levels:
        print(f"\n{'='*100}")
        print(f"NOISE LEVEL: σ = {noise_std}")
        print(f"{'='*100}")
        
        for reg in regularization_levels:
            reg_str = f"Regularized (λ={reg})" if reg > 0 else "Unregularized"
            print(f"\n{reg_str}")
            print("-" * 100)
            print(f"{'Method':<12} {'Loss':<12} {'Pred Mean':<12} {'Pred Std':<12} {'Unc Mean':<12} {'Unc Max':<12} {'Norm Unc':<12} {'L2 Dist':<12}")
            print("-" * 100)
            
            for method_name in ['SGD_1000', 'SGD_2000', 'OLS']:
                results = all_results[noise_std][reg][method_name]
                print(f"{method_name:<12} "
                      f"{np.mean(results['training_loss']):<12.6f} "
                      f"{np.mean(results['pred_mean']):<12.6f} "
                      f"{np.mean(results['pred_std']):<12.6f} "
                      f"{np.mean(results['uncertainty_mean']):<12.6f} "
                      f"{np.mean(results['uncertainty_max']):<12.6f} "
                      f"{np.mean(results['norm_unc_mean']):<12.6f} "
                      f"{np.mean(results['l2_distance']):<12.6f}")
    
    # Print weights and biases comparison
    print("\n" + "=" * 100)
    print("WEIGHTS AND BIASES COMPARISON (from final run)")
    print("=" * 100)
    
    for noise_std in noise_levels:
        print(f"\n{'='*100}")
        print(f"NOISE LEVEL: σ = {noise_std}")
        print(f"{'='*100}")
        
        for reg in regularization_levels:
            reg_str = f"Regularized (λ={reg})" if reg > 0 else "Unregularized"
            print(f"\n{reg_str}")
            print("-" * 100)
            print(f"{'Method':<12} {'Weight L2':<15} {'Weight Mean':<15} {'Weight Std':<15} {'Bias':<15}")
            print("-" * 100)
            
            for method_name in ['SGD_1000', 'SGD_2000', 'OLS']:
                w = weights_biases[noise_std][reg][method_name]['weights']
                b = weights_biases[noise_std][reg][method_name]['bias']
                if w is not None:
                    print(f"{method_name:<12} "
                          f"{np.linalg.norm(w):<15.6f} "
                          f"{np.mean(w):<15.6f} "
                          f"{np.std(w):<15.6f} "
                          f"{b:<15.6f}")
            
            # Compute similarities
            if weights_biases[noise_std][reg]['SGD_2000']['weights'] is not None:
                w_sgd = weights_biases[noise_std][reg]['SGD_2000']['weights']
                w_ols = weights_biases[noise_std][reg]['OLS']['weights']
                if w_ols is not None:
                    cos_sim = np.dot(w_sgd, w_ols) / (np.linalg.norm(w_sgd) * np.linalg.norm(w_ols))
                    l2_diff = np.linalg.norm(w_sgd - w_ols)
                    print(f"\n  SGD_2000 vs OLS: Cosine similarity = {cos_sim:.6f}, L2 difference = {l2_diff:.6f}")
    
    return all_results, weights_biases

if __name__ == '__main__':
    results, weights = run_comprehensive_comparison(
        num_runs=10,  # Reduced from 20 to 10 for faster results
        num_samples=100,
        noise_levels=[0.0, 0.1, 0.5],
        regularization_levels=[0.0, 1e-4],
        lambda_reg=1e-4
    )

