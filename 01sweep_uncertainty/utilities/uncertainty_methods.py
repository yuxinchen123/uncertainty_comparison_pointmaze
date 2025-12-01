import torch
import torch.nn as nn
import numpy as np

class RNDMethod:
    def __init__(self, hidden_dims, output_dim, device='cpu'):
        self.device = device
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build architecture: [2] + hidden_dims + [output_dim]
        architecture = [2] + hidden_dims + [output_dim]
        
        # Target network (fixed)
        self.target_net = self._build_network(architecture).to(device)
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        # Predictor network (trainable)
        self.predictor_net = self._build_network(architecture).to(device)
        self.optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def _build_network(self, architecture):
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture) - 2:  # No activation on output layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """Train predictor to match target - use all provided positions"""
        print(f"Training RND on {len(positions)} positions for {num_epochs} epochs")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        losses = []
        
        # Fix: Assign fixed noise per unique position at training start
        noise_map = {}
        if gaussian_noise > 0:
            # Get unique positions (as tuples for dictionary keys)
            unique_positions = {}
            for i, pos in enumerate(positions[:, :2]):
                pos_tuple = tuple(pos.tolist())
                if pos_tuple not in unique_positions:
                    unique_positions[pos_tuple] = i
            
            # Get target output for unique positions to determine noise shape
            unique_indices = list(unique_positions.values())
            unique_coords = torch.FloatTensor(positions[unique_indices, :2]).to(self.device)
            with torch.no_grad():
                unique_target_output = self.target_net(unique_coords)  # [unique_N, output_dim]
            
            # Assign fixed noise to each unique position (in the same order as unique_indices)
            pos_tuple_list = [tuple(positions[i, :2].tolist()) for i in unique_indices]
            for idx, pos_tuple in enumerate(pos_tuple_list):
                noise_map[pos_tuple] = torch.randn_like(unique_target_output[idx:idx+1]) * gaussian_noise
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Target output (with fixed noise per position)
            with torch.no_grad():
                target_output = self.target_net(coords_tensor)
                if gaussian_noise > 0:
                    # Look up fixed noise for each position
                    noise_values = []
                    for pos in positions[:, :2]:
                        pos_tuple = tuple(pos.tolist())
                        noise_values.append(noise_map[pos_tuple])
                    noise = torch.cat(noise_values, dim=0)  # [N, output_dim]
                    target_output += noise
            
            # Predictor output
            pred_output = self.predictor_net(coords_tensor)
            
            # Loss
            loss = self.criterion(pred_output, target_output)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        return losses
    
    def get_uncertainty(self, coordinates):
        """Get uncertainty as L1 distance between target and predictor"""
        self.predictor_net.eval()
        with torch.no_grad():
            if isinstance(coordinates, np.ndarray):
                coordinates = torch.FloatTensor(coordinates).to(self.device)
            
            target_output = self.target_net(coordinates)
            pred_output = self.predictor_net(coordinates)
            
            # L1 uncertainty: mean absolute difference across output dimensions
            uncertainty = torch.abs(target_output - pred_output).mean(dim=1)
            
            return uncertainty.cpu().numpy()

# RND Linear with SGD Training
class RNDLinearSGDMethod:
    def __init__(self, feature_dim, device='cpu', phi_weights=None, theta_seed=42, predictor_seed=None, regularization=1e-2, lr_decay_step_size=250, lr_decay_gamma=0.5):
        self.device = device
        self.feature_dim = feature_dim
        self.regularization = regularization
        self.lr_decay_step_size = lr_decay_step_size
        self.lr_decay_gamma = lr_decay_gamma
        
        # Shared φ(s): 2D -> feature_dim
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False
            
        # Target θ vector (deterministic based on seed)
        torch.manual_seed(theta_seed)
        self.theta = torch.randn(feature_dim).to(device)
        
        # Predictor: feature_dim -> 1 (seed initialization if provided)
        if predictor_seed is not None:
            torch.manual_seed(predictor_seed)
        self.predictor = nn.Linear(feature_dim, 1).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.01, weight_decay=regularization)
        # Learning rate scheduler: decay by gamma every step_size epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_gamma)
        self.criterion = nn.MSELoss()
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0, noise_seed=None):
        """Train predictor to match φ(s)ᵀθ using SGD - use all provided positions"""
        print(f"Training RND-Linear (SGD) on {len(positions)} positions for {num_epochs} epochs")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        
        # Fix: Assign fixed noise per unique position at training start
        noise_map = {}
        if gaussian_noise > 0:
            # Seed noise generation deterministically if seed provided
            if noise_seed is not None:
                torch.manual_seed(noise_seed)
            
            # Get unique positions (as tuples for dictionary keys) - sort for deterministic order
            unique_positions_list = []
            seen = set()
            for i, pos in enumerate(positions[:, :2]):
                pos_tuple = tuple(pos.tolist())
                if pos_tuple not in seen:
                    unique_positions_list.append(pos_tuple)
                    seen.add(pos_tuple)
            
            # Sort for deterministic iteration order
            unique_positions_list.sort()
            
            # Assign fixed scalar noise to each unique position
            for pos_tuple in unique_positions_list:
                noise_map[pos_tuple] = torch.randn(1).item() * gaussian_noise
        
        losses = []
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Compute features φ(s)
            phi_output = self.phi(coords_tensor)
            
            # Target: φ(s)ᵀθ (with fixed noise per position)
            with torch.no_grad():
                target_output = torch.matmul(phi_output, self.theta)
                if gaussian_noise > 0:
                    # Look up fixed noise for each position
                    noise_values = torch.zeros(len(positions), device=self.device)
                    for i, pos in enumerate(positions[:, :2]):
                        pos_tuple = tuple(pos.tolist())
                        noise_values[i] = noise_map[pos_tuple]
                    target_output += noise_values
            
            # Predictor output
            pred_output = self.predictor(phi_output).squeeze()
            
            # Loss
            loss = self.criterion(pred_output, target_output)
            loss.backward()
            self.optimizer.step()
            # Step learning rate scheduler
            self.scheduler.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
        
        return losses
    
    def get_uncertainty(self, coordinates):
        """Get uncertainty as |φ(s)ᵀθ - predictor(φ(s))|"""
        self.predictor.eval()
        with torch.no_grad():
            if isinstance(coordinates, np.ndarray):
                coordinates = torch.FloatTensor(coordinates).to(self.device)
            
            phi_output = self.phi(coordinates)
            target_output = torch.matmul(phi_output, self.theta)
            pred_output = self.predictor(phi_output).squeeze()
            
            uncertainty = torch.abs(target_output - pred_output)
            return uncertainty.cpu().numpy()

# RND Linear with Least Square Fit
class RNDLinearLSMethod:
    def __init__(self, feature_dim, device='cpu', phi_weights=None, theta_seed=42, regularization=1e-2):
        self.device = device
        self.feature_dim = feature_dim
        self.regularization = regularization
        
        # Shared φ(s): 2D -> feature_dim
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False
            
        # Target θ vector (deterministic based on seed)
        torch.manual_seed(theta_seed)
        self.theta = torch.randn(feature_dim).to(device)
        
        # Predictor weights (will be set via least squares)
        self.predictor_weights = None
        self.predictor_intercept = None
    
    def least_squares_fit(self, X, y, add_intercept=True):
        """Ridge regression (regularized least squares): min ||X_aug w - y||^2 + λ||w||^2"""
        if y.ndim == 1:
            y = y[:, None]
        if add_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X
        
        # Regularized least squares: (X^T X + λI)^-1 X^T y
        XTX = X_aug.T @ X_aug
        XTX += np.eye(XTX.shape[0]) * self.regularization
        
        try:
            w = np.linalg.solve(XTX, X_aug.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        
        w = w.squeeze()
        intercept = w[0] if add_intercept else 0.0
        coefs = w[1:] if add_intercept else w
        
        # Compute residuals for compatibility
        pred = X_aug @ w
        residuals = np.sum((y - pred) ** 2)
        return intercept, coefs, residuals
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0, noise_seed=None):
        """Fit predictor to match φ(s)ᵀθ using closed-form least squares"""
        print(f"Training RND-Linear on {len(positions)} positions (least squares fit)")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        
        # Fix: Assign fixed noise per unique position at training start
        noise_map = {}
        if gaussian_noise > 0:
            # Seed noise generation deterministically if seed provided
            if noise_seed is not None:
                torch.manual_seed(noise_seed)
            
            # Get unique positions (as tuples for dictionary keys) - sort for deterministic order
            unique_positions_list = []
            seen = set()
            for i, pos in enumerate(positions[:, :2]):
                pos_tuple = tuple(pos.tolist())
                if pos_tuple not in seen:
                    unique_positions_list.append(pos_tuple)
                    seen.add(pos_tuple)
            
            # Sort for deterministic iteration order
            unique_positions_list.sort()
            
            # Assign fixed scalar noise to each unique position
            for pos_tuple in unique_positions_list:
                noise_map[pos_tuple] = torch.randn(1).item() * gaussian_noise
        
        # Compute features and targets
        with torch.no_grad():
            phi_output = self.phi(coords_tensor)  # [N, feature_dim]
            target_output = torch.matmul(phi_output, self.theta)  # [N]
            
            if gaussian_noise > 0:
                # Look up fixed noise for each position
                noise_values = torch.zeros(len(positions), device=self.device)
                for i, pos in enumerate(positions[:, :2]):
                    pos_tuple = tuple(pos.tolist())
                    noise_values[i] = noise_map[pos_tuple]
                target_output += noise_values
        
        # Convert to numpy for least squares
        X = phi_output.cpu().numpy()  # [N, feature_dim]
        y = target_output.cpu().numpy()  # [N]
        
        # Fit: predictor(φ(s)) = intercept + φ(s)ᵀ * weights ≈ φ(s)ᵀθ
        self.predictor_intercept, self.predictor_weights, residuals = self.least_squares_fit(X, y, add_intercept=True)
        
        # Convert back to torch tensors
        self.predictor_weights = torch.FloatTensor(self.predictor_weights).to(self.device)
        self.predictor_intercept = torch.FloatTensor([self.predictor_intercept]).to(self.device)
        
        # Compute final loss for logging
        pred_output = torch.matmul(phi_output, self.predictor_weights) + self.predictor_intercept
        final_loss = torch.mean((pred_output - target_output) ** 2).item()
        
        print(f"  Least squares fit complete. Final MSE: {final_loss:.6f}")
        if residuals is not None:
            if isinstance(residuals, (list, tuple, np.ndarray)) and len(residuals) > 0:
                print(f"  Residual sum of squares: {residuals[0]:.6f}")
            elif isinstance(residuals, (int, float, np.number)):
                print(f"  Residual sum of squares: {residuals:.6f}")
        
        # Return dummy loss history for compatibility
        return [final_loss] * num_epochs
    
    def get_uncertainty(self, coordinates):
        """Get uncertainty as |φ(s)ᵀθ - predictor(φ(s))|"""
        if self.predictor_weights is None:
            raise ValueError("Model not trained yet. Call train_on_positions first.")
            
        with torch.no_grad():
            if isinstance(coordinates, np.ndarray):
                coordinates = torch.FloatTensor(coordinates).to(self.device)
            
            phi_output = self.phi(coordinates)
            target_output = torch.matmul(phi_output, self.theta)
            pred_output = torch.matmul(phi_output, self.predictor_weights) + self.predictor_intercept
            
            uncertainty = torch.abs(target_output - pred_output.squeeze())
            return uncertainty.cpu().numpy()

# Keep RNDLinearMethod as an alias for RNDLinearLSMethod for backward compatibility
RNDLinearMethod = RNDLinearLSMethod
        
class EllipticalBonusMethod:
    def __init__(self, feature_dim, device='cpu', phi_weights=None, regularization=1e-6):
        self.device = device
        self.feature_dim = feature_dim
        self.regularization = regularization

        # Shared φ(s): 2D -> feature_dim (frozen)
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False

        # Initialize Λ and its inverse with regularization
        self.covariance = torch.eye(feature_dim, device=device) * regularization
        self.covariance_inv = torch.eye(feature_dim, device=device) / regularization

        # Normalization stats (computed from data on first update)
        self.obs_mean = None
        self.obs_std = None

    def _normalize_observations(self, observations):
        if self.obs_mean is None:
            self.obs_mean = torch.mean(observations, dim=0, keepdim=True)
            self.obs_std = torch.std(observations, dim=0, keepdim=True) + 1e-8
        return (observations - self.obs_mean) / self.obs_std

    def _compute_features(self, observations):
        self.phi.eval()
        norm_obs = self._normalize_observations(observations)
        with torch.no_grad():
            return self.phi(norm_obs)

    def update_covariance_from_positions(self, positions):
        """
        Correct covariance calculation:
        Λ = (1/n) Σ φ(s_i) φ(s_i)^T + regularization * I
        Compute in batches for memory efficiency, then invert (with fallback).
        """
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)

        # compute normalization stats and features in batches
        features_list = []
        batch_size = 1000
        for i in range(0, len(coords_tensor), batch_size):
            batch = coords_tensor[i:i+batch_size]
            batch_feats = self._compute_features(batch)
            features_list.append(batch_feats)

        all_features = torch.cat(features_list, dim=0)  # [n_samples, feature_dim]
        n_samples = all_features.shape[0]
        if n_samples == 0:
            return

        # accumulate sum of outer-products in batches to form (unnormalized) covariance
        accumulated = torch.zeros(self.feature_dim, self.feature_dim, device=self.device)
        batch_size_cov = 5000
        for i in range(0, n_samples, batch_size_cov):
            batch_feats = all_features[i:i+batch_size_cov]  # [b, d]
            accumulated += batch_feats.T @ batch_feats  # [d, d]

        # normalize and add regularization
        Lambda = accumulated / float(n_samples)
        Lambda += torch.eye(self.feature_dim, device=self.device) * self.regularization
        self.covariance = Lambda

        # invert with numerical fallback
        try:
            self.covariance_inv = torch.linalg.inv(self.covariance)
        except Exception:
            self.covariance_inv = torch.pinverse(self.covariance)

        # optional diagnostics (kept minimal)
        try:
            eigvals = torch.linalg.eigvals(self.covariance).real
            cond = (torch.max(eigvals) / torch.min(eigvals)).item()
            print(f"  Covariance updated (n={n_samples}), condition number: {cond:.2e}")
        except Exception:
            print("  Covariance updated (condition number unavailable)")

    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """Train/update covariance from positions (analytical; no optimizer)"""
        print(f"Training Elliptical Bonus on {len(positions)} positions (updating covariance)")
        # compute normalization from first chunk to stabilise features
        coords = torch.FloatTensor(positions[:, :2]).to(self.device)
        _ = self._normalize_observations(coords[:1000])
        self.update_covariance_from_positions(positions)
        return [0.0] * num_epochs

    def get_uncertainty(self, coordinates):
        """Elliptical bonus: sqrt(φ(s)^T Σ^{-1} φ(s))"""
        with torch.no_grad():
            if isinstance(coordinates, np.ndarray):
                coordinates = torch.FloatTensor(coordinates).to(self.device)

            phi_output = self._compute_features(coordinates)  # [N, feature_dim]
            tmp = phi_output @ self.covariance_inv  # [N, feature_dim]
            quadratic_form = torch.sum(tmp * phi_output, dim=1)
            uncertainty = torch.sqrt(torch.clamp(quadratic_form, min=1e-8))
            return uncertainty.cpu().numpy()