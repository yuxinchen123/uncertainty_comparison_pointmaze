import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple

class ScalarEnsembleBootstrapRNDMethod:
    """
    Scalar Ensemble-Based RND with bootstrap sampling diversity.
    
    Key features:
    - K predictors (configurable)
    - Bootstrap sampling with replacement for each predictor
    - Scalar projection layer: output_dim -> 1 (trained)
    - Optional gaussian_noise (fixed per unique position, shared across predictors)
    - Uncertainty = STD of scalar predictions across K predictors
    """
    
    def __init__(self, hidden_dims, output_dim, K=10, device='cpu'):
        self.device = device
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.K = K
        
        # Build base architecture: [2] + hidden_dims + [output_dim]
        architecture = [2] + hidden_dims + [output_dim]
        
        # Single frozen target network (shared across all predictors)
        self.target_net = self._build_network(architecture).to(device)
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Scalar projection for target (frozen after initialization)
        self.target_scalar_proj = nn.Linear(output_dim, 1).to(device)
        for param in self.target_scalar_proj.parameters():
            param.requires_grad = False
            
        # K predictor networks (each trained independently)
        self.predictor_nets = []
        self.predictor_scalar_projs = []  # Scalar projection layers for each predictor
        self.optimizers = []
        self.criterion = nn.MSELoss()
        
        for k in range(K):
            predictor_net = self._build_network(architecture).to(device)
            scalar_proj = nn.Linear(output_dim, 1).to(device)
            # Combine predictor and scalar projection for optimizer
            optimizer = torch.optim.Adam(
                list(predictor_net.parameters()) + list(scalar_proj.parameters()), 
                lr=0.001
            )
            self.predictor_nets.append(predictor_net)
            self.predictor_scalar_projs.append(scalar_proj)
            self.optimizers.append(optimizer)
        
    def _build_network(self, architecture):
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture) - 2:  # No activation on output layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _bootstrap_sample(self, positions, sample_size=None):
        """Bootstrap sampling with replacement from the dataset."""
        if sample_size is None:
            sample_size = len(positions)
        
        indices = np.random.choice(len(positions), size=sample_size, replace=True)
        bootstrap_positions = positions[indices]
        
        # Show bootstrap sampling diversity
        unique_count = len(np.unique(bootstrap_positions, axis=0))
        print(f"    Bootstrap sample: {unique_count}/{sample_size} unique positions")
        
        return bootstrap_positions
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """
        Train K predictors using bootstrap sampling.
        Each predictor samples its own D_i with replacement from the data pool.
        """
        print(f"Training Scalar Ensemble Bootstrap RND with K={self.K} predictors on {len(positions)} positions")
        
        # Fix: Assign fixed noise per unique position at training start (before bootstrap sampling)
        noise_map = {}
        if gaussian_noise > 0:
            # Get unique positions from the original dataset
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
                unique_target_scalar = self.target_scalar_proj(unique_target_output).squeeze()  # [unique_N]
            
            # Assign fixed scalar noise to each unique position
            pos_tuple_list = [tuple(positions[i, :2].tolist()) for i in unique_indices]
            for idx, pos_tuple in enumerate(pos_tuple_list):
                noise_map[pos_tuple] = torch.randn(1).item() * gaussian_noise
        
        all_losses = []
        
        # Train each predictor independently with its own bootstrap sample
        for k in range(self.K):
            print(f"Training predictor {k+1}/{self.K}")
            
            # Bootstrap sample for this predictor
            bootstrap_positions = self._bootstrap_sample(positions)
            coords_tensor = torch.FloatTensor(bootstrap_positions[:, :2]).to(self.device)
            
            predictor_losses = []
            
            for epoch in range(num_epochs):
                self.optimizers[k].zero_grad()
                
                # Target output (scalar projection with fixed noise per position)
                with torch.no_grad():
                    target_output = self.target_net(coords_tensor)  # [N, output_dim]
                    target_scalar = self.target_scalar_proj(target_output).squeeze()  # [N]
                    if gaussian_noise > 0:
                        # Look up fixed noise for each position in bootstrap sample
                        noise_values = torch.zeros(len(bootstrap_positions), device=self.device)
                        for i, pos in enumerate(bootstrap_positions[:, :2]):
                            pos_tuple = tuple(pos.tolist())
                            noise_values[i] = noise_map[pos_tuple]
                        target_scalar += noise_values
                
                # Predictor output: scalar projection of predictor network
                pred_output = self.predictor_nets[k](coords_tensor)  # [N, output_dim]
                pred_scalar = self.predictor_scalar_projs[k](pred_output).squeeze()  # [N]
                
                # Loss: MSE between scalar predictions
                loss = self.criterion(pred_scalar, target_scalar)
                loss.backward()
                self.optimizers[k].step()
                
                predictor_losses.append(loss.item())
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Predictor {k+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            
            all_losses.append(predictor_losses)
        
        return all_losses
    
    def get_uncertainty(self, coordinates):
        """
        Calculate ensemble uncertainty as STD of scalar predictions across K predictors.
        Uncertainty = Std[pred_1(s), pred_2(s), ..., pred_K(s)]
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Get scalar predictions from all K predictors
            scalar_predictions = []
            
            for k in range(self.K):
                pred_output = self.predictor_nets[k](coordinates)  # [N, output_dim]
                pred_scalar = self.predictor_scalar_projs[k](pred_output).squeeze()  # [N]
                scalar_predictions.append(pred_scalar)
            
            # Stack: [K, N]
            scalar_predictions = torch.stack(scalar_predictions, dim=0)
            
            # Calculate STD across K predictors
            ensemble_uncertainty = torch.std(scalar_predictions, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()


class ScalarEnsembleBootstrapRNDLinearSGDMethod:
    """
    Scalar Ensemble-Based RND-Linear (SGD) with bootstrap sampling diversity.
    
    Key features:
    - K predictors (configurable)
    - Bootstrap sampling with replacement for each predictor
    - Output already scalar, use directly
    - Optional gaussian_noise (fixed per unique position, shared across predictors)
    - Regularization support (L2 regularization / weight decay)
    - Uncertainty = STD of scalar predictions across K predictors
    """
    
    def __init__(self, feature_dim, K=10, device='cpu', phi_weights=None, regularization=1e-2):
        self.device = device
        self.feature_dim = feature_dim
        self.K = K
        self.regularization = regularization
        
        # Shared φ(s): 2D -> feature_dim (frozen)
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False
            
        # Frozen target vector θ̂ (randomly initialized and kept frozen)
        self.theta_hat = torch.randn(feature_dim).to(device)
        
        # K predictor networks (each trained independently with SGD)
        self.predictor_nets = []
        self.optimizers = []
        self.criterion = nn.MSELoss()
        
        for k in range(K):
            # Each predictor: feature_dim -> 1 (already scalar)
            predictor_net = nn.Linear(feature_dim, 1).to(device)
            optimizer = torch.optim.Adam(predictor_net.parameters(), lr=0.01, weight_decay=regularization)
            self.predictor_nets.append(predictor_net)
            self.optimizers.append(optimizer)
    
    def _bootstrap_sample(self, positions, sample_size=None):
        """Bootstrap sampling with replacement from the dataset."""
        if sample_size is None:
            sample_size = len(positions)
        
        indices = np.random.choice(len(positions), size=sample_size, replace=True)
        bootstrap_positions = positions[indices]
        
        # Show bootstrap sampling diversity
        unique_count = len(np.unique(bootstrap_positions, axis=0))
        print(f"    Bootstrap sample: {unique_count}/{sample_size} unique positions")
        
        return bootstrap_positions
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """
        Train K predictors using bootstrap sampling and SGD.
        Each predictor samples its own D_i with replacement from the data pool.
        """
        print(f"Training Scalar Ensemble Bootstrap RND-Linear (SGD) with K={self.K} predictors on {len(positions)} positions")
        
        # Fix: Assign fixed noise per unique position at training start (before bootstrap sampling)
        noise_map = {}
        if gaussian_noise > 0:
            # Get unique positions from the original dataset
            unique_positions = {}
            for i, pos in enumerate(positions[:, :2]):
                pos_tuple = tuple(pos.tolist())
                if pos_tuple not in unique_positions:
                    unique_positions[pos_tuple] = i
            
            # Assign fixed scalar noise to each unique position
            for pos_tuple in unique_positions.keys():
                noise_map[pos_tuple] = torch.randn(1).item() * gaussian_noise
        
        all_losses = []
        
        # Train each predictor independently with its own bootstrap sample
        for k in range(self.K):
            print(f"Training predictor {k+1}/{self.K}")
            
            # Bootstrap sample for this predictor
            bootstrap_positions = self._bootstrap_sample(positions)
            coords_tensor = torch.FloatTensor(bootstrap_positions[:, :2]).to(self.device)
            
            predictor_losses = []
            
            for epoch in range(num_epochs):
                self.optimizers[k].zero_grad()
                
                # Compute features φ(s)
                phi_output = self.phi(coords_tensor)  # [N, feature_dim]
                
                # Target: φ(s)ᵀθ̂ (with fixed noise per position)
                with torch.no_grad():
                    target_output = torch.matmul(phi_output, self.theta_hat)  # [N]
                    if gaussian_noise > 0:
                        # Look up fixed noise for each position in bootstrap sample
                        noise_values = torch.zeros(len(bootstrap_positions), device=self.device)
                        for i, pos in enumerate(bootstrap_positions[:, :2]):
                            pos_tuple = tuple(pos.tolist())
                            noise_values[i] = noise_map[pos_tuple]
                        target_output += noise_values
                
                # Predictor output (already scalar)
                pred_output = self.predictor_nets[k](phi_output).squeeze()  # [N]
                
                # Loss
                loss = self.criterion(pred_output, target_output)
                loss.backward()
                self.optimizers[k].step()
                
                predictor_losses.append(loss.item())
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Predictor {k+1}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            
            all_losses.append(predictor_losses)
        
        return all_losses
    
    def get_uncertainty(self, coordinates):
        """
        Calculate ensemble uncertainty as STD of scalar predictions across K predictors.
        Uncertainty = Std[pred_1(s), pred_2(s), ..., pred_K(s)]
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Compute features φ(s)
            phi_output = self.phi(coordinates)  # [N, feature_dim]
            
            # Get scalar predictions from all K predictors
            scalar_predictions = []
            
            for k in range(self.K):
                pred_scalar = self.predictor_nets[k](phi_output).squeeze()  # [N]
                scalar_predictions.append(pred_scalar)
            
            # Stack: [K, N]
            scalar_predictions = torch.stack(scalar_predictions, dim=0)
            
            # Calculate STD across K predictors
            ensemble_uncertainty = torch.std(scalar_predictions, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()


class ScalarEnsembleBootstrapRNDLinearLSMethod:
    """
    Scalar Ensemble-Based RND-Linear (LS) with bootstrap sampling diversity.
    
    Key features:
    - K predictors (configurable)
    - Bootstrap sampling with replacement for each predictor
    - Least squares fitting for each predictor
    - Output already scalar, use directly
    - Optional gaussian_noise (fixed per unique position, shared across predictors)
    - Regularization support (ridge regression)
    - Uncertainty = STD of scalar predictions across K predictors
    """
    
    def __init__(self, feature_dim, K=10, device='cpu', phi_weights=None, regularization=1e-2):
        self.device = device
        self.feature_dim = feature_dim
        self.K = K
        self.regularization = regularization
        
        # Shared φ(s): 2D -> feature_dim (frozen)
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False
            
        # Frozen target vector θ̂ (randomly initialized and kept frozen)
        self.theta_hat = torch.randn(feature_dim).to(device)
        
        # K predictor weights (will be set via least squares)
        self.predictor_weights = [None] * K
        self.predictor_intercepts = [None] * K
    
    def _bootstrap_sample(self, positions, sample_size=None):
        """Bootstrap sampling with replacement from the dataset."""
        if sample_size is None:
            sample_size = len(positions)
        
        indices = np.random.choice(len(positions), size=sample_size, replace=True)
        bootstrap_positions = positions[indices]
        
        # Show bootstrap sampling diversity
        unique_count = len(np.unique(bootstrap_positions, axis=0))
        print(f"    Bootstrap sample: {unique_count}/{sample_size} unique positions")
        
        return bootstrap_positions
    
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
        return intercept, coefs
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """
        Train K predictors using bootstrap sampling and least squares.
        Each predictor samples its own D_i with replacement from the data pool.
        """
        print(f"Training Scalar Ensemble Bootstrap RND-Linear (LS) with K={self.K} predictors on {len(positions)} positions")
        
        # Fix: Assign fixed noise per unique position at training start (before bootstrap sampling)
        noise_map = {}
        if gaussian_noise > 0:
            # Get unique positions from the original dataset
            unique_positions = {}
            for i, pos in enumerate(positions[:, :2]):
                pos_tuple = tuple(pos.tolist())
                if pos_tuple not in unique_positions:
                    unique_positions[pos_tuple] = i
            
            # Assign fixed scalar noise to each unique position
            for pos_tuple in unique_positions.keys():
                noise_map[pos_tuple] = torch.randn(1).item() * gaussian_noise
        
        all_losses = []
        
        # Train each predictor independently with its own bootstrap sample
        for k in range(self.K):
            print(f"Training predictor {k+1}/{self.K}")
            
            # Bootstrap sample for this predictor
            bootstrap_positions = self._bootstrap_sample(positions)
            coords_tensor = torch.FloatTensor(bootstrap_positions[:, :2]).to(self.device)
            
            # Compute features and targets
            with torch.no_grad():
                phi_output = self.phi(coords_tensor)  # [N, feature_dim]
                target_output = torch.matmul(phi_output, self.theta_hat)  # [N]
                
                if gaussian_noise > 0:
                    # Look up fixed noise for each position in bootstrap sample
                    noise_values = torch.zeros(len(bootstrap_positions), device=self.device)
                    for i, pos in enumerate(bootstrap_positions[:, :2]):
                        pos_tuple = tuple(pos.tolist())
                        noise_values[i] = noise_map[pos_tuple]
                    target_output += noise_values
            
            # Convert to numpy for least squares
            X = phi_output.cpu().numpy()  # [N, feature_dim]
            y = target_output.cpu().numpy()  # [N]
            
            # Fit: predictor(φ(s)) = intercept + φ(s)ᵀ * weights ≈ φ(s)ᵀθ̂
            intercept, weights = self.least_squares_fit(X, y, add_intercept=True)
            
            # Store weights
            self.predictor_intercepts[k] = torch.FloatTensor([intercept]).to(self.device)
            self.predictor_weights[k] = torch.FloatTensor(weights).to(self.device)
            
            # Compute final loss for logging
            pred_output = torch.matmul(phi_output, self.predictor_weights[k]) + self.predictor_intercepts[k]
            final_loss = torch.mean((pred_output.squeeze() - target_output) ** 2).item()
            
            print(f"  Predictor {k+1} least squares fit complete. Final MSE: {final_loss:.6f}")
            
            # Return dummy loss history for compatibility
            all_losses.append([final_loss] * num_epochs)
        
        return all_losses
    
    def get_uncertainty(self, coordinates):
        """
        Calculate ensemble uncertainty as STD of scalar predictions across K predictors.
        Uncertainty = Std[pred_1(s), pred_2(s), ..., pred_K(s)]
        """
        if any(w is None for w in self.predictor_weights):
            raise ValueError("Model not trained yet. Call train_on_positions first.")
        
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Compute features φ(s)
            phi_output = self.phi(coordinates)  # [N, feature_dim]
            
            # Get scalar predictions from all K predictors
            scalar_predictions = []
            
            for k in range(self.K):
                pred_scalar = torch.matmul(phi_output, self.predictor_weights[k]) + self.predictor_intercepts[k]
                pred_scalar = pred_scalar.squeeze()  # [N]
                scalar_predictions.append(pred_scalar)
            
            # Stack: [K, N]
            scalar_predictions = torch.stack(scalar_predictions, dim=0)
            
            # Calculate STD across K predictors
            ensemble_uncertainty = torch.std(scalar_predictions, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()

