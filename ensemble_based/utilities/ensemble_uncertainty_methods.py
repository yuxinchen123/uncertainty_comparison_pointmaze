import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple

class EnsembleRNDMethod:
    """
    Ensemble-Based RND implementation following the project proposal.
    
    Key features:
    - K=10 predictors (configurable)
    - Bootstrap sampling with replacement for each predictor
    - Frozen target network shared across all predictors
    - Ensemble uncertainty = std deviation across K predictors
    """
    
    def __init__(self, hidden_dims, output_dim, K=10, device='cpu'):
        self.device = device
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.K = K
        
        # Build architecture: [2] + hidden_dims + [output_dim]
        architecture = [2] + hidden_dims + [output_dim]
        
        # Single frozen target network (shared across all predictors)
        self.target_net = self._build_network(architecture).to(device)
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        # K predictor networks (each trained independently)
        self.predictor_nets = []
        self.optimizers = []
        self.criterion = nn.MSELoss()
        
        for k in range(K):
            predictor_net = self._build_network(architecture).to(device)
            optimizer = torch.optim.Adam(predictor_net.parameters(), lr=0.001)
            self.predictor_nets.append(predictor_net)
            self.optimizers.append(optimizer)
        
    def _build_network(self, architecture):
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture) - 2:  # No activation on output layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _bootstrap_sample(self, positions, sample_size=None):
        """
        Bootstrap sampling with replacement from the dataset.
        Each predictor gets its own bootstrap sample.
        """
        if sample_size is None:
            sample_size = len(positions)
        
        # Sample with replacement
        indices = np.random.choice(len(positions), size=sample_size, replace=True)
        return positions[indices]
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """
        Train K predictors using bootstrap sampling.
        Each predictor samples its own D_i with replacement from the 10k data pool.
        """
        print(f"Training Ensemble RND with K={self.K} predictors on {len(positions)} positions")
        print(f"Each predictor will bootstrap sample {len(positions)} positions with replacement")
        
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
                
                # Target output (with noise)
                with torch.no_grad():
                    target_output = self.target_net(coords_tensor)
                    if gaussian_noise > 0:
                        noise = torch.randn_like(target_output) * gaussian_noise
                        target_output += noise
                
                # Predictor output
                pred_output = self.predictor_nets[k](coords_tensor)
                
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
        Calculate ensemble uncertainty as standard deviation across K predictors.
        r_int(s) = Std_{i=1,...,K}[e_i(s)]
        where e_i(s) = ||f_theta_i(s) - f_hat_theta(s)||
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        # Get target output (same for all predictors)
        with torch.no_grad():
            target_output = self.target_net(coordinates)  # [N, output_dim]
            
            # Calculate per-model errors for each predictor
            per_model_errors = []
            
            for k in range(self.K):
                pred_output = self.predictor_nets[k](coordinates)  # [N, output_dim]
                
                # L1 distance between target and predictor (per sample)
                error = torch.abs(target_output - pred_output).mean(dim=1)  # [N]
                per_model_errors.append(error)
            
            # Stack errors: [K, N]
            per_model_errors = torch.stack(per_model_errors, dim=0)
            
            # Calculate ensemble uncertainty as standard deviation across K predictors
            ensemble_uncertainty = torch.std(per_model_errors, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()


class EnsembleRNDLinearSGDMethod:
    """
    Ensemble-Based RND-Linear (SGD) implementation.
    
    Key features:
    - K=10 predictors (configurable)
    - Bootstrap sampling with replacement for each predictor
    - Frozen target vector θ̂ shared across all predictors
    - SGD training for each predictor
    - Ensemble uncertainty = std deviation across K predictors
    """
    
    def __init__(self, feature_dim, K=10, device='cpu', phi_weights=None):
        self.device = device
        self.feature_dim = feature_dim
        self.K = K
        
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
            # Each predictor: feature_dim -> 1
            predictor_net = nn.Linear(feature_dim, 1).to(device)
            optimizer = torch.optim.Adam(predictor_net.parameters(), lr=0.001)
            self.predictor_nets.append(predictor_net)
            self.optimizers.append(optimizer)
    
    def _bootstrap_sample(self, positions, sample_size=None):
        """Bootstrap sampling with replacement from the dataset."""
        if sample_size is None:
            sample_size = len(positions)
        
        indices = np.random.choice(len(positions), size=sample_size, replace=True)
        return positions[indices]
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """
        Train K predictors using bootstrap sampling and SGD.
        Each predictor samples its own D_i with replacement from the 10k data pool.
        """
        print(f"Training Ensemble RND-Linear (SGD) with K={self.K} predictors on {len(positions)} positions")
        print(f"Each predictor will bootstrap sample {len(positions)} positions with replacement")
        
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
                
                # Target: φ(s)ᵀθ̂ (with noise)
                with torch.no_grad():
                    target_output = torch.matmul(phi_output, self.theta_hat)  # [N]
                    if gaussian_noise > 0:
                        noise = torch.randn_like(target_output) * gaussian_noise
                        target_output += noise
                
                # Predictor output
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
        Calculate ensemble uncertainty as standard deviation across K predictors.
        r_int(s) = Std_{i=1,...,K}[e_i(s)]
        where e_i(s) = |⟨φ(s), θ_i⟩ - ⟨φ(s), θ̂⟩|
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Compute features φ(s)
            phi_output = self.phi(coordinates)  # [N, feature_dim]
            
            # Target output: φ(s)ᵀθ̂
            target_output = torch.matmul(phi_output, self.theta_hat)  # [N]
            
            # Calculate per-model errors for each predictor
            per_model_errors = []
            
            for k in range(self.K):
                pred_output = self.predictor_nets[k](phi_output).squeeze()  # [N]
                
                # Absolute difference between predictor and target
                error = torch.abs(target_output - pred_output)  # [N]
                per_model_errors.append(error)
            
            # Stack errors: [K, N]
            per_model_errors = torch.stack(per_model_errors, dim=0)
            
            # Calculate ensemble uncertainty as standard deviation across K predictors
            ensemble_uncertainty = torch.std(per_model_errors, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()


class EnsembleRNDLinearLSMethod:
    """
    Ensemble-Based RND-Linear (LS) implementation using least squares.
    
    Key features:
    - K=10 predictors (configurable)
    - Bootstrap sampling with replacement for each predictor
    - Frozen target vector θ̂ shared across all predictors
    - Least squares fitting for each predictor
    - Ensemble uncertainty = std deviation across K predictors
    """
    
    def __init__(self, feature_dim, K=10, device='cpu', phi_weights=None, regularization=1e-6):
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
        return positions[indices]
    
    def least_squares_fit(self, X, y, add_intercept=True):
        """Direct least squares solution with regularization."""
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
        Each predictor samples its own D_i with replacement from the 10k data pool.
        """
        print(f"Training Ensemble RND-Linear (LS) with K={self.K} predictors on {len(positions)} positions")
        print(f"Each predictor will bootstrap sample {len(positions)} positions with replacement")
        
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
                    noise = torch.randn_like(target_output) * gaussian_noise
                    target_output += noise
            
            # Convert to numpy for least squares
            X = phi_output.cpu().numpy()  # [N, feature_dim]
            y = target_output.cpu().numpy()  # [N]
            
            # Fit: predictor(φ(s)) = intercept + φ(s)ᵀ * weights ≈ φ(s)ᵀθ̂
            intercept, weights, _ = self.least_squares_fit(X, y, add_intercept=True)
            
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
        Calculate ensemble uncertainty as standard deviation across K predictors.
        r_int(s) = Std_{i=1,...,K}[e_i(s)]
        where e_i(s) = |⟨φ(s), θ_i⟩ - ⟨φ(s), θ̂⟩|
        """
        if any(w is None for w in self.predictor_weights):
            raise ValueError("Model not trained yet. Call train_on_positions first.")
        
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Compute features φ(s)
            phi_output = self.phi(coordinates)  # [N, feature_dim]
            
            # Target output: φ(s)ᵀθ̂
            target_output = torch.matmul(phi_output, self.theta_hat)  # [N]
            
            # Calculate per-model errors for each predictor
            per_model_errors = []
            
            for k in range(self.K):
                pred_output = torch.matmul(phi_output, self.predictor_weights[k]) + self.predictor_intercepts[k]
                pred_output = pred_output.squeeze()  # [N]
                
                # Absolute difference between predictor and target
                error = torch.abs(target_output - pred_output)  # [N]
                per_model_errors.append(error)
            
            # Stack errors: [K, N]
            per_model_errors = torch.stack(per_model_errors, dim=0)
            
            # Calculate ensemble uncertainty as standard deviation across K predictors
            ensemble_uncertainty = torch.std(per_model_errors, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()

