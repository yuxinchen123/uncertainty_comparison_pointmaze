import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple

class ScalarEnsembleRNDMethod:
    """
    Scalar Ensemble-Based RND with noise-based diversity.
    
    Key features:
    - K predictors (configurable)
    - Same sub-dataset for all K heads (no bootstrap sampling)
    - Different Gaussian noise per head: w_t^i ~ N(0, noise_sigma^2) for each position
    - Scalar projection layer: output_dim -> 1 (trained)
    - Uncertainty = STD of scalar predictions across K heads
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
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, noise_sigma=0.0):
        """
        Train K predictors using same positions but different noise per head.
        For head i: Sample W_i = {w_t^i ~ N(0, noise_sigma^2)} for each position t
        Training target for head i: scalar_proj(target_net(x_t)) + w_t^i
        """
        print(f"Training Scalar Ensemble RND with K={self.K} predictors on {len(positions)} positions")
        print(f"Using same positions for all heads, noise_sigma={noise_sigma}")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        N = len(positions)
        
        # Compute clean target scalar once (same for all heads)
        with torch.no_grad():
            target_output = self.target_net(coords_tensor)  # [N, output_dim]
            target_scalar = self.target_scalar_proj(target_output).squeeze()  # [N]
        
        all_losses = []
        
        # Train each predictor independently with different noise
        for k in range(self.K):
            print(f"Training predictor {k+1}/{self.K}")
            
            # Sample noise for this head: W_k = {w_t^k ~ N(0, noise_sigma^2)}
            if noise_sigma > 0:
                noise = torch.randn(N, device=self.device) * noise_sigma  # [N]
            else:
                noise = torch.zeros(N, device=self.device)
            
            # Target for this head: clean_target_scalar + noise
            target_with_noise = target_scalar + noise
            
            predictor_losses = []
            
            for epoch in range(num_epochs):
                self.optimizers[k].zero_grad()
                
                # Predictor output: scalar projection of predictor network
                pred_output = self.predictor_nets[k](coords_tensor)  # [N, output_dim]
                pred_scalar = self.predictor_scalar_projs[k](pred_output).squeeze()  # [N]
                
                # Loss: MSE between scalar predictions
                loss = self.criterion(pred_scalar, target_with_noise)
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


class ScalarEnsembleRNDLinearSGDMethod:
    """
    Scalar Ensemble-Based RND-Linear (SGD) with noise-based diversity.
    
    Key features:
    - K predictors (configurable)
    - Same sub-dataset for all K heads (no bootstrap sampling)
    - Different Gaussian noise per head: w_t^i ~ N(0, noise_sigma^2)
    - Output already scalar, use directly
    - Uncertainty = STD of scalar predictions across K heads
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
            # Each predictor: feature_dim -> 1 (already scalar)
            predictor_net = nn.Linear(feature_dim, 1).to(device)
            # Use higher learning rate and weight decay for better convergence on ill-conditioned problems
            optimizer = torch.optim.Adam(predictor_net.parameters(), lr=0.01, weight_decay=1e-6)
            self.predictor_nets.append(predictor_net)
            self.optimizers.append(optimizer)
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, noise_sigma=0.0):
        """
        Train K predictors using same positions but different noise per head.
        For head i: Sample W_i = {w_t^i ~ N(0, noise_sigma^2)} for each position t
        Training target for head i: φ(s)ᵀθ̂ + w_t^i
        """
        print(f"Training Scalar Ensemble RND-Linear (SGD) with K={self.K} predictors on {len(positions)} positions")
        print(f"Using same positions for all heads, noise_sigma={noise_sigma}")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        N = len(positions)
        
        # Compute features φ(s) once (same for all heads)
        with torch.no_grad():
            phi_output = self.phi(coords_tensor)  # [N, feature_dim]
            target_scalar = torch.matmul(phi_output, self.theta_hat)  # [N]
        
        all_losses = []
        
        # Train each predictor independently with different noise
        for k in range(self.K):
            print(f"Training predictor {k+1}/{self.K}")
            
            # Sample noise for this head: W_k = {w_t^k ~ N(0, noise_sigma^2)}
            if noise_sigma > 0:
                noise = torch.randn(N, device=self.device) * noise_sigma  # [N]
            else:
                noise = torch.zeros(N, device=self.device)
            
            # Target for this head: clean_target_scalar + noise
            target_with_noise = target_scalar + noise
            
            predictor_losses = []
            
            for epoch in range(num_epochs):
                self.optimizers[k].zero_grad()
                
                # Predictor output (already scalar)
                pred_scalar = self.predictor_nets[k](phi_output).squeeze()  # [N]
                
                # Loss
                loss = self.criterion(pred_scalar, target_with_noise)
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
            
            # Debug: Check if predictions are identical (would cause STD=0)
            if scalar_predictions.shape[0] > 1:
                first_pred = scalar_predictions[0]
                max_diff = torch.max(torch.abs(scalar_predictions - first_pred)).item()
                if max_diff < 1e-6:
                    print(f"Warning: All {self.K} predictors produce nearly identical predictions (max_diff={max_diff:.2e}). STD will be ~0.")
            
            # Calculate STD across K predictors
            ensemble_uncertainty = torch.std(scalar_predictions, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()


class ScalarEnsembleRNDLinearLSMethod:
    """
    Scalar Ensemble-Based RND-Linear (LS) with noise-based diversity.
    
    Key features:
    - K predictors (configurable)
    - Same sub-dataset for all K heads (no bootstrap sampling)
    - Different Gaussian noise per head: w_t^i ~ N(0, noise_sigma^2)
    - Least squares fitting for each predictor
    - Output already scalar, use directly
    - Uncertainty = STD of scalar predictions across K heads
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
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, noise_sigma=0.0):
        """
        Train K predictors using same positions but different noise per head.
        For head i: Sample W_i = {w_t^i ~ N(0, noise_sigma^2)} for each position t
        Training target for head i: φ(s)ᵀθ̂ + w_t^i
        """
        print(f"Training Scalar Ensemble RND-Linear (LS) with K={self.K} predictors on {len(positions)} positions")
        print(f"Using same positions for all heads, noise_sigma={noise_sigma}")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        N = len(positions)
        
        # Compute features φ(s) once (same for all heads)
        with torch.no_grad():
            phi_output = self.phi(coords_tensor)  # [N, feature_dim]
            target_scalar = torch.matmul(phi_output, self.theta_hat)  # [N]
        
        # Convert to numpy for least squares
        X = phi_output.cpu().numpy()  # [N, feature_dim]
        
        all_losses = []
        
        # Train each predictor independently with different noise
        for k in range(self.K):
            print(f"Training predictor {k+1}/{self.K}")
            
            # Sample noise for this head: W_k = {w_t^k ~ N(0, noise_sigma^2)}
            if noise_sigma > 0:
                noise = np.random.randn(N) * noise_sigma
            else:
                noise = np.zeros(N)
            
            # Target for this head: clean_target_scalar + noise
            y = target_scalar.cpu().numpy() + noise  # [N]
            
            # Fit: predictor(φ(s)) = intercept + φ(s)ᵀ * weights
            intercept, weights = self.least_squares_fit(X, y, add_intercept=True)
            
            # Store weights
            self.predictor_intercepts[k] = torch.FloatTensor([intercept]).to(self.device)
            self.predictor_weights[k] = torch.FloatTensor(weights).to(self.device)
            
            # Compute final loss for logging
            pred_output = torch.matmul(phi_output, self.predictor_weights[k]) + self.predictor_intercepts[k]
            final_loss = torch.mean((pred_output.squeeze() - target_scalar) ** 2).item()
            
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
            
            # Debug: Check if predictions are identical (would cause STD=0)
            if scalar_predictions.shape[0] > 1:
                first_pred = scalar_predictions[0]
                max_diff = torch.max(torch.abs(scalar_predictions - first_pred)).item()
                if max_diff < 1e-6:
                    print(f"Warning: All {self.K} predictors produce nearly identical predictions (max_diff={max_diff:.2e}). STD will be ~0.")
            
            # Calculate STD across K predictors
            ensemble_uncertainty = torch.std(scalar_predictions, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()

