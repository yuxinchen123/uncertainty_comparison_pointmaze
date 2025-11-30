import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple

class ScalarEnsembleRandomInitRNDMethod:
    """
    Scalar Ensemble-Based RND with random initialization diversity.
    
    Key features:
    - K predictors (configurable)
    - Same dataset for all K heads (no bootstrap sampling)
    - Random initialization per head (default PyTorch behavior)
    - Optional gaussian_noise (fixed per unique position, shared across predictors)
    - Scalar projection layer: output_dim -> 1 (trained)
    - Two uncertainty methods:
      - get_uncertainty_errors(): STD of |target - prediction|
      - get_uncertainty_predictions(): STD of predictions
    """
    
    def __init__(self, hidden_dims, output_dim, num_heads=10, device='cpu'):
        self.device = device
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        
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
            
        # K predictor networks (each trained independently with different random initialization)
        self.predictor_nets = []
        self.predictor_scalar_projs = []  # Scalar projection layers for each predictor
        self.optimizers = []
        self.criterion = nn.MSELoss()
        
        for k in range(num_heads):
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
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """
        Train K predictors using same dataset (no bootstrap).
        Each predictor has different random initialization.
        """
        print(f"Training Scalar Ensemble Random Init RND with {self.num_heads} predictors on {len(positions)} positions")
        print(f"Using same positions for all heads, diversity from random initialization only")
        
        # Fix: Assign fixed noise per unique position at training start (like folder 04)
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
        
        # Compute target once (same for all heads)
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        with torch.no_grad():
            target_output = self.target_net(coords_tensor)  # [N, output_dim]
            target_scalar = self.target_scalar_proj(target_output).squeeze()  # [N]
            
            if gaussian_noise > 0:
                # Look up fixed noise for each position
                noise_values = torch.zeros(len(positions), device=self.device)
                for i, pos in enumerate(positions[:, :2]):
                    pos_tuple = tuple(pos.tolist())
                    noise_values[i] = noise_map[pos_tuple]
                target_scalar += noise_values
        
        all_losses = []
        
        # Train each predictor independently on same dataset (different random init)
        for k in range(self.num_heads):
            print(f"Training predictor {k+1}/{self.num_heads}")
            
            predictor_losses = []
            
            for epoch in range(num_epochs):
                self.optimizers[k].zero_grad()
                
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
    
    def get_uncertainty_errors(self, coordinates):
        """
        Calculate ensemble uncertainty as STD of errors across K predictors.
        Uncertainty = Std[|target - pred_1|, |target - pred_2|, ..., |target - pred_K|]
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Compute target output
            target_output = self.target_net(coordinates)  # [N, output_dim]
            target_scalar = self.target_scalar_proj(target_output).squeeze()  # [N]
            
            # Calculate per-model errors for each predictor
            per_model_errors = []
            
            for k in range(self.num_heads):
                pred_output = self.predictor_nets[k](coordinates)  # [N, output_dim]
                pred_scalar = self.predictor_scalar_projs[k](pred_output).squeeze()  # [N]
                
                # Absolute difference between predictor and target
                error = torch.abs(target_scalar - pred_scalar)  # [N]
                per_model_errors.append(error)
            
            # Stack errors: [K, N]
            per_model_errors = torch.stack(per_model_errors, dim=0)
            
            # Calculate ensemble uncertainty as standard deviation across K predictors
            ensemble_uncertainty = torch.std(per_model_errors, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()
    
    def get_uncertainty_predictions(self, coordinates):
        """
        Calculate ensemble uncertainty as STD of scalar predictions across K predictors.
        Uncertainty = Std[pred_1(s), pred_2(s), ..., pred_K(s)]
        """
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.FloatTensor(coordinates).to(self.device)
        
        with torch.no_grad():
            # Get scalar predictions from all K predictors
            scalar_predictions = []
            
            for k in range(self.num_heads):
                pred_output = self.predictor_nets[k](coordinates)  # [N, output_dim]
                pred_scalar = self.predictor_scalar_projs[k](pred_output).squeeze()  # [N]
                scalar_predictions.append(pred_scalar)
            
            # Stack: [K, N]
            scalar_predictions = torch.stack(scalar_predictions, dim=0)
            
            # Calculate STD across K predictors
            ensemble_uncertainty = torch.std(scalar_predictions, dim=0)  # [N]
            
            return ensemble_uncertainty.cpu().numpy()

