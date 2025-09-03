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
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Target output (with noise)
            with torch.no_grad():
                target_output = self.target_net(coords_tensor)
                if gaussian_noise > 0:
                    noise = torch.randn_like(target_output) * gaussian_noise
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

class RNDLinearMethod:
    def __init__(self, feature_dim, device='cpu', phi_weights=None):
        self.device = device
        self.feature_dim = feature_dim
        
        # Shared φ(s): 2D -> feature_dim
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False
            
        # Target θ vector (random)
        self.theta = torch.randn(feature_dim).to(device)
        
        # Predictor: feature_dim -> 1
        self.predictor = nn.Linear(feature_dim, 1).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """Train predictor to match φ(s)ᵀθ - use all provided positions"""
        print(f"Training RND-Linear on {len(positions)} positions for {num_epochs} epochs")
        
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        losses = []
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Target: φ(s)ᵀθ (with noise)
            with torch.no_grad():
                phi_output = self.phi(coords_tensor)
                target_output = torch.matmul(phi_output, self.theta)
                if gaussian_noise > 0:
                    noise = torch.randn_like(target_output) * gaussian_noise
                    target_output += noise
            
            # Predictor output
            phi_output = self.phi(coords_tensor)
            pred_output = self.predictor(phi_output).squeeze()
            
            # Loss
            loss = self.criterion(pred_output, target_output)
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        
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

class EllipticalBonusMethod:
    def __init__(self, feature_dim, device='cpu', phi_weights=None):
        self.device = device
        self.feature_dim = feature_dim
        
        # Shared φ(s): Same as RND_Linear
        self.phi = nn.Linear(2, feature_dim).to(device)
        if phi_weights is not None:
            self.phi.load_state_dict(phi_weights)
        for param in self.phi.parameters():
            param.requires_grad = False
        
        # Covariance matrix (will be updated)
        self.covariance = torch.eye(feature_dim).to(device)
        self.covariance_inv = torch.eye(feature_dim).to(device)
    
    def update_covariance_from_positions(self, positions):
        """Update covariance matrix from positions"""
        coords_tensor = torch.FloatTensor(positions[:, :2]).to(self.device)
        
        # Compute φ(s) for all coordinates
        with torch.no_grad():
            phi_outputs = self.phi(coords_tensor)  # [N, feature_dim]
            
            # Update covariance matrix
            if len(phi_outputs) > 1:
                self.covariance = torch.cov(phi_outputs.T)
                # Add regularization for numerical stability
                self.covariance += torch.eye(self.covariance.shape[0]).to(self.device) * 1e-6
                self.covariance_inv = torch.linalg.inv(self.covariance)
    
    def train_on_positions(self, positions, num_epochs=30, subset_ratio=1.0, gaussian_noise=0.0):
        """Train elliptical bonus method - use all provided positions"""
        print(f"Training Elliptical Bonus on {len(positions)} positions (updating covariance)")
        
        # Update covariance from all provided positions
        self.update_covariance_from_positions(positions)
        
        # Simple progress logging
        for epoch in range(min(5, num_epochs)):  # Only need a few epochs for covariance update
            if (epoch + 1) % 2 == 0:
                print(f"  Step {epoch+1}, Covariance matrix updated")
        
        return [0.0] * num_epochs  # Dummy losses since this is analytical
    
    def get_uncertainty(self, coordinates):
        """Get uncertainty as φ(s)ᵀ Σ⁻¹ φ(s)"""
        with torch.no_grad():
            if isinstance(coordinates, np.ndarray):
                coordinates = torch.FloatTensor(coordinates).to(self.device)
            
            phi_output = self.phi(coordinates)  # [N, feature_dim]
            
            # Compute φ(s)ᵀ Σ⁻¹ φ(s) for each point
            uncertainty = torch.sum(phi_output @ self.covariance_inv * phi_output, dim=1)
            return uncertainty.cpu().numpy()