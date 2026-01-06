"""
Adapter classes to integrate uncertainty methods from folders 01-05 with RL training.
Provides unified interface for online updates and uncertainty computation.
"""
import numpy as np
import torch
import sys
import os

# Import uncertainty methods from different folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
from uncertainty_methods import RNDMethod, RNDLinearSGDMethod, RNDLinearLSMethod, EllipticalBonusMethod

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../02ensemble_nonscalarRND_bootstrap/utilities'))
try:
    from ensemble_uncertainty_methods import (
        EnsembleRNDMethod,
        EnsembleRNDLinearSGDMethod,
        EnsembleRNDLinearLSMethod
    )
except ImportError:
    EnsembleRNDMethod = None
    EnsembleRNDLinearSGDMethod = None
    EnsembleRNDLinearLSMethod = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../03scalar_ensemble_noise_based/utilities'))
try:
    from scalar_ensemble_methods import (
        ScalarEnsembleRNDMethod,
        ScalarEnsembleRNDLinearSGDMethod,
        ScalarEnsembleRNDLinearLSMethod
    )
except ImportError:
    ScalarEnsembleRNDMethod = None
    ScalarEnsembleRNDLinearSGDMethod = None
    ScalarEnsembleRNDLinearLSMethod = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../04scalar_ensemble_bootstrapping/utilities'))
try:
    import importlib.util
    bootstrap_path = os.path.join(os.path.dirname(__file__), '../../04scalar_ensemble_bootstrapping/utilities/scalar_ensemble_bootstrapping_methods.py')
    if os.path.exists(bootstrap_path):
        spec = importlib.util.spec_from_file_location("scalar_ensemble_bootstrapping_methods", bootstrap_path)
        bootstrap_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap_module)
        ScalarEnsembleBootstrapRNDMethod = getattr(bootstrap_module, 'ScalarEnsembleBootstrapRNDMethod', None)
        ScalarEnsembleBootstrapRNDLinearSGDMethod = getattr(bootstrap_module, 'ScalarEnsembleBootstrapRNDLinearSGDMethod', None)
        ScalarEnsembleBootstrapRNDLinearLSMethod = getattr(bootstrap_module, 'ScalarEnsembleBootstrapRNDLinearLSMethod', None)
    else:
        ScalarEnsembleBootstrapRNDMethod = None
        ScalarEnsembleBootstrapRNDLinearSGDMethod = None
        ScalarEnsembleBootstrapRNDLinearLSMethod = None
except (ImportError, Exception) as e:
    ScalarEnsembleBootstrapRNDMethod = None
    ScalarEnsembleBootstrapRNDLinearSGDMethod = None
    ScalarEnsembleBootstrapRNDLinearLSMethod = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../05scalar_ensemble_random_init/utilities'))
try:
    import importlib.util
    random_init_path = os.path.join(os.path.dirname(__file__), '../../05scalar_ensemble_random_init/utilities/scalar_ensemble_random_init_methods.py')
    if os.path.exists(random_init_path):
        spec = importlib.util.spec_from_file_location("scalar_ensemble_random_init_methods", random_init_path)
        random_init_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(random_init_module)
        ScalarEnsembleRandomInitRNDMethod = getattr(random_init_module, 'ScalarEnsembleRandomInitRNDMethod', None)
    else:
        ScalarEnsembleRandomInitRNDMethod = None
except (ImportError, Exception) as e:
    ScalarEnsembleRandomInitRNDMethod = None


class UncertaintyMethodAdapter:
    """
    Base adapter class that provides unified interface for all uncertainty methods.
    Handles online updates during RL training.
    """
    
    def __init__(self, method, update_frequency=1, num_epochs_per_update=1):
        """
        Args:
            method: Uncertainty method object (RND, RND-Linear, etc.)
            update_frequency: Update every N steps (1 = every step)
            num_epochs_per_update: Number of training epochs per update
        """
        self.method = method
        self.update_frequency = update_frequency
        self.num_epochs_per_update = num_epochs_per_update
        self.step_count = 0
        self.buffer = []  # Buffer for accumulating states before update
        
    def get_uncertainty(self, coordinates):
        """
        Get uncertainty for given coordinates.
        
        Args:
            coordinates: Array of shape (N, 2) with (x, y) positions
            
        Returns:
            uncertainties: Array of shape (N,) with uncertainty values
        """
        return self.method.get_uncertainty(coordinates)
    
    def update_with_batch(self, states, force_update=False):
        """
        Update uncertainty model with new states (online learning).
        
        Args:
            states: Array of shape (N, 2) with (x, y) positions
            force_update: If True, force update regardless of frequency
        """
        if states is None or len(states) == 0:
            return
        
        # Add to buffer
        if isinstance(states, list):
            self.buffer.extend(states)
        else:
            self.buffer.extend(states.tolist() if isinstance(states, np.ndarray) else [states])
        
        self.step_count += 1
        
        # Update if frequency reached or forced
        if force_update or (self.update_frequency > 0 and self.step_count % self.update_frequency == 0):
            if len(self.buffer) > 0:
                states_array = np.array(self.buffer)
                self._update_method(states_array)
                self.buffer = []  # Clear buffer after update
    
    def _update_method(self, states):
        """Internal method to update the underlying uncertainty method"""
        # Default: train on positions with specified epochs
        if hasattr(self.method, 'train_on_positions'):
            self.method.train_on_positions(
                states,
                num_epochs=self.num_epochs_per_update,
                subset_ratio=1.0,
                gaussian_noise=0.0
            )
        else:
            raise NotImplementedError(f"Method {type(self.method)} does not support train_on_positions")
    
    def flush_buffer(self):
        """Force update with any remaining buffered states"""
        if len(self.buffer) > 0:
            self.update_with_batch([], force_update=True)


def create_uncertainty_method(method_name, method_config, device='cpu'):
    """
    Factory function to create uncertainty method from config.
    
    Args:
        method_name: Name of method (e.g., 'rnd', 'rnd_linear_ls', 'gt')
        method_config: Dictionary with method-specific parameters
        device: Device for computation ('cpu' or 'cuda')
    
    Returns:
        UncertaintyMethodAdapter wrapping the method
    """
    method_name_lower = method_name.lower()
    
    # Single-model methods from 01sweep_uncertainty
    if method_name_lower == 'rnd':
        method = RNDMethod(
            hidden_dims=method_config.get('hidden_dims', [128, 128]),
            output_dim=method_config.get('output_dim', 128),
            device=device
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'rnd_linear_sgd':
        method = RNDLinearSGDMethod(
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            predictor_seed=method_config.get('predictor_seed', None),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower in ['rnd_linear_ls', 'rnd_linear']:
        method = RNDLinearLSMethod(
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'elliptical':
        method = EllipticalBonusMethod(
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            regularization=method_config.get('regularization', 1e-6)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    # Ensemble methods from 02ensemble_nonscalarRND_bootstrap
    elif method_name_lower == 'ensemble_rnd' and EnsembleRNDMethod is not None:
        method = EnsembleRNDMethod(
            num_heads=method_config.get('num_heads', 10),
            hidden_dims=method_config.get('hidden_dims', [128, 128]),
            output_dim=method_config.get('output_dim', 128),
            device=device
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'ensemble_rnd_linear_sgd' and EnsembleRNDLinearSGDMethod is not None:
        method = EnsembleRNDLinearSGDMethod(
            num_heads=method_config.get('num_heads', 10),
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'ensemble_rnd_linear_ls' and EnsembleRNDLinearLSMethod is not None:
        method = EnsembleRNDLinearLSMethod(
            num_heads=method_config.get('num_heads', 10),
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    # Scalar ensemble methods from 03scalar_ensemble_noise_based
    elif method_name_lower == 'scalar_ensemble_rnd' and ScalarEnsembleRNDMethod is not None:
        method = ScalarEnsembleRNDMethod(
            num_heads=method_config.get('num_heads', 10),
            hidden_dims=method_config.get('hidden_dims', [128, 128]),
            output_dim=method_config.get('output_dim', 128),
            device=device
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'scalar_ensemble_rnd_linear_sgd' and ScalarEnsembleRNDLinearSGDMethod is not None:
        method = ScalarEnsembleRNDLinearSGDMethod(
            num_heads=method_config.get('num_heads', 10),
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'scalar_ensemble_rnd_linear_ls' and ScalarEnsembleRNDLinearLSMethod is not None:
        method = ScalarEnsembleRNDLinearLSMethod(
            num_heads=method_config.get('num_heads', 10),
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    # Scalar ensemble bootstrapping methods from 04scalar_ensemble_bootstrapping
    elif method_name_lower == 'scalar_ensemble_bootstrap_rnd' and ScalarEnsembleBootstrapRNDMethod is not None:
        method = ScalarEnsembleBootstrapRNDMethod(
            num_heads=method_config.get('num_heads', 10),
            hidden_dims=method_config.get('hidden_dims', [128, 128]),
            output_dim=method_config.get('output_dim', 128),
            device=device
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'scalar_ensemble_bootstrap_rnd_linear_sgd' and ScalarEnsembleBootstrapRNDLinearSGDMethod is not None:
        method = ScalarEnsembleBootstrapRNDLinearSGDMethod(
            num_heads=method_config.get('num_heads', 10),
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    elif method_name_lower == 'scalar_ensemble_bootstrap_rnd_linear_ls' and ScalarEnsembleBootstrapRNDLinearLSMethod is not None:
        method = ScalarEnsembleBootstrapRNDLinearLSMethod(
            num_heads=method_config.get('num_heads', 10),
            feature_dim=method_config.get('feature_dim', 128),
            device=device,
            phi_weights=method_config.get('phi_weights', None),
            theta_seed=method_config.get('theta_seed', 42),
            regularization=method_config.get('regularization', 1e-2)
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    # Scalar ensemble random init from 05scalar_ensemble_random_init
    elif method_name_lower == 'scalar_ensemble_random_init_rnd' and ScalarEnsembleRandomInitRNDMethod is not None:
        method = ScalarEnsembleRandomInitRNDMethod(
            num_heads=method_config.get('num_heads', 10),
            hidden_dims=method_config.get('hidden_dims', [128, 128]),
            output_dim=method_config.get('output_dim', 128),
            device=device
        )
        return UncertaintyMethodAdapter(
            method,
            update_frequency=method_config.get('update_frequency', 1),
            num_epochs_per_update=method_config.get('num_epochs_per_update', 1)
        )
    
    else:
        raise ValueError(f"Unknown uncertainty method: {method_name}")

