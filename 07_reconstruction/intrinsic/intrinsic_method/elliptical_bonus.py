"""
Elliptical episodic bonus (E3B-style) intrinsic reward model.
Bonus = sqrt(φ(s)^T Λ^{-1} φ(s)) — Mahalanobis distance in embedding space;
encourages visiting states whose embedding lies outside the fitted ellipse.
Λ is updated from batches (running covariance of φ(s)); φ(s) is always frozen (not trainable).
Implements IntrinsicRewardModel for VectorIntrinsicReplayBuffer.
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from utilities.format import to_tensor

from .base import IntrinsicRewardModel


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class PhiEncoder(nn.Module):
    """MLP mapping obs -> feature_dim (φ(s)). Always frozen, not trainable."""

    def __init__(self, obs_dim: int, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            _layer_init(nn.Linear(256, feature_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EllipticalBonus(IntrinsicRewardModel):
    """
    Elliptical bonus: intrinsic reward = sqrt(φ(s)^T Λ^{-1} φ(s)).
    Λ is a running covariance of φ over seen states; φ is always frozen.
    Higher reward for states whose embedding is far from the visited distribution (outside ellipse).
    Uses full observation (no obs_slice).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        feature_dim: int = 128,
        device: str = "cpu",
        regularization: float = 1e-6,
    ):
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        obs_dim = self.obs_shape[0]
        self._input_dim = obs_dim

        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.regularization = float(regularization)

        self.phi = PhiEncoder(self._input_dim, feature_dim).to(self.device)
        for p in self.phi.parameters():
            p.requires_grad = False

        # Λ = covariance, Λ_inv = its inverse. Start with regularized identity.
        self._cov = torch.eye(feature_dim, device=self.device, dtype=torch.float32) * self.regularization
        self._cov_inv = torch.eye(feature_dim, device=self.device, dtype=torch.float32) / self.regularization

    def _samples_to_features(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Extract next_observations from samples and return φ(s) (batch, feature_dim). Phi is frozen."""
        x = to_tensor(samples["next_observations"], self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.phi.eval()
        with torch.no_grad():
            return self.phi(x)

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Return elliptical bonus per sample: sqrt(φ(s)^T Λ^{-1} φ(s)). Shape (batch_size,)."""
        phi = self._samples_to_features(samples)
        # quadratic = φ^T Λ^{-1} φ = (φ Λ^{-1}) · φ
        tmp = phi @ self._cov_inv
        quadratic = (tmp * phi).sum(dim=1).clamp(min=1e-8)
        bonus = torch.sqrt(quadratic)
        return bonus

    def update(self, samples: Dict[str, Any]) -> None:
        """Update Λ from batch: Λ = (1/n) φ^T φ + reg*I, then set Λ_inv."""
        phi = self._samples_to_features(samples)
        n = phi.shape[0]
        if n == 0:
            return
        self._cov = (phi.T @ phi) / float(n) + torch.eye(
            self.feature_dim, device=self.device
        ) * self.regularization
        try:
            self._cov_inv = torch.linalg.inv(self._cov)
        except Exception:
            self._cov_inv = torch.linalg.pinv(self._cov)
