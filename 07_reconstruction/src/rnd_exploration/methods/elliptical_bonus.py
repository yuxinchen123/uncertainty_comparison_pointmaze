"""
Elliptical/UCB bonus: intrinsic reward = sqrt(φ(s,a)^T Λ^{-1} φ(s,a)) — Mahalanobis norm in (s,a) feature space.
Uses current state s and current action a; φ(s, a) is always frozen. Λ is updated from batches.
Implements IntrinsicRewardModel for VectorIntrinsicReplayBuffer.
"""
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from rnd_exploration.common.format import to_tensor

from .base import IntrinsicRewardModel


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class PhiEncoder(nn.Module):
    """MLP mapping (s, a) -> feature_dim, i.e. input dim = obs_dim + action_dim. Always frozen."""

    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(input_dim, 256)),
            nn.ReLU(),
            _layer_init(nn.Linear(256, feature_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EllipticalBonus(IntrinsicRewardModel):
    """
    Elliptical/UCB bonus: intrinsic reward = sqrt(φ(s,a)^T Λ^{-1} φ(s,a)).
    Uses current state s and current action a. Λ is covariance of φ(s,a) over seen (s,a); φ is frozen.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        feature_dim: int = 128,
        device: str = "cpu",
        regularization: float = 1e-6,
    ):
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        obs_dim = self.obs_shape[0]
        self.action_dim = int(action_dim)
        self._input_dim = obs_dim + self.action_dim

        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.regularization = float(regularization)

        self.phi = PhiEncoder(self._input_dim, feature_dim).to(self.device)
        for p in self.phi.parameters():
            p.requires_grad = False

        self._cov = torch.eye(feature_dim, device=self.device, dtype=torch.float32) * self.regularization
        self._cov_inv = torch.eye(feature_dim, device=self.device, dtype=torch.float32) / self.regularization

    def _samples_to_features(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Build (s, a) from observations and actions; return φ(s, a) (batch, feature_dim)."""
        s = to_tensor(samples["observations"], self.device).float()
        a = to_tensor(samples["actions"], self.device).float()
        if s.dim() == 1:
            s = s.unsqueeze(0)
            a = a.unsqueeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = torch.cat([s, a], dim=-1)
        self.phi.eval()
        with torch.no_grad():
            return self.phi(x)

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Return bonus per sample: sqrt(φ(s,a)^T Λ^{-1} φ(s,a)). Shape (batch_size,)."""
        phi = self._samples_to_features(samples)
        tmp = phi @ self._cov_inv
        quadratic = (tmp * phi).sum(dim=1).clamp(min=1e-8)
        return torch.sqrt(quadratic)

    def update(self, samples: Dict[str, Any]) -> None:
        """Update Λ from batch (s,a): Λ = (1/n) φ(s,a)^T φ(s,a) + reg*I, then set Λ_inv."""
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
