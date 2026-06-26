"""
Elliptical/UCB bonus: intrinsic reward = sqrt(φ(s,a)^T Λ^{-1} φ(s,a)) — Mahalanobis norm in (s,a) feature space.
Uses current state s and current action a; φ(s, a) is always frozen. Λ is updated from batches.
Implements IntrinsicRewardModel for VectorIntrinsicReplayBuffer.

Feature normalization (see development_document/main.tex, "Raw features and normalized features").
The covariance Λ is built from a normalized feature φ(x) = N_{ν_φ}(z_φ(x)), not the raw encoder output
z_φ(x), because the raw feature norm is an uncontrolled random-network artifact. Three modes:
  - "unit" (default): φ = z / (||z||_2 + ε_φ), so ||φ||_2 <= 1 and the ridge λ is interpretable.
  - "rms_unit":        per-coordinate standardize with running feature stats, then unit-normalize.
  - "none":            φ = z (raw features); diagnostic only.

Quadratic form q = φ^T Λ^{-1} φ (see "Stable quadratic-form computation"). Two paths, equal in exact
arithmetic:
  - "cholesky" (default, inverse-free): factor A = sym(Λ) + ε_chol·I = L L^T, solve L y = φ, then
    q = ||y||_2^2 = ||L^{-1} φ||_2^2. No explicit inverse is ever formed.
  - "inverse" (kept, not default): form Λ^{-1} once and read q = φ^T Λ^{-1} φ.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers.utils import RunningMeanStd

from rnd_exploration.common.format import to_tensor

from .base import IntrinsicRewardModel


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def normalize_features(
    z: torch.Tensor,
    mode: str,
    eps: float,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the elliptical feature normalization N_{ν_φ}(z) -> φ; shapes (batch, d) -> (batch, d).

    mode "none":     φ = z (raw frozen-encoder features; diagnostic only).
    mode "unit":     φ = z / (||z||_2 + eps); then ||φ||_2 <= 1, making the ridge λ interpretable.
    mode "rms_unit": u = (z - mean) / (std + eps); φ = u / (||u||_2 + eps). mean/std are the running
                     per-coordinate feature statistics (μ_t^φ, σ_t^φ), broadcast over the batch axis.
    eps is ε_φ (fixed 10^{-8} by default).
    """
    # none: pass the raw features straight through (no normalization).
    if mode == "none":
        return z
    # unit: divide each row by its 2-norm so every feature vector has length at most 1.
    if mode == "unit":
        return z / (z.norm(dim=-1, keepdim=True) + eps)
    # rms_unit: per-coordinate standardize with the running stats, then unit-normalize the result.
    if mode == "rms_unit":
        u = (z - mean) / (std + eps)
        return u / (u.norm(dim=-1, keepdim=True) + eps)
    raise ValueError(f"feature normalization mode must be 'unit', 'rms_unit', or 'none'; got {mode!r}")


def cholesky_quadratic_form(chol_L: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Inverse-free quadratic form q_i = φ_i^T (L L^T)^{-1} φ_i = ||L^{-1} φ_i||_2^2 via one triangular solve.

    chol_L is the lower-triangular Cholesky factor (d, d) of A = L L^T; phi is (batch, d). No inverse
    is formed: we solve L · sol = φ^T for sol = L^{-1} φ^T, then q_i is the squared norm of column i.
    Returns shape (batch,).
    """
    # Solve the lower-triangular system L @ sol = phi^T -> sol = L^{-1} phi^T, shape (d, batch).
    # before: phi (batch, d); after: sol (d, batch) with column i = L^{-1} φ_i.
    sol = torch.linalg.solve_triangular(chol_L, phi.transpose(-1, -2), upper=False)
    # q_i is the squared 2-norm of column i of sol (sum over the feature axis).
    return (sol ** 2).sum(dim=0)


def inverse_quadratic_form(cov_inv: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Explicit-inverse quadratic form q_i = φ_i^T Λ^{-1} φ_i (kept, not the default path).

    cov_inv is Λ^{-1} (d, d); phi is (batch, d). Returns shape (batch,).
    """
    # (φ Λ^{-1}) elementwise-times φ, summed over the feature axis -> q_i = φ_i^T Λ^{-1} φ_i.
    return ((phi @ cov_inv) * phi).sum(dim=1)


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
    Elliptical/UCB bonus: intrinsic reward = sqrt(max(φ(s,a)^T Λ^{-1} φ(s,a), ε_q)).
    Uses current state s and current action a. Λ is covariance of the normalized feature φ(s,a) over
    seen (s,a); φ is a frozen random encoder followed by the configured feature normalization. The
    quadratic form is computed inverse-free by default (Cholesky factor + triangular solve).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        feature_dim: int = 128,
        device: str = "cpu",
        regularization: float = 1e-6,
        feature_normalization: str = "unit",  # ν_φ: "unit" (default) | "rms_unit" | "none"
        feature_norm_eps: float = 1e-8,        # ε_φ: denominator constant in feature normalization
        bonus_method: str = "cholesky",        # "cholesky" (default, inverse-free) | "inverse" (explicit Λ^{-1})
        cholesky_jitter: float = 0.0,          # ε_chol: diagonal jitter before factorization; set 1e-8 if Cholesky fails
        quadratic_floor: float = 1e-12,        # ε_q: floor in sqrt(max(q, ε_q)) guarding round-off
    ):
        if feature_normalization not in ("unit", "rms_unit", "none"):
            raise ValueError(
                f"feature_normalization must be 'unit', 'rms_unit', or 'none'; got {feature_normalization!r}"
            )
        if bonus_method not in ("cholesky", "inverse"):
            raise ValueError(f"bonus_method must be 'cholesky' or 'inverse'; got {bonus_method!r}")
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        obs_dim = self.obs_shape[0]
        self.action_dim = int(action_dim)
        self._input_dim = obs_dim + self.action_dim

        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.regularization = float(regularization)
        self.feature_normalization = feature_normalization
        self.feature_norm_eps = float(feature_norm_eps)
        self.bonus_method = bonus_method
        self.cholesky_jitter = float(cholesky_jitter)
        self.quadratic_floor = float(quadratic_floor)

        self.phi = PhiEncoder(self._input_dim, feature_dim).to(self.device)
        for p in self.phi.parameters():
            p.requires_grad = False

        # rms_unit keeps running per-coordinate statistics of the RAW features z_φ; the other modes are stateless.
        self._feat_rms = (
            RunningMeanStd(shape=(feature_dim,)) if feature_normalization == "rms_unit" else None
        )

        # Λ_0 = λ I; build the active method's solve structure (Cholesky factor or explicit inverse).
        self._eye = torch.eye(feature_dim, device=self.device, dtype=torch.float32)
        self._cov = self._eye * self.regularization
        self._prepare_solver()

    def _prepare_solver(self) -> None:
        """Build the solve structure for the active bonus_method from the current Λ (self._cov)."""
        if self.bonus_method == "cholesky":
            # A = sym(Λ) + ε_chol·I, then A = L L^T. sym() removes round-off asymmetry before factoring.
            A = 0.5 * (self._cov + self._cov.T) + self.cholesky_jitter * self._eye
            self._chol_L = torch.linalg.cholesky(A)
        else:
            # explicit inverse path (kept for comparison; not the default)
            self._cov_inv = torch.linalg.inv(self._cov)

    def _raw_features(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Build (s, a) from observations and actions; return the RAW frozen-encoder features z_φ(s,a) (batch, d)."""
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

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the configured feature normalization N_{ν_φ} to raw features z -> φ (batch, d)."""
        # rms_unit needs the running per-coordinate stats as tensors on z's device; other modes are stateless.
        if self.feature_normalization == "rms_unit":
            mean = torch.as_tensor(self._feat_rms.mean, device=z.device, dtype=z.dtype)
            std = torch.sqrt(torch.as_tensor(self._feat_rms.var, device=z.device, dtype=z.dtype))
            return normalize_features(z, "rms_unit", self.feature_norm_eps, mean, std)
        return normalize_features(z, self.feature_normalization, self.feature_norm_eps)

    def _samples_to_features(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Return the normalized feature φ(s,a) = N_{ν_φ}(z_φ(s,a)) used in Λ and the bonus (batch, d)."""
        return self._normalize(self._raw_features(samples))

    def _quadratic_form(self, phi: torch.Tensor) -> torch.Tensor:
        """Dispatch q = φ^T Λ^{-1} φ to the active bonus_method (inverse-free Cholesky or explicit inverse)."""
        if self.bonus_method == "cholesky":
            return cholesky_quadratic_form(self._chol_L, phi)
        return inverse_quadratic_form(self._cov_inv, phi)

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Return bonus per sample: sqrt(max(φ(s,a)^T Λ^{-1} φ(s,a), ε_q)). Shape (batch_size,)."""
        phi = self._samples_to_features(samples)
        quadratic = self._quadratic_form(phi).clamp(min=self.quadratic_floor)
        return torch.sqrt(quadratic)

    def update(self, samples: Dict[str, Any]) -> None:
        """Update Λ = (1/n) φ^T φ + λ I from the batch, refreshing rms_unit stats and the solve structure."""
        z = self._raw_features(samples)
        n = z.shape[0]
        if n == 0:
            return
        # rms_unit: refresh running per-coordinate feature stats from this batch's RAW features before
        # normalizing. The buffer recomputes the bonus and refreshes these stats on every batch sampled
        # from the replay buffer (compute() then update() on the same sampled batch), so the stats are
        # built from the replay-sampled features. before: stats at update t-1; after: stats include this batch's z.
        if self.feature_normalization == "rms_unit":
            self._feat_rms.update(z.detach().cpu().numpy())
        phi = self._normalize(z)
        self._cov = (phi.T @ phi) / float(n) + self._eye * self.regularization
        self._prepare_solver()
