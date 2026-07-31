"""
Elliptical/UCB bonus family: intrinsic reward = sqrt(φ(x)^T Λ^{-1} φ(x)) — Mahalanobis norm in a
random-feature space. The encoder input x is selected by feature_input: "state_action" (default) uses
the RAW current state s and current action a, x=(s,a); "next_state" uses only the next observation s'
(no action), x=s'. No observation RMS — the only normalization is the in-family feature normalization
below; φ(x) is always frozen.
Implements IntrinsicRewardModel for VectorIntrinsicReplayBuffer.

The family is one bonus with two orthogonal choices:
  1. The covariance rule (how Λ changes given this call's feature batch Φ_U of size n), one class each:
     - EllipticalBonus (batch):           Λ = (1/n) Φ_U^T Φ_U + λ I        (replace).
     - GlobalEllipticalBonus (global):    Λ_t = Λ_{t-1} + Φ_U^T Φ_U        (iterative cumulative).
     The global form keeps ONE running matrix (Λ_0 = λ I) and never recomputes Λ from history.
  2. The update timing (when that rule fires), shared field update_timing:
     - "sample" (default): fired in update() on each replay minibatch (buffer.sample()) — same path as
       RND/batch; the covariance then tracks replay draws.
     - "add":              fired in observe() on each freshly collected transition (buffer.add()) — each
       visited (s,a) enters exactly once, so the covariance is a true environment-visitation count.

Feature normalization (see development_document/RND_development_document.tex, "Raw features and normalized features").
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

The bonus is clipped to bonus_clip (default 5): r = min(sqrt(max(q, ε_q)), bonus_clip). With a small
ridge an under-covered direction can give a very large raw bonus; the clip bounds rare reward spikes.
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
    """MLP mapping the encoder input x -> feature_dim. The caller sets input_dim: obs_dim + action_dim
    for the (s, a) input, or obs_dim for the next-state-only input. Always frozen."""

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
    Elliptical/UCB bonus (BATCH covariance rule, the family base): intrinsic reward =
    min(sqrt(max(φ(s,a)^T Λ^{-1} φ(s,a), ε_q)), bonus_clip).
    Uses the raw current state s and current action a. Λ is the covariance of the normalized feature
    φ(s,a); φ is a frozen random encoder followed by the configured feature normalization. The
    quadratic form is computed inverse-free by default (Cholesky factor + triangular solve).

    This base class also implements the shared update-timing dispatch (update() on sample / observe()
    on add) and a single _update_covariance() with the BATCH rule (Λ replaced from the current batch).
    Subclasses override only _update_covariance() to get the global / discounted-global rules.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        feature_dim: int = 128,
        device: str = "cpu",
        regularization: float = 1e-2,
        feature_normalization: str = "unit",  # ν_φ: "unit" (default) | "rms_unit" | "none"
        feature_norm_eps: float = 1e-8,        # ε_φ: denominator constant in feature normalization
        bonus_method: str = "cholesky",        # "cholesky" (default, inverse-free) | "inverse" (explicit Λ^{-1})
        cholesky_jitter: float = 0.0,          # ε_chol: diagonal jitter before factorization; set 1e-8 if Cholesky fails
        quadratic_floor: float = 1e-12,        # ε_q: floor in sqrt(max(q, ε_q)) guarding round-off
        bonus_clip: float = 5.0,               # r_max^E: clip the bonus to this max; float("inf") disables clipping
        update_timing: str = "sample",         # when the covariance rule fires: "sample" (default) | "add"
        feature_input: str = "state_action",   # encoder input x: "state_action" x=(s,a) | "next_state" x=s' (no action)
    ):
        if feature_normalization not in ("unit", "rms_unit", "none"):
            raise ValueError(
                f"feature_normalization must be 'unit', 'rms_unit', or 'none'; got {feature_normalization!r}"
            )
        if bonus_method not in ("cholesky", "inverse"):
            raise ValueError(f"bonus_method must be 'cholesky' or 'inverse'; got {bonus_method!r}")
        if update_timing not in ("sample", "add"):
            raise ValueError(f"update_timing must be 'sample' or 'add'; got {update_timing!r}")
        if feature_input not in ("state_action", "next_state"):
            raise ValueError(f"feature_input must be 'state_action' or 'next_state'; got {feature_input!r}")
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        obs_dim = self.obs_shape[0]
        self.action_dim = int(action_dim)
        self.feature_input = feature_input
        # encoder input dim: (s,a) adds the action dims; next-state-only is the observation dims alone.
        self._input_dim = obs_dim + self.action_dim if feature_input == "state_action" else obs_dim

        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.regularization = float(regularization)
        self.feature_normalization = feature_normalization
        self.feature_norm_eps = float(feature_norm_eps)
        self.bonus_method = bonus_method
        self.cholesky_jitter = float(cholesky_jitter)
        self.quadratic_floor = float(quadratic_floor)
        self.bonus_clip = float(bonus_clip)
        self.update_timing = update_timing

        self.phi = PhiEncoder(self._input_dim, feature_dim).to(self.device)
        for p in self.phi.parameters():
            p.requires_grad = False

        # rms_unit keeps running per-coordinate statistics of the RAW features z_φ; the other modes are stateless.
        self._feat_rms = (
            RunningMeanStd(shape=(feature_dim,)) if feature_normalization == "rms_unit" else None
        )

        # diagnostics counters (reported once per run via diagnostics()): feature-scale sums over the
        # covariance's own input stream, and clip-saturation counts over all computed bonuses.
        self._raw_sq_norm_sum = 0.0    # sum of ||z||^2 (raw encoder features)
        self._norm_sq_norm_sum = 0.0   # sum of ||φ||^2 (normalized features that build Λ)
        self._feature_count = 0        # rows summed above
        self._clip_hits = 0            # bonuses that exceeded bonus_clip before clamping
        self._bonus_count = 0          # bonuses computed in total

        # Λ_0 = λ I; build the active method's solve structure (Cholesky factor or explicit inverse).
        # The covariance and its factorization are kept in FLOAT64: the global rule accumulates Λ over the
        # whole run (Λ_t = λ I + Σ φφ^T), so Λ grows large and ill-conditioned, and a float32 Cholesky fails
        # on it (the leading minor is not positive-definite). float64 handles condition numbers up to ~1e15,
        # far above the ~1e10 this run reaches. The batch rule is well-conditioned, so this only affects it
        # at the ~1e-12 level. Features stay float32 and are cast to float64 at the covariance/solve boundary.
        self._eye = torch.eye(feature_dim, device=self.device, dtype=torch.float64)
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
        """Build the encoder input x from the configured feature_input and return the RAW frozen-encoder
        features z_φ(x) (batch, d). "state_action": x=[s;a] from observations+actions; "next_state":
        x=s' from next_observations only (action dropped)."""
        # state_action: concat current state and action into x=[s;a].
        if self.feature_input == "state_action":
            s = to_tensor(samples["observations"], self.device).float()
            a = to_tensor(samples["actions"], self.device).float()
            if s.dim() == 1:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            if a.dim() == 1:
                a = a.unsqueeze(-1)
            x = torch.cat([s, a], dim=-1)
        # next_state: use only the next observation s' (no action) as x.
        else:
            x = to_tensor(samples["next_observations"], self.device).float()
            if x.dim() == 1:
                x = x.unsqueeze(0)
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
        """Dispatch q = φ^T Λ^{-1} φ to the active bonus_method (inverse-free Cholesky or explicit inverse).
        φ is cast to float64 to match the float64 solve structure (see __init__: the global covariance can be
        ill-conditioned and a float32 solve is unstable). Returns a float64 tensor."""
        phi64 = phi.double()
        if self.bonus_method == "cholesky":
            return cholesky_quadratic_form(self._chol_L, phi64)
        return inverse_quadratic_form(self._cov_inv, phi64)

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Return bonus per sample: min(sqrt(max(φ^T Λ^{-1} φ, ε_q)), bonus_clip). Shape (batch_size,).
        The quadratic form is computed in float64; the bonus is cast back to float32 for the reward."""
        phi = self._samples_to_features(samples)
        quadratic = self._quadratic_form(phi).clamp(min=self.quadratic_floor)
        unclipped = torch.sqrt(quadratic)
        # clip-saturation diagnostics: count bonuses above the cap (always zero when bonus_clip is +inf)
        self._clip_hits += int((unclipped > self.bonus_clip).sum())
        self._bonus_count += int(unclipped.numel())
        # clip the raw Mahalanobis bonus to bonus_clip (no-op when bonus_clip is +inf), then back to float32.
        return unclipped.clamp(max=self.bonus_clip).float()

    def _features_for_update(self, samples: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Return normalized φ for a covariance update, or None for an empty batch. Refreshes the
        rms_unit running per-coordinate feature stats from this batch's RAW features first (so the
        stats track whatever stream — replay draws on "sample" timing, fresh transitions on "add"
        timing — feeds the covariance). before: stats at update t-1; after: stats include this batch's z."""
        z = self._raw_features(samples)
        if z.shape[0] == 0:
            return None
        if self.feature_normalization == "rms_unit":
            self._feat_rms.update(z.detach().cpu().numpy())
        phi = self._normalize(z)
        # feature-scale diagnostics over the covariance's own input stream: mean ||z||^2 (raw) and
        # mean ||φ||^2 (normalized) feed the effective ridge ratio ρ = λ d / mean ||φ||^2.
        # before: sums cover the batches of updates 1..t-1; after: this batch's rows are included.
        self._raw_sq_norm_sum += float((z ** 2).sum())
        self._norm_sq_norm_sum += float((phi ** 2).sum())
        self._feature_count += int(z.shape[0])
        return phi

    def _update_covariance(self, samples: Dict[str, Any]) -> None:
        """BATCH rule: replace Λ with this batch's covariance Λ = (1/n) φ^T φ + λ I, then refactor.
        Overridden by the global / discounted-global subclasses."""
        phi = self._features_for_update(samples)
        if phi is None:
            return
        # cast to float64 for the covariance (matches the float64 solve structure; see __init__)
        phi = phi.double()
        n = phi.shape[0]
        self._cov = (phi.T @ phi) / float(n) + self._eye * self.regularization
        self._prepare_solver()

    def diagnostics(self) -> Dict[str, float]:
        """Run-level summary for the local log: feature-scale means (raw and normalized — the normalized
        one gives the effective ridge ratio ρ = λ d / mean ||φ||^2), the covariance eigenvalue range, and
        the clip saturation fraction. Called once per checkpoint/final write; the eigvalsh is one cheap
        d×d call."""
        eigs = torch.linalg.eigvalsh(self._cov)
        n = max(1, self._feature_count)
        mean_norm_sq = self._norm_sq_norm_sum / n
        return {
            "mean_raw_feature_sq_norm": self._raw_sq_norm_sum / n,
            "mean_normalized_feature_sq_norm": mean_norm_sq,
            # ρ: how large the ridge is relative to the data covariance's average eigenvalue; inf-like
            # before any covariance update (mean_norm_sq = 0).
            "effective_ridge_ratio": (self.regularization * self.feature_dim / mean_norm_sq
                                      if mean_norm_sq > 0 else float("inf")),
            "cov_eig_min": float(eigs[0]),
            "cov_eig_max": float(eigs[-1]),
            "clip_hit_fraction": self._clip_hits / max(1, self._bonus_count),
            "n_bonuses_computed": self._bonus_count,
            "n_update_features": self._feature_count,
        }

    def update(self, samples: Dict[str, Any]) -> None:
        """Covariance update on the replay minibatch (buffer.sample()). Fires only on "sample" timing."""
        if self.update_timing == "sample":
            self._update_covariance(samples)

    def observe(self, samples: Dict[str, Any]) -> None:
        """Covariance update on freshly added transition(s) (buffer.add()). Fires only on "add" timing."""
        if self.update_timing == "add":
            self._update_covariance(samples)


class GlobalEllipticalBonus(EllipticalBonus):
    """Global (cumulative) elliptical bonus: one running covariance accumulated over all seen features,
    Λ_t = Λ_{t-1} + Φ_U^T Φ_U with Λ_0 = λ I. Differs from the batch base only in the covariance rule;
    the bonus, the solve, the clip, and the update-timing dispatch are inherited unchanged. Maintained
    iteratively (never recomputed from history). Closest of the family to a lifetime count."""

    def _update_covariance(self, samples: Dict[str, Any]) -> None:
        """GLOBAL rule: add this batch's outer products to the running Λ (no 1/n, no re-added ridge),
        then refactor. Λ_0 = λ I is set in the base __init__, so Λ_t = λ I + sum over all seen φ φ^T."""
        phi = self._features_for_update(samples)
        if phi is None:
            return
        # cast to float64 for the running covariance (it accumulates over the whole run and becomes
        # ill-conditioned; a float32 Cholesky fails on it -- see __init__).
        phi = phi.double()
        # accumulate onto the running covariance; the ridge λ I lives only in the Λ_0 init.
        # before: self._cov = Λ_{t-1}; after: self._cov = Λ_{t-1} + Φ_U^T Φ_U.
        self._cov = self._cov + phi.T @ phi
        self._prepare_solver()
