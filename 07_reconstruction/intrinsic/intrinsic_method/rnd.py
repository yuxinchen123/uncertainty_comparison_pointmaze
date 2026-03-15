"""
RND (Random Network Distillation) intrinsic reward model.
Predictor vs frozen target; intrinsic reward = distance(predictor(obs), target(obs)).
All learning data comes from VectorIntrinsicReplayBuffer sample().
"""
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers.utils import RunningMeanStd

from utilities.format import to_tensor

from .base import IntrinsicRewardModel


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal init (same as ppo_rnd_envpool.py)."""
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ObservationEncoder(nn.Module):
    """MLP encoder for flat observations: (batch_size, obs_dim) -> (batch_size, output_dim)."""

    def __init__(self, obs_shape: Tuple[int, ...], output_dim: int):
        super().__init__()
        obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
        self.network = nn.Sequential(layer_init(nn.Linear(obs_dim, 256)), nn.ReLU(), layer_init(nn.Linear(256, output_dim)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class EnsembleObservationEncoder(nn.Module):
    """
    n independent MLP encoders in one module; single batched forward to (batch_size, n, output_dim).
    Weights stored as (n, ...) and applied with einsum for parallelism (no Python loop over n).
    """

    def __init__(self, obs_shape: Tuple[int, ...], output_dim: int, n_predictors: int = 5):
        super().__init__()
        self.n_predictors = n_predictors
        obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
        w1 = torch.empty(n_predictors, obs_dim, 256)
        b1 = torch.empty(n_predictors, 256)
        w2 = torch.empty(n_predictors, 256, output_dim)
        b2 = torch.empty(n_predictors, output_dim)
        for i in range(n_predictors):
            l1 = layer_init(nn.Linear(obs_dim, 256))
            l2 = layer_init(nn.Linear(256, output_dim))
            w1[i] = l1.weight.T
            b1[i] = l1.bias
            w2[i] = l2.weight.T
            b2[i] = l2.bias
        self.w1 = nn.Parameter(w1)
        self.b1 = nn.Parameter(b1)
        self.w2 = nn.Parameter(w2)
        self.b2 = nn.Parameter(b2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bd,qdi->bqi", obs, self.w1) + self.b1
        h = h.relu()
        out = torch.einsum("bqi,qio->bqo", h, self.w2) + self.b2
        return out


FEATURE_CHOICES = ("next_state", "state", "state_action", "state_action_next_state")


class RND(IntrinsicRewardModel):
    """
    RND intrinsic reward model. Input is built from samples according to feature:
    - next_state: next_observations only (obs_slice applied only here to next state).
    - state: observations only.
    - state_action: concat(observations, actions).
    - state_action_next_state: concat(observations, actions, next_observations).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        output_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        device: str = "cpu",
        use_obs_norm: bool = False,
        distance: str = "mse",
        obs_slice: Optional[Tuple[int, int]] = None,
        n_predictors: int = 5,
        beta_std: float = 0.0,
        feature: str = "next_state",
        action_dim: Optional[int] = None,
    ):
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        self.obs_slice = obs_slice
        self.action_dim = int(action_dim) if action_dim is not None else 0

        obs_dim = self.obs_shape[0]
        if self.feature == "next_state":
            if obs_slice is not None:
                start, end = obs_slice
                if start < 0 or end > obs_dim or start >= end:
                    raise ValueError(f"obs_slice must be (start, end) with 0 <= start < end <= obs_dim; got {obs_slice}")
                self._rnd_input_dim = end - start
            else:
                self._rnd_input_dim = obs_dim
        elif self.feature == "state":
            self._rnd_input_dim = obs_dim
        elif self.feature == "state_action":
            self._rnd_input_dim = obs_dim + self.action_dim
        else:
            self._rnd_input_dim = obs_dim + self.action_dim + obs_dim
        self._rnd_obs_shape = (self._rnd_input_dim,)

        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.use_obs_norm = bool(use_obs_norm)
        self.distance = str(distance).lower()
        if self.distance not in {"mse", "abs"}:
            raise ValueError("distance must be one of: 'mse', 'abs'")

        self.n_predictors = n_predictors
        self.beta_std = beta_std

        self.predictor = EnsembleObservationEncoder(
            self._rnd_obs_shape, output_dim, n_predictors=n_predictors
        ).to(self.device)
        self.target = ObservationEncoder(self._rnd_obs_shape, output_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms: Optional[object] = None
        if self.use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=self._rnd_obs_shape)

    def _slice_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Apply obs_slice only when used for next_state (feature='next_state')."""
        if self.obs_slice is None:
            return x
        start, end = self.obs_slice
        return x[..., start:end]

    def _get_feature_tensor(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Build input tensor (batch_size, _rnd_input_dim) from samples. obs_slice only for next_state when feature='next_state'."""
        if self.feature == "next_state":
            next_obs = to_tensor(samples["next_observations"], self.device).float()
            return self._slice_obs(next_obs)
        if self.feature == "state":
            return to_tensor(samples["observations"], self.device).float()
        if self.feature == "state_action":
            obs = to_tensor(samples["observations"], self.device).float()
            actions = to_tensor(samples["actions"], self.device).float()
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            return torch.cat([obs, actions], dim=-1)
        if self.feature == "state_action_next_state":
            obs = to_tensor(samples["observations"], self.device).float()
            actions = to_tensor(samples["actions"], self.device).float()
            next_obs = to_tensor(samples["next_observations"], self.device).float()
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            return torch.cat([obs, actions, next_obs], dim=-1)
        raise ValueError(f"feature must be one of {FEATURE_CHOICES}; got {self.feature!r}")

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_obs_norm or self.obs_rms is None:
            return x
        mean = torch.as_tensor(getattr(self.obs_rms, "mean"), device=x.device, dtype=x.dtype)
        var = torch.as_tensor(getattr(self.obs_rms, "var"), device=x.device, dtype=x.dtype)
        x = (x - mean) / torch.sqrt(var + 1e-8)
        return x.clamp(-5.0, 5.0)

    def _dist_ensemble(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        if self.distance == "mse":
            diff = tgt.unsqueeze(1) - src
            return 0.5 * diff.pow(2).sum(dim=2)
        diff = (tgt.unsqueeze(1) - src).abs()
        return diff.sum(dim=2)

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Intrinsic reward from feature built from samples. Returns shape (batch_size,)."""
        x = self._get_feature_tensor(samples)
        x = self._normalize_obs(x)
        with torch.no_grad():
            src = self.predictor(x)
            tgt = self.target(x)
            distances = self._dist_ensemble(src, tgt)
        mean_dist = distances.mean(dim=1)
        if self.n_predictors == 1:
            std_dist = torch.zeros_like(mean_dist, device=mean_dist.device, dtype=mean_dist.dtype)
        else:
            std_dist = distances.std(dim=1, unbiased=True)
        return mean_dist + self.beta_std * std_dist

    def update(self, samples: Dict[str, Any]) -> None:
        """Train predictor on feature built from samples."""
        x = self._get_feature_tensor(samples)
        if self.use_obs_norm and self.obs_rms is not None:
            self.obs_rms.update(x.detach().cpu().numpy())
        x = self._normalize_obs(x)
        self.opt.zero_grad()
        src = self.predictor(x)
        with torch.no_grad():
            tgt = self.target(x)
        loss = self._dist_ensemble(src, tgt).mean()
        loss.backward()
        self.opt.step()
