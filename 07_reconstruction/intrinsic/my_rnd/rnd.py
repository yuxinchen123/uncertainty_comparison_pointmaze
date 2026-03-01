"""
RND (Random Network Distillation) for flat observations from ReplayBuffer.sample()
and for single-step intrinsic reward queries from env wrappers.
Predictor vs frozen target; intrinsic reward = distance(predictor(obs), target(obs)).
No update_proportion, kappa, or reward normalization. All learning data comes from buffer sample().
"""
from typing import Any, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from gymnasium.wrappers.utils import RunningMeanStd
from utilities.format import to_tensor


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


class MyRND:
    """
    RND that acts directly on batch-shaped data (batch_size, obs_dim).
    All inputs come from replay buffer sample(); no update in add().
    If obs_slice is set, only obs[..., start:end] is used for intrinsic reward and predictor training.
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
    ):
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        self.obs_slice = obs_slice
        if obs_slice is not None:
            start, end = obs_slice
            if start < 0 or end > self.obs_shape[0] or start >= end:
                raise ValueError(f"obs_slice must be (start, end) with 0 <= start < end <= obs_dim; got {obs_slice}")
            self._rnd_input_dim = end - start
            self._rnd_obs_shape = (self._rnd_input_dim,)
        else:
            self._rnd_input_dim = self.obs_shape[0]
            self._rnd_obs_shape = self.obs_shape

        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.use_obs_norm = bool(use_obs_norm)
        self.distance = str(distance).lower()
        if self.distance not in {"mse", "abs"}:
            raise ValueError("distance must be one of: 'mse', 'abs'")

        self.predictor = ObservationEncoder(self._rnd_obs_shape, output_dim).to(self.device)
        self.target = ObservationEncoder(self._rnd_obs_shape, output_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.obs_rms: Optional[object] = None
        if self.use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=self._rnd_obs_shape)

    def _slice_obs(self, x: torch.Tensor) -> torch.Tensor:
        """If obs_slice is set, return x[..., start:end]; else return x."""
        if self.obs_slice is None:
            return x
        start, end = self.obs_slice
        return x[..., start:end]

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_obs_norm or self.obs_rms is None:
            return x
        mean = torch.as_tensor(getattr(self.obs_rms, "mean"), device=x.device, dtype=x.dtype)
        var = torch.as_tensor(getattr(self.obs_rms, "var"), device=x.device, dtype=x.dtype)
        x = (x - mean) / torch.sqrt(var + 1e-8)
        return x.clamp(-5.0, 5.0)

    def _dist(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Per-sample distance vector of shape (batch_size,)."""
        if self.distance == "mse":
            return 0.5 * (tgt - src).pow(2).sum(dim=1)
        # abs / l1
        return (tgt - src).abs().sum(dim=1)

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """
        Intrinsic reward from next_observations. Only "next_observations" is used.

        Expected input:

            samples = {
                "next_observations": torch.Tensor or np.ndarray of shape (batch_size, obs_dim),
            }

        Returns a 1D tensor of shape (batch_size,).
        """
        if not isinstance(samples, dict) or "next_observations" not in samples:
            raise TypeError("samples must be a dict with 'next_observations' key")

        next_obs_raw = samples["next_observations"]
        next_obs = to_tensor(next_obs_raw, self.device)
        next_obs = self._slice_obs(next_obs)
        next_obs = self._normalize_obs(next_obs)
        with torch.no_grad():
            src = self.predictor(next_obs)
            tgt = self.target(next_obs)
            dist = self._dist(src, tgt)
        return dist

    def update(self, samples: Dict[str, torch.Tensor]) -> None:
        """Train predictor on observations. samples["observations"]: (batch_size, obs_dim)."""
        obs = samples["observations"].to(self.device).float()
        obs = self._slice_obs(obs)
        if self.use_obs_norm and self.obs_rms is not None:
            self.obs_rms.update(obs.detach().cpu().numpy())
        obs = self._normalize_obs(obs)
        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        for batch in loader:
            o = batch[0]
            self.opt.zero_grad()
            src = self.predictor(o)
            with torch.no_grad():
                tgt = self.target(o)
            per = self._dist(src, tgt)
            loss = per.mean()
            loss.backward()
            self.opt.step()
