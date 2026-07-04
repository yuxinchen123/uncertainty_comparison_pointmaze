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

from rnd_exploration.common.format import to_tensor

from .base import IntrinsicRewardModel


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal init (same as ppo_rnd_envpool.py)."""
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ObservationEncoder(nn.Module):
    """
    MLP encoder for flat observations: (batch_size, obs_dim) -> (batch_size, output_dim).

    - Standard RND (default ``linear_mode=False``): one ``network`` Sequential (256 hidden).
    - Linear RND building block (``linear_mode=True``): ``body`` (obs -> 256) + ``head`` (256 -> out).
      Use two instances: a frozen target encoder and a predictor encoder whose ``body`` is copied
      from the target; only the predictor's ``head`` is trained.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        output_dim: int,
        linear_mode: bool = False,
    ):
        super().__init__()
        obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
        self.linear_mode = linear_mode
        if linear_mode:
            self.body = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                nn.ReLU(),
            )
            self.head = layer_init(nn.Linear(256, output_dim))
            self.network = None
        else:
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, output_dim)),
            )
            self.body = None
            self.head = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.linear_mode:
            return self.head(self.body(obs))
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


FEATURE_CHOICES = (
    "rnd_next_state",                  # next_state
    "rnd_next_state_position_only",    # next_state with position only (fixed slice)
    "rnd_state",                       # state
    "rnd_state_action",                # state + action
    "rnd_state_action_next_state",      # state + action + next_state
)


class RND(IntrinsicRewardModel):
    """
    RND intrinsic reward model. Input is built from samples according to feature:
    - rnd_next_state: next_observations only (full).
    - rnd_next_state_position_only: first 2 dims of next_observations only.
    - rnd_state: observations only.
    - rnd_state_action: concat(observations, actions).
    - rnd_state_action_next_state: concat(observations, actions, next_observations).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        output_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        device: str = "cpu",
        use_obs_norm: bool = True,
        distance: str = "mse",
        n_predictors: int = 1,
        beta_std: float = 0.0,
        feature: str = "rnd_next_state",
        action_dim: Optional[int] = None,
        linear_rnd: bool = False,
        optimizer: str = "adam",
        bonus_readout: str = "mse",
        sgd_eta0: float = 1e-2,
        sgd_t0: float = 1e3,
    ):
        self.obs_shape = obs_shape if isinstance(obs_shape, tuple) else (int(obs_shape),)
        self.feature = feature
        self.action_dim = int(action_dim) if action_dim is not None else 0
        self.linear_rnd = bool(linear_rnd)

        obs_dim = self.obs_shape[0]
        if self.feature == "rnd_next_state":
            self._rnd_input_dim = obs_dim
        elif self.feature == "rnd_next_state_position_only":
            self._rnd_input_dim = 2
        elif self.feature == "rnd_state":
            self._rnd_input_dim = obs_dim
        elif self.feature == "rnd_state_action":
            self._rnd_input_dim = obs_dim + self.action_dim
        elif self.feature == "rnd_state_action_next_state":
            self._rnd_input_dim = obs_dim + self.action_dim + obs_dim
        else:
            raise ValueError(f"feature must be one of {FEATURE_CHOICES}; got {self.feature!r}")
        self._rnd_obs_shape = (self._rnd_input_dim,)

        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.use_obs_norm = bool(use_obs_norm)
        self.distance = str(distance).lower()
        if self.distance not in {"mse", "abs"}:
            raise ValueError("distance must be one of: 'mse', 'abs'")

        self.n_predictors = 1 if self.linear_rnd else n_predictors
        self.beta_std = beta_std

        # predictor optimizer + bonus readout (run 3.2.1 switches). adam/mse are the historical
        # defaults (bit-identical to the pre-switch code). The l2 readout is only defined on top of
        # the squared-error distance (||e||_2 = sqrt(2 * B_mse)), so it requires distance='mse'.
        self.lr = lr
        self.optimizer = str(optimizer).lower()
        if self.optimizer not in {"adam", "adagrad", "sgd1t"}:
            raise ValueError("optimizer must be one of: 'adam', 'adagrad', 'sgd1t'")
        self.bonus_readout = str(bonus_readout).lower()
        if self.bonus_readout not in {"mse", "l2"}:
            raise ValueError("bonus_readout must be one of: 'mse', 'l2'")
        if self.bonus_readout == "l2" and self.distance != "mse":
            raise ValueError("bonus_readout='l2' requires distance='mse' (l2 = sqrt(2*mse))")
        self.sgd_eta0 = float(sgd_eta0)
        self.sgd_t0 = float(sgd_t0)
        if self.sgd_t0 <= 0:
            raise ValueError("sgd_t0 must be > 0 (divides the update count in the 1/t schedule)")
        self._n_updates = 0  # predictor-update counter t for the sgd1t schedule (0-based)

        if self.linear_rnd:
            # Frozen target: body + head; never updated.
            self.target = ObservationEncoder(
                self._rnd_obs_shape, output_dim, linear_mode=True
            ).to(self.device)
            for p in self.target.parameters():
                p.requires_grad = False
            # Predictor: same body weights as target; fresh random head (trainable).
            self.predictor = ObservationEncoder(
                self._rnd_obs_shape, output_dim, linear_mode=True
            ).to(self.device)
            with torch.no_grad():
                self.predictor.body.load_state_dict(self.target.body.state_dict())
            for p in self.predictor.body.parameters():
                p.requires_grad = False
            self.opt = self._build_optimizer(self.predictor.head.parameters())
        else:
            if n_predictors == 1:
                self.predictor = ObservationEncoder(self._rnd_obs_shape, output_dim).to(self.device)
            else:
                self.predictor = EnsembleObservationEncoder(
                    self._rnd_obs_shape, output_dim, n_predictors=n_predictors
                ).to(self.device)
            self.target = ObservationEncoder(self._rnd_obs_shape, output_dim).to(self.device)
            for p in self.target.parameters():
                p.requires_grad = False
            self.opt = self._build_optimizer(self.predictor.parameters())
        self.obs_rms: Optional[object] = None
        if self.use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=self._rnd_obs_shape)

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        """Construct the predictor optimizer named by self.optimizer (run 3.2.1 methods O1/O2/O3)."""
        # adam (O1): the historical construction, bit-identical to the pre-switch code (factory lr=1e-3,
        # PyTorch default betas (0.9, 0.999) and eps 1e-8)
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        # adagrad (O2): PyTorch defaults written out for self-documentation (the Adam lr is ignored)
        if self.optimizer == "adagrad":
            return torch.optim.Adagrad(params, lr=1e-2, eps=1e-10, initial_accumulator_value=0)
        # sgd1t (O3): plain SGD (momentum 0, no weight decay); update() recomputes the lr from the
        # 1/t schedule before every step, and lr=eta0 here equals the schedule's t=0 value
        return torch.optim.SGD(params, lr=self.sgd_eta0)

    @staticmethod
    def _sgd_1t_lr(eta0: float, t0: float, t: int, eta_min: float = 0.0) -> float:
        """Shifted 1/t learning-rate schedule: eta_t = max(eta_min, eta0 / (1 + t/t0)), t 0-based."""
        return max(eta_min, eta0 / (1.0 + t / t0))

    def _get_feature_tensor(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Build input tensor (batch_size, _rnd_input_dim) from samples."""
        if self.feature == "rnd_next_state":
            next_obs = to_tensor(samples["next_observations"], self.device).float()
            return next_obs
        if self.feature == "rnd_next_state_position_only":
            next_obs = to_tensor(samples["next_observations"], self.device).float()
            return next_obs[..., 0:2]
        if self.feature == "rnd_state":
            return to_tensor(samples["observations"], self.device).float()
        if self.feature == "rnd_state_action":
            obs = to_tensor(samples["observations"], self.device).float()
            actions = to_tensor(samples["actions"], self.device).float()
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            return torch.cat([obs, actions], dim=-1)
        if self.feature == "rnd_state_action_next_state":
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

    def _compute_linear_rnd(self, x: torch.Tensor) -> torch.Tensor:
        """RND-Linear reward: i_t = || f_hat_theta(s) - f_theta(s) || (L2)."""
        with torch.no_grad():
            tgt = self.target(x)
            pred = self.predictor(x)
        diff = pred - tgt
        return diff.pow(2).sum(dim=1).clamp(min=1e-8).sqrt()

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Intrinsic reward from feature built from samples. Returns shape (batch_size,)."""
        x = self._get_feature_tensor(samples)
        x = self._normalize_obs(x)
        if self.linear_rnd:
            return self._compute_linear_rnd(x)
        with torch.no_grad():
            src = self.predictor(x)
            if self.n_predictors == 1:
                src = src.unsqueeze(1)
            tgt = self.target(x)
            distances = self._dist_ensemble(src, tgt)
            # l2 readout: with distance='mse' (enforced in __init__), distances holds per-predictor
            # B_mse = 0.5*sum_j e_j^2, so ||e||_2 = sqrt(2 * distances). before: [[1.5]] (residual
            # all-ones over 3 dims) -> after: [[1.7321]] (= sqrt(3)). The mse readout skips this.
            if self.bonus_readout == "l2":
                distances = (2.0 * distances).clamp(min=1e-8).sqrt()
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
        if self.linear_rnd:
            tgt = self.target(x).detach()
            pred = self.predictor(x)
            loss = (pred - tgt).pow(2).mean()
        else:
            src = self.predictor(x)
            if self.n_predictors == 1:
                src = src.unsqueeze(1)
            with torch.no_grad():
                tgt = self.target(x)
            loss = self._dist_ensemble(src, tgt).mean()
        loss.backward()
        # sgd1t: set this step's lr from the 1/t schedule; t = predictor updates done so far (0-based,
        # so the first step uses eta0 exactly). The loss above is ALWAYS the mse objective, for every
        # readout — the bonus readout never enters training (the exact norm is nonsmooth at zero).
        if self.optimizer == "sgd1t":
            lr_t = self._sgd_1t_lr(self.sgd_eta0, self.sgd_t0, self._n_updates)
            for group in self.opt.param_groups:
                group["lr"] = lr_t
        self.opt.step()
        self._n_updates += 1
