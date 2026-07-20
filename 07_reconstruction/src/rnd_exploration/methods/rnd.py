"""
RND (Random Network Distillation) intrinsic reward model.
Predictor vs frozen target; intrinsic reward = distance(predictor(obs), target(obs)).
All learning data comes from VectorIntrinsicReplayBuffer sample().
"""
import hashlib
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


def _make_activation(name: str) -> nn.Module:
    """Build the hidden activation module named by `name` for the RND encoders.
    'relu' -> nn.ReLU() (historical default, bit-identical to the pre-switch code);
    'leaky_relu' -> nn.LeakyReLU(0.2) (the original RND's tf.nn.leaky_relu slope 0.2, NOT torch's
    0.01 default). Both are parameter-free, so this never consumes the RNG."""
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"activation must be 'relu' or 'leaky_relu'; got {name!r}")


def keyed_gen(*parts) -> torch.Generator:
    """One torch.Generator seeded by a hash of stable names (reproducible-seeding rule). The key
    parts are IDENTICAL to the 2026-07-10 bias-ablation's keyed_gen, so a training run's bias draws
    are byte-equal to the ablation's untrained-bonus fields for the same seed."""
    # e.g. keyed_gen(111, 'bias-normal', 'target', 0) -> a fixed generator independent of any sigma
    key = "::".join(str(p) for p in parts)
    g = torch.Generator()
    g.manual_seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0x7FFFFFFF)
    return g


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
        activation: str = "relu",
        extra_layers: int = 0,
    ):
        super().__init__()
        obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
        self.linear_mode = linear_mode
        # extra_layers deepens the predictor (original RND's deeper-predictor asymmetry); it is only
        # defined for the standard Sequential path, never the linear_rnd body/head split.
        if linear_mode and extra_layers > 0:
            raise ValueError("extra_layers > 0 is not supported with linear_mode")
        if linear_mode:
            self.body = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 256)),
                _make_activation(activation),
            )
            self.head = layer_init(nn.Linear(256, output_dim))
            self.network = None
        else:
            # base 2-Linear MLP, then `extra_layers` appended [activation, Linear(out, out)] blocks on
            # the OUTPUT width. extra_layers=0 (default) yields exactly [Linear, act, Linear] -> two
            # layer_init draws in the historical order, so the default net is byte-identical.
            # before (extra_layers=0): [Lin(obs,256), act, Lin(256,out)]
            # after  (extra_layers=1): [Lin(obs,256), act, Lin(256,out), act, Lin(out,out)]  (out stays)
            modules = [
                layer_init(nn.Linear(obs_dim, 256)),
                _make_activation(activation),
                layer_init(nn.Linear(256, output_dim)),
            ]
            for _ in range(int(extra_layers)):
                modules.append(_make_activation(activation))
                modules.append(layer_init(nn.Linear(output_dim, output_dim)))
            self.network = nn.Sequential(*modules)
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
        weight_init: str = "orthogonal",
        bias_init: str = "zero",
        bias_seed: int = 0,
        reward_norm: bool = False,
        reward_norm_gamma: float = 0.99,
        activation: str = "relu",
        predictor_extra_layers: int = 0,
        update_proportion: float = 1.0,
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
        if self.bonus_readout not in {"mse", "l2", "mse_mean"}:
            raise ValueError("bonus_readout must be one of: 'mse', 'l2', 'mse_mean'")
        # l2 = sqrt(2*B_mse) and mse_mean = (1/m)*sum e_j^2 = (2/m)*B_mse both build on the squared-error
        # distance, so both require distance='mse'. mse_mean is the original RND paper's mean-over-dims
        # readout (no 1/2 factor); mse (0.5*sum) is the historical default.
        if self.bonus_readout in {"l2", "mse_mean"} and self.distance != "mse":
            raise ValueError(f"bonus_readout={self.bonus_readout!r} requires distance='mse'")
        self.sgd_eta0 = float(sgd_eta0)
        self.sgd_t0 = float(sgd_t0)
        if self.sgd_t0 <= 0:
            raise ValueError("sgd_t0 must be > 0 (divides the update count in the 1/t schedule)")
        self._n_updates = 0  # predictor-update counter t for the sgd1t schedule (0-based)

        # weight initialization for BOTH nets' Linear weights (convergence-run-1 switch).
        # "orthogonal" is the historical default (layer_init's orthogonal, gain sqrt(2) —
        # bit-identical to the pre-switch code); "pytorch_default" overwrites the weights with the
        # full PyTorch Linear default law U(-1/sqrt(fan_in), +1/sqrt(fan_in)) (what
        # reset_parameters()'s kaiming_uniform_(a=sqrt(5)) reduces to for Linear), drawn from its
        # own keyed stream so no other draw (orthogonal stream, bias streams) ever shifts.
        self.weight_init = str(weight_init)
        if self.weight_init not in {"orthogonal", "pytorch_default"}:
            raise ValueError(
                f"weight_init must be 'orthogonal' or 'pytorch_default'; got {self.weight_init!r}")
        if self.weight_init != "orthogonal" and (self.linear_rnd or n_predictors > 1):
            raise ValueError(
                "weight_init != 'orthogonal' is only implemented for the standard single-predictor "
                "RND architecture (not linear_rnd, not the ensemble encoder)")

        # bias initialization for BOTH nets' Linear biases (run 3.2.3 switch). "zero" is the
        # historical default (layer_init already zeroes biases, bit-identical to the pre-switch
        # code); "pytorch_default" = U(-1/sqrt(fan_in), +1/sqrt(fan_in)); "normal_<sigma>" =
        # N(0, sigma^2). Weights are touched only by the weight_init switch above, never by any
        # bias scheme.
        self.bias_init = str(bias_init)
        self.bias_seed = int(bias_seed)
        if self.bias_init != "zero":
            if self.linear_rnd or n_predictors > 1:
                raise ValueError(
                    "bias_init != 'zero' is only implemented for the standard single-predictor "
                    "RND architecture (not linear_rnd, not the ensemble encoder)")
            if self.bias_init != "pytorch_default":
                if not self.bias_init.startswith("normal_"):
                    raise ValueError(
                        "bias_init must be 'zero', 'pytorch_default', or 'normal_<sigma>'; "
                        f"got {self.bias_init!r}")
                sigma = float(self.bias_init.split("_", 1)[1])  # raises ValueError on a bad suffix
                if sigma <= 0:
                    raise ValueError(f"bias_init normal_<sigma> needs sigma > 0; got {sigma}")

        # original-RND architecture/optimization knobs (train run 5), all defaulting to the historical
        # behavior. `activation` is the hidden activation for BOTH nets ('relu' default | 'leaky_relu'
        # = slope 0.2). `predictor_extra_layers` deepens the PREDICTOR only by that many
        # [activation, Linear(out, out)] blocks (the original RND's deeper predictor); 0 = symmetric
        # (historical). `update_proportion` is the fraction of the batch kept in the predictor loss
        # (CleanRL keep-mask); 1.0 = train on the whole batch (historical, no torch.rand draw). The
        # deeper predictor and non-relu activation are only defined for the standard single-predictor
        # architecture (not linear_rnd, not the ensemble encoder).
        self.activation = str(activation)
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError(f"activation must be 'relu' or 'leaky_relu'; got {self.activation!r}")
        self.predictor_extra_layers = int(predictor_extra_layers)
        if self.predictor_extra_layers < 0:
            raise ValueError(f"predictor_extra_layers must be >= 0; got {self.predictor_extra_layers}")
        if (self.predictor_extra_layers > 0 or self.activation != "relu") and (
                self.linear_rnd or n_predictors > 1):
            raise ValueError(
                "predictor_extra_layers > 0 / activation != 'relu' are only implemented for the "
                "standard single-predictor RND architecture (not linear_rnd, not the ensemble encoder)")
        self.update_proportion = float(update_proportion)
        if not (0.0 < self.update_proportion <= 1.0):
            raise ValueError(f"update_proportion must be in (0, 1]; got {self.update_proportion}")

        # classic-RND intrinsic reward normalization (run 3.2.3 switch, off by default): divide the
        # bonus by a running std of the forward-filtered (discounted) intrinsic return, the recipe of
        # the original RND paper. observe() (fresh transitions, in time order) updates the filter and
        # the running std; compute() applies the division, so the replay-sampled training reward and
        # the wrapper's logged intrinsic are normalized consistently while eval rollouts (which only
        # call compute()) never move the statistics.
        self.reward_norm = bool(reward_norm)
        self.reward_norm_gamma = float(reward_norm_gamma)
        if not (0.0 < self.reward_norm_gamma < 1.0):
            raise ValueError(f"reward_norm_gamma must be in (0, 1); got {self.reward_norm_gamma}")
        self.reward_rms: Optional[object] = RunningMeanStd(shape=()) if self.reward_norm else None
        self._rff_return: Optional[np.ndarray] = None  # per-env forward-filter state R_t, lazily sized

        if self.linear_rnd:
            # Frozen target: body + head; never updated.
            self.target = ObservationEncoder(
                self._rnd_obs_shape, output_dim, linear_mode=True, activation=self.activation
            ).to(self.device)
            for p in self.target.parameters():
                p.requires_grad = False
            # Predictor: same body weights as target; fresh random head (trainable).
            self.predictor = ObservationEncoder(
                self._rnd_obs_shape, output_dim, linear_mode=True, activation=self.activation
            ).to(self.device)
            with torch.no_grad():
                self.predictor.body.load_state_dict(self.target.body.state_dict())
            for p in self.predictor.body.parameters():
                p.requires_grad = False
            self.opt = self._build_optimizer(self.predictor.head.parameters())
        else:
            # predictor gets the extra layers (deeper than the target); the target keeps the base depth.
            # extra_layers=0 + activation='relu' reproduce the historical identical-shape nets exactly.
            if n_predictors == 1:
                self.predictor = ObservationEncoder(
                    self._rnd_obs_shape, output_dim,
                    activation=self.activation, extra_layers=self.predictor_extra_layers
                ).to(self.device)
            else:
                self.predictor = EnsembleObservationEncoder(
                    self._rnd_obs_shape, output_dim, n_predictors=n_predictors
                ).to(self.device)
            self.target = ObservationEncoder(
                self._rnd_obs_shape, output_dim, activation=self.activation
            ).to(self.device)
            for p in self.target.parameters():
                p.requires_grad = False
            self.opt = self._build_optimizer(self.predictor.parameters())
        # overwrite the orthogonal weights with the requested scheme (validated above; no-op for
        # "orthogonal"), then the zero biases with theirs (no-op for "zero"). The two schemes draw
        # from independent keyed streams and never interact.
        if self.weight_init != "orthogonal":
            self._apply_weight_init()
        if self.bias_init != "zero":
            self._apply_bias_init()
        self.obs_rms: Optional[object] = None
        if self.use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=self._rnd_obs_shape)

    def _apply_weight_init(self) -> None:
        """Overwrite both nets' two Linear weight matrices in place per self.weight_init; biases are
        untouched here (the bias scheme is applied separately), and the draws come from their own
        keyed stream so every other draw stays byte-identical to a run without this switch."""
        # walk each net's Sequential by POSITION index and key every Linear's draw by that index. For the
        # base 2-Linear net the Linears sit at positions {0, 2}, so this reproduces the old hardcoded
        # (0, 2) keys exactly; a deeper predictor's extra Linears take NEW keys {4, 6, ...}, so the base
        # {0, 2} streams stay frozen (rng-seeding rule).
        for net_name, net in (("target", self.target.network), ("predictor", self.predictor.network)):
            for layer_idx, layer in enumerate(net):
                if not isinstance(layer, nn.Linear):
                    continue
                fan_in = layer.in_features
                with torch.no_grad():
                    # PyTorch Linear default weight law U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
                    # before: u ~ U(-1, 1), shape (fan_out, fan_in); after: W = u / sqrt(fan_in)
                    #   e.g. fan_in=4 -> entries in [-0.5, 0.5]; fan_in=256 -> [-0.0625, 0.0625]
                    g = keyed_gen(self.bias_seed, "weight-uniform", net_name, layer_idx)
                    u = torch.rand(layer.weight.shape, generator=g) * 2.0 - 1.0
                    layer.weight.copy_(u / (fan_in ** 0.5))

    def _apply_bias_init(self) -> None:
        """Overwrite both nets' two Linear biases in place per self.bias_init; weights are never
        touched, so a run's weight draws stay byte-identical to a zero-bias run at the same seed."""
        # walk each net's Sequential by POSITION index; each Linear's bias gets its own keyed draw
        # (same key parts as the 2026-07-10 bias-ablation: tag, net name, layer index). The base
        # Linears keep positions {0, 2} (byte-identical to the old hardcoded (0, 2)); a deeper
        # predictor's extra Linears take NEW keys {4, 6, ...}, so the base streams stay frozen.
        for net_name, net in (("target", self.target.network), ("predictor", self.predictor.network)):
            for layer_idx, layer in enumerate(net):
                if not isinstance(layer, nn.Linear):
                    continue
                fan_in = layer.in_features
                with torch.no_grad():
                    if self.bias_init == "pytorch_default":
                        # PyTorch Linear default bias law U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
                        # before: u ~ U(-1, 1), shape (fan_out,); after: bias = u / sqrt(fan_in)
                        #   e.g. fan_in=4 -> entries in [-0.5, 0.5]; fan_in=256 -> [-0.0625, 0.0625]
                        g = keyed_gen(self.bias_seed, "bias-uniform", net_name, layer_idx)
                        u = torch.rand(layer.bias.shape, generator=g) * 2.0 - 1.0
                        layer.bias.copy_(u / (fan_in ** 0.5))
                    else:
                        # normal_<sigma>: b = sigma * z, z ~ N(0,1). z is keyed WITHOUT sigma, so
                        # every scale reuses the SAME z direction (a pure scale sweep).
                        # before: z fixed per (net, layer); after: sigma * z
                        sigma = float(self.bias_init.split("_", 1)[1])
                        g = keyed_gen(self.bias_seed, "bias-normal", net_name, layer_idx)
                        z = torch.randn(layer.bias.shape, generator=g)
                        layer.bias.copy_(sigma * z)

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

    def _masked_mean(self, per_sample: torch.Tensor) -> torch.Tensor:
        """Mean of the per-sample loss over a random Bernoulli(update_proportion) keep-mask (CleanRL's
        proportion-of-experience mask). Called only when update_proportion < 1.0."""
        # keep each sample with probability update_proportion; average over the kept ones (>=1 to avoid
        # a divide-by-zero when the mask is all-False on a small batch).
        # before: per_sample=[l0,l1,l2,l3], p=0.5 -> mask~[1,0,1,0]; after: (l0+l2)/2
        mask = (torch.rand(per_sample.shape[0], device=per_sample.device) < self.update_proportion
                ).to(per_sample.dtype)
        return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)

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

    def _raw_bonus(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Unnormalized intrinsic bonus (the readout value) for a batch. Returns shape (batch_size,)."""
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
            # mse_mean readout (original RND): (1/m)*sum_j e_j^2 = (2/m)*B_mse (distances hold B_mse).
            # before: [[1.5]] (residual all-ones over 3 dims, m=3) -> after: [[1.0]] (= 3/3). Never
            # scales the mse readout (which stays 0.5*sum) or the training loss.
            elif self.bonus_readout == "mse_mean":
                distances = (2.0 / self.output_dim) * distances
        mean_dist = distances.mean(dim=1)
        if self.n_predictors == 1:
            std_dist = torch.zeros_like(mean_dist, device=mean_dist.device, dtype=mean_dist.dtype)
        else:
            std_dist = distances.std(dim=1, unbiased=True)
        return mean_dist + self.beta_std * std_dist

    def compute(self, samples: Dict[str, Any]) -> torch.Tensor:
        """Intrinsic reward from feature built from samples; with reward_norm, the raw readout
        divided by the running std of the filtered intrinsic return. Returns shape (batch_size,)."""
        bonus = self._raw_bonus(samples)
        # classic-RND normalization: r_t / std(filtered returns). The std comes from observe()'s
        # running statistics (initial var = 1, so the division is a no-op until data arrives).
        if self.reward_norm:
            bonus = bonus / float(np.sqrt(self.reward_rms.var))
        return bonus

    def observe(self, samples: Dict[str, Any]) -> None:
        """Update the reward-normalization statistics from freshly collected transition(s) (called by
        the buffer on add(), i.e. once per env step in time order). No-op unless reward_norm."""
        if not self.reward_norm:
            return
        # raw readout on the fresh transition(s); grads are off inside _raw_bonus, obs_rms untouched
        # (only update() moves the observation statistics, exactly as before)
        vals = self._raw_bonus(samples).detach().cpu().numpy().ravel().astype(np.float64)
        # per-env forward filter over time: R_t = gamma * R_{t-1} + r_t (one state per env row).
        # before: vals = [r_t] (n_envs=1 here), _rff_return = [R_{t-1}] or None at t=0
        # after:  _rff_return = [R_t]; the running std is updated with R_t
        if self._rff_return is None:
            self._rff_return = vals.copy()
        else:
            self._rff_return = self.reward_norm_gamma * self._rff_return + vals
        self.reward_rms.update(self._rff_return)

    def update(self, samples: Dict[str, Any]) -> None:
        """Train predictor on feature built from samples."""
        x = self._get_feature_tensor(samples)
        if self.use_obs_norm and self.obs_rms is not None:
            self.obs_rms.update(x.detach().cpu().numpy())
        x = self._normalize_obs(x)
        self.opt.zero_grad()
        # update_proportion >= 1.0 keeps the LITERAL historical expression (full-batch mean) and draws
        # no torch.rand, so the shared global RNG stream is untouched for every existing config; a
        # proportion < 1.0 (original RND / CleanRL 0.25) masks the per-sample loss via _masked_mean.
        if self.linear_rnd:
            tgt = self.target(x).detach()
            pred = self.predictor(x)
            if self.update_proportion >= 1.0:
                loss = (pred - tgt).pow(2).mean()
            else:
                loss = self._masked_mean((pred - tgt).pow(2).mean(dim=1))
        else:
            src = self.predictor(x)
            if self.n_predictors == 1:
                src = src.unsqueeze(1)
            with torch.no_grad():
                tgt = self.target(x)
            dist = self._dist_ensemble(src, tgt)
            if self.update_proportion >= 1.0:
                loss = dist.mean()
            else:
                loss = self._masked_mean(dist.mean(dim=1))
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
