"""THE editable method file (program.md): everything about how the bonus is produced lives
here — networks, initialization, optimizer, loss, readout. The experiment loop edits ONLY this
file; run_experiment.py and decay_harness/ are fixed.

Interface contract (run_experiment.py relies on exactly this):
  - class Method:
      name: str                       # short slug, used in the experiment folder name
      __init__(self, seed, obs_dim)   # build everything; ALL randomness keyed off `seed`
      update(self, x)                 # ONE optimizer step on the (B, obs_dim) float32 batch x
      bonus(self, x) -> np.ndarray    # per-point bonus readout, shape (B,), float64
  - update() is called once per training step with the batch the regime chose (full point set
    in uniform_fullbatch; a sampled batch in nonuniform). bonus() is called at checkpoints on
    the FULL point set and must not train or mutate optimizer state.
  - Readout rule (program.md): bonus() may use the input, the networks, and running statistics
    an online implementation could keep — never an explicit function of the step counter or of
    per-position visit counts. The learning-rate schedule inside update() MAY use the step
    counter (standard optimizer practice).

Current method: see the Method class docstring (one method per experiment; the history of
methods is the git history of this file plus results.tsv).
"""
import copy
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn


def keyed_gen(*parts) -> torch.Generator:
    """One torch.Generator seeded by a hash of stable names (reproducible-seeding rule): each
    named draw gets its own stream, so adding a new random quantity never shifts existing ones."""
    # e.g. keyed_gen(7, 'target', 'weight', 0) -> generator for the target net's first Linear
    key = "::".join(str(p) for p in parts)
    g = torch.Generator()
    g.manual_seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0x7FFFFFFF)
    return g


def make_mlp(obs_dim: int, hidden: int, out_dim: int, seed: int, stream: str) -> nn.Sequential:
    """The prior work's RND encoder: Linear(obs, hidden) -> ReLU -> Linear(hidden, out), every
    Linear orthogonal-initialized with gain sqrt(2) and zero bias, drawn from keyed streams."""
    net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    # orthogonal init per layer from its own keyed stream (layer index = position in the net)
    for idx, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            g = keyed_gen(seed, stream, "weight", idx)
            nn.init.orthogonal_(layer.weight, gain=float(np.sqrt(2.0)), generator=g)
            nn.init.constant_(layer.bias, 0.0)
    return net




def make_trunk(obs_dim, out_dim: int, seed: int, stream: str) -> nn.Module:
    """Obs-shape-adaptive trunk with the project's init law (orthogonal gain sqrt(2), zero
    biases, keyed streams). Vectors get obs -> 256 -> ReLU -> 256 -> ReLU -> out; images get
    the RND conv stack (32/64/64) -> flatten -> Linear -> ReLU -> out."""
    if isinstance(obs_dim, int):
        net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))
    else:
        h, w = obs_dim
        conv_out = ((h - 8) // 4 + 1, (w - 8) // 4 + 1)
        conv_out = ((conv_out[0] - 4) // 2 + 1, (conv_out[1] - 4) // 2 + 1)
        conv_out = ((conv_out[0] - 3) // 1 + 1, (conv_out[1] - 3) // 1 + 1)
        flat = 64 * conv_out[0] * conv_out[1]
        net = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
                            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                            nn.Flatten(), nn.Linear(flat, out_dim))
    idx = 0
    for layer in net:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            g = keyed_gen(seed, stream, "weight", idx)
            nn.init.orthogonal_(layer.weight.view(layer.weight.shape[0], -1), gain=float(np.sqrt(2.0)),
                                generator=g)
            nn.init.constant_(layer.bias, 0.0)
        idx += 1
    return net




def add_trunk_biases(net: nn.Module, sigma: float, seed: int, stream: str) -> None:
    """Overwrite a trunk's zero biases with keyed N(0, sigma^2) draws (used for LOW-dimensional
    inputs, whose bias-free ReLU features are homogeneous and nearly collinear)."""
    idx = 0
    for layer in net:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            with torch.no_grad():
                layer.bias.copy_(sigma * torch.randn(
                    layer.bias.shape, generator=keyed_gen(seed, stream, "bias", idx)))
        idx += 1


class Method:
    """Deep random features + exact coin-flip least-squares head: the RND-pluggable NEURAL
    counting bonus (per-state visit counts under any visitation, by statistics).

    A frozen deep trunk (vector MLP or RND conv stack; N(0, 0.5^2) biases only for inputs of
    dimension <= 8) produces 256-d features. Every visit draws a fresh Rademacher coin vector
    (d = 512) and accumulates the ridge normal equations of the regression from features onto
    (coin - prior); the head solve is cached and refreshed at readout. The frozen prior net is
    unit-normalized per input and the head starts empty, so the bonus
    ||head(x) + prior(x)|| / sqrt(d) is exactly 1 at every never-visited input, and a state
    visited m times reads the norm of its own m-coin running mean — m^(-1/2) in expectation,
    with the chi fluctuation floor and whatever feature collinearity shrinks. Whitener frozen
    after the first visited batch (as the shrink variant)."""

    name = "deep_lastlayer_cfn"

    D_COINS = 512
    RIDGE = 1e-8
    BIAS_SIGMA_LOWDIM = 0.5

    def __init__(self, seed: int, obs_dim=4):
        """Frozen trunk + frozen unit-normalized prior, empty normal equations, whitener."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.is_image = not isinstance(obs_dim, int)
        d = 256 if self.is_image else self.D_COINS
        self.d = d
        self.trunk = make_trunk(obs_dim, 256, seed, "feat")
        self.prior_raw = make_trunk(obs_dim, d, seed, "prior")
        if not self.is_image and obs_dim <= 8:
            add_trunk_biases(self.trunk, self.BIAS_SIGMA_LOWDIM, seed, "feat")
            add_trunk_biases(self.prior_raw, self.BIAS_SIGMA_LOWDIM, seed, "prior")
        for net in (self.trunk, self.prior_raw):
            for p in net.parameters():
                p.requires_grad_(False)
        self.device = torch.device(os.environ.get("METHOD_DEVICE", "cpu"))
        self.trunk.to(self.device)
        self.prior_raw.to(self.device)
        self.Lam = self.RIDGE * torch.eye(256, dtype=torch.float64, device=self.device)
        self.Bmat = torch.zeros(256, d, dtype=torch.float64, device=self.device)
        self.coin_gen = keyed_gen(seed, "coins")
        self._mean = None
        self._var = None

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Whitener frozen at the first update batch; identity before it; clip +-5."""
        z = x.to(self.device)
        if self._mean is not None:
            z = ((z - self._mean) / (self._var + 1e-8).sqrt()).clamp(-5.0, 5.0)
        return z.unsqueeze(1) if self.is_image else z

    def _freeze_whitener(self, x: torch.Tensor) -> None:
        """Initialize the whitener from the first visited batch, then never move it."""
        if self._mean is None:
            with torch.no_grad():
                z = x.to(self.device)
                self._mean = z.mean(dim=0)
                self._var = z.var(dim=0, unbiased=False) + 1e-8

    def _prior(self, z: torch.Tensor) -> torch.Tensor:
        """Frozen prior with ||prior(x)|| = sqrt(d) exactly at every input."""
        p = self.prior_raw(z).double()
        return float(np.sqrt(self.d)) * p / p.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def update(self, x: torch.Tensor) -> None:
        """One visit per batch row: fresh coins into the normal equations."""
        with torch.no_grad():
            self._freeze_whitener(x)
            z = self._whiten(x)
            phi = self.trunk(z).double()
            c = (torch.randint(0, 2, (x.shape[0], self.d), generator=self.coin_gen,
                               dtype=torch.float64) * 2.0 - 1.0).to(self.device)
            tgt = c - self._prior(z)
            self.Lam += phi.T @ phi
            self.Bmat += phi.T @ tgt

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """||head(x) + prior(x)|| / sqrt(d); exactly 1 before any visit."""
        with torch.no_grad():
            z = self._whiten(x)
            phi = self.trunk(z).double()
            W = torch.linalg.solve(self.Lam, self.Bmat)
            out = phi @ W + self._prior(z)
            return (out.norm(dim=1) / float(np.sqrt(self.d))).cpu().numpy()
