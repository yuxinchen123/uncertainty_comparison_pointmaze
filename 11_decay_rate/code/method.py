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
    """Deep random features + exact last-layer shrink head: the RND-pluggable NEURAL form of
    the campaign's exact construction.

    A frozen deep trunk (vector MLP or RND conv stack; random biases N(0, 0.5^2) added only
    for low-dimensional inputs, which need them to break ReLU homogeneity) produces 256-d
    features phi(x); a frozen deep target f(x) (128-d) plays RND's target role. The bonus is
    r(x) = ||W phi(x) - f(x)|| / ||f(x)|| with the head W zero-initialized, so r = 1 exactly
    at every never-visited input. Each visit applies the residual-encoded per-visit map
    r -> r / sqrt(1 + r^2) to the visited samples, realized by the minimum-Frobenius-change
    head update dW = E^T (Phi Phi^T + 1e-8 I)^{-1} Phi in float64 — a functional step that
    does not pass through the kernel spectrum, self-correcting under capacity stress and
    feature collinearity (the campaign's exps 010/023). Inputs pass through a whitener frozen
    after the first visited batch (stationarity for the convergence measurement; the online
    version would keep updating it)."""

    name = "deep_lastlayer_shrink"

    RIDGE = 1e-8
    BIAS_SIGMA_LOWDIM = 0.5   # trunk/target bias scale for inputs of dimension <= 8

    def __init__(self, seed: int, obs_dim=4):
        """Frozen trunk + frozen target (+ low-dim biases), zero head, whitener state."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.is_image = not isinstance(obs_dim, int)
        self.trunk = make_trunk(obs_dim, 256, seed, "feat")
        self.target = make_trunk(obs_dim, 128, seed, "target")
        if not self.is_image and obs_dim <= 8:
            add_trunk_biases(self.trunk, self.BIAS_SIGMA_LOWDIM, seed, "feat")
            add_trunk_biases(self.target, self.BIAS_SIGMA_LOWDIM, seed, "target")
        for net in (self.trunk, self.target):
            for p in net.parameters():
                p.requires_grad_(False)
        self.device = torch.device(os.environ.get("METHOD_DEVICE", "cpu"))
        self.trunk.to(self.device)
        self.target.to(self.device)
        self.W = torch.zeros(128, 256, dtype=torch.float64, device=self.device)
        self._mean = None
        self._var = None

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Whitener frozen at the FIRST update batch; identity before it; clip +-5."""
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

    def update(self, x: torch.Tensor) -> None:
        """Shrink the visited samples' residuals by 1/sqrt(1 + r^2) via the min-change solve."""
        with torch.no_grad():
            self._freeze_whitener(x)
            z = self._whiten(x)
            phi = self.trunk(z).double()                     # (B, 256)
            f = self.target(z).double()                      # (B, 128)
            e = phi @ self.W.T - f                           # residual vectors (B, 128)
            r = e.norm(dim=1) / f.norm(dim=1).clamp(min=1e-12)
            s = 1.0 / torch.sqrt(1.0 + r.pow(2))             # per-visit shrink (B,)
            E = (s - 1.0).unsqueeze(1) * e
            G = phi @ phi.T + self.RIDGE * torch.eye(phi.shape[0], dtype=torch.float64,
                                                     device=self.device)
            self.W += E.T @ torch.linalg.solve(G, phi)

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """r(x) = ||W phi(x) - f(x)|| / ||f(x)|| (exactly 1 at initialization)."""
        with torch.no_grad():
            z = self._whiten(x)
            phi = self.trunk(z).double()
            f = self.target(z).double()
            e = (phi @ self.W.T - f).norm(dim=1)
            return (e / f.norm(dim=1).clamp(min=1e-12)).cpu().numpy()
