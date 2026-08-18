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




class Method:
    """Residual-encoded shrink + exact min-change interpolation on a GROWING data-dependent
    kernel dictionary (the exp-010 mechanism on the exp-019 basis).

    The head starts at zero over an empty dictionary and the readout is normalized by the
    target's own norm, r(x) = ||W phi(x) - f(x)|| / ||f(x)||, so the bonus is exactly 1 at
    every never-visited input with no stored copy needed. Visited states grow the dictionary
    exactly as in exp 019 (insert beyond 0.05, bandwidth = half the nearest-center distance,
    clipped to [0.05, 0.75]); each visit shrinks the visited samples' residual vectors by
    s = 1/sqrt(1 + r^2) (the map whose iterates from 1 are 1/sqrt(2), 1/sqrt(3), ...), realized
    exactly by the minimum-Frobenius-change head update on the batch. Near-orthogonal local
    features keep the interference that broke exp 011 small, and unlike the coin-flip family
    there is no chi fluctuation floor — the recurrence is deterministic."""

    name = "linhead_shrink_adaptive"

    TAU_ADD = 0.05
    SIGMA_MAX = 0.75
    SIGMA_MIN = 0.05
    MAX_CENTERS = 1024
    RIDGE = 1e-8

    def __init__(self, seed: int, obs_dim: int = 4):
        """Frozen target MLP; empty dictionary; zero head over it."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(obs_dim, 256, 128, seed, "target")
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.centers = torch.zeros(0, 2, dtype=torch.float64)
        self.sigmas = torch.zeros(0, dtype=torch.float64)
        self.W = torch.zeros(128, 0, dtype=torch.float64)

    def _grow(self, z: torch.Tensor) -> None:
        """Insert new centers for batch rows far from the dictionary (sequential, as exp 019);
        the head gets a zero column per new center, leaving its function unchanged."""
        for row in z:
            if self.centers.shape[0] >= self.MAX_CENTERS:
                return
            if self.centers.shape[0] == 0:
                d_nn = float("inf")
            else:
                d_nn = float(((self.centers - row) ** 2).sum(-1).min().sqrt())
            if d_nn > self.TAU_ADD:
                sig = min(self.SIGMA_MAX, max(self.SIGMA_MIN, 0.5 * d_nn))
                self.centers = torch.cat([self.centers, row[None, :]])
                self.sigmas = torch.cat([self.sigmas,
                                         torch.tensor([sig], dtype=torch.float64)])
                self.W = torch.cat([self.W, torch.zeros(128, 1, dtype=torch.float64)], dim=1)

    def feat(self, x: torch.Tensor) -> torch.Tensor:
        """Per-center Gaussian features with per-center bandwidths."""
        z = x[:, :2].double()
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * self.sigmas[None, :] ** 2))

    def update(self, x: torch.Tensor) -> None:
        """Grow the dictionary, then shrink each visited sample's residual by its own factor
        via the minimum-change interpolation update."""
        with torch.no_grad():
            self._grow(x[:, :2].double())
            phi = self.feat(x)                                   # (B, K)
            f = self.target(x).double()                          # (B, 128)
            e = phi @ self.W.T - f                               # residual vectors (B, 128)
            r = e.norm(dim=1) / f.norm(dim=1)                    # normalized residuals (B,)
            s = 1.0 / torch.sqrt(1.0 + r.pow(2))                 # per-visit shrink (B,)
            E = (s - 1.0).unsqueeze(1) * e                       # desired residual change
            G = phi @ phi.T + self.RIDGE * torch.eye(phi.shape[0], dtype=torch.float64)
            self.W += E.T @ torch.linalg.solve(G, phi)

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """r(x) = ||W phi(x) - f(x)|| / ||f(x)|| (exactly 1 over an empty dictionary)."""
        with torch.no_grad():
            f = self.target(x).double()
            if self.centers.shape[0] == 0:
                return torch.ones(x.shape[0], dtype=torch.float64).numpy()
            e = (self.feat(x) @ self.W.T - f).norm(dim=1)
            return (e / f.norm(dim=1)).cpu().numpy()
