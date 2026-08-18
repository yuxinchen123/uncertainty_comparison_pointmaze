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
    """Coin-flip counting + exact RLS head on LOCALIZED radial-basis features.

    exp 014/015 failed for a representational reason: zero-bias ReLU features are homogeneous
    half-plane hinges, rank-deficient (106/108) and conditioned at 1e10 on the cell-midpoint
    set, so the least-squares head cannot reach each position's running coin mean and the
    readout saturates on the unrepresentable component. This experiment swaps in 256 Gaussian
    radial-basis features (centers uniform over the maze extent, bandwidth sigma = 0.75 world
    units, Gram condition about 1e2), keeping everything else: fresh Rademacher coins per
    visit, exact ridge normal equations, unit-normalized frozen prior and zero head (bonus
    exactly 1 before and at the first visit). With a representable basis the least-squares
    optimum at each position IS its running coin mean, so the squared readout tracks 1/m_i per
    position under any visitation pattern, up to the chi fluctuation floor (~1/sqrt(2d))."""

    name = "coinflip_rls_rbf"

    D_COINS = 512
    N_FEAT = 256
    SIGMA = 0.75    # RBF bandwidth in world units (cells are 1x1)
    RIDGE = 1e-8

    def __init__(self, seed: int, obs_dim: int = 4):
        """Draw RBF centers, the frozen unit-normalized prior, and zero the RLS state."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        g = keyed_gen(seed, "rbf_centers")
        ext_lo = torch.tensor([-6.0, -4.5], dtype=torch.float64)
        ext_hi = torch.tensor([6.0, 4.5], dtype=torch.float64)
        self.centers = torch.rand(self.N_FEAT, 2, generator=g).double() * (ext_hi - ext_lo) + ext_lo
        self.prior_raw = make_mlp(obs_dim, 256, self.D_COINS, seed, "prior")
        for p in self.prior_raw.parameters():
            p.requires_grad_(False)
        self.Lam = self.RIDGE * torch.eye(self.N_FEAT, dtype=torch.float64)
        self.Bmat = torch.zeros(self.N_FEAT, self.D_COINS, dtype=torch.float64)
        self.coin_gen = keyed_gen(seed, "coins")

    def feat(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian radial-basis features of the position coordinates.
        before: x row [-5.5, 4.0, 0, 0]; after: 256 values exp(-||(x,y)-mu_j||^2 / (2 sigma^2))"""
        z = x[:, :2].double()
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * self.SIGMA ** 2))

    def prior(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen prior with ||prior(x)|| = sqrt(d) exactly at every input."""
        p = self.prior_raw(x).double()
        return float(np.sqrt(self.D_COINS)) * p / p.norm(dim=1, keepdim=True)

    def update(self, x: torch.Tensor) -> None:
        """One visit per batch row: fresh Rademacher coins into the normal equations."""
        with torch.no_grad():
            phi = self.feat(x)
            c = torch.randint(0, 2, (x.shape[0], self.D_COINS), generator=self.coin_gen,
                              dtype=torch.float64) * 2.0 - 1.0
            tgt = c - self.prior(x)
            self.Lam += phi.T @ phi
            self.Bmat += phi.T @ tgt

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """||head(x) + prior(x)|| / sqrt(d)."""
        with torch.no_grad():
            phi = self.feat(x)
            W = torch.linalg.solve(self.Lam, self.Bmat)
            out = phi @ W + self.prior(x)
            return (out.norm(dim=1) / float(np.sqrt(self.D_COINS))).cpu().numpy()
