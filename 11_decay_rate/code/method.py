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
    """Coin-flip counting + exact RLS head on a GROWING data-dependent kernel dictionary.

    exp 016/017 showed the coin-flip construction works once the basis can represent
    per-position running means, but a fixed bandwidth cannot serve both the cell-scale set and
    the 0.1-spaced center square. Here the basis adapts: the dictionary starts empty, and each
    visited state farther than 0.05 world units from every existing center is inserted as a new
    center whose bandwidth is half its nearest-center distance at insertion (clipped to
    [0.05, 0.75]); the normal-equation state grows with the dictionary (new rows start at
    zero). Resolution therefore concentrates exactly where states are visited, at the spacing
    they are visited at — the online analog of a growing radial-basis network. Coins, prior,
    and readout as exp 016 (fresh Rademacher per visit, unit-normalized frozen prior, zero
    head: bonus exactly 1 before and at the first visit)."""

    name = "coinflip_hadamard_perm"

    D_COINS = 512
    TAU_ADD = 0.05      # insert a visited state as a center beyond this distance
    SIGMA_MAX = 0.75    # bandwidth clip range for inserted centers
    SIGMA_MIN = 0.05
    MAX_CENTERS = 1024
    RIDGE = 1e-8

    def __init__(self, seed: int, obs_dim: int = 4):
        """Empty dictionary, frozen unit-normalized prior, empty normal-equation state."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.centers = torch.zeros(0, 2, dtype=torch.float64)
        self.sigmas = torch.zeros(0, dtype=torch.float64)
        self.prior_raw = make_mlp(obs_dim, 256, self.D_COINS, seed, "prior")
        for p in self.prior_raw.parameters():
            p.requires_grad_(False)
        self.Lam = torch.zeros(0, 0, dtype=torch.float64)
        self.Bmat = torch.zeros(0, self.D_COINS, dtype=torch.float64)
        self.coin_gen = keyed_gen(seed, "coins")
        # scrambled-Hadamard coin state (exp 034): rows of H within one block are mutually
        # orthogonal, so a position's first n coins sum to squared norm exactly n*d and the
        # independent-coin chi fluctuation vanishes within a block. One visit counter and one
        # per-block sign vector per center (dictionary state, like the normal equations).
        from scipy.linalg import hadamard
        self.H = torch.as_tensor(hadamard(self.D_COINS), dtype=torch.float64)
        self.hvisits = []
        self.block_signs = []
        self.block_perms = []   # per-center per-block column permutation (exp 035 fix: a
        # sign-only scramble leaves every full block's sum on coordinate 0, so consecutive
        # blocks cancel and the running-mean norm collapses past 512 visits; permuting the
        # columns per block sends each block's sum to its own random coordinate)

    def _grow(self, z: torch.Tensor) -> None:
        """Insert new centers for batch rows far from the dictionary (sequentially, so a row
        inserted first can cover later rows of the same batch)."""
        for row in z:
            if self.centers.shape[0] >= self.MAX_CENTERS:
                return
            if self.centers.shape[0] == 0:
                d_nn = float("inf")
            else:
                d_nn = float(((self.centers - row) ** 2).sum(-1).min().sqrt())
            if d_nn > self.TAU_ADD:
                # bandwidth = half the nearest-center distance at insertion, clipped
                # before: first center -> d_nn = inf -> sigma = SIGMA_MAX (0.75)
                # after: a center_square point 0.1 from its neighbor -> sigma = 0.05
                sig = min(self.SIGMA_MAX, max(self.SIGMA_MIN, 0.5 * d_nn))
                self.centers = torch.cat([self.centers, row[None, :]])
                self.sigmas = torch.cat([self.sigmas,
                                         torch.tensor([sig], dtype=torch.float64)])
                # grow the normal-equation state with zero rows/columns for the new feature
                k = self.Lam.shape[0]
                lam = torch.zeros(k + 1, k + 1, dtype=torch.float64)
                lam[:k, :k] = self.Lam
                self.Lam = lam
                self.Bmat = torch.cat([self.Bmat,
                                       torch.zeros(1, self.D_COINS, dtype=torch.float64)])
                self.hvisits.append(0)
                self.block_signs.append(None)
                self.block_perms.append(None)

    def feat(self, x: torch.Tensor) -> torch.Tensor:
        """Per-center Gaussian features with per-center bandwidths."""
        z = x[:, :2].double()
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * self.sigmas[None, :] ** 2))

    def prior(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen prior with ||prior(x)|| = sqrt(d) exactly at every input."""
        p = self.prior_raw(x).double()
        return float(np.sqrt(self.D_COINS)) * p / p.norm(dim=1, keepdim=True)

    def update(self, x: torch.Tensor) -> None:
        """Grow the dictionary from the batch, then accumulate coin normal equations."""
        with torch.no_grad():
            self._grow(x[:, :2].double())
            phi = self.feat(x)
            # per-row coin: the nearest center's next scrambled-Hadamard row; a fresh random
            # sign vector scrambles each new 512-row block (and the first visit).
            # before: center j at visit k=0,1,2 -> rows H[0]*s, H[1]*s, H[2]*s (orthogonal)
            z = x[:, :2].double()
            d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
            nearest = d2.argmin(dim=1)
            rows = []
            for j in nearest.tolist():
                k = self.hvisits[j]
                if k % self.D_COINS == 0:
                    self.block_signs[j] = (torch.randint(0, 2, (self.D_COINS,),
                                           generator=self.coin_gen, dtype=torch.float64)
                                           * 2.0 - 1.0)
                    self.block_perms[j] = torch.randperm(self.D_COINS,
                                                         generator=self.coin_gen)
                rows.append((self.H[k % self.D_COINS]
                             * self.block_signs[j])[self.block_perms[j]])
                self.hvisits[j] = k + 1
            c = torch.stack(rows)
            tgt = c - self.prior(x)
            self.Lam += phi.T @ phi
            self.Bmat += phi.T @ tgt

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """||head(x) + prior(x)|| / sqrt(d); the prior alone before any visit."""
        with torch.no_grad():
            pr = self.prior(x)
            if self.centers.shape[0] == 0:
                out = pr
            else:
                phi = self.feat(x)
                lam = self.Lam + self.RIDGE * torch.eye(self.Lam.shape[0], dtype=torch.float64)
                W = torch.linalg.solve(lam, self.Bmat)
                out = phi @ W + pr
            return (out.norm(dim=1) / float(np.sqrt(self.D_COINS))).cpu().numpy()
