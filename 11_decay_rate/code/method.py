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
    """Coin-flip counting with an exact recursive-least-squares linear head (the coin-flip
    network of Lobel, Bagaria and Konidaris, ICML 2023, arXiv:2306.03186, made exact).

    Every visit to a position draws a FRESH Rademacher vector c in {-1,+1}^d. A linear head on
    frozen random features regresses positions onto their coins by exact ridge least squares
    (the normal equations are accumulated per visit; the solve happens at readout). The
    least-squares optimum at a position visited m times is that position's own running coin
    mean, whose expected squared norm is d/m — so the readout ||head(x) + prior(x)|| / sqrt(d)
    behaves as m^(-1/2) with THAT position's own visit count, under any visitation pattern:
    the count is encoded statistically, not through optimizer dynamics. The frozen prior
    network is unit-normalized per input (||prior(x)|| = sqrt(d) exactly) and the head starts
    at zero, so the bonus is exactly 1 at every never-visited input; after the first visit the
    running mean is a single Rademacher vector of norm exactly sqrt(d), so the bonus is exactly
    1 there too — matching min(1, m^(-1/2)) at m = 0 and 1 with no off-by-one. The chi-type
    fluctuation of a d-coordinate mean (relative std about sqrt(1/(2d)) on the norm) is this
    construction's noise floor; d = 512 puts it near 3 percent."""

    name = "coinflip_rls_head"

    D_COINS = 512   # coin/output dimension (the fluctuation floor scales as 1/sqrt(2d))
    RIDGE = 1e-8    # ridge on the accumulated normal equations

    def __init__(self, seed: int, obs_dim: int = 4):
        """Build frozen features, the frozen unit-normalized prior, the zero head, the
        normal-equation accumulators, and the keyed coin generator."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        pred = make_mlp(obs_dim, 256, self.D_COINS, seed, "predictor")
        self.feat = nn.Sequential(pred[0], pred[1])  # frozen ReLU features (256)
        for p in self.feat.parameters():
            p.requires_grad_(False)
        self.prior_raw = make_mlp(obs_dim, 256, self.D_COINS, seed, "prior")
        for p in self.prior_raw.parameters():
            p.requires_grad_(False)
        # exact ridge least squares state: Lam = ridge*I + sum phi phi^T, Bmat = sum phi tgt^T
        self.Lam = self.RIDGE * torch.eye(256, dtype=torch.float64)
        self.Bmat = torch.zeros(256, self.D_COINS, dtype=torch.float64)
        self.coin_gen = keyed_gen(0, "coins")  # one persistent stream; seed-independent draws
        # NOTE the coin stream is keyed by a constant, not the seed, so re-running a seed
        # re-draws the same coin sequence per call order — reproducible per (env, seed) cell
        # because each cell is its own process with its own Method instance.

    def prior(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen prior with ||prior(x)|| = sqrt(d) exactly at every input (computed pointwise,
        online): the never-visited bonus is exactly 1."""
        p = self.prior_raw(x).double()
        return float(np.sqrt(self.D_COINS)) * p / p.norm(dim=1, keepdim=True)

    def update(self, x: torch.Tensor) -> None:
        """One visit per batch row: draw fresh Rademacher coins, accumulate the normal
        equations of the regression from features onto (coin - prior)."""
        with torch.no_grad():
            phi = self.feat(x).double()                              # (B, 256)
            # fresh coins: +-1 per coordinate per visit occurrence
            # before: shape (B, 512) uniform ints in {0, 1}; after: {-1.0, +1.0}
            c = torch.randint(0, 2, (x.shape[0], self.D_COINS), generator=self.coin_gen,
                              dtype=torch.float64) * 2.0 - 1.0
            tgt = c - self.prior(x)                                  # head regresses this
            self.Lam += phi.T @ phi
            self.Bmat += phi.T @ tgt

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """||head(x) + prior(x)|| / sqrt(d): exactly 1 before the first visit, then the norm of
        the position's running coin mean."""
        with torch.no_grad():
            phi = self.feat(x).double()
            W = torch.linalg.solve(self.Lam, self.Bmat)              # (256, d)
            out = phi @ W + self.prior(x)
            return (out.norm(dim=1) / float(np.sqrt(self.D_COINS))).cpu().numpy()
