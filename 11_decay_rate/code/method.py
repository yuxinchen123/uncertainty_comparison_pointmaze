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
    """Linear head on frozen random features + exact minimum-change interpolation + a
    residual-encoded per-visit shrink.

    Architecture: frozen random features phi(x) = ReLU(W1 x + b1) (256-dim, the usual first
    layer of both networks) with a trainable linear head W (128 x 256); target f is the usual
    frozen MLP. The normalized residual r(x) = ||W phi(x) - f(x)|| / ||W0 phi(x) - f(x)||
    starts at exactly 1.

    Update: the count oracle's curve satisfies r(n) = (n+1)^(-1/2) under the per-visit map
    m -> m + 1 on m = 1/r^2, i.e. r' = r / sqrt(1 + r^2) — a pure function of the CURRENT
    residual, so the bonus value itself encodes the visit count and no counter or clock is
    needed (works under any visitation pattern). Each update shrinks every visited sample's
    residual VECTOR by its own factor s_i = 1 / sqrt(1 + r_i^2) and realizes the shrunk
    residual field exactly with the minimum-Frobenius-change head update
    dW = E^T G^{-1} Phi (G = Phi Phi^T + 1e-8 I in float64), which interpolates the P batch
    constraints exactly (P <= 108 < 256 features) while moving W as little as possible.
    Iterating from r = 1 gives exactly 1/sqrt(2), 1/sqrt(3), ... at every visited position
    simultaneously; the only deviations are the (n+1 vs n) off-by-one and feature-collinearity
    numerics."""

    name = "linhead_rls_residual_count"

    def __init__(self, seed: int, obs_dim: int = 4):
        """Build the frozen target MLP, the frozen feature layer + trainable linear head, and
        the frozen initial head (readout denominator)."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(obs_dim, 256, 128, seed, "target")
        for p in self.target.parameters():
            p.requires_grad_(False)
        # predictor = frozen random features + linear head, taken from the SAME init law as the
        # usual predictor MLP (its two Linear layers), so the initial bonus field matches
        pred = make_mlp(obs_dim, 256, 128, seed, "predictor")
        self.feat = nn.Sequential(pred[0], pred[1])  # Linear(obs, 256) + ReLU, frozen
        for p in self.feat.parameters():
            p.requires_grad_(False)
        self.W = pred[2].weight.detach().clone().double()   # (128, 256) trainable head
        self.b = pred[2].bias.detach().clone().double()     # (128,) trainable head bias
        self.W0, self.b0 = self.W.clone(), self.b.clone()   # frozen initial head (denominator)

    def _predict(self, phi: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Head output W phi + b for a (P, 256) float64 feature matrix -> (P, 128)."""
        return phi @ W.T + b

    def update(self, x: torch.Tensor) -> None:
        """Shrink each visited sample's residual by s = 1/sqrt(1 + r^2) exactly via the
        minimum-change interpolation update of the head."""
        with torch.no_grad():
            phi = self.feat(x).double()                      # (P, 256)
            f = self.target(x).double()                      # (P, 128)
            e = self._predict(phi, self.W, self.b) - f       # current residual vectors (P, 128)
            e0 = self._predict(phi, self.W0, self.b0) - f    # initial residual vectors (P, 128)
            r = e.norm(dim=1) / e0.norm(dim=1)               # normalized residual r_i (P,)
            s = 1.0 / torch.sqrt(1.0 + r.pow(2))             # per-visit shrink s_i (P,)
            # desired change of the residual field on the batch: E_i = (s_i - 1) e_i, so the
            # new residual is s_i e_i exactly.
            # before: r_i = 1 everywhere at t=0 -> s_i = 0.7071 -> new r_i = 0.7071
            # after another visit: r = 0.7071 -> s = 0.8165 -> new r = 0.5774 = 1/sqrt(3)
            E = (s - 1.0).unsqueeze(1) * e                   # (P, 128)
            # minimum-Frobenius-change head update interpolating the P constraints exactly:
            # augment phi with a constant column so the bias participates in the solve
            phi_a = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=torch.float64)], dim=1)
            G = phi_a @ phi_a.T + 1e-8 * torch.eye(phi.shape[0], dtype=torch.float64)
            dWa = E.T @ torch.linalg.solve(G, phi_a)         # (128, 257)
            self.W += dWa[:, :-1]
            self.b += dWa[:, -1]

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """Per-point normalized residual r(x) (exactly 1 at t=0)."""
        with torch.no_grad():
            phi = self.feat(x).double()
            f = self.target(x).double()
            e = (self._predict(phi, self.W, self.b) - f).norm(dim=1)
            e0 = (self._predict(phi, self.W0, self.b0) - f).norm(dim=1)
            return (e / e0).cpu().numpy().astype(np.float64)
