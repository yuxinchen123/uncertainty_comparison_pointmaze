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
    """Closed-form elliptical posterior readout on unit-normalized adaptive-dictionary
    features (the deterministic ideal of the coin-flip family).

    No residual and no training at all: the method keeps the running feature second-moment
    matrix A = I + sum over visits of phihat phihat^T with phihat(x) = phi(x)/||phi(x)|| the
    unit-normalized features on the growing dictionary of exp 019, and reads out the
    Gaussian-process posterior standard deviation b(x) = sqrt(phihat(x)^T A^{-1} phihat(x)).
    Before any visit A = I so b = ||phihat|| = 1 exactly at every input; for mutually
    orthogonal features a position visited m times reads exactly (1 + m)^{-1/2} — the count
    lives in A, which is precisely the elliptical-bonus matrix the project already studies.
    Ash et al. (ICLR 2022) is the bridge: this quantity is what coin-flip regression estimates
    by sampling, with the chi fluctuation removed."""

    name = "elliptical_adaptive_dict"

    TAU_ADD = 0.05
    SIGMA_MAX = 0.75
    SIGMA_MIN = 0.05
    MAX_CENTERS = 1024

    def __init__(self, seed: int, obs_dim: int = 4):
        """Empty dictionary and an empty accumulator (grown with the dictionary)."""
        torch.manual_seed(seed)  # draws nothing; the method has no random state of its own
        self.centers = torch.zeros(0, 2, dtype=torch.float64)
        self.sigmas = torch.zeros(0, dtype=torch.float64)
        self.A = torch.zeros(0, 0, dtype=torch.float64)  # holds A - I (grown with zeros)

    def _grow(self, z: torch.Tensor) -> None:
        """Insert new centers for batch rows far from the dictionary (as exp 019); the
        accumulator gains zero rows/columns (its identity part is added at readout)."""
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
                k = self.A.shape[0]
                a = torch.zeros(k + 1, k + 1, dtype=torch.float64)
                a[:k, :k] = self.A
                self.A = a

    def feat_hat(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-normalized per-center Gaussian features (norm 1 at every input, so the
        never-visited readout is exactly 1)."""
        z = x[:, :2].double()
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        phi = torch.exp(-d2 / (2.0 * self.sigmas[None, :] ** 2))
        return phi / phi.norm(dim=1, keepdim=True)

    def update(self, x: torch.Tensor) -> None:
        """Grow the dictionary, then accumulate the visited features' second moments."""
        with torch.no_grad():
            self._grow(x[:, :2].double())
            ph = self.feat_hat(x)
            self.A += ph.T @ ph

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """sqrt(phihat^T (I + A)^{-1} phihat), exactly 1 over an empty dictionary."""
        with torch.no_grad():
            if self.centers.shape[0] == 0:
                return torch.ones(x.shape[0], dtype=torch.float64).numpy()
            ph = self.feat_hat(x)
            lam = self.A + torch.eye(self.A.shape[0], dtype=torch.float64)
            sol = torch.linalg.solve(lam, ph.T)                    # (K, B)
            q = (ph * sol.T).sum(dim=1).clamp(min=0.0)
            return q.sqrt().cpu().numpy()
