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
    """Residual-encoded shrink + exact min-change interpolation on LOCALIZED radial-basis
    features (the exp-010 construction with the exp-016 basis).

    exp 011 showed the exp-010 method does not track per-position counts under non-uniform
    visitation: the min-change solve on ill-spread global ReLU features drags unvisited
    positions along. Here the head lives on 256 Gaussian radial-basis features (centers
    uniform over the extent, sigma = 0.75, Gram condition about 1e2): the batch constraints
    are realized exactly while the update's effect is spatially LOCAL, so an unvisited
    position moves only insofar as a visited one sits within a bandwidth of it. Same per-visit
    map r -> r / sqrt(1 + r^2) (the bonus value itself encodes the visit count), same target
    network and initial-copy readout as exp 010; only the feature layer changed."""

    name = "linhead_shrink_rbf"

    N_FEAT = 256
    SIGMA = 0.75
    RIDGE = 1e-8

    def __init__(self, seed: int, obs_dim: int = 4):
        """Frozen target MLP, RBF features, head initialized from the usual predictor's output
        layer statistics, frozen initial head for the readout denominator."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(obs_dim, 256, 128, seed, "target")
        for p in self.target.parameters():
            p.requires_grad_(False)
        g = keyed_gen(seed, "rbf_centers")
        ext_lo = torch.tensor([-6.0, -4.5], dtype=torch.float64)
        ext_hi = torch.tensor([6.0, 4.5], dtype=torch.float64)
        self.centers = torch.rand(self.N_FEAT, 2, generator=g).double() * (ext_hi - ext_lo) + ext_lo
        # head init: orthogonal draw from its own keyed stream (an arbitrary nonzero start; the
        # readout is normalized by the frozen initial head, so only its randomness matters)
        W0 = torch.empty(128, self.N_FEAT)
        torch.nn.init.orthogonal_(W0, gain=float(np.sqrt(2.0)), generator=keyed_gen(seed, "head"))
        self.W = W0.double().clone()
        self.b = torch.zeros(128, dtype=torch.float64)
        self.W0, self.b0 = self.W.clone(), self.b.clone()

    def feat(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian radial-basis features of the position coordinates (as exp 016)."""
        z = x[:, :2].double()
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * self.SIGMA ** 2))

    def _predict(self, phi: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Head output for (P, N_FEAT) float64 features -> (P, 128)."""
        return phi @ W.T + b

    def update(self, x: torch.Tensor) -> None:
        """Shrink each visited sample's residual by 1/sqrt(1 + r^2) exactly via the
        minimum-change interpolation update of the head (bias included in the solve)."""
        with torch.no_grad():
            phi = self.feat(x)
            f = self.target(x).double()
            e = self._predict(phi, self.W, self.b) - f
            e0 = self._predict(phi, self.W0, self.b0) - f
            r = e.norm(dim=1) / e0.norm(dim=1)
            s = 1.0 / torch.sqrt(1.0 + r.pow(2))
            E = (s - 1.0).unsqueeze(1) * e
            phi_a = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=torch.float64)], dim=1)
            G = phi_a @ phi_a.T + self.RIDGE * torch.eye(phi.shape[0], dtype=torch.float64)
            dWa = E.T @ torch.linalg.solve(G, phi_a)
            self.W += dWa[:, :-1]
            self.b += dWa[:, -1]

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """Per-point normalized residual (exactly 1 at t=0)."""
        with torch.no_grad():
            phi = self.feat(x)
            f = self.target(x).double()
            e = (self._predict(phi, self.W, self.b) - f).norm(dim=1)
            e0 = (self._predict(phi, self.W0, self.b0) - f).norm(dim=1)
            return (e / e0).cpu().numpy()
