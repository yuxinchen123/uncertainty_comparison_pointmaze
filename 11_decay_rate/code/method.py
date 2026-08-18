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
    """MLP + per-sample residual-encoded shrink targets + inner SGD fitting (the gradient-only
    analog of the exact linear-head method of exp 010).

    Each update computes every visited sample's normalized residual r_i (initial-copy
    denominator), forms the shrink target tau_i = f(x_i) + s_i e_i with s_i = 1/sqrt(1+r_i^2)
    (the per-visit map whose iterates from 1 are exactly 1/sqrt(2), 1/sqrt(3), ...), and runs
    inner SGD (lr 1e-2, up to 40 steps) on ||g - tau||^2 until every sample's fit error is
    below 5% of its own target residual scale. Unlike exp 005's inner Adam (whose fixed-size
    steps overshoot the nearby target), SGD steps are proportional to the distance to tau and
    contract monotonically. The open question this experiment measures: whether inner
    gradient fitting can realize the per-position shrink through the tangent kernel, or slow
    kernel modes leave some positions under-shrunk."""

    name = "mlp_resshrink_innersgd"

    INNER_LR = 1e-2
    INNER_MAX_STEPS = 40
    INNER_TOL = 0.05  # per-sample fit error allowed, relative to the sample's target scale

    def __init__(self, seed: int, obs_dim: int = 4):
        """Build target + predictor (4 -> 256 -> ReLU -> 128), the frozen init copy (readout
        denominator), and the inner SGD optimizer."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(obs_dim, 256, 128, seed, "target")
        self.predictor = make_mlp(obs_dim, 256, 128, seed, "predictor")
        for p in self.target.parameters():
            p.requires_grad_(False)
        # frozen initial predictor: the per-position readout denominator (starts the bonus at 1)
        self.predictor0 = copy.deepcopy(self.predictor)
        for p in self.predictor0.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.SGD(self.predictor.parameters(), lr=self.INNER_LR)

    def update(self, x: torch.Tensor) -> None:
        """Form the per-sample shrink targets, then inner-SGD until every sample fits."""
        with torch.no_grad():
            f = self.target(x)
            e = self.predictor(x) - f                       # current residual vectors (B, 128)
            e0n = (self.predictor0(x) - f).norm(dim=1)      # initial residual norms (B,)
            r = e.norm(dim=1) / e0n                         # normalized residuals (B,)
            s = 1.0 / torch.sqrt(1.0 + r.pow(2))            # per-visit shrink factors (B,)
            tau = f + s.unsqueeze(1) * e                    # shrink targets (B, 128)
            # per-sample tolerance: 5% of the target's own residual norm
            # before: r_i = 1, ||e_i|| = 7 -> s = 0.707, scale = 4.95, tol = 0.247
            tol = self.INNER_TOL * (s * e.norm(dim=1))
        for _ in range(self.INNER_MAX_STEPS):
            d = self.predictor(x) - tau
            with torch.no_grad():
                if bool((d.norm(dim=1) <= tol).all()):
                    break
            loss = 0.5 * d.pow(2).sum(dim=1).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """Per-point normalized residual ||g(x)-f(x)|| / ||g_0(x)-f(x)|| (exactly 1 at t=0)."""
        with torch.no_grad():
            f = self.target(x)
            e = (self.predictor(x) - f).pow(2).sum(dim=1).sqrt()
            e0 = (self.predictor0(x) - f).pow(2).sum(dim=1).sqrt()
            return (e / e0).cpu().numpy().astype(np.float64)
