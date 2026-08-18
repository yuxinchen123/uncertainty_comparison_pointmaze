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
    """AdaGrad + initial-copy readout: the cumulative-squared-gradient optimizer (the closest
    off-the-shelf optimizer to a soft visit count — its accumulator never forgets) at the
    project's O2 settings (lr 1e-2, eps 1e-10, zero initial accumulator), on the plain MSE
    distillation loss. Readout: initial-copy-normalized residual norm."""

    name = "adagrad_initcopy_norm"

    def __init__(self, seed: int, obs_dim: int = 4):
        """Build target + predictor (4 -> 256 -> ReLU -> 128), the frozen init copy (readout
        denominator), and AdaGrad."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(obs_dim, 256, 128, seed, "target")
        self.predictor = make_mlp(obs_dim, 256, 128, seed, "predictor")
        for p in self.target.parameters():
            p.requires_grad_(False)
        # frozen initial predictor: the per-position readout denominator (starts the bonus at 1)
        self.predictor0 = copy.deepcopy(self.predictor)
        for p in self.predictor0.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adagrad(self.predictor.parameters(), lr=1e-2, eps=1e-10,
                                       initial_accumulator_value=0)

    def update(self, x: torch.Tensor) -> None:
        """One AdaGrad step on the MSE distillation loss."""
        e = self.predictor(x) - self.target(x)
        loss = 0.5 * e.pow(2).sum(dim=1).mean()
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
