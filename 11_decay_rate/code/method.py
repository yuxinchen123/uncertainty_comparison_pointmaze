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
    """Sphere-projected inputs + AdaGrad + initial-copy readout.

    exp 007's diagnostics: the worst positions are the maze corners — the largest-radius
    inputs, whose tangent-kernel diagonal K(x, x) is largest for a ReLU MLP, so they train
    fastest (slopes -0.67 vs -0.55 near the center). Fix at the source: lift each input to
    [x, y, c] with a constant homogeneous coordinate c = 4 and L2-normalize, so every network
    input lies on the unit sphere and K(x, x) is the SAME at every position; angular geometry
    still separates the positions. Training and readout otherwise identical to exp 007
    (AdaGrad lr 1e-2, MSE loss, initial-copy-normalized residual norm)."""

    name = "sphere_input_adagrad"

    HOMOGENEOUS_C = 4.0  # the lifted constant coordinate (comparable to the +-5.5 input range)

    def __init__(self, seed: int, obs_dim: int = 4):
        """Build target + predictor (3 -> 256 -> ReLU -> 128 on the sphere-lifted input), the
        frozen init copy (readout denominator), and AdaGrad."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.target = make_mlp(3, 256, 128, seed, "target")
        self.predictor = make_mlp(3, 256, 128, seed, "predictor")
        for p in self.target.parameters():
            p.requires_grad_(False)
        # frozen initial predictor: the per-position readout denominator (starts the bonus at 1)
        self.predictor0 = copy.deepcopy(self.predictor)
        for p in self.predictor0.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adagrad(self.predictor.parameters(), lr=1e-2, eps=1e-10,
                                       initial_accumulator_value=0)

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        """[x, y, vx, vy] -> [x, y, c] / ||[x, y, c]||: every input lands on the unit sphere.
        before: rows [-5.5, 4.0, 0, 0] and [0.05, -0.05, 0, 0]
        after:  [-0.68, 0.49, 0.49] (norm 1) and [0.012, -0.012, 1.0] (norm 1)"""
        z = torch.cat([x[:, :2], torch.full_like(x[:, :1], self.HOMOGENEOUS_C)], dim=1)
        return z / z.pow(2).sum(dim=1, keepdim=True).sqrt()

    def update(self, x: torch.Tensor) -> None:
        """One AdaGrad step on the MSE distillation loss over sphere-lifted inputs."""
        z = self.lift(x)
        e = self.predictor(z) - self.target(z)
        loss = 0.5 * e.pow(2).sum(dim=1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """Initial-copy-normalized residual norm on the lifted input (exactly 1 at t=0)."""
        with torch.no_grad():
            z = self.lift(x)
            f = self.target(z)
            e = (self.predictor(z) - f).pow(2).sum(dim=1).sqrt()
            e0 = (self.predictor0(z) - f).pow(2).sum(dim=1).sqrt()
            return (e / e0).cpu().numpy().astype(np.float64)
