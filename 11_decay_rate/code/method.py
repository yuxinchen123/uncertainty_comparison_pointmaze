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
import os

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




def make_trunk(obs_dim, out_dim: int, seed: int, stream: str) -> nn.Module:
    """Obs-shape-adaptive trunk with the project's init law (orthogonal gain sqrt(2), zero
    biases, keyed streams). Vectors get obs -> 256 -> ReLU -> 256 -> ReLU -> out; images get
    the RND conv stack (32/64/64) -> flatten -> Linear -> ReLU -> out."""
    if isinstance(obs_dim, int):
        net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, out_dim))
    else:
        h, w = obs_dim
        conv_out = ((h - 8) // 4 + 1, (w - 8) // 4 + 1)
        conv_out = ((conv_out[0] - 4) // 2 + 1, (conv_out[1] - 4) // 2 + 1)
        conv_out = ((conv_out[0] - 3) // 1 + 1, (conv_out[1] - 3) // 1 + 1)
        flat = 64 * conv_out[0] * conv_out[1]
        net = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
                            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                            nn.Flatten(), nn.Linear(flat, out_dim))
    idx = 0
    for layer in net:
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            g = keyed_gen(seed, stream, "weight", idx)
            nn.init.orthogonal_(layer.weight.view(layer.weight.shape[0], -1), gain=float(np.sqrt(2.0)),
                                generator=g)
            nn.init.constant_(layer.bias, 0.0)
        idx += 1
    return net


class Method:
    """Gradient-trained coin-flip network (the published construction of Lobel, Bagaria and
    Konidaris, ICML 2023, in this campaign's exact-start form): a NEURAL, RND-pluggable
    counting bonus.

    The trainable network f-hat (vector MLP or RND conv trunk, head zero-initialized via a
    final zero Linear) is trained by plain gradient steps toward fresh Rademacher coins c per
    visit: loss = mean ||f-hat(x) - (c - prior(x))||^2, with a frozen same-architecture prior
    unit-normalized per input, so the bonus ||f-hat(x) + prior(x)|| / sqrt(d) is exactly 1 at
    every never-visited input. If optimization tracks the running least-squares optimum, the
    bonus at a state visited m times behaves as m^(-1/2) with that state's own count —
    statistically, at every state, under any visitation. THE question this experiment
    measures: how closely gradient training tracks that optimum per state. Observations pass
    through the method's own running whitener (mean/std over visited batches, clip +-5) — the
    standard RND input treatment, required for the heterogeneous-scale AntMaze dimensions.
    Optimizer: this variant uses OPTIMIZER below (the CFN paper trains with Adam 1e-4)."""

    name = "cfn_neural_adam1e-4"

    D_COINS = 512
    OPTIMIZER = ("adam", 1e-4)      # ("adam", lr) | ("adagrad", lr) | ("sgd1t", eta0, t0)
    ZERO_HEAD = True

    def __init__(self, seed: int, obs_dim=4):
        """Trainable net (zero head), frozen unit-normalized prior, whitener, optimizer."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.is_image = not isinstance(obs_dim, int)
        d = self.D_COINS if not self.is_image else 256
        self.d = d
        self.net = make_trunk(obs_dim, d, seed, "predictor")
        # zero the OUTPUT layer so f-hat is exactly 0 at initialization (start-at-1 exactness)
        last = [m for m in self.net if isinstance(m, (nn.Linear, nn.Conv2d))][-1]
        if self.ZERO_HEAD:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.prior_raw = make_trunk(obs_dim, d, seed, "prior")
        for p in self.prior_raw.parameters():
            p.requires_grad_(False)
        # running whitener over visited batches (RND's obs treatment): mean/var per input dim
        self._count = 1e-4
        self._mean = None
        self._var = None
        self.coin_gen = keyed_gen(seed, "coins")
        self.t = 0
        kind = self.OPTIMIZER[0]
        if kind == "adam":
            self.opt = torch.optim.Adam(self.net.parameters(), lr=self.OPTIMIZER[1])
        elif kind == "adagrad":
            self.opt = torch.optim.Adagrad(self.net.parameters(), lr=self.OPTIMIZER[1],
                                           eps=1e-10, initial_accumulator_value=0)
        elif kind == "sgd1t":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=self.OPTIMIZER[1])
        else:
            raise ValueError(f"unknown optimizer kind {kind!r}")
        # device: METHOD_DEVICE=cuda for GPU jobs (conv/atari); default cpu
        self.device = torch.device(os.environ.get("METHOD_DEVICE", "cpu"))
        self.net.to(self.device)
        self.prior_raw.to(self.device)

    def _whiten(self, x: torch.Tensor, update: bool) -> torch.Tensor:
        """Running mean/std whitening with a +-5 clip; the image branch adds the channel dim.
        before: antmaze rows with qvel entries in the tens; after: zero-mean unit-var, clipped"""
        z = x
        if update:
            with torch.no_grad():
                b_mean = z.mean(dim=0)
                b_var = z.var(dim=0, unbiased=False)
                n = float(z.shape[0])
                if self._mean is None:
                    self._mean, self._var = b_mean.clone(), b_var.clone() + 1e-8
                    self._count = n
                else:
                    tot = self._count + n
                    delta = b_mean - self._mean
                    self._mean = self._mean + delta * (n / tot)
                    self._var = (self._var * self._count + b_var * n
                                 + delta.pow(2) * self._count * n / tot) / tot
                    self._count = tot
        if self._mean is None:
            out = z
        else:
            out = ((z - self._mean) / (self._var + 1e-8).sqrt()).clamp(-5.0, 5.0)
        return out.unsqueeze(1) if self.is_image else out

    def update(self, x: torch.Tensor) -> None:
        """One optimizer step toward fresh Rademacher coins (whitener updated first)."""
        self.t += 1
        if self.OPTIMIZER[0] == "sgd1t":
            eta0, t0 = self.OPTIMIZER[1], self.OPTIMIZER[2]
            for group in self.opt.param_groups:
                group["lr"] = eta0 / (1.0 + (self.t - 1) / t0)
        z = self._whiten(x.to(self.device), update=True)
        with torch.no_grad():
            c = (torch.randint(0, 2, (x.shape[0], self.d), generator=self.coin_gen,
                               dtype=torch.float32) * 2.0 - 1.0).to(self.device)
            p = self.prior_raw(z)
            p = float(np.sqrt(self.d)) * p / p.norm(dim=1, keepdim=True).clamp(min=1e-12)
            tgt = c - p
        e = self.net(z) - tgt
        loss = 0.5 * e.pow(2).sum(dim=1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """||f-hat(x) + prior(x)|| / sqrt(d) on whitened inputs (whitener NOT updated)."""
        with torch.no_grad():
            z = self._whiten(x.to(self.device), update=False)
            p = self.prior_raw(z)
            p = float(np.sqrt(self.d)) * p / p.norm(dim=1, keepdim=True).clamp(min=1e-12)
            out = self.net(z) + p
            return (out.norm(dim=1) / float(np.sqrt(self.d))).cpu().numpy().astype(np.float64)
