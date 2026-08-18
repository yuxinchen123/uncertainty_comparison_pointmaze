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
    """Replay-buffer coin-flip network: the published CFN training regime (gradient steps over
    a replay of stored (state, coin) pairs), measured against the count oracle.

    Every visit appends (x, fresh Rademacher coin) to a ring buffer (capacity 150k pairs) and
    runs K = 4 Adam minibatch steps (batch 256) on the regression toward (coin - prior). The
    least-squares optimum over the buffer is each state's running mean of its BUFFERED coins,
    so two gaps separate this from the exact head: the optimization lag of gradient steps, and
    the ring's forgetting (a state's effective count saturates near capacity / distinct
    states). Zero-initialized output layer + unit-normalized frozen prior keep the bonus
    exactly 1 at never-visited inputs; whitener runs online as in the plain gradient CFN."""

    name = "cfn_replay_adam1e-3"

    D_COINS = 512
    RING = 150_000
    K_STEPS = 4
    BATCH = 256
    LR = 1e-3

    def __init__(self, seed: int, obs_dim=4):
        """Trainable net (zero head), frozen unit-normalized prior, ring buffer, Adam."""
        torch.manual_seed(seed)  # belt-and-braces; every draw below uses keyed generators
        self.is_image = not isinstance(obs_dim, int)
        self.d = 256 if self.is_image else self.D_COINS
        self.net = make_trunk(obs_dim, self.d, seed, "predictor")
        last = [m for m in self.net if isinstance(m, (nn.Linear, nn.Conv2d))][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.prior_raw = make_trunk(obs_dim, self.d, seed, "prior")
        for p in self.prior_raw.parameters():
            p.requires_grad_(False)
        self.device = torch.device(os.environ.get("METHOD_DEVICE", "cpu"))
        self.net.to(self.device)
        self.prior_raw.to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.coin_gen = keyed_gen(seed, "coins")
        self.sample_gen = keyed_gen(seed, "replay")
        self.buf_x = None      # ring storage, allocated on first visit
        self.buf_c = None
        self.buf_n = 0         # total pairs ever appended (ring position = n mod RING)
        self._count = 1e-4
        self._mean = None
        self._var = None

    def _whiten(self, x: torch.Tensor, update: bool) -> torch.Tensor:
        """Running mean/std whitening with a +-5 clip (see the plain gradient CFN)."""
        z = x.to(self.device)
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
        if self._mean is not None:
            z = ((z - self._mean) / (self._var + 1e-8).sqrt()).clamp(-5.0, 5.0)
        return z.unsqueeze(1) if self.is_image else z

    def _prior(self, z: torch.Tensor) -> torch.Tensor:
        """Frozen prior with ||prior(x)|| = sqrt(d) exactly at every input."""
        p = self.prior_raw(z)
        return float(np.sqrt(self.d)) * p / p.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def _append(self, x: torch.Tensor, c: torch.Tensor) -> None:
        """Ring append of raw (x, coin) rows. before: buf_n=150000 (full) -> new rows
        overwrite positions 150000 mod RING onward (the oldest entries)."""
        if self.buf_x is None:
            shape = (self.RING,) + tuple(x.shape[1:])
            self.buf_x = torch.zeros(shape, dtype=x.dtype)
            self.buf_c = torch.zeros(self.RING, self.d, dtype=torch.float32)
        idx = (torch.arange(x.shape[0]) + self.buf_n) % self.RING
        self.buf_x[idx] = x.cpu()
        self.buf_c[idx] = c.cpu()
        self.buf_n += x.shape[0]

    def update(self, x: torch.Tensor) -> None:
        """Append fresh coins for the visited batch, then K Adam minibatch replay steps."""
        with torch.no_grad():
            self._whiten(x, update=True)   # whitener statistics track visited batches
            c = (torch.randint(0, 2, (x.shape[0], self.d), generator=self.coin_gen,
                               dtype=torch.float32) * 2.0 - 1.0)
            self._append(x, c)
        live = min(self.buf_n, self.RING)
        for _ in range(self.K_STEPS):
            idx = torch.randint(0, live, (min(self.BATCH, live),), generator=self.sample_gen)
            xb = self.buf_x[idx].to(self.device)
            cb = self.buf_c[idx].to(self.device)
            z = self._whiten(xb, update=False)
            with torch.no_grad():
                tgt = cb - self._prior(z)
            e = self.net(z) - tgt
            loss = 0.5 * e.pow(2).sum(dim=1).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def bonus(self, x: torch.Tensor) -> np.ndarray:
        """||net(x) + prior(x)|| / sqrt(d) on whitened inputs (whitener not updated)."""
        with torch.no_grad():
            z = self._whiten(x, update=False)
            out = self.net(z) + self._prior(z)
            return (out.norm(dim=1) / float(np.sqrt(self.d))).cpu().numpy().astype(np.float64)
