"""Coin-flip pseudo-count bonus with an exact recursive-least-squares head (11_decay_rate
campaign winner, adapted for the RL loop; construction of Lobel, Bagaria and Konidaris,
"Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning", ICML
2023, made exact by closed-form ridge least squares).

Every environment visit (buffer add) draws a fresh Rademacher vector and accumulates the
normal equations of a regression from localized kernel features of the visited position onto
(coin - prior). The least-squares optimum at a position visited m times is that position's own
running coin mean, whose norm behaves as m^(-1/2) with THAT position's count under any
visitation pattern; the frozen prior is unit-normalized per input and the head starts at zero,
so the bonus is exactly 1 at every never-visited input. The 11_decay_rate development document
measures this construction against the count oracle's curve min(1, m^(-1/2)) per position.

RL-loop adaptations relative to the campaign version: counting fires at ADD time (environment
visitation, like the elliptical bonus's update_timing="add"), not on replay draws; the kernel
dictionary works at maze-cell granularity (insertion radius 0.5, the oracle VisitCount's cell
scale) so the pseudo-count matches gt_position's resolution; the head solve is cached and
refreshed every SOLVE_EVERY visits (compute() runs on every SAC gradient step and must not
pay a matrix solve each time).
"""
import hashlib
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from .base import IntrinsicRewardModel


def _keyed_gen(*parts) -> torch.Generator:
    """One torch.Generator seeded by a hash of stable names (reproducible-seeding rule)."""
    key = "::".join(str(p) for p in parts)
    g = torch.Generator()
    g.manual_seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0x7FFFFFFF)
    return g


class CoinFlipCount(IntrinsicRewardModel):
    """Coin-flip pseudo-count bonus on a growing kernel dictionary (exact RLS head)."""

    def __init__(
        self,
        obs_shape,
        d_coins: int = 128,
        tau_add: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 0.5,
        max_centers: int = 512,
        solve_every: int = 256,
        ridge: float = 1e-8,
        seed: int = 0,
    ):
        """Build the frozen unit-normalized prior and the empty dictionary/normal-equation
        state. Positions are the first two observation coordinates (x, y), raw."""
        obs_dim = obs_shape[0] if isinstance(obs_shape, (tuple, list)) else int(obs_shape)
        self.d = int(d_coins)
        self.tau_add = float(tau_add)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.max_centers = int(max_centers)
        self.solve_every = int(solve_every)
        self.ridge = float(ridge)
        # frozen prior MLP: orthogonal weights gain sqrt(2) plus N(0, 0.5^2) biases (the
        # project's normal_0.5 law) — the biases matter: a zero-bias ReLU net outputs the
        # zero vector at the origin input, which would make the unit-normalized prior (and
        # hence the never-visited bonus) degenerate exactly there
        self.prior_raw = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                       nn.Linear(256, self.d))
        for idx, layer in enumerate(self.prior_raw):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=float(np.sqrt(2.0)),
                                    generator=_keyed_gen(seed, "coinflip_prior", idx))
                with torch.no_grad():
                    layer.bias.copy_(0.5 * torch.randn(
                        layer.bias.shape, generator=_keyed_gen(seed, "coinflip_prior_bias", idx)))
        for p in self.prior_raw.parameters():
            p.requires_grad_(False)
        self.coin_gen = _keyed_gen(seed, "coinflip_coins")
        self.centers = torch.zeros(0, 2, dtype=torch.float64)
        self.sigmas = torch.zeros(0, dtype=torch.float64)
        self.Lam = torch.zeros(0, 0, dtype=torch.float64)
        self.Bmat = torch.zeros(0, self.d, dtype=torch.float64)
        self.W = None                # cached head solve; None = dictionary still empty
        self._adds_since_solve = 0

    def _grow(self, z: torch.Tensor) -> None:
        """Insert new centers for rows farther than tau_add from the dictionary (sequential,
        so a row inserted first covers later rows of the same call)."""
        for row in z:
            if self.centers.shape[0] >= self.max_centers:
                return
            if self.centers.shape[0] == 0:
                d_nn = float("inf")
            else:
                d_nn = float(((self.centers - row) ** 2).sum(-1).min().sqrt())
            if d_nn > self.tau_add:
                sig = min(self.sigma_max, max(self.sigma_min, 0.5 * d_nn))
                self.centers = torch.cat([self.centers, row[None, :]])
                self.sigmas = torch.cat([self.sigmas,
                                         torch.tensor([sig], dtype=torch.float64)])
                k = self.Lam.shape[0]
                lam = torch.zeros(k + 1, k + 1, dtype=torch.float64)
                lam[:k, :k] = self.Lam
                self.Lam = lam
                self.Bmat = torch.cat([self.Bmat,
                                       torch.zeros(1, self.d, dtype=torch.float64)])
                self.W = None  # dictionary changed shape: force a re-solve

    def _feat(self, z: torch.Tensor) -> torch.Tensor:
        """Per-center Gaussian features of (B, 2) float64 positions."""
        d2 = ((z[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)
        return torch.exp(-d2 / (2.0 * self.sigmas[None, :] ** 2))

    def _prior(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen prior with ||prior(x)|| = sqrt(d) exactly at every input."""
        p = self.prior_raw(x).double()
        return float(np.sqrt(self.d)) * p / p.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def _positions(self, samples: Dict[str, Any]) -> torch.Tensor:
        """(B, 2) float64 raw positions from the samples' next observations."""
        nxt = samples["next_observations"]
        t = torch.as_tensor(np.asarray(nxt), dtype=torch.float64)
        return t.reshape(t.shape[0], -1)[:, :2]

    def observe(self, samples: Dict[str, Any]) -> None:
        """One VISIT per freshly-added transition: grow the dictionary, draw fresh coins,
        accumulate the normal equations (called by the buffer on add())."""
        with torch.no_grad():
            z = self._positions(samples)
            self._grow(z)
            phi = self._feat(z)
            c = torch.randint(0, 2, (z.shape[0], self.d), generator=self.coin_gen,
                              dtype=torch.float64) * 2.0 - 1.0
            x_full = torch.as_tensor(np.asarray(samples["next_observations"]),
                                     dtype=torch.float32).reshape(z.shape[0], -1)
            tgt = c - self._prior(x_full)
            self.Lam += phi.T @ phi
            self.Bmat += phi.T @ tgt
            self._adds_since_solve += z.shape[0]

    def _head(self) -> torch.Tensor:
        """The cached ridge solve, refreshed every solve_every visits."""
        if self.W is None or self._adds_since_solve >= self.solve_every:
            lam = self.Lam + self.ridge * torch.eye(self.Lam.shape[0], dtype=torch.float64)
            self.W = torch.linalg.solve(lam, self.Bmat)
            self._adds_since_solve = 0
        return self.W

    def compute(self, samples: Dict[str, Any]) -> np.ndarray:
        """Bonus per sample: ||head(x) + prior(x)|| / sqrt(d); exactly 1 before any visit."""
        with torch.no_grad():
            z = self._positions(samples)
            x_full = torch.as_tensor(np.asarray(samples["next_observations"]),
                                     dtype=torch.float32).reshape(z.shape[0], -1)
            pr = self._prior(x_full)
            if self.centers.shape[0] == 0:
                out = pr
            else:
                out = self._feat(z) @ self._head() + pr
            return (out.norm(dim=1) / float(np.sqrt(self.d))).cpu().numpy()

    def update(self, samples: Dict[str, Any]) -> None:
        """No replay-time learning: all statistics accumulate at add time (observe)."""
