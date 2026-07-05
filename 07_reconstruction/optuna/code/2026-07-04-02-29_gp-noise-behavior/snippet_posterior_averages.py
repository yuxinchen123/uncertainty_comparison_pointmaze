"""The GP posterior at a point tracks the MEAN of repeated noisy draws there, not the last draw."""
import hashlib

import numpy as np
import torch

import optuna._gp.gp as gp
import optuna._gp.prior as prior


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bimodal(rng, mean, n):
    # project noise model: success -> Normal(80,10), else -> Normal(0,3)
    hit = rng.random(n) < mean / 80.0
    return np.where(hit, rng.normal(80, 10, n), rng.normal(0, 3, n))


def posterior_mean_raw(X, y_raw, x_query, deterministic):
    # fit optuna's GP the way GPSampler does; return posterior mean at x_query in raw reward units
    m, s = y_raw.mean(), max(1e-10, y_raw.std())
    gpr = gp.fit_kernel_params(
        X=X, Y=(y_raw - m) / s, is_categorical=np.zeros(X.shape[1], dtype=bool),
        log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic, gpr_cache=None)
    pm, _ = gpr.posterior(torch.from_numpy(np.asarray(x_query, dtype=np.float64)))
    return float(pm.item()) * s + m


# Three anchor points elsewhere give the GP spatial structure; x_target=0.5 (true mean 30).
anchor_x = np.array([0.05, 0.30, 0.95])
anchor_y = np.array([bimodal(substream(0, "anchor", i), mu, 1)[0] for i, mu in enumerate([10, 60, 5])])
draws = bimodal(substream(0, "target"), 30.0, 10)  # 10 draws at x_target, true mean 30
print("10 draws at x_target (true mean 30):", np.array2string(draws, precision=1))
print()
print(f"{'k':>3} {'sample_mean':>12} {'last_draw':>10} {'post_mean (fitted noise)':>25} {'post_mean (det=True)':>21}")
for k in (1, 3, 10):
    # stack k draws at x_target=0.5 plus the 3 anchors, read posterior mean back at x_target
    X = np.vstack([np.full((k, 1), 0.5), anchor_x.reshape(-1, 1)])
    y = np.concatenate([draws[:k], anchor_y])
    fit = posterior_mean_raw(X, y, np.array([0.5]), deterministic=False)
    det = posterior_mean_raw(X, y, np.array([0.5]), deterministic=True)
    print(f"{k:>3} {draws[:k].mean():>12.2f} {draws[k-1]:>10.2f} {fit:>25.2f} {det:>21.2f}")
print()
print("At k=10 the fitted-noise posterior mean tracks the sample mean, far from the last draw.")
