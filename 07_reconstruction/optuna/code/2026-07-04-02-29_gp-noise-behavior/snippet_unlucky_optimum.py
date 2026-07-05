"""Three unlucky draws at the true optimum: fitted noise keeps the GP's estimate there high.

Landscape m(x) = 50*exp(-(x-8)^2/2), optimum at x=8. We give the GP 20 informative points that
reveal the bump, plus 3 forced ~0 draws sitting exactly at x=8 (three unlucky seeds). With the noise
fitted (the default) the posterior mean at x=8 stays well above 0; with the noise pinned to the
floor (deterministic_objective=True) it is dragged down onto those zeros.
"""
import hashlib

import numpy as np
import torch

import optuna._gp.gp as gp
import optuna._gp.prior as prior


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def true_mean(x):
    # reward-mean landscape: a Gaussian bump peaking at x=8 with height 50
    return 50.0 * np.exp(-((x - 8.0) ** 2) / 2.0)


def posterior_mean_at_8(X, y_raw, deterministic):
    # fit optuna's GP the way GPSampler does; return the posterior mean at x=8 in raw reward units
    m, s = y_raw.mean(), max(1e-10, y_raw.std())
    gpr = gp.fit_kernel_params(
        X=X, Y=(y_raw - m) / s, is_categorical=np.zeros(1, dtype=bool),
        log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic, gpr_cache=None)
    pm, _ = gpr.posterior(torch.from_numpy(np.array([8.0 / 10.0])))  # x normalized to [0,1]
    return float(pm.item()) * s + m


# 20 informative points at random x valued at the true mean, plus 3 forced ~0 draws at x=8.
xr = substream(0, "informative_x").uniform(0, 10, 20)
X = np.concatenate([xr, [8.0, 8.0, 8.0]]).reshape(-1, 1) / 10.0  # normalize params to [0,1]
zeros = np.abs(substream(0, "forced_zeros").normal(0, 1, 3))
y_raw = np.concatenate([true_mean(xr), zeros])

print("true mean at the optimum m(8) =", round(float(true_mean(8.0)), 2))
print("the 3 forced draws at x=8     =", np.array2string(zeros, precision=2))
print("posterior mean at x=8, noise FITTED (default)      = %.2f" % posterior_mean_at_8(X, y_raw, False))
print("posterior mean at x=8, noise PINNED (det=True)     = %.2f" % posterior_mean_at_8(X, y_raw, True))
print("Fitted noise keeps the optimum's estimate positive; pinned noise sits on the unlucky zeros.")
