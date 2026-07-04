"""Check that optuna 4.9.0 GPSampler asks land where the from-scratch EI landscape is highest.

Same 1-D problem as 02_gp_ei_from_scratch.py: x = log10(ridge) on [-6,-2], reward bump at x=-4,
the six flank observations imported via study.add_trial. We then ask GPSampler where to sample and
compare against the from-scratch EI ranking. Exact agreement is NOT expected: optuna standardizes y,
fits its own kernel hyperparameters by marginal-likelihood, and uses logEI, not the fixed-hyper EI here.
"""

import warnings
from math import erf

import numpy as np

import optuna
from optuna.exceptions import ExperimentalWarning

# Quiet the experimental-API warning and per-trial logging so the comparison table is readable.
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----- identical toy setup to the from-scratch script -----
TRUE = lambda x: 50.0 * np.exp(-(((x + 4.0) / 1.2) ** 2))
X_OBS = np.array([-6.0, -5.5, -5.0, -3.0, -2.5, -2.0])
NOISE_OFFSETS = np.array([+0.3, -0.2, +0.1, -0.1, +0.2, -0.3])
Y_OBS = TRUE(X_OBS) + NOISE_OFFSETS
S2, L, NOISE_VAR = 400.0, 0.8, 1.0
DIST = optuna.distributions.FloatDistribution(-6.0, -2.0)


def matern52(xa, xb, s2, ell):
    """Matern 5/2 covariance matrix between 1-D input vectors xa and xb."""
    # Pairwise distances then the closed-form Matern 5/2 expression.
    r = np.abs(xa[:, None] - xb[None, :])
    s = np.sqrt(5.0) * r / ell
    return s2 * (1.0 + s + (s * s) / 3.0) * np.exp(-s)


def ei_landscape(x_grid):
    """Return the from-scratch EI over x_grid, reusing the fixed-hyperparameter GP of script 02."""
    # Posterior mean/std via Cholesky solves.
    K = matern52(X_OBS, X_OBS, S2, L) + NOISE_VAR * np.eye(len(X_OBS))
    Lc = np.linalg.cholesky(K)
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, Y_OBS))
    Ks = matern52(x_grid, X_OBS, S2, L)
    mu = Ks @ alpha
    v = np.linalg.solve(Lc, Ks.T)
    sigma = np.sqrt(np.clip(S2 - np.sum(v * v, axis=0), 1e-12, None))
    # Closed-form EI against the incumbent.
    f_best = Y_OBS.max()
    z = (mu - f_best) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    return (mu - f_best) * Phi + sigma * phi


def seed_study(seed):
    """Create a maximize study with GPSampler and import the six fixed observations."""
    # Fresh study; n_startup_trials=5 < 6 completed, so the GP (not random) drives the ask.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=5),
    )
    # Import each observation as a finished trial with a fixed param value and reward.
    for xi, yi in zip(X_OBS, Y_OBS):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(xi)}, distributions={"x": DIST}, value=float(yi)
            )
        )
    return study


# From-scratch EI landscape on a fine grid: argmax and the high-EI band (EI >= 0.9 * max).
grid = np.round(np.arange(-6.0, -2.0 + 1e-9, 0.02), 4)
ei = ei_landscape(grid)
argmax_x = grid[int(np.argmax(ei))]
band = grid[ei >= 0.9 * ei.max()]
band_lo, band_hi = band.min(), band.max()
print("From-scratch EI: argmax at x=%+.2f ; high-EI band (EI>=0.9*max) = [%+.2f, %+.2f]"
      % (argmax_x, band_lo, band_hi))
print()

# --- Test A: 10 independent single asks (fresh study per seed) ---
# Each fresh study sees exactly the 6 observations, so each ask is the GP's argmax-EI suggestion.
print("Test A: 10 independent GPSampler asks, one per seed (fresh study each, 6 obs, no pending):")
asks_A = []
for seed in range(10):
    study = seed_study(seed)
    trial = study.ask()
    asks_A.append(trial.suggest_float("x", -6.0, -2.0))
asks_A = np.array(asks_A)
in_band_A = int(np.sum((asks_A >= band_lo) & (asks_A <= band_hi)))
for seed, xa in enumerate(asks_A):
    print("  seed %d -> ask x=%+.3f  %s" % (seed, xa,
          "(in high-EI band)" if band_lo <= xa <= band_hi else "(outside band)"))
print("  asks sorted:", np.array2string(np.sort(asks_A), precision=3, floatmode="fixed"))
print("  mean ask x = %+.3f ; in high-EI band: %d / 10" % (asks_A.mean(), in_band_A))
print()

# --- Test B: 10 sequential asks on ONE study without telling results ---
# Un-told asks become RUNNING trials; GPSampler applies a constant-liar to pending points, so the
# later asks are pushed to spread out rather than repeating the single argmax.
print("Test B: 10 sequential asks on one study WITHOUT tell (pending trials accumulate):")
study = seed_study(123)
asks_B = []
for i in range(10):
    trial = study.ask()
    asks_B.append(trial.suggest_float("x", -6.0, -2.0))
asks_B = np.array(asks_B)
in_band_B = int(np.sum((asks_B >= band_lo) & (asks_B <= band_hi)))
for i, xb in enumerate(asks_B):
    print("  ask %2d -> x=%+.3f  %s" % (i, xb,
          "(in high-EI band)" if band_lo <= xb <= band_hi else "(outside band)"))
print("  in high-EI band: %d / 10 (spread wider than Test A due to pending-trial constant-liar)"
      % in_band_B)
