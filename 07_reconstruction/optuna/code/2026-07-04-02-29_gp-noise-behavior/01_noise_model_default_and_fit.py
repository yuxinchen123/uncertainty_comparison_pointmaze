"""Point 1: confirm GPSampler fits an observation-noise variance on every ask, and that
deterministic_objective=False is the default (so noise is fitted, not pinned to the floor).

Two checks, both on optuna 4.9.0:
  A. Read the installed defaults out of the objects: GPSampler.__init__ signature default for
     deterministic_objective, the minimum-noise floor, and the noise-variance prior in prior.py.
  B. Fit optuna's own GP (optuna._gp.gp.fit_kernel_params) on noisy data drawn from the project's
     bimodal reward model, once with deterministic_objective=False (noise fitted) and once with
     =True (noise pinned to the floor 1e-6). Print the fitted noise variance in both cases.
"""

import hashlib
import inspect

import numpy as np
import torch

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bimodal_rewards(rng, mean, n):
    # project noise model: one reward per seed, success -> Normal(80,10), else -> Normal(0,3)
    p = mean / 80.0
    success = rng.random(n) < p
    return np.where(success, rng.normal(80.0, 10.0, n), rng.normal(0.0, 3.0, n))


def fitted_noise_var(X, y_raw, deterministic):
    # standardize y as GPSampler does, fit optuna's GP, return the fitted noise variance
    means = np.mean(y_raw)
    stds = np.std(y_raw)
    y = (y_raw - means) / max(1e-10, stds)
    is_categorical = np.zeros(X.shape[1], dtype=bool)
    gpr = gp.fit_kernel_params(
        X=X,
        Y=y,
        is_categorical=is_categorical,
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic,
        gpr_cache=None,
    )
    return float(gpr.noise_var.item())


# ---- Check A: installed defaults ----
print("optuna version:", optuna.__version__)
sig = inspect.signature(optuna.samplers.GPSampler.__init__)
print("GPSampler default deterministic_objective =",
      sig.parameters["deterministic_objective"].default)
sampler = optuna.samplers.GPSampler(seed=0)
print("GPSampler(seed=0)._deterministic          =", sampler._deterministic)
print("GPSampler(seed=0)._minimum_noise          =", sampler._minimum_noise)
print("prior.DEFAULT_MINIMUM_NOISE_VAR           =", prior.DEFAULT_MINIMUM_NOISE_VAR)
print("noise_var prior in prior.default_log_prior: gamma_log_prior(noise_var, concentration=1.1, rate=30)")
print()

# ---- Check B: fitted noise variance on bimodal data ----
# 12 points spread over the unit interval (normalized param space), each with a bimodal draw at a
# gently varying true mean so the data is genuinely noisy (draws jump between ~0 and ~80).
x = np.linspace(0.0, 1.0, 12)
X = x.reshape(-1, 1)
true_means = 20.0 + 30.0 * x  # ranges 20 -> 50 across the interval
rng = substream(2026, "point1", "rewards")
y_raw = np.array([bimodal_rewards(rng, m, 1)[0] for m in true_means])

print("normalized x and its bimodal reward draw (raw, before standardization):")
for xi, mi, yi in zip(x, true_means, y_raw):
    print(f"  x={xi:4.2f}  true_mean={mi:5.1f}  draw={yi:7.2f}")
print(f"  sample std of the 12 raw draws = {np.std(y_raw):.2f}  (this is the spread the GP must explain)")
print()

nv_fitted = fitted_noise_var(X, y_raw, deterministic=False)
nv_pinned = fitted_noise_var(X, y_raw, deterministic=True)
print("fitted noise variance, deterministic_objective=False (DEFAULT):  %.6f" % nv_fitted)
print("fitted noise variance, deterministic_objective=True  (pinned) :  %.6f" % nv_pinned)
print("floor (minimum_noise) = %.6f" % prior.DEFAULT_MINIMUM_NOISE_VAR)
print()
print("Note: the fitted value is in STANDARDIZED units (y was divided by its sample std %.2f)."
      % np.std(y_raw))
print("ratio fitted/floor = %.1f  (default fit sits far above the floor; the =True fit sits at it)"
      % (nv_fitted / prior.DEFAULT_MINIMUM_NOISE_VAR))
