"""GPSampler fits an observation-noise variance on every ask; the default is NOT deterministic."""
import hashlib
import inspect

import numpy as np

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior

optuna.logging.set_verbosity(optuna.logging.WARNING)


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bimodal(rng, mean, n):
    # project noise model: success -> Normal(80,10), else -> Normal(0,3)
    hit = rng.random(n) < mean / 80.0
    return np.where(hit, rng.normal(80, 10, n), rng.normal(0, 3, n))


def fitted_noise_var(X, y_raw, deterministic):
    # standardize y as GPSampler does, fit optuna's own GP, return the fitted noise variance
    y = (y_raw - y_raw.mean()) / max(1e-10, y_raw.std())
    gpr = gp.fit_kernel_params(
        X=X, Y=y, is_categorical=np.zeros(X.shape[1], dtype=bool),
        log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic, gpr_cache=None)
    return float(gpr.noise_var.item())


# The default: noise is fitted, not pinned to the floor.
default = inspect.signature(optuna.samplers.GPSampler.__init__).parameters["deterministic_objective"].default
print("GPSampler default deterministic_objective =", default)
print("noise floor (minimum_noise) =", prior.DEFAULT_MINIMUM_NOISE_VAR)
print("noise_var prior = Gamma(concentration=1.1, rate=30)   # optuna/_gp/prior.py")
print()

# 12 points across the unit interval, each one bimodal draw (jumps between ~0 and ~80).
x = np.linspace(0, 1, 12)
true_means = 20 + 30 * x
y_raw = np.array([bimodal(substream(0, "y", i), m, 1)[0] for i, m in enumerate(true_means)])
X = x.reshape(-1, 1)

print("fitted noise_var, deterministic_objective=False (DEFAULT) = %.4f" % fitted_noise_var(X, y_raw, False))
print("fitted noise_var, deterministic_objective=True  (pinned)  = %.6f" % fitted_noise_var(X, y_raw, True))
print("(values are in standardized units: y was divided by its sample std %.1f)" % y_raw.std())
