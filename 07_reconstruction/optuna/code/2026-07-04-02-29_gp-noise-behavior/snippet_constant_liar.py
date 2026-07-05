"""Running (not-yet-finished) trials are imputed with the best completed value (constant liar)."""
import hashlib

import numpy as np

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior
import optuna._gp.search_space as gp_search_space
from optuna._gp.acqf import LogEI

optuna.logging.set_verbosity(optuna.logging.WARNING)
DIST = optuna.distributions.FloatDistribution(0.0, 10.0)


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


# Fit optuna's GP on 30 toy observations, then build the exact LogEI acquisition GPSampler uses.
rng = substream(0, "obs")
x = rng.uniform(0, 10, 30)
y_raw = rng.normal(0, 1, 30)
y = (y_raw - y_raw.mean()) / max(1e-10, y_raw.std())
gpr = gp.fit_kernel_params(
    X=(x / 10.0).reshape(-1, 1), Y=y, is_categorical=np.zeros(1, dtype=bool),
    log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
    deterministic_objective=False, gpr_cache=None)

# Two pretend running trials; LogEI appends imputed values for them at construction time.
running = np.array([[0.42], [0.77]])
acqf = LogEI(gpr, gp_search_space.SearchSpace({"x": DIST}),
             threshold=float(y.max()), normalized_params_of_running_trials=running)
imputed = acqf._gpr._y_all[-running.shape[0]:].detach().numpy()

print("best completed value (standardized) =", round(float(y.max()), 4))
print("value LogEI assigned to running pts  =", np.array2string(imputed, precision=4))
print("equal to the best value?             =", bool(np.allclose(imputed, y.max())))
print("source: optuna/_gp/acqf.py -> LogEI: constant_liar_value = self._gpr._y_train.max()")
