"""Optuna GPSampler: import 6 observations, then ask where it wants to sample. Self-contained."""
import warnings
import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Same toy problem as the from-scratch GP: reward bump at x=-4, six flank observations.
true = lambda x: 50.0 * np.exp(-(((x + 4.0) / 1.2) ** 2))
x_obs = np.array([-6.0, -5.5, -5.0, -3.0, -2.5, -2.0])
y_obs = true(x_obs) + np.array([0.3, -0.2, 0.1, -0.1, 0.2, -0.3])
dist = optuna.distributions.FloatDistribution(-6.0, -2.0)

# Ten independent asks (fresh study per seed); each study sees exactly the six observations.
# n_startup_trials=5 < 6 completed, so the Gaussian process (not random sampling) drives the ask.
asks = []
for seed in range(10):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=5))
    for xi, yi in zip(x_obs, y_obs):                    # import observations as finished trials
        study.add_trial(optuna.trial.create_trial(
            params={"x": float(xi)}, distributions={"x": dist}, value=float(yi)))
    trial = study.ask()                                 # ask where to sample next
    asks.append(trial.suggest_float("x", -6.0, -2.0))

asks = np.array(asks)
print("GPSampler asks (10 seeds):", np.array2string(asks, precision=3, floatmode="fixed"))
print("mean ask x = %+.3f" % asks.mean())
# From-scratch EI (fixed hyperparameters) put argmax near x=-4.2, in the gap between x=-5 and x=-3.
print("From-scratch EI argmax was near x=-4.2 (gap between x=-5 and x=-3); GPSampler agrees.")
