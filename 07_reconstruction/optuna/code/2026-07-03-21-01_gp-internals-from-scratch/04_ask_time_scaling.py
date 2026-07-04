"""Measure how GPSampler's per-ask wall time grows with the number of completed trials, 20 vs 200.

Each ask refits the GP (a Cholesky factorization of an N-by-N matrix, O(N^3)) and optimizes the
acquisition function (evaluate the posterior over N points at 2048 QMC candidates plus local search),
so the cost per ask rises with the count of completed trials N. We also confirm the torch dependency.
"""

import hashlib
import time
import warnings

import numpy as np

import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def substream(base_seed, *parts):
    """Return an independent numpy generator keyed by a stable name (per-quantity seeding)."""
    # Hash a stable string of the parts into a 32-bit seed for a dedicated generator.
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


# Two float parameters; objective is a smooth bowl so the GP has real structure to fit.
DIST = optuna.distributions.FloatDistribution(-5.0, 5.0)


def build_study_with_completed(n_completed, seed):
    """Create a maximize study with GPSampler and import n_completed random finished trials."""
    # Fresh study; n_startup_trials small so the GP engages at both N=20 and N=200.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=5),
    )
    # Draw parameter values and objective values from named substreams (reproducible, independent).
    ga = substream(seed, "param_a", n_completed)
    gb = substream(seed, "param_b", n_completed)
    a = ga.uniform(-5.0, 5.0, size=n_completed)
    b = gb.uniform(-5.0, 5.0, size=n_completed)
    # Objective: negative sphere (maximized at the origin), deterministic given the params.
    y = -(a**2 + b**2)
    # Import each row as a finished trial.
    for ai, bi, yi in zip(a, b, y):
        study.add_trial(
            optuna.trial.create_trial(
                params={"a": float(ai), "b": float(bi)},
                distributions={"a": DIST, "b": DIST},
                value=float(yi),
            )
        )
    return study


def time_one_ask(n_completed, seed):
    """Return seconds for a single GPSampler ask (GP fit + acquisition optimization) at N trials."""
    # Build the study first (not timed), then time exactly one ask that suggests both params.
    study = build_study_with_completed(n_completed, seed)
    t0 = time.perf_counter()
    trial = study.ask()
    trial.suggest_float("a", -5.0, 5.0)
    trial.suggest_float("b", -5.0, 5.0)
    return time.perf_counter() - t0


# Confirm torch is present and used by the GP module.
import torch  # noqa: E402

print("torch version:", torch.__version__)
print("GPSampler requires scipy and torch (from the class docstring, verified in 01_read_gp_source).")
print()

# One untimed warmup ask: triggers the lazy torch/scipy imports and first-call JIT so the timings
# below measure the actual GP fit + acquisition optimization, not one-time library startup.
time_one_ask(20, seed=999)

# Time three fresh asks at each size to report a stable minimum and mean (first ask is cold, no cache).
for n in (20, 200):
    times = [time_one_ask(n, seed) for seed in range(3)]
    print("N_completed=%3d : ask times (s) = %s ; min=%.3f mean=%.3f"
          % (n, ["%.3f" % t for t in times], float(np.min(times)), float(np.mean(times))))

# Report the ratio so the growth with N is explicit.
t20 = float(np.min([time_one_ask(20, s) for s in range(3)]))
t200 = float(np.min([time_one_ask(200, s) for s in range(3)]))
print()
print("min ask time at N=20  : %.3f s" % t20)
print("min ask time at N=200 : %.3f s" % t200)
print("ratio (200 vs 20)     : %.1fx" % (t200 / t20))
