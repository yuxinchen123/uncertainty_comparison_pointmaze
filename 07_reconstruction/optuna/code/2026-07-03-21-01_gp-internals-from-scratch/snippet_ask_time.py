"""Measure GPSampler per-ask time at 20 vs 200 completed trials. Self-contained."""
import time, warnings
import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
dist = optuna.distributions.FloatDistribution(-5.0, 5.0)

def time_one_ask(n, seed):
    """Build a study with n completed trials, then time a single GPSampler ask (GP fit + acqf opt)."""
    rng = np.random.default_rng(seed)                       # reproducible params for this study
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=5))
    a, b = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)     # random 2-D params
    for ai, bi in zip(a, b):                                # objective: negative sphere (max at origin)
        study.add_trial(optuna.trial.create_trial(
            params={"a": float(ai), "b": float(bi)},
            distributions={"a": dist, "b": dist}, value=float(-(ai**2 + bi**2))))
    t0 = time.perf_counter()                                # time only the ask, not the setup
    trial = study.ask()
    trial.suggest_float("a", -5, 5); trial.suggest_float("b", -5, 5)
    return time.perf_counter() - t0

time_one_ask(20, 999)                                       # untimed warmup (lazy torch/scipy import)
for n in (20, 200):
    times = [time_one_ask(n, s) for s in range(3)]
    print("N_completed=%3d : per-ask time min=%.3f s  mean=%.3f s" % (n, min(times), np.mean(times)))
