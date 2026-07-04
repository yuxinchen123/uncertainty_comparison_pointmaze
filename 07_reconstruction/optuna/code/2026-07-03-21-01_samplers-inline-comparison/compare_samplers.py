# Four samplers, equal 60-trial budget, on a noisy landscape shaped like the ridge-beta sweep.
import hashlib
import warnings

import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)  # GPSampler is experimental
optuna.logging.set_verbosity(optuna.logging.WARNING)
PEAK = np.array([-6.0, -2.0])  # (log10 ridge, log10 beta) of the peak


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def make_objective(name):
    # noisy bump: height 50, width one decade, additive noise SD 10, keyed per (sampler, trial)
    def objective(trial):
        r = trial.suggest_float("log10_ridge", -6.0, -2.0)
        b = trial.suggest_float("log10_beta", -3.0, -1.0)
        clean = 50.0 * np.exp(-((r - PEAK[0]) ** 2 + (b - PEAK[1]) ** 2) / 2.0)
        return clean + substream(0, name, "noise", trial.number).normal(0.0, 10.0)
    return objective


GRID = {"log10_ridge": list(np.linspace(-6, -2, 4)), "log10_beta": [-3.0, -2.0, -1.0],
        "rep": [0, 1, 2, 3, 4]}  # rep axis makes the 12-cell grid spend all 60 trials


def grid_objective(trial):
    # the grid objective must suggest every grid key, including the rep axis
    trial.suggest_categorical("rep", GRID["rep"])
    return make_objective("GridSampler")(trial)


SAMPLERS = {
    "RandomSampler": optuna.samplers.RandomSampler(seed=0),
    "TPESampler": optuna.samplers.TPESampler(seed=0, n_startup_trials=10),
    "GPSampler": optuna.samplers.GPSampler(seed=0),
    "GridSampler": optuna.samplers.GridSampler(GRID, seed=0),
}
print(f"{'sampler':>14} {'best':>6} {'near-peak trials of 60':>24}")
for name, sampler in SAMPLERS.items():
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(grid_objective if name == "GridSampler" else make_objective(name), n_trials=60)
    pts = np.array([[t.params["log10_ridge"], t.params["log10_beta"]] for t in study.trials])
    near = int(np.sum(np.max(np.abs(pts - PEAK), axis=1) < 0.5))  # within half a decade in BOTH axes
    print(f"{name:>14} {study.best_value:6.1f} {near:>24}")
