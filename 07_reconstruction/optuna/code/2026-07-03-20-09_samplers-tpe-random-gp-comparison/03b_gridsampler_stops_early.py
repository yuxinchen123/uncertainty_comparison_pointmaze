"""GridSampler over 12 cells stops after 12 trials even when n_trials=60 is requested.

This documents a sharp edge: a plain GridSampler with a 12-point grid does NOT pad out to the
requested 60 trials by repeating cells. It runs each cell once (12 trials) and then stops the
study. To spend a larger budget you must add a repetition dimension to the grid (see
03_sampler_comparison.py, which adds a rep index of 5 to reach 60 trials).
"""

import collections

import numpy as np
import optuna

import landscape as L

# The same 4x3 log grid used in the main comparison, WITHOUT a rep dimension (12 cells).
GRID_RIDGE = [10.0 ** x for x in np.linspace(-6.0, -2.0, 4)]
GRID_BETA = [10.0 ** x for x in np.linspace(-3.0, -1.0, 3)]


# Objective: suggest ridge and beta on a log scale and return the noisy reward.
def objective(trial):
    # ridge and beta on log intervals
    ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
    beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
    # fixed noise seed
    return L.noisy_reward(ridge, beta, noise_seed=11)


# Ask GridSampler for 60 trials over 12 cells and report how many it actually ran.
def main():
    # announce version and quiet logs
    print("optuna", optuna.__version__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # grid of 12 cells; request 60 trials
    sampler = optuna.samplers.GridSampler({"ridge": GRID_RIDGE, "beta": GRID_BETA}, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=60)
    # report: trials requested vs actually run, and reps per cell
    print(f"n_trials requested = 60, grid cells = {len(GRID_RIDGE) * len(GRID_BETA)}")
    print(f"trials actually run = {len(study.trials)}")
    # tally reps per (log10 ridge, log10 beta) cell
    cells = collections.Counter(
        (round(np.log10(t.params["ridge"]), 3), round(np.log10(t.params["beta"]), 3))
        for t in study.trials)
    print(f"distinct cells visited = {len(cells)}")
    print(f"reps per cell = {sorted(cells.values())}")


if __name__ == "__main__":
    main()
