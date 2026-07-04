"""TPESampler on a mixed search space: one categorical + one log float in the same objective.

The categorical is feature normalization in {none, unit}; "unit" is made the better choice by
adding a bonus to the reward. We confirm TPE runs on the mixed space and that its categorical
suggestions concentrate on the better category: we print the counts of each category in the
first 15 vs the last 15 of 60 trials.
"""

import collections

import numpy as np
import optuna

import landscape as L

# How much extra reward the better normalization ("unit") earns, so TPE has a signal to find.
UNIT_BONUS = 25.0


# Objective over {normalization category, ridge log float}: "unit" is worth an extra bonus.
def objective(trial):
    # categorical knob: which feature normalization
    norm = trial.suggest_categorical("normalization", ["none", "unit"])
    # log float knob: ridge lambda, peak-favored near 1e-6
    ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
    # base reward from the ridge bump at the peak beta, plus a bonus only when normalization is "unit"
    base = L.clean_reward(np.log10(ridge), L.PEAK_LOG10_BETA)
    bonus = UNIT_BONUS if norm == "unit" else 0.0
    # one reproducible noise draw keyed by the exact (norm, ridge) point
    rng = L.substream(0, "mixed_reward", norm, round(float(np.log10(ridge)), 6))
    return float(base + bonus + rng.normal(0.0, L.NOISE_SD))


# Run TPE on the mixed space and report category counts in the first vs last 15 trials.
def main():
    # announce version and quiet logs
    print("optuna", optuna.__version__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # TPE with an explicit seed and the default 10 random startup trials
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
    # 60 trials so the last 15 are well past the 10 startup trials
    study.optimize(objective, n_trials=60)
    # tally the categorical in the first 15 and last 15 trials
    first15 = collections.Counter(t.params["normalization"] for t in study.trials[:15])
    last15 = collections.Counter(t.params["normalization"] for t in study.trials[-15:])
    # report
    print(f"better category is 'unit' (worth +{UNIT_BONUS:.0f} reward)")
    print(f"first 15 trials: none={first15['none']:2d}  unit={first15['unit']:2d}")
    print(f"last  15 trials: none={last15['none']:2d}  unit={last15['unit']:2d}")
    print(f"best trial normalization = {study.best_params['normalization']!r}, "
          f"best_value = {study.best_value:.3f}")


if __name__ == "__main__":
    main()
