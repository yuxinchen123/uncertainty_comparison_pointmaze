"""Compare four Optuna samplers at an equal budget of 60 trials on the toy sweep landscape.

Samplers: RandomSampler(seed=0), TPESampler(seed=0, n_startup_trials=10), GPSampler(seed=0),
and GridSampler over a 4x3 log grid (12 cells) with a rep dimension of 5 so it runs a real
60 trials (each cell 5 times). For each sampler and each of three noise seeds we report the
best observed value, the params of the best trial, and the count of trials that landed within
half a decade of the true peak in BOTH axes. Higher reward is better, so every study maximizes.

Note on GridSampler: without the rep dimension, GridSampler stops after the 12 cells are
covered (it does not pad to n_trials=60). The rep dimension is what makes it spend the full
budget. On this landscape the reward is fixed once (ridge, beta) is fixed, so the 5 reps of a
cell return identical values -- the reps spend budget without adding information. This mirrors
the project sweep's grid but shows a real limitation of grid on a deterministic objective.
"""

import numpy as np
import optuna

import landscape as L

# Budget and the noise seeds we repeat the whole comparison over.
N_TRIALS = 60
NOISE_SEEDS = [11, 22, 33]

# 4x3 log grid: 4 ridge decades across [1e-6, 1e-2], 3 beta decades across [1e-3, 1e-1].
GRID_RIDGE = [10.0 ** x for x in np.linspace(-6.0, -2.0, 4)]
GRID_BETA = [10.0 ** x for x in np.linspace(-3.0, -1.0, 3)]
# rep index 0..4 turns the 12-cell grid into 60 trials (each cell evaluated 5 times).
GRID_REP = [0, 1, 2, 3, 4]


# Objective for the continuous samplers: suggest ridge and beta on a log scale, return noisy reward.
def make_objective(noise_seed):
    # inner objective closes over the noise seed so reruns are reproducible
    def objective(trial):
        # ridge lambda on a log interval matching the searchable domain
        ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
        # beta on a log interval matching the searchable domain
        beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
        # noisy reward at this point
        return L.noisy_reward(ridge, beta, noise_seed)
    return objective


# Objective for GridSampler: same reward, plus an inert rep so the grid spends all 60 trials.
def make_grid_objective(noise_seed):
    # inner objective closes over the noise seed
    def objective(trial):
        # same two knobs as the continuous objective
        ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
        beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
        # rep is required by the grid search space but does not enter the reward
        trial.suggest_int("rep", 0, 4)
        # identical reward function to the continuous objective
        return L.noisy_reward(ridge, beta, noise_seed)
    return objective


# Construct a fresh sampler by name so each study starts clean.
def make_sampler(name):
    # random search, fixed seed
    if name == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=0)
    # Tree-structured Parzen Estimator with 10 random startup trials
    if name == "TPESampler":
        return optuna.samplers.TPESampler(seed=0, n_startup_trials=10)
    # Gaussian-process sampler (needs torch)
    if name == "GPSampler":
        return optuna.samplers.GPSampler(seed=0)
    # exhaustive grid over the 4x3 log cells x 5 reps = 60 points
    if name == "GridSampler":
        return optuna.samplers.GridSampler(
            {"ridge": GRID_RIDGE, "beta": GRID_BETA, "rep": GRID_REP}, seed=0)
    raise ValueError(name)


# Run one sampler on one noise seed and return (best_value, best_params, near_peak_count, n_trials).
def run_one(name, noise_seed):
    # silence per-trial INFO logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # maximize reward with the named sampler
    study = optuna.create_study(direction="maximize", sampler=make_sampler(name))
    # grid uses its own objective (with the inert rep); the others share one
    objective = make_grid_objective(noise_seed) if name == "GridSampler" else make_objective(noise_seed)
    study.optimize(objective, n_trials=N_TRIALS)
    # count how many completed trials sat within half a decade of the peak in both axes
    near = sum(L.near_peak(t.params["ridge"], t.params["beta"]) for t in study.trials)
    return study.best_value, study.best_params, near, len(study.trials)


# Run the full comparison and print one text table per noise seed plus a summary of near-peak counts.
def main():
    # announce version
    print("optuna", optuna.__version__)
    print(f"budget = {N_TRIALS} trials; near-peak = within 0.5 decade of (ridge=1e-6, beta=1e-2) in BOTH axes\n")
    samplers = ["RandomSampler", "TPESampler", "GPSampler", "GridSampler"]
    # collect near-peak counts per sampler across seeds for the final summary
    near_by_sampler = {s: [] for s in samplers}
    # one detailed table per noise seed
    for noise_seed in NOISE_SEEDS:
        print(f"=== noise_seed = {noise_seed} ===")
        print(f"{'sampler':<15}{'n_trials':>9}{'best_value':>12}{'best_ridge':>13}{'best_beta':>12}{'near_peak':>11}")
        for name in samplers:
            best_value, best_params, near, n = run_one(name, noise_seed)
            near_by_sampler[name].append(near)
            print(f"{name:<15}{n:>9d}{best_value:>12.3f}{best_params['ridge']:>13.2e}"
                  f"{best_params['beta']:>12.2e}{near:>11d}")
        print()
    # summary: near-peak count per sampler across the three seeds
    print("=== near-peak counts per sampler across seeds (the key comparison) ===")
    print(f"{'sampler':<15}{'seed 11':>9}{'seed 22':>9}{'seed 33':>9}{'range':>12}")
    for name in samplers:
        counts = near_by_sampler[name]
        print(f"{name:<15}{counts[0]:>9d}{counts[1]:>9d}{counts[2]:>9d}"
              f"{min(counts):>7d}-{max(counts):<4d}")


if __name__ == "__main__":
    main()
