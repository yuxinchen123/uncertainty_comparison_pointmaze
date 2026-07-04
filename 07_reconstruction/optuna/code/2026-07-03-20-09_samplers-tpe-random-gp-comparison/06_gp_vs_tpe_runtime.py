"""GPSampler runs in this env (needs torch) and is slower per trial than TPE at 60 trials.

We time a full 60-trial study for GPSampler(seed=0) and for TPESampler(seed=0), on the same
objective, and print total wall time and per-trial wall time for each. torch is imported to
confirm the dependency GPSampler needs is present.
"""

import time

import optuna
import torch

import landscape as L


# Objective: suggest ridge and beta on a log scale and return the noisy reward.
def objective(trial):
    # ridge lambda on a log interval
    ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
    # beta on a log interval
    beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
    # fixed noise seed
    return L.noisy_reward(ridge, beta, noise_seed=11)


# Time a full n_trials study with the given sampler and return (total_seconds, best_value).
def time_study(sampler, n_trials):
    # quiet logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # fresh study
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # wall-clock the optimize loop
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.perf_counter() - t0
    return elapsed, study.best_value


# Time GP vs TPE at an equal budget and print the comparison.
def main():
    # confirm torch (GPSampler's dependency) is importable and report its version
    print("optuna", optuna.__version__, "| torch", torch.__version__)
    n_trials = 60
    # time TPE first
    tpe_secs, tpe_best = time_study(optuna.samplers.TPESampler(seed=0, n_startup_trials=10), n_trials)
    # time GP (experimental) on the identical objective and budget
    gp_secs, gp_best = time_study(optuna.samplers.GPSampler(seed=0), n_trials)
    # print a small table of total and per-trial wall time
    print(f"\nbudget = {n_trials} trials, same objective\n")
    print(f"{'sampler':<14}{'total_s':>10}{'per_trial_s':>14}{'best_value':>12}")
    print(f"{'TPESampler':<14}{tpe_secs:>10.3f}{tpe_secs / n_trials:>14.4f}{tpe_best:>12.3f}")
    print(f"{'GPSampler':<14}{gp_secs:>10.3f}{gp_secs / n_trials:>14.4f}{gp_best:>12.3f}")
    print(f"\nGP / TPE total-time ratio = {gp_secs / tpe_secs:.1f}x")


if __name__ == "__main__":
    main()
