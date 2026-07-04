"""Show the four value-suggestion calls Optuna offers inside objective(trial).

Prints, over a few trials, the concrete values returned by
suggest_categorical, suggest_float (linear), suggest_float(log=True), and suggest_int.
Optuna version is printed so the tutorial can cite it.
"""

import optuna


# One trial's objective: call each suggest_* type once and record what came back.
def objective(trial):
    # categorical: pick from a finite set of labels
    norm = trial.suggest_categorical("normalization", ["none", "unit"])
    # float on a linear interval
    beta_linear = trial.suggest_float("beta_linear", 0.001, 0.1)
    # float on a log interval (values spread evenly across decades)
    ridge_log = trial.suggest_float("ridge_log", 1e-6, 1e-2, log=True)
    # integer on an interval
    n_layers = trial.suggest_int("n_layers", 1, 4)
    # print one row per trial so the reader sees real returned values
    print(f"trial {trial.number}: normalization={norm!r:8s} "
          f"beta_linear={beta_linear:.5f} ridge_log={ridge_log:.3e} n_layers={n_layers}")
    # objective value is irrelevant here; return a constant
    return 0.0


# Run a handful of trials with a fixed-seed random sampler so the printout is reproducible.
def main():
    # announce version for the tutorial
    print("optuna", optuna.__version__)
    # quiet the per-trial INFO logging so only our prints show
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # explicit seed on the sampler for reproducibility
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    # 6 trials is enough to see the value ranges
    study.optimize(objective, n_trials=6)


if __name__ == "__main__":
    main()
