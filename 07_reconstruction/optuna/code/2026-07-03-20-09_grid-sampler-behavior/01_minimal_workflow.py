"""Verify the minimal Optuna workflow: define objective, create study, optimize, read results."""
import optuna


def objective(trial):
    """Toy objective on two suggested params; returns a value Optuna will maximize."""
    # suggest a float in a range and a categorical value
    beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
    normalization = trial.suggest_categorical("normalization", ["none", "unit"])
    # deterministic scalar: peak near beta=1e-2 and normalization="unit"
    score = -abs(beta - 1e-2) * 100.0 + (5.0 if normalization == "unit" else 0.0)
    return score


def main():
    """Run a short study with a fixed-seed TPE sampler and print every result field."""
    # quiet the per-trial INFO logs so the printed output is only what we assert
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # create a maximize study with a seeded sampler for reproducibility
    sampler = optuna.samplers.TPESampler(seed=12345)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # run 15 trials
    study.optimize(objective, n_trials=15)

    # read and print the headline results
    print("optuna version:", optuna.__version__)
    print("len(study.trials):", len(study.trials))
    print("best_trial.number:", study.best_trial.number)
    print("best_trial.value:", study.best_trial.value)
    print("best_trial.params:", study.best_trial.params)
    print("best_params:", study.best_params)
    print("best_value:", study.best_value)

    # print the trials_dataframe column names
    df = study.trials_dataframe()
    print("trials_dataframe shape:", df.shape)
    print("trials_dataframe columns:", list(df.columns))


if __name__ == "__main__":
    main()
