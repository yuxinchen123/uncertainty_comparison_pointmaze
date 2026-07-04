"""Verify study.enqueue_trial: enqueued param combinations are used by the first trials, in order."""
import optuna


def make_objective(seen):
    """Return an objective recording each trial's params; free search ranges so TPE could pick anything."""
    def objective(trial):
        # suggest a float and a categorical over broad ranges
        beta = trial.suggest_float("beta", 1e-4, 1.0, log=True)
        normalization = trial.suggest_categorical("normalization", ["none", "unit"])
        # record what this trial actually used
        seen.append({"beta": beta, "normalization": normalization})
        return -abs(beta - 1e-2)
    return objective


def main():
    """Enqueue two specific combinations on a TPE study; confirm the first two trials use them exactly."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("optuna version:", optuna.__version__)

    # seeded TPE study
    sampler = optuna.samplers.TPESampler(seed=2024)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # enqueue two exact parameter combinations before optimizing
    study.enqueue_trial({"beta": 1e-3, "normalization": "unit"})
    study.enqueue_trial({"beta": 5e-2, "normalization": "none"})

    # run 5 trials: first two should be the enqueued combos, the rest TPE-sampled
    seen = []
    study.optimize(make_objective(seen), n_trials=5)

    # print what each trial used
    for i, params in enumerate(seen):
        print(f"trial {i} params: {params}")

    # explicit checks on the first two
    print("\ntrial 0 matches first enqueue:",
          seen[0] == {"beta": 1e-3, "normalization": "unit"})
    print("trial 1 matches second enqueue:",
          seen[1] == {"beta": 5e-2, "normalization": "none"})


if __name__ == "__main__":
    main()
