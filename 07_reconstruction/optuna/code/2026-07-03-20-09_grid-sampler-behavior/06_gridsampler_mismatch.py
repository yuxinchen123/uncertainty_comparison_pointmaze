"""Verify GridSampler edge cases:
6a - objective suggests a parameter NOT in the search_space dict;
6b - suggest_float low/high vs the grid value (grid value outside the suggested [low, high])."""
import optuna


def test_extra_param():
    """6a: objective suggests a param absent from the grid search_space; observe the exact error."""
    print("=== 6a: objective suggests a param not in search_space ===")
    # grid only defines "beta"; objective will also suggest "ridge"
    search_space = {"beta": [1e-3, 1e-2, 1e-1]}
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # objective asks for an undefined "ridge" param
    def objective(trial):
        beta = trial.suggest_categorical("beta", [1e-3, 1e-2, 1e-1])
        ridge = trial.suggest_categorical("ridge", [1e-6, 1e-4])
        return beta + ridge

    # catch exactly the exception type Optuna raises here (point of the test)
    try:
        study.optimize(objective, n_trials=None)
        print("6a: no exception raised; len(trials):", len(study.trials))
    except ValueError as e:
        print("6a: raised ValueError:", e)


def test_missing_param():
    """6a-variant: grid defines a param the objective never suggests; observe behavior."""
    print("\n=== 6a-variant: search_space has an extra key the objective ignores ===")
    # grid defines beta AND ridge, objective only suggests beta
    search_space = {"beta": [1e-3, 1e-2], "ridge": [1e-6, 1e-4]}
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        beta = trial.suggest_categorical("beta", [1e-3, 1e-2])
        return beta

    # run and report how many trials happened and their recorded params
    study.optimize(objective, n_trials=None)
    print("6a-variant: len(trials):", len(study.trials))
    print("6a-variant: distinct trial.params:", {tuple(sorted(t.params.items())) for t in study.trials})


def test_float_range_inside():
    """6b-inside: suggest_float range wide enough to contain every grid value."""
    print("\n=== 6b-inside: grid values all within suggest_float [low, high] ===")
    # grid beta values 1e-3..1e-1, suggest range 1e-4..1.0 contains them all
    search_space = {"beta": [1e-3, 1e-2, 1e-1]}
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        beta = trial.suggest_float("beta", 1e-4, 1.0, log=True)
        return beta

    study.optimize(objective, n_trials=None)
    used = sorted(t.params["beta"] for t in study.trials)
    print("6b-inside: len(trials):", len(study.trials))
    print("6b-inside: beta values used:", used)


def test_float_range_outside():
    """6b-outside: a grid value lies OUTSIDE the suggest_float [low, high]; report the exact result."""
    print("\n=== 6b-outside: a grid value below suggest_float low ===")
    # grid includes 1e-3, but suggest_float low is 1e-2 -> 1e-3 is below the stated low
    search_space = {"beta": [1e-3, 1e-2, 1e-1]}
    sampler = optuna.samplers.GridSampler(search_space, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        # stated low=1e-2 excludes the grid value 1e-3
        beta = trial.suggest_float("beta", 1e-2, 1.0, log=True)
        return beta

    # observe whether Optuna raises, warns, clips, or accepts the out-of-range grid value
    study.optimize(objective, n_trials=None)
    used = sorted(t.params["beta"] for t in study.trials)
    print("6b-outside: len(trials):", len(study.trials))
    print("6b-outside: beta values used:", used)
    print("6b-outside: 1e-3 (below stated low) present in used values:", 1e-3 in used)


def main():
    """Run all four GridSampler mismatch checks."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("optuna version:", optuna.__version__, "\n")
    test_extra_param()
    test_missing_param()
    test_float_range_inside()
    test_float_range_outside()


if __name__ == "__main__":
    main()
