"""Verify GridSampler enumerates all 360 combinations when the random seed is one more grid axis."""
import optuna

# same four sweep axes plus a_seed in 0..9 -> 2 x 3 x 2 x 3 x 10 = 360 combinations
SEARCH_SPACE = {
    "normalization": ["none", "unit"],
    "ridge": [1e-6, 1e-4, 1e-2],
    "clip": ["inf", "5"],
    "beta": [1e-3, 1e-2, 1e-1],
    "a_seed": list(range(10)),
}


def make_objective(seen):
    """Return an instant objective recording each visited full param tuple."""
    def objective(trial):
        # suggest every axis, including the seed axis
        p = (
            trial.suggest_categorical("normalization", SEARCH_SPACE["normalization"]),
            trial.suggest_categorical("ridge", SEARCH_SPACE["ridge"]),
            trial.suggest_categorical("clip", SEARCH_SPACE["clip"]),
            trial.suggest_categorical("beta", SEARCH_SPACE["beta"]),
            trial.suggest_categorical("a_seed", SEARCH_SPACE["a_seed"]),
        )
        # record for coverage check; return a trivial value
        seen.append(p)
        return 0.0
    return objective


def main():
    """Run the 360-combination grid and confirm exact full coverage."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("optuna version:", optuna.__version__)
    n_combos = 2 * 3 * 2 * 3 * 10
    print("expected combinations:", n_combos)

    # run the whole grid with n_trials=None
    seen = []
    sampler = optuna.samplers.GridSampler(SEARCH_SPACE, seed=7)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(seen), n_trials=None)

    # report totals and coverage
    print("len(study.trials):", len(study.trials))
    print("objective calls:", len(seen))
    print("unique combinations visited:", len(set(seen)))
    print("all 360 visited exactly once:", len(set(seen)) == n_combos and len(seen) == n_combos)


if __name__ == "__main__":
    main()
