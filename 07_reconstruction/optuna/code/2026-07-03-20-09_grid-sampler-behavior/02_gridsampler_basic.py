"""Verify GridSampler with a 36-combination search space mirroring the project sweep:
n_trials=None stop behavior, over-large n_trials, full coverage, and a second optimize call."""
import optuna

# search space mirroring the project sweep axes (2 x 3 x 2 x 3 = 36 combinations)
SEARCH_SPACE = {
    "normalization": ["none", "unit"],
    "ridge": [1e-6, 1e-4, 1e-2],
    "clip": ["inf", "5"],
    "beta": [1e-3, 1e-2, 1e-1],
}


def make_objective(seen):
    """Return an objective that records each visited param tuple into the given set."""
    def objective(trial):
        # suggest each grid axis with its matching suggest_* call
        normalization = trial.suggest_categorical("normalization", SEARCH_SPACE["normalization"])
        ridge = trial.suggest_categorical("ridge", SEARCH_SPACE["ridge"])
        clip = trial.suggest_categorical("clip", SEARCH_SPACE["clip"])
        beta = trial.suggest_categorical("beta", SEARCH_SPACE["beta"])
        # record the exact combination so we can check coverage afterwards
        seen.append((normalization, ridge, clip, beta))
        # trivial deterministic score
        return beta - ridge
    return objective


def main():
    """Run the four GridSampler checks (a)-(d) and print what actually happened."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("optuna version:", optuna.__version__)
    n_combos = 2 * 3 * 2 * 3
    print("expected combinations:", n_combos)

    # (a) n_trials=None: does the study stop by itself at exactly 36?
    seen_a = []
    sampler_a = optuna.samplers.GridSampler(SEARCH_SPACE, seed=42)
    study_a = optuna.create_study(direction="maximize", sampler=sampler_a)
    study_a.optimize(make_objective(seen_a), n_trials=None)
    print("\n(a) n_trials=None -> len(study.trials):", len(study_a.trials))
    print("(a) number of objective calls:", len(seen_a))

    # (c) every combination visited exactly once?
    unique = set(seen_a)
    print("(c) unique combinations visited:", len(unique))
    print("(c) all 36 visited exactly once:", len(unique) == n_combos and len(seen_a) == n_combos)

    # (b) n_trials larger than 36 (50): stop at 36, rerun, or raise?
    seen_b = []
    sampler_b = optuna.samplers.GridSampler(SEARCH_SPACE, seed=42)
    study_b = optuna.create_study(direction="maximize", sampler=sampler_b)
    study_b.optimize(make_objective(seen_b), n_trials=50)
    print("\n(b) n_trials=50 -> len(study.trials):", len(study_b.trials))
    print("(b) number of objective calls:", len(seen_b))
    from collections import Counter
    counts_b = Counter(seen_b)
    print("(b) distinct combinations:", len(counts_b))
    print("(b) max visits of any single combination:", max(counts_b.values()))

    # (d) a SECOND optimize call after the grid is exhausted
    seen_d = []
    sampler_d = optuna.samplers.GridSampler(SEARCH_SPACE, seed=42)
    study_d = optuna.create_study(direction="maximize", sampler=sampler_d)
    study_d.optimize(make_objective(seen_d), n_trials=None)
    print("\n(d) after first optimize -> len(study.trials):", len(study_d.trials))
    calls_after_first = len(seen_d)
    study_d.optimize(make_objective(seen_d), n_trials=None)
    print("(d) after second optimize -> len(study.trials):", len(study_d.trials))
    print("(d) new objective calls in second optimize:", len(seen_d) - calls_after_first)


if __name__ == "__main__":
    main()
