"""TPESampler determinism: same seed + same objective -> identical suggested params across studies.

We build two fresh studies, each with TPESampler(seed=0), run the identical objective, and check
that the first 10 suggested (ridge, beta) pairs match exactly between the two studies.
"""

import optuna

import landscape as L


# Objective: suggest ridge and beta on a log scale and return the noisy reward.
def objective(trial):
    # ridge lambda on a log interval
    ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
    # beta on a log interval
    beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
    # fixed noise seed so both studies see the identical function
    return L.noisy_reward(ridge, beta, noise_seed=99)


# Build one fresh TPE study with the given seed and return its first n suggested (ridge, beta) pairs.
def run_study(seed, n_trials):
    # quiet logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # fresh study, TPE with the given seed
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials)
    # return the suggested-parameter sequence
    return [(t.params["ridge"], t.params["beta"]) for t in study.trials]


# Compare two fresh seed-0 studies on their first 10 trials.
def main():
    # announce version
    print("optuna", optuna.__version__)
    # two independent studies, identical seed and objective
    seq_a = run_study(seed=0, n_trials=10)
    seq_b = run_study(seed=0, n_trials=10)
    # a third study with a different seed, to show the sequence does change with the seed
    seq_c = run_study(seed=1, n_trials=10)
    # print the first few pairs of each
    print("\nfirst 5 (ridge, beta) suggestions:")
    print(f"{'trial':<6}{'study A seed0':>28}{'study B seed0':>28}")
    for i in range(5):
        print(f"{i:<6}{str((f'{seq_a[i][0]:.3e}', f'{seq_a[i][1]:.3e}')):>28}"
              f"{str((f'{seq_b[i][0]:.3e}', f'{seq_b[i][1]:.3e}')):>28}")
    # exact-match checks over all 10 trials
    print(f"\nseed 0 vs seed 0, first 10 trials identical: {seq_a == seq_b}")
    print(f"seed 0 vs seed 1, first 10 trials identical: {seq_a == seq_c}")


if __name__ == "__main__":
    main()
