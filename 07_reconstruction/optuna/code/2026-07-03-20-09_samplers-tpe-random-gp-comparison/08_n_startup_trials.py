"""n_startup_trials: TPE's first n_startup_trials are random; after that it concentrates.

Documented behavior: for the first n_startup_trials, TPE draws parameters at random (like
RandomSampler); only after that does it fit the Parzen models and steer toward good regions.
We verify this two ways on the toy landscape:
  1. Direct: with n_startup_trials=60 of 60, TPE's suggested params should equal RandomSampler's
     exactly (all trials are the random startup phase).
  2. Indirect: near-peak allocation for TPE with n_startup_trials=59 (almost all random) should
     be close to RandomSampler and far below n_startup_trials=10 (mostly steered).
"""

import optuna

import landscape as L

# Budget and noise seed for this test.
N_TRIALS = 60
NOISE_SEED = 11


# Objective: suggest ridge and beta on a log scale and return the noisy reward.
def objective(trial):
    # ridge lambda on a log interval
    ridge = trial.suggest_float("ridge", 1e-6, 1e-2, log=True)
    # beta on a log interval
    beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
    # fixed noise seed
    return L.noisy_reward(ridge, beta, NOISE_SEED)


# Run one sampler, return its suggested (ridge, beta) sequence and near-peak count.
def run(sampler):
    # quiet logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # fresh study
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    # collect the parameter sequence and the near-peak allocation
    seq = [(t.params["ridge"], t.params["beta"]) for t in study.trials]
    near = sum(L.near_peak(r, b) for r, b in seq)
    return seq, near


# Verify the startup phase is random and that shrinking it lets TPE concentrate.
def main():
    # announce version
    print("optuna", optuna.__version__)
    # random baseline
    rand_seq, rand_near = run(optuna.samplers.RandomSampler(seed=0))
    # TPE with ALL trials as random startup -- should match Random exactly
    tpe_allrand_seq, tpe_allrand_near = run(
        optuna.samplers.TPESampler(seed=0, n_startup_trials=N_TRIALS))
    # TPE with almost all random startup (59 of 60)
    _, tpe_59_near = run(optuna.samplers.TPESampler(seed=0, n_startup_trials=59))
    # TPE with the usual small startup (10 of 60) -- should concentrate
    _, tpe_10_near = run(optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
    # direct check: full-startup TPE reproduces RandomSampler's exact suggestions
    print(f"\nTPE(n_startup=60) params identical to RandomSampler: {rand_seq == tpe_allrand_seq}")
    # indirect check: near-peak allocation rises as the startup phase shrinks
    print(f"\nnear-peak allocation (out of {N_TRIALS}) vs startup size:")
    print(f"  RandomSampler                 : {rand_near}")
    print(f"  TPE n_startup_trials=60 (all random) : {tpe_allrand_near}")
    print(f"  TPE n_startup_trials=59 (almost random): {tpe_59_near}")
    print(f"  TPE n_startup_trials=10 (steered)    : {tpe_10_near}")


if __name__ == "__main__":
    main()
