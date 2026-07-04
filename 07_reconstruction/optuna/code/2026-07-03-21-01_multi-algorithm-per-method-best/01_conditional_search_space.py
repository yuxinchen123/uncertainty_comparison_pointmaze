"""Verify a conditional search space in ONE Optuna study under the default TPE sampler.

The objective picks method as a categorical, and suggests eta0/t0 ONLY inside the sgd1t
branch. We confirm the study runs, then show that trials_dataframe has params_eta0/params_t0
missing (NaN) for the adam/adagrad rows, and print how many trials each parameter appears in.
"""

import numpy as np
import optuna

from toy import reward

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    """One trial: choose method, then suggest branch-specific params, return noisy reward."""
    # method is the top-level categorical; readout and beta exist for every method
    method = trial.suggest_categorical("method", ["adam", "adagrad", "sgd1t"])
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    # eta0 and t0 are suggested ONLY in the sgd1t branch, so they never exist for adam/adagrad
    if method == "sgd1t":
        eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
        t0 = trial.suggest_categorical("t0", [1e3, 1e4])
        return reward(method, readout, log10_beta, eta0=eta0, t0=t0, base_seed=0)
    return reward(method, readout, log10_beta, base_seed=0)


def main():
    """Run one conditional study, then inspect which trials carry the branch-only params."""
    # default TPE (multivariate=False -> independent per-parameter sampling), fixed seed
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=60)

    # dump the trials as a dataframe and show the columns Optuna created
    df = study.trials_dataframe()
    print("trials_dataframe columns:")
    print(list(df.columns))
    print()

    # show a few adam/adagrad rows: their params_eta0 / params_t0 cells must be missing (NaN)
    cols = ["number", "value", "params_method", "params_readout", "params_log10_beta",
            "params_eta0", "params_t0"]
    print("first 8 rows (note eta0/t0 are NaN unless method == sgd1t):")
    with np.printoptions(precision=3):
        print(df[cols].head(8).to_string(index=False))
    print()

    # count, per parameter, how many trials actually contain it (this is what TPE fits on)
    n_total = len(study.trials)
    n_sgd = sum(1 for t in study.trials if t.params.get("method") == "sgd1t")
    n_eta0 = sum(1 for t in study.trials if "eta0" in t.params)
    n_t0 = sum(1 for t in study.trials if "t0" in t.params)
    n_beta = sum(1 for t in study.trials if "log10_beta" in t.params)
    print(f"total trials: {n_total}")
    print(f"trials with method == sgd1t : {n_sgd}")
    print(f"trials containing eta0      : {n_eta0}   (present only in sgd1t trials: {n_eta0 == n_sgd})")
    print(f"trials containing t0        : {n_t0}   (present only in sgd1t trials: {n_t0 == n_sgd})")
    print(f"trials containing log10_beta: {n_beta}   (present in every trial: {n_beta == n_total})")
    print()

    # confirm the NaN pattern precisely from the dataframe
    non_sgd = df[df["params_method"] != "sgd1t"]
    print(f"non-sgd1t rows: {len(non_sgd)}; "
          f"all have params_eta0 NaN: {non_sgd['params_eta0'].isna().all()}; "
          f"all have params_t0 NaN: {non_sgd['params_t0'].isna().all()}")


if __name__ == "__main__":
    main()
