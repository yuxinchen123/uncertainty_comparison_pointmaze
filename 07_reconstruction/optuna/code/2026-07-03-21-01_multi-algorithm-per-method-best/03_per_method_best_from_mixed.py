"""Extract the per-method best configuration from a single mixed (shared) study.

If you already have one study whose trials mix all three methods, you can still recover the
best configuration PER method: filter COMPLETE trials by params["method"], then take the
argmax by value within each method group. This prints the three winners with their params.
"""

import optuna

from toy import reward

optuna.logging.set_verbosity(optuna.logging.WARNING)

METHODS = ["adam", "adagrad", "sgd1t"]


def objective(trial):
    """One trial of the mixed study: suggest method, then branch-specific params, return reward."""
    # method is a suggested categorical; readout and beta apply to all methods
    method = trial.suggest_categorical("method", METHODS)
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    # eta0/t0 only in the sgd1t branch
    if method == "sgd1t":
        eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
        t0 = trial.suggest_categorical("t0", [1e3, 1e4])
        return reward(method, readout, log10_beta, eta0=eta0, t0=t0, base_seed=0)
    return reward(method, readout, log10_beta, base_seed=0)


def per_method_best(study):
    """Return {method: best FrozenTrial} by filtering COMPLETE trials on params['method']."""
    # start with no winner for any method
    winners = {}
    # scan only COMPLETE trials so failed/running trials never win
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        m = t.params.get("method")
        if m is None:
            continue
        # keep the highest-value trial seen so far for this method (direction is maximize)
        if m not in winners or t.value > winners[m].value:
            winners[m] = t
    return winners


def main():
    """Build one mixed study, then print the best trial per method with its full params."""
    # one shared TPE study over all three methods
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=90)

    # recover the per-method winners from the single mixed study
    winners = per_method_best(study)

    print("Per-method best configuration recovered from ONE mixed study (90 trials):\n")
    for m in METHODS:
        t = winners[m]
        # count how many of the 90 trials this method received (fewer -> less-searched winner)
        n_m = sum(1 for x in study.trials if x.params.get("method") == m)
        print(f"method = {m}")
        print(f"  trials for this method : {n_m}")
        print(f"  best trial number      : {t.number}")
        print(f"  best value (noisy)     : {t.value:.3f}")
        print(f"  params                 : {t.params}")
        print()


if __name__ == "__main__":
    main()
