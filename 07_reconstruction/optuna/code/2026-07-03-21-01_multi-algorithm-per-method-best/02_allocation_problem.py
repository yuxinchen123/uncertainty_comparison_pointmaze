"""Central evidence: one shared study vs three per-method studies at equal total budget.

A shared TPE study with a categorical `method` parameter spends most of its post-startup
budget on the single best method (sgd1t), so the two weaker methods (adam, adagrad) get few
trials and a worse per-method best. Splitting the same total budget into one study per method
gives each method a full search. We run both at 3 seeds and print the counts and bests.
"""

import numpy as np
import optuna

from toy import reward, reward_mean, TRUE_PEAK

optuna.logging.set_verbosity(optuna.logging.WARNING)

METHODS = ["adam", "adagrad", "sgd1t"]


def trial_true_mean(method, params):
    """Return the noise-free mean reward for a trial's suggested params (search-quality metric)."""
    # sgd1t trials carry eta0/t0; adam/adagrad trials do not
    if method == "sgd1t":
        return reward_mean(method, params["readout"], params["log10_beta"],
                           eta0=params["eta0"], t0=params["t0"])
    return reward_mean(method, params["readout"], params["log10_beta"])


def shared_objective(base_seed):
    """Build the conditional objective for the SHARED study (method is a suggested categorical)."""
    def _obj(trial):
        # method is chosen by the sampler; readout and beta exist for all methods
        method = trial.suggest_categorical("method", METHODS)
        readout = trial.suggest_categorical("readout", ["mse", "l2"])
        log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
        # eta0/t0 are suggested only inside the sgd1t branch
        if method == "sgd1t":
            eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
            t0 = trial.suggest_categorical("t0", [1e3, 1e4])
            return reward(method, readout, log10_beta, eta0=eta0, t0=t0, base_seed=base_seed)
        return reward(method, readout, log10_beta, base_seed=base_seed)
    return _obj


def per_method_objective(method, base_seed):
    """Build the objective for ONE per-method study (method is fixed, not suggested)."""
    def _obj(trial):
        # method is fixed for this study; only that method's own axes are searched
        readout = trial.suggest_categorical("readout", ["mse", "l2"])
        log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
        if method == "sgd1t":
            eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
            t0 = trial.suggest_categorical("t0", [1e3, 1e4])
            return reward(method, readout, log10_beta, eta0=eta0, t0=t0, base_seed=base_seed)
        return reward(method, readout, log10_beta, base_seed=base_seed)
    return _obj


def best_per_method_shared(study):
    """Return {method: (n_trials, best_noisy, best_true_mean)} from a shared study's trials."""
    # group completed trials by their suggested method
    out = {}
    for m in METHODS:
        trs = [t for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE and t.params.get("method") == m]
        # best_noisy: the max objective value Optuna sees (upward-biased by noise + sample count)
        best_noisy = max((t.value for t in trs), default=float("nan"))
        # best_true_mean: the best noise-free region the search actually visited (search quality)
        best_true = max((trial_true_mean(m, t.params) for t in trs), default=float("nan"))
        out[m] = (len(trs), best_noisy, best_true)
    return out


def best_per_method_split(study, method):
    """Return (n_trials, best_noisy, best_true_mean) for a single-method study."""
    trs = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_noisy = max((t.value for t in trs), default=float("nan"))
    best_true = max((trial_true_mean(method, t.params) for t in trs), default=float("nan"))
    return (len(trs), best_noisy, best_true)


def main():
    """Run shared vs split at 3 seeds and print a comparison table of counts and bests."""
    seeds = [0, 1, 2]
    total_budget = 90
    per_method_budget = total_budget // len(METHODS)  # 30 each -> same 90-trial total

    # accumulate rows: (seed, method, shared_n, shared_best, split_n, split_best, true_peak)
    rows = []
    for seed in seeds:
        # --- shared study: one TPE study, 90 trials, method is a suggested parameter ---
        shared = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=seed))
        shared.optimize(shared_objective(seed), n_trials=total_budget)
        shared_stats = best_per_method_shared(shared)

        # --- split: three studies, 30 trials each, each searches one method fully ---
        split_stats = {}
        for m in METHODS:
            st = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=seed))
            st.optimize(per_method_objective(m, seed), n_trials=per_method_budget)
            split_stats[m] = best_per_method_split(st, m)

        # record one row per (seed, method): counts, noisy best, and noise-free best mean
        for m in METHODS:
            sn, sb_noisy, sb_true = shared_stats[m]
            pn, pb_noisy, pb_true = split_stats[m]
            rows.append((seed, m, sn, sb_noisy, sb_true, pn, pb_noisy, pb_true, TRUE_PEAK[m]))

    # print the per-seed comparison table
    print(f"Total budget per approach: {total_budget} trials "
          f"(split = {per_method_budget} trials x {len(METHODS)} methods).")
    print("TPE default n_startup_trials = 10 (random) before the model takes over.")
    print("best_true = best NOISE-FREE mean reward the search reached (search quality, no max-of-noise bias).")
    print("best_noisy = max noisy objective value = study.best_value (biased upward by noise and by #trials).\n")
    header = f"{'seed':>4} {'method':>8} {'shared_n':>9} {'shared_true':>12} {'shared_noisy':>13} " \
             f"{'split_n':>8} {'split_true':>11} {'split_noisy':>12} {'true_peak':>10}"
    print(header)
    print("-" * len(header))
    for seed, m, sn, sbn, sbt, pn, pbn, pbt, tp in rows:
        print(f"{seed:>4} {m:>8} {sn:>9} {sbt:>12.2f} {sbn:>13.2f} "
              f"{pn:>8} {pbt:>11.2f} {pbn:>12.2f} {tp:>10.1f}")

    # print the shared-study allocation share averaged over seeds (the concentration effect)
    print("\nShared-study trial allocation per method (share of 90 trials), averaged over seeds:")
    for m in METHODS:
        counts = [r[2] for r in rows if r[1] == m]
        print(f"  {m:>8}: mean {np.mean(counts):5.1f} trials  "
              f"(per-seed counts: {counts})")

    # summarize search quality: how close did each approach get to the true peak (noise-free)?
    print("\nSearch quality: gap of best_true to the true peak (0 = reached the peak), averaged over seeds:")
    print(f"{'method':>8} {'shared_gap':>11} {'split_gap':>10}")
    for m in METHODS:
        sgaps = [r[8] - r[4] for r in rows if r[1] == m]   # true_peak - shared_true
        pgaps = [r[8] - r[7] for r in rows if r[1] == m]   # true_peak - split_true
        print(f"{m:>8} {np.mean(sgaps):>11.2f} {np.mean(pgaps):>10.2f}")


if __name__ == "__main__":
    main()
