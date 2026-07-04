"""Three per-method studies in ONE JournalStorage file, then reload each by name.

This is the recommended layout for train run 3.2.1: one journal file in the run folder,
one study per optimizer method. We create adam / adagrad / sgd1t studies in the same
storage, optimize each briefly, list all study names from the storage, reload each by
name, and confirm the trial counts are independent (no cross-talk between studies).
"""

import os

import optuna
# current (v4.x) import paths: the journal classes live under optuna.storages.journal
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

from toy import reward

optuna.logging.set_verbosity(optuna.logging.WARNING)

# per-method trial budgets, deliberately different so we can see each study is independent
METHOD_TRIALS = {"adam": 12, "adagrad": 8, "sgd1t": 20}

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "journal_demo", "runs_3_2_1.log")


def per_method_objective(method):
    """Build a fixed-method objective (method is not suggested; only its own axes are searched)."""
    def _obj(trial):
        # method is fixed for this study; readout and beta apply to every method
        readout = trial.suggest_categorical("readout", ["mse", "l2"])
        log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
        # only sgd1t carries eta0/t0
        if method == "sgd1t":
            eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
            t0 = trial.suggest_categorical("t0", [1e3, 1e4])
            return reward(method, readout, log10_beta, eta0=eta0, t0=t0, base_seed=0)
        return reward(method, readout, log10_beta, base_seed=0)
    return _obj


def open_storage(path):
    """Open a JournalStorage backed by one file, using the open-file lock (current import path)."""
    # JournalFileOpenLock is the NFSv3+ open()-with-O_EXCL lock; pass the same path it protects
    lock = JournalFileOpenLock(path)
    return JournalStorage(JournalFileBackend(path, lock_obj=lock))


def main():
    """Create three named studies in one journal file, optimize, then reload each by name."""
    # start from a clean file so reruns are reproducible (delete the log and any lock leftovers)
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    for p in (JOURNAL_PATH, JOURNAL_PATH + ".lock"):
        if os.path.exists(p):
            os.remove(p)

    # one storage object over the single journal file
    storage = open_storage(JOURNAL_PATH)

    # create + optimize one study per method, all in the same storage
    for method, n in METHOD_TRIALS.items():
        study = optuna.create_study(direction="maximize",
                                    study_name=f"3_2_1_{method}",
                                    sampler=optuna.samplers.TPESampler(seed=0),
                                    storage=storage)
        study.optimize(per_method_objective(method), n_trials=n)

    print(f"journal file: {JOURNAL_PATH}")
    print(f"file size on disk: {os.path.getsize(JOURNAL_PATH)} bytes\n")

    # list every study name recorded in the storage
    names = optuna.get_all_study_names(storage)
    print(f"optuna.get_all_study_names(storage) -> {sorted(names)}\n")

    # reload each study by name from a FRESH storage handle and check its trial count
    reloaded_storage = open_storage(JOURNAL_PATH)
    print("Reloaded each study by name (fresh storage handle):")
    print(f"{'study_name':>16} {'n_trials':>9} {'expected':>9} {'best_value':>11}")
    for method, expected in METHOD_TRIALS.items():
        study = optuna.load_study(study_name=f"3_2_1_{method}", storage=reloaded_storage)
        n = len(study.trials)
        # each study must hold exactly its own budget, proving no cross-talk between studies
        print(f"{('3_2_1_' + method):>16} {n:>9} {expected:>9} {study.best_value:>11.3f}")

    # explicit independence check: totals add up and every study is isolated
    total = sum(len(optuna.load_study(study_name=f"3_2_1_{m}", storage=reloaded_storage).trials)
                for m in METHOD_TRIALS)
    print(f"\nsum of per-study trial counts: {total} (== total submitted {sum(METHOD_TRIALS.values())}: "
          f"{total == sum(METHOD_TRIALS.values())})")
    all_isolated = all(
        len(optuna.load_study(study_name=f"3_2_1_{m}", storage=reloaded_storage).trials) == METHOD_TRIALS[m]
        for m in METHOD_TRIALS)
    print(f"every study holds exactly its own budget (no cross-talk): {all_isolated}")


if __name__ == "__main__":
    main()
