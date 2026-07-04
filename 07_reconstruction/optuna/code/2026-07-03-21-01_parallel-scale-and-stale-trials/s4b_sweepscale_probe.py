"""Isolate journal size and cold-load time at sweep scale (~9600 trials).

Uses RandomSampler so ask() is constant-time (the default TPESampler refits over all
completed trials on every ask, which dominates and confounds a pure storage measurement).
No intermediate reports: one value per trial, matching the RND objective (final reward).
"""

import os
import time

import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)


# Build the shared journal storage on the real folder.
def open_storage(path):
    # a fresh storage object replays the journal from the first record
    lock = optuna.storages.journal.JournalFileOpenLock(path)
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path, lock_obj=lock))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "s4d_journal.log")
    if os.path.exists(path):
        os.remove(path)
    n = 9600
    # fill with a constant-time sampler and a one-value objective
    study = optuna.create_study(
        study_name="s4d", storage=open_storage(path), direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    t0 = time.perf_counter()
    study.optimize(lambda t: t.suggest_float("x", -10.0, 10.0) ** 2, n_trials=n)
    fill = time.perf_counter() - t0
    size_mb = os.path.getsize(path) / 1e6
    print(f"fill {n} trials (RandomSampler, one value each): {fill:.1f} s, journal {size_mb:.2f} MB")

    # time a genuinely cold open + trial-list read from a fresh storage object
    t1 = time.perf_counter()
    reopened = optuna.load_study(study_name="s4d", storage=open_storage(path))
    loaded = len(reopened.trials)
    load = time.perf_counter() - t1
    complete = sum(1 for tr in reopened.trials if tr.state == optuna.trial.TrialState.COMPLETE)
    print(f"cold open + len(study.trials)={loaded} (complete={complete}): {load:.3f} s")


if __name__ == "__main__":
    main()
