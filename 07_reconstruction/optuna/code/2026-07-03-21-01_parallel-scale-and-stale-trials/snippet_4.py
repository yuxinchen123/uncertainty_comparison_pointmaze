import optuna, tempfile, os, time
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One journal-file study; fill it with many trivial trials.
path = os.path.join(tempfile.mkdtemp(), "journal.log")
def open_storage():
    # a fresh storage object replays the whole journal from the first record on open
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

study = optuna.create_study(study_name="demo", storage=open_storage(), direction="minimize")
study.optimize(lambda t: t.suggest_float("x", -10, 10) ** 2, n_trials=1000)
print(f"journal size at 1000 trials: {os.path.getsize(path)/1e6:.3f} MB")

# Time a cold open + trial-list read (what every worker start / controller poll pays).
t0 = time.perf_counter()
reopened = optuna.load_study(study_name="demo", storage=open_storage())
n = len(reopened.trials)
print(f"cold open + len(study.trials)={n}: {time.perf_counter() - t0:.3f} s")
