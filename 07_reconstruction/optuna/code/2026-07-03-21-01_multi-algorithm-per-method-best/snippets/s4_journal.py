import os
import tempfile
import optuna
# current (v4.x) import paths — the journal classes live under optuna.storages.journal
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHOD_TRIALS = {"adam": 12, "adagrad": 8, "sgd1t": 20}   # different budgets to show independence


def objective(trial):
    # trivial method-agnostic objective; the point is the storage layout, not the landscape
    x = trial.suggest_float("x", -5.0, 5.0)
    return -(x - 1.0) ** 2


# one journal file (one file in the run folder) holds one study per method
path = os.path.join(tempfile.mkdtemp(), "runs_3_2_1.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

# create + optimize one study per method in the SAME storage
for method, n in METHOD_TRIALS.items():
    study = optuna.create_study(direction="maximize", study_name=f"3_2_1_{method}",
                                sampler=optuna.samplers.TPESampler(seed=0), storage=storage)
    study.optimize(objective, n_trials=n)

# list every study name recorded in the storage
print("study names:", sorted(optuna.get_all_study_names(storage)))

# reload each study by name from a FRESH storage handle; each holds exactly its own budget
reloaded = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
for method, expected in METHOD_TRIALS.items():
    study = optuna.load_study(study_name=f"3_2_1_{method}", storage=reloaded)
    print(f"  3_2_1_{method:<8} n_trials={len(study.trials):2d}  expected={expected:2d}  "
          f"isolated={len(study.trials) == expected}")
