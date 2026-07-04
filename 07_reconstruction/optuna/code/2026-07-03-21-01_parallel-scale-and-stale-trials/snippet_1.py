import optuna, tempfile, os
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Build a shared journal-file study (what every worker opens on the NFS folder).
path = os.path.join(tempfile.mkdtemp(), "journal.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

# One process completes a good trial: high intermediate value at every step.
study = optuna.create_study(study_name="demo", storage=storage, direction="maximize",
                            pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=0))
good = study.ask()
for step in range(10):
    good.report(1.0, step)
study.tell(good, 1.0)

# A second view WITHOUT passing a pruner: it silently gets the DEFAULT
# MedianPruner(n_startup_trials=5), so with one completed trial it never prunes.
view_default = optuna.load_study(study_name="demo", storage=storage)
bad = view_default.ask()
bad.report(0.0, 0)
print("no pruner passed  -> should_prune():", bad.should_prune())
view_default.tell(bad, state=TrialState.FAIL)

# A third view that passes the SAME pruner: now the low trial is pruned using the
# good trial's history, which reached this process through the shared journal.
view_pruner = optuna.load_study(study_name="demo", storage=storage,
                                pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=0))
bad2 = view_pruner.ask()
bad2.report(0.0, 0)
print("same pruner passed -> should_prune():", bad2.should_prune())
