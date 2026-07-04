import optuna, tempfile, os
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One journal-file study on the shared folder.
path = os.path.join(tempfile.mkdtemp(), "journal.log")
def open_storage():
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

# Controller pattern: the controller keeps K trials asked-but-untold ("in flight").
K = 8
controller = optuna.create_study(study_name="demo", storage=open_storage())
in_flight = [controller.ask() for _ in range(K)]           # ask() runs the sampler here
for t in in_flight:
    t.suggest_float("x", -5, 5)                             # controller fixes the params

# Any other process (here: a fresh open of the same journal) sees those K trials RUNNING.
observer = optuna.load_study(study_name="demo", storage=open_storage())
running = [t for t in observer.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
print(f"asked-but-untold trials visible as RUNNING from another view: {len(running)}")

# When a worker returns a result, the controller tells that trial number.
controller.tell(in_flight[0].number, 0.123)
observer2 = optuna.load_study(study_name="demo", storage=open_storage())
n_running = sum(t.state == TrialState.RUNNING for t in observer2.get_trials(deepcopy=False))
n_done = sum(t.state == TrialState.COMPLETE for t in observer2.get_trials(deepcopy=False))
print(f"after telling one: RUNNING={n_running} COMPLETE={n_done}")
