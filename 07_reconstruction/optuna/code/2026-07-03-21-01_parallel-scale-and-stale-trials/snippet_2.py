import optuna, tempfile, os, warnings
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.storages._heartbeat import BaseHeartbeat, is_heartbeat_enabled
from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)  # fail_stale_trials is experimental

# Journal-file study on the shared folder.
path = os.path.join(tempfile.mkdtemp(), "journal.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
study = optuna.create_study(study_name="demo", storage=storage)

# Journal storage does NOT implement the heartbeat mechanism (that is RDB-only).
print("is BaseHeartbeat:", isinstance(storage, BaseHeartbeat))
print("heartbeat enabled:", is_heartbeat_enabled(storage))

# A worker that dies mid-trial leaves the trial RUNNING. Simulate the leftover trial.
orphan = study.ask()
orphan.suggest_float("x", 0.0, 1.0)
print("orphan trial state:", study.trials[orphan.number].state.name)

# fail_stale_trials is a silent no-op on journal storage: nothing changes, no error.
optuna.storages.fail_stale_trials(study)
print("after fail_stale_trials:", study.trials[orphan.number].state.name)

# The cleanup a controller must do itself: tell the trial number FAIL.
# tell accepts an int trial number, so a process that never created the trial can fail it.
study.tell(orphan.number, state=TrialState.FAIL)
print("after tell(number, FAIL):", study.trials[orphan.number].state.name)
