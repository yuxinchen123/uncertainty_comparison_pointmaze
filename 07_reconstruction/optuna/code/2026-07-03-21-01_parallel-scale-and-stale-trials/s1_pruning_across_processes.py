"""Verify MedianPruner state flows across separate OS processes through a shared journal file.

Process A completes one good trial (intermediate values high at steps 1..10).
Process B then runs a bad trial, calling trial.should_prune() each step.
If pruner state crosses the process boundary via the journal file, B is pruned
using A's completed history. Genuine separate processes: this file re-invokes
itself with a role argument through subprocess, so nothing is shared in memory.
"""

import os
import subprocess
import sys

import optuna


# Build a JournalStorage on the shared folder with the file-open lock, as the sweep uses.
def make_storage(journal_path):
    # one journal backend guarded by the cross-process file-open lock
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    backend = optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


# Role A: complete one trial whose per-step intermediate values are all high.
def role_process_a(journal_path, study_name):
    # attach to the shared study (already created by the driver)
    storage = make_storage(journal_path)
    study = optuna.load_study(study_name=study_name, storage=storage)
    # ask one trial and report a high intermediate value at each of steps 1..10
    trial = study.ask()
    for step in range(1, 11):
        trial.report(1.0, step)
    # finish it as COMPLETE with a high final value
    study.tell(trial, 1.0)
    print(f"[A] completed trial number={trial.number} with per-step value 1.0 at steps 1..10")


# Role B: run a bad trial, checking should_prune() each step against A's history.
def role_process_b(journal_path, study_name):
    # attach to the same shared study from a different process.
    # The pruner is NOT stored in storage; each process must pass its own, or it
    # silently gets the default MedianPruner(n_startup_trials=5) and never prunes here.
    storage = make_storage(journal_path)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0, interval_steps=1),
    )
    # report how many completed trials this fresh process sees from the journal
    n_completed = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    print(f"[B] on load, completed trials visible from journal = {n_completed}; direction={study.direction.name}")
    # ask a trial and report a low value each step; check should_prune each step
    trial = study.ask()
    pruned_at = None
    for step in range(1, 11):
        trial.report(0.0, step)
        # should_prune reads completed-trial history from the shared storage
        if trial.should_prune():
            pruned_at = step
            break
    # record the outcome exactly as the pruner decided
    if pruned_at is not None:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        print(f"[B] trial number={trial.number} should_prune()==True at step={pruned_at}; told PRUNED")
    else:
        study.tell(trial, 0.0)
        print(f"[B] trial number={trial.number} never pruned; told COMPLETE")


# Driver: create the shared study, run A then B as subprocesses, print final states.
def driver():
    # place the journal on the real project NFS folder next to this script
    here = os.path.dirname(os.path.abspath(__file__))
    journal_path = os.path.join(here, "s1_journal.log")
    # start from a clean journal so the run is reproducible
    if os.path.exists(journal_path):
        os.remove(journal_path)
    study_name = "s1_pruning_demo"
    # create the study once, up front, with MedianPruner that starts after 1 completed trial
    storage = make_storage(journal_path)
    optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0, interval_steps=1),
    )
    # run process A to completion, then process B, each a fresh python process
    subprocess.run([sys.executable, os.path.abspath(__file__), "A", journal_path, study_name], check=True)
    subprocess.run([sys.executable, os.path.abspath(__file__), "B", journal_path, study_name], check=True)
    # reload from a third fresh view and print both trials' final states
    study = optuna.load_study(study_name=study_name, storage=make_storage(journal_path))
    print("\nFinal trial states read back from the shared journal:")
    for t in study.trials:
        print(f"  trial number={t.number} state={t.state.name} value={t.value} "
              f"intermediate_steps={sorted(t.intermediate_values)}")


# Dispatch on the role argument so subprocess children run the right function.
if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver()
    else:
        role = sys.argv[1]
        if role == "A":
            role_process_a(sys.argv[2], sys.argv[3])
        elif role == "B":
            role_process_b(sys.argv[2], sys.argv[3])
        else:
            raise ValueError(f"unknown role {role!r}")
