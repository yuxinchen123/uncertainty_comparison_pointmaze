"""Verify stale-trial handling with journal storage: no heartbeat, and manual FAIL recovery.

(a) Journal storage does not support the heartbeat mechanism (RDB only). Show that
    optuna.storages.fail_stale_trials on a journal-backed study is a silent no-op.
(b) A subprocess asks a trial (now RUNNING in the shared journal), then is SIGKILLed
    mid-trial. Confirm the trial is still RUNNING afterward from a different process.
(c) Recovery: from the driver process, load_study then study.tell(number, FAIL) to
    clean up the orphaned RUNNING trial that a dead worker left behind.
"""

import os
import signal
import subprocess
import sys
import time

import optuna
from optuna.storages._heartbeat import BaseHeartbeat, is_heartbeat_enabled


# Build the shared journal storage exactly as the sweep does.
def make_storage(journal_path):
    # journal backend guarded by the cross-process file-open lock
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    backend = optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


# Worker role: ask a trial (marks it RUNNING in the journal), announce its number, then sleep.
def role_dead_worker(journal_path, study_name, number_file):
    # attach to the shared study and start one trial
    study = optuna.load_study(study_name=study_name, storage=make_storage(journal_path))
    trial = study.ask()
    trial.suggest_float("x", 0.0, 1.0)  # give it a parameter so it looks like a real trial
    # tell the driver which trial number is now RUNNING, then block so it can be killed
    with open(number_file, "w") as f:
        f.write(str(trial.number))
    sys.stdout.flush()
    time.sleep(120)  # driver will SIGKILL this process during the sleep


# Driver: run the three parts and print observed states.
def driver():
    here = os.path.dirname(os.path.abspath(__file__))
    journal_path = os.path.join(here, "s2_journal.log")
    number_file = os.path.join(here, "s2_running_number.txt")
    # clean slate for reproducibility
    for p in (journal_path, number_file):
        if os.path.exists(p):
            os.remove(p)
    study_name = "s2_stale_demo"
    storage = make_storage(journal_path)
    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    # ---- (a) journal storage has no heartbeat; fail_stale_trials is a no-op ----
    print("== part (a): heartbeat support on journal storage ==")
    print(f"JournalStorage isinstance BaseHeartbeat = {isinstance(storage, BaseHeartbeat)}")
    print(f"is_heartbeat_enabled(journal storage) = {is_heartbeat_enabled(storage)}")
    # ask a throwaway trial and call fail_stale_trials: it must NOT change or raise
    probe = study.ask()
    optuna.storages.fail_stale_trials(study)  # experimental warning is expected
    probe_state = optuna.load_study(study_name=study_name, storage=make_storage(journal_path)).trials[probe.number].state
    print(f"trial {probe.number} state after fail_stale_trials on journal = {probe_state.name} (unchanged; no-op)")

    # ---- (b) a subprocess asks a trial then is SIGKILLed; trial stays RUNNING ----
    print("\n== part (b): dead worker leaves a RUNNING trial ==")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "worker", journal_path, study_name, number_file]
    )
    # wait until the worker has announced its running trial number
    while not os.path.exists(number_file):
        time.sleep(0.05)
    time.sleep(0.5)  # let the sleep(120) actually begin
    with open(number_file) as f:
        dead_number = int(f.read().strip())
    # kill the worker hard (mimics Slurm walltime / node failure): no cleanup runs
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    print(f"worker asked trial number={dead_number}, then was SIGKILLed (returncode={proc.returncode})")
    # from THIS (different) process, read the trial's state out of the shared journal
    view = optuna.load_study(study_name=study_name, storage=make_storage(journal_path))
    before = view.trials[dead_number].state
    print(f"state of trial {dead_number} seen from the controller BEFORE cleanup = {before.name}")

    # ---- (c) recovery: tell the orphaned RUNNING trial FAIL from the controller ----
    print("\n== part (c): controller fails the orphaned trial ==")
    # study.tell can move a RUNNING trial (created by the dead worker) to FAIL
    view.tell(dead_number, state=optuna.trial.TrialState.FAIL)
    after = optuna.load_study(study_name=study_name, storage=make_storage(journal_path)).trials[dead_number].state
    print(f"state of trial {dead_number} AFTER controller study.tell(..., FAIL) = {after.name}")


# Dispatch on the role argument for subprocess children.
if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver()
    elif sys.argv[1] == "worker":
        role_dead_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise ValueError(f"unknown role {sys.argv[1]!r}")
