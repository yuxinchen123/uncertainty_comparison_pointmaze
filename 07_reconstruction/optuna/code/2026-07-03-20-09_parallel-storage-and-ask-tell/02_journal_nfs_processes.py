"""Point 2: create a JournalStorage study on the real NFS filesystem, then launch 8 concurrent
worker PROCESSES that each attach and run 5 trials. After all exit, assert the study holds
exactly 40 COMPLETE trials. This verifies many processes on NFS share one study through the
journal file with the NFSv3 open-lock."""

import os
import sys
import subprocess
import optuna


def build_storage(journal_path):
    """Build the JournalStorage backed by a journal file with the NFSv3-recommended open lock."""
    # JournalFileOpenLock is the docstring-recommended lock for NFSv3 or later
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    )


def main():
    """Create the study, spawn 8 worker processes, wait, then check trial count and states."""
    # place the journal file inside this run folder, which is on NFS (verified vers=3)
    here = os.path.dirname(os.path.abspath(__file__))
    journal_path = os.path.join(here, "02_study_journal.log")
    # start from a clean file so the assertion of exactly 40 trials is meaningful
    if os.path.exists(journal_path):
        os.remove(journal_path)

    # create the shared study once (minimize the noisy quadratic)
    study_name = "point2_nfs_shared"
    storage = build_storage(journal_path)
    optuna.create_study(study_name=study_name, storage=storage, direction="minimize")
    print(f"created study {study_name!r} on {journal_path}", flush=True)

    # launch 8 worker processes; each runs 02_journal_worker.py doing 5 trials => 40 total
    worker_script = os.path.join(here, "02_journal_worker.py")
    procs = []
    for i in range(8):
        # each Popen is a distinct OS process, not a thread
        p = subprocess.Popen(
            [sys.executable, worker_script, study_name, journal_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append(p)
    print("launched 8 worker processes", flush=True)

    # wait for every worker and print its output; a nonzero exit is surfaced (no swallowing)
    for i, p in enumerate(procs):
        out, _ = p.communicate()
        print(f"--- worker {i} exit={p.returncode} ---")
        print(out.strip())
        assert p.returncode == 0, f"worker {i} failed with exit {p.returncode}"

    # reload the study from the shared journal and check the aggregate result
    final_study = optuna.load_study(study_name=study_name, storage=build_storage(journal_path))
    trials = final_study.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"total trials in study: {len(trials)}")
    print(f"COMPLETE trials: {len(complete)}")
    # the point-2 assertions: exactly 40 trials and all COMPLETE
    assert len(trials) == 40, f"expected 40 trials, got {len(trials)}"
    assert len(complete) == 40, f"expected 40 COMPLETE, got {len(complete)}"
    print(f"best value: {final_study.best_value:.4f}")
    print(f"best params: {final_study.best_params}")
    print("POINT 2 OK: 8 processes shared one NFS journal study; 40/40 COMPLETE")


main()
