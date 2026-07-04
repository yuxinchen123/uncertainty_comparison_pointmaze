"""Measure journal-file storage under many concurrent workers and on load.

(a) 32 worker processes, each running study.optimize(n_trials=10) into ONE shared
    journal study = 320 trials. Measure wall time; verify exactly 320 COMPLETE with
    unique numbers (no lost or duplicated trials under the file lock).
(b) Grow a separate study to 1000 trials; report the journal file size in MB and the
    time a fresh process needs to load_study and read len(study.trials) (the journal
    is replayed from the first record on load, so this cost grows with trial count).
"""

import os
import subprocess
import sys
import time

import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)


# Build the shared journal storage on the real folder, as the sweep does.
def make_storage(journal_path):
    # journal backend guarded by the cross-process file-open lock
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    backend = optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


# A trivial fast objective so the run is dominated by storage, not compute.
def objective(trial):
    # one parameter, cheap arithmetic value
    x = trial.suggest_float("x", -10.0, 10.0)
    return x * x


# Worker role for part (a): load the shared study and run n_trials via optimize.
def role_worker(journal_path, study_name, n_trials):
    # each worker uses the default sampler; contention is on the journal file lock
    study = optuna.load_study(study_name=study_name, storage=make_storage(journal_path))
    study.optimize(objective, n_trials=int(n_trials))


# Role for part (b) fresh-load timing: time load_study + len(study.trials) from cold.
def role_coldload(journal_path, study_name):
    # measure only the replay-on-load + trial-list read
    t0 = time.perf_counter()
    study = optuna.load_study(study_name=study_name, storage=make_storage(journal_path))
    n = len(study.trials)
    dt = time.perf_counter() - t0
    print(f"COLDLOAD n_trials={n} load_seconds={dt:.3f}")


# Driver: run part (a) then part (b), printing measured numbers.
def driver():
    here = os.path.dirname(os.path.abspath(__file__))

    # ---- part (a): 32 workers x 10 trials into one journal study ----
    journal_a = os.path.join(here, "s4a_journal.log")
    if os.path.exists(journal_a):
        os.remove(journal_a)
    study_name_a = "s4a_scale"
    optuna.create_study(study_name=study_name_a, storage=make_storage(journal_a), direction="minimize")
    n_workers, per_worker = 32, 10
    print(f"== part (a): {n_workers} workers x {per_worker} trials = {n_workers * per_worker} into one journal ==")
    t0 = time.perf_counter()
    # launch all workers roughly at once
    procs = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker",
                          journal_a, study_name_a, str(per_worker)])
        for _ in range(n_workers)
    ]
    # wait for every worker to finish and check none crashed
    for p in procs:
        p.wait()
        assert p.returncode == 0, f"a worker exited with {p.returncode}"
    wall = time.perf_counter() - t0
    # read the final study once and check counts + uniqueness of trial numbers
    study = optuna.load_study(study_name=study_name_a, storage=make_storage(journal_a))
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    numbers = [t.number for t in trials]
    print(f"total trials={len(trials)} complete={len(complete)} "
          f"unique_numbers={len(set(numbers))} min={min(numbers)} max={max(numbers)}")
    print(f"wall_seconds={wall:.2f} throughput={len(trials)/wall:.1f} trials/s "
          f"journal_MB={os.path.getsize(journal_a)/1e6:.3f}")

    # ---- part (b): a 1000-trial study, file size and cold-load time ----
    journal_b = os.path.join(here, "s4b_journal.log")
    if os.path.exists(journal_b):
        os.remove(journal_b)
    study_name_b = "s4b_thousand"
    print(f"\n== part (b): grow one study to 1000 trials, then time a fresh-process load ==")
    study_b = optuna.create_study(study_name=study_name_b, storage=make_storage(journal_b), direction="minimize")
    # single process fills 1000 trials quickly
    study_b.optimize(objective, n_trials=1000)
    size_mb = os.path.getsize(journal_b) / 1e6
    print(f"journal file size at 1000 trials = {size_mb:.3f} MB")
    # time a genuinely fresh process doing load_study + len(study.trials)
    subprocess.run([sys.executable, os.path.abspath(__file__), "coldload", journal_b, study_name_b], check=True)


# Dispatch on role for subprocess children.
if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver()
    elif sys.argv[1] == "worker":
        role_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "coldload":
        role_coldload(sys.argv[2], sys.argv[3])
    else:
        raise ValueError(f"unknown role {sys.argv[1]!r}")
