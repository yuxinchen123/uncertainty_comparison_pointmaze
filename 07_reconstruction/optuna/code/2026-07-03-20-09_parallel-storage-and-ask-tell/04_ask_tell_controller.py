"""Point 4: drive Optuna by ask-and-tell across a file work queue, the exact pattern the project
uses. The controller owns the study (TPESampler + JournalStorage), asks trials and writes their
params as pending JSONs; four separate worker PROCESSES (04_ask_tell_worker.py) run the objective
and write result JSONs; the controller polls results and tells the study, keeping ~8 trials in
flight until 40 are told. Also demonstrates telling a plain int trial number and telling one
deliberately failed trial with TrialState.FAIL, showing the study keeps going afterward."""

import os
import sys
import json
import time
import shutil
import subprocess
import optuna


def build_storage(journal_path):
    """Build the JournalStorage backed by a journal file with the NFSv3-recommended open lock."""
    # same construction the workers would use if they attached (here only the controller attaches)
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    )


def fresh_dir(path):
    """Remove and recreate a directory so each run starts from an empty queue."""
    # wipe any leftovers from a previous run, then make the empty directory
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def ask_and_write(study, pending_dir):
    """Ask one trial, fix its search space with suggest_*, write its params as a pending JSON."""
    # ask() returns a Trial whose params are fixed by calling suggest_* right away
    trial = study.ask()
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_float("y", -10.0, 10.0)
    # write the config a worker will claim: identity + fixed params
    config = {"trial_number": trial.number, "params": {"x": x, "y": y}}
    tmp = os.path.join(pending_dir, f"{trial.number}.json.tmp")
    final = os.path.join(pending_dir, f"{trial.number}.json")
    with open(tmp, "w") as f:
        json.dump(config, f)
    os.rename(tmp, final)
    return trial.number


def main():
    """Set up the study and queue, demonstrate a FAIL tell, then run the ask/tell file-queue loop."""
    # journal file on NFS + fresh queue directories mirroring the project layout
    here = os.path.dirname(os.path.abspath(__file__))
    journal_path = os.path.join(here, "04_study_journal.log")
    if os.path.exists(journal_path):
        os.remove(journal_path)
    queue_dir = os.path.join(here, "04_queue")
    for sub in ("pending", "running", "done", "results"):
        fresh_dir(os.path.join(queue_dir, sub))
    pending_dir = os.path.join(queue_dir, "pending")
    results_dir = os.path.join(queue_dir, "results")
    stop_path = os.path.join(queue_dir, "STOP")

    # create the study with a seeded TPE sampler so the ask sequence is reproducible
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(
        study_name="point4_ask_tell",
        storage=build_storage(journal_path),
        sampler=sampler,
        direction="minimize",
    )

    # --- demonstrate telling a deliberately failed trial, then that the study continues ---
    fail_trial = study.ask()
    fail_trial.suggest_float("x", -10.0, 10.0)
    fail_trial.suggest_float("y", -10.0, 10.0)
    study.tell(fail_trial, state=optuna.trial.TrialState.FAIL)
    fail_state = study.get_trials(deepcopy=False)[fail_trial.number].state
    print(f"deliberately failed trial number={fail_trial.number} -> state={fail_state.name}", flush=True)
    # prove the study still accepts a fresh ask after a FAIL (it does not stop the study)
    probe = study.ask()
    print(f"study still asks after FAIL: got trial number={probe.number}; telling it back as FAIL too", flush=True)
    study.tell(probe.number, state=optuna.trial.TrialState.FAIL)

    # --- launch 4 worker processes that will run the objective from the file queue ---
    # each worker's stdout/stderr goes to its own log file so a crash is visible, not swallowed
    worker_script = os.path.join(here, "04_ask_tell_worker.py")
    worker_logs = []
    workers = []
    for i in range(4):
        log_handle = open(os.path.join(queue_dir, f"worker_{i}.log"), "w")
        worker_logs.append(log_handle)
        workers.append(
            subprocess.Popen(
                [sys.executable, worker_script, queue_dir],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        )
    print("launched 4 worker processes", flush=True)

    # --- main ask/tell loop: keep ~8 trials in flight until 40 COMPLETE tells happen ---
    target_complete = 40
    max_in_flight = 8
    in_flight = set()
    told_complete = 0
    asked = 0
    told_via_int = 0
    while told_complete < target_complete:
        # top up the queue so up to 8 trials are outstanding, never asking past the target
        while len(in_flight) < max_in_flight and (asked < target_complete):
            number = ask_and_write(study, pending_dir)
            in_flight.add(number)
            asked += 1

        # poll results/: each finished file carries {trial_number, value}
        for fname in sorted(os.listdir(results_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(results_dir, fname)) as f:
                result = json.load(f)
            trial_number = result["trial_number"]
            value = result["value"]
            # tell the study using the plain INTEGER trial number (not the Trial object)
            assert isinstance(trial_number, int)
            study.tell(trial_number, value)
            told_via_int += 1
            # clear the result marker and drop it from the in-flight set
            os.remove(os.path.join(results_dir, fname))
            in_flight.discard(trial_number)
            told_complete += 1
        # if every worker has already exited before we are done, stop instead of hanging forever
        if told_complete < target_complete and all(w.poll() is not None for w in workers):
            raise RuntimeError(
                f"all workers exited early with only {told_complete}/{target_complete} told; "
                f"see worker_*.log in {queue_dir}"
            )
        # brief pause so we are not busy-spinning the filesystem
        time.sleep(0.05)

    # signal workers to stop, then wait for them to exit cleanly
    open(stop_path, "w").close()
    for w in workers:
        w.wait()
        assert w.returncode == 0, f"a worker exited with {w.returncode}"
    for log_handle in worker_logs:
        log_handle.close()

    # --- final checks on the study ---
    all_trials = study.get_trials(deepcopy=False)
    complete = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"total trials: {len(all_trials)}")
    print(f"COMPLETE: {len(complete)}  FAIL: {len(failed)}")
    print(f"tells made with a plain int trial number: {told_via_int}")
    # the point-4 assertions
    assert len(complete) == 40, f"expected 40 COMPLETE, got {len(complete)}"
    assert len(failed) == 2, f"expected 2 FAIL (the two demonstrated), got {len(failed)}"
    print(f"best value: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")
    print("POINT 4 OK: 40 COMPLETE via file-queue ask/tell; int-number tell works; FAIL tell does not stop the study")


main()
