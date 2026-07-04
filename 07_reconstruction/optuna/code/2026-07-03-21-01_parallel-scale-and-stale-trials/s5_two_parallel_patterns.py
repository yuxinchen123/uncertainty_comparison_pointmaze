"""Compare the two parallel Optuna patterns on shared journal storage.

Pattern (a) worker-owned optimize: 8 worker processes each load_study +
    study.optimize(...). The sampler runs INSIDE EACH WORKER. While workers sleep
    inside their trials, a separate observer process lists study.trials and counts
    how many trials from OTHER processes are visible as RUNNING.

Pattern (b) controller ask/tell: ONE controller calls study.ask()/study.tell()
    and keeps K=8 trials asked-but-untold at a time; workers only evaluate the
    objective (via a file queue) and never touch the sampler. The sampler runs
    only in the controller.
"""

import json
import os
import subprocess
import sys
import time

import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)


# Build the shared journal storage on the real folder.
def make_storage(journal_path):
    # journal backend with the cross-process file-open lock
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    backend = optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


# ---------------- pattern (a) ----------------

# Worker for pattern (a): its OWN study object runs the sampler via optimize.
def role_worker_a(journal_path, study_name, n_trials):
    # each worker loads the study and drives ask/evaluate/tell itself through optimize
    study = optuna.load_study(study_name=study_name, storage=make_storage(journal_path))

    # objective that sleeps so trials overlap across processes
    def objective(trial):
        # a slow trial keeps this process's trial RUNNING for a while
        x = trial.suggest_float("x", -5.0, 5.0)
        time.sleep(0.4)
        return x * x

    study.optimize(objective, n_trials=int(n_trials))


# Driver for pattern (a): launch workers, observe RUNNING trials from another process.
def driver_pattern_a(here):
    journal = os.path.join(here, "s5a_journal.log")
    if os.path.exists(journal):
        os.remove(journal)
    study_name = "s5a_worker_optimize"
    optuna.create_study(study_name=study_name, storage=make_storage(journal), direction="minimize")
    n_workers, per_worker = 8, 2
    print(f"== pattern (a) worker-owned optimize: {n_workers} workers x {per_worker} trials ==")
    procs = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker_a",
                          journal, study_name, str(per_worker)])
        for _ in range(n_workers)
    ]
    # observer loop in THIS process: how many RUNNING trials from other processes are visible?
    max_running_seen = 0
    while any(p.poll() is None for p in procs):
        # fresh read of the shared study from the observer process
        view = optuna.load_study(study_name=study_name, storage=make_storage(journal))
        running = [t for t in view.get_trials(deepcopy=False)
                   if t.state == optuna.trial.TrialState.RUNNING]
        max_running_seen = max(max_running_seen, len(running))
        time.sleep(0.05)
    for p in procs:
        p.wait()
    print(f"max concurrent RUNNING trials visible from the observer process = {max_running_seen}")
    print(f"(these RUNNING trials were created by the {n_workers} separate worker processes)")
    print("sampler location for pattern (a): runs INSIDE EACH WORKER (each worker's study.ask)")


# ---------------- pattern (b) ----------------

# Worker for pattern (b): evaluate only; never call the sampler.
def role_worker_b(ask_dir, done_dir, sentinel):
    # poll the queue: claim an ask file, evaluate its params, write the result
    while True:
        # stop once the controller signals done and no asks remain
        names = [n for n in os.listdir(ask_dir) if n.endswith(".json")]
        if not names:
            if os.path.exists(sentinel):
                return
            time.sleep(0.02)
            continue
        name = sorted(names)[0]
        src = os.path.join(ask_dir, name)
        claimed = src + ".claimed"
        # atomic claim: exactly one worker wins the rename; the loser sees FileNotFoundError
        try:
            os.rename(src, claimed)
        except FileNotFoundError:
            # another worker claimed it first; try again
            continue
        with open(claimed) as f:
            payload = json.load(f)
        # the objective evaluation (the only thing a worker does)
        x = payload["x"]
        time.sleep(0.2)
        value = x * x
        # publish the result and remove the claimed file
        with open(os.path.join(done_dir, f"{payload['number']}.json"), "w") as f:
            json.dump({"number": payload["number"], "value": value}, f)
        os.remove(claimed)


# Driver for pattern (b): the single controller runs the sampler; keeps K outstanding.
def driver_pattern_b(here):
    journal = os.path.join(here, "s5b_journal.log")
    ask_dir = os.path.join(here, "s5b_ask")
    done_dir = os.path.join(here, "s5b_done")
    sentinel = os.path.join(here, "s5b_sentinel")
    # clean slate
    import shutil
    for d in (ask_dir, done_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for p in (journal, sentinel):
        if os.path.exists(p):
            os.remove(p)
    study_name = "s5b_controller"
    # the controller owns the ONLY sampler in this pattern
    study = optuna.create_study(
        study_name=study_name, storage=make_storage(journal), direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0, constant_liar=True),
    )
    n_workers, N, K = 8, 24, 8
    print(f"\n== pattern (b) controller ask/tell: 1 controller, {n_workers} eval workers, "
          f"keep K={K} outstanding, N={N} total ==")
    workers = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker_b",
                          ask_dir, done_dir, sentinel])
        for _ in range(n_workers)
    ]
    outstanding = {}   # trial number -> True while asked-but-untold
    asked = 0
    told = 0
    max_outstanding = 0
    # controller loop: refill asks up to K, then collect any finished results and tell
    while told < N:
        # refill: ask new trials until K are in flight or we've asked all N
        while len(outstanding) < K and asked < N:
            trial = study.ask()                       # sampler executes HERE, in the controller
            x = trial.suggest_float("x", -5.0, 5.0)   # controller fixes the params
            with open(os.path.join(ask_dir, f"{trial.number}.json"), "w") as f:
                json.dump({"number": trial.number, "x": x}, f)
            outstanding[trial.number] = True
            asked += 1
        max_outstanding = max(max_outstanding, len(outstanding))
        # collect finished results and tell the controller's study
        for name in list(os.listdir(done_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(done_dir, name)
            with open(path) as f:
                res = json.load(f)
            study.tell(res["number"], res["value"])   # tell executes HERE, in the controller
            outstanding.pop(res["number"], None)
            os.remove(path)
            told += 1
        time.sleep(0.02)
    # signal workers to stop and wait for them
    open(sentinel, "w").close()
    for w in workers:
        w.wait()
    complete = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    print(f"trials told COMPLETE = {complete}; max asked-but-untold trials in flight = {max_outstanding}")
    print("sampler location for pattern (b): runs ONLY in the controller (study.ask); "
          "workers only evaluate")


# Driver: run both patterns.
def driver():
    here = os.path.dirname(os.path.abspath(__file__))
    driver_pattern_a(here)
    driver_pattern_b(here)


# Dispatch on role for subprocess children.
if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver()
    elif sys.argv[1] == "worker_a":
        role_worker_a(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "worker_b":
        role_worker_b(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise ValueError(f"unknown role {sys.argv[1]!r}")
