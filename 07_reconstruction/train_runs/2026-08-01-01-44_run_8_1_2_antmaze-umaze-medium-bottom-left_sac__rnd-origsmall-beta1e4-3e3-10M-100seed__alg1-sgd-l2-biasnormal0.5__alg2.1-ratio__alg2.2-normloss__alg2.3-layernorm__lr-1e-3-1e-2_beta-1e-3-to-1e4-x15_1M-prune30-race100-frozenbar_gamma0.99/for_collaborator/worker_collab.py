#!/usr/bin/env python
"""Collaborator worker for Run 8.1.2: the run's own claim-run-mark loop plus a fail-fast guard.

Reuses the run's own slurm/worker.py unchanged — its two-pool claim window (WORKER_POOLS priority
order plus the 10M walltime guard), its train.py argv builder (per-config `params`, the global
`fixed` args, and the WORKER_DEVICE override), and its marker moves. The one addition: if the first
MAX_FAST_FAILURES claimed configs ALL fail after less than FAST_FAIL_SECONDS each (an import-time /
environment / permissions failure, not 15 h of real training), this worker STOPS claiming, writes ONE
problem report naming the markers it sent to failed/, and exits loudly. It NEVER renames anything
back itself — the collaborator side never repairs shared state; the owner's monitor loop
(slurm/requeue_orphans.py) reads the report and re-pends the named markers into their origin pool
(coordination contract).

Env (set by the *_collab.slurm scripts): RUN_DIR, PROJ_DIR, SWEEP_ID, WORKER_POOLS, WORKER_DEVICE=cpu.
"""
import getpass
import json
import os
import random
import subprocess
import sys
import time

RUN = os.environ["RUN_DIR"]
sys.path.insert(0, os.path.join(RUN, "slurm"))
import worker  # noqa: E402  (the run's own worker module: claim(), build_cmd(), log(), paths)

FAST_FAIL_SECONDS = 600        # a run dying faster than this never reached real training
MAX_FAST_FAILURES = 3          # consecutive fast failures before this worker gives up
PROBLEMS_OPEN = os.path.join(RUN, "for_collaborator", "problems", "open")


def write_problem_report(fast_failed):
    """Write ONE new machine-readable report naming the markers this worker wrongly failed.
    One file per event, named by timestamp+user+worker id -> single writer, no appends."""
    # one report file per fail-fast event, under problems/open/ for the owner's monitor to pick up
    os.makedirs(PROBLEMS_OPEN, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(PROBLEMS_OPEN, f"{stamp}_{getpass.getuser()}_fast-failures_{worker.WID}.md")
    with open(path, "w") as fh:
        fh.write(f"# Fast-failure report — worker {worker.WID} ({getpass.getuser()})\n\n"
                 f"{len(fast_failed)} consecutive claimed configs failed in under "
                 f"{FAST_FAIL_SECONDS}s each — this points at a broken environment or permissions\n"
                 f"on the collaborator side, not at the configs. The markers below now sit in\n"
                 f"failed/ and should be re-pended into their origin pool by the owner's monitor\n"
                 f"(requeue_orphans.py parses the MARKER lines). This worker stopped claiming and\n"
                 f"exited.\n\n")
        for name in fast_failed:
            fh.write(f"MARKER: {name}\n")
    return path


def main():
    """worker.py's two-pool claim-run-mark loop with the consecutive-fast-failure guard around it."""
    # same startup jitter as the shared worker (avoids simultaneous torch imports / env builds)
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    worker.log(f"collaborator worker: pools={worker.POOLS} "
               f"device={os.environ.get('WORKER_DEVICE', '(queue default)')} "
               f"fail-fast after {MAX_FAST_FAILURES} runs under {FAST_FAIL_SECONDS}s")
    n_done = n_fail = 0
    fast_failures = []  # consecutive fast-fail marker names; any success or slow run resets it
    while True:
        # claim one pending config atomically from the first claimable pool in WORKER_POOLS order;
        # `blocked` is True when only walltime-blocked 10M items remain (this job cannot finish one)
        name, path, blocked = worker.claim()
        if name is None:
            # report BEFORE exiting: 1-2 accumulated fast failures would otherwise strand their
            # markers in failed/ with no report naming them
            if fast_failures:
                report = write_problem_report(fast_failures)
                worker.log(f"nothing claimable with {len(fast_failures)} unreported fast failures — "
                           f"problem report: {report}")
            reason = ("only walltime-blocked 10M items remain" if blocked else "queue empty")
            worker.log(f"{reason}; exiting after {n_done} done / {n_fail} failed")
            return 0
        # load the claimed config and run train.py with the run's own argv builder
        with open(path) as fh:
            cfg = json.load(fh)
        worker.log(f"claimed {name} :: task={cfg.get('task', '?')} {cfg['env_setup']} {cfg['arm']} "
                   f"beta={cfg['beta']} seed={cfg['a_seed']} steps={cfg['fixed']['total_timesteps']}")
        t0 = time.time()
        rc = subprocess.call(worker.build_cmd(cfg), cwd=worker.PROJ)
        elapsed = time.time() - t0
        # move the marker to done/ (rc 0) or failed/ (otherwise), same as the run's own worker
        dest = worker.DONE if rc == 0 else worker.FAILED
        try:
            os.rename(path, os.path.join(dest, name))
        except OSError:
            pass
        n_done += (rc == 0)
        n_fail += (rc != 0)
        worker.log(f"{name} rc={rc} in {elapsed:.0f}s -> {os.path.basename(dest)} "
                   f"(running totals: {n_done} done, {n_fail} failed)")
        # the guard: consecutive FAST failures mean the environment is broken, not the configs
        if rc != 0 and elapsed < FAST_FAIL_SECONDS:
            fast_failures.append(name)
            if len(fast_failures) >= MAX_FAST_FAILURES:
                report = write_problem_report(fast_failures)
                worker.log(f"FAIL-FAST: {len(fast_failures)} consecutive fast failures — "
                           f"stopping this worker; problem report: {report}")
                return 4
        else:
            fast_failures = []


if __name__ == "__main__":
    sys.exit(main())
