#!/usr/bin/env python
"""Collaborator worker: the run's own claim-run-mark loop plus a fail-fast guard.

Reuses the run's own `slurm/worker.py` unchanged — its claim window, its train.py argv builder and
its marker moves — so a collaborator's runs are byte-for-byte the same work as the owner's. The one
addition: if MAX_FAST_FAILURES claimed configurations in a row all fail in under FAST_FAIL_SECONDS
each (an import-time, environment or permissions failure, not hours of real training), this worker
STOPS claiming, writes ONE problem report naming the markers it sent to failed/, and exits loudly.

It never renames anything back itself. The collaborator side never repairs shared state; the owner's
monitor loop (`slurm/requeue_orphans.py`) reads the report and re-pends the named markers.

Env (set by the *_collab.slurm scripts): RUN_DIR, PROJ_DIR, SWEEP_ID.
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

    One file per event, named by timestamp, user and worker id — a single writer per file, never an
    append, so two workers reporting at the same moment cannot collide.
    """
    os.makedirs(PROBLEMS_OPEN, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(PROBLEMS_OPEN, f"{stamp}_{getpass.getuser()}_fast-failures_{worker.WID}.md")
    with open(path, "w") as fh:
        fh.write(f"# Fast-failure report — worker {worker.WID} ({getpass.getuser()})\n\n"
                 f"{len(fast_failed)} claimed configurations in a row failed in under "
                 f"{FAST_FAIL_SECONDS}s each. That points at a broken environment or a permissions\n"
                 f"problem on the submitting side, not at the configurations. The markers below now\n"
                 f"sit in failed/ and should be re-pended by the owner's monitor "
                 f"(requeue_orphans.py\nparses the MARKER lines). This worker stopped claiming and "
                 f"exited.\n\n")
        for name in fast_failed:
            fh.write(f"MARKER: {name}\n")
    return path


def main():
    """The run's own claim-run-mark loop with the consecutive-fast-failure guard around it."""
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "30"))))
    worker.log(f"collaborator worker started; fail-fast after {MAX_FAST_FAILURES} runs under "
               f"{FAST_FAIL_SECONDS}s")
    n_done = n_fail = 0
    fast_failures = []  # consecutive fast-fail marker names; any success or slow run resets it
    while True:
        name, path = worker.claim()
        if name is None:
            # report BEFORE exiting: one or two accumulated fast failures would otherwise strand
            # their markers in failed/ with no report naming them
            if fast_failures:
                report = write_problem_report(fast_failures)
                worker.log(f"queue empty with {len(fast_failures)} unreported fast failures — "
                           f"problem report: {report}")
            worker.log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return 0
        with open(path) as fh:
            cfg = json.load(fh)
        worker.log(f"claimed {name} :: {cfg['env_setup']} beta={cfg['beta']} seed={cfg['a_seed']} "
                   f"steps={cfg['fixed']['total_timesteps']}")
        t0 = time.time()
        rc = subprocess.call(worker.build_cmd(cfg), cwd=worker.PROJ)
        elapsed = time.time() - t0
        dest = worker.DONE if rc == 0 else worker.FAILED
        try:
            os.rename(path, os.path.join(dest, name))
        except OSError:
            pass
        n_done += (rc == 0)
        n_fail += (rc != 0)
        worker.log(f"{name} rc={rc} in {elapsed:.0f}s -> {os.path.basename(dest)} "
                   f"(running totals: {n_done} done, {n_fail} failed)")
        # the guard: consecutive FAST failures mean the environment is broken, not the configurations
        if rc != 0 and elapsed < FAST_FAIL_SECONDS:
            fast_failures.append(name)
            if len(fast_failures) >= MAX_FAST_FAILURES:
                report = write_problem_report(fast_failures)
                worker.log(f"FAIL-FAST: {len(fast_failures)} consecutive fast failures — stopping "
                           f"this worker; problem report: {report}")
                return 4
        else:
            fast_failures = []


if __name__ == "__main__":
    sys.exit(main())
