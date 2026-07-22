#!/usr/bin/env python
"""Collaborator worker: the run's own claim-run-mark loop plus a fail-fast guard (report, don't
repair).

Reuses the run's own slurm/worker.py (claim window, convergence_train.py argv, marker moves)
unchanged. The one addition: if MAX_FAST_FAILURES claimed configs in a row ALL fail after less than
FAST_FAIL_SECONDS each, that points at a broken environment or permissions on the collaborator
side (not at the configs), so this worker STOPS claiming, writes ONE problem report naming the
markers it sent to failed/, and exits loudly. It never renames anything back itself — the
collaborator side never repairs shared state; the owner's monitor loop (slurm/requeue_orphans.py)
reads the report's MARKER lines and requeues them (coordination contract).

A healthy convergence run finishes in ~15 s and returns rc=0 (never counted by the guard, which
only counts rc!=0). A FAILURE in under FAST_FAIL_SECONDS therefore means the run died before/at
real work — the fingerprint of a broken env, not of a genuine long run.

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
import worker  # noqa: E402  (the run's own worker module: claim(), build_cmd(), log(), paths, WID)

FAST_FAIL_SECONDS = 30         # a convergence run is ~15 s; a failure faster than this never worked
MAX_FAST_FAILURES = 3          # consecutive fast failures before this worker gives up
PROBLEMS_OPEN = os.path.join(RUN, "for_collaborator", "problems", "open")


def write_problem_report(fast_failed):
    """Write ONE new machine-readable report naming the markers this worker wrongly failed. One
    file per event, named by timestamp+user+worker id -> single writer, no appends."""
    # one report per fail-fast event; the owner's requeue_orphans.py parses the MARKER lines
    os.makedirs(PROBLEMS_OPEN, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(PROBLEMS_OPEN, f"{stamp}_{getpass.getuser()}_fast-failures_{worker.WID}.md")
    with open(path, "w") as fh:
        fh.write(f"# Fast-failure report — worker {worker.WID} ({getpass.getuser()})\n\n"
                 f"{len(fast_failed)} consecutive claimed configs failed in under "
                 f"{FAST_FAIL_SECONDS}s each — this points at a broken environment or permissions\n"
                 f"on the collaborator side, not at the configs. The markers below now sit in\n"
                 f"failed/ and should be requeued by the owner's monitor (requeue_orphans.py\n"
                 f"parses the MARKER lines). This worker stopped claiming and exited.\n\n")
        for name in fast_failed:
            fh.write(f"MARKER: {name}\n")
    return path


def main():
    """worker.py's claim-run-mark loop with the consecutive-fast-failure guard around it."""
    # same startup jitter as the shared worker (avoids simultaneous torch imports)
    time.sleep(random.uniform(0, float(os.environ.get("WORKER_JITTER_MAX", "15"))))
    n_done = n_fail = 0
    fast_failures = []  # consecutive fast-fail marker names; any success or slow run resets it
    while True:
        name, path = worker.claim()
        if name is None:
            # report BEFORE exiting: 1-2 accumulated fast failures would otherwise strand their
            # markers in failed/ with no report naming them
            if fast_failures:
                report = write_problem_report(fast_failures)
                worker.log(f"queue empty with {len(fast_failures)} unreported fast failures — "
                           f"problem report: {report}")
            worker.log(f"queue empty; exiting after {n_done} done / {n_fail} failed")
            return 0
        with open(path) as fh:
            cfg = json.load(fh)
        worker.log(f"claimed {name} :: {cfg['config_key']} seed={cfg['a_seed']} "
                   f"params={cfg.get('params', {})}")
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
