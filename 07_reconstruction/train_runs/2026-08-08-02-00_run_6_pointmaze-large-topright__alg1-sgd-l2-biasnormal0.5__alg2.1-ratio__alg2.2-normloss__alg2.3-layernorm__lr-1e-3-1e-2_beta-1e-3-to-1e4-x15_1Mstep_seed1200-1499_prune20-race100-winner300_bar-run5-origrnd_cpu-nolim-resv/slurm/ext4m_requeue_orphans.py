#!/usr/bin/env python
"""Owner-only orphan reclaim for the ext4m sweep, called each monitor cycle.

Simpler than the 1M sweep's requeue_orphans.py because every run is CHECKPOINTED: an orphan
(running/ marker whose claiming job is terminal) goes STRAIGHT BACK TO PENDING — the next claimer
resumes from the newest checkpoint, losing at most the steps since it (<= 0.5M). The partial
record JSON is NOT archived: it is live resume state, continued in place. There is no truncation
race, so no pruned/ redirect either.

Phases:
1. ORPHANS: running/ markers whose claiming job id (from the worker logs' "claimed" lines, owner
   and collaborator dirs) is in a terminal sacct state -> back to pending/, appended to the ledger
   slurm/killed_orphans_<sweep_id>.txt. A marker with no claim line falls back to staleness: its
   record JSON unmodified for STALE_SECONDS (a live 4M run flushes every eval, ~1 h at worst).
2. FAILED RE-PEND: markers in failed/ re-pend (the code is tested; a failure is an infrastructure
   kill mid-write — the checkpoint still stands).
3. PROBLEM REPORTS: for_collaborator/problems/open/*.md with "MARKER: <name>" lines get those
   markers re-pended; the report moves to resolved/ with a footer.

Safety: never touches a marker whose job still shows in squeue or whose sacct state is not
terminal; any read failure skips that marker until the next cycle.

Usage:  python ext4m_requeue_orphans.py --sweep_id <id> [--dry]
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)

STALE_SECONDS = 3 * 3600
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                   "OUT_OF_MEMORY", "PREEMPTED", "DEADLINE", "REVOKED", "BOOT_FAIL")


def log(msg):
    """Timestamped reclaim line, flushed."""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')} ext4m_requeue] {msg}", flush=True)


def last_claim_jobid(marker_name):
    """The job id of the LAST 'claimed <marker>' line across both log dirs, or None.
    before: '[2026-08-13T04:00:01 worker 6537001.12.4141 sweep=..] claimed 003_of_900_...json ::'
    after:  '6537001' (from the newest-mtime log containing the newest such line)."""
    hits = []
    for pat in (os.path.join(RUN_DIR, "logs", "*.log"),
                os.path.join(RUN_DIR, "for_collaborator", "logs", "*.log")):
        for path in glob.glob(pat):
            try:
                with open(path, errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            for m in re.finditer(r"\[(\S+) worker (\d+)\.\d+\.\d+ [^\]]*\] claimed "
                                 + re.escape(marker_name), text):
                hits.append((m.group(1), m.group(2)))
    return max(hits)[1] if hits else None


def job_state(jobid):
    """The job's sacct state string, or None on any read failure (fail-safe: do nothing)."""
    try:
        out = subprocess.run(["sacct", "-j", jobid, "-X", "-n", "-o", "State"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.split()[0] if out else None


def record_mtime(sweep_id, marker_name):
    """mtime of the run's record JSON, or None when it does not exist yet."""
    run_id = int(marker_name.split("_")[0])
    total = marker_name.split("_of_")[1].split("_")[0]
    path = os.path.join(RUN_DIR, "data", sweep_id, "local",
                        f"{run_id:0{len(total)}d}_of_{total}.json")
    return os.path.getmtime(path) if os.path.exists(path) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    q = os.path.join(RUN_DIR, "queue", args.sweep_id)
    pending, running, failed = (os.path.join(q, s) for s in ("pending", "running", "failed"))
    ledger = os.path.join(HERE, f"killed_orphans_{args.sweep_id}.txt")
    if not os.path.isdir(running):
        sys.exit(f"no queue for sweep {args.sweep_id}")

    # phase 1: orphaned running markers -> pending (the checkpoint carries the progress)
    n_orphan = 0
    for name in sorted(os.listdir(running)):
        jid = last_claim_jobid(name)
        if jid is not None:
            state = job_state(jid)
            if state is None or not state.startswith(TERMINAL_STATES):
                continue
            why = f"job {jid} {state}"
        else:
            # no claim line found anywhere: fall back to record staleness
            mt = record_mtime(args.sweep_id, name)
            if mt is None or time.time() - mt < STALE_SECONDS:
                continue
            why = f"no claim line; record stale {int((time.time() - mt) / 3600)} h"
        log(f"orphan {name} ({why}) -> pending")
        n_orphan += 1
        if args.dry:
            continue
        os.rename(os.path.join(running, name), os.path.join(pending, name))
        with open(ledger, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {name} {why}\n")

    # phase 2: failed markers re-pend (checkpointed, so a retry is cheap)
    n_failed = 0
    for name in sorted(os.listdir(failed)):
        log(f"failed marker {name} -> pending")
        n_failed += 1
        if not args.dry:
            os.rename(os.path.join(failed, name), os.path.join(pending, name))

    # phase 3: problem reports naming markers
    open_dir = os.path.join(RUN_DIR, "for_collaborator", "problems", "open")
    resolved_dir = os.path.join(RUN_DIR, "for_collaborator", "problems", "resolved")
    for report in sorted(glob.glob(os.path.join(open_dir, "*.md"))):
        with open(report) as fh:
            text = fh.read()
        markers = re.findall(r"^MARKER:\s*(\S+)", text, re.M)
        moved = 0
        for name in markers:
            src = os.path.join(running, name)
            if os.path.exists(src) and not args.dry:
                os.rename(src, os.path.join(pending, name))
                moved += 1
        if markers and not args.dry:
            with open(report, "a") as fh:
                fh.write(f"\n\nRESOLVED {time.strftime('%Y-%m-%dT%H:%M:%S')}: "
                         f"{moved} marker(s) re-pended by ext4m_requeue_orphans.\n")
            os.makedirs(resolved_dir, exist_ok=True)
            os.rename(report, os.path.join(resolved_dir, os.path.basename(report)))
            log(f"problem report {os.path.basename(report)}: {moved} marker(s) re-pended")
        elif markers:
            log(f"[dry] problem report {os.path.basename(report)} names {len(markers)} marker(s)")
        else:
            log(f"problem report {os.path.basename(report)} has no MARKER lines; surfacing only")

    log(f"done: {n_orphan} orphan(s), {n_failed} failed marker(s) returned to pending")


if __name__ == "__main__":
    main()
