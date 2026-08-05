#!/usr/bin/env python
"""Owner-only reclaim step, called each cycle by monitor.sh (adapted from train run 1.2's, itself
the run-3.2.3 hardening of 2026-07-11; this run has ONE pending pool, so the pool routing is gone).

Four phases, all through the visible failed/ folder:
1. ORPHAN DETECTION: a running/ marker whose claiming Slurm job is terminal (walltime kill, node
   failure, cancellation) moves running/ -> failed/ and is recorded in the append-only ledger
   slurm/killed_orphans_<sweep_id>.txt. The claiming job id comes from the worker logs' "claimed"
   lines (owner logs/ AND for_collaborator/logs/); when no claim line is found, a staleness
   fallback fires (per-run JSON checkpoint present but unmoved for STALE_SECONDS).
2. REQUEUE: every ledger-recorded orphan still sitting in failed/ has its partial JSON archived to
   data/<sweep>/killed_attempts_<date>/ and its marker returned failed/ -> pending/. This run
   writes no model checkpoints, so a killed run restarts from the beginning.
2b. BLANKET RE-PEND of anything else left in failed/: the run code is tested and no configuration
   fails deterministically, so a failed marker is a retryable infrastructure kill.
3. PROBLEM REPORTS: files in for_collaborator/problems/open/*.md with machine-readable
   "MARKER: <name>" lines get those markers requeued the same way, then the report moves to
   problems/resolved/ with a resolution footer. Reports without MARKER lines are only surfaced.

Decided-configuration redirect: a configuration with a truncation verdict (truncated OR survivor,
from slurm/truncation_decisions_<sweep_id>.jsonl) sends its orphans to pruned/ instead of pending/ —
a truncated configuration's race is over, and a survivor already has its 100 completed seeds, so a
rerun would only burn hours.

Safety: acts only inside the given sweep's queue; never requeues a marker whose job still appears in
squeue or whose sacct state is not terminal; any scontrol/sacct/read failure skips that marker until
the next cycle (fail-safe = do nothing).

Usage:  python requeue_orphans.py --sweep_id <id> [--dry] [--repend_failed]
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)

STALE_SECONDS = 6 * 3600      # fallback: a checkpoint JSON EXISTS but has not moved for 6 h
TERMINAL_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                   "OUT_OF_MEMORY", "PREEMPTED", "DEADLINE", "REVOKED", "BOOT_FAIL")


def log(msg):
    """Print a timestamped reclaim line, flushed (monitor.log is the live trace)."""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')} requeue_orphans] {msg}", flush=True)


def sweep_paths(sweep_id):
    """The sweep's queue subdirs, data dir, ledger and problem dirs as one dict."""
    q = os.path.join(RUN_DIR, "queue", sweep_id)
    return {
        "sweep_id": sweep_id,
        "pending": os.path.join(q, "pending"),
        "running": os.path.join(q, "running"),
        "failed": os.path.join(q, "failed"),
        "pruned": os.path.join(q, "pruned"),
        "local": os.path.join(RUN_DIR, "data", sweep_id, "local"),
        "archive": os.path.join(RUN_DIR, "data", sweep_id,
                                f"killed_attempts_{time.strftime('%Y-%m-%d')}"),
        "ledger": os.path.join(HERE, f"killed_orphans_{sweep_id}.txt"),
        "problems_open": os.path.join(RUN_DIR, "for_collaborator", "problems", "open"),
        "problems_resolved": os.path.join(RUN_DIR, "for_collaborator", "problems", "resolved"),
    }


def last_claim_jobid(marker_name):
    """(job id, claim timestamp) of the latest visible claim line for this marker, or None.
    before: a log line '[2026-08-05T18:02:11 worker 6540123.4.19 sweep=..] claimed 0030_of_4500_...json ..'
    after:  ('6540123', '2026-08-05T18:02:11'), taken from the LAST such line across both log dirs."""
    patterns = [os.path.join(RUN_DIR, "logs", "*.log"),
                os.path.join(RUN_DIR, "for_collaborator", "logs", "*.log")]
    files = [f for pat in patterns for f in glob.glob(pat)]
    if not files:
        return None
    proc = subprocess.run(["grep", "-h", f"claimed {marker_name}", *files],
                          capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    # lines start with "[YYYY-MM-DDTHH:MM:SS ...", so a lexical sort IS chronological
    last = sorted(lines)[-1]
    m = re.search(r"worker (\d+)\.", last)
    ts = re.match(r"\[(\S+) ", last)
    return (m.group(1), ts.group(1) if ts else "") if m else None


def job_is_terminal(jobid):
    """True only when the job is absent from squeue AND sacct reports a terminal state. Any command
    failure returns False (fail-safe: treat as possibly alive and skip this cycle)."""
    try:
        alive = subprocess.run(["squeue", "-h", "-j", jobid, "-o", "%i"],
                               capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    if alive.stdout.strip():
        return False
    # -P prints the FULL state; fixed-width output truncates OUT_OF_MEMORY to "OUT_OF_ME+", which
    # would leave out-of-memory orphans unreclaimed forever
    try:
        acct = subprocess.run(["sacct", "-j", jobid, "-X", "-n", "-P", "--format=State"],
                              capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    state = acct.stdout.strip().split("\n")[0].strip().split()[0] if acct.stdout.strip() else ""
    return any(state.startswith(t) for t in TERMINAL_STATES)


def json_path_for(marker_name, local_dir):
    """The per-run JSON path for a marker. before: '0030_of_4500_AntMaze...json'
    after: <local>/0030_of_4500.json (train.py names the record by id_of_total only)."""
    m = re.match(r"(\d+_of_\d+)_", marker_name)
    return os.path.join(local_dir, m.group(1) + ".json") if m else None


def checkpoint_is_stale(marker_name, p):
    """Staleness fallback when no claim line exists: True ONLY when the run's JSON checkpoint EXISTS
    and has not moved for STALE_SECONDS. A marker with no claim line and no checkpoint is NEVER
    declared stale (os.rename preserves mtime, so the marker's own age proves nothing)."""
    jp = json_path_for(marker_name, p["local"])
    if not jp or not os.path.exists(jp):
        return False
    try:
        return (time.time() - os.path.getmtime(jp)) > STALE_SECONDS
    except OSError:
        return False


def ledger_read(p):
    """Per-marker COUNTS of kill and requeue events (counts, not sets, so a marker whose requeued
    attempt is itself killed can be requeued again)."""
    killed, requeued = Counter(), Counter()
    if os.path.exists(p["ledger"]):
        with open(p["ledger"]) as fh:
            for line in fh:
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0] in ("killed", "problem"):
                    killed[parts[1].strip()] += 1
                elif len(parts) >= 2 and parts[0] in ("requeued", "pruned-instead"):
                    requeued[parts[1].strip()] += 1
    return killed, requeued


def decided_config_keys(p):
    """Configuration keys with a truncation verdict — their orphans go to pruned/, not pending/."""
    decided = set()
    decisions = os.path.join(HERE, f"truncation_decisions_{p['sweep_id']}.jsonl")
    if os.path.exists(decisions):
        with open(decisions) as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("verdict") in ("truncated", "survivor"):
                    decided.add(entry["config_key"])
    return decided


def marker_config_key(path):
    """The configuration key read from the marker JSON itself; None on read failure."""
    try:
        with open(path) as fh:
            return json.load(fh).get("config_key")
    except (json.JSONDecodeError, OSError):
        return None


def ledger_append(p, kind, name, extra=""):
    """Append one journal line: kind<TAB>marker<TAB>extra<TAB>timestamp."""
    with open(p["ledger"], "a") as fh:
        fh.write(f"{kind}\t{name}\t{extra}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")


def archive_partial(marker_name, p, dry):
    """Move the marker's completed=false JSON into the killed-attempts archive (evidence keeping);
    a completed=true or missing JSON is left alone."""
    jp = json_path_for(marker_name, p["local"])
    if not jp or not os.path.exists(jp):
        return
    try:
        with open(jp) as fh:
            completed = json.load(fh).get("completed", True)
    except (json.JSONDecodeError, OSError):
        completed = False  # a half-readable record is partial evidence: archive it
    if completed:
        return
    if not dry:
        os.makedirs(p["archive"], exist_ok=True)
        shutil.move(jp, os.path.join(p["archive"], os.path.basename(jp)))
    log(f"archived partial {os.path.basename(jp)} -> {os.path.basename(p['archive'])}/")


def requeue_one(name, p, dry, why=""):
    """Return one failed/ marker to pending/ (or to pruned/ for a decided configuration); ledger it."""
    src = os.path.join(p["failed"], name)
    if not os.path.exists(src):
        return False
    archive_partial(name, p, dry)
    key = marker_config_key(src)
    decided = key is not None and key in decided_config_keys(p)
    dest_state = "pruned" if decided else "pending"
    log(f"requeue: {name} failed/ -> {dest_state}/" + (f" ({why})" if why else "")
        + (" (configuration already decided)" if decided else ""))
    if not dry:
        try:
            os.rename(src, os.path.join(p[dest_state], name))
        except OSError:
            return False
        ledger_append(p, "pruned-instead" if decided else "requeued", name)
    return True


def phase1_detect(p, dry):
    """Move each orphaned running/ marker to failed/ and record it in the ledger."""
    # a claim line OLDER than the marker's last requeue is STALE information — declaring an orphan
    # from it would double-run a live run
    last_requeue = {}
    if os.path.exists(p["ledger"]):
        with open(p["ledger"]) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[0] in ("requeued", "pruned-instead"):
                    last_requeue[parts[1]] = max(last_requeue.get(parts[1], ""), parts[3])
    for name in sorted(os.listdir(p["running"])) if os.path.isdir(p["running"]) else []:
        if not name.endswith(".json"):
            continue
        claim = last_claim_jobid(name)
        if claim is not None:
            jobid, claim_ts = claim
            if name in last_requeue and claim_ts <= last_requeue[name]:
                log(f"skip {name}: claim info stale (claim {claim_ts} <= requeue "
                    f"{last_requeue[name]}) — waiting for the new claim line to appear")
                continue
            orphan, why = job_is_terminal(jobid), f"job {jobid} terminal"
        else:
            orphan, why = checkpoint_is_stale(name, p), f"no claim line; stale > {STALE_SECONDS}s"
        if not orphan:
            continue
        log(f"orphan: {name} ({why}) -> failed/")
        if not dry:
            try:
                os.rename(os.path.join(p["running"], name), os.path.join(p["failed"], name))
            except OSError:
                continue  # the worker finished in the same instant; nothing to reclaim
            ledger_append(p, "killed", name, why)


def phase2_requeue(p, dry):
    """Return every ledger-recorded orphan still in failed/ to pending/."""
    killed, requeued = ledger_read(p)
    for name in sorted(n for n in killed if killed[n] > requeued[n]):
        requeue_one(name, p, dry)


def phase2b_repend_all_failed(p, dry):
    """Blanket re-pend of everything left in failed/: no configuration fails deterministically here,
    so a failed marker is a retryable infrastructure kill. Runs AFTER phase 2."""
    if not os.path.isdir(p["failed"]):
        return
    n = 0
    for name in sorted(os.listdir(p["failed"])):
        if name.endswith(".json") and requeue_one(name, p, dry, why="blanket re-pend"):
            n += 1
    log(f"blanket re-pend: {n} failed marker(s) returned to pending/")


def phase3_problems(p, dry):
    """Requeue markers named in collaborator problem reports; move handled reports to resolved/."""
    if not os.path.isdir(p["problems_open"]):
        return
    for path in sorted(glob.glob(os.path.join(p["problems_open"], "*.md"))):
        with open(path) as fh:
            text = fh.read()
        markers = re.findall(r"^MARKER:\s*(\S+)$", text, flags=re.MULTILINE)
        if not markers:
            # free-text report: a human decision is needed; surface it every cycle, leave it open
            log(f"PROBLEM-REPORT needs attention: {path}")
            continue
        handled = []
        for name in markers:
            if requeue_one(name, p, dry, why=f"report {os.path.basename(path)}"):
                if not dry:
                    ledger_append(p, "problem", name, os.path.basename(path))
                handled.append(name)
        if not dry:
            os.makedirs(p["problems_resolved"], exist_ok=True)
            with open(path, "a") as fh:
                fh.write(f"\n---\nRESOLVED {time.strftime('%Y-%m-%dT%H:%M:%S')} by "
                         f"requeue_orphans.py: requeued {len(handled)}/{len(markers)} markers.\n")
            shutil.move(path, os.path.join(p["problems_resolved"], os.path.basename(path)))
        log(f"resolved report {os.path.basename(path)} ({len(handled)}/{len(markers)} requeued)")


def main():
    """One reclaim pass (detection, requeue, problem reports) for one sweep."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", required=True)
    ap.add_argument("--dry", action="store_true", help="print actions without renaming anything")
    ap.add_argument("--repend_failed", action="store_true",
                    help="also blanket re-pend everything left in failed/ (monitor tick step 2b)")
    args = ap.parse_args()
    p = sweep_paths(args.sweep_id)
    phase1_detect(p, args.dry)
    phase2_requeue(p, args.dry)
    if args.repend_failed:
        phase2b_repend_all_failed(p, args.dry)
    phase3_problems(p, args.dry)


if __name__ == "__main__":
    main()
