#!/usr/bin/env python
"""Owner-only reclaim step, called each cycle by monitor.sh (adapted from run 8.1's, itself the
run-3.2.3 hardening of 2026-07-11).

Three phases, all through the visible failed/ folder (user decision):
1. ORPHAN DETECTION: a running/ marker whose claiming Slurm job is terminal (walltime kill, node
   failure, cancellation) moves running/ -> failed/ and is recorded in the append-only ledger
   slurm/killed_orphans_<sweep_id>.txt. The claiming job id comes from the worker logs' "claimed"
   lines (owner logs/ AND for_collaborator/logs/); when no claim line is found, a staleness
   fallback fires (per-run JSON checkpoint older than STALE_SECONDS).
2. REQUEUE: every ledger-recorded orphan still sitting in failed/ has its partial JSON archived
   to data/<sweep>/killed_attempts_<date>/ and its marker returned failed/ -> its ORIGIN POOL
   (pending_1m or pending_10m, read from the marker JSON's "pool" field — the run-8.1.2 change; a
   killed 10M baseline run restarts from scratch, this run has no checkpoints). Application
   failures (worker-recorded rc!=0, NOT in the ledger) stay in failed/ forever, so a crash-looping
   config is never resurrected.
3. PROBLEM REPORTS: files in for_collaborator/problems/open/*.md with machine-readable
   "MARKER: <name>" lines get those markers requeued the same way, then the report moves to
   problems/resolved/ with a resolution footer. Reports without MARKER lines are only surfaced.

Run-8.1.2 decided-config redirect: config keys with a stage-1 verdict (pruned OR survivor, from
slurm/stage1_decisions_<sweep_id>.jsonl) send their orphans to pruned/ instead of a pending pool —
a pruned config's race is over, and a survivor already reached its 100 completed seeds, so a rerun
would only burn hours and delay the completion sentinel. The config key is read from the MARKER
JSON itself (the run-8.1 id-arithmetic shortcut assumed that run's layout and is gone).

Safety: acts only inside the given sweep's queue; never requeues a marker whose job still appears
in squeue or whose sacct state is not terminal; any scontrol/sacct/read failure skips that marker
until the next cycle (fail-safe = do nothing).

Usage:  python requeue_orphans.py --sweep_id <id> [--dry]
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
POOLS = ("pending_1m", "pending_10m")


def log(msg):
    """Print a timestamped reclaim line, flushed (monitor.log is the live trace)."""
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')} requeue_orphans] {msg}", flush=True)


def sweep_paths(sweep_id):
    """The sweep's queue subdirs, data dir, ledger, and problems dirs as one dict."""
    q = os.path.join(RUN_DIR, "queue", sweep_id)
    return {
        "sweep_id": sweep_id,
        "pruned": os.path.join(q, "pruned"),
        "running": os.path.join(q, "running"),
        "failed": os.path.join(q, "failed"),
        "pending_1m": os.path.join(q, "pending_1m"),
        "pending_10m": os.path.join(q, "pending_10m"),
        "local": os.path.join(RUN_DIR, "data", sweep_id, "local"),
        "archive": os.path.join(RUN_DIR, "data", sweep_id,
                                f"killed_attempts_{time.strftime('%Y-%m-%d')}"),
        "ledger": os.path.join(HERE, f"killed_orphans_{sweep_id}.txt"),
        "problems_open": os.path.join(RUN_DIR, "for_collaborator", "problems", "open"),
        "problems_resolved": os.path.join(RUN_DIR, "for_collaborator", "problems", "resolved"),
    }


def last_claim_jobid(marker_name):
    """(jobid, claim timestamp) of the latest visible claim line for this marker, or None.
    before: logs contain '[.. worker 6532xxx.4.19 sweep=..] claimed 00030_of_24200_...json ..'
    after:  '6532xxx' (the WID's first dot-field), from the LAST such line across both log dirs."""
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
    # lines start with "[YYYY-MM-DDTHH:MM:SS ...", so a lexical sort IS chronological; take the
    # LATEST claim (the attempt that currently owns the marker)
    last = sorted(lines)[-1]
    m = re.search(r"worker (\d+)\.", last)
    ts = re.match(r"\[(\S+) ", last)
    return (m.group(1), ts.group(1) if ts else "") if m else None


def job_is_terminal(jobid):
    """True only when the job is absent from squeue AND sacct reports a terminal state.
    Any command failure returns False (fail-safe: treat as possibly alive, skip this cycle)."""
    try:
        alive = subprocess.run(["squeue", "-h", "-j", jobid, "-o", "%i"],
                               capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    if alive.stdout.strip():
        return False
    # sacct parent line: terminal state required; -P prints the FULL state (fixed-width output
    # truncates OUT_OF_MEMORY to "OUT_OF_ME+", which would leave OOM orphans unreclaimed)
    try:
        acct = subprocess.run(["sacct", "-j", jobid, "-X", "-n", "-P", "--format=State"],
                              capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    state = acct.stdout.strip().split("\n")[0].strip().split()[0] if acct.stdout.strip() else ""
    return any(state.startswith(t) for t in TERMINAL_STATES)


def json_path_for(marker_name, local_dir):
    """The per-run JSON path for a marker. before: '00030_of_24200_...json' -> after:
    <local>/00030_of_24200.json (train.py names the record by id_of_total only)."""
    m = re.match(r"(\d+_of_\d+)_", marker_name)
    return os.path.join(local_dir, m.group(1) + ".json") if m else None


def checkpoint_is_stale(marker_name, p):
    """Staleness fallback when no claim line exists: True ONLY when the run's JSON checkpoint
    EXISTS and has not moved for STALE_SECONDS. A marker with no claim line and no checkpoint is
    NEVER declared stale (the 2026-07-11 double-execution hazard: os.rename preserves mtime)."""
    jp = json_path_for(marker_name, p["local"])
    if not jp or not os.path.exists(jp):
        return False
    try:
        return (time.time() - os.path.getmtime(jp)) > STALE_SECONDS
    except OSError:
        return False


def ledger_read(p):
    """Parse the append-only ledger into per-marker COUNTS of kill and requeue events (counts, not
    sets, so a marker whose requeued attempt is itself killed can be requeued again)."""
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
    """Config keys with a stage-1 verdict (pruned or survivor) — their orphans go to pruned/."""
    decided = set()
    decisions = os.path.join(HERE, f"stage1_decisions_{p['sweep_id']}.jsonl")
    if os.path.exists(decisions):
        with open(decisions) as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("verdict") in ("pruned", "survivor"):
                    decided.add(entry["config_key"])
    return decided


def marker_fields(path):
    """(config_key, pool) read from the marker JSON itself; (None, 'pending_1m') on read failure.
    before: {'config_key': 'AntMaze...|alg2.1|lr0.01|b3', 'pool': 'pending_1m', ...}
    after:  ('AntMaze...|alg2.1|lr0.01|b3', 'pending_1m')"""
    try:
        with open(path) as fh:
            d = json.load(fh)
        pool = d.get("pool", "pending_1m")
        return d.get("config_key"), pool if pool in POOLS else "pending_1m"
    except (json.JSONDecodeError, OSError):
        return None, "pending_1m"


def ledger_append(p, kind, name, extra=""):
    """Append one journal line: kind<TAB>marker<TAB>extra<TAB>timestamp."""
    with open(p["ledger"], "a") as fh:
        fh.write(f"{kind}\t{name}\t{extra}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")


def archive_partial(marker_name, p, dry):
    """Move the marker's completed=false JSON to the killed_attempts archive (evidence keeping);
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
    """Return one failed/ marker to its origin pool (or pruned/ for a decided config); ledger it."""
    src = os.path.join(p["failed"], name)
    if not os.path.exists(src):
        return False
    archive_partial(name, p, dry)
    key, pool = marker_fields(src)
    decided = key is not None and key in decided_config_keys(p)
    dest_state = "pruned" if decided else pool
    log(f"requeue: {name} failed/ -> {dest_state}/" + (f" ({why})" if why else "")
        + (" (config already decided)" if decided else ""))
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
    # from it double-runs a live run (race hit 2026-07-19, five markers)
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
    """Return every ledger-recorded orphan still in failed/ to its origin pool."""
    killed, requeued = ledger_read(p)
    for name in sorted(n for n in killed if killed[n] > requeued[n]):
        requeue_one(name, p, dry)


def phase3_problems(p, dry):
    """Requeue markers named in collaborator problem reports; move handled reports to resolved/."""
    if not os.path.isdir(p["problems_open"]):
        return
    for path in sorted(glob.glob(os.path.join(p["problems_open"], "*.md"))):
        with open(path) as fh:
            text = fh.read()
        markers = re.findall(r"^MARKER:\s*(\S+)$", text, flags=re.MULTILINE)
        if not markers:
            # free-text report: a human decision is needed; surface it every cycle, leave in open/
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


def phase2b_repend_all_failed(p, dry):
    """Blanket re-pend of EVERYTHING left in failed/ into its origin pool (the run-8.1 monitor.sh
    step 2b, made pool-aware): the run code is tested and no config fails deterministically, so
    every failed marker is a retryable infrastructure kill. Runs AFTER phase 2, so ledger-recorded
    orphans were already handled; this catches rc!=0 exits with no ledger line."""
    if not os.path.isdir(p["failed"]):
        return
    n = 0
    for name in sorted(os.listdir(p["failed"])):
        if name.endswith(".json") and requeue_one(name, p, dry, why="blanket re-pend"):
            n += 1
    log(f"blanket re-pend: {n} failed marker(s) returned to their pools")


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
