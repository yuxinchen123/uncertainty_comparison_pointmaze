#!/usr/bin/env python
"""Size and submit the ONE-SHOT worker jobs of the ext96h sweep: cpu -> nolim, NO reservation
(user rule 2026-08-13: the reservation is not used for this sweep).

Every worker claims exactly one FRESH 96-hour run (the run ends at the job's wall or the 10M
step cap; no checkpoints, no resume), so each job's slots translate one-to-one into runs.
Replenishment is a LADDER OF PENDING JOBS: beyond the jobs that can start now on free nodes,
unpinned filler jobs are queued that start as the 96-hour wave's nodes free.

WORKLOAD SHARE CAP. Each submitter owns a fixed share of the 900 runs — the owner 600 (2/3),
a collaborator 300 — enforced through an append-only slots ledger next to the id file
(ext96h_slots_<sweep>_<user>.txt, one "jobid ntasks" line per submitted job). Because workers
are one-shot, the sum of a submitter's non-cancelled ledger slots IS the number of runs their
jobs will consume; planning stops when that sum reaches the share. Cancelled jobs' slots are
released back to the budget (their claims re-pend).

Usage:
  python ext96h_plan_jobs.py --sweep_id <id>            # print the plan
  python ext96h_plan_jobs.py --sweep_id <id> --submit   # print AND submit, appending ids
  python ext96h_plan_jobs.py --sweep_id <id> --submit --script <collab .slurm> --jobname <prefix> \
                            --idfile <for_collaborator/...>   # collaborator form
"""
import argparse
import fcntl
import getpass
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# the run-6 sizing machinery is reused as a library; only the buckets and caps differ
from plan_jobs import (POOLS, HEADROOM, MIN_TASKS, candidate_nodes, pool_room,  # noqa: E402
                       reserved_nodes, walltime_for, _to_hours)


def walltime_96h(partition):
    """The job --time: exactly 96 hours (the run design — every run is one 96-hour attempt),
    unless the partition limit or an approaching maintenance window caps it lower (walltime_for
    already handles both)."""
    limit, why = walltime_for(partition)
    if _to_hours(limit) > 96:
        return "4-00:00:00", f"96-hour run design (partition allows {limit})"
    return limit, why


SENTINEL = os.path.join(RUN_DIR, "SWEEP96H_COMPLETE")
# each submitter's share of the 900 runs (one-shot workers: slots == runs)
SHARE = {"sl5nw": 600}
DEFAULT_SHARE = 300            # any collaborator
FILLER_NTASKS = 30             # unpinned pending jobs; schedule on the >=30-core class as it frees
FILLER_MEM_MB = FILLER_NTASKS * 2048


def ledger_path(idfile, sweep_id):
    """The slots ledger next to this submitter's id file."""
    return os.path.join(os.path.dirname(idfile),
                        f"ext96h_slots_{sweep_id}_{getpass.getuser()}.txt")


def read_ledger(path):
    """[(jobid, ntasks)] rows of the ledger, or [] when it does not exist."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            jid, n = line.split()
            rows.append((jid, int(n)))
    return rows


def cancelled_ids(jobids):
    """The subset of jobids whose sacct state is CANCELLED (their slots return to the budget)."""
    if not jobids:
        return set()
    out = subprocess.run(["sacct", "-j", ",".join(jobids), "-X", "-n", "-o", "JobID,State%20"],
                         capture_output=True, text=True).stdout
    return {parts[0] for line in out.splitlines()
            if (parts := line.split()) and len(parts) >= 2 and parts[1].startswith("CANCELLED")}


def budget_remaining(ledger_rows, cancelled, share):
    """Runs this submitter may still queue: share minus non-cancelled ledger slots.
    before: share=600, ledger=[('101', 32), ('102', 30)], cancelled={'102'}
    after:  600 - 32 = 568."""
    used = sum(n for jid, n in ledger_rows if jid not in cancelled)
    return max(0, share - used)


def open_pool_plan():
    """The cpu + nolim start-now plan (run-6 logic), excluding every reserved node."""
    excluded = reserved_nodes()
    plan = []
    for partition, (qos, cap) in POOLS.items():
        room, used = pool_room(qos, cap, partition)
        walltime, why = walltime_96h(partition)
        print(f"[{partition}] cap {cap}, in use {used}, headroom {HEADROOM} -> {room} threads to "
              f"fill; --time {walltime} ({why})")
        for node in candidate_nodes(partition, excluded):
            free = node["free"]
            while room >= MIN_TASKS and free >= MIN_TASKS:
                n = min(32, node["cores"], free, room)
                if n < MIN_TASKS:
                    break
                first_on_node = not any(j["node"] == node["name"] for j in plan)
                per_socket = (-(-n // node["sockets"])
                              if node["sockets"] > 1 and node["idle"] and first_on_node else None)
                plan.append({"partition": partition, "node": node["name"], "ntasks": n,
                             "mem_mb": n * node["mem"], "walltime": walltime,
                             "ntasks_per_socket": per_socket,
                             "ntasks_per_core": 2 if node["threads_per_core"] >= 2 else None})
                free -= n
                room -= n
            if room < MIN_TASKS:
                break
    return plan


def filler_jobs(n_slots, walltime):
    """Unpinned pending jobs covering n_slots runs — the replenishment ladder. They wait in the
    Slurm queue (accruing age) and start on whichever >=30-core cpu node frees first."""
    jobs = []
    while n_slots >= MIN_TASKS:
        n = min(FILLER_NTASKS, n_slots)
        jobs.append({"partition": "cpu", "node": None, "ntasks": n, "mem_mb": n * 2048,
                     "walltime": walltime, "ntasks_per_socket": None, "ntasks_per_core": 2})
        n_slots -= n
    return jobs


def sbatch_argv(job, sweep_id, jobname, script, comment):
    """The full sbatch command line for one planned job (no --nodelist for unpinned fillers)."""
    argv = ["sbatch", "--parsable", f"--job-name={jobname}", f"--comment={comment}",
            f"--partition={job['partition']}", "--nodes=1",
            f"--ntasks={job['ntasks']}", "--cpus-per-task=1",
            f"--mem={job['mem_mb']}M", f"--time={job['walltime']}",
            f"--export=ALL,SWEEP_ID={sweep_id}", script]
    if job.get("node"):
        argv.insert(-1, f"--nodelist={job['node']}")
    if job["ntasks_per_core"]:
        argv.insert(-1, f"--ntasks-per-core={job['ntasks_per_core']}")
    if job.get("ntasks_per_socket"):
        argv.insert(-1, f"--ntasks-per-socket={job['ntasks_per_socket']}")
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--jobname", default="r6e96")
    p.add_argument("--script", default=os.path.join(HERE, "ext96h_worker_cpu.slurm"))
    p.add_argument("--idfile", default=None)
    args = p.parse_args()
    if os.path.exists(SENTINEL):
        sys.exit("SWEEP96H_COMPLETE exists — this sweep is finished; not planning any jobs")
    idfile = args.idfile or os.path.join(
        HERE, f"submitted_jobids_{args.sweep_id}_{getpass.getuser()}.txt")
    ledger = ledger_path(idfile, args.sweep_id)
    rows = read_ledger(ledger)
    budget = budget_remaining(rows, cancelled_ids([j for j, _ in rows]),
                              SHARE.get(getpass.getuser(), DEFAULT_SHARE))
    pending_dir = os.path.join(RUN_DIR, "queue", args.sweep_id, "pending")
    left = len(os.listdir(pending_dir)) if os.path.isdir(pending_dir) else 0
    print(f"[share] budget remaining {budget} runs (ledger {ledger}); queue has {left} pending")

    # start-now jobs on free nodes, then the pending filler ladder, both under budget AND queue
    plan = open_pool_plan()
    allow = min(budget, left)
    trimmed, slots = [], 0
    for job in plan:
        if slots + job["ntasks"] > allow:
            job["ntasks"] = allow - slots          # shrink the last job to fit the cap exactly
            job["mem_mb"] = job["ntasks"] * 2048
        if job["ntasks"] < MIN_TASKS:
            break
        trimmed.append(job)
        slots += job["ntasks"]
    plan = trimmed
    walltime, _ = walltime_96h("cpu")
    plan += filler_jobs(allow - slots, walltime)
    total = sum(j["ntasks"] for j in plan)
    n_fill = sum(1 for j in plan if j.get("node") is None)
    print(f"\n[plan] {len(plan)} jobs ({n_fill} unpinned pending fillers), {total} worker slots "
          f"= runs\n")
    comment = f"{args.sweep_id}_{getpass.getuser()}"
    for i, job in enumerate(plan, 1):
        argv = sbatch_argv(job, args.sweep_id, f"{args.jobname}{i}", args.script, comment)
        print(" ".join(argv))
        if not args.submit:
            continue
        out = subprocess.run(argv, capture_output=True, text=True)
        jid = out.stdout.strip().split(";")[0]
        if not jid.isdigit():
            print(f"  [FAILED] {out.stderr.strip()}")
            continue
        # append id file AND slots ledger the INSTANT sbatch returns (cancel-own-ids + share cap)
        for path, line in ((idfile, jid), (ledger, f"{jid} {job['ntasks']}")):
            with open(path, "a") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.write(line + "\n")
                fh.flush()
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        print(f"  -> {jid}")
        if i % 10 == 0:
            print("  [throttle] 10 submitted; sleeping 30 s")
            time.sleep(30)
    if args.submit:
        print(f"\n[done] ids appended to {idfile}")


if __name__ == "__main__":
    main()
