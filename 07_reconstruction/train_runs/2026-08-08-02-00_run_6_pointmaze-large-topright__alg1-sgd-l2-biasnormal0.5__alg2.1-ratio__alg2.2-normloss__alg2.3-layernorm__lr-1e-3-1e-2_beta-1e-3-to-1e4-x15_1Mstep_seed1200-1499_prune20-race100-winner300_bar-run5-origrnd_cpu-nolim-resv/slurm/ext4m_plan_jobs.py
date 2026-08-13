#!/usr/bin/env python
"""Size and submit the worker jobs of the ext4m sweep: cpu -> nolim -> the owner's reservation on
puma01 ONLY (no jaguar03) — the user's submission order for this sweep.

The cpu/nolim sizing is the run-6 plan_jobs.py logic unchanged (per-node shapes from live scontrol
numbers, the two-reading pool-room guard, the socket spread, the queue-depth guard). New here:

- a RESERVATION bucket after the open pools (owner only): puma01 at the 2-CPUs-per-worker shape —
  `--ntasks=<free//2> --cpus-per-task=2 --ntasks-per-core=2` with 3000 MB per worker — per the
  2026-08-13 addition to the shared uva-submit-cpu-sweep skill (1-thread packing ran puma01 at
  half speed and broke the walltime guard; one full core per worker recovers it). jaguar03 is
  excluded by the user's instruction. The reservation name is discovered live, never hardcoded.
- sentinel SWEEP4M_COMPLETE stops all planning.

Usage:
  python ext4m_plan_jobs.py --sweep_id <id>            # print the plan
  python ext4m_plan_jobs.py --sweep_id <id> --submit   # print AND submit, appending ids
  python ext4m_plan_jobs.py --sweep_id <id> --submit --script <collab .slurm> --jobname <prefix> \
                            --idfile <for_collaborator/...>   # collaborator form (open pools only)
"""
import argparse
import fcntl
import getpass
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# the run-6 sizing machinery is reused as a library; only the buckets and defaults differ
from plan_jobs import (POOLS, HEADROOM, MIN_TASKS, candidate_nodes, pool_room,  # noqa: E402
                       reserved_nodes, walltime_for, partition_time_limit,
                       next_maintenance_start, _to_hours, _fields)

RESERVATION_NODE = "puma01"       # the ONLY reserved node this sweep uses (no jaguar03)
RESERVATION_CPUS_PER_TASK = 2     # one full physical core per worker (skill rule 2026-08-13)
RESERVATION_MEM_PER_TASK_MB = 3000
SENTINEL = os.path.join(RUN_DIR, "SWEEP4M_COMPLETE")


def my_reservation():
    """(name, end_epoch) of this user's active reservation covering puma01, or (None, None)."""
    out = subprocess.run(["scontrol", "show", "reservation", "--oneliner"],
                         capture_output=True, text=True).stdout
    user = getpass.getuser()
    for line in out.splitlines():
        f = _fields(line)
        if user not in f.get("Users", ""):
            continue
        nodes = subprocess.run(["scontrol", "show", "hostnames", f.get("Nodes", "")],
                               capture_output=True, text=True).stdout.split()
        if RESERVATION_NODE not in nodes:
            continue
        end = time.mktime(time.strptime(f["EndTime"], "%Y-%m-%dT%H:%M:%S"))
        return f["ReservationName"], end
    return None, None


def reservation_job(sweep_id):
    """The one puma01 reservation job, sized from the node's live free threads, or None."""
    name, end_epoch = my_reservation()
    if name is None:
        print("[reservation] no active reservation covering puma01 for this user; skipping")
        return None
    out = subprocess.run(["scontrol", "show", "node", RESERVATION_NODE, "--oneliner"],
                         capture_output=True, text=True).stdout
    f = _fields(out.splitlines()[0])
    free = int(f["CPUEfctv"]) - int(f["CPUAlloc"])
    ntasks = free // RESERVATION_CPUS_PER_TASK
    if ntasks < MIN_TASKS:
        print(f"[reservation] puma01 has only {free} free threads; skipping")
        return None
    # walltime: the cpu partition limit, capped by reservation end and any maintenance window
    limit = partition_time_limit("cpu")
    hours = _to_hours(limit)
    res_hours = (end_epoch - time.time() - 1800) / 3600
    maint = next_maintenance_start()
    if maint is not None:
        res_hours = min(res_hours, (maint - time.time() - 1800) / 3600)
    if res_hours < 1:
        print("[reservation] under an hour of reservation left; skipping")
        return None
    hours = min(hours, int(res_hours))
    days, rem = divmod(int(hours), 24)
    walltime = f"{days}-{rem:02d}:00:00"
    print(f"[reservation] {name} on {RESERVATION_NODE}: {free} free threads -> {ntasks} workers "
          f"x {RESERVATION_CPUS_PER_TASK} cpus; --time {walltime}")
    return {"partition": "cpu", "node": RESERVATION_NODE, "ntasks": ntasks,
            "cpus_per_task": RESERVATION_CPUS_PER_TASK,
            "mem_mb": ntasks * RESERVATION_MEM_PER_TASK_MB, "walltime": walltime,
            "ntasks_per_socket": None, "ntasks_per_core": 2, "reservation": name}


def open_pool_plan():
    """The cpu + nolim plan (run-6 logic), excluding every reserved node."""
    excluded = reserved_nodes()
    plan = []
    for partition, (qos, cap) in POOLS.items():
        room, used = pool_room(qos, cap, partition)
        walltime, why = walltime_for(partition)
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
                             "cpus_per_task": 1, "mem_mb": n * node["mem"], "walltime": walltime,
                             "ntasks_per_socket": per_socket, "reservation": None,
                             "ntasks_per_core": 2 if node["threads_per_core"] >= 2 else None})
                free -= n
                room -= n
            if room < MIN_TASKS:
                break
    return plan


def sbatch_argv(job, sweep_id, jobname, script, comment):
    """The full sbatch command line for one planned job."""
    argv = ["sbatch", "--parsable", f"--job-name={jobname}", f"--comment={comment}",
            f"--partition={job['partition']}", f"--nodelist={job['node']}", "--nodes=1",
            f"--ntasks={job['ntasks']}", f"--cpus-per-task={job['cpus_per_task']}",
            f"--mem={job['mem_mb']}M", f"--time={job['walltime']}",
            f"--export=ALL,SWEEP_ID={sweep_id}", script]
    if job["ntasks_per_core"]:
        argv.insert(-1, f"--ntasks-per-core={job['ntasks_per_core']}")
    if job.get("ntasks_per_socket"):
        argv.insert(-1, f"--ntasks-per-socket={job['ntasks_per_socket']}")
    if job.get("reservation"):
        argv.insert(-1, f"--reservation={job['reservation']}")
        argv.insert(-1, "--qos=csresnolim")
    return argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--jobname", default="r6e4m")
    p.add_argument("--script", default=os.path.join(HERE, "ext4m_worker_cpu.slurm"))
    p.add_argument("--idfile", default=None)
    args = p.parse_args()
    if os.path.exists(SENTINEL):
        sys.exit("SWEEP4M_COMPLETE exists — this sweep is finished; not planning any jobs")
    plan = open_pool_plan()
    # the reservation bucket comes LAST (open pools first per the capacity-ordering rule) and only
    # for the owner — a collaborator has no access to the reservation
    if getpass.getuser() == "sl5nw":
        res = reservation_job(args.sweep_id)
        if res is not None:
            plan.append(res)
    # queue-depth guard: never more worker slots than runs left to claim
    pending = os.path.join(RUN_DIR, "queue", args.sweep_id, "pending")
    left = len(os.listdir(pending)) if os.path.isdir(pending) else 0
    trimmed, slots = [], 0
    for job in plan:
        if slots >= left:
            break
        trimmed.append(job)
        slots += job["ntasks"]
    if len(trimmed) < len(plan):
        print(f"[guard] {left} runs left to claim: dropping {len(plan) - len(trimmed)} of "
              f"{len(plan)} planned jobs so workers never outnumber the work")
    plan = trimmed
    print(f"\n[plan] {len(plan)} jobs, {sum(j['ntasks'] for j in plan)} worker slots\n")
    comment = f"{args.sweep_id}_{getpass.getuser()}"
    idfile = args.idfile or os.path.join(
        HERE, f"submitted_jobids_{args.sweep_id}_{getpass.getuser()}.txt")
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
        # append the id the INSTANT sbatch returns it (only-cancel-your-own-ids rule)
        with open(idfile, "a") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(jid + "\n")
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
