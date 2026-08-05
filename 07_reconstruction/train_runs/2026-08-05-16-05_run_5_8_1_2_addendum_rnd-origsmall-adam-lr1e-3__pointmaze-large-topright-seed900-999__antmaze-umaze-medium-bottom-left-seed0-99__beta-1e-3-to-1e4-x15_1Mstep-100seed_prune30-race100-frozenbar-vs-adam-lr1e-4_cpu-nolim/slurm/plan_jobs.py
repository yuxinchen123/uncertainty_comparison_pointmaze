#!/usr/bin/env python
"""Size the worker jobs of this run from the LIVE cluster state, for the cpu and nolim partitions.

This run is submitted to the cpu and nolim partitions only — no gpu, no gnolim, no reservation (the
user's instruction for this sweep). Both partitions are heterogeneous, so instead of guessing a
fixed job shape this script reads every candidate node and derives its shape from four hard limits:

1. `--ntasks` must not exceed the node's PHYSICAL CORE count (Sockets x CoresPerSocket). sbatch
   rejects a larger ask outright ("Requested node configuration is not available") — hardware
   threads do not count here.
2. `--ntasks` must not exceed the node's FREE ALLOCATABLE threads, CPUEfctv - CPUAlloc. CPUEfctv is
   CPUTot minus the system's reserved core, so a set of jobs summing to CPUTot leaves the last one
   pending on Resources forever.
3. Memory per task must fit: a whole node's worth of 2 GB tasks exceeds RealMemory on the
   high-core-count nodes (puma01: 158 allocatable threads, 252 GB), so the per-task memory is
   min(2 GB, 92% of RealMemory / CPUEfctv) and a node that cannot give a worker at least
   MIN_MEM_MB is skipped rather than run at risk of an out-of-memory kill.
4. The per-user pool cap of the partition (cpu 400, nolim 80), minus what the user already has
   running there, minus HEADROOM threads left free for that user's own interactive jobs.

`--ntasks-per-core=2` is added only on nodes that actually have two threads per core; on a
ThreadsPerCore=1 node it would be rejected.

The same sizing serves the owner and any collaborator: only the worker script, the job name, the id
file and the `--comment` recovery tag differ, and each submitter appends to their OWN id file. Two
guards apply to every submitter: nothing is planned once the SWEEP_COMPLETE sentinel exists, and the
plan is trimmed so worker slots never outnumber the runs still waiting to be claimed.

Usage:
  python plan_jobs.py --sweep_id <id>            # print the plan and the sbatch lines
  python plan_jobs.py --sweep_id <id> --submit   # print AND submit, appending ids to the id file
  python plan_jobs.py --sweep_id <id> --submit --script <collab .slurm> --jobname <prefix> \
                      --idfile <for_collaborator/...>      # the collaborator form
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

# per-user thread cap of each partition's QOS, and the QOS name the cap is read from live
POOLS = {"cpu": ("cspartcpu", 400), "nolim": ("cspartnolim", 80)}
HEADROOM = 16          # threads left unallocated under each cap for the user's own manual jobs
MAX_TASKS = 32         # largest job shape; bigger jobs fit fewer nodes and pend longer
MIN_TASKS = 6          # below this a job is not worth an id
MAX_MEM_MB = 2048      # memory per worker when the node can afford it (worker RSS plateaus ~1.2 GB)
MIN_MEM_MB = 1400      # below this the node is skipped rather than risking an out-of-memory kill
MEM_FRACTION = 0.92    # share of RealMemory we are willing to allocate across a node's threads


def _fields(line):
    """Parse one `scontrol --oneliner` record into a dict of its KEY=VALUE tokens."""
    # before: "NodeName=struct01 CPUAlloc=0 CPUEfctv=26 ... Partitions=cpu ..."
    # after:  {"NodeName": "struct01", "CPUAlloc": "0", "CPUEfctv": "26", ..., "Partitions": "cpu"}
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def reserved_nodes():
    """Every node currently inside ANY reservation, expanded to individual names.

    A reserved node cannot run a normal job (the reservations here carry IGNORE_JOBS,SPEC_NODES), so
    a job placed on one sits pending forever. This run holds no reservation, so every reserved node
    is simply excluded.
    """
    out = subprocess.run(["scontrol", "show", "reservation", "--oneliner"],
                         capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        f = _fields(line)
        spec = f.get("Nodes", "")
        if not spec or spec == "(null)":
            continue
        expanded = subprocess.run(["scontrol", "show", "hostnames", spec],
                                  capture_output=True, text=True).stdout.split()
        names.update(expanded)
    return names


def candidate_nodes(partition, excluded):
    """Usable nodes of one partition, each as (name, free_threads, cores, mem_per_task_mb).

    Skips anything not up and available, anything reserved, and anything too memory-poor per thread.
    """
    out = subprocess.run(["scontrol", "show", "node", "--oneliner"],
                         capture_output=True, text=True).stdout
    nodes = []
    for line in out.splitlines():
        f = _fields(line)
        name = f.get("NodeName")
        if not name or name in excluded:
            continue
        if partition not in f.get("Partitions", "").split(","):
            continue
        state = f.get("State", "")
        if any(bad in state for bad in ("DOWN", "DRAIN", "FAIL", "MAINT", "RESERVED", "UNKNOWN")):
            continue
        efctv, alloc = int(f["CPUEfctv"]), int(f["CPUAlloc"])
        sockets = int(f["Sockets"])
        cores = sockets * int(f["CoresPerSocket"])
        threads_per_core = int(f["ThreadsPerCore"])
        free = efctv - alloc
        mem_per_task = min(MAX_MEM_MB, int(int(f["RealMemory"]) * MEM_FRACTION) // max(efctv, 1))
        if free < MIN_TASKS or mem_per_task < MIN_MEM_MB:
            continue
        nodes.append({"name": name, "free": free, "cores": cores, "mem": mem_per_task,
                      "threads_per_core": threads_per_core, "sockets": sockets})
    # fill the roomiest nodes first so the plan uses few, large jobs before many small ones
    return sorted(nodes, key=lambda n: -n["free"])


def my_cpus_in(partition):
    """CPUs this user already has RUNNING or PENDING in one partition, summed from squeue.

    A pending job counts: it will start and take those CPUs, so a top-up that ignores it submits
    work twice. Returns 0 if squeue fails, which only makes the QOS reading below the binding one.
    """
    out = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-t", "R,PD",
                          "-p", partition, "-o", "%C"], capture_output=True, text=True).stdout
    return sum(int(v) for v in out.split() if v.isdigit())


def pool_room(qos, cap, partition):
    """Threads still available under this user's cap in one partition, minus HEADROOM.

    Usage is the LARGER of two readings, because each can under-report on its own:
    - the partition QOS counter, which also sees jobs submitted outside this run;
    - this user's own running-plus-pending CPUs in the partition, from squeue.
    The QOS counter was observed reading 400(0) on the cpu partition while 384 of this user's CPUs
    were running there (2026-08-05), and a top-up that trusted it alone resubmitted the entire
    fleet a second time. squeue alone would miss usage the QOS counts but squeue cannot see.
    """
    out = subprocess.run(["scontrol", "show", "assoc_mgr", f"qos={qos}", "flags=qos"],
                         capture_output=True, text=True).stdout
    m = re.search(rf"MaxTRESPU=cpu={cap}\((\d+)\)", out)
    qos_used = int(m.group(1)) if m else 0
    used = max(qos_used, my_cpus_in(partition))
    return max(0, cap - used - HEADROOM), used


def partition_time_limit(partition):
    """The partition's MaxTime string, which is this run's --time (no maintenance window is open and
    this run pins no reservation, so the partition limit is the binding cap).

    Hard-fails if the partition reports no limit: this cluster accepts an over-limit --time and then
    holds the job PENDING (PartitionTimeLimit) forever with no warning.
    """
    out = subprocess.run(["scontrol", "show", "partition", partition],
                         capture_output=True, text=True).stdout
    m = re.search(r"MaxTime=(\S+)", out)
    if not m or m.group(1) in ("UNLIMITED", "INFINITE"):
        sys.exit(f"cannot read a finite MaxTime for partition {partition}")
    return m.group(1)


def next_maintenance_start():
    """Unix epoch of the next maintenance reservation start, or None when none is scheduled."""
    out = subprocess.run(["scontrol", "show", "reservation", "--oneliner"],
                         capture_output=True, text=True).stdout
    starts = []
    for line in out.splitlines():
        if "MAINT" not in line:
            continue
        m = re.search(r"StartTime=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
        if m:
            starts.append(time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")))
    future = [s for s in starts if s > time.time()]
    return min(future) if future else None


def walltime_for(partition):
    """The --time to request: the partition limit, capped by the next maintenance window if one
    would start before a full-length job finished."""
    limit = partition_time_limit(partition)
    maint = next_maintenance_start()
    if maint is None:
        return limit, f"partition MaxTime {limit}; no maintenance scheduled"
    # leave 30 minutes of margin before maintenance starts
    hours = int((maint - time.time() - 1800) // 3600)
    if hours < 1:
        sys.exit("maintenance starts within the hour — do not submit now")
    days, rem = divmod(hours, 24)
    capped = f"{days}-{rem:02d}:00:00"
    # the partition limit still binds; take whichever is smaller by comparing day counts crudely
    return (capped if _to_hours(capped) < _to_hours(limit) else limit,
            f"partition MaxTime {limit}; maintenance caps at {capped}")


def _to_hours(t):
    """Hours in a Slurm time string of the form D-HH:MM:SS or HH:MM:SS."""
    days, _, rest = t.partition("-")
    if not rest:
        rest, days = days, "0"
    h, m, s = (rest.split(":") + ["0", "0"])[:3]
    return int(days) * 24 + int(h) + int(m) / 60 + int(s) / 3600


def build_plan():
    """The whole submission plan: a list of job dicts across both partitions."""
    excluded = reserved_nodes()
    plan = []
    for partition, (qos, cap) in POOLS.items():
        room, used = pool_room(qos, cap, partition)
        walltime, why = walltime_for(partition)
        print(f"[{partition}] cap {cap}, in use {used}, headroom {HEADROOM} -> {room} threads to "
              f"fill; --time {walltime} ({why})")
        if excluded:
            print(f"[{partition}] excluded reserved nodes: {' '.join(sorted(excluded))}")
        for node in candidate_nodes(partition, excluded):
            free = node["free"]
            while room >= MIN_TASKS and free >= MIN_TASKS:
                n = min(MAX_TASKS, node["cores"], free, room)
                if n < MIN_TASKS:
                    break
                # Spread the tasks evenly over the node's sockets. Without this, Slurm packs a job
                # onto as few sockets as it fits on: on slurm4 (2 sockets x 12 cores) a 24-task job
                # took ALL of socket 0, so its 24 workers shared one memory controller while an
                # 8-task job on socket 1 had a whole controller for eight. Both showed 100% CPU;
                # the crowded job ran at less than half the throughput (measured 2026-08-05 — this
                # workload is memory-latency-bound, so bandwidth per task, not cycles, sets its
                # speed). ceil(n / sockets) forces the even split.
                per_socket = -(-n // node["sockets"]) if node["sockets"] > 1 else None
                plan.append({"partition": partition, "node": node["name"], "ntasks": n,
                             "mem_mb": n * node["mem"], "walltime": walltime,
                             "ntasks_per_socket": per_socket,
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
            f"--ntasks={job['ntasks']}", "--cpus-per-task=1",
            f"--mem={job['mem_mb']}M", f"--time={job['walltime']}",
            f"--export=ALL,SWEEP_ID={sweep_id}", script]
    if job["ntasks_per_core"]:
        argv.insert(-1, f"--ntasks-per-core={job['ntasks_per_core']}")
    if job.get("ntasks_per_socket"):
        argv.insert(-1, f"--ntasks-per-socket={job['ntasks_per_socket']}")
    return argv


def unclaimed(sweep_id):
    """How many runs are still waiting to be claimed — the ceiling on useful worker slots."""
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending")
    return len(os.listdir(pending)) if os.path.isdir(pending) else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--submit", action="store_true", help="actually submit (default: print only)")
    p.add_argument("--jobname", default="lr1e3", help="base job name (a numbered suffix is added)")
    p.add_argument("--script", default=os.path.join(HERE, "worker_cpu.slurm"),
                   help="the worker batch script to submit (collaborators pass their own)")
    p.add_argument("--idfile", default=None,
                   help="where submitted ids are appended (default: this user's own file in slurm/)")
    args = p.parse_args()
    # a finished sweep must never receive new workers, whoever submits
    if os.path.exists(os.path.join(RUN_DIR, "SWEEP_COMPLETE")):
        sys.exit("SWEEP_COMPLETE exists — this sweep is finished; not planning any jobs")
    plan = build_plan()
    # queue-depth guard: never more worker slots than runs left to claim
    left = unclaimed(args.sweep_id)
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
    total = sum(j["ntasks"] for j in plan)
    print(f"\n[plan] {len(plan)} jobs, {total} worker slots\n")
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
        # Append the id the INSTANT sbatch returns it — the only-cancel-your-own-ids rule. The lock
        # matters because a standing monitor loop can be topping up at the same moment as a manual
        # launch under the same uid, and both append to this one file.
        with open(idfile, "a") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(jid + "\n")
            fh.flush()
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        print(f"  -> {jid}")
        # batch-of-10 etiquette: pause so many workers do not initialize at the same instant
        if i % 10 == 0:
            print("  [throttle] 10 submitted; sleeping 30 s")
            time.sleep(30)
    if args.submit:
        print(f"\n[done] ids appended to {idfile}")


if __name__ == "__main__":
    main()
