"""Submit generic worker jobs of the maze campaign to UVA under both accounts, and the persistent monitor job.

    python submit.py --run-dir RUN --phase PHASE [--gpu-types a,b] [--max-new-cards N] [--dry-run]
    python submit.py --run-dir RUN --monitor [--top-up-phase PHASE] [--complete-when-drained]

A worker job holds no unit: each of its cards runs worker.py, which claims a unit after the job
has started. New worker cards never exceed the pending units that have no worker yet. Jobs are
sent in this order: open partitions (most capable card first) pinned to nodes with free cards,
which start now; then the reservation's nodes; then, for the units still without a worker, jobs
by card type that wait in the scheduler's queue. Every job id is appended at submit time to
RUN/slurm/submitted_jobids_<phase>_<account>.txt by the shared wrapper submit_as.sh.

Standard library only. This file holds no cancel command (see cancel_ledger.py).
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from grid_campaign import common, load
import unit_queue

grid_submit = load("submit")
IDENTITIES = ("uva-sl5nw", "uva-yuxinchen")
# processors per card: 8 where the per-user caps allow it (skill rnd-jax-submission, section 7);
# gnolim's cap is 80 processors for 20 cards
CPUS_PER_CARD = {"gpu": 8, "gnolim": 4}
MEMORY_GIB_PER_CARD = 16
MAX_WAITING_JOBS = 40    # per account; the scheduler tests only 50 waiting jobs of one user per pass


def set_request_per_card(cpus=None, memory_gib=None):
    """Replace the processors and host memory a worker asks for per card, for this process."""
    # measured need of a PointMaze unit of 256 policies: about one busy processor and 2.6 GiB of host
    # memory per card (sacct MaxRSS of jobs 6874063 and 6874078), far below the defaults; a smaller
    # request fits on nodes whose memory is mostly held by other jobs
    global MEMORY_GIB_PER_CARD
    if cpus is not None:
        for partition in CPUS_PER_CARD:
            CPUS_PER_CARD[partition] = cpus
    if memory_gib is not None:
        MEMORY_GIB_PER_CARD = memory_gib
OBSERVED = common.PROJECT / "10_jax_exploration_platform/device/observed"
# the JAX job environment of skill rnd-jax-submission, section 7, for one worker process per card
WORKER_ENVIRONMENT = """# Sourced by every job script of this run folder. Written by scripts/maze_campaign/submit.py.
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"""


def canonical_python():
    """Read from the environment registry which Python runs the JAX platform."""
    # before (one row of .venvs/ENVS.md): | platform_jax | `/p/rlprojects/RND/.venvs/platform_jax` |
    #                                     **canonical for `10_jax_exploration_platform/`** | ... |
    # after:  /p/rlprojects/RND/.venvs/platform_jax/bin/python
    for row in (common.PROJECT / ".venvs/ENVS.md").read_text().splitlines():
        cells = [cell.strip() for cell in row.split("|")]
        if len(cells) > 3 and "canonical for `10_jax_exploration_platform/`" in cells[3]:
            return Path(cells[2].strip("`")) / "bin/python"
    raise RuntimeError("ENVS.md names no canonical environment for 10_jax_exploration_platform/")


def frozen_code(run_dir):
    """The platform folder and the commit of the run's frozen code copy; jobs never run the working tree."""
    path = Path(run_dir) / "source.json"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: commit the code, then run freeze_code.py --run-dir {run_dir}")
    source = common.read_json(path)
    return Path(source["snapshot"]) / "10_jax_exploration_platform", source["commit"]


def ledger(run_dir, phase, account):
    """Name the job-id file of one phase and one account."""
    return Path(run_dir) / "slurm" / f"submitted_jobids_{phase}_{account}.txt"


def worker_job_text(run_dir, platform, python, cards):
    """The bash script of one worker job: one worker.py per card, and a stop signal passed on to each."""
    worker = shlex.join([str(python), str(platform / "scripts/maze_campaign/worker.py"), "--run-dir", str(run_dir),
                         "--observed-dir", str(OBSERVED)])
    return "\n".join([
        "#!/bin/bash", "set -uo pipefail", "umask 0002",
        f"source {shlex.quote(str(run_dir / 'slurm/worker_env.sh'))}",
        f"cd {shlex.quote(str(common.PROJECT))}",
        "# one worker per card of this job; each takes units from the queue until none fits its card",
        'IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"',
        f'[ "${{#devices[@]}}" -eq {cards} ] || {{ echo "this job holds ${{#devices[@]}} cards, not {cards}"; exit 2; }}',
        "pids=()",
        'for device in "${devices[@]}"; do',
        f'  CUDA_VISIBLE_DEVICES="$device" {worker} &',
        '  pids+=("$!")',
        "done",
        "# a stop signal goes on to the workers, which let their units save their state",
        "trap 'kill -TERM \"${pids[@]}\" 2> /dev/null' TERM",
        "status=0",
        'for pid in "${pids[@]}"; do',
        '  wait "$pid"; code=$?',
        "  # the signal interrupts the wait; wait again until that worker has ended",
        '  while [ "$code" -gt 128 ] && kill -0 "$pid" 2> /dev/null; do wait "$pid"; code=$?; done',
        '  [ "$code" -eq 0 ] || status=1',
        "done",
        'exit "$status"', ""])


def monitor_job_text(run_dir, platform, python, options):
    """The bash script of the persistent monitor job."""
    command = shlex.join([str(python), str(platform / "scripts/maze_campaign/monitor.py"), "--run-dir", str(run_dir),
                          "--loop", *options])
    return "\n".join(["#!/bin/bash", "set -uo pipefail", "umask 0002", "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1",
                      f"cd {shlex.quote(str(common.PROJECT))}", f"exec {command}", ""])


def submit_job(run_dir, phase, job, text, dry_run=False):
    """Write one job script, submit it through the account's wrapper, and record the job; returns its id."""
    run_dir = Path(run_dir)
    account = job["identity"].removeprefix("uva-")
    name = f"maze-{job['kind']}_{phase}_{time.time_ns()}"
    seconds = grid_submit.allowed_seconds(job["partition"], job["reservation"])
    # Slurm reads a bare number as minutes, so the limit is spelled as hours:minutes:seconds
    arguments = ["--job-name", name, "--partition", job["partition"], "--nodes", "1", "--ntasks", "1",
                 "--cpus-per-task", str(job["cpus"]), "--mem", f"{job['memory_gib']}G",
                 "--time", f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}",
                 "--output", str(run_dir / "logs" / "%x_%j.log"), "--open-mode", "append", "--requeue"]
    arguments += (["--gres", f"gpu:{job['gpu_type']}:{job['cards']}"] if job["cards"] else ["--gpus-per-node", "0"])
    if job["node"]:
        arguments += ["--nodelist", job["node"]]
    if job["reservation"]:
        arguments += ["--reservation", job["reservation"], "--qos", "csresnolim"]
    script = run_dir / "slurm/scripts" / f"{name}.sh"
    command = [str(common.ACCOUNTS / "submit_as.sh"), job["identity"], "--idfile", str(ledger(run_dir, phase, account)),
               "--dir", str(common.PROJECT), "--", *arguments, str(script)]
    if dry_run:
        print("    would run: " + shlex.join(command))
        return None
    script.parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    script.write_text(text)
    # an empty standard input: the wrapper reaches the other account over ssh, which reads its input
    output = subprocess.check_output(command, text=True, stdin=subprocess.DEVNULL).strip()
    job_id = output.splitlines()[-1] if output else ""
    if not job_id.isdecimal():
        raise ValueError(f"the submission returned no job id: {output!r}")
    common.write_json(run_dir / "slurm/submissions" / f"{job_id}.json", dict(
        job, job_id=job_id, account=account, phase=phase, qos="csresnolim" if job["reservation"] else None,
        source_commit=frozen_code(run_dir)[1], submitted_at=common.now(), script=str(script), wall_seconds=seconds))
    return job_id


def plan_jobs(unit_memories, workers_without_unit, classes, nodes, budgets, reservation_budgets, held_nodes,
              type_room, waiting, allowed_types=None, max_new_cards=None, max_waiting=MAX_WAITING_JOBS):
    """Decide the worker jobs of one submission from the queue and the scheduler's state; changes nothing.

    unit_memories: the least card memory (MB) of every pending unit. workers_without_unit: cards of
    this run's live jobs that hold no unit. classes: the card classes of the resource catalogue.
    nodes: {node: free cards, cpus, memory_gib, reservation}. budgets: {(identity, partition): free
    cards, cpus, memory_gib} within the per-user caps. reservation_budgets: the same per node of the
    owner's reservation. held_nodes: nodes where a job of this run still waits. type_room: cards of
    each card type on nodes outside every reservation that this run does not ask for yet (a job
    that names no reservation cannot start on a reserved node). waiting: {identity: jobs of this
    run that wait}.

    after: [{"identity": "uva-sl5nw", "partition": "gpu", "node": "nekomata01", "gpu_type": "...",
             "cards": 2, "cpus": 16, "memory_gib": 32, "reservation": None, "kind": "worker"}, ...]
    """
    budgets, reservation_budgets = deepcopy(budgets), deepcopy(reservation_budgets)
    type_room, waiting, nodes = dict(type_room), dict(waiting), deepcopy(nodes)
    # never more worker cards than pending units that have no worker yet
    remaining = len(unit_memories) - workers_without_unit
    if max_new_cards is not None:
        remaining = min(remaining, max_new_cards)
    jobs = []

    def usable(record):
        """A card class takes part when its type is allowed and its card can hold a pending unit."""
        return ((allowed_types is None or record["gres_type"] in allowed_types)
                and any(memory <= record["gpu_mem_mb"] for memory in unit_memories))

    def request(partition, cards):
        """Processors and host memory of a job with this many cards."""
        return CPUS_PER_CARD[partition] * cards, MEMORY_GIB_PER_CARD * cards

    # the open partitions before the reservation, and the most capable card first; a job is pinned
    # only to a node that has free cards now, so no pinned job waits
    ordered = sorted(classes, key=lambda record: (record["partition"] != "gpu", record["capability_rank"]))
    for reserved in (False, True):
        for record in ordered:
            if not usable(record):
                continue
            for node in record["nodes"]:
                if remaining <= 0:
                    return jobs
                if node not in nodes or node in held_nodes or (node in reservation_budgets) != reserved:
                    continue
                physical = nodes[node]
                # a node reserved for someone else takes none of our jobs
                if not reserved and physical["reservation"] not in (None, "(null)"):
                    continue
                cpus_per_card, memory_per_card = request(record["partition"], 1)
                identities = IDENTITIES[:1] if reserved else sorted(
                    IDENTITIES, key=lambda identity: -budgets.get((identity, record["partition"]), {}).get("cards", 0))
                for identity in identities:
                    budget = reservation_budgets[node] if reserved else budgets.get((identity, record["partition"]))
                    if not budget:
                        continue
                    cards = int(min(physical["cards"], physical["cpus"] // cpus_per_card,
                                    physical["memory_gib"] // memory_per_card, budget["cards"],
                                    budget["cpus"] // cpus_per_card, budget["memory_gib"] // memory_per_card,
                                    remaining))
                    # outside the reservation a card type is asked for at most as often as it has open cards
                    if not reserved:
                        cards = min(cards, type_room.get(record["gres_type"], 0))
                    if cards < 1:
                        continue
                    cpus, memory = request(record["partition"], cards)
                    jobs.append({"identity": identity, "partition": record["partition"], "node": node,
                                 "gpu_type": record["gres_type"], "cards": cards, "cpus": cpus, "memory_gib": memory,
                                 "reservation": reservation_budgets[node]["name"] if reserved else None,
                                 "kind": "worker"})
                    remaining -= cards
                    if not reserved:
                        type_room[record["gres_type"]] -= cards
                    for resource, amount in (("cards", cards), ("cpus", cpus), ("memory_gib", memory)):
                        budget[resource] -= amount
                        physical[resource] -= amount
                    break

    # the units still without a worker get jobs that wait in the scheduler's queue: one card each,
    # asked by card type and not pinned, the most capable type first, shared between the accounts
    for gpu_type, partition in dict.fromkeys((record["gres_type"], record["partition"])
                                             for record in ordered if usable(record)):
        added = True
        while added and remaining > 0:
            added = False
            for identity in IDENTITIES:
                if (remaining <= 0 or type_room.get(gpu_type, 0) <= 0 or (identity, partition) not in budgets
                        or waiting.get(identity, 0) >= max_waiting):
                    continue
                cpus, memory = request(partition, 1)
                jobs.append({"identity": identity, "partition": partition, "node": None, "gpu_type": gpu_type,
                             "cards": 1, "cpus": cpus, "memory_gib": memory, "reservation": None, "kind": "worker"})
                remaining -= 1
                type_room[gpu_type] -= 1
                waiting[identity] = waiting.get(identity, 0) + 1
                added = True
    return jobs


def scheduler_state(run_dir):
    """Read what a submission plan needs: this run's live jobs, the accounts' free room, and the nodes."""
    live = common.own_live_jobs(run_dir)
    submissions = [common.read_json(path) for path in (Path(run_dir) / "slurm/submissions").glob("*.json")]
    active = [record for record in submissions if record["job_id"] in live and record["kind"] == "worker"]
    # room within the per-user caps, with this run's waiting jobs already counted against it
    capacity = json.loads(subprocess.check_output(
        [sys.executable, str(common.ACCOUNTS / "capacity.py"), "--no-mit", "--json"], text=True))
    common.write_json(Path(run_dir) / "slurm/capacity_latest.json", capacity)
    budgets = grid_submit.capacity_budgets(capacity, submissions, live)
    nodes = grid_submit.node_resources(subprocess.check_output(["scontrol", "show", "nodes", "-o"], text=True))
    classes = common.read_json(grid_submit.RESOURCE_FILE)["classes"]
    # the owner's reservation is read live; its nodes are planned last and from their own free resources
    reservation = grid_submit.owner_reservation()
    reservation_budgets = {
        node: dict(grid_submit.reservation_budget(node, physical, submissions, live, reservation), name=reservation)
        for node, physical in nodes.items() if reservation and physical["reservation"] == reservation}
    # cards of each type on usable nodes outside every reservation, less the cards this run
    # already asks for there; a job that names no reservation cannot start on a reserved node
    type_room = {}
    for record in classes:
        open_nodes = [node for node in record["nodes"]
                      if node in nodes and nodes[node]["reservation"] in (None, "(null)")]
        type_room[record["gres_type"]] = (type_room.get(record["gres_type"], 0)
                                          + record["gpus_per_node"] * len(open_nodes))
    for record in active:
        if not record["reservation"]:
            type_room[record["gpu_type"]] -= record["cards"]
    waiting_jobs = [record for record in active if live[record["job_id"]][3] != "RUNNING"]
    waiting = {identity: sum(record["identity"] == identity for record in waiting_jobs) for identity in IDENTITIES}
    held_nodes = {record["node"] for record in waiting_jobs if record["node"]}
    return {"active_cards": sum(record["cards"] for record in active),
            "waiting_cards": sum(record["cards"] for record in waiting_jobs), "classes": classes, "nodes": nodes,
            "budgets": budgets, "reservation_budgets": reservation_budgets, "held_nodes": held_nodes,
            "type_room": type_room, "waiting": waiting}


def top_up(run_dir, phase, gpu_types=None, max_new_cards=None, dry_run=False):
    """Submit worker jobs for the pending units that have no worker yet; returns the new job ids."""
    run_dir = Path(run_dir)
    if (run_dir / "STOP").exists() or (run_dir / "SWEEP_COMPLETE").exists():
        print("RUN/STOP or RUN/SWEEP_COMPLETE exists: nothing is submitted")
        return []
    platform, _ = frozen_code(run_dir)
    pending = unit_queue.read_units(run_dir, "pending")
    running = unit_queue.counts(run_dir)["running"]
    state = scheduler_state(run_dir)
    # worker cards of this run that hold no unit: the cards of its waiting jobs. A running job's card
    # whose worker found the queue empty has ended its worker (the job goes on for its other cards), so
    # running cards beyond the running units are not workers any more and are not counted
    # (2026-11-12: three new units waited 16 minutes behind two such cards of a four-card job).
    spare = state["waiting_cards"]
    jobs = plan_jobs([unit["minimum_device_memory_mb"] for unit in pending], spare,
                     state["classes"], state["nodes"], state["budgets"], state["reservation_budgets"],
                     state["held_nodes"], state["type_room"], state["waiting"], gpu_types, max_new_cards)
    # the depth of the submission, said before anything is sent
    pinned = sum(job["cards"] for job in jobs if job["node"])
    print(f"{len(pending)} units pending, {running} running; {spare} worker cards of this "
          f"run hold no unit yet; {len(jobs)} new jobs: {pinned} cards on nodes with free cards now, "
          f"{sum(job['cards'] for job in jobs) - pinned} cards that wait in the scheduler's queue")
    for identity in IDENTITIES:
        print(f"    {identity}: {sum(job['cards'] for job in jobs if job['identity'] == identity)} cards in "
              f"{sum(job['identity'] == identity for job in jobs)} jobs")
    if not dry_run and jobs:
        (run_dir / "slurm").mkdir(exist_ok=True)
        (run_dir / "slurm/worker_env.sh").write_text(WORKER_ENVIRONMENT)
    python, job_ids = canonical_python(), []
    for index, job in enumerate(jobs):
        # at most ten submissions, then a pause of 30 seconds, for the scheduler's request rate
        if index and index % 10 == 0 and not dry_run:
            time.sleep(30)
        print(f"  {job['identity']} {job['partition']} {job['node'] or 'any node'} {job['cards']} x {job['gpu_type']}"
              + (f" reservation {job['reservation']}" if job["reservation"] else ""))
        job_ids.append(submit_job(run_dir, phase, job, worker_job_text(run_dir, platform, python, job["cards"]), dry_run))
    # every job that was sent must be in its account's ledger before this call reports success
    for job, job_id in zip(jobs, job_ids):
        account = job["identity"].removeprefix("uva-")
        if not dry_run and job_id not in ledger(run_dir, phase, account).read_text().split():
            raise RuntimeError(f"job {job_id} is missing from {ledger(run_dir, phase, account)}")
    return job_ids


def submit_monitor(run_dir, options, dry_run=False):
    """Submit the persistent monitor: one processor on a long open partition, no card."""
    platform, _ = frozen_code(run_dir)
    # one monitor per run folder: a live one is named and no second one is sent
    live = common.own_live_jobs(run_dir)
    for path in (Path(run_dir) / "slurm/submissions").glob("*.json"):
        record = common.read_json(path)
        if record["kind"] == "monitor" and record["job_id"] in live:
            print(f"monitor job {record['job_id']} of this run folder is still live; no second one is submitted")
            return record["job_id"]
    job = {"identity": IDENTITIES[0], "partition": "nolim", "node": None, "gpu_type": None, "cards": 0, "cpus": 1,
           "memory_gib": 16, "reservation": None, "kind": "monitor"}
    return submit_job(run_dir, "monitor", job, monitor_job_text(Path(run_dir), platform, canonical_python(), options),
                      dry_run)


def main(argv=None):
    """Submit worker jobs for one phase, or the persistent monitor job."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", help="the phase the job ids are recorded under (words joined by hyphens)")
    parser.add_argument("--gpu-types", help="comma-separated card types (gres names) the workers may ask for; "
                                            "default: every type whose card can hold a pending unit")
    parser.add_argument("--max-new-cards", type=int, help="send at most this many worker cards in this call")
    parser.add_argument("--cpus-per-card", type=int,
                        help="processors a worker asks for per card (default 8 in gpu, 4 in gnolim)")
    parser.add_argument("--memory-gib-per-card", type=int,
                        help="host memory in GiB a worker asks for per card (default 16); size it from the "
                             "measured MaxRSS of finished units of the same kind, plus a margin")
    parser.add_argument("--monitor", action="store_true", help="submit the persistent monitor job instead")
    parser.add_argument("--top-up-phase", help="with --monitor: every pass also submits workers under this phase")
    parser.add_argument("--complete-when-drained", action="store_true",
                        help="with --monitor: write RUN/SWEEP_COMPLETE when no unit is pending, running or held")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the commands; submit nothing")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    set_request_per_card(args.cpus_per_card, args.memory_gib_per_card)
    if args.monitor:
        options = (["--top-up-phase", args.top_up_phase] if args.top_up_phase else []) + (
            ["--complete-when-drained"] if args.complete_when_drained else [])
        print(submit_monitor(run_dir, options, args.dry_run))
    elif args.phase:
        print(top_up(run_dir, args.phase, args.gpu_types.split(",") if args.gpu_types else None,
                     args.max_new_cards, args.dry_run))
    else:
        parser.error("give --phase to submit workers, or --monitor")


if __name__ == "__main__":
    main()
