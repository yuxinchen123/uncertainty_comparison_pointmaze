"""Submit generic worker jobs of the pixel sweep to UVA under both accounts, and the persistent monitor job.

    python submit.py --run-dir RUN --phase PHASE [--only-cards-free-now] [--leave-cards N] [--gpu-types a,b]
                     [--max-new-cards N] [--accounts uva-sl5nw] [--worker-hours H] [--cancel-waiting-after S]
                     [--dry-run]
    python submit.py --run-dir RUN --monitor [--top-up-phase PHASE] [--complete-when-drained] [the options above]

A worker job holds no unit: each of its cards runs worker.py, which claims a unit after the job has
started. The plan is the maze campaign's (scripts/maze_campaign/submit.py, imported): nodes with
free cards in the open partitions first (these start now), the owner's reservation after them, and
last, unless --only-cards-free-now is given, jobs by card type that wait in the scheduler's queue.
What this file adds:

  - Only card types of an allow-list are asked for (default: RTX 2080 Ti, RTX A4000, RTX A4500, A40,
    RTX 5080, A100, H100 NVL; --gpu-types replaces it). The RTX 4000 Ada is left out: gate 6 measured
    3.039 s per update on it against the RTX A4000's 2.038 s, the plan's bar.
  - The plan is made once per execution profile: cards under 60 GB for the pending single units,
    larger cards for the pending four-copy units, and never more worker cards of a profile than
    pending units of that profile that have no worker yet.
  - --leave-cards N keeps N cards that are free now out of the plan, for another study's waiting
    units (the most capable allowed types first, or the types of --leave-card-types).
  - Each card asks for 4 processors and 16 GiB of host memory (the archived baseline used about
    6 GB per one-copy run; --memory-gib-per-card changes the memory), and every job is requeued by
    the scheduler when it is preempted.
  - --cancel-waiting-after S: a job of this call that still waits in the scheduler's queue S seconds
    after the last submission is cancelled by its id (cancel_ledger.py --job-id); its units stay
    pending for a later call. No worker job is then left waiting in UVA's queue.

Every job id is appended at submit time to RUN/slurm/submitted_jobids_<phase>_<account>.txt by the
shared wrapper submit_as.sh (sl5nw in this shell, yuxinchen over ssh). Standard library only. This
file holds no cancel command: the jobs of --cancel-waiting-after are cancelled through
scripts/maze_campaign/cancel_ledger.py, the only cancel path.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from campaign_imports import MAZE, common, grid, maze, unit_queue
import pixel_queue

maze_submit = maze("submit")
grid_submit = grid("submit")
# the card types the user allows for this sweep (plan, decision 7): every UVA type equal to or better than
# the RTX A4000 and the RTX 2080 Ti; GTX 1080 Ti, GTX 1080, Quadro RTX 4000, Titan, P100 and A16 are left out.
# The RTX 4000 Ada is left out too: below the RTX A4000's speed in gate 6 (3.039 against 2.038 s per update)
DEFAULT_CARD_TYPES = ("nvidia_geforce_rtx_2080_ti", "nvidia_rtx_a4000", "nvidia_rtx_a4500", "nvidia_a40", "a40",
                      "nvidia_geforce_rtx_5080", "a100", "nvidia_a100-pcie-40gb", "nvidia_h100_nvl")
CPUS_PER_CARD = 4
MEMORY_GIB_PER_CARD = 16       # `set_memory_per_card` replaces it for one process
PIXEL_PYTHON = common.PROJECT / ".venvs/atari_pixels_jax/bin/python"
ASSETS = common.PROJECT / "reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png"
# the job environment of the implementation design ("Environment for every job"), as the archived baseline's units set it
WORKER_ENVIRONMENT = f"""# Sourced by every worker job of this run folder. Written by scripts/pixel_sweep_campaign/submit.py.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export XDG_DATA_HOME={ASSETS}
export JAXTARI_SKIP_SPRITE_UPDATE=1
"""


def set_memory_per_card(gib=None):
    """Replace the host memory a worker asks for per card, for this process."""
    # measured 2026-11-18: one-copy jobs held 5.0 to 6.4 GiB (sacct MaxRSS of gate jobs 6955816 to 6956643, the
    # gate-7 unit's worker and trainer together 6.35 GiB); 16 GiB per card fits only one card on a node with
    # 30 GiB unallocated
    # before: None ; after: 16 GiB as always      before: 10 ; after: every card asks for 10 GiB
    global MEMORY_GIB_PER_CARD
    if gib is None:
        return
    if gib <= 0:
        raise ValueError(f"the host memory per card must be positive; it is {gib} GiB")
    MEMORY_GIB_PER_CARD = gib


def worker_job_text(run_dir, platform, python, cards):
    """The bash script of one worker job: one worker.py per card, and a stop signal passed on to each."""
    # copied from scripts/maze_campaign/submit.py, worker_job_text, with this folder's worker
    worker = shlex.join([str(python), str(platform / "scripts/pixel_sweep_campaign/worker.py"), "--run-dir", str(run_dir)])
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
        "# a stop signal goes on to the workers, which pass it to their trainers; a trainer then saves and ends",
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
    command = shlex.join([str(python), str(platform / "scripts/pixel_sweep_campaign/monitor.py"), "--run-dir",
                          str(run_dir), "--loop", *options])
    return "\n".join(["#!/bin/bash", "set -uo pipefail", "umask 0002", "export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1",
                      f"cd {shlex.quote(str(common.PROJECT))}", f"exec {command}", ""])


def submit_job(run_dir, phase, job, text, dry_run=False):
    """Write one job script, submit it through the account's wrapper, and record the job; returns its id."""
    # copied from scripts/maze_campaign/submit.py, submit_job: the job's name says it belongs to the pixel sweep
    run_dir = Path(run_dir)
    account = job["identity"].removeprefix("uva-")
    name = f"pixel-{job['kind']}_{phase}_{time.time_ns()}"
    seconds = maze_submit.wall_seconds(job["kind"], grid_submit.allowed_seconds(job["partition"], job["reservation"]))
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
    command = [str(common.ACCOUNTS / "submit_as.sh"), job["identity"], "--idfile",
               str(maze_submit.ledger(run_dir, phase, account)), "--dir", str(common.PROJECT), "--", *arguments, str(script)]
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
        source_commit=maze_submit.frozen_code(run_dir)[1], submitted_at=common.now(), script=str(script),
        wall_seconds=seconds))
    return job_id


def profile_of_type(classes, profiles, gpu_type):
    """Name the execution profile of a card type, by the card memory the resource catalogue records for it."""
    # before: "nvidia_rtx_a4500" (20,470 MB) ; after: "one copy"     before: "nvidia_h100_nvl" ; after: "four copies"
    memory = next((record["gpu_mem_mb"] for record in classes if record["gres_type"] == gpu_type), None)
    return None if memory is None else pixel_queue.profile_for_card(profiles, memory)


def units_of_profile(pending, name, profile):
    """The pending units a worker of one profile may take: its number of copies, and no other profile fixed."""
    return [unit for unit in pending
            if unit["copies"] == profile["copies"] and unit.get("execution_profile") in (None, name)]


def leave_free_cards(state, classes, count, types=None):
    """Take cards that are free now out of a plan's reach, the most capable card types first; returns what was left.

    before: nodes {"serval06": {"cards": 1, ...}, "nekomata01": {"cards": 2, ...}}, type room
            {"nvidia_h100_nvl": 1, "nvidia_geforce_rtx_5080": 4}, count 2
    after:  nodes {"serval06": {"cards": 0, ...}, "nekomata01": {"cards": 1, ...}}, type room
            {"nvidia_h100_nvl": 0, "nvidia_geforce_rtx_5080": 3}; returns [("serval06", "nvidia_h100_nvl", 1),
            ("nekomata01", "nvidia_geforce_rtx_5080", 1)]
    `state` holds the planner's inputs (nodes, type_room, reservation_budgets) and is changed in place.
    """
    left = []
    for record in sorted(classes, key=lambda record: record["capability_rank"]):
        if types is not None and record["gres_type"] not in types:
            continue
        for node in record["nodes"]:
            physical = state["nodes"].get(node)
            reserved = node in state["reservation_budgets"]
            # only nodes this run could plan on: open ones and the owner's reservation
            if count <= 0 or physical is None or (not reserved and physical["reservation"] not in (None, "(null)")):
                continue
            cards = min(count, physical["cards"])
            if cards < 1:
                continue
            physical["cards"] -= cards
            if reserved:
                state["reservation_budgets"][node]["cards"] = max(0, state["reservation_budgets"][node]["cards"] - cards)
            else:
                state["type_room"][record["gres_type"]] = max(0, state["type_room"].get(record["gres_type"], 0) - cards)
            left.append((node, record["gres_type"], cards))
            count -= cards
    return left


def take_out(state, jobs):
    """Take the resources of planned jobs out of the planner's inputs, so a second plan cannot use them again."""
    # before: budgets {("uva-sl5nw", "gpu"): {"cards": 40, "cpus": 384, "memory_gib": 1000}}, one planned job of
    #         2 cards, 8 processors and 32 GiB on nekomata01 under uva-sl5nw
    # after:  budgets {("uva-sl5nw", "gpu"): {"cards": 38, "cpus": 376, "memory_gib": 968}}, and nekomata01's
    #         free cards, processors and memory lower by the same amounts
    for job in jobs:
        amounts = (("cards", job["cards"]), ("cpus", job["cpus"]), ("memory_gib", job["memory_gib"]))
        if job["node"] is None:
            # a job that waits in the scheduler's queue holds one place of its card type and of its account
            state["type_room"][job["gpu_type"]] -= job["cards"]
            state["waiting"][job["identity"]] = state["waiting"].get(job["identity"], 0) + 1
            continue
        budget = (state["reservation_budgets"][job["node"]] if job["reservation"]
                  else state["budgets"][(job["identity"], job["partition"])])
        for resource, amount in amounts:
            budget[resource] -= amount
            state["nodes"][job["node"]][resource] -= amount
        if not job["reservation"]:
            state["type_room"][job["gpu_type"]] -= job["cards"]


def plan_profiles(pending, profiles, waiting_cards, state, allowed_types, max_new_cards=None, max_waiting=None):
    """Plan the worker jobs of one submission, one execution profile after the other; changes nothing outside.

    pending: the pending units' queue records. waiting_cards: {profile name: cards of this run's jobs
    that wait and hold no unit}. state: the maze planner's inputs (classes, nodes, budgets,
    reservation_budgets, held_nodes, type_room, waiting). Returns the planned jobs, each with the
    profile its cards belong to.
    """
    state = deepcopy(state)
    jobs = []
    for name, profile in profiles.items():
        units = units_of_profile(pending, name, profile)
        # only the card classes of this profile's memory range take part in its plan
        classes = [record for record in state["classes"] if pixel_queue.holds_card(profile, record["gpu_mem_mb"])]
        memories = [max(unit["minimum_device_memory_mb"], profile["minimum_card_memory_mb"]) for unit in units]
        room = None if max_new_cards is None else max_new_cards - sum(job["cards"] for job in jobs)
        planned = maze_submit.plan_jobs(
            memories, waiting_cards.get(name, 0), classes, state["nodes"], state["budgets"],
            state["reservation_budgets"], state["held_nodes"], state["type_room"], state["waiting"],
            allowed_types, room, maze_submit.MAX_WAITING_JOBS if max_waiting is None else max_waiting)
        take_out(state, planned)
        jobs += [dict(job, execution_profile=name) for job in planned]
    return jobs


def waiting_cards_by_profile(run_dir, profiles, classes, live):
    """Count, per profile, the cards of this run's worker jobs that still wait in the scheduler's queue."""
    # after: {"one copy": 6, "four copies": 0}
    counts = {name: 0 for name in profiles}
    for path in (Path(run_dir) / "slurm/submissions").glob("*.json"):
        record = common.read_json(path)
        job = live.get(record["job_id"])
        if record["kind"] == "worker" and job is not None and job[3] != "RUNNING":
            name = record.get("execution_profile") or profile_of_type(classes, profiles, record["gpu_type"])
            if name in counts:
                counts[name] += record["cards"]
    return counts


def cancel_still_waiting(run_dir, phase, sent, wait_seconds, sleep=time.sleep):
    """Wait, then cancel by id every job of `sent` that still waits in the scheduler's queue; returns their ids.

    before: sent [(job under uva-sl5nw, "6957001"), (job under uva-yuxinchen, "6957002")], wait 120 s; after 120 s
            6957001 runs and 6957002 waits on Priority
    after:  6957002 cancelled through cancel_ledger.py with RUN/slurm/submitted_jobids_<phase>_yuxinchen.txt;
            returns ["6957002"]
    """
    sleep(wait_seconds)
    live = common.own_live_jobs(run_dir)
    cancelled = []
    for job, job_id in sent:
        if job_id not in live or live[job_id][3] != "PENDING":
            continue
        # the only cancel path: one id of this run's own ledger, checked again by the account's wrapper
        ledger = maze_submit.ledger(run_dir, phase, job["identity"].removeprefix("uva-"))
        subprocess.run([sys.executable, str(MAZE / "cancel_ledger.py"), "--ledger", str(ledger), "--job-id", job_id],
                       check=True, stdin=subprocess.DEVNULL)
        print(f"  cancelled {job_id} ({job['identity']}, {job['node'] or 'any node'}, {job['gpu_type']}): it still "
              f"waited after {wait_seconds} s ({live[job_id][5]}); its units stay pending")
        cancelled.append(job_id)
    return cancelled


# The card names a worker records ("card" in its queue history) and the scheduler's card types (gres names) they are
SPEED_FILE = "infrastructure/card_seconds_per_update.json"
CARD_TYPES_OF_DEVICE = {"NVIDIA H100 NVL": ("nvidia_h100_nvl",), "NVIDIA A100 80GB PCIe": ("a100",),
                        "NVIDIA A100-PCIE-40GB": ("nvidia_a100-pcie-40gb",), "NVIDIA A40": ("nvidia_a40", "a40"),
                        "NVIDIA RTX A4500": ("nvidia_rtx_a4500",), "Quadro RTX 6000": ("quadro_rtx_6000",),
                        "NVIDIA GeForce RTX 2080 Ti": ("nvidia_geforce_rtx_2080_ti",),
                        "NVIDIA RTX A4000": ("nvidia_rtx_a4000",),
                        "NVIDIA RTX 4000 Ada Generation": ("nvidia_rtx_4000_ada_generation",),
                        "NVIDIA GeForce RTX 5080": ("nvidia_geforce_rtx_5080",)}


def measured_card_speeds(run_dir, profile="one copy"):
    """Median seconds per update of each UVA card type over a run's finished units of one execution profile.

    Read from every attempt in queue/done that recorded its card and its speed; cards of other servers (A30, H100
    on Gautschi) have no UVA card type and are left out.
    before: done units whose attempts ran on "NVIDIA RTX A4500" at 1.55 and 1.60 s and on "NVIDIA A40" at 1.38 s ;
    after:  {"nvidia_rtx_a4500": 1.575, "nvidia_a40": 1.38, "a40": 1.38}
    """
    seconds = {}
    # every attempt of every finished unit of the profile, by the card it ran on
    for path in sorted((Path(run_dir) / "queue/done").glob("*.json")):
        record = common.read_json(path)
        if record.get("execution_profile") != profile:
            continue
        for attempt in record.get("history") or []:
            speed = attempt.get("seconds_per_update")
            if speed is None and attempt is record["history"][-1]:
                speed = (record.get("result") or {}).get("seconds_per_update")
            for card_type in CARD_TYPES_OF_DEVICE.get(attempt.get("card"), ()) if speed else ():
                seconds.setdefault(card_type, []).append(speed)
    return {card_type: statistics.median(values) for card_type, values in sorted(seconds.items())}


def read_card_speeds(run_dir):
    """The run's table of measured seconds per update by card type, or an empty one when the run has none."""
    path = Path(run_dir) / SPEED_FILE
    return common.read_json(path)["seconds_per_update"] if path.exists() else {}


def order_by_measured_speed(classes, speeds):
    """Card classes ranked by this work's measured seconds per update, fastest first; unmeasured types after them.

    The planner asks for the card types in capability_rank order. The resource catalogue's rank put the RTX 4000 Ada
    (3.01 s per update on pixel Montezuma) before the RTX 2080 Ti (1.88 s); rule pure-rl-gpu-placement.md item 3
    orders by the work's own speed. Changed 2026-11-20 18:12 PT.
    before: [rtx_4000_ada rank 6, rtx_2080_ti rank 10, h100 rank 1], speeds {ada: 3.01, 2080 ti: 1.88} ;
    after:  [2080 ti rank 0, ada rank 1, h100 rank 2] (the unmeasured H100 keeps its catalogue order after them)
    """
    ordered = sorted(classes, key=lambda record: (0, speeds[record["gres_type"]]) if record["gres_type"] in speeds
                     else (1, record["capability_rank"]))
    return [dict(record, capability_rank=rank) for rank, record in enumerate(ordered)]


def top_up(run_dir, phase, gpu_types=None, max_new_cards=None, dry_run=False, only_free_now=False,
           leave_cards=0, leave_card_types=None, cancel_waiting_after=None, python=PIXEL_PYTHON):
    """Submit worker jobs for the pending units that have no worker yet; returns the ids of the jobs kept."""
    run_dir = Path(run_dir)
    if (run_dir / "STOP").exists() or (run_dir / "SWEEP_COMPLETE").exists():
        print("RUN/STOP or RUN/SWEEP_COMPLETE exists: nothing is submitted")
        return []
    platform, _ = maze_submit.frozen_code(run_dir)
    allowed = tuple(gpu_types) if gpu_types else DEFAULT_CARD_TYPES
    profiles = pixel_queue.read_profiles()
    pending = unit_queue.read_units(run_dir, "pending")
    # the request of every card of this sweep, set in the maze planner this process holds
    maze_submit.set_request_per_card(CPUS_PER_CARD, MEMORY_GIB_PER_CARD)
    state = maze_submit.scheduler_state(run_dir)
    # the faster card types on this work are asked for first, when the run has a speed table
    state["classes"] = order_by_measured_speed(state["classes"], read_card_speeds(run_dir))
    waiting_cards = waiting_cards_by_profile(run_dir, profiles, state["classes"], common.own_live_jobs(run_dir))
    # cards another study's waiting units need stay out of the plan
    left = leave_free_cards(state, state["classes"], leave_cards, leave_card_types or allowed)
    jobs = plan_profiles(pending, profiles, waiting_cards, state, allowed, max_new_cards,
                         0 if only_free_now else None)

    # the depth of the submission, said before anything is sent
    print(f"{len(pending)} units pending, {unit_queue.counts(run_dir)['running']} running; allowed card types: "
          f"{', '.join(allowed)}")
    if leave_cards:
        print(f"    left free for another study ({sum(cards for _, _, cards in left)} of {leave_cards} cards asked): "
              + (", ".join(f"{cards} x {gpu_type} on {node}" for node, gpu_type, cards in left) or "none are free"))
    for name, profile in profiles.items():
        mine = [job for job in jobs if job["execution_profile"] == name]
        pinned = sum(job["cards"] for job in mine if job["node"])
        print(f"    profile {name!r}: {len(units_of_profile(pending, name, profile))} pending units, "
              f"{waiting_cards[name]} waiting worker cards of this run without a unit; {len(mine)} new jobs: "
              f"{pinned} cards on nodes with free cards now, {sum(job['cards'] for job in mine) - pinned} cards "
              "that wait in the scheduler's queue")
    for identity in maze_submit.IDENTITIES:
        print(f"    {identity}: {sum(job['cards'] for job in jobs if job['identity'] == identity)} cards in "
              f"{sum(job['identity'] == identity for job in jobs)} jobs")
    if not dry_run and jobs:
        (run_dir / "slurm").mkdir(exist_ok=True)
        (run_dir / "slurm/worker_env.sh").write_text(WORKER_ENVIRONMENT)
    sent = []
    for index, job in enumerate(jobs):
        # at most ten submissions, then a pause of 30 seconds, for the scheduler's request rate
        if index and index % 10 == 0 and not dry_run:
            time.sleep(30)
        print(f"  {job['identity']} {job['partition']} {job['node'] or 'any node'} {job['cards']} x {job['gpu_type']}"
              f" ({job['execution_profile']})" + (f" reservation {job['reservation']}" if job["reservation"] else ""))
        try:
            job_id = submit_job(run_dir, phase, job, worker_job_text(run_dir, platform, python, job["cards"]), dry_run)
        except subprocess.CalledProcessError as error:
            # one refused job stops neither the other jobs nor the monitor that called this
            print(f"  NOT SUBMITTED: the scheduler refused this job (exit {error.returncode}); the next ones are still sent")
            continue
        sent.append((job, job_id))
    # every job that was sent must be in its account's ledger before this call reports success
    for job, job_id in sent:
        account = job["identity"].removeprefix("uva-")
        if not dry_run and job_id not in maze_submit.ledger(run_dir, phase, account).read_text().split():
            raise RuntimeError(f"job {job_id} is missing from {maze_submit.ledger(run_dir, phase, account)}")
    # a job that has not started a short while after its submission is not left waiting in the queue
    cancelled = cancel_still_waiting(run_dir, phase, sent, cancel_waiting_after) if (
        cancel_waiting_after and sent and not dry_run) else []
    return [job_id for _, job_id in sent if job_id not in cancelled]


def submit_monitor(run_dir, options, dry_run=False, python=PIXEL_PYTHON):
    """Submit the persistent monitor: one processor on a long open partition, no card."""
    # copied from scripts/maze_campaign/submit.py, submit_monitor, with this folder's monitor
    platform, _ = maze_submit.frozen_code(run_dir)
    # one monitor per run folder: a live one is named and no second one is sent
    live = common.own_live_jobs(run_dir)
    for path in (Path(run_dir) / "slurm/submissions").glob("*.json"):
        record = common.read_json(path)
        if record["kind"] == "monitor" and record["job_id"] in live:
            print(f"monitor job {record['job_id']} of this run folder is still live; no second one is submitted")
            return record["job_id"]
    job = {"identity": maze_submit.IDENTITIES[0], "partition": "nolim", "node": None, "gpu_type": None, "cards": 0,
           "cpus": 1, "memory_gib": 2, "reservation": None, "kind": "monitor"}
    return submit_job(run_dir, "monitor", job, monitor_job_text(Path(run_dir), platform, python, options), dry_run)


def add_plan_options(parser):
    """The options that shape a worker submission; the monitor takes the same ones for its top-ups."""
    parser.add_argument("--gpu-types", help="comma-separated card types (gres names) the workers may ask for; "
                                            f"default: {','.join(DEFAULT_CARD_TYPES)}")
    parser.add_argument("--only-cards-free-now", action="store_true",
                        help="send only jobs pinned to nodes with a free card, processors and memory now; "
                             "none that would wait in the scheduler's queue")
    parser.add_argument("--leave-cards", type=int, default=0,
                        help="keep this many cards that are free now out of the plan, for another study's waiting units")
    parser.add_argument("--leave-card-types", help="comma-separated card types the cards of --leave-cards are "
                                                   "taken from; default: the allowed types, the most capable first")
    parser.add_argument("--worker-hours", type=float,
                        help="time limit of each worker job in hours (default: the longest the partition allows); "
                             "a short limit lets the scheduler place a low-priority worker on a free card")
    parser.add_argument("--accounts", help="comma-separated accounts new workers are submitted under "
                                           "(uva-sl5nw, uva-yuxinchen); default: both")
    parser.add_argument("--memory-gib-per-card", type=int,
                        help=f"host memory each worker card asks for, in GiB (default {MEMORY_GIB_PER_CARD})")
    parser.add_argument("--cancel-waiting-after", type=int,
                        help="cancel, by id, each job of this call that still waits in the scheduler's queue this "
                             "many seconds after the submission; its units stay pending")


def plan_arguments(args):
    """Turn the parsed plan options into top_up's keyword arguments, and set the accounts and the time limit."""
    maze_submit.set_worker_hours(args.worker_hours)
    maze_submit.set_identities(args.accounts.split(",") if args.accounts else None)
    set_memory_per_card(args.memory_gib_per_card)
    return {"gpu_types": args.gpu_types.split(",") if args.gpu_types else None,
            "only_free_now": args.only_cards_free_now, "leave_cards": args.leave_cards,
            "leave_card_types": args.leave_card_types.split(",") if args.leave_card_types else None,
            "cancel_waiting_after": args.cancel_waiting_after}


def main(argv=None):
    """Submit worker jobs for one phase, or the persistent monitor job."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", help="the phase the job ids are recorded under (words joined by hyphens)")
    parser.add_argument("--max-new-cards", type=int, help="send at most this many worker cards in this call")
    add_plan_options(parser)
    parser.add_argument("--monitor", action="store_true", help="submit the persistent monitor job instead")
    parser.add_argument("--top-up-phase", help="with --monitor: every pass also submits workers under this phase")
    parser.add_argument("--complete-when-drained", action="store_true",
                        help="with --monitor: write RUN/SWEEP_COMPLETE when no unit is pending, running or held")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the commands; submit nothing")
    parser.add_argument("--card-speeds-from", type=Path,
                        help="write RUN/infrastructure/card_seconds_per_update.json from this finished run's records")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    # the speed table of a new run, measured on an earlier run of the same work
    if args.card_speeds_from:
        speeds = measured_card_speeds(args.card_speeds_from.resolve())
        common.write_json(run_dir / SPEED_FILE, {"seconds_per_update": speeds, "profile": "one copy",
                                                 "measured_on": str(args.card_speeds_from.resolve()),
                                                 "written_at": common.now()})
        print("card speeds, seconds per update: " + ", ".join(f"{key} {value:.3f}" for key, value in speeds.items()))
        return
    plan = plan_arguments(args)
    if args.monitor:
        options = (["--top-up-phase", args.top_up_phase] if args.top_up_phase else []) + (
            ["--complete-when-drained"] if args.complete_when_drained else [])
        # the monitor's top-ups ask for what this call's workers would
        for name in ("gpu_types", "leave_card_types", "worker_hours", "accounts", "cancel_waiting_after",
                     "memory_gib_per_card"):
            if getattr(args, name) is not None:
                options += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        options += ["--leave-cards", str(args.leave_cards)] if args.leave_cards else []
        options += ["--only-cards-free-now"] if args.only_cards_free_now else []
        job_id = submit_monitor(run_dir, options, args.dry_run)
        print("dry run: nothing was submitted" if args.dry_run else f"monitor job id: {job_id}")
    elif args.phase:
        job_ids = top_up(run_dir, args.phase, max_new_cards=args.max_new_cards, dry_run=args.dry_run, **plan)
        print("dry run: nothing was submitted" if args.dry_run else f"submitted job ids: {' '.join(job_ids) or 'none'}")
    else:
        parser.error("give --phase to submit workers, or --monitor")


if __name__ == "__main__":
    main()
