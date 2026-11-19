"""The steps of one pass of the Montezuma stage-1 hand-over loop that are not remote_fleet.py's own tick (with MIT).

    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN gautschi-share   # one ssh call to Gautschi
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN mit-connection   # no login: ssh -O check mit only
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN mit-share        # one call through the shared connection
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN pack-top-up
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN smoke-wait --server gilbreth --job 12434389
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN held-pool-smoke --server mit --pool preemptable-h200 --job 29603828
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN queue-check      # exit 1 when a unit is in two folders
    python3 montezuma_hand_over_steps_with_mit.py --run-dir RUN finished         # exit 0 when no unit is pending or remote

Standard library only; it runs on the UVA login node beside montezuma_hand_over_loop_with_mit.sh. It is
montezuma_hand_over_steps.py (the helper of the loop without MIT) with five changes: the steps mit-connection,
mit-share and held-pool-smoke, pack-top-up counting one limit per server and making packs of each pool's own size
(four copies for Gautschi, eight for MIT's H200 since 2026-11-18 about 22:30 PT), and smoke-wait waiting for several
jobs.

gautschi-share: account dsimchil holds 4 H100 at once on Gautschi, shared with the group's other member
long635. The cards the other member's running and waiting `ai` jobs ask for are taken out of this run's
Gautschi pool before every tick: max_jobs = 4 - those cards, and units held at once = 2 x max_jobs.
mit-connection: MIT is reached only through the shared connection held on portal04, which only the user can open
again (Duo push or web login). `ssh -O check mit` is read (it is no login); while the connection is closed, MIT is
marked not ready in the server table with the reason and the marker `not_ready_by` set to this step, so the tick
leaves MIT alone; once it is open again, MIT is made ready again, but only when this step was what made it not ready.
mit-share: account shuzel holds 4 cards at once in the Advanced pool of mit_normal_gpu (the free-tier jobs of that
partition share the counter) and 4 on mit_preemptable, shared by every session of the account. The cards that
other jobs of the account ask for (running or waiting; every job that is not a worker or smoke job of this run's
MIT pools, this run's pack-size profile job included) are taken out of this run's MIT pools before every tick:
max_jobs = 4 - those cards in each pool, and units held at once (the first pool's limit, which hand_over applies
to the whole server) = 2 x the jobs of both pools.
pack-top-up: the pack pools take packs only: four-copy pools (Gautschi) four-copy packs, eight-copy pools (MIT's
H200) eight-copy packs. For each pack size, packs are made from never-started singles (build_units.py --pack-pending
--pack-size N) so that the packs of that size pending on UVA and held by each ready server reach that server's limit
of units held at once: the largest units_at_once of its pools of that profile, because remote_fleet.hand_over counts
every unit of a profile a server holds against the limit of each of its pools of that profile.
smoke-wait: once every one of a server's smoke jobs has ended, `remote_fleet.py smoke-result` reads the verdicts
once (a marker file in RUN/slurm/ records the reading); the server becomes ready only if every pool passed.
held-pool-smoke: a pool held back in pools_held_back (MIT's preemptable-h200 until its own smoke job passes) is
brought back by `remote_fleet.py smoke-result --pool` once its smoke job has ended and only if it passed; read once.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ACCOUNT_CARDS = 4                       # GrpTRES gres/hp_gpu=4 of account dsimchil (read 2026-11-13 PT)
OWN_USER = "liu4869"
SHARE_COMMAND = ("squeue -A dsimchil -h -p ai -t RUNNING,PENDING "
                 "-O 'UserName:20,State:12,tres-per-node:60,tres-per-job:60,NumNodes:6'")
MIT_CARDS = {"advanced-h200": 4, "preemptable-h200": 4}     # the account's caps (sacctmgr, read 2026-11-18 PT)
MIT_PARTITIONS = {"mit_normal_gpu": "advanced-h200", "mit_preemptable": "preemptable-h200"}
MIT_SHARE_COMMAND = ("squeue -u shuzel -h -t RUNNING,PENDING "
                     "-O 'JobID:14,Name:150,Partition:24,tres-per-node:60,tres-per-job:60,NumNodes:6,State:12'")
MIT_CHECK = ["ssh", "-O", "check", "mit"]
# While another session of the account has jobs on MIT's GPU partitions, this run's waiting MIT jobs get this nice
# (user, 2026-11-18 about 23:50 PT: "if there is another session that also submits to MIT, please hold or low
# priority the jobs of this RND runs on MIT (choose one that cumulate aging)"). MIT sets no ACCRUE_ALWAYS, so a held
# job stops gaining age, while a job with a higher nice stays eligible and keeps it. 3,000,000 is above the largest
# gap between two jobs of the account on those partitions (age 50,000 + partition 600,000 + QOS 2,000,000 at most;
# fair share is the same for both), so every job of another session starts before this run's waiting jobs.
YIELD_NICE = 3000000
YIELD_STATE = "slurm/mit_yield_state.json"
CONNECTION_MARK = "the hand-over loop's connection check (mit-connection)"
BUILD_UNITS = Path("/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign/build_units.py")
REMOTE_FLEET = BUILD_UNITS.parent / "remote_fleet.py"
# the profiles that run packs -> the word in their packs' names and the copies of a pack (build_units.PACK_WORDS)
PACK_PROFILES = {"four copies": ("four-copies", 4), "eight copies": ("eight-copies", 8)}
JOB_PREFIX_LENGTH = 8                   # remote_fleet.job_prefix: "pixelfleet-" and 8 characters of a sha1


def table_path(run_dir):
    """Name the run folder's server table."""
    return run_dir / "infrastructure/remote_fleet_servers.json"


def write_table(run_dir, table):
    """Write the server table in one rename, so a reader never sees half a file."""
    path = table_path(run_dir)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(table, indent=2) + "\n")
    os.replace(temporary, path)


def cards_of(line):
    """Read a squeue line of the share command; returns (user, state, cards asked for)."""
    # before: "long635   PENDING   gres/gpu:h100:2,gres/hp_gpu:2   N/A   1" ; after: ("long635", "PENDING", 2)
    # before: "long635   RUNNING   N/A   gres/gpu:h100:4   1"                ; after: ("long635", "RUNNING", 4)
    # a job of the ai partition always holds a card: one whose fields name no count counts as one per node
    user, state, per_node, per_job, nodes = line.split()
    for field, factor in ((per_node, int(nodes) if nodes.isdecimal() else 1), (per_job, 1)):
        counts = [int(value) for value in re.findall(r"gpu(?::[a-z0-9_]+)?:(\d+)", field)]
        if counts:
            return user, state, max(counts) * factor
    return user, state, int(nodes) if nodes.isdecimal() else 1


def gautschi_share(run_dir):
    """Lower or raise this run's Gautschi pool by the cards the account's other member asks for; prints what it set."""
    table = json.loads(table_path(run_dir).read_text())
    entry = table["gautschi"]
    if not entry["ready"]:
        print("gautschi-share: Gautschi is not ready in this run folder; nothing read")
        return
    answer = subprocess.run(entry["ssh"] + [SHARE_COMMAND], capture_output=True, text=True, timeout=120,
                            stdin=subprocess.DEVNULL)
    if answer.returncode != 0:
        raise SystemExit(f"gautschi-share: squeue on Gautschi ended with {answer.returncode}: {answer.stderr.strip()}")
    # the other member's running and waiting cards on ai; ours are counted by the tick as live jobs of the pool
    jobs = [cards_of(line) for line in answer.stdout.splitlines() if line.strip()]
    others = sum(cards for user, _, cards in jobs if user != OWN_USER)
    allowed = max(0, ACCOUNT_CARDS - others)
    # the table is changed under the fleet's lock, as every remote_fleet command does
    with (run_dir / "slurm/remote_fleet.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        table = json.loads(table_path(run_dir).read_text())
        pool = table["gautschi"]["pools"]["ai-h100"]
        limits = table["gautschi"]["hand_over"]["pools"]["ai-h100"]
        before = (pool["max_jobs"], limits["units_at_once"])
        pool["max_jobs"], limits["units_at_once"] = allowed, 2 * allowed
        if before != (allowed, 2 * allowed):
            write_table(run_dir, table)
    print(f"gautschi-share: account dsimchil, cards of other members on ai (running or waiting): {others}; "
          f"ours {sum(cards for user, _, cards in jobs if user == OWN_USER)}; this run's Gautschi pool: at most "
          f"{allowed} job(s) and {2 * allowed} unit(s) held" + (f" (was {before[0]} and {before[1]})"
                                                                if before != (allowed, 2 * allowed) else " (unchanged)"))


def job_prefix(run_dir):
    """The first part of the name of every remote job of the run folder (remote_fleet.job_prefix)."""
    # before: .../runs/2026-11-18-12-00_pixel-montezuma_... ; after: "pixelfleet-9d411784"
    return "pixelfleet-" + hashlib.sha1(Path(run_dir).resolve().name.encode()).hexdigest()[:JOB_PREFIX_LENGTH]


def mit_connection(run_dir):
    """Read whether the shared MIT connection on this machine is open, and mark MIT not ready or ready again by it."""
    is_open = subprocess.run(MIT_CHECK, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL).returncode == 0
    # the table is changed under the fleet's lock, as every remote_fleet command does
    with (run_dir / "slurm/remote_fleet.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        table = json.loads(table_path(run_dir).read_text())
        entry = table.get("mit")
        if entry is None:
            print("mit-connection: the table has no MIT entry")
            return
        # before: ready, connection closed ; after: not ready, the reason and the marker of this step
        # before: not ready by this step, connection open ; after: ready, reason and marker removed
        # a server made not ready by anything else (its smoke job, failed calls) is left as it is
        if not is_open and entry["ready"]:
            entry.update(ready=False, not_ready_by=CONNECTION_MARK, not_ready_reason=(
                f"the shared connection to MIT on portal04 is closed ({datetime.now().astimezone().isoformat()}); "
                "only the user can open it again (Duo push or the web login at orcd-ood.mit.edu)"))
            write_table(run_dir, table)
            print("mit-connection: the shared connection is CLOSED; MIT is marked not ready until it is open again "
                  "(no login was tried); tell the user")
        elif is_open and not entry["ready"] and entry.get("not_ready_by") == CONNECTION_MARK:
            entry["ready"] = True
            entry.pop("not_ready_by")
            entry.pop("not_ready_reason", None)
            write_table(run_dir, table)
            print("mit-connection: the shared connection is open again; MIT is ready again")
        else:
            print(f"mit-connection: connection {'open' if is_open else 'closed'}; MIT "
                  f"{'ready' if entry['ready'] else 'not ready: ' + entry.get('not_ready_reason', '')}")


def mit_cards_of(line):
    """Read a squeue line of the MIT share command; returns (job id, job name, pool, cards asked for)."""
    # before: "29603853  montezuma-pack-size-profile-h200  mit_normal_gpu  N/A  gres/gpu:h200:1  1  PENDING"
    # after:  ("29603853", "montezuma-pack-size-profile-h200", "advanced-h200", 1)
    job, name, partition, per_node, per_job, nodes, _ = line.split()
    cards = 0
    for field, factor in ((per_node, int(nodes) if nodes.isdecimal() else 1), (per_job, 1)):
        counts = [int(value) for value in re.findall(r"gpu(?::[a-z0-9_]+)?:(\d+)", field)]
        if counts:
            cards = max(counts) * factor
            break
    return job, name, MIT_PARTITIONS.get(partition), cards


def yield_decision(lines, ledger, lowered):
    """Decide which of this run's MIT jobs to lower and which to restore; returns (others, lower, restore)."""
    # before: lines of MIT_SHARE_COMMAND, ledger {"29619724", ...}, lowered {"29606828"} (from the state file)
    #         "29700001  other-session-job  mit_normal_gpu  N/A  gres/gpu:h200:1  1  PENDING"
    #         "29619724  pixelfleet-...-1   mit_normal_gpu  N/A  gres/gpu:h200:1  1  RUNNING"
    #         "29619730  pixelfleet-...-3   mit_normal_gpu  N/A  gres/gpu:h200:1  1  PENDING"
    # after:  others ["29700001"], lower ["29619730"] (waiting, ours, not yet lowered), restore []
    #         with no other job: others [], lower [], restore the lowered ids still in the queue
    others, ours_waiting, present = [], [], set()
    for line in lines:
        if not line.strip():
            continue
        job, _, pool, _ = mit_cards_of(line)
        state = line.split()[-1]
        present.add(job)
        if job in ledger:
            if state == "PENDING":
                ours_waiting.append(job)
        elif pool is not None:
            others.append(job)
    if others:
        return others, [job for job in ours_waiting if job not in lowered], []
    return others, [], sorted(job for job in lowered if job in present)


def mit_yield(run_dir, entry, squeue_text):
    """Lower this run's waiting MIT jobs below the account's other jobs while another session has jobs there; restore them after."""
    ledger = set((run_dir / "slurm/submitted_jobids_mit.txt").read_text().split())
    state_path = run_dir / YIELD_STATE
    state = json.loads(state_path.read_text()) if state_path.exists() else {"lowered": {}}
    others, lower, restore = yield_decision(squeue_text.splitlines(), ledger, set(state["lowered"]))
    # one ssh call for every change; each id is one of this run's own ledger
    changes = [f"scontrol update JobId={job} Nice={YIELD_NICE}" for job in lower] + \
              [f"scontrol update JobId={job} Nice=0" for job in restore]
    if changes:
        answer = subprocess.run(entry["ssh"] + [" && ".join(changes)], capture_output=True, text=True, timeout=120,
                                stdin=subprocess.DEVNULL)
        if answer.returncode != 0:
            raise SystemExit(f"mit-yield: scontrol on MIT ended with {answer.returncode}: {answer.stderr.strip()}")
    stamp = datetime.now().astimezone().isoformat()
    for job in lower:
        state["lowered"][job] = stamp
    for job in restore:
        del state["lowered"][job]
    if not others:
        # a lowered job that left the queue (started or ended) needs nothing more
        state["lowered"] = {}
    state["last_read"] = {"at": stamp, "other_jobs": others}
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, state_path)
    print(f"mit-yield: other jobs of the account on MIT's GPU partitions: {len(others)}"
          + (f" ({', '.join(others[:6])}{', ...' if len(others) > 6 else ''})" if others else "")
          + f"; this run's waiting jobs lowered now: {lower or 'none'}; restored now: {restore or 'none'}"
          + f"; lowered in all: {sorted(state['lowered']) or 'none'}")


def mit_share(run_dir):
    """Lower or raise this run's MIT pools by the cards other jobs of the account hold or wait for; prints what it set."""
    table = json.loads(table_path(run_dir).read_text())
    entry = table.get("mit")
    if entry is None or not entry["ready"]:
        print("mit-share: MIT is not ready in this run folder; nothing read")
        return
    if subprocess.run(MIT_CHECK, capture_output=True, timeout=60, stdin=subprocess.DEVNULL).returncode != 0:
        print("mit-share: the shared connection is closed; nothing read")
        return
    answer = subprocess.run(entry["ssh"] + [MIT_SHARE_COMMAND], capture_output=True, text=True, timeout=120,
                            stdin=subprocess.DEVNULL)
    if answer.returncode != 0:
        raise SystemExit(f"mit-share: squeue on MIT ended with {answer.returncode}: {answer.stderr.strip()}")
    # this run yields to another session's MIT jobs (lower priority, age kept), and comes back when they are gone
    mit_yield(run_dir, entry, answer.stdout)
    # this run's worker and smoke jobs of a pool are counted by the tick as live jobs of that pool; every other
    # job of the account on the two partitions takes cards from them
    # before: "pixelfleet-9d411784-mit-advanced-h200-20261118-200000-1" ; after: a job of pool advanced-h200, not counted
    pool_job = f"{job_prefix(run_dir)}-mit-"
    others = {pool: 0 for pool in MIT_CARDS}
    for line in answer.stdout.splitlines():
        if not line.strip():
            continue
        _, name, pool, cards = mit_cards_of(line)
        if pool is None:
            continue
        if name.startswith(pool_job) and name[len(pool_job):].rsplit("-", 3)[0] in entry["pools"]:
            continue
        others[pool] += cards
    with (run_dir / "slurm/remote_fleet.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        table = json.loads(table_path(run_dir).read_text())
        pools, limits = table["mit"]["pools"], table["mit"]["hand_over"]["pools"]
        # only the pools that take units: a pool held back until its smoke job passes has no jobs
        # before: pools advanced-h200 and preemptable-h200, others {"advanced-h200": 1, "preemptable-h200": 0}
        # after:  max_jobs 3 and 4, units_at_once 14 in the first pool and 0 in the second
        taking = [pool for pool in MIT_CARDS if pool in pools]
        allowed = {pool: max(0, MIT_CARDS[pool] - others[pool]) for pool in taking}
        before = {pool: (pools[pool]["max_jobs"], limits[pool]["units_at_once"]) for pool in taking}
        for pool in taking:
            pools[pool]["max_jobs"] = allowed[pool]
            limits[pool]["units_at_once"] = 0
        # the server-wide limit stands in the first pool (hand_over counts all of the server's four-copy units)
        limits[taking[0]]["units_at_once"] = 2 * sum(allowed.values())
        after = {pool: (pools[pool]["max_jobs"], limits[pool]["units_at_once"]) for pool in taking}
        if after != before:
            write_table(run_dir, table)
    print(f"mit-share: account shuzel, cards of other jobs (running or waiting): {others}; this run's MIT pools: "
          + ", ".join(f"{pool} at most {after[pool][0]} job(s)" for pool in taking)
          + f", {after[taking[0]][1]} unit(s) held on the server"
          + (" (unchanged)" if after == before else f" (was {before})"))


def pack_top_up(run_dir):
    """Make packs of each pack size from never-started singles until the pools of that size on ready servers can be filled."""
    table = json.loads(table_path(run_dir).read_text())
    for profile, (word, copies) in PACK_PROFILES.items():
        # the units the pools of this profile on every ready server may hold at once (one limit per server:
        # hand_over counts all of a server's units of the profile against each pool's limit), and the packs of
        # this size those servers hold
        # before: gautschi ready, ai-h100 four copies with units_at_once 8, 8 four-copy packs in queue/remote_gautschi;
        #         mit ready, advanced-h200 eight copies with 8 and preemptable-h200 0, 2 four-copy and 1 eight-copy
        #         pack there
        # after:  four copies: wanted 8, held 8 (MIT's four-copy packs are not counted: MIT has no four-copy pool);
        #         eight copies: wanted 8, held 1
        wanted = held = 0
        for name, entry in table.items():
            limits = [limit["units_at_once"] for pool, limit in entry.get("hand_over", {}).get("pools", {}).items()
                      if entry["ready"] and entry["pools"][pool]["execution_profile"] == profile]
            if limits:
                wanted += max(limits)
                held += sum(word in path.stem for path in (run_dir / "queue" / f"remote_{name}").glob("*.json"))
        pending = sum(word in path.stem for path in (run_dir / "queue/pending").glob("*.json"))
        needed = max(0, wanted - held - pending)
        print(f"pack-top-up: {profile} units wanted at once {wanted}, held by servers {held}, pending on UVA "
              f"{pending}: {needed} new pack(s) asked for")
        # the builder takes the first never-started singles in claim order and holds the queue lock itself
        if needed:
            made = subprocess.run([sys.executable, str(BUILD_UNITS), "--run-dir", str(run_dir), "--pack-pending",
                                   "--packs", str(needed), "--pack-size", str(copies)], capture_output=True, text=True,
                                  timeout=600, stdin=subprocess.DEVNULL)
            print("pack-top-up: " + (made.stdout.strip().splitlines() or ["(no output)"])[-1] + " " + made.stderr.strip())
            if made.returncode != 0:
                raise SystemExit(f"pack-top-up: build_units.py ended with {made.returncode}")


def smoke_wait(run_dir, server, jobs):
    """Read a server's smoke verdicts once all its smoke jobs have ended; remote_fleet.py makes it ready only if they passed."""
    table = json.loads(table_path(run_dir).read_text())
    entry = table[server]
    # before: jobs ["12434389"] ; after: slurm/smoke_verdict_read_gilbreth_job-12434389.txt (the name of one job's marker)
    # before: jobs ["29603827", "29603828"] ; after: slurm/smoke_verdict_read_mit_job-29603827-29603828.txt
    marker = run_dir / "slurm" / f"smoke_verdict_read_{server}_job-{'-'.join(jobs)}.txt"
    # a ready server, or verdicts read before, need nothing more
    if entry["ready"] or marker.exists():
        print(f"smoke-wait: {server} is {'ready' if entry['ready'] else 'not ready'}; the verdict of job(s) {', '.join(jobs)} "
              f"{'was read at ' + marker.read_text().strip() if marker.exists() else 'is not needed'}")
        return
    # a server reached through a shared connection that is closed is not asked (no login)
    if entry.get("check") and subprocess.run(entry["check"], capture_output=True, timeout=60,
                                             stdin=subprocess.DEVNULL).returncode != 0:
        print(f"smoke-wait: the shared connection to {server} is closed; nothing read")
        return
    # the jobs' states, through the shared connection
    answer = subprocess.run(entry["ssh"] + [f"sacct -j {','.join(str(int(job)) for job in jobs)} -X -n -P -o JobID,State"],
                            capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    if answer.returncode != 0:
        raise SystemExit(f"smoke-wait: sacct on {server} ended with {answer.returncode}: {answer.stderr.strip()}")
    # before: "29603827|COMPLETED\n29603828|RUNNING" ; after: {"29603827": "COMPLETED", "29603828": "RUNNING"}
    states = dict(line.split("|", 1) for line in answer.stdout.split() if "|" in line)
    states = {job: states.get(job, "UNKNOWN").split()[0] for job in jobs}
    waiting = {job: state for job, state in states.items() if state in (
        "PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "UNKNOWN")}
    if waiting:
        print(f"smoke-wait: {server} smoke job(s) not ended: {waiting}; {server} stays not ready")
        return
    # every job has ended: smoke-result reads the verdicts (deciding the first update again) and sets ready
    read = subprocess.run([sys.executable, str(REMOTE_FLEET), "--run-dir", str(run_dir), "smoke-result",
                           "--server", server], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    print(read.stdout.strip(), read.stderr.strip())
    if read.returncode != 0:
        raise SystemExit(f"smoke-wait: smoke-result ended with {read.returncode}")
    ready = json.loads(table_path(run_dir).read_text())[server]["ready"]
    marker.write_text(f"{datetime.now().astimezone().isoformat()}, job states {states}, "
                      f"{server} {'READY' if ready else 'NOT ready'}\n")
    print(f"smoke-wait: smoke job(s) {', '.join(jobs)} ended ({states}); {server} is {'READY' if ready else 'NOT ready'}")


def held_pool_smoke(run_dir, server, pool, job):
    """Once the smoke job of a held-back pool has ended, read its verdict once; the pool takes units again if it passed."""
    entry = json.loads(table_path(run_dir).read_text())[server]
    marker = run_dir / "slurm" / f"smoke_verdict_read_{server}_pool-{pool}_job-{job}.txt"
    # a pool that is not held back, or a verdict read before, needs nothing more
    held = {name for name, settings in entry.get("pools_held_back", {}).items() if isinstance(settings, dict)}
    if pool not in held or marker.exists():
        print(f"held-pool-smoke: {server} pool {pool} is {'held back' if pool in held else 'taking units'}; the verdict "
              f"of job {job} {'was read at ' + marker.read_text().strip() if marker.exists() else 'is not needed'}")
        return
    # a server reached through a shared connection that is closed is not asked (no login)
    if entry.get("check") and subprocess.run(entry["check"], capture_output=True, timeout=60,
                                             stdin=subprocess.DEVNULL).returncode != 0:
        print(f"held-pool-smoke: the shared connection to {server} is closed; nothing read")
        return
    answer = subprocess.run(entry["ssh"] + [f"sacct -j {int(job)} -X -n -P -o State"], capture_output=True, text=True,
                            timeout=120, stdin=subprocess.DEVNULL)
    if answer.returncode != 0:
        raise SystemExit(f"held-pool-smoke: sacct on {server} ended with {answer.returncode}: {answer.stderr.strip()}")
    words = answer.stdout.split()
    state = words[0] if words else "UNKNOWN"
    if state in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "UNKNOWN"):
        print(f"held-pool-smoke: {server} pool {pool} smoke job {job} is {state}; the pool stays held back")
        return
    # the job has ended: remote_fleet.py reads the pool's verdict and brings the pool back only if it passed
    read = subprocess.run([sys.executable, str(REMOTE_FLEET), "--run-dir", str(run_dir), "smoke-result", "--server", server,
                           "--pool", pool], capture_output=True, text=True, timeout=900, stdin=subprocess.DEVNULL)
    print(read.stdout.strip(), read.stderr.strip())
    if read.returncode != 0:
        raise SystemExit(f"held-pool-smoke: smoke-result ended with {read.returncode}")
    back = pool in json.loads(table_path(run_dir).read_text())[server]["pools"]
    marker.write_text(f"{datetime.now().astimezone().isoformat()}, job state {state}, pool "
                      f"{'takes units again' if back else 'stays held back'}\n")
    print(f"held-pool-smoke: smoke job {job} ended ({state}); {server} pool {pool} "
          f"{'TAKES UNITS AGAIN' if back else 'stays held back'}")


def queue_check(run_dir):
    """Say whether every unit stands in one queue folder only (so no unit has a job on UVA and one on a server)."""
    # before: queue/running/a.json and queue/remote_gautschi/a.json ; after: "a: running, remote_gautschi", exit 1
    folders = {}
    for folder in sorted(path for path in (run_dir / "queue").iterdir() if path.is_dir()):
        for path in folder.glob("*.json"):
            folders.setdefault(path.stem, []).append(folder.name)
    twice = {unit: places for unit, places in folders.items() if len(places) > 1}
    counts = {}
    for places in folders.values():
        for place in places:
            counts[place] = counts.get(place, 0) + 1
    print("queue-check: " + ", ".join(f"{place} {count}" for place, count in sorted(counts.items()))
          + ("; every unit in one folder" if not twice else
             "; IN TWO FOLDERS: " + "; ".join(f"{unit}: {', '.join(places)}" for unit, places in sorted(twice.items()))))
    return 1 if twice else 0


def finished(run_dir):
    """Say whether no unit is pending on UVA or held by a remote server; exit code 0 when so."""
    queue = run_dir / "queue"
    pending = len(list((queue / "pending").glob("*.json")))
    remote = sum(len(list(folder.glob("*.json"))) for folder in queue.glob("remote_*") if folder.is_dir())
    print(f"finished: pending on UVA {pending}, held by remote servers {remote}")
    return 0 if pending == 0 and remote == 0 else 1


def main():
    """Run one step for a run folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("step", choices=("gautschi-share", "mit-connection", "mit-share", "pack-top-up", "smoke-wait",
                                         "held-pool-smoke", "queue-check", "finished"))
    parser.add_argument("--pool", help="held-pool-smoke: the held-back pool whose smoke job is waited for")
    parser.add_argument("--server", help="smoke-wait: the server whose smoke jobs are waited for")
    parser.add_argument("--job", action="append", help="smoke-wait: the id of one smoke job (from the server's ledger); "
                                                       "give one --job per pool")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.step == "gautschi-share":
        gautschi_share(run_dir)
    elif args.step == "mit-connection":
        mit_connection(run_dir)
    elif args.step == "mit-share":
        mit_share(run_dir)
    elif args.step == "pack-top-up":
        pack_top_up(run_dir)
    elif args.step == "smoke-wait":
        smoke_wait(run_dir, args.server, args.job)
    elif args.step == "held-pool-smoke":
        held_pool_smoke(run_dir, args.server, args.pool, args.job[0])
    else:
        raise SystemExit(queue_check(run_dir) if args.step == "queue-check" else finished(run_dir))


if __name__ == "__main__":
    main()
