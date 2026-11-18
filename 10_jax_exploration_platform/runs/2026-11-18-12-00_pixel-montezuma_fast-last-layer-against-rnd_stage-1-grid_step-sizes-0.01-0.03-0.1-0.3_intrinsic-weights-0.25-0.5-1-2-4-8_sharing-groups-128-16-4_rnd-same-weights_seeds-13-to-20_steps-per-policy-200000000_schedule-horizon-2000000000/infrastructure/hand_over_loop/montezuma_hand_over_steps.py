"""The steps of one pass of the Montezuma stage-1 hand-over loop that are not remote_fleet.py's own tick.

    python3 montezuma_hand_over_steps.py --run-dir RUN gautschi-share   # one ssh call to Gautschi
    python3 montezuma_hand_over_steps.py --run-dir RUN pack-top-up
    python3 montezuma_hand_over_steps.py --run-dir RUN smoke-wait --server gilbreth --job 12434389
    python3 montezuma_hand_over_steps.py --run-dir RUN queue-check      # exit 1 when a unit is in two folders
    python3 montezuma_hand_over_steps.py --run-dir RUN finished         # exit 0 when no unit is pending or remote

Standard library only; it runs on the UVA login node beside montezuma_hand_over_loop.sh.

gautschi-share: account dsimchil holds 4 H100 at once on Gautschi, shared with the group's other member
long635. The cards the other member's running and waiting `ai` jobs ask for are taken out of this run's
Gautschi pool before every tick: max_jobs = 4 - those cards, and units held at once = 2 x max_jobs.
pack-top-up: Gautschi takes four-copy packs only. Packs are made from never-started singles
(build_units.py --pack-pending) so that the packs pending on UVA and held by the four-copy pools of ready
servers reach those pools' limits of units held at once.
smoke-wait: once a server's smoke job has ended, `remote_fleet.py smoke-result` reads its verdict once (a marker
file in RUN/slurm/ records the reading); the server becomes ready only if the verdict passed.
"""
import argparse
from datetime import datetime
import fcntl
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
BUILD_UNITS = Path("/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign/build_units.py")
REMOTE_FLEET = BUILD_UNITS.parent / "remote_fleet.py"
PACK_WORD = "four-copies"


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


def pack_top_up(run_dir):
    """Make packs from never-started singles until the four-copy pools of ready servers can be filled; prints the count."""
    table = json.loads(table_path(run_dir).read_text())
    # the units the four-copy pools of every ready server may hold at once, and the packs those servers hold
    # before: gautschi ready, ai-h100 four copies with units_at_once 8, 3 packs in queue/remote_gautschi
    # after:  wanted 8, held 3
    wanted = held = 0
    for name, entry in table.items():
        limits = [limit["units_at_once"] for pool, limit in entry.get("hand_over", {}).get("pools", {}).items()
                  if entry["ready"] and entry["pools"][pool]["execution_profile"] == "four copies"]
        if limits:
            wanted += sum(limits)
            held += sum(PACK_WORD in path.stem for path in (run_dir / "queue" / f"remote_{name}").glob("*.json"))
    pending = sum(PACK_WORD in path.stem for path in (run_dir / "queue/pending").glob("*.json"))
    needed = max(0, wanted - held - pending)
    print(f"pack-top-up: four-copy units wanted at once {wanted}, held by servers {held}, pending on UVA {pending}: "
          f"{needed} new pack(s) asked for")
    # the builder takes the first never-started singles in claim order and holds the queue lock itself
    if needed:
        made = subprocess.run([sys.executable, str(BUILD_UNITS), "--run-dir", str(run_dir), "--pack-pending",
                               "--packs", str(needed)], capture_output=True, text=True, timeout=600,
                              stdin=subprocess.DEVNULL)
        print("pack-top-up: " + (made.stdout.strip().splitlines() or ["(no output)"])[-1] + " " + made.stderr.strip())
        if made.returncode != 0:
            raise SystemExit(f"pack-top-up: build_units.py ended with {made.returncode}")


def smoke_wait(run_dir, server, job):
    """Read a server's smoke verdict once its smoke job has ended; remote_fleet.py makes it ready only if it passed."""
    table = json.loads(table_path(run_dir).read_text())
    entry = table[server]
    marker = run_dir / "slurm" / f"smoke_verdict_read_{server}_job-{job}.txt"
    # a ready server, or a verdict read before, needs nothing more
    if entry["ready"] or marker.exists():
        print(f"smoke-wait: {server} is {'ready' if entry['ready'] else 'not ready'}; the verdict of job {job} "
              f"{'was read at ' + marker.read_text().strip() if marker.exists() else 'is not needed'}")
        return
    # the job's state, through the shared connection
    answer = subprocess.run(entry["ssh"] + [f"sacct -j {int(job)} -X -n -P -o State"], capture_output=True, text=True,
                            timeout=120, stdin=subprocess.DEVNULL)
    if answer.returncode != 0:
        raise SystemExit(f"smoke-wait: sacct on {server} ended with {answer.returncode}: {answer.stderr.strip()}")
    words = answer.stdout.split()
    state = words[0] if words else "UNKNOWN"
    if state in ("PENDING", "RUNNING", "REQUEUED", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "UNKNOWN"):
        print(f"smoke-wait: {server} smoke job {job} is {state}; {server} stays not ready")
        return
    # the job has ended: smoke-result reads its verdict (deciding the first update again) and sets ready
    read = subprocess.run([sys.executable, str(REMOTE_FLEET), "--run-dir", str(run_dir), "smoke-result",
                           "--server", server], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    print(read.stdout.strip(), read.stderr.strip())
    if read.returncode != 0:
        raise SystemExit(f"smoke-wait: smoke-result ended with {read.returncode}")
    ready = json.loads(table_path(run_dir).read_text())[server]["ready"]
    marker.write_text(f"{datetime.now().astimezone().isoformat()}, job state {state}, "
                      f"{server} {'READY' if ready else 'NOT ready'}\n")
    print(f"smoke-wait: smoke job {job} ended ({state}); {server} is {'READY' if ready else 'NOT ready'}")


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
    parser.add_argument("step", choices=("gautschi-share", "pack-top-up", "smoke-wait", "queue-check", "finished"))
    parser.add_argument("--server", help="smoke-wait: the server whose smoke job is waited for")
    parser.add_argument("--job", help="smoke-wait: the id of that smoke job (from the server's ledger)")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.step == "gautschi-share":
        gautschi_share(run_dir)
    elif args.step == "pack-top-up":
        pack_top_up(run_dir)
    elif args.step == "smoke-wait":
        smoke_wait(run_dir, args.server, args.job)
    else:
        raise SystemExit(queue_check(run_dir) if args.step == "queue-check" else finished(run_dir))


if __name__ == "__main__":
    main()
