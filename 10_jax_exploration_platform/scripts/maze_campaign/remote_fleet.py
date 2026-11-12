"""Run units of the maze campaign on MIT, Purdue Gautschi, TAMU ACES and NCSA DeltaAI.

ADAPTED FROM THE GRID CAMPAIGN AND NOT EXERCISED: nothing in this file has run against a server.

    python remote_fleet.py --run-dir RUN stage --server S --units N [--match TEXT] [--from-end] [--dry-run]
    python remote_fleet.py --run-dir RUN submit --server S --jobs K [--pool P] [--dry-run]
    python remote_fleet.py --run-dir RUN tick [--dry-run]
    python remote_fleet.py --run-dir RUN status
    python remote_fleet.py --run-dir RUN take-back --server S --units N [--match TEXT] [--from-end] [--hold]

These are the commands of `scripts/grid_initialization_campaign/remote_fleet.py`, run unchanged:
the server table RUN/infrastructure/remote_fleet_servers.json, the queue folders on a server, one
status call per server and tick, no new login, and units that no job claimed within two hours
going back to UVA are all described at the head of that file. Four things are replaced here:

  1. The worker job on a server starts this folder's worker.py with --remote-records, which
     writes the claim records and the completed-unit folders the grid campaign's tick reads.
  2. Preparing a server sends the job scripts and the run's frozen code copy. There is no
     first-unit comparison with UVA.
  3. A pulled unit is checked by its shard: it must end with a unit_complete record. Its shard
     and log then move to RUN/data and RUN/logs, and its line for the copies-per-card skill is
     added on UVA.
  4. Only complete units come back with their records. A unit that returns unfinished starts
     again from its beginning on UVA; while it stays on a server it continues from its checkpoint there.

H100 and H200 work goes to the first server of SERVER_ORDER that can start it (rule
h100-h200-server-order.md). This file offers no cancel command (see cancel_ledger.py).
"""
import os
from pathlib import Path
import shlex
import sys

from grid_campaign import common, load, own
import unit_queue

fleet = load("remote_fleet")
SERVER_ORDER = ("mit", "gautschi", "aces", "deltaai")
OBSERVED = common.PROJECT / "10_jax_exploration_platform/device/observed"
REMOTE_FOLDERS = tuple(f"queue/{state}" for state in fleet.REMOTE_STATES) + (
    "specifications", "data/research", "checkpoints", "logs", "slurm/jobs")

WORKER_TEMPLATE = r'''#!/bin/bash
# Worker job of the maze study on @SERVER@, pool @POOL@. Written by scripts/maze_campaign/remote_fleet.py:
# change that, not this copy. One card per job. It checks the card, then runs the study's worker.py, which
# takes units from queue/pending one at a time until none is left or too little of the job is left.
# No speed test or benchmark is run.
RUN=@RUN@
PY=@PYTHON@
CODE=@CODE@
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
@CARD_GUARD@
echo "job $JOB on $(hostname), start $(date -Is), card class @GPU_CLASS@"

# Nothing to do: give the card back before starting Python.
if ! ls queue/pending/*.json > /dev/null 2>&1; then echo "nothing pending; this job ends"; exit 0; fi

# The environment must use the card, or every unit taken would fail.
"$PY" -c 'import jax; print("jax devices:", jax.devices()); assert jax.default_backend() == "gpu"' \
  || { echo "jax does not use the card; this job ends without taking a unit"; exit 3; }

exec "$PY" "$CODE/scripts/maze_campaign/worker.py" --run-dir "$RUN" --remote-records --minimum-seconds @UNIT_SECONDS@
'''


def prepare(run_dir, table, name):
    """Create a server's run folders and send its worker job scripts and the run's frozen code copy."""
    entry = table[name]
    # the server's files are built locally, laid out as in its run folder, so one copy sends them all
    tree = run_dir / "infrastructure/remote_fleet" / name
    for folder in REMOTE_FOLDERS:
        (tree / folder).mkdir(parents=True, exist_ok=True)
    for pool_name, pool in entry["pools"].items():
        (tree / f"worker_job_{pool_name}.sh").write_text(fleet.worker_script(name, entry, pool_name, pool))
    fleet.remote(run_dir, table, name,
                 f"mkdir -p {shlex.quote(entry['remote_run'])} {shlex.quote(entry['code'])}\n")
    fleet.rsync(run_dir, table, name, [f"{tree}/", fleet.remote_path(entry)])
    # the same frozen code copy as on UVA; nothing is deleted on the server
    platform, _ = own("submit").frozen_code(run_dir)
    fleet.rsync(run_dir, table, name,
                ["--exclude", "__pycache__", f"{platform}/", f"{entry['ssh'][-1]}:{entry['code']}/"])


def verify_unit(run_dir, unit):
    """Move a pulled unit's shard and log into the run's own folders and check the shard; None when it is complete."""
    folder = run_dir / "data/research" / unit
    if not (folder / "summary.json").exists():
        return "summary.json is missing"
    # the server's shard replaces any partial shard UVA still held for this unit
    for name, target in ((f"{unit}.jsonl", run_dir / "data"), (f"{unit}.log", run_dir / "logs")):
        if (folder / name).exists():
            target.mkdir(exist_ok=True)
            os.replace(folder / name, target / name)
    if not unit_queue.shard_progress(run_dir / "data" / f"{unit}.jsonl")["complete"]:
        return "the shard does not end with a unit_complete record"
    # a checkpoint UVA kept from an earlier attempt has no use now
    (run_dir / "checkpoints" / f"{unit}.npz").unlink(missing_ok=True)
    # the first real units on a remote card correct that card's copies-per-card estimate
    observed = common.read_json(folder / "summary.json").get("observed")
    if observed:
        common.append_json(OBSERVED / f"{observed['workload']}.jsonl", observed)
    return None


def adapt():
    """Give the grid campaign's remote fleet this study's worker job, preparation, folders and unit check."""
    fleet.WORKER_TEMPLATE = WORKER_TEMPLATE
    fleet.REMOTE_FOLDERS = REMOTE_FOLDERS
    fleet.prepare = prepare
    fleet.verify_unit = verify_unit
    # jobs of this study are told from the grid study's by their name
    fleet.JOB_PREFIX = "mazefleet"
    # a unit whose job ended continues from its checkpoint on the same server, so it may be put
    # back in that server's queue more often than a grid unit, which started again from zero
    fleet.RECOVERY_LIMIT = 20


def main():
    """Run one command of the grid campaign's remote fleet, adapted to this study."""
    arguments = sys.argv[1:]
    if "cancel-pending" in arguments:
        raise SystemExit("this file offers no cancel command; see cancel_ledger.py")
    # a reminder of the order of servers when units are sent to one of them
    if "stage" in arguments and "--server" in arguments and "--run-dir" in arguments:
        server = arguments[arguments.index("--server") + 1]
        table = common.read_json(fleet.table_path(Path(arguments[arguments.index("--run-dir") + 1]).resolve()))
        earlier = [name for name in SERVER_ORDER[:SERVER_ORDER.index(server)]
                   if table.get(name, {}).get("ready")] if server in SERVER_ORDER else []
        if earlier:
            print(f"order of servers: {', '.join(earlier)} come before {server} and are marked ready; "
                  f"use {server} only when they cannot start the work")
    adapt()
    fleet.main()


if __name__ == "__main__":
    main()
