"""The queue of units of the maze campaign: claims by atomic rename, returns, results, and progress read from shards.

A unit's queue record is one JSON file. It is in exactly one folder of RUN/queue at any time:
pending, running, done or failed (and remote_<server> or deferred while the remote fleet holds
it). Every change of state is one rename of that file. The unit's definition is a second file,
RUN/specifications/<unit id>.json, which never changes. Standard library only.
"""
import json
import os
from pathlib import Path
import re

from grid_campaign import common

STATES = ("pending", "running", "done", "failed")
FAILURE_LIMIT = 3     # a unit whose process failed this many times goes to failed
RECOVERY_LIMIT = 5    # a unit whose job ended without a result more often than this goes to failed
# what a claim adds to a unit's record; an ended attempt moves these fields into the record's history
CLAIM_FIELDS = ("job_id", "restart_count", "account", "hostname", "card", "card_memory_mb",
                "source_commit", "started_at")
TAIL_BYTES = 4 * 2 ** 20   # how much of a shard's end is read for its progress


def unit_path(run_dir, state, name):
    """Name the queue record of one unit in one queue folder."""
    return Path(run_dir) / "queue" / state / f"{name}.json"


def configuration_path(run_dir, name):
    """Name the configuration file run_training.py reads for one unit (the unit's cells are in it)."""
    return Path(run_dir) / "slurm/unit_configurations" / f"{name}.toml"


def enqueue(run_dir, specification, record, replace_pending=False):
    """Add one unit: its specification, its configuration file, then its pending record.

    Returns "added", or "" when the unit already exists and is left as it is. With replace_pending,
    a unit that is pending and never started gets the new specification ("replaced") and keeps its
    place in the order; a unit that started, finished or is held elsewhere is never replaced.
    """
    run_dir = Path(run_dir)
    for state in STATES:
        (run_dir / "queue" / state).mkdir(parents=True, exist_ok=True)
    with common.queue_lock(run_dir):
        # a unit is known by its name in ANY queue folder, the remote servers' and the held ones included
        folders = [folder for folder in (run_dir / "queue").iterdir() if folder.is_dir()]
        found = [folder / f"{record['name']}.json" for folder in folders if (folder / f"{record['name']}.json").exists()]
        outcome = "added"
        if found:
            pending = unit_path(run_dir, "pending", record["name"])
            if not (replace_pending and found == [pending] and common.read_json(pending)["attempts"] == 0):
                return ""
            record, outcome = dict(record, order=common.read_json(pending)["order"]), "replaced"
        # the pending record goes last: a worker may claim the unit the moment that record exists
        common.write_json(run_dir / record["specification"], specification)
        configuration = configuration_path(run_dir, record["name"])
        configuration.parent.mkdir(parents=True, exist_ok=True)
        configuration.write_text(specification["configuration_toml"])
        common.write_json(unit_path(run_dir, "pending", record["name"]), dict(record, created_at=common.now()))
    return outcome


def close_attempt(unit, outcome, **details):
    """Move the fields of a unit's open claim into its history, with how the attempt ended."""
    # before: {"name": "u", "job_id": "71", "hostname": "jaguar03", "started_at": "2026-...", "history": []}
    # after:  {"name": "u", "history": [{"job_id": "71", "hostname": "jaguar03", "started_at": "2026-...",
    #                                    "ended_at": "2026-...", "outcome": "stopped_with_state_saved"}]}
    entry = {field: unit.pop(field) for field in CLAIM_FIELDS if field in unit}
    entry.update(ended_at=common.now(), outcome=outcome, **details)
    unit.setdefault("history", []).append(entry)


def move(run_dir, unit, source, target):
    """Write a unit's record where it is, then rename it into another queue folder."""
    path = unit_path(run_dir, source, unit["name"])
    common.write_json(path, unit)
    destination = unit_path(run_dir, target, unit["name"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(path, destination)


def claim(run_dir, card_memory_mb, claimant):
    """Claim the first pending unit this card can hold; the rename into running/ is the claim."""
    run_dir = Path(run_dir)
    with common.queue_lock(run_dir):
        # units are taken in the order the builder gave them, then by name
        units = sorted((common.read_json(path) for path in (run_dir / "queue/pending").glob("*.json")),
                       key=lambda unit: (unit["order"], unit["name"]))
        for unit in units:
            if unit["minimum_device_memory_mb"] > card_memory_mb:
                continue
            os.rename(unit_path(run_dir, "pending", unit["name"]), unit_path(run_dir, "running", unit["name"]))
            # a unit put back by a remote server's tick still carries the claim of its ended job
            if "job_id" in unit:
                close_attempt(unit, "job_ended_without_a_result")
            unit.update(claimant, started_at=common.now(), attempts=unit["attempts"] + 1)
            common.write_json(unit_path(run_dir, "running", unit["name"]), unit)
            return unit
    return None


def next_folder(unit, counts_as_failure):
    """Say where a unit that did not complete goes: pending, or failed at its last allowed failure."""
    return "failed" if counts_as_failure and unit["failures"] + 1 >= FAILURE_LIMIT else "pending"


def give_back(run_dir, unit, outcome, counts_as_failure=False, minimum_device_memory_mb=None, **details):
    """Close a claimed unit's attempt and move it to pending, or to failed at its third failure."""
    run_dir = Path(run_dir)
    with common.queue_lock(run_dir):
        target = next_folder(unit, counts_as_failure)
        close_attempt(unit, outcome, **details)
        if counts_as_failure:
            unit["failures"] += 1
        # a unit that ran out of card memory is left to cards with more memory than the one it failed on
        if minimum_device_memory_mb is not None:
            unit["minimum_device_memory_mb"] = minimum_device_memory_mb
            unit["memory_failures"] = unit.get("memory_failures", 0) + 1
        if target == "failed":
            unit.update(failure_reason=f"its process failed {FAILURE_LIMIT} times", failure_log=details.get("log"),
                        finished_at=common.now())
        move(run_dir, unit, "running", target)
    common.append_json(run_dir / "infrastructure_events.jsonl", {
        "at": common.now(), "event": "unit_did_not_complete", "unit": unit["name"], "outcome": outcome,
        "destination": target, "log": details.get("log")})
    return target


def finish(run_dir, unit, result):
    """Record a complete unit's result and move it to done."""
    with common.queue_lock(run_dir):
        close_attempt(unit, "complete", wall_seconds=result.get("wall_seconds"), log=result.get("log"))
        unit.update(result=result, finished_at=common.now())
        move(run_dir, unit, "running", "done")


def release_claims_of_job(run_dir, job_id, restart_count):
    """Return to pending the units an earlier start of this same job claimed; returns their names."""
    run_dir, released = Path(run_dir), []
    with common.queue_lock(run_dir):
        for path in sorted((run_dir / "queue/running").glob("*.json")):
            unit = common.read_json(path)
            # the other workers of this start of the job (one per card) hold claims with the same count
            if unit.get("job_id") != job_id or unit.get("restart_count", 0) >= restart_count:
                continue
            close_attempt(unit, "job_started_again")
            move(run_dir, unit, "running", "pending")
            released.append(unit["name"])
    return released


def recover_claims(run_dir, live):
    """Return to pending every claimed unit whose Slurm job no longer runs; returns {unit name: folder}.

    `live` maps the job ids of this run's own ledgers that the scheduler still lists to their
    squeue rows (state in column 3), as common.own_live_jobs returns it. A claim made outside
    Slurm (a job id that is not a number) is not judged here.
    """
    run_dir, moved = Path(run_dir), {}
    with common.queue_lock(run_dir):
        for path in sorted((run_dir / "queue/running").glob("*.json")):
            unit = common.read_json(path)
            job_id = str(unit.get("job_id", ""))
            job = live.get(job_id)
            # a job that runs holds its unit; a job that waits again after a requeue does not; a
            # claim with no job id was cut off between its rename and its record
            if (job_id and not job_id.isdecimal()) or (job is not None and job[3] != "PENDING"):
                continue
            close_attempt(unit, "job_ended_without_a_result")
            unit["recoveries"] = unit.get("recoveries", 0) + 1
            target = "pending" if unit["recoveries"] <= RECOVERY_LIMIT else "failed"
            if target == "failed":
                unit.update(failure_reason=f"its job ended without a result more than {RECOVERY_LIMIT} times",
                            finished_at=common.now())
            move(run_dir, unit, "running", target)
            moved[unit["name"]] = target
            common.append_json(run_dir / "infrastructure_events.jsonl", {
                "at": common.now(), "event": "own_job_ended_with_unfinished_unit", "unit": unit["name"],
                "job_id": job_id, "destination": target})
    return moved


def counts(run_dir):
    """Count the units of every queue folder, the remote servers' and the held ones included."""
    # after: {"pending": 3, "running": 2, "done": 14, "failed": 0, "remote_mit": 4}
    folders = sorted(folder for folder in (Path(run_dir) / "queue").iterdir() if folder.is_dir())
    return {folder.name: len(list(folder.glob("*.json"))) for folder in folders}


def read_units(run_dir, state):
    """Read every queue record of one folder under the queue lock, so no record is caught mid-move."""
    with common.queue_lock(run_dir):
        return [common.read_json(path) for path in sorted((Path(run_dir) / "queue" / state).glob("*.json"))]


def shard_progress(path):
    """Read from the end of a unit's shard whether it is complete and how many windows it holds.

    before (the last lines of data/<unit>.jsonl):
        {"record": "episode_window", "unit_id": "u", "window_index": 41, "first_iteration": 2563,
         "last_iteration": 2624, ...}
        {"record": "unit_complete", "unit_id": "u", "windows": 42, ...}
    after:  {"exists": True, "complete": True, "windows_written": 42, "last_iteration": 2624, "bytes": 912345,
             "changed_at": 1794285410.0, "completion": {the unit_complete record}}
    `windows_written` is None when the last TAIL_BYTES of a larger shard hold no whole window line.
    """
    path = Path(path)
    if not path.exists():
        return {"exists": False, "complete": False, "windows_written": 0, "last_iteration": 0, "bytes": 0,
                "changed_at": None, "completion": None}
    status = path.stat()
    with path.open("rb") as handle:
        handle.seek(max(0, status.st_size - TAIL_BYTES))
        tail = handle.read()
    # only whole lines count: the text after the last newline is a line still being written, and
    # the first line of a tail cut out of a larger file has lost its beginning
    lines = tail.split(b"\n")[:-1]
    whole_file = status.st_size <= TAIL_BYTES
    if not whole_file:
        lines = lines[1:]
    progress = {"exists": True, "complete": False, "windows_written": 0 if whole_file else None,
                "last_iteration": 0 if whole_file else None, "bytes": status.st_size,
                "changed_at": status.st_mtime, "completion": None}
    # the unit_complete record is short and is the last line run_training.py ever writes
    if lines and len(lines[-1]) < 65536:
        last = json.loads(lines[-1])
        if last.get("record") == "unit_complete":
            progress.update(complete=True, completion=last)
    # a window line is long (one number per policy and measure); its counters stand at its beginning
    for line in reversed(lines):
        head = line[:2048]
        if b'"record": "episode_window"' in head:
            progress["windows_written"] = int(re.search(rb'"window_index": (\d+)', head).group(1)) + 1
            progress["last_iteration"] = int(re.search(rb'"last_iteration": (\d+)', head).group(1))
            break
    return progress
