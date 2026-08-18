"""What the 20-minute tick reports: this run's own job states, progress and estimate against actual.

Jobs are read ONLY from `slurm/submitted_jobids.txt`, never by enumerating a user, a state or a job
name — the uid is shared with other sessions and with jobs started by hand.

Two views:
  --table   one line per chunk: job, node, state, windows written, per cent done, projected finish
  --tick    one compact line for a monitoring notification, plus a line per chunk that failed or
            whose canary rate is far from the plan

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/status.py --table
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RUN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import plan_submission as ps  # noqa: E402  (the run's shape lives there)

PACIFIC = ZoneInfo("America/Los_Angeles")
WINDOWS = ps.ITERATIONS // ps.WINDOW_ITERATIONS
# a window record's own iteration marker, matched against the raw bytes of a shard's tail
LAST_ITERATION = re.compile(rb'"last_iteration": (\d+)')
# a canary this far from its planned seconds per iteration is worth a line of its own in the tick
CANARY_TOLERANCE = 0.25


def own_job_ids() -> list:
    """This run's job ids, in submission order, from the run's own id file."""
    path = RUN_DIR / "slurm" / "submitted_jobids.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip().isdigit()]


def job_states() -> dict:
    """State, node and elapsed time of every id in the run's id file, keyed by id."""
    ids = own_job_ids()
    output = subprocess.run(
        ["sacct", "-j", ",".join(ids), "-X", "--format=JobID,JobName%40,State,NodeList,Elapsed",
         "-P", "-n"], capture_output=True, text=True, check=True).stdout
    states = {}
    for line in output.splitlines():
        job, name, state, node, elapsed = line.split("|")
        states[job] = {"name": name, "state": state.split()[0], "node": node, "elapsed": elapsed}
    return states


def chunk_of_job(name: str) -> str:
    """The unit and chunk a job name carries, e.g. "unit-1-chunk-7-of-32"; "" for a probe."""
    match = re.match(r"pmjax2-(unit-\d+-chunk-\d+-of-\d+)$", name)
    return match.group(1) if match else ""


# how much of a shard's tail to read to find its last complete record. One window line of the
# widest chunk is about 25 kB, so half a megabyte always contains several.
TAIL_BYTES = 512 * 1024


def windows_written(shard: Path) -> int:
    """How many windows a shard has written, from its LAST record rather than by counting lines.

    A shard of this run grows to about 240 MB, and forty of them are read at every tick, so
    counting lines would mean reading 4 GB every twenty minutes. The last window record carries
    the iteration it ends at, and windows are a fixed 200 iterations, so the count follows from it.

    before: a shard whose last complete line ends at iteration 61,600;
    after:  308 windows.
    """
    if not shard.exists() or shard.stat().st_size == 0:
        return 0
    with open(shard, "rb") as handle:
        handle.seek(max(0, shard.stat().st_size - TAIL_BYTES))
        lines = handle.read().split(b"\n")
    for line in reversed(lines):
        if b'"record": "episode_window"' in line:
            return int(LAST_ITERATION.search(line).group(1)) // ps.WINDOW_ITERATIONS
    return 0


def queue_entries() -> dict:
    """Every queue entry with the state its folder says it is in, keyed by unit id."""
    entries = {}
    for state in ("pending", "running", "done", "failed"):
        for path in (RUN_DIR / "queue" / state).glob("*.json"):
            entries[path.stem] = dict(json.loads(path.read_text()), queue_state=state)
    return entries


def canary_rate(unit_id: str) -> float:
    """The steady seconds per iteration this chunk's canary measured, or None if it has not run."""
    shard = RUN_DIR / "canary" / "data" / f"{unit_id}.jsonl"
    if not shard.exists():
        return None
    for line in reversed(shard.read_text().splitlines()):
        record = json.loads(line)
        if record.get("record") == "unit_complete":
            return record["seconds_per_iteration_steady"]
    return None


def rows() -> list:
    """One row per chunk: its queue entry, its job, its progress and its canary rate."""
    states = job_states()
    by_chunk = {}
    for job, entry in states.items():
        chunk = chunk_of_job(entry["name"])
        if chunk:
            # a chunk resubmitted after a failure has two ids; the later one is the live job
            by_chunk[chunk] = max(by_chunk.get(chunk, ("0", None))[0], job), entry
    result = []
    for unit_id, record in sorted(queue_entries().items(),
                                  key=lambda item: (item[1]["order"], item[1]["chunk"])):
        key = f"unit-{record['order']}-chunk-{record['chunk'] + 1}-of-{record['chunks']}"
        job, entry = by_chunk.get(key, (None, None))
        written = windows_written(RUN_DIR / "data" / f"{unit_id}.jsonl")
        measured = canary_rate(unit_id)
        planned_seconds_per_iteration = record["planned_seconds"] / ps.ITERATIONS
        result.append({
            "chunk": key, "unit_id": unit_id, "job": job,
            "state": entry["state"] if entry else "not submitted",
            "node": entry["node"] if entry else record["planned_node"],
            "elapsed": entry["elapsed"] if entry else "",
            "queue_state": record["queue_state"], "windows": written,
            "percent": 100.0 * written / WINDOWS,
            "planned_hours": record["planned_seconds"] / 3600,
            "canary_seconds_per_iteration": measured,
            "canary_projected_hours": (measured * ps.ITERATIONS / 3600 if measured else None),
            "planned_seconds_per_iteration": planned_seconds_per_iteration,
        })
    return result


def print_table(result: list) -> None:
    """One line per chunk, in unit and chunk order."""
    print(f"{'chunk':22} {'job':>9} {'node':>11} {'state':>10} {'queue':>8} {'windows':>9} "
          f"{'done':>6} {'plan h':>7} {'canary h':>9}")
    for row in result:
        canary = f"{row['canary_projected_hours']:.2f}" if row["canary_projected_hours"] else "-"
        print(f"{row['chunk']:22} {str(row['job']):>9} {row['node']:>11} {row['state']:>10} "
              f"{row['queue_state']:>8} {row['windows']:>9} {row['percent']:>5.1f}% "
              f"{row['planned_hours']:>7.2f} {canary:>9}")


def print_tick(result: list) -> None:
    """One compact line for a notification, plus a line per chunk that needs attention."""
    counts = {}
    for row in result:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    written = sum(row["windows"] for row in result)
    total = WINDOWS * len(result)
    now = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")
    state_text = " ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    print(f"[{now}] {state_text} | {written}/{total} windows ({100.0 * written / total:.1f}%)")
    for row in result:
        if row["state"] in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"):
            print(f"  {row['state']}: {row['chunk']} job {row['job']} on {row['node']}")
        measured = row["canary_seconds_per_iteration"]
        if measured is None:
            continue
        planned = row["planned_seconds_per_iteration"]
        if abs(measured - planned) / planned > CANARY_TOLERANCE:
            print(f"  canary off plan: {row['chunk']} on {row['node']} measured "
                  f"{row['canary_projected_hours']:.2f} h against a planned "
                  f"{row['planned_hours']:.2f} h")


def main() -> None:
    """Print whichever view was asked for."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--tick", action="store_true")
    args = parser.parse_args()
    result = rows()
    if args.table:
        print_table(result)
    if args.tick or not args.table:
        print_tick(result)


if __name__ == "__main__":
    main()
