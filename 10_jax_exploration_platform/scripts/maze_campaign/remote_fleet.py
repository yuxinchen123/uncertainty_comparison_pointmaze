"""Run units of the maze campaign on remote servers: Purdue Gautschi and Gilbreth, MIT, TAMU ACES, NCSA DeltaAI.

    python remote_fleet.py --run-dir RUN prepare --server S [--again]
    python remote_fleet.py --run-dir RUN hand-over-tick [--dry-run]      # the one command of a monitoring tick
    python remote_fleet.py --run-dir RUN stage --server S --units N [--match TEXT] [--from-end] [--dry-run]
    python remote_fleet.py --run-dir RUN submit --server S --jobs K [--pool P] [--dry-run]
    python remote_fleet.py --run-dir RUN tick [--dry-run]
    python remote_fleet.py --run-dir RUN status
    python remote_fleet.py --run-dir RUN take-back --server S --units N [--match TEXT] [--from-end] [--hold]

`stage`, `submit`, `tick`, `status` and `take-back` are the commands of
`scripts/grid_initialization_campaign/remote_fleet.py`: the server table
RUN/infrastructure/remote_fleet_servers.json, the queue folders on a server, one status call per
server, no new login, and units that no job claimed within two hours going back to UVA are all
described at the head of that file. This file replaces these parts of it:

  1. The worker job on a server starts this folder's worker.py (the frozen copy) with
     --remote-records, which writes the claim records and the completed-unit folders the tick reads.
  2. `prepare` sends the job scripts and the run's frozen code copy, and makes the folder above the
     copy a git repository there, because run_training.py records the commit of that folder and
     stops where git cannot read one. There is no first-unit comparison of arrays with UVA.
  3. A pulled unit is checked by its shard: a unit_complete record and the expected number of
     windows. The first complete unit of a server is also compared in format (record kinds, keys,
     kinds of values) with a unit trained on UVA; until that has passed the server holds one unit.
     A verified shard and log move to RUN/data and RUN/logs, and the unit's line for the
     copies-per-card skill is added on UVA.
  4. A unit that returns unfinished brings its checkpoint and its shard when the server has both,
     and continues on UVA; otherwise it starts again. While it stays on a server it continues from
     its checkpoint there, in a later job.
  5. `hand-over-tick` does a whole tick for every server whose table entry has `hand_over`: read
     the server once, pull and verify finished units, move pending units from UVA to the server up
     to its limit, submit worker jobs for units no job will take, and print one line per unit.

Job ids go to RUN/slurm/submitted_jobids_<server>.txt (one file per server). A name with a phase
in it (submitted_jobids_<phase>_<server>.txt) would be read by UVA's monitor as UVA job ids.
H100 and H200 work goes to the first server of SERVER_ORDER that can start it (rule
h100-h200-server-order.md). This file offers no cancel command (see cancel_ledger.py).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

from grid_campaign import common, load
import unit_queue

fleet = load("remote_fleet")
SERVER_ORDER = ("mit", "gautschi", "aces", "deltaai")
OBSERVED = common.PROJECT / "10_jax_exploration_platform/device/observed"
REMOTE_FOLDERS = tuple(f"queue/{state}" for state in fleet.REMOTE_STATES) + (
    "specifications", "data/research", "checkpoints", "logs", "slurm/jobs")
# how much of a shard's end the status call reads on the server to find the last window
PROGRESS_BYTES = 4 * 2 ** 20
# the record kinds every complete shard holds; their keys are compared with a UVA unit's
RECORD_KINDS = ("job", "unit_start", "episode_window", "unit_complete")
# the grid campaign's own functions, kept once per process before adapt() replaces them
if not hasattr(fleet, "grid_functions"):
    fleet.grid_functions = {"status_script": fleet.status_script, "parse_status": fleet.parse_status,
                            "result_block": fleet.result_block}
# the server table the running command holds, so a check made deep inside a tick changes that copy
CURRENT = {"table": None}

WORKER_TEMPLATE = r'''#!/bin/bash
# Worker job of the maze study on @SERVER@, pool @POOL@. Written by scripts/maze_campaign/remote_fleet.py:
# change that, not this copy. One card per job. It checks the card, then runs the study's worker.py, which
# takes units from queue/pending one at a time until none is left or too little of the job is left.
# No speed test or benchmark is run.
RUN=@RUN@
PY=@PYTHON@
CODE=@CODE@
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
@CARD_GUARD@
echo "job $JOB on $(hostname), start $(date -Is), card class @GPU_CLASS@"

# Nothing to do: give the card back before starting Python.
if ! ls queue/pending/*.json > /dev/null 2>&1; then echo "nothing pending; this job ends"; exit 0; fi

# run_training.py records the commit of the folder above the code copy and stops where git cannot read it.
git -C "$CODE/.." rev-parse HEAD > /dev/null \
  || { echo "git cannot read a commit above the code copy; this job ends without taking a unit"; exit 5; }

# The environment must use the card and hold every library the trainer imports at its start, or every unit taken
# would fail: the modules run_training.py imports are imported here first, before a unit is claimed.
PYTHONPATH="$CODE/src:$CODE/scripts" "$PY" -c 'import jax, experiment_configuration
from exploration_platform.training import checkpoint, episode_log, sweep
from exploration_platform.training.runner import Runner
from exploration_platform.agents.ppo import config, presets
from exploration_platform.bonuses import registry
from exploration_platform.envs.antmaze import am_common
from exploration_platform.envs.pointmaze import pm_common
print("jax devices:", jax.devices()); assert jax.default_backend() == "gpu"' \
  || { echo "jax does not use the card, or a module the trainer imports is missing; this job ends without taking a unit"; exit 3; }

exec "$PY" "$CODE/scripts/maze_campaign/worker.py" --run-dir "$RUN" --remote-records --minimum-seconds @UNIT_SECONDS@
'''


def repository_script(code_parent, commit):
    """Write the bash script that makes the folder above a server's code copy a git repository, once."""
    message = (f"Copy of the frozen code of UVA commit {commit} "
               "(10_jax_exploration_platform: src, scripts, configs, SOURCE_COMMIT)")
    return "\n".join([
        f"cd {shlex.quote(code_parent)} || exit 3",
        "if [ ! -d .git ]; then",
        "  git init -q . || exit 4",
        "  echo '__pycache__/' >> .git/info/exclude",
        "  git add -A . && git -c user.name='maze remote fleet' -c user.email='remote-fleet@localhost' "
        f"commit -q -m {shlex.quote(message)} && git tag copy-of-uva-commit-{commit} || exit 4",
        "fi",
        f'echo "{fleet.MARK}repository $(git rev-parse HEAD) $(git status --porcelain | wc -l)"',
        f'echo "{fleet.MARK}end"', "exit 0"]) + "\n"


def prepare(run_dir, table, name, again=False):
    """Create a server's run folders and send its worker job scripts and the run's frozen code copy."""
    entry = table[name]
    source = common.read_json(run_dir / "source.json")
    commit = source["commit"]
    # a server that already holds this code copy and its job scripts is left as it is
    if entry.get("prepared_commit") == commit and not again:
        return
    code_parent = f"{entry['code_root']}/source_{commit}"
    entry["code"] = f"{code_parent}/10_jax_exploration_platform"
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
    fleet.rsync(run_dir, table, name, ["--exclude", "__pycache__",
                                       f"{source['snapshot']}/10_jax_exploration_platform/",
                                       f"{entry['ssh'][-1]}:{entry['code']}/"])
    # before: "@@fleet:repository 3f2a...e1 0"; after: the server's own commit of the copy, 0 changed files
    answer = fleet.remote(run_dir, table, name, repository_script(code_parent, commit))
    remote_commit, changed = next(line for line in answer.splitlines()
                                  if line.startswith(f"{fleet.MARK}repository ")).split()[1:]
    entry.update(prepared_commit=commit, code_repository_commit=remote_commit)
    common.write_json(fleet.table_path(run_dir), table)
    fleet.log(run_dir, f"{name} | prepare | job scripts and the code copy of commit {commit} are on the server; "
              f"the copy's own git commit there is {remote_commit} ({changed} file(s) differ from it)")


def status_script(name, entry, assigned):
    """The grid campaign's status script with one more section: how far each assigned unit's shard has come."""
    text = fleet.grid_functions["status_script"](name, entry, assigned)
    end = f'echo; echo "{fleet.MARK}end"\nexit 0\n'
    if not text.endswith(end):
        raise RuntimeError("the grid campaign's status script no longer ends with its end marker")
    lines = [f'echo; echo "{fleet.MARK}progress"']
    if assigned:
        # a window line begins with its counters; the last one in the shard's end is the last window written
        lines += [f'for unit in {" ".join(shlex.quote(unit) for unit in assigned)}; do',
                  '  shard="data/$unit.jsonl"; [ -e "$shard" ] || continue',
                  f'  window=$(tail -c {PROGRESS_BYTES} "$shard" | grep -a -o \'"window_index": [0-9]*\' | tail -n 1)',
                  '  tail -c 4096 "$shard" | grep -a -q \'"record": "unit_complete"\' && complete=yes || complete=no',
                  '  [ -e "checkpoints/$unit.npz" ] && saved=yes || saved=no',
                  '  echo "$unit window=${window##* } complete=$complete checkpoint=$saved"', 'done']
    # the scheduler's start estimate of each waiting job, as seconds from now on the server's own clock
    lines += [f'echo; echo "{fleet.MARK}estimates"',
              'squeue -h -u "$(id -un)" -t PENDING -o "%i %S" | while read -r job start; do',
              '  case "$start" in N/A|"") echo "$job unknown" ;; '
              '*) echo "$job $(( $(date -d "$start" +%s) - $(date +%s) ))" ;; esac', 'done']
    return text[:-len(end)] + "\n".join(lines) + "\n" + end


def parse_status(text):
    """Read one status answer as the grid campaign does, and each assigned unit's progress beside it."""
    # before: "@@fleet:progress\nunit-a window=41 complete=no checkpoint=yes\nunit-b window= complete=no checkpoint=no"
    # after:  status["progress"] = {"unit-a": {"windows_written": 42, "complete": False, "checkpoint": True},
    #                               "unit-b": {"windows_written": 0, "complete": False, "checkpoint": False}}
    # before: "@@fleet:estimates\n18433051 207000\n18433052 unknown"
    # after:  status["start_in_seconds"] = {"18433051": 207000, "18433052": None}
    status = fleet.grid_functions["parse_status"](text)
    status["progress"], status["start_in_seconds"], section = {}, {}, None
    for line in text.splitlines():
        if line.startswith(fleet.MARK):
            section = line[len(fleet.MARK):].strip()
        elif section == "progress" and line.strip():
            unit, *fields = line.split()
            values = dict(field.split("=", 1) for field in fields)
            status["progress"][unit] = {"windows_written": int(values["window"]) + 1 if values["window"] else 0,
                                        "complete": values["complete"] == "yes",
                                        "checkpoint": values["checkpoint"] == "yes"}
        elif section == "estimates" and line.strip():
            job, seconds = line.split()
            status["start_in_seconds"][job] = None if seconds == "unknown" else int(seconds)
    return status


def result_block(name, claim, summary=None):
    """The grid campaign's result of a remote unit, with the unit's windows and its observed line."""
    block = fleet.grid_functions["result_block"](name, claim, summary)
    if summary is not None:
        block.update(windows_written=summary.get("windows"), observed=summary.get("observed"))
    return block


def value_kind(value, copies):
    """Name the kind of one JSON value of a shard record."""
    # before: [0.1, 0.2] in a unit of 2 copies; {"gamma": 0.99, "n_envs": 4}; None
    # after:  "list, one entry per copy";       "object with keys gamma, n_envs"; "null"
    if isinstance(value, dict):
        return "object with keys " + ", ".join(sorted(value))
    if isinstance(value, list):
        return "list, one entry per copy" if len(value) == copies else "list"
    return {type(None): "null", bool: "true or false", int: "whole number", float: "number", str: "text"}[type(value)]


def shard_format(path):
    """Describe a shard's format: per record kind, how many records, their keys and the kind of each value."""
    # before (two lines of the shard of a unit with 2 copies):
    #   {"record": "unit_start", "unit_id": "u", "copies": 2}
    #   {"record": "episode_window", "window_index": 0, "loss": 0.5, "coverage_per_copy": [0.1, 0.2]}
    # after: {"unit_start": {"count": 1, "key_sets": 1, "keys": {"record": "text", "unit_id": "text",
    #                                                            "copies": "whole number"}},
    #         "episode_window": {"count": 1, "key_sets": 1, "keys": {"record": "text", "window_index": "whole number",
    #                            "loss": "number", "coverage_per_copy": "list, one entry per copy"}}}
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    copies = next(record["copies"] for record in records if record["record"] == "unit_start")
    kinds = {}
    for record in records:
        kind = kinds.setdefault(record["record"], {"count": 0, "key_sets": set(), "keys": None})
        kind["count"] += 1
        kind["key_sets"].add(frozenset(record))
        # the kinds of values are read from the first record of each kind
        if kind["keys"] is None:
            kind["keys"] = {key: value_kind(value, copies) for key, value in record.items()}
    return {name: dict(kind, key_sets=len(kind["key_sets"])) for name, kind in kinds.items()}


def format_differences(reference, remote):
    """List how a remote unit's shard differs in format from a UVA unit's; an empty list when it does not."""
    differences = []
    for kind in RECORD_KINDS:
        if kind not in remote or kind not in reference:
            differences.append(f"{kind}: " + ("missing in the remote shard" if kind in reference else
                                              "missing in the UVA shard"))
            continue
        ours, theirs = reference[kind], remote[kind]
        # one unit_start and one unit_complete per unit; every record of a kind has the same keys
        if kind in ("unit_start", "unit_complete") and theirs["count"] != 1:
            differences.append(f"{kind}: {theirs['count']} records in the remote shard, expected 1")
        if theirs["key_sets"] != ours["key_sets"]:
            differences.append(f"{kind}: {theirs['key_sets']} different key sets in the remote shard, "
                               f"{ours['key_sets']} in the UVA shard")
        for key in sorted(set(ours["keys"]) | set(theirs["keys"])):
            if ours["keys"].get(key) != theirs["keys"].get(key):
                differences.append(f"{kind}.{key}: UVA has {ours['keys'].get(key, 'no such key')}, "
                                   f"the remote shard has {theirs['keys'].get(key, 'no such key')}")
    return differences


def uva_reference(done, unit):
    """Name a complete unit trained on UVA with a unit's workload and reward setting; None when there is none."""
    # before: done records [{"name": "a", "workload": "w", "reward_setting": "r", "method": "m1", "result": {...}},
    #                       {"name": "b", "workload": "w", "reward_setting": "r", "method": "m2",
    #                        "result": {"server": "gautschi"}}], unit {"workload": "w", "reward_setting": "r", ...}
    # after:  "a" (unit b was trained on a remote server, so it cannot stand for UVA's format)
    matching = [record for record in done if "server" not in record.get("result", {})
                and record["workload"] == unit["workload"] and record["reward_setting"] == unit["reward_setting"]]
    # the same method first, then the unit that finished last
    matching.sort(key=lambda record: (record["method"] == unit["method"], record.get("finished_at", "")))
    return matching[-1]["name"] if matching else None


def first_unit_problem(run_dir, unit, shard):
    """Check the format of a server's first complete unit against a UVA unit; None when the server may go on."""
    server = fleet.load_assignment(run_dir).get(unit, {}).get("server")
    # a unit no server was given (records copied by hand) has no server to check
    if server is None:
        return None
    table = CURRENT["table"] or common.read_json(fleet.table_path(run_dir))
    if table[server].get("first_unit", {}).get("state") == "passed":
        return None
    specification = common.read_json(run_dir / "specifications" / f"{unit}.json")
    reference = uva_reference(unit_queue.read_units(run_dir, "done"), specification)
    # with no UVA unit of this kind yet the unit waits where it is and is checked again at a later tick
    if reference is None:
        return "no unit trained on UVA with the same workload and reward setting is done yet to compare its format with"
    differences = format_differences(shard_format(run_dir / "data" / f"{reference}.jsonl"), shard_format(shard))
    table[server]["first_unit"] = {"state": "failed" if differences else "passed", "unit": unit,
                                   "compared_with_uva_unit": reference, "differences": differences,
                                   "checked_at": common.now()}
    common.write_json(fleet.table_path(run_dir), table)
    fleet.log(run_dir, f"{server} | first unit | format of {unit} against UVA's {reference}: "
              + (f"{len(differences)} difference(s): {'; '.join(differences[:5])}" if differences else
                 "the same record kinds, keys and kinds of values"))
    return f"its records differ in format from UVA's ({'; '.join(differences[:3])})" if differences else None


def verify_unit(run_dir, unit):
    """Check a pulled unit's shard, then move shard and log into the run's own folders; None when it is complete."""
    folder = run_dir / "data/research" / unit
    if not (folder / "summary.json").exists():
        return "summary.json is missing"
    # a unit verified at an earlier tick whose queue move did not happen has its shard in RUN/data already
    pulled = folder / f"{unit}.jsonl"
    shard = pulled if pulled.exists() else run_dir / "data" / f"{unit}.jsonl"
    progress = unit_queue.shard_progress(shard)
    if not progress["complete"]:
        return "the shard does not end with a unit_complete record"
    expected = common.read_json(run_dir / "specifications" / f"{unit}.json")["windows_expected"]
    if progress["windows_written"] != expected or progress["completion"]["windows"] != expected:
        return (f"the shard's last window is number {progress['windows_written']} and its unit_complete record "
                f"says {progress['completion']['windows']} windows; {expected} are expected")
    problem = first_unit_problem(run_dir, unit, shard)
    if problem:
        return problem
    # only a checked shard replaces what UVA held for this unit (a partial shard of an earlier attempt)
    for name, target in ((f"{unit}.jsonl", run_dir / "data"), (f"{unit}.log", run_dir / "logs")):
        if (folder / name).exists():
            target.mkdir(exist_ok=True)
            os.replace(folder / name, target / name)
    # a checkpoint UVA kept from an earlier attempt has no use now
    (run_dir / "checkpoints" / f"{unit}.npz").unlink(missing_ok=True)
    # one line per finished unit for the copies-per-card skill: the speed this card type gave
    observed = common.read_json(folder / "summary.json").get("observed")
    if observed:
        common.append_json(OBSERVED / f"{observed['workload']}.jsonl", observed)
    return None


def bring_back(run_dir, table, name, returned, command, folder="pending"):
    """Put units a server gave up back in a UVA queue folder, each with its saved state when the server has one."""
    entry = table[name]
    for unit, _ in returned:
        # the server's record of the unit, its checkpoint and its shard land in a folder of their own first
        landing = run_dir / "slurm/remote_returned" / name / unit
        landing.mkdir(parents=True, exist_ok=True)
        fleet.rsync(run_dir, table, name, [
            "--ignore-missing-args", fleet.remote_path(entry, f"queue/withdrawn/{unit}.json"),
            f":{entry['remote_run']}/checkpoints/{unit}.npz", f":{entry['remote_run']}/data/{unit}.jsonl",
            f"{landing}/"])
        # checkpoint and shard belong together: both replace UVA's, or neither does and UVA keeps what it had
        checkpoint, shard = landing / f"{unit}.npz", landing / f"{unit}.jsonl"
        saved = checkpoint.exists() and shard.exists()
        if saved:
            for source, target in ((shard, run_dir / "data"), (checkpoint, run_dir / "checkpoints")):
                target.mkdir(exist_ok=True)
                os.replace(source, target / source.name)
        # the server's record holds the unit's attempts there; UVA's record takes them over
        record = landing / f"{unit}.json"
        fleet.move_unit(run_dir, unit, f"remote_{name}", folder, common.read_json(record) if record.exists() else None)
        fleet.finish_assignment(run_dir, [unit], "returned")
        fleet.log(run_dir, f"{name} | {command} | {unit} is back in queue/{folder} on UVA, "
                  + ("with its checkpoint and shard: it continues" if saved else
                     "without a checkpoint from the server: it starts again or continues from what UVA held"))


def hand_over(run_dir, table, name, dry_run):
    """Move pending units from UVA to a server, up to the server's limit; returns the names handed over."""
    entry, settings = table[name], table[name]["hand_over"]
    held = run_dir / "queue" / f"remote_{name}"
    # the server's first unit is its check: until that unit is verified the server holds one unit,
    # and after a failed check it gets none
    first = entry.get("first_unit", {}).get("state")
    limit = {"passed": settings["units_at_once"], "failed": 0}.get(first, 1)
    # the done records are read before the queue lock is taken, because reading them takes it too
    done = unit_queue.read_units(run_dir, "done")
    with common.queue_lock(run_dir):
        # UVA's workers claim from the front of the order, so a server takes from its end; before
        # the first unit is verified only a unit whose format can be compared with a UVA unit goes
        pending = sorted((common.read_json(path) for path in (run_dir / "queue/pending").glob("*.json")),
                         key=lambda unit: (unit["order"], unit["name"]), reverse=True)
        candidates = [unit["name"] for unit in pending if settings["match"] in unit["name"]
                      and (first == "passed" or uva_reference(done, unit))]
        chosen = candidates[:max(0, limit - len(fleet.unit_names(held)))]
        if chosen and not dry_run:
            # the rename out of pending is the hand-over: no UVA worker can claim the unit after it
            held.mkdir(exist_ok=True)
            for unit in chosen:
                os.rename(run_dir / "queue/pending" / f"{unit}.json", held / f"{unit}.json")
            assignment = fleet.load_assignment(run_dir)
            assignment.update({unit: {"server": name, "staged_at": common.now(), "origin": "pending",
                                      "recoveries": 0, "state": "staged"} for unit in chosen})
            common.write_json(fleet.assignment_path(run_dir), assignment)
    fleet.log(run_dir, f"{name} | hand-over | limit {limit} unit(s) at once (first unit: {first or 'not checked yet'}), "
              f"{len(fleet.unit_names(held)) - (0 if dry_run else len(chosen))} held before, {len(pending)} pending on UVA: "
              f"{len(chosen)} unit(s) {'would be' if dry_run else ''} handed over", dry_run)
    for unit in chosen:
        fleet.log(run_dir, f"{name} | hand-over | {unit}", dry_run)
    if not chosen or dry_run:
        return chosen
    # one copy sends everything: specification, queue record (to incoming/, out of the workers' sight) and,
    # for a unit UVA began earlier, its checkpoint with its shard, so it continues on the server
    stamp = int(time.time())
    tree = run_dir / "slurm/remote_hand_over" / f"{name}_{stamp}"
    for unit in chosen:
        files = [(run_dir / "specifications" / f"{unit}.json", "specifications"),
                 (held / f"{unit}.json", "queue/incoming")]
        checkpoint, shard = run_dir / "checkpoints" / f"{unit}.npz", run_dir / "data" / f"{unit}.jsonl"
        if checkpoint.exists() and shard.exists():
            files += [(checkpoint, "checkpoints"), (shard, "data")]
        for source, folder in files:
            (tree / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, tree / folder / source.name)
    try:
        fleet.rsync(run_dir, table, name, [f"{tree}/", fleet.remote_path(entry)])
    except (fleet.ServerUnreachable, fleet.RemoteCommandFailed) as error:
        # nothing reached the server's pending folder: the units go back where they came from
        with common.queue_lock(run_dir):
            for unit in chosen:
                os.rename(held / f"{unit}.json", run_dir / "queue/pending" / f"{unit}.json")
            assignment = fleet.load_assignment(run_dir)
            for unit in chosen:
                del assignment[unit]
            common.write_json(fleet.assignment_path(run_dir), assignment)
        fleet.log(run_dir, f"{name} | hand-over | FAILED to copy, {len(chosen)} unit(s) moved back on UVA: {error}")
        raise
    finally:
        shutil.rmtree(tree)
    # one call makes the units visible to the workers; if it fails the next tick finishes the move
    moved = fleet.parse_moves(fleet.remote(run_dir, table, name, fleet.move_script(
        entry, {"place": chosen, "recover": [], "pull_back": [], "withdraw": []}, stamp)))
    fleet.log(run_dir, f"{name} | hand-over | {len(moved)} of {len(chosen)} unit(s) now pending on the server")
    return chosen


def start_expected(entry, live, estimates):
    """Say whether units may wait on a server: a job of ours runs there, or one is expected to start soon."""
    # before: limit 7200 s; jobs {"12": PENDING, estimated to start in 207000 s}; after: False
    # before: the same with job "13" RUNNING beside it, or with no estimate yet for job "12"; after: True
    limit = entry["hand_over"].get("start_within_seconds")
    waiting = [job for job, row in live.items() if row["state"] != "RUNNING"]
    # no such rule for this server, a job that runs, or no job yet (the first hand-over submits one)
    if limit is None or not waiting or len(waiting) < len(live):
        return True
    # a job the scheduler has not estimated yet counts as starting soon
    known = [estimates[job] for job in waiting if estimates.get(job) is not None]
    return len(known) < len(waiting) or min(known) <= limit


def units_without_job(name, status, assigned, ledger):
    """Count, by one status reading, the assigned units that wait on a server or whose job is gone."""
    # before: unit-a pending on the server; unit-b running under job 12, which the scheduler no longer lists;
    #         unit-c running under job 13, which runs
    # after:  2
    live = fleet.fleet_jobs(name, status, ledger)
    waiting = set(assigned) & (set(status["queue"]["pending"]) | set(status["queue"]["incoming"]))
    orphaned = [unit for unit in set(assigned) & set(status["queue"]["running"])
                if live.get(status["claims"].get(("running", unit), {}).get("job"), {}).get("state") != "RUNNING"]
    return len(waiting) + len(orphaned)


def unit_lines(run_dir, name, status, assigned, handed):
    """One line per unit a server holds: cluster, job id, card, state there, windows written of windows expected."""
    # after: "gautschi  job 18420011   NVIDIA L40S   running    120 of 256 windows  on UVA: queue/remote_gautschi  <unit>"
    lines = []
    for unit in list(assigned) + [unit for unit in handed if unit not in assigned]:
        state = next((folder for folder, units in status["queue"].items() if unit in units),
                     "handed over at this tick" if unit in handed else "not on the server")
        claim = status["claims"].get((state, unit), {})
        expected = common.read_json(run_dir / "specifications" / f"{unit}.json")["windows_expected"]
        written = status["progress"].get(unit, {}).get("windows_written", 0)
        on_uva = next((folder.name for folder in sorted((run_dir / "queue").iterdir())
                       if folder.is_dir() and (folder / f"{unit}.json").exists()), "no queue folder")
        lines.append(f"{name:<9} job {claim.get('job', '-'):<10} {claim.get('gpu', '-').replace('_', ' '):<26} "
                     f"{state:<25} {written:>4} of {expected} windows  on UVA: queue/{on_uva}  {unit}")
    return lines


def hand_over_tick(run_dir, table, dry_run):
    """One whole tick: read each server once, pull and verify, hand over pending units, submit jobs, print."""
    CURRENT["table"] = table
    commit = common.read_json(run_dir / "source.json")["commit"]
    lines, jobs = [], []
    for name, entry in table.items():
        if not entry["ready"] or "hand_over" not in entry:
            print(f"{name}: not part of this tick ({entry.get('not_ready_reason', 'no hand_over entry in the table')})")
            continue
        # an unreachable server is reported and the others are still done
        try:
            if entry.get("prepared_commit") != commit and not dry_run:
                prepare(run_dir, table, name)
            # (a) the one status reading; (b) finished units pulled, verified and moved to done, units
            # whose job ended put back, long-waiting units returned to UVA
            status, assigned = fleet.tick_server(run_dir, table, name, dry_run)
            # units an earlier tick withdrew on the server but could not copy back
            stranded = sorted(set(assigned) & set(status["queue"]["withdrawn"]))
            if stranded and not dry_run:
                bring_back(run_dir, table, name, [(unit, False) for unit in stranded], "tick")
            pool_name = entry["hand_over"]["pool"]
            ledger = set(fleet.ledger_ids(run_dir, name))
            live = fleet.fleet_jobs(name, status, ledger)
            # units wait on a server only while a job of ours runs there or is expected to start soon:
            # otherwise the units no job has claimed come back to UVA now and no further unit goes there
            expected = start_expected(entry, live, status["start_in_seconds"])
            if not expected:
                soonest = min(seconds for job, seconds in status["start_in_seconds"].items()
                              if job in live and seconds is not None)
                waiting = sorted(set(assigned) & set(status["queue"]["pending"]))
                fleet.log(run_dir, f"{name} | hand-over | none: no job of this run runs there, and the scheduler "
                          f"expects the first waiting one to start in {soonest / 3600:.1f} h (limit "
                          f"{entry['hand_over']['start_within_seconds'] / 3600:.1f} h); {len(waiting)} waiting unit(s) "
                          f"{'would ' if dry_run else ''}come back to UVA; the waiting job(s) stay in the queue", dry_run)
                if waiting and not dry_run:
                    moved = fleet.parse_moves(fleet.remote(run_dir, table, name, fleet.move_script(
                        entry, {"place": [], "recover": [], "pull_back": [], "withdraw": waiting}, int(time.time()))))
                    bring_back(run_dir, table, name, [(unit, data) for _, unit, data in moved], "hand-over-tick")
            # (c) further pending units, up to the server's limit
            handed = hand_over(run_dir, table, name, dry_run) if expected else []
            # worker jobs for the units no job will take; the scheduler is asked only when one is due
            due = 0 if not expected else fleet.allowed_jobs(
                entry["pools"][pool_name]["max_jobs"], entry["pools"][pool_name]["max_jobs"],
                sum(fleet.pool_of(name, row["name"]) == pool_name for row in live.values()),
                units_without_job(name, status, assigned, ledger) + len(handed),
                sum(row["state"] != "RUNNING" for row in live.values()))
            if due and not dry_run:
                fleet.submit(run_dir, table, argparse.Namespace(
                    server=name, jobs=entry["pools"][pool_name]["max_jobs"], pool=pool_name, dry_run=False))
            elif due:
                fleet.log(run_dir, f"{name} | submit | {due} worker job(s) would be submitted to pool {pool_name}", True)
        except (fleet.ServerUnreachable, fleet.RemoteCommandFailed) as error:
            fleet.log(run_dir, f"{name} | hand-over-tick | NOT DONE: {error}", dry_run)
            continue
        lines += unit_lines(run_dir, name, status, assigned, handed)
        jobs += [f"{name:<9} job {job:<10} {row['state']} ({row['reason']}), elapsed {row['elapsed']}, "
                 f"partition {row['partition']}, {row['name']}" for job, row in sorted(live.items())]
        first = entry.get("first_unit", {})
        jobs.append(f"{name:<9} first unit: {first.get('state', 'not checked yet')}"
                    + (f" ({first['unit']} against UVA's {first['compared_with_uva_unit']})" if first else ""))
    print("\nUnits on the servers (as read at the start of this tick; this tick's moves are in the lines above):")
    print("\n".join(lines) or "none")
    print("\nJobs of this run on the servers (as read at the start of this tick), and each server's first-unit check:")
    print("\n".join(jobs) or "none")
    print("\nUnits on UVA by queue folder: "
          + ", ".join(f"{folder} {count}" for folder, count in unit_queue.counts(run_dir).items()))


def job_prefix(run_dir):
    """The first part of the name of every remote job of one run folder."""
    # before: .../runs/2026-11-12-02-24_cleanrl-ppo-rnd-baseline_...  ; after: "mazefleet-3f9a1c2e"
    # The eight characters come from the folder's name, so two run folders never share a prefix. With one
    # shared prefix (until 2026-11-12 15:05 PT) each folder took the other's jobs on a server for its own:
    # it wrote their ids into its ledger and submitted no job for its own waiting units.
    return "mazefleet-" + hashlib.sha1(Path(run_dir).resolve().name.encode()).hexdigest()[:8]


def adapt(run_dir):
    """Give the grid campaign's remote fleet this study's worker job, preparation, status, checks and returns."""
    fleet.WORKER_TEMPLATE = WORKER_TEMPLATE
    fleet.REMOTE_FOLDERS = REMOTE_FOLDERS
    fleet.prepare = prepare
    fleet.verify_unit = verify_unit
    fleet.status_script = status_script
    fleet.parse_status = parse_status
    fleet.result_block = result_block
    fleet.bring_back = bring_back
    # jobs of this run folder are told from every other folder's and from the grid study's by their name
    fleet.JOB_PREFIX = job_prefix(run_dir)
    # a unit whose job ended continues from its checkpoint on the same server, so it may be put
    # back in that server's queue more often than a grid unit, which started again from zero
    fleet.RECOVERY_LIMIT = 20


def main():
    """Run one command: this file's own (prepare, hand-over-tick), or one of the grid campaign's, adapted."""
    arguments = sys.argv[1:]
    if "cancel-pending" in arguments:
        raise SystemExit("this file offers no cancel command; see cancel_ledger.py")
    if "--run-dir" not in arguments or arguments.index("--run-dir") + 1 >= len(arguments):
        raise SystemExit("give --run-dir RUN before the command")
    adapt(arguments[arguments.index("--run-dir") + 1])
    if "prepare" in arguments or "hand-over-tick" in arguments:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--run-dir", type=Path, required=True)
        commands = parser.add_subparsers(dest="command", required=True)
        preparing = commands.add_parser("prepare")
        preparing.add_argument("--server", required=True)
        preparing.add_argument("--again", action="store_true", help="send the scripts and the code copy once more")
        commands.add_parser("hand-over-tick").add_argument("--dry-run", action="store_true")
        args = parser.parse_args()
        run_dir = args.run_dir.resolve()
        table = common.read_json(fleet.table_path(run_dir))
        with fleet.fleet_lock(run_dir):
            if args.command == "prepare":
                prepare(run_dir, table, args.server, args.again)
            else:
                hand_over_tick(run_dir, table, args.dry_run)
        return
    # a reminder of the order of servers when units are sent to one of them
    if "stage" in arguments and "--server" in arguments and "--run-dir" in arguments:
        server = arguments[arguments.index("--server") + 1]
        table = common.read_json(fleet.table_path(Path(arguments[arguments.index("--run-dir") + 1]).resolve()))
        earlier = [name for name in SERVER_ORDER[:SERVER_ORDER.index(server)]
                   if table.get(name, {}).get("ready")] if server in SERVER_ORDER else []
        if earlier:
            print(f"order of servers: {', '.join(earlier)} come before {server} and are marked ready; "
                  f"use {server} only when they cannot start the work")
    fleet.main()


if __name__ == "__main__":
    main()
