"""Run units of the pixel sweep on remote servers: Purdue Gautschi and Gilbreth, TAMU ACES, NCSA DeltaAI, MIT.

    python remote_fleet.py --run-dir RUN init-table [--force]
    python remote_fleet.py --run-dir RUN prepare --server S [--again]
    python remote_fleet.py --run-dir RUN smoke --server S [--pool P] [--dry-run]
    python remote_fleet.py --run-dir RUN smoke-result --server S [--pool P]   # P: a held-back pool
    python remote_fleet.py --run-dir RUN hand-over-tick [--dry-run]      # the one command of a monitoring tick
    python remote_fleet.py --run-dir RUN hand-over --server S --pool P --units N [--match TEXT] [--dry-run]
    python remote_fleet.py --run-dir RUN submit --server S --pool P --jobs N [--dry-run]
    python remote_fleet.py --run-dir RUN status
    python remote_fleet.py --run-dir RUN take-back --server S --units N [--match TEXT] [--from-end] [--hold]

The queue folders on a server, the one status call per server, no new login, and the server table
RUN/infrastructure/remote_fleet_servers.json are those of the grid campaign's remote fleet
(scripts/grid_initialization_campaign/remote_fleet.py, imported), as the maze campaign uses it
(scripts/maze_campaign/remote_fleet.py, imported for the parts that are the same). What differs here:

  1. Every server starts "not ready" (init-table writes the table from server_table.json). A server
     becomes ready only when its smoke job has passed: `smoke` submits one job of at most 20 minutes
     per pool (smoke_check.py: imports and card, the asset pack's hashes, the start room's geometry,
     20 default updates compared with the archived baseline, 20 updates with the fast last layer
     and the game events, speed and peak card memory), and `smoke-result` reads its verdict and
     marks the server ready. `smoke-result` decides the comparison with the archive again from the
     job's own 20 default metric lines, under this folder's rule (smoke_check.FIRST_LINE_RULE), since a
     smoke job runs the comparison of the run's frozen code copy, which may hold an older rule.
  2. `prepare` sends the job scripts, the run's frozen code copy, the corrected asset pack, its
     hashes, and the first archived metric lines the smoke job compares with.
  3. The worker job on a server starts this folder's worker.py with --remote-records. A pool names
     its execution profile ("eight copies" on the H200, "four copies" on H100 and A100 80 GB cards,
     "one copy" on smaller ones); only units of that profile are handed to it, so an H100 pool gets
     four-copy packs only and an H200 pool eight-copy packs only. A pool may set
     `worker_minimum_seconds` and `worker_margin_seconds` (the worker's --minimum-seconds and
     --margin-seconds; default the server's unit_seconds and the worker's 900 s), so a short check
     pool's job can train.
  4. A unit goes to a server with its specification and queue record and, when UVA has begun it,
     its whole run folder (checkpoint, metrics, record files, run.json, events, progress). It
     comes back with the same files. A finished unit is verified before it counts as done: a
     completed progress record at the unit's target, one metric line per update, the record files
     folded to the target, run.json with the unit's seeds and settings, the trainer's completed
     event, and a checkpoint.
  5. At every tick the small record files of the units a server holds are copied to
     RUN/slurm/remote_live/<server>/, so the early-signals report shows those runs while they train.
  6. `hand-over-tick` does a whole tick for every ready server: read it once, pull and verify
     finished units, put back units whose job ended, hand over pending units of each pool's
     profile up to the pool's limit (the first of UVA's claim order first), submit worker jobs for
     units no job will take, and print one line per unit. It uses only the pools of the table's
     `hand_over.pools`.
  7. `hand-over` and `submit` do one of those steps by hand for one pool, also a pool outside
     `hand_over.pools` (a check pool the tick leaves alone): up to N pending units of the pool's
     profile (those whose name holds --match) go to the server; N worker jobs within the pool's
     max_jobs and the server's pending units.

Job ids go to RUN/slurm/submitted_jobids_<server>.txt (one file per server). This file offers no
cancel command (scripts/maze_campaign/cancel_ledger.py --server cancels a server's waiting jobs).
Standard library only.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import time
from zoneinfo import ZoneInfo

from campaign_imports import HERE, common, grid, maze, own, unit_queue
import pixel_queue

fleet = grid("remote_fleet")
maze_fleet = maze("remote_fleet")
smoke_check = own("smoke_check")
worker = own("worker")
ARCHIVE_CHECK = "default updates against the archived baseline"
SERVER_TABLE = HERE / "server_table.json"
ASSETS = common.PROJECT / "reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png"
ARCHIVE_RUNS = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs")
ARCHIVE_PATTERN = "2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_*/data"
REMOTE_FOLDERS = tuple(f"queue/{state}" for state in fleet.REMOTE_STATES) + (
    "specifications", "data/research", "logs", "slurm/jobs", "smoke")
SMOKE_UPDATES = 20
SMOKE_TIME_LIMIT = "00:20:00"
WINDOW_UPDATES = 61

JOB_HEAD = r'''#!/bin/bash@SHELL_OPTION@
# @PURPOSE@ of the pixel Montezuma sweep on @SERVER@, pool @POOL@. Written by
# scripts/pixel_sweep_campaign/remote_fleet.py: change that, not this copy. One card per job.
RUN=@RUN@
PY=@PYTHON@
CODE=@CODE@
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export XDG_DATA_HOME=@ASSETS@
export JAXTARI_SKIP_SPRITE_UPDATE=1
@JOB_SETUP_LINES@
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
@CARD_GUARD@
echo "job $JOB on $(hostname), start $(date -Is), card class @GPU_CLASS@"
'''

WORKER_TEMPLATE = JOB_HEAD.replace("@PURPOSE@", "Worker job") + r'''
# Nothing to do: give the card back before starting Python.
if ! ls queue/pending/*.json > /dev/null 2>&1; then echo "nothing pending; this job ends"; exit 0; fi

# The environment must use the card and hold every module the trainer imports at its start, or every unit taken
# would fail: they are imported here first, before a unit is claimed.
PYTHONPATH="$CODE/src" "$PY" -c 'import jax, jaxtari
from exploration_platform.atari_pixels.config import PixelPPOConfig
from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
from exploration_platform.atari_pixels.training import PixelPPORND
print("jax devices:", jax.devices()); assert jax.default_backend() == "gpu"' \
  || { echo "jax does not use the card, or a module the trainer imports is missing; this job ends without taking a unit"; exit 3; }

exec "$PY" "$CODE/scripts/pixel_sweep_campaign/worker.py" --run-dir "$RUN" --remote-records --server @SERVER@ \
  --minimum-seconds @UNIT_SECONDS@ --margin-seconds @MARGIN_SECONDS@
'''

SMOKE_TEMPLATE = JOB_HEAD.replace("@PURPOSE@", "Smoke job (the server's check and its speed check)") + r'''
# One job of at most 20 minutes: imports and card, asset hashes, start-room geometry, 20 default updates compared
# with the archived baseline, 20 updates with the fast last layer and the game events, speed and peak card memory.
# Its verdict is smoke/result_@POOL@.json; `remote_fleet.py smoke-result` reads it.
exec "$PY" "$CODE/scripts/pixel_sweep_campaign/smoke_check.py" --run-dir "$RUN" --code "$CODE" --server @SERVER@ \
  --pool @POOL@ --profile @PROFILE@ --updates @SMOKE_UPDATES@
'''


def job_prefix(run_dir):
    """The first part of the name of every remote job of one run folder."""
    # before: .../runs/2026-11-18-02-00_pixel-montezuma_grid_...  ; after: "pixelfleet-3f9a1c2e"
    # the eight characters come from the folder's name, so two run folders never share a prefix
    return "pixelfleet-" + hashlib.sha1(Path(run_dir).resolve().name.encode()).hexdigest()[:8]


def profile_words(name):
    """An execution profile's name as one word, for a file name and a command line."""
    # before: "four copies" ; after: "four-copies"
    return name.replace(" ", "-")


def job_script(template, name, entry, pool_name, pool):
    """Fill one job script: the grid campaign's fields, the server's shell option and lines, the assets, the profile."""
    # the maze campaign's writer fills @SERVER@, @POOL@, @RUN@, @PYTHON@, @CODE@, @UNIT_SECONDS@, @GPU_CLASS@,
    # @CARD_GUARD@, @SHELL_OPTION@ and @JOB_SETUP_LINES@ of whatever template the grid module holds; the
    # worker's least training time and its margin may be the pool's own (a check pool of 20-minute jobs)
    # before: a pool without the two keys ; after: "--minimum-seconds 1800 --margin-seconds 900" (the server's unit_seconds)
    # before: {"worker_minimum_seconds": 240, "worker_margin_seconds": 240} ; after: "--minimum-seconds 240 --margin-seconds 240"
    template = (template.replace("@UNIT_SECONDS@", str(int(pool.get("worker_minimum_seconds", entry["unit_seconds"]))))
                .replace("@MARGIN_SECONDS@", str(int(pool.get("worker_margin_seconds", worker.MARGIN_SECONDS)))))
    held = fleet.WORKER_TEMPLATE
    fleet.WORKER_TEMPLATE = template
    try:
        text = maze_fleet.worker_script(name, entry, pool_name, pool)
    finally:
        fleet.WORKER_TEMPLATE = held
    return (text.replace("@ASSETS@", shlex.quote(entry["assets"]))
            .replace("@PROFILE@", profile_words(pool["execution_profile"]))
            .replace("@SMOKE_UPDATES@", str(SMOKE_UPDATES)))


def worker_script(name, entry, pool_name, pool):
    """Write one pool's worker job script."""
    return job_script(WORKER_TEMPLATE, name, entry, pool_name, pool)


def asset_hashes(root=ASSETS):
    """The SHA-256 of every file of the asset pack, by its path inside the pack."""
    # after: {"jaxtari/asset_source.json": "6e52...", "jaxtari/sprites/montezumarevenge/...": "...", ...}
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def joined_lines(parts):
    """Join the metric lines of several archived runs, update by update, into the lines of one run of all their copies."""
    # before: [[{"update": 1, "unix": 5.0, "loss": [0.6, 0.5, 0.5, 0.6], "steps_per_policy": 16384}],
    #          [{"update": 1, "unix": 7.0, "loss": [0.7], "steps_per_policy": 16384}]]
    # after:  [{"update": 1, "unix": 5.0, "loss": [0.6, 0.5, 0.5, 0.6, 0.7], "steps_per_policy": 16384}]
    # a list holds one value per copy and is joined in the runs' order; a number is the same for every copy
    # (update, steps) or the write time, and is the first run's
    joined = []
    for lines in zip(*parts):
        line = dict(lines[0])
        for field, value in lines[0].items():
            if isinstance(value, list):
                line[field] = [item for other in lines for item in other[field]]
            elif field != "unix" and any(other[field] != value for other in lines):
                raise ValueError(f"the archived runs differ in {field!r} at update {line['update']}")
        joined.append(line)
    return joined


def archive_reference(profile, lines=SMOKE_UPDATES):
    """The first metric lines of the archived run(s) of one profile, which a smoke job compares with, and their seeds."""
    # before: profile "eight copies" with the archived runs of seeds 1 to 4 (four copies) and of seeds 5, 6, 7, 8
    # after:  20 lines of eight copies each, and the seeds [1, 2, 3, 4, 5, 6, 7, 8]
    data = next(iter(sorted(ARCHIVE_RUNS.glob(ARCHIVE_PATTERN))), None)
    if data is None:
        raise FileNotFoundError(f"the archived baseline's data folder ({ARCHIVE_RUNS}/{ARCHIVE_PATTERN}) does not "
                                "exist on this machine; the smoke job's comparison needs its first metric lines")
    parts, seeds = [], []
    for name in profile["archived_runs"]:
        run = data / name / "run"
        with (run / "metrics.jsonl").open() as stream:
            parts.append([json.loads(next(stream)) for _ in range(lines)])
        seeds += common.read_json(run / "run.json")["copy_seeds"]
    if len(seeds) != profile["copies"]:
        raise ValueError(f"the archived runs {profile['archived_runs']} hold {len(seeds)} copies; the profile has "
                         f"{profile['copies']}")
    return "".join(json.dumps(line) + "\n" for line in joined_lines(parts)), seeds


def prepare(run_dir, table, name, again=False):
    """Create a server's run folders and send its job scripts, the frozen code copy, the assets and the smoke files."""
    entry = table[name]
    source = common.read_json(run_dir / "source.json")
    commit = source["commit"]
    # a server that already holds this code copy and its job scripts is left as it is
    if entry.get("prepared_commit") == commit and not again:
        return
    entry["code"] = f"{entry['code_root']}/source_{commit}/10_jax_exploration_platform"
    # the server's files are built locally, laid out as in its run folder, so one copy sends them all
    tree = run_dir / "infrastructure/remote_fleet" / name
    for folder in REMOTE_FOLDERS:
        (tree / folder).mkdir(parents=True, exist_ok=True)
    profiles = pixel_queue.read_profiles()
    for pool_name, pool in entry["pools"].items():
        (tree / f"worker_job_{pool_name}.sh").write_text(worker_script(name, entry, pool_name, pool))
        (tree / f"smoke_job_{pool_name}.sh").write_text(job_script(SMOKE_TEMPLATE, name, entry, pool_name, pool))
        # what the pool's smoke job compares its 20 default updates with, and the seeds that run had
        text, seeds = archive_reference(profiles[pool["execution_profile"]])
        words = profile_words(pool["execution_profile"])
        (tree / "smoke" / f"reference_{words}.jsonl").write_text(text)
        common.write_json(tree / "smoke" / f"reference_{words}_seeds.json", seeds)
    common.write_json(tree / "smoke/asset_sha256.json", asset_hashes())
    fleet.remote(run_dir, table, name, f"mkdir -p {shlex.quote(entry['remote_run'])} {shlex.quote(entry['code'])} "
                                       f"{shlex.quote(entry['assets'])}\n")
    fleet.rsync(run_dir, table, name, [f"{tree}/", fleet.remote_path(entry)])
    # the same frozen code copy as on UVA (its SOURCE_COMMIT and SOURCE_COPY_NUMBER go with it), and the
    # asset pack file by file; nothing is deleted on the server
    fleet.rsync(run_dir, table, name, ["--exclude", "__pycache__",
                                       f"{source['snapshot']}/10_jax_exploration_platform/",
                                       f"{entry['ssh'][-1]}:{entry['code']}/"])
    fleet.rsync(run_dir, table, name, ["--checksum", f"{ASSETS}/", f"{entry['ssh'][-1]}:{entry['assets']}/"])
    number = source.get("copy_number", 0)
    entry.update(prepared_commit=commit, prepared_copy_number=number)
    common.write_json(fleet.table_path(run_dir), table)
    fleet.log(run_dir, f"{name} | prepare | job scripts, the code copy of commit {commit} (copy number {number}), "
              "the asset pack and the smoke files are on the server")


def status_script(name, entry, assigned):
    """The grid campaign's status script with two more sections: each assigned unit's progress, and start estimates."""
    # the same additions as scripts/maze_campaign/remote_fleet.py, status_script; the progress is read from
    # the trainer's progress.json instead of a shard
    text = fleet.grid_functions["status_script"](name, entry, assigned)
    end = f'echo; echo "{fleet.MARK}end"\nexit 0\n'
    if not text.endswith(end):
        raise RuntimeError("the grid campaign's status script no longer ends with its end marker")
    lines = [f'echo; echo "{fleet.MARK}progress"']
    if assigned:
        lines += [f'for unit in {" ".join(shlex.quote(unit) for unit in assigned)}; do',
                  '  folder="data/$unit/run"; [ -d "$folder" ] || continue',
                  '  update=$(grep -a -o \'"update": [0-9]*\' "$folder/progress.json" 2> /dev/null | head -n 1)',
                  '  grep -a -q \'"completed": true\' "$folder/progress.json" 2> /dev/null && completed=yes || completed=no',
                  '  [ -e "$folder/checkpoint.pkl" ] && saved=yes || saved=no',
                  '  echo "$unit update=${update##* } completed=$completed checkpoint=$saved"', 'done']
    # the scheduler's start estimate of each waiting job, as seconds from now on the server's own clock
    lines += [f'echo; echo "{fleet.MARK}estimates"',
              'squeue -h -u "$(id -un)" -t PENDING -o "%i %S" | while read -r job start; do',
              '  case "$start" in N/A|"") echo "$job unknown" ;; '
              '*) echo "$job $(( $(date -d "$start" +%s) - $(date +%s) ))" ;; esac', 'done']
    return text[:-len(end)] + "\n".join(lines) + "\n" + end


def parse_status(text):
    """Read one status answer as the grid campaign does, and each assigned unit's progress beside it."""
    # before: "@@fleet:progress\nunit-a update=4120 completed=no checkpoint=yes\nunit-b update= completed=no checkpoint=no"
    # after:  status["progress"] = {"unit-a": {"update": 4120, "completed": False, "checkpoint": True},
    #                               "unit-b": {"update": 0, "completed": False, "checkpoint": False}}
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
            status["progress"][unit] = {"update": int(values["update"]) if values["update"] else 0,
                                        "completed": values["completed"] == "yes",
                                        "checkpoint": values["checkpoint"] == "yes"}
        elif section == "estimates" and line.strip():
            job, seconds = line.split()
            status["start_in_seconds"][job] = None if seconds == "unknown" else int(seconds)
    return status


def result_block(name, claim, summary=None):
    """The grid campaign's result of a remote unit, with the card, the profile and the speed of its last attempt."""
    block = fleet.grid_functions["result_block"](name, claim, summary)
    if summary is not None:
        attempt = summary.get("attempt", {})
        block.update(card=summary.get("card"), execution_profile=summary.get("execution_profile"),
                     seconds_per_update=attempt.get("seconds_per_update"),
                     update_at_end=attempt.get("update_at_end"))
    return block


def metrics_facts(path):
    """Count a metrics file's lines and read the update of its first and of its last line."""
    # before: a file of 12,207 lines, the first {"update": 1, ...}, the last {"update": 12207, ...}
    # after:  (12207, 1, 12207)
    count, first, last = 0, None, None
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                count += 1
                first = line if first is None else first
                last = line
    return count, json.loads(first)["update"] if first else None, json.loads(last)["update"] if last else None


def returned_unit_problem(folder, record, specification):
    """Check a finished unit's files as a server returned them; None when they are complete, else what is wrong."""
    folder, target = Path(folder), record["target_updates"]
    missing = [name for name in ("run.json", "progress.json", "events.jsonl", "metrics.jsonl", "checkpoint.pkl")
               if not (folder / name).exists() or (folder / name).stat().st_size == 0]
    if missing:
        return f"missing or empty: {', '.join(missing)}"
    # the trainer's own word that it reached the target, and one metric line per update up to it
    if not pixel_queue.is_complete(pixel_queue.read_progress(folder), target):
        return f"progress.json does not say completed at update {target}"
    lines, first, last = metrics_facts(folder / "metrics.jsonl")
    if (lines, first, last) != (target, 1, target):
        return f"metrics.jsonl holds {lines} lines from update {first} to {last}; expected updates 1 to {target}"
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line.strip()]
    if not any(event.get("event") == "completed" and event.get("update") == target for event in events):
        return f"events.jsonl holds no completed event of the trainer at update {target}"
    # run.json must describe this unit: its seeds and the learner settings of its command
    described = common.read_json(folder / "run.json")
    if described["copy_seeds"] != specification["copy_seeds"]:
        return f"run.json is of the seeds {described['copy_seeds']}, the unit's are {specification['copy_seeds']}"
    learner, _ = pixel_queue.trainer_settings(record, specification)
    differing = [key for key, value in learner.items()
                 if key != "per_copy_scalar_settings" and described["learner"].get(key) != value]
    if differing:
        return f"run.json's learner differs from the unit's in {sorted(differing)}"
    # the record files, where the unit records the game events: folded to the target, one line per whole window
    if specification["learner"].get("record_game_events"):
        if not (folder / "game_first_times.json").exists() or not (folder / "game_windows.jsonl").exists():
            return "the record files game_first_times.json and game_windows.jsonl are not both there"
        first_times = common.read_json(folder / "game_first_times.json")
        windows = [line for line in (folder / "game_windows.jsonl").read_text().splitlines() if line.strip()]
        if first_times["updates_folded"] != target or first_times["has_event_fields"] is not True:
            return (f"game_first_times.json is folded to update {first_times['updates_folded']} "
                    f"(events recorded: {first_times['has_event_fields']}); expected update {target} with events")
        if len(windows) != target // first_times["window_updates"]:
            return (f"game_windows.jsonl holds {len(windows)} windows; "
                    f"{target // first_times['window_updates']} are expected")
    return None


def held_record_path(run_dir, unit):
    """Name the UVA queue record of a unit a server holds."""
    server = fleet.load_assignment(run_dir)[unit]["server"]
    return run_dir / "queue" / f"remote_{server}" / f"{unit}.json"


def verify_unit(run_dir, unit):
    """Check a pulled unit's files, then move them into its run folder; None when the unit is complete."""
    pulled = run_dir / "data/research" / unit
    if not (pulled / "summary.json").exists():
        return "summary.json is missing"
    summary = common.read_json(pulled / "summary.json")
    if summary.get("completed") is not True:
        return "summary.json does not say completed"
    record = common.read_json(held_record_path(run_dir, unit))
    specification = common.read_json(run_dir / "specifications" / f"{unit}.json")
    # the server's worker fixed the profile when the unit first started there; UVA's record takes it over
    record.update({key: summary[key] for key in ("execution_profile", "execution_profile_settings")})
    # a unit verified at an earlier tick whose queue move did not happen has its files in its run folder already
    target = pixel_queue.run_folder(run_dir, unit)
    source = pulled if (pulled / "metrics.jsonl").exists() else target
    problem = returned_unit_problem(source, record, specification)
    if problem:
        return problem
    # only checked files replace what UVA held for this unit (the files of an earlier attempt)
    target.mkdir(parents=True, exist_ok=True)
    for name in pixel_queue.RUN_FILES:
        if source != target and (source / name).exists():
            os.replace(source / name, target / name)
    # the attempts the unit had on the server go into UVA's record, the last one with its speed
    # before: UVA's record {"history": [], "execution_profile": None}
    # after:  {"history": [..., {"card": "NVIDIA H100 80GB HBM3", "outcome": "complete", "server": "gautschi",
    #          "seconds_per_update": 2.1, ...}], "execution_profile": "four copies", ...}
    last = dict(summary.get("attempt", {}), card=summary.get("card"), outcome="complete",
                server=fleet.load_assignment(run_dir)[unit]["server"])
    known = {(attempt.get("job_id"), attempt.get("started_at")) for attempt in record.get("history", [])}
    earlier = [attempt for attempt in summary.get("earlier_attempts", [])
               if (attempt.get("job_id"), attempt.get("started_at")) not in known]
    record["history"] = record.get("history", []) + earlier + [last]
    with common.queue_lock(run_dir):
        common.write_json(held_record_path(run_dir, unit), record)
    return None


def bring_back(run_dir, table, name, returned, command, folder="pending"):
    """Put units a server gave up back in a UVA queue folder, each with its run folder when the server saved one."""
    # the same steps as scripts/maze_campaign/remote_fleet.py, bring_back; the saved state of a pixel unit is
    # its whole run folder (checkpoint.pkl with the metrics and the record files that belong to it)
    entry = table[name]
    for unit, _ in returned:
        arrived = run_dir / "slurm/remote_returned" / name / unit
        arrived.mkdir(parents=True, exist_ok=True)
        # relative paths ("/./"), so a unit that never started there (no data/<unit>/ folder) is skipped like a
        # missing file instead of ending the copy with rsync's exit 23
        # before (server): queue/withdrawn/<unit>.json, data/<unit>/run/{checkpoint.pkl, metrics.jsonl, ...}
        # after (UVA):     slurm/remote_returned/<server>/<unit>/<unit>.json, .../<unit>/<unit>/run/...
        fleet.rsync(run_dir, table, name, [
            "--relative", "--ignore-missing-args", fleet.remote_path(entry, f"queue/withdrawn/./{unit}.json"),
            f":{entry['remote_run']}/data/./{unit}/run", f"{arrived}/"])
        # checkpoint and metrics belong together: the run folder replaces UVA's as a whole, or not at all
        copied = arrived / unit / "run"
        saved = (copied / "checkpoint.pkl").exists() and (copied / "metrics.jsonl").exists()
        if saved:
            target = pixel_queue.run_folder(run_dir, unit)
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted(copied.iterdir()):
                os.replace(path, target / path.name)
        # the server's record holds the unit's attempts and its profile; UVA's record takes them over
        record = arrived / f"{unit}.json"
        returned_record = common.read_json(record) if record.exists() else None
        if returned_record is not None:
            # the lowest code copy that may run the unit never falls
            held_record = common.read_json(run_dir / "queue" / f"remote_{name}" / f"{unit}.json")
            needed = max(unit_queue.required_copy(returned_record), unit_queue.required_copy(held_record))
            if needed:
                returned_record["minimum_copy_number"] = needed
        fleet.move_unit(run_dir, unit, f"remote_{name}", folder, returned_record)
        fleet.finish_assignment(run_dir, [unit], "returned")
        fleet.log(run_dir, f"{name} | {command} | {unit} is back in queue/{folder} on UVA, "
                  + ("with its run folder: it continues from the server's checkpoint" if saved else
                     "without a checkpoint from the server: it starts again or continues from what UVA held"))


def held_units(run_dir, name):
    """Read the UVA queue records of the units one server holds, by unit name."""
    folder = run_dir / "queue" / f"remote_{name}"
    return {path.stem: common.read_json(path) for path in sorted(folder.glob("*.json"))}


def hand_over(run_dir, table, name, pool_name, dry_run, units_at_once=None, match=None):
    """Move pending units of one pool's profile from UVA to a server, up to the pool's limit; returns their names.

    units_at_once and match replace the table's limit (hand_over.pools.<pool>.units_at_once) and its
    hand_over.match, for the `hand-over` command (a pool the tick does not use has no limit there).
    """
    entry, pool = table[name], table[name]["pools"][pool_name]
    settings = entry["hand_over"]
    limit = settings["pools"][pool_name]["units_at_once"] if units_at_once is None else units_at_once
    match = settings.get("match", "") if match is None else match
    profile_name = pool["execution_profile"]
    profile = pixel_queue.read_profiles()[profile_name]
    held = run_dir / "queue" / f"remote_{name}"
    with common.queue_lock(run_dir):
        # the units of this profile the server holds already count against the pool's limit
        holding = sum(unit["copies"] == profile["copies"] for unit in held_units(run_dir, name).values())
        pending = sorted((common.read_json(path) for path in (run_dir / "queue/pending").glob("*.json")),
                         key=lambda unit: (unit["order"], unit["name"]))
        # the first of the claim order first, so early results cover every configuration at the first seeds
        candidates = [unit["name"] for unit in pending
                      if unit["copies"] == profile["copies"] and unit.get("execution_profile") in (None, profile_name)
                      and match in unit["name"]]
        chosen = candidates[:max(0, limit - holding)]
        if chosen and not dry_run:
            # the rename out of pending is the hand-over: no UVA worker can claim the unit after it
            held.mkdir(exist_ok=True)
            for unit in chosen:
                os.rename(run_dir / "queue/pending" / f"{unit}.json", held / f"{unit}.json")
            assignment = fleet.load_assignment(run_dir)
            assignment.update({unit: {"server": name, "pool": pool_name, "staged_at": common.now(), "origin": "pending",
                                      "recoveries": 0, "state": "staged"} for unit in chosen})
            common.write_json(fleet.assignment_path(run_dir), assignment)
    fleet.log(run_dir, f"{name} | hand-over | pool {pool_name} ({profile_name}): limit "
              f"{limit} unit(s) at once, {holding} held before, "
              f"{len(candidates)} pending on UVA fit: {len(chosen)} unit(s) {'would be ' if dry_run else ''}handed over",
              dry_run)
    for unit in chosen:
        fleet.log(run_dir, f"{name} | hand-over | {unit}", dry_run)
    if not chosen or dry_run:
        return chosen
    # one copy sends everything: specification, queue record (to incoming/, out of the workers' sight) and,
    # for a unit UVA began earlier, its run folder, so it continues on the server
    stamp = int(time.time())
    tree = run_dir / "slurm/remote_hand_over" / f"{name}_{stamp}"
    for unit in chosen:
        files = [(run_dir / "specifications" / f"{unit}.json", "specifications"),
                 (held / f"{unit}.json", "queue/incoming")]
        folder = pixel_queue.run_folder(run_dir, unit)
        if (folder / "checkpoint.pkl").exists() and (folder / "metrics.jsonl").exists():
            files += [(path, f"data/{unit}/run") for path in sorted(folder.iterdir()) if path.is_file()]
        for source, target in files:
            (tree / target).mkdir(parents=True, exist_ok=True)
            # a hard link: a checkpoint is up to 230 MB and need not be copied to be sent
            os.link(source, tree / target / source.name)
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


def mirror_records(run_dir, table, name):
    """Copy the small record files of every unit a server holds to RUN/slurm/remote_live/<server>/."""
    # before (on the server): data/<unit>/run/{progress.json, game_first_times.json, game_windows.jsonl,
    #         events.jsonl, run.json, metrics.jsonl, checkpoint.pkl}
    # after (on UVA): slurm/remote_live/<server>/<unit>/run/ with the first five; metrics and checkpoint stay
    entry = table[name]
    target = run_dir / "slurm/remote_live" / name
    target.mkdir(parents=True, exist_ok=True)
    fleet.rsync(run_dir, table, name, ["--prune-empty-dirs", "--include=*/"]
                + [f"--include=*/run/{file_name}" for file_name in pixel_queue.RECORD_FILES]
                + ["--exclude=*", f"{entry['ssh'][-1]}:{entry['remote_run']}/data/", f"{target}/"])


def units_without_job(name, status, units, ledger):
    """Name the given units that wait on a server or whose job is gone, by one status reading."""
    # the same rule as scripts/maze_campaign/remote_fleet.py, units_without_job, for a subset of the units
    live = fleet.fleet_jobs(name, status, ledger)
    waiting = set(units) & (set(status["queue"]["pending"]) | set(status["queue"]["incoming"]))
    orphaned = {unit for unit in set(units) & set(status["queue"]["running"])
                if live.get(status["claims"].get(("running", unit), {}).get("job"), {}).get("state") != "RUNNING"}
    return sorted(waiting | orphaned)


def unit_lines(name, status, records, handed):
    """One line per unit a server holds: server, job id, card, state there, updates done of the unit's target."""
    # after: "gautschi  job 18420011   NVIDIA H100 80GB HBM3      running      4120 of 12207 updates  <unit>"
    lines = []
    for unit, record in records.items():
        state = next((folder for folder, units in status["queue"].items() if unit in units),
                     "handed over at this tick" if unit in handed else "not on the server")
        claim = status["claims"].get((state, unit), {})
        update = status["progress"].get(unit, {}).get("update", 0)
        lines.append(f"{name:<9} job {claim.get('job', '-'):<10} {claim.get('gpu', '-').replace('_', ' '):<26} "
                     f"{state:<25} {update:>6} of {record['target_updates']} updates  {unit}")
    return lines


def hand_over_tick(run_dir, table, dry_run):
    """One whole tick: read each ready server once, pull and verify, hand over pending units, submit jobs, print."""
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
            # the one status reading; finished units pulled, verified and moved to done, units whose job
            # ended put back in the server's queue, long-waiting units returned to UVA
            status, assigned = fleet.tick_server(run_dir, table, name, dry_run)
            # units an earlier tick withdrew on the server but could not copy back
            stranded = maze_fleet.stranded_units(assigned, status["queue"])
            if stranded and not dry_run:
                bring_back(run_dir, table, name, [(unit, False) for unit in stranded], "tick")
            ledger = set(fleet.ledger_ids(run_dir, name))
            live = fleet.fleet_jobs(name, status, ledger)
            # units wait on a server only while a job of ours runs there or is expected to start soon:
            # otherwise the units no job has claimed come back to UVA now and no further unit goes there
            expected = maze_fleet.start_expected(entry, live, status["start_in_seconds"])
            if not expected:
                waiting = sorted(set(assigned) & set(status["queue"]["pending"]))
                fleet.log(run_dir, f"{name} | hand-over | none: no job of this run runs there, and the scheduler expects "
                          f"none to start within {entry['hand_over']['start_within_seconds'] / 3600:.1f} h; "
                          f"{len(waiting)} waiting unit(s) {'would ' if dry_run else ''}come back to UVA; the waiting "
                          "job(s) stay in the queue", dry_run)
                if waiting and not dry_run:
                    moved = fleet.parse_moves(fleet.remote(run_dir, table, name, fleet.move_script(
                        entry, {"place": [], "recover": [], "pull_back": [], "withdraw": waiting}, int(time.time()))))
                    bring_back(run_dir, table, name, [(unit, data) for _, unit, data in moved], "hand-over-tick")
            handed = []
            profiles = pixel_queue.read_profiles()
            for pool_name, limits in entry["hand_over"]["pools"].items() if expected else []:
                pool = entry["pools"][pool_name]
                new = hand_over(run_dir, table, name, pool_name, dry_run) if limits["units_at_once"] > 0 else []
                handed += new
                # worker jobs for the units of this pool's profile that no job will take
                copies = profiles[pool["execution_profile"]]["copies"]
                mine = [unit for unit, record in held_units(run_dir, name).items()
                        if record["copies"] == copies and unit not in new]
                in_pool = [row for row in live.values() if fleet.pool_of(name, row["name"]) == pool_name]
                due = fleet.allowed_jobs(pool["max_jobs"], pool["max_jobs"], len(in_pool),
                                         len(units_without_job(name, status, mine, ledger)) + len(new),
                                         sum(row["state"] != "RUNNING" for row in in_pool))
                if due and not dry_run:
                    fleet.submit(run_dir, table, argparse.Namespace(server=name, jobs=due, pool=pool_name, dry_run=False))
                elif due:
                    fleet.log(run_dir, f"{name} | submit | {due} worker job(s) would be submitted to pool {pool_name}", True)
            # the record files of the units that train there, for the early-signals report
            if not dry_run and held_units(run_dir, name):
                mirror_records(run_dir, table, name)
        except (fleet.ServerUnreachable, fleet.RemoteCommandFailed) as error:
            fleet.log(run_dir, f"{name} | hand-over-tick | NOT DONE: {error}", dry_run)
            continue
        lines += unit_lines(name, status, held_units(run_dir, name), handed)
        jobs += [f"{name:<9} job {job:<10} {row['state']} ({row['reason']}), elapsed {row['elapsed']}, "
                 f"partition {row['partition']}, {row['name']}" for job, row in sorted(live.items())]
    print("\nUnits on the servers (as read at the start of this tick; this tick's moves are in the lines above):")
    print("\n".join(lines) or "none")
    print("\nJobs of this run on the servers (as read at the start of this tick):")
    print("\n".join(jobs) or "none")
    print("\nUnits on UVA by queue folder: "
          + ", ".join(f"{folder} {count}" for folder, count in unit_queue.counts(run_dir).items()))


def init_table(run_dir, force=False):
    """Write the run folder's server table from this folder's server_table.json: every server not ready."""
    path = fleet.table_path(run_dir)
    if path.exists() and not force:
        raise SystemExit(f"{path} exists already; it is this run folder's own table (--force writes it anew)")
    table = common.read_json(SERVER_TABLE)
    for entry in table.values():
        # before: "/scratch/gautschi/liu4869/rnd_pixel/runs/{run_name}" ; after: ".../runs/<the run folder's name>"
        entry["remote_run"] = entry["remote_run"].replace("{run_name}", run_dir.name)
        if entry["ready"]:
            raise ValueError("server_table.json must mark every server not ready: a smoke job makes a server ready")
    common.write_json(path, table)
    print(f"wrote {path}: {', '.join(table)}; every server is not ready until its smoke job has passed")


def smoke_script(run_dir, name, entry, pools, stamp):
    """The bash script that submits one smoke job per pool on a server and reports each job id."""
    lines = [f"cd {shlex.quote(entry['remote_run'])} || exit 3"]
    for pool in pools:
        options = shlex.join(entry["pools"][pool]["sbatch_options"])
        job_name = f"{job_prefix(run_dir)}-{name}-{pool}-smoke-{stamp}"
        output = shlex.quote(f"{entry['remote_run']}/smoke/smoke_{pool}_%j.out")
        script = shlex.quote(f"{entry['remote_run']}/smoke_job_{pool}.sh")
        command = f"sbatch --parsable -o {output} -t {SMOKE_TIME_LIMIT} {options} -J {job_name} {script}"
        # the id is the last line sbatch prints; it goes to the server's ledger in the command that created it
        lines += [f"out=$({command} 2>&1); code=$?", 'id=$(echo "$out" | tail -n 1); id=${id%%;*}',
                  f'if [ "$code" -eq 0 ] && [ -n "$id" ] && [ -z "${{id//[0-9]/}}" ]; then '
                  f'echo "$id" >> slurm/submitted_jobids_{name}.txt; echo "{fleet.MARK}submitted {pool} $id"; '
                  f'else echo "{fleet.MARK}refused {pool} $(echo "$out" | tr "\\n" " ")"; fi']
    return "\n".join(lines + [f'echo "{fleet.MARK}end"', "exit 0"]) + "\n"


def smoke(run_dir, table, name, pool_name=None, dry_run=False):
    """Submit one smoke job per pool of a server (or the named pool); each id goes to the server's ledger."""
    entry = table[name]
    pools = [pool_name] if pool_name else list(entry["pools"])
    stamp = datetime.now().astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    script = smoke_script(run_dir, name, entry, pools, stamp)
    fleet.log(run_dir, f"{name} | smoke | one job of at most {SMOKE_TIME_LIMIT} for each of the pools "
              f"{', '.join(pools)}; the server stays not ready until smoke-result has read a passed verdict", dry_run)
    if dry_run:
        print(script)
        return []
    prepare(run_dir, table, name, again=True)
    answer = fleet.remote(run_dir, table, name, script, timeout=300)
    submitted = []
    for line in answer.splitlines():
        if line.startswith(f"{fleet.MARK}submitted "):
            _, pool, job = line.split()
            with fleet.ledger_path(run_dir, name).open("a") as handle:
                handle.write(job + "\n")
            submitted.append(job)
            fleet.log(run_dir, f"{name} | smoke | job {job} submitted for pool {pool}; read it with smoke-result")
        elif line.startswith(f"{fleet.MARK}refused "):
            fleet.log(run_dir, f"{name} | smoke | REFUSED by the scheduler: {line[len(fleet.MARK) + 8:]}")
    return submitted


def smoke_verdict(entry, results):
    """Decide from the pools' smoke results whether a server is ready; returns (ready, reason when it is not)."""
    # before: pools "a" and "b" take units; results {"a": {"passed": True, ...}} ; after: (False, "pool b has no smoke result")
    wanted = list(entry["pools"])
    for pool in wanted:
        if pool not in results:
            return False, f"pool {pool} has no smoke result yet"
        if results[pool].get("passed") is not True:
            failed = [check["name"] for check in results[pool].get("checks", []) if not check.get("passed")]
            return False, f"the smoke job of pool {pool} did not pass ({', '.join(failed) or 'no check listed'})"
    return True, None


def read_first_update_again(result, reference, lines):
    """Decide a smoke result's comparison with the archive again under this folder's rule, from the job's own lines."""
    # A smoke job runs the comparison of the run's frozen code copy. The rule of this folder's smoke_check.py
    # decides instead (decision of 2026-11-18: |new - archive| <= 1e-3 * |archive| + 1e-6, whole numbers equal),
    # read from the 20 default metric lines the job left on the server and the reference it compared them with.
    # before: checks [..., {"name": ARCHIVE_CHECK, "passed": False, "detail": "... 3.33e-03 (limit 1e-03) ..."}],
    #         passed False, the only check that did not pass
    # after:  that check replaced by the comparison under the rule ("passed": True, "detail": "... 0 number(s)
    #         outside the rule ..."), the job's own check kept as "job_archive_check", passed True
    # a job that left no default lines (it ended before them) keeps its own verdict
    if not reference or not lines:
        return result
    checks = [dict(smoke_check.compare_with_archive(reference, lines), read_again_on="UVA")
              if check["name"] == ARCHIVE_CHECK else check for check in result.get("checks", [])]
    job_check = next((check for check in result.get("checks", []) if check["name"] == ARCHIVE_CHECK), None)
    return dict(result, checks=checks, passed=bool(checks) and all(check["passed"] for check in checks),
                job_archive_check=job_check, first_update_rule=smoke_check.FIRST_LINE_RULE)


def held_back_pools(entry):
    """A server's pools held back from taking units, by name (the table's free-text note left out)."""
    # before: {"standby-a10": {...pool...}, "note": "Not used until ..."} ; after: {"standby-a10": {...pool...}}
    return {pool: settings for pool, settings in entry.get("pools_held_back", {}).items() if isinstance(settings, dict)}


def read_smoke_results(run_dir, table, name, pools):
    """Read the smoke results of some pools of a server in one call, decide each again under this folder's rule, print them."""
    entry = table[name]
    known = {**held_back_pools(entry), **entry["pools"]}
    script = [f"cd {shlex.quote(entry['remote_run'])} || exit 3"]
    for pool in pools:
        settings = known[pool]
        # the verdict, the job's own default metric lines and the reference they were compared with
        script += [f'echo; echo "{fleet.MARK}result {pool}"', f"cat smoke/result_{pool}.json 2> /dev/null",
                   f"job=$(grep -o '\"job_id\": \"[0-9]*\"' smoke/result_{pool}.json 2> /dev/null | head -n 1 "
                   "| cut -d '\"' -f 4)",
                   f'echo; echo "{fleet.MARK}default {pool}"',
                   f'[ -n "$job" ] && head -n {SMOKE_UPDATES} "smoke/{pool}_$job/default/metrics.jsonl" 2> /dev/null',
                   f'echo; echo "{fleet.MARK}reference {pool}"',
                   f"cat smoke/reference_{profile_words(settings['execution_profile'])}.jsonl 2> /dev/null"]
    answer = fleet.remote(run_dir, table, name, "\n".join(script + [f'echo; echo "{fleet.MARK}end"', "exit 0"]) + "\n")
    # before: "@@fleet:result ai-h100\n{...json...}\n@@fleet:default ai-h100\n{line}\n...\n@@fleet:end"
    # after:  sections {("result", "ai-h100"): "{...json...}", ("default", "ai-h100"): "{line}\n...", ...}
    sections, key, text = {}, None, []
    for line in answer.splitlines() + [f"{fleet.MARK}end"]:
        if line.startswith(fleet.MARK):
            if key and "".join(text).strip():
                sections[key] = "\n".join(text)
            words = line[len(fleet.MARK):].split()
            key, text = (tuple(words) if len(words) == 2 else None), []
        else:
            text.append(line)
    results = {}
    for (kind, pool), text in sections.items():
        if kind == "result":
            lines = {part: [json.loads(line) for line in sections.get((part, pool), "").splitlines() if line.strip()]
                     for part in ("reference", "default")}
            results[pool] = read_first_update_again(json.loads(text), lines["reference"], lines["default"])
    for pool, result in results.items():
        print(f"{name}, pool {pool}: {'PASSED' if result.get('passed') else 'NOT PASSED'} (job {result.get('job_id')}, "
              f"{result.get('device')}, written {common.pacific(result['written_at'])})")
        for check in result.get("checks", []):
            print(f"    {'ok ' if check['passed'] else 'NOT'} {check['name']}: {check['detail']}"
                  + (" (read again on UVA)" if check.get("read_again_on") else ""))
        if result.get("job_archive_check"):
            job_check = result["job_archive_check"]
            print(f"    the job's own comparison, by its frozen code: {'passed' if job_check['passed'] else 'NOT passed'}"
                  f", {job_check['detail']}")
        for line in result.get("speed_lines", []):
            print("    " + line)
    # the table keeps each pool's last verdict read
    entry.setdefault("smoke", {}).update(
        {pool: {key: result.get(key) for key in ("passed", "job_id", "device", "written_at", "speed", "first_update_rule")}
         for pool, result in results.items()})
    return results


def smoke_result(run_dir, table, name):
    """Read a server's smoke results, print them, and mark the server ready when every pool passed."""
    entry = table[name]
    results = read_smoke_results(run_dir, table, name, list(entry["pools"]))
    ready, reason = smoke_verdict(entry, results)
    entry["ready"] = ready
    if ready:
        entry.pop("not_ready_reason", None)
    else:
        entry["not_ready_reason"] = reason
    common.write_json(fleet.table_path(run_dir), table)
    fleet.log(run_dir, f"{name} | smoke-result | " + ("READY: every pool's smoke job passed" if ready
                                                       else f"not ready: {reason}"))
    return ready


def held_pool_result(run_dir, table, name, pool):
    """Read the smoke result of one held-back pool; a passed pool goes back to the pools that take units. Returns passed."""
    # before: pools {"advanced-h200": ...}, pools_held_back {"preemptable-h200": {..., "units_at_once_when_back": 0}},
    #         the pool's smoke job passed
    # after:  pools {"advanced-h200": ..., "preemptable-h200": {...}}, hand_over.pools["preemptable-h200"] =
    #         {"units_at_once": 0}, nothing held back; whether the server is ready is not changed
    entry = table[name]
    if pool not in held_back_pools(entry):
        raise SystemExit(f"{name} holds no pool {pool!r} back (pools_held_back: {sorted(held_back_pools(entry))})")
    if "units_at_once_when_back" not in entry["pools_held_back"][pool]:
        raise SystemExit(f"the held-back pool {pool!r} of {name} names no units_at_once_when_back")
    result = read_smoke_results(run_dir, table, name, [pool]).get(pool)
    passed = result is not None and result.get("passed") is True
    if passed:
        settings = dict(entry["pools_held_back"].pop(pool))
        entry["hand_over"]["pools"][pool] = {"units_at_once": settings.pop("units_at_once_when_back")}
        entry["pools"][pool] = settings
    common.write_json(fleet.table_path(run_dir), table)
    fleet.log(run_dir, f"{name} | smoke-result | pool {pool}: " + (
        "PASSED; it takes units again" if passed else
        f"{'no smoke result yet' if result is None else 'NOT passed'}; it stays held back"))
    # the pool's job script is written again, so it holds this run folder's code copy
    if passed:
        prepare(run_dir, table, name, again=True)
    return passed


def adapt(run_dir):
    """Give the grid campaign's remote fleet this study's worker job, preparation, status, checks and returns."""
    fleet.WORKER_TEMPLATE = WORKER_TEMPLATE
    fleet.REMOTE_FOLDERS = REMOTE_FOLDERS
    fleet.worker_script = worker_script
    fleet.prepare = prepare
    fleet.verify_unit = verify_unit
    fleet.status_script = status_script
    fleet.parse_status = parse_status
    fleet.result_block = result_block
    fleet.bring_back = bring_back
    # jobs of this run folder are told from every other folder's by their name
    fleet.JOB_PREFIX = job_prefix(run_dir)
    # a unit whose job ended continues from its checkpoint on the same server, so it may be put back in
    # that server's queue many times (a 4-hour idle-card job ends long before a unit does)
    fleet.RECOVERY_LIMIT = 100


def main(argv=None):
    """Run one command under the fleet lock."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-table").add_argument("--force", action="store_true")
    preparing = commands.add_parser("prepare")
    preparing.add_argument("--server", required=True)
    preparing.add_argument("--again", action="store_true", help="send the scripts, the code copy and the assets once more")
    smoking = commands.add_parser("smoke")
    smoking.add_argument("--server", required=True)
    smoking.add_argument("--pool", help="only this pool; default: every pool of the server")
    smoking.add_argument("--dry-run", action="store_true")
    reading = commands.add_parser("smoke-result")
    reading.add_argument("--server", required=True)
    reading.add_argument("--pool", help="a pool of pools_held_back: read its verdict only; when it passed, the pool "
                                        "takes units again (the server's ready state is not changed)")
    commands.add_parser("hand-over-tick").add_argument("--dry-run", action="store_true")
    handing = commands.add_parser("hand-over", help="hand pending units of one pool's profile to a server")
    handing.add_argument("--server", required=True)
    handing.add_argument("--pool", required=True)
    handing.add_argument("--units", type=int, required=True,
                         help="units of the pool's profile the server may hold at once after this call")
    handing.add_argument("--match", default="", help="only units whose name holds this text")
    handing.add_argument("--dry-run", action="store_true")
    submitting = commands.add_parser("submit", help="submit worker jobs to one pool of a ready server")
    submitting.add_argument("--server", required=True)
    submitting.add_argument("--pool", required=True)
    submitting.add_argument("--jobs", type=int, required=True)
    submitting.add_argument("--dry-run", action="store_true")
    commands.add_parser("status")
    taking = commands.add_parser("take-back")
    taking.add_argument("--server", required=True)
    taking.add_argument("--units", type=int, required=True)
    taking.add_argument("--from-end", action="store_true")
    taking.add_argument("--match")
    taking.add_argument("--hold", action="store_true",
                        help="put the units in queue/deferred, out of the UVA workers' reach")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    adapt(run_dir)
    # the fleet's lock, its log and the ledgers live in RUN/slurm
    (run_dir / "slurm").mkdir(parents=True, exist_ok=True)
    if args.command == "init-table":
        init_table(run_dir, args.force)
        return
    if args.command == "status":
        # status writes nothing, so it reads the table without the lock
        args.dry_run = True
        fleet.tick(run_dir, common.read_json(fleet.table_path(run_dir)), args, act=False)
        return
    with fleet.fleet_lock(run_dir):
        # the table is read only once the lock is held, so a command that waited on the lock never writes
        # back a table read before another writer's change
        table = common.read_json(fleet.table_path(run_dir))
        if args.command == "prepare":
            prepare(run_dir, table, args.server, args.again)
        elif args.command == "smoke":
            smoke(run_dir, table, args.server, args.pool, args.dry_run)
        elif args.command == "smoke-result" and args.pool:
            held_pool_result(run_dir, table, args.server, args.pool)
        elif args.command == "smoke-result":
            smoke_result(run_dir, table, args.server)
        elif args.command == "hand-over-tick":
            hand_over_tick(run_dir, table, args.dry_run)
        elif args.command == "hand-over":
            fleet.require_ready(table, args.server)
            hand_over(run_dir, table, args.server, args.pool, args.dry_run, args.units, args.match)
        elif args.command == "submit":
            fleet.submit(run_dir, table, args)
        else:
            fleet.take_back(run_dir, table, args)


if __name__ == "__main__":
    main()
