"""A generic worker of the maze campaign: one process on one card that takes units from the queue until none fits.

    python worker.py --run-dir RUN [--observed-dir DIR] [--remote-records] [--max-seconds N]

The worker claims the next pending unit its card can hold and runs scripts/run_training.py on it,
with a checkpoint every 20 minutes and a time limit taken from what is left of the job. The unit's
process runs with this worker's own interpreter, or with the one the unit's specification names
(`interpreter`: a unit whose AntMaze physics is MuJoCo Warp needs platform_jax_warp). Then
  - a unit whose shard ends with a unit_complete record moves to done, and one line for the
    copies-per-card skill is added to <observed dir>/<workload>.jsonl;
  - a unit that stopped with its state saved (time limit, stop signal) returns to pending, and a
    later job continues it from its checkpoint;
  - a unit whose process failed returns to pending, and moves to failed at its third failure,
    with the path of its log; a unit that ran out of card memory returns to pending for cards
    with more memory than this one.
The worker ends when nothing it can run is pending, when too little of the job is left, after a
unit that did not complete, or when RUN/STOP exists. A queued job holds no unit: the claim is made
here, after the job has started. The worker itself imports no numerical library.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from grid_campaign import common, load
import unit_queue

PLATFORM = Path(__file__).resolve().parents[2]
TRAINER = PLATFORM / "scripts/run_training.py"
CHECKPOINT_SECONDS = 1200      # the unit saves its state at the first window end after this long
MARGIN_SECONDS = 600           # kept at the end of the job for the unit to save its state and stop
MINIMUM_SECONDS = 900          # a unit is started only when it can train at least this long
MEMORY_SAMPLE_SECONDS = 60     # how often the card's used memory is read while a unit runs
LOG_TAIL_BYTES = 65536
# run_training.py ends a unit it stopped at the time limit or on a stop signal with this sentence
STOP_SENTENCE = "with its state saved"
# what the runtime prints when the card's memory is the cause (the list of device/series.py)
OUT_OF_MEMORY_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory", "OUT_OF_MEMORY",
                         "CUDA_ERROR_OUT_OF_MEMORY")
# set by a stop signal; the unit that runs is told to stop, and no further unit is claimed
STOP = {"requested": False, "process": None}


def request_stop(*_):
    """Pass a stop signal on to the unit that runs, which then saves its state and ends."""
    STOP["requested"] = True
    process = STOP["process"]
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)


def source_commit():
    """Name the frozen code copy this worker runs from; a worker started from the working tree says so."""
    marker = PLATFORM / "SOURCE_COMMIT"
    return marker.read_text().strip() if marker.exists() else "working tree"


def job_end(job_id):
    """Read from the scheduler when this job ends, in seconds since 1970."""
    row = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True)
    return datetime.fromisoformat(load("submit").slurm_fields(row)["EndTime"]).astimezone().timestamp()


def seconds_left(started, max_seconds):
    """Seconds until the job ends: read from the scheduler inside a job, counted from the start outside one."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return job_end(job_id) - time.time()
    return max_seconds - (time.monotonic() - started)


def card_memory_used_mb(card):
    """Read the card's used memory now, in MB; None when the query gives no number."""
    visible = (card.get("cuda_visible_devices") or "0").split(",")[0]
    result = subprocess.run(["nvidia-smi", "-i", visible if visible.isdecimal() else "0",
                             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True)
    return int(result.stdout.split()[0]) if result.returncode == 0 and result.stdout.strip() else None


def write_unit_configuration(run_dir, specification):
    """Make sure the unit's configuration file holds exactly the text of its specification; returns its path."""
    path = unit_queue.configuration_path(run_dir, specification["unit_id"])
    if not path.exists():
        # a server the unit was copied to has only the specification; the file is written under
        # another name first, so no reader sees half of it
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(specification["configuration_toml"])
        os.replace(temporary, path)
    elif path.read_text() != specification["configuration_toml"]:
        raise ValueError(f"{path} differs from the configuration in the unit's specification")
    return path


def unit_interpreter(specification, remote=False):
    """The interpreter that runs a unit: the one its specification names, else this worker's own."""
    # before: {"unit_id": "...", "interpreter": "/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python"}
    # after:  that path; a specification without the key gives sys.executable, as before the key existed
    # On a remote server the named path is UVA's and does not exist. The server table names ONE environment
    # per server (the worker's own), built to hold every package a unit of the study needs, MuJoCo Warp
    # included, so there the worker's own interpreter runs every unit.
    if remote:
        return sys.executable
    interpreter = specification.get("interpreter", sys.executable)
    if not Path(interpreter).exists():
        raise FileNotFoundError(f"unit {specification['unit_id']} needs the interpreter {interpreter}, "
                                "which does not exist on this machine")
    return interpreter


def unit_command(run_dir, specification, configuration, limit_seconds, trainer, remote=False):
    """The run_training.py command of one unit; running it again continues the unit from its checkpoint."""
    return [unit_interpreter(specification, remote), str(trainer), "--run-dir", str(run_dir), "--unit-id", specification["unit_id"],
            "--configuration-file", str(configuration), *specification["options"],
            "--checkpoint-seconds", str(CHECKPOINT_SECONDS), "--time-limit-seconds", str(int(limit_seconds)),
            "--no-aggregate"]


def classify(shard_complete, return_code, log_tail, stop_was_requested):
    """Name how a unit's process ended: complete, stopped, out_of_card_memory or failed."""
    # the shard decides completion, never the exit code: a complete unit is done whatever came after
    if shard_complete:
        return "complete"
    # a stop the unit carried out itself, or a stop signal that ended it before it could
    if return_code != 0 and (STOP_SENTENCE in log_tail or stop_was_requested):
        return "stopped"
    if any(marker in log_tail for marker in OUT_OF_MEMORY_MARKERS):
        return "out_of_card_memory"
    return "failed"


def observed_line(specification, card, completion, job_id, card_memory_used):
    """One line for the copies-per-card skill: the size and the speed this unit had on this card."""
    # after: {"workload": "...", "device_name": "NVIDIA RTX A4500", "copies": 960,
    #         "seconds_per_iteration": 0.071, "host": "jaguar03", "unit_id": "...", ...}
    line = {"workload": specification["workload"], "device_name": card["gpu"], "copies": completion["copies"],
            "seconds_per_iteration": completion["seconds_per_iteration_steady"],
            "environments_per_copy": specification["environments_per_policy"],
            "iterations_measured": completion.get("iterations_this_attempt", completion["iterations"]),
            "host": card["hostname"], "unit_id": specification["unit_id"], "job_id": job_id,
            "written_at": common.now()}
    # the skill's peak is the framework's own peak in use; the card's used memory is kept under
    # its own name, because the framework holds more than it uses
    if completion.get("peak_memory_mb"):
        line["peak_memory_mb"] = completion["peak_memory_mb"]
    if card_memory_used:
        line["card_memory_used_mb_largest_reading"] = card_memory_used
    return line


def write_claim_record(run_dir, unit):
    """Write the claim record a remote server's status call reads: job, host, card and start."""
    # after: "job=2331041 host=ac053 gpu=NVIDIA_H100_PCIe started=2026-11-09T22:24:58-06:00"
    (run_dir / "queue/running" / f"{unit['name']}.job").write_text(
        f"job={unit['job_id']} host={unit['hostname']} gpu={unit['card'].replace(' ', '_')} "
        f"started={unit['started_at']}\n")


def close_claim_record(run_dir, unit, folder, exit_code, wall_seconds):
    """Add the attempt's end to the claim record and move it beside the unit's next place."""
    record = run_dir / "queue/running" / f"{unit['name']}.job"
    with record.open("a") as handle:
        handle.write(f"exit={exit_code} wall_seconds={int(wall_seconds)} ended={common.now()}\n")
    # a unit that returns to pending keeps no claim record in the queue; that record goes to the logs
    target = (run_dir / "queue" / folder / record.name if folder in ("done", "failed") else
              run_dir / "logs" / f"{unit['name']}_{unit['job_id']}_released_{int(time.time())}.job")
    os.rename(record, target)


def completed_unit_folder(run_dir, name, summary):
    """Put a complete unit's shard, log and summary in data/research/<unit>, the folder a remote tick copies to UVA."""
    folder = run_dir / "data/research" / name
    folder.mkdir(parents=True, exist_ok=True)
    for source in (run_dir / "data" / f"{name}.jsonl", run_dir / "logs" / f"{name}.log"):
        link = folder / source.name
        link.unlink(missing_ok=True)
        os.link(source, link)
    common.write_json(folder / "summary.json", summary)


def run_unit(run_dir, unit, specification, card, limit_seconds, trainer, left, sample_memory, remote=False):
    """Run one claimed unit until it ends or its time limit; returns (how it ended, details of the attempt)."""
    configuration = write_unit_configuration(run_dir, specification)
    command = unit_command(run_dir, specification, configuration, limit_seconds, trainer, remote)
    log_path = run_dir / "logs" / f"{unit['name']}_job-{unit['job_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started, largest = time.monotonic(), None
    with log_path.open("a") as log:
        log.write(f"[{common.pacific()}] job {unit['job_id']} on {unit['hostname']}, {unit['card']}: "
                  f"{' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=PLATFORM,
                                   start_new_session=True)
        STOP["process"] = process
        try:
            # a stop signal that came between the claim and this start is passed on now
            if STOP["requested"]:
                request_stop()
            while process.poll() is None:
                try:
                    process.wait(timeout=MEMORY_SAMPLE_SECONDS)
                except subprocess.TimeoutExpired:
                    reading = sample_memory(card) if sample_memory else None
                    largest = max((value for value in (largest, reading) if value), default=None)
                    # the unit stops by itself at its time limit; this asks again shortly before the job ends
                    if left() < MARGIN_SECONDS / 2 and not STOP["requested"]:
                        request_stop()
        finally:
            # this worker never leaves a unit's process behind: a second writer would spoil the shard
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            STOP["process"] = None

    # the shard says whether the unit is complete; the end of the log says why it is not
    progress = unit_queue.shard_progress(run_dir / "data" / f"{unit['name']}.jsonl")
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - LOG_TAIL_BYTES))
        log_tail = handle.read().decode(errors="replace")
    outcome = classify(progress["complete"], process.returncode, log_tail, STOP["requested"])
    return outcome, {"exit_code": process.returncode, "wall_seconds": round(time.monotonic() - started, 1),
                     "log": str(log_path), "windows_written": progress["windows_written"],
                     "card_memory_used_mb": largest, "completion": progress["completion"]}


def main(argv=None, card=None, trainer=TRAINER, sample_memory=card_memory_used_mb):
    """Take units from the queue one after another until this worker has reason to end; returns its exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--observed-dir", type=Path,
                        help="the platform's device/observed folder; a complete unit adds one line there")
    parser.add_argument("--remote-records", action="store_true",
                        help="on a remote server: also write the claim records and the completed-unit "
                             "folders the grid campaign's remote fleet reads")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="outside a Slurm job: how long this worker may run")
    parser.add_argument("--minimum-seconds", type=float, default=MINIMUM_SECONDS,
                        help="a unit is started only when it can train at least this long")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    if job_id == "local" and args.max_seconds <= 0:
        parser.error("outside a Slurm job, --max-seconds says how long this worker may run")
    os.umask(0o002)
    started = time.monotonic()

    # the card is read with nvidia-smi before anything else, and every start of the job is recorded
    card = card or load("worker").hardware()
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
    visible = (card.get("cuda_visible_devices") or "0").split(",")[0]
    folder = run_dir / "slurm/jobs" / f"{job_id}_{card['hostname']}_card-{visible}"
    common.write_json(folder / "hardware.json", card)
    with (run_dir / "slurm/job_starts.txt").open("a") as handle:
        handle.write(f"{job_id} {card['hostname']} {common.now()} restart={restart_count}\n")
    claimant = {"job_id": job_id, "restart_count": restart_count, "account": os.environ.get("USER"),
                "hostname": card["hostname"], "card": card["gpu"], "card_memory_mb": card["memory_mb"],
                "source_commit": source_commit()}
    # a requeued job starts again under the same id: what its earlier start claimed goes back to pending
    for name in unit_queue.release_claims_of_job(run_dir, job_id, restart_count):
        print(f"[{common.pacific()}] released {name}, claimed by an earlier start of this job", flush=True)

    STOP.update(requested=False, process=None)
    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    exit_code = 0
    try:
        while not STOP["requested"] and not (run_dir / "STOP").exists():
            # the job's end is read again before every claim; the unit must be able to save its
            # state and stop before that end
            ends_at = time.monotonic() + seconds_left(started, args.max_seconds)
            limit = ends_at - time.monotonic() - MARGIN_SECONDS
            if limit < args.minimum_seconds:
                print(f"[{common.pacific()}] {int(limit)} s of training time are left, under "
                      f"{int(args.minimum_seconds)} s; this worker ends", flush=True)
                break
            unit = unit_queue.claim(run_dir, card["memory_mb"], claimant)
            if unit is None:
                print(f"[{common.pacific()}] nothing this card can run is pending; this worker ends", flush=True)
                break
            print(f"[{common.pacific()}] start {unit['name']} (attempt {unit['attempts']})", flush=True)
            common.write_json(folder / "assignment.json", unit)
            if args.remote_records:
                write_claim_record(run_dir, unit)
            try:
                specification = common.read_json(run_dir / unit["specification"])
                outcome, details = run_unit(run_dir, unit, specification, card, limit, trainer,
                                            lambda: ends_at - time.monotonic(), sample_memory,
                                            remote=args.remote_records)
            except BaseException:
                # a worker that cannot go on gives its claim back first; the error is raised again
                unit_queue.give_back(run_dir, unit, "worker_error", counts_as_failure=True)
                raise
            completion = details.pop("completion")
            details = {key: value for key, value in details.items() if value is not None}

            if outcome == "complete":
                observed = observed_line(specification, card, completion, job_id, details.get("card_memory_used_mb"))
                if args.observed_dir:
                    common.append_json(args.observed_dir / f"{specification['workload']}.jsonl", observed)
                if args.remote_records:
                    # the claim record moves first, so a unit seen in done/ has its record beside it
                    close_claim_record(run_dir, unit, "done", details["exit_code"], details["wall_seconds"])
                    completed_unit_folder(run_dir, unit["name"], {
                        "completed": True, "unit_id": unit["name"], "wall_seconds": details["wall_seconds"],
                        "windows": completion["windows"], "observed": observed})
                unit_queue.finish(run_dir, unit, dict(details, observed=observed))
                print(f"[{common.pacific()}] done {unit['name']} in {details['wall_seconds']} s", flush=True)
                continue

            # a unit that did not complete goes back to the queue, and this worker takes no further
            # unit: one fault on this node must not move a row of units toward failed
            failed = outcome == "failed"
            if args.remote_records:
                close_claim_record(run_dir, unit, unit_queue.next_folder(unit, failed), details["exit_code"],
                                   details["wall_seconds"])
            target = unit_queue.give_back(
                run_dir, unit, {"stopped": "stopped_with_state_saved"}.get(outcome, outcome), counts_as_failure=failed,
                minimum_device_memory_mb=card["memory_mb"] + 1 if outcome == "out_of_card_memory" else None,
                **details)
            print(f"[{common.pacific()}] {unit['name']} {outcome.replace('_', ' ')} (exit code "
                  f"{details['exit_code']}); it is now in queue/{target}; log {details['log']}", flush=True)
            exit_code = 0 if outcome == "stopped" else 1
            break
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
