"""A generic worker of the pixel sweep: one process on one card that takes units from the queue until none fits.

    python worker.py --run-dir RUN [--card-profiles FILE] [--remote-records] [--server NAME] [--max-seconds N]
                     [--minimum-seconds N] [--margin-seconds N]

The worker reads its card, takes the execution profile of that card (card_profiles.json: "one copy"
under 60 GB, "four copies" from 60 GB, "eight copies" from 140 GB) and claims the next pending unit
of that profile, or a unit that started under a profile its own continues (an eight-copy worker
continues four-copy packs). It runs
scripts/atari_pixel_campaign/train_final.py on the unit in RUN/data/<unit>/run/ and then
  - a unit whose progress.json says completed at its target moves to done;
  - a unit whose trainer ended with exit code 0 without that stopped with its state saved (time
    limit or stop signal): it returns to pending, and a later job continues it from its checkpoint;
  - a unit that ran out of card memory returns to pending for cards with more memory;
  - a unit whose trainer failed returns to pending, and moves to failed at its third failure.
The trainer saves its own checkpoints and stops by itself at its time limit, which is what is left
of the job (read from the scheduler) less a margin. The worker only ever sends the trainer SIGTERM,
on which the trainer finishes its update, saves and ends; it never ends the trainer in a way that
loses the checkpoint. Before each start it adds a line with the card and the copies in the process
(the pack size) to the run folder's events.jsonl, so a unit that continued on another card type can
be told, and every run's record says how many copies shared its process.

The worker ends when nothing it can run is pending, when too little of the job is left, after a unit
that did not complete, or when RUN/STOP exists. A queued job holds no unit: the claim is made here,
after the job has started. The worker imports no numerical library.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from campaign_imports import PLATFORM, common, grid, maze, unit_queue
import pixel_queue

maze_worker = maze("worker")
TRAINER = PLATFORM / "scripts/atari_pixel_campaign/train_final.py"
MARGIN_SECONDS = 900           # kept at the end of the job for the trainer to finish its update, save and end
MINIMUM_SECONDS = 1800         # a unit is started only when it can train at least this long
SAMPLE_SECONDS = 60            # how often the card's used memory and the trainer's speed are read
# the share of card memory JAX may take, by card name where the job environment's 0.85 is too little: on the
# RTX 2080 Ti the default update ran out of card memory at 0.85 (gate 6, job 6955822: one allocation of
# 5.10 GiB) and ran at 0.95 (job 6955829). Every other card keeps 0.85, the setting of the archived baseline
MEMORY_FRACTION_BY_CARD = {"NVIDIA GeForce RTX 2080 Ti": "0.95"}
# set by a stop signal; the trainer that runs is told to stop, and no further unit is claimed
STOP = {"requested": False, "process": None}


def request_stop(*_):
    """Pass a stop signal on to the trainer, which finishes its update, saves its state and ends."""
    STOP["requested"] = True
    process = STOP["process"]
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)


def trainer_command(run_dir, unit, specification, limit_seconds, trainer):
    """The train_final.py command of one unit; running it again continues the unit from its checkpoint."""
    learner, environment = pixel_queue.trainer_settings(unit, specification)
    return [sys.executable, "-u", str(trainer), "--run-dir", str(pixel_queue.run_folder(run_dir, unit["name"])),
            "--copies", str(specification["copies"]),
            "--copy-seeds", ",".join(str(seed) for seed in specification["copy_seeds"]),
            "--target-updates", str(unit["target_updates"]),
            "--observation-warmup-iterations", str(specification["observation_warmup_iterations"]),
            "--checkpoint-seconds", str(specification["checkpoint_seconds"]),
            "--time-limit-seconds", str(int(limit_seconds)),
            "--learner", json.dumps(learner), "--environment", json.dumps(environment)]


def trainer_environment(card):
    """The environment of the trainer's process: this worker's, with a compilation cache on the node's own disk."""
    # the cache is kept per user and card type, as the archived baseline's units did
    # before: card "NVIDIA RTX A4500", user sl5nw ; after: JAX_COMPILATION_CACHE_DIR=/tmp/sl5nw_pixel_sweep_cache_nvidia-rtx-a4500
    environment = dict(os.environ)
    if "JAX_COMPILATION_CACHE_DIR" not in environment:
        words = re.sub(r"[^a-z0-9]+", "-", card["gpu"].lower()).strip("-")
        cache = Path(environment.get("TMPDIR", "/tmp")) / f"{environment.get('USER', 'user')}_pixel_sweep_cache_{words}"
        cache.mkdir(parents=True, exist_ok=True)
        environment["JAX_COMPILATION_CACHE_DIR"] = str(cache)
    # a card whose program needs a larger share of its memory gets it; the others keep the job's setting
    # before: card "NVIDIA GeForce RTX 2080 Ti", XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ; after: 0.95
    # before: card "NVIDIA RTX A4500", XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 ; after: 0.85
    if card["gpu"] in MEMORY_FRACTION_BY_CARD:
        environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = MEMORY_FRACTION_BY_CARD[card["gpu"]]
    return environment


def append_event(folder, kind, unit, **values):
    """Add one line of this worker to the run folder's events.jsonl, beside the trainer's own lines."""
    # after: {"event": "worker_start", "unix": 1795000000.0, "host": "jaguar03", "slurm_job_id": "6900001",
    #         "source_commit": "19d5...", "device": "NVIDIA RTX A4500", "card_memory_mb": 20470,
    #         "execution_profile": "one copy", "copies": 1, "server": "uva", "attempt": 1, "update": 0,
    #         "memory_fraction": "0.85"}
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": kind, "unix": time.time(), "host": unit["hostname"],
                                 "slurm_job_id": unit["job_id"], "source_commit": unit["source_commit"],
                                 **values}) + "\n")


def classify(complete, return_code, has_checkpoint, log_tail, stop_was_requested):
    """Name how a unit's trainer ended: complete, stopped, out_of_card_memory or failed."""
    # progress.json decides completion, never the exit code
    if complete:
        return "complete"
    # the trainer ends with exit code 0 after it saved its state at the time limit or on a stop signal
    if return_code == 0 and has_checkpoint:
        return "stopped"
    if any(marker in log_tail for marker in maze_worker.OUT_OF_MEMORY_MARKERS):
        return "out_of_card_memory"
    # a stop signal that ended the trainer before its save: it continues from its last checkpoint
    if return_code != 0 and stop_was_requested:
        return "stopped"
    return "failed"


def completed_unit_folder(run_dir, name, summary):
    """Put a complete unit's files and a summary in data/research/<unit>, the folder a remote tick copies to UVA."""
    folder, source = run_dir / "data/research" / name, pixel_queue.run_folder(run_dir, name)
    folder.mkdir(parents=True, exist_ok=True)
    for file_name in pixel_queue.RUN_FILES:
        if (source / file_name).exists():
            link = folder / file_name
            link.unlink(missing_ok=True)
            os.link(source / file_name, link)
    common.write_json(folder / "summary.json", summary)


def run_unit(run_dir, unit, specification, card, limit_seconds, trainer, left, sample_memory, server,
             margin_seconds=MARGIN_SECONDS):
    """Run one claimed unit until its trainer ends; returns (how it ended, details of the attempt)."""
    folder = pixel_queue.run_folder(run_dir, unit["name"])
    command = trainer_command(run_dir, unit, specification, limit_seconds, trainer)
    log_path = run_dir / "logs" / f"{unit['name']}_job-{unit['job_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before = pixel_queue.read_progress(folder)
    # a unit whose trainer reached the target in a job that ended before it could say so needs no trainer
    if pixel_queue.is_complete(before, unit["target_updates"]):
        return "complete", {"exit_code": 0, "wall_seconds": 0.0, "log": str(log_path),
                            "update_at_start": before["update"], "update_at_end": before["update"],
                            "seconds_per_update": None, "card_memory_used_mb": None}
    environment = trainer_environment(card)
    append_event(folder, "worker_start", unit, device=card["gpu"], card_memory_mb=card["memory_mb"],
                 execution_profile=unit["execution_profile"], copies=specification["copies"], server=server,
                 attempt=unit["attempts"], update=before["update"],
                 memory_fraction=environment.get("XLA_PYTHON_CLIENT_MEM_FRACTION"))
    started, largest, speeds = time.monotonic(), None, []
    with log_path.open("a") as log:
        log.write(f"[{common.pacific()}] job {unit['job_id']} on {unit['hostname']}, {unit['card']}: "
                  f"{' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=PLATFORM,
                                   env=environment, start_new_session=True)
        STOP["process"] = process
        try:
            # a stop signal that came between the claim and this start is passed on now
            if STOP["requested"]:
                request_stop()
            while process.poll() is None:
                try:
                    process.wait(timeout=SAMPLE_SECONDS)
                except subprocess.TimeoutExpired:
                    reading = sample_memory(card) if sample_memory else None
                    largest = max((value for value in (largest, reading) if value), default=None)
                    # the trainer's own rate of its last minute, kept only once it has trained in this attempt
                    progress = pixel_queue.read_progress(folder)
                    if progress["seconds_per_update"] and progress["update"] > before["update"]:
                        speeds.append(progress["seconds_per_update"])
                    # the trainer stops by itself at its time limit; this asks again shortly before the job ends
                    if left() < margin_seconds / 2 and not STOP["requested"]:
                        request_stop()
        finally:
            # this worker never leaves a trainer behind (a second writer would spoil the run folder); the
            # signal is the one the trainer saves its state on, and the worker waits for it to end
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            STOP["process"] = None

    # progress.json says whether the unit is complete; the end of the log says why it is not
    after = pixel_queue.read_progress(folder)
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - maze_worker.LOG_TAIL_BYTES))
        log_tail = handle.read().decode(errors="replace")
    outcome = classify(pixel_queue.is_complete(after, unit["target_updates"]), process.returncode,
                       (folder / "checkpoint.pkl").exists(), log_tail, STOP["requested"])
    details = {"exit_code": process.returncode, "wall_seconds": round(time.monotonic() - started, 1),
               "log": str(log_path), "update_at_start": before["update"], "update_at_end": after["update"],
               "seconds_per_update": round(statistics.median(speeds), 4) if speeds else None,
               "card_memory_used_mb": largest}
    append_event(folder, "worker_end", unit, device=card["gpu"], outcome=outcome, update=after["update"],
                 exit_code=process.returncode)
    return outcome, details


def main(argv=None, card=None, trainer=TRAINER, sample_memory=maze_worker.card_memory_used_mb):
    """Take units from the queue one after another until this worker has reason to end; returns its exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--card-profiles", type=Path, default=pixel_queue.PROFILES_FILE,
                        help="the execution profiles; default: the file beside this worker")
    parser.add_argument("--remote-records", action="store_true",
                        help="on a remote server: also write the claim records and the completed-unit "
                             "folders the remote fleet's tick reads")
    parser.add_argument("--server", default="uva", help="the server's name, written into every start record")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="outside a Slurm job: how long this worker may run")
    parser.add_argument("--minimum-seconds", type=float, default=MINIMUM_SECONDS,
                        help="a unit is started only when it can train at least this long")
    parser.add_argument("--margin-seconds", type=float, default=MARGIN_SECONDS,
                        help="kept at the end of the job for the trainer to finish its update, save and end "
                             "(a short check job keeps less than the default)")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    if job_id == "local" and args.max_seconds <= 0:
        parser.error("outside a Slurm job, --max-seconds says how long this worker may run")
    os.umask(0o002)
    started = time.monotonic()

    # the card is read with nvidia-smi before anything else, and every start of the job is recorded
    card = card or grid("worker").hardware()
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
    visible = (card.get("cuda_visible_devices") or "0").split(",")[0]
    folder = run_dir / "slurm/jobs" / f"{job_id}_{card['hostname']}_card-{visible}"
    common.write_json(folder / "hardware.json", card)
    with (run_dir / "slurm/job_starts.txt").open("a") as handle:
        handle.write(f"{job_id} {card['hostname']} {common.now()} restart={restart_count}\n")
    # the card decides the profile, and the profile decides which units this worker may take
    profiles = pixel_queue.read_profiles(args.card_profiles)
    profile_name = pixel_queue.profile_for_card(profiles, card["memory_mb"])
    if profile_name is None:
        print(f"[{common.pacific()}] the card {card['gpu']} ({card['memory_mb']} MB) is below every execution "
              "profile's memory; this worker ends", flush=True)
        return 0
    claimant = {"job_id": job_id, "restart_count": restart_count, "account": os.environ.get("USER"),
                "hostname": card["hostname"], "card": card["gpu"], "card_memory_mb": card["memory_mb"],
                "source_commit": maze_worker.source_commit(), "copy_number": maze_worker.source_copy_number()}
    # a requeued job starts again under the same id: what its earlier start claimed goes back to pending
    for name in unit_queue.release_claims_of_job(run_dir, job_id, restart_count):
        print(f"[{common.pacific()}] released {name}, claimed by an earlier start of this job", flush=True)

    STOP.update(requested=False, process=None)
    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    exit_code = 0
    try:
        while not STOP["requested"] and not (run_dir / "STOP").exists():
            # the job's end is read again before every claim; the trainer must be able to save its
            # state and end before that
            ends_at = time.monotonic() + maze_worker.seconds_left(started, args.max_seconds)
            limit = ends_at - time.monotonic() - args.margin_seconds
            if limit < args.minimum_seconds:
                print(f"[{common.pacific()}] {int(limit)} s of training time are left, under "
                      f"{int(args.minimum_seconds)} s; this worker ends", flush=True)
                break
            unit = pixel_queue.claim(run_dir, card["memory_mb"], claimant, profile_name, profiles[profile_name])
            if unit is None:
                left = pixel_queue.pending_for_others(run_dir, profile_name, profiles[profile_name],
                                                      card["memory_mb"], claimant["copy_number"])
                others = ", ".join(f"{count} for {reason}" for reason, count in left.items() if count)
                print(f"[{common.pacific()}] nothing a card of the profile {profile_name!r} can run is pending"
                      + (f" (pending units left: {others})" if others else "") + "; this worker ends", flush=True)
                break
            print(f"[{common.pacific()}] start {unit['name']} (attempt {unit['attempts']}, profile "
                  f"{unit['execution_profile']!r})", flush=True)
            common.write_json(folder / "assignment.json", unit)
            if args.remote_records:
                maze_worker.write_claim_record(run_dir, unit)
            try:
                specification = common.read_json(run_dir / unit["specification"])
                outcome, details = run_unit(run_dir, unit, specification, card, limit, trainer,
                                            lambda: ends_at - time.monotonic(), sample_memory, args.server,
                                            args.margin_seconds)
            except BaseException:
                # a worker that cannot go on gives its claim back first; the error is raised again
                unit_queue.give_back(run_dir, unit, "worker_error", counts_as_failure=True)
                raise
            details = {key: value for key, value in details.items() if value is not None}

            if outcome == "complete":
                if args.remote_records:
                    # the claim record moves first, so a unit seen in done/ has its record beside it
                    maze_worker.close_claim_record(run_dir, unit, "done", details["exit_code"], details["wall_seconds"])
                    completed_unit_folder(run_dir, unit["name"], {
                        "completed": True, "unit_id": unit["name"], "target_updates": unit["target_updates"],
                        "execution_profile": unit["execution_profile"],
                        "execution_profile_settings": unit["execution_profile_settings"],
                        "card": unit["card"], "attempt": details, "earlier_attempts": unit.get("history", [])})
                unit_queue.finish(run_dir, unit, details)
                print(f"[{common.pacific()}] done {unit['name']} in {details['wall_seconds']} s", flush=True)
                continue

            # a unit that did not complete goes back to the queue, and this worker takes no further
            # unit: one fault on this node must not move a row of units toward failed
            failed = outcome == "failed"
            if args.remote_records:
                maze_worker.close_claim_record(run_dir, unit, unit_queue.next_folder(unit, failed),
                                               details["exit_code"], details["wall_seconds"])
            target = unit_queue.give_back(
                run_dir, unit, {"stopped": "stopped_with_state_saved"}.get(outcome, outcome), counts_as_failure=failed,
                minimum_device_memory_mb=card["memory_mb"] + 1 if outcome == "out_of_card_memory" else None,
                **details)
            print(f"[{common.pacific()}] {unit['name']} {outcome.replace('_', ' ')} at update "
                  f"{details['update_at_end']} of {unit['target_updates']} (exit code {details['exit_code']}); "
                  f"it is now in queue/{target}; log {details['log']}", flush=True)
            exit_code = 0 if outcome == "stopped" else 1
            break
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
