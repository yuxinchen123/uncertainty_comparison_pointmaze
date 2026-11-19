"""The pack-size profile of the stage-1 grid on one MIT H200: four copies against eight copies in one process.

    python pack_size_profile.py --code CODE --out OUT [--setting NAME ...] [--updates 60]

It runs inside one Slurm job holding one H200 (pack_size_profile_job.sh). Each setting is one train_final.py
process of the run's frozen code copy, with the "four copies" execution profile's learner and environment
settings, on its own run folder OUT/<setting>/run. A setting saves and ends by itself after TRAINER_SECONDS
(startup, compilation and the 50 random warm-up rollouts included) and is stopped with SIGTERM at
KILL_SECONDS, so no setting holds the card for more than 15 minutes.

The settings:
  default_copies-4_seeds-1-to-4         the default program (RND), seeds 1 to 4: the archived four-copy run's seeds
  default_copies-8_seeds-1-to-8         the same with eight copies, seeds 1 to 8; the reset lane budget doubled
                                        (16 -> 32) with the copies, which changes the speed only
  fast-layer_copies-4_seeds-13-to-16    the fast last layer at sharing group 16 with the game events and the bonus
                                        profile, the four step sizes of a stage-1 pack, weight 1
  fast-layer_copies-8_seeds-13-to-20    the same with eight copies: the four step sizes at weights 1 and 2

What each setting leaves in OUT/<setting>/: the trainer's run folder, its log, and summary.json (exit code,
updates written, seconds per update, peak card memory). Seconds per update is the median gap between the writes
of consecutive metric lines, the first two updates (compilation) left out, as smoke_check.py measures it.
Standard library only; nothing numerical is imported here.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

STEPS_PER_UPDATE = 128 * 128
TRAINER_SECONDS = 780        # the trainer saves and ends after its first update past this many seconds
KILL_SECONDS = 870           # a setting still running then gets SIGTERM (it finishes its update and saves)
STEP_SIZES = [0.01, 0.03, 0.1, 0.3]
FAST_LAYER = {"fast_last_layer_step": 0.01, "fast_last_layer_sharing_group": 16, "intrinsic_coefficient": 1.0,
              "record_game_events": True, "record_bonus_profile": True}


def settings(profile):
    """The four settings: copies, copy seeds, learner and environment of each, from the four-copy profile."""
    # before: profile {"learner": {"convolution_algorithm": "space_to_depth", ...}, "environment": {..., "reset_lane_budget": 16}}
    # after:  {"default_copies-8_seeds-1-to-8": (8, [1, ..., 8], {learner of the profile},
    #          {environment of the profile, "reset_lane_budget": 32}), ...}
    learner, environment = profile["learner"], profile["environment"]
    eight = dict(environment, reset_lane_budget=2 * environment["reset_lane_budget"])
    fast = {**learner, **FAST_LAYER}
    return {
        "default_copies-4_seeds-1-to-4": (4, [1, 2, 3, 4], dict(learner), dict(environment)),
        "default_copies-8_seeds-1-to-8": (8, list(range(1, 9)), dict(learner), eight),
        "fast-layer_copies-4_seeds-13-to-16": (
            4, [13, 14, 15, 16], dict(fast, per_copy_scalar_settings={"fast_last_layer_step": STEP_SIZES}),
            dict(environment, report_game_events=True)),
        "fast-layer_copies-8_seeds-13-to-20": (
            8, list(range(13, 21)), dict(fast, per_copy_scalar_settings={
                "fast_last_layer_step": STEP_SIZES * 2, "intrinsic_coefficient": [1.0] * 4 + [2.0] * 4}),
            dict(eight, report_game_events=True)),
    }


def card_memory_used():
    """Read the card's used memory now, in MB."""
    answer = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(answer.split()[0])


def run_setting(python, code, folder, copies, seeds, learner, environment, updates):
    """Run one setting in its own trainer process; returns its summary."""
    run = folder / "run"
    command = [python, "-u", str(code / "scripts/atari_pixel_campaign/train_final.py"), "--run-dir", str(run),
               "--copies", str(copies), "--copy-seeds", ",".join(map(str, seeds)), "--target-updates", str(updates),
               "--observation-warmup-iterations", "50", "--checkpoint-seconds", "1200",
               "--time-limit-seconds", str(TRAINER_SECONDS), "--learner", json.dumps(learner),
               "--environment", json.dumps(environment)]
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] start {folder.name}: {' '.join(command)}", flush=True)
    peak, started, stopped = 0, time.monotonic(), False
    with (folder / "trainer.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=code, start_new_session=True)
        # the card's used memory every two seconds while the trainer runs; past KILL_SECONDS the stop signal
        while process.poll() is None:
            peak = max(peak, card_memory_used())
            if time.monotonic() - started > KILL_SECONDS and not stopped:
                os.killpg(process.pid, signal.SIGTERM)
                stopped = True
            time.sleep(2)
    metrics = run / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
    # seconds per update: the median gap between consecutive writes, the first two updates left out
    gaps = [after["unix"] - before["unix"] for before, after in zip(lines[2:], lines[3:])]
    seconds = statistics.median(gaps) if gaps else None
    summary = {"setting": folder.name, "copies": copies, "copy_seeds": seeds, "exit_code": process.returncode,
               "stopped_by_signal": stopped, "wall_seconds": round(time.monotonic() - started, 1),
               "updates_written": len(lines), "timed_gaps": len(gaps), "seconds_per_update": seconds,
               "gap_range": [min(gaps), max(gaps)] if gaps else None, "peak_card_memory_mb": peak,
               "total_steps_per_second": copies * STEPS_PER_UPDATE / seconds if seconds else None}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{time.strftime('%H:%M:%S')}] end {folder.name}: " + json.dumps(summary), flush=True)
    return summary


def main(argv=None):
    """Run the chosen settings one after another on the job's card."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", type=Path, required=True, help="the platform folder of the frozen code copy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--setting", action="append", help="a setting's name; default: all four, in order")
    parser.add_argument("--updates", type=int, default=60)
    args = parser.parse_args(argv)
    profile = json.loads((args.code / "scripts/pixel_sweep_campaign/card_profiles.json").read_text())["profiles"]["four copies"]
    chosen = settings(profile)
    for name in args.setting or list(chosen):
        copies, seeds, learner, environment = chosen[name]
        run_setting(sys.executable, args.code, args.out / name, copies, seeds, learner, environment, args.updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
