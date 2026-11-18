"""The smoke job of a server (or of a card type on UVA): its check before it gets units, and its speed check.

    python smoke_check.py --run-dir RUN --code CODE --server S --pool P --profile four-copies [--updates 20]

It runs inside one Slurm job of at most 20 minutes, on the card the job holds, and does in order:

  1. the card: its name and memory (nvidia-smi), and that its execution profile is the pool's;
  2. the asset pack: the SHA-256 of every file under XDG_DATA_HOME against RUN/smoke/asset_sha256.json;
  3. imports, the card under jax, and the start room's geometry (smoke_trainer_checks.py, its own process);
  4. 20 default updates (train_final.py with the profile's settings and the archived run's seeds, no
     fast layer, no event records), compared line by line with the archived baseline's first lines
     (RUN/smoke/reference_<profile>.jsonl). The first update must agree: whole numbers and true/false
     values equal, and every other number within |new - archive| <= 0.001 * |archive| + 1e-6 (decision
     of 2026-11-18: the bits differ between card types, and a value near zero, such as a policy loss of
     1.5e-5, may differ by 5e-8 in single-precision rounding). Later updates on another card type may
     differ in their last bits, so for them the number of differing fields is reported and does not decide;
  5. 20 updates with the fast last layer (step size 0.1, sharing group 16) and the game events: the
     run must complete with both record files folded to its last update;
  6. speed and peak card memory of both runs, with the five columns of every throughput table.

The verdict goes to RUN/smoke/result_<pool>.json and to the job's output. This file itself imports
no numerical library; steps 3 to 5 run in processes of their own.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STEPS_PER_UPDATE = 128 * 128
FIRST_LINE_RELATIVE = 1e-3  # a number of the first update may differ by this share of the archived value
FIRST_LINE_ABSOLUTE = 1e-6  # plus this much, so a value near zero may differ in its last bits
FIRST_LINE_RULE = "|new - archive| <= 1e-3 * |archive| + 1e-6, whole numbers equal"
TRAINER_SECONDS = 420       # each of the two short trainings saves and ends by itself after this long
FAST_LAYER = {"fast_last_layer_step": 0.1, "fast_last_layer_sharing_group": 16, "record_game_events": True,
              "record_bonus_profile": True}


def card():
    """Read the job's card: its name and its memory in MB."""
    answer = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                     text=True).splitlines()[0]
    name, memory = (part.strip() for part in answer.split(","))
    return name, int(memory)


def card_memory_used():
    """Read the card's used memory now, in MB."""
    answer = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(answer.split()[0])


def asset_check(root, expected):
    """Compare every file of the asset pack with the hashes UVA sent; returns a check record."""
    # before: expected {"jaxtari/asset_source.json": "6e52...", ...}, the pack under XDG_DATA_HOME
    # after:  {"name": "the asset pack equals UVA's, file by file", "passed": True, "detail": "4,531 files"}
    found = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(Path(root).rglob("*")) if path.is_file()}
    # every file UVA has must be here with the same bytes; a file only this server has is named and does not decide
    differing = sorted(name for name in expected if found.get(name) != expected[name])
    extra = len(set(found) - set(expected))
    detail = (f"{len(expected):,} files equal" if not differing else
              f"{len(differing)} file(s) missing or different, the first: {differing[0]}")
    detail += f"; {extra} file(s) only on this server" if extra else ""
    return {"name": "the asset pack equals UVA's, file by file", "passed": not differing, "detail": detail}


def relative_difference(archived, new):
    """The largest relative difference between two parsed JSON values of the same shape; inf when shapes differ."""
    # before: [1.0, [true, 2.0]] against [1.0, [true, 2.002]] ; after: 0.001 (about)
    if isinstance(archived, list) or isinstance(new, list):
        if not (isinstance(archived, list) and isinstance(new, list) and len(archived) == len(new)):
            return math.inf
        return max((relative_difference(left, right) for left, right in zip(archived, new)), default=0.0)
    if isinstance(archived, bool) or isinstance(new, bool) or isinstance(archived, int) and isinstance(new, int):
        return 0.0 if archived == new else math.inf
    if isinstance(archived, float) and isinstance(new, float) and math.isnan(archived) and math.isnan(new):
        return 0.0
    if not all(isinstance(value, (int, float)) for value in (archived, new)):
        return 0.0 if archived == new else math.inf
    return abs(archived - new) / max(abs(archived), abs(new), 1e-30) if archived != new else 0.0


def outside_first_line_rule(archived, new):
    """Count the numbers of two parsed JSON values that differ by more than the first update allows."""
    # before: [-1.5e-05, [25, true]] against [-1.505e-05, [25, true]] ; after: 0 (5e-8 <= 1e-3 * 1.5e-5 + 1e-6)
    # before: [-1.5e-05] against [-2.5e-05] ; after: 1 (1e-5 > 1.015e-6)
    # before: [25] against [26] ; after: 1 (whole numbers must be equal) ; a list of another length counts 1
    if isinstance(archived, list) or isinstance(new, list):
        if not (isinstance(archived, list) and isinstance(new, list) and len(archived) == len(new)):
            return 1
        return sum(outside_first_line_rule(left, right) for left, right in zip(archived, new))
    # true/false, whole numbers and anything that is not a number must be equal
    if (isinstance(archived, bool) or isinstance(new, bool) or isinstance(archived, int) and isinstance(new, int)
            or not all(isinstance(value, (int, float)) for value in (archived, new))):
        return int(archived != new)
    if math.isnan(archived) and math.isnan(new):
        return 0
    return int(not abs(new - archived) <= FIRST_LINE_RELATIVE * abs(archived) + FIRST_LINE_ABSOLUTE)


def compare_with_archive(reference, lines):
    """Compare a short run's metric lines with the archived baseline's first lines; returns a check record."""
    # every field of the archived line except the time of the write; fields only the new line has are left out
    differing, worst_first, outside_first = 0, 0.0, 0
    for index, (archived, new) in enumerate(zip(reference, lines)):
        for field, value in archived.items():
            if field == "unix":
                continue
            difference = relative_difference(value, new[field]) if field in new else math.inf
            differing += difference > 0
            # only the first update decides; a field missing from it counts as one number outside the rule
            if index == 0:
                worst_first = max(worst_first, difference)
                outside_first += outside_first_line_rule(value, new[field]) if field in new else 1
    enough = len(lines) >= len(reference) > 0
    passed = enough and outside_first == 0
    return {"name": "default updates against the archived baseline", "passed": passed,
            "detail": (f"{len(lines)} of {len(reference)} updates written; first update: largest relative difference "
                       f"{worst_first:.2e}, {outside_first} number(s) outside the rule {FIRST_LINE_RULE}; all "
                       f"updates: {differing} differing field(s)" + (", every field equal" if not differing else ""))}


def finite(value):
    """Say whether every number of a parsed JSON value is finite."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def train(python, code, folder, copies, seeds, learner, environment, updates):
    """Run one short training in its own process; returns its exit code, metric lines, speed and peak card memory."""
    command = [str(python), "-u", str(Path(code) / "scripts/atari_pixel_campaign/train_final.py"),
               "--run-dir", str(folder), "--copies", str(copies), "--copy-seeds", ",".join(map(str, seeds)),
               "--target-updates", str(updates), "--observation-warmup-iterations", "50", "--checkpoint-seconds", "1200",
               "--time-limit-seconds", str(TRAINER_SECONDS), "--learner", json.dumps(learner),
               "--environment", json.dumps(environment)]
    print("start: " + " ".join(command), flush=True)
    peak = 0
    with (Path(folder).parent / f"{Path(folder).name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=code)
        # the card's used memory is read every two seconds while the trainer runs; its largest reading is the peak
        while process.poll() is None:
            peak = max(peak, card_memory_used())
            time.sleep(2)
    metrics = Path(folder) / "metrics.jsonl"
    lines = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
    # seconds per update: the median gap between the writes of consecutive lines, the first two (compilation) left out
    gaps = [after["unix"] - before["unix"] for before, after in zip(lines[2:], lines[3:])]
    return {"exit_code": process.returncode, "lines": lines, "peak_card_memory_mb": peak,
            "seconds_per_update": statistics.median(gaps) if gaps else None}


def speed_line(label, copies, run):
    """One row of the throughput table: copies, seconds per iteration, total and per-copy steps per second, hours."""
    # after: "default: copies 4, 2.08 s per iteration, total 0.0315 million steps per second, 7.9 thousand steps
    #         per second per copy, 0.0353 hours per million steps per copy, peak card memory 61,200 MB"
    if not run["seconds_per_update"]:
        return f"{label}: copies {copies}, too few updates to measure a speed"
    per_copy = STEPS_PER_UPDATE / run["seconds_per_update"]
    return (f"{label}: copies {copies}, {run['seconds_per_update']:.2f} s per iteration, total "
            f"{copies * per_copy / 1e6:.4f} million steps per second, {per_copy / 1e3:.1f} thousand steps per second "
            f"per copy, {1e6 / (3600 * per_copy):.4f} hours per million steps per copy, peak card memory "
            f"{run['peak_card_memory_mb']:,} MB")


def main(argv=None):
    """Run the smoke job's steps in order and write the verdict."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True, help="the platform folder of the frozen code copy")
    parser.add_argument("--server", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--profile", required=True, help="the pool's execution profile, as one word (four-copies)")
    parser.add_argument("--updates", type=int, default=20)
    args = parser.parse_args(argv)
    job = os.environ.get("SLURM_JOB_ID", "local")
    smoke = args.run_dir / "smoke"
    work = smoke / f"{args.pool}_{job}"
    work.mkdir(parents=True, exist_ok=True)
    profiles = json.loads((HERE / "card_profiles.json").read_text())["profiles"]
    profile_name = args.profile.replace("-", " ")
    profile = profiles[profile_name]
    checks = []

    # 1. the card and its profile
    name, memory = card()
    below = profile["card_memory_below_mb"]
    fits = profile["minimum_card_memory_mb"] <= memory and (below is None or memory < below)
    checks.append({"name": "the card belongs to the pool's execution profile", "passed": fits,
                   "detail": f"{name}, {memory:,} MB, profile {profile_name!r}"})
    # 2. the asset pack, byte for byte
    checks.append(asset_check(Path(os.environ["XDG_DATA_HOME"]), json.loads((smoke / "asset_sha256.json").read_text())))
    # 3. imports, the card under jax, the start room's geometry
    environment = dict(os.environ, PYTHONPATH=str(args.code / "src"))
    answer = subprocess.run([sys.executable, str(HERE / "smoke_trainer_checks.py")], capture_output=True, text=True,
                            env=environment)
    if answer.returncode == 0:
        checks += json.loads(answer.stdout.strip().splitlines()[-1])["checks"]
    else:
        checks.append({"name": "imports, the card under jax, the start room's geometry", "passed": False,
                       "detail": answer.stderr.strip()[-600:]})
    # 4. default updates, compared with the archived baseline's first lines
    reference = [json.loads(line) for line in (smoke / f"reference_{args.profile}.jsonl").read_text().splitlines()]
    seeds = json.loads((smoke / f"reference_{args.profile}_seeds.json").read_text())
    default = train(sys.executable, args.code, work / "default", profile["copies"], seeds, profile["learner"],
                    profile["environment"], args.updates)
    checks.append(compare_with_archive(reference[:args.updates], default["lines"]))
    checks.append({"name": "default updates: exit code 0 and finite values",
                   "passed": default["exit_code"] == 0 and finite(default["lines"]),
                   "detail": f"exit code {default['exit_code']}, {len(default['lines'])} updates"})
    # 5. the fast last layer with the game events
    fast = train(sys.executable, args.code, work / "fast_layer", profile["copies"], seeds,
                 {**profile["learner"], **FAST_LAYER}, {**profile["environment"], "report_game_events": True},
                 args.updates)
    first_times = work / "fast_layer/game_first_times.json"
    folded = json.loads(first_times.read_text()) if first_times.exists() else {}
    checks.append({"name": "fast last layer with game events: completed, records folded, finite values",
                   "passed": (fast["exit_code"] == 0 and len(fast["lines"]) == args.updates and finite(fast["lines"])
                              and folded.get("updates_folded") == args.updates and folded.get("has_event_fields") is True
                              and (work / "fast_layer/game_windows.jsonl").exists()),
                   "detail": f"exit code {fast['exit_code']}, {len(fast['lines'])} updates, record files folded to "
                             f"update {folded.get('updates_folded')}"})
    # 6. speed and peak card memory
    lines = [speed_line("default", profile["copies"], default),
             speed_line("fast last layer (step size 0.1, sharing group 16) with game events", profile["copies"], fast)]
    result = {"passed": all(check["passed"] for check in checks), "server": args.server, "pool": args.pool,
              "execution_profile": profile_name, "job_id": job, "host": os.uname().nodename, "device": name,
              "card_memory_mb": memory, "written_at": datetime.now().astimezone().isoformat(), "checks": checks,
              "speed_lines": lines,
              "speed": {"copies": profile["copies"], "default_seconds_per_update": default["seconds_per_update"],
                        "fast_layer_seconds_per_update": fast["seconds_per_update"],
                        "default_peak_card_memory_mb": default["peak_card_memory_mb"],
                        "fast_layer_peak_card_memory_mb": fast["peak_card_memory_mb"]}}
    path = smoke / f"result_{args.pool}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for check in checks:
        print(f"{'ok ' if check['passed'] else 'NOT'} {check['name']}: {check['detail']}", flush=True)
    for line in lines:
        print(line, flush=True)
    print(f"{'PASSED' if result['passed'] else 'NOT PASSED'}; the verdict is in {path}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
