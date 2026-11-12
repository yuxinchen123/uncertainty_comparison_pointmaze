"""Total steps per second of two training processes that share one card, against one process.

Idea 27 of the PointMaze speed study. Measurements on the same card, one after another, each
made by `compare_study_arms.py` with the study's learner:

  1. one process holding 2 x P policies;
  2. with --also-one-process-alone, one process holding P policies;
  3. two processes holding P policies each, started together; their timed rounds begin at the
     same moment, so each is timed while the other one trains.

The record has the shape `compare_study_arms.py` writes, so the same tables read it: the
reference arm is the one process of 2 x P policies, and the two processes together are an arm
whose total steps per second is the sum of the two processes' and whose ratio is that total over
the reference's. The ratio is of two medians, not paired by round.

Usage (inside a Slurm job on one card):
  python two_processes_on_one_card.py --envs-per-copy 4 --copies-per-process 256 --output <file>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study_learner import machine_record, throughput_columns  # noqa: E402


def launch(folder: Path, tag: str, envs_per_copy: int, copies: int, start_at: float,
           rounds: int, budget: float):
    """Start one measuring process; returns (process, the path of its record)."""
    output = folder / f"{tag}.json"
    command = [sys.executable, str(HERE / "compare_study_arms.py"), "--idea", "27_two-processes",
               "--shapes", f"{envs_per_copy}:{copies}", "--arm", "the study's learner",
               "--output", str(output), "--max-rounds", str(rounds),
               "--budget-seconds", str(budget), "--start-rounds-at", str(start_at)]
    log = open(folder / f"{tag}.log", "w")
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), output


def arm_of(path: Path) -> tuple:
    """(the one arm of a finished measuring process's record, its number of rounds)."""
    shape = json.loads(path.read_text())["shapes"][0]
    return next(iter(shape["arms"].values())), shape["rounds"]


def main():
    """Run the three measurements and write one record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs-per-copy", type=int, default=4)
    ap.add_argument("--copies-per-process", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--also-one-process-alone", action="store_true",
                    help="also time one process of P policies alone (about 3 more minutes)")
    ap.add_argument("--lead-seconds", type=float, default=240.0,
                    help="how long after their start the two processes begin their rounds")
    args = ap.parse_args()
    started = time.time()
    N, P = args.envs_per_copy, args.copies_per_process
    folder = args.output.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)

    # one process at a time: 2 x P policies, then P policies
    singles = {}
    for copies in ((2 * P, P) if args.also_one_process_alone else (2 * P,)):
        process, path = launch(folder, f"one-process_policies-{copies}", N, copies, 0.0,
                               args.rounds, 240.0)
        if process.wait() != 0:
            raise RuntimeError(f"the process of {copies} policies failed; see {path.with_suffix('.log')}")
        singles[copies] = arm_of(path)
        print(f"[{time.time() - started:6.1f} s] one process, {copies} policies: "
              f"{singles[copies][0]['throughput']['total_steps_per_second_millions']:.3f} "
              "million steps per second", flush=True)

    # two processes together; both begin their rounds at the same moment
    start_at = time.time() + args.lead_seconds
    pair = [launch(folder, f"two-processes_process-{k}_policies-{P}", N, P, start_at,
                   args.rounds, args.lead_seconds + 120.0) for k in (1, 2)]
    for process, path in pair:
        if process.wait() != 0:
            raise RuntimeError(f"a process of the pair failed; see {path.with_suffix('.log')}")
    together = [arm_of(path) for _, path in pair]
    total = sum(arm["throughput"]["total_steps_per_second_millions"] for arm, _ in together)
    # the rollout length, read back from a record: steps per policy per iteration / environments
    one = singles[2 * P][0]["throughput"]
    T = round(one["steps_per_second_per_copy_thousands"] * 1e3 * one["seconds_per_iteration"] / N)

    def as_arm(throughput: dict, ratio: float, rounds: int) -> dict:
        """One arm in the comparison record's form."""
        return {"throughput": throughput, "paired_ratio_median": ratio,
                "paired_ratio_least": ratio, "paired_ratio_most": ratio,
                "rounds_faster_than_reference": rounds if ratio > 1.0 else 0}

    # the pair as one setting of 2 x P policies: the time one iteration of all of them takes
    reference = singles[2 * P][0]["throughput"]["total_steps_per_second_millions"]
    pair_seconds = 2 * P * N * T / (total * 1e6)
    rounds = min(count for _, count in together)
    record = {
        "record": "two processes on one card against one process", "idea": "27_two-processes-per-card",
        "environment": "pointmaze-large", "machine": machine_record(),
        "reference": f"one process of {2 * P} policies",
        "shapes": [{
            "copies": 2 * P, "envs_per_copy": N, "rounds": rounds, "conclusive": rounds >= 5,
            "not_measured": [], "ratio_is_of_two_medians_not_paired": True,
            "arms": {
                f"one process of {2 * P} policies": as_arm(singles[2 * P][0]["throughput"], 1.0,
                                                           rounds),
                f"two processes of {P} policies each, at the same time": as_arm(
                    throughput_columns(2 * P, N, T, pair_seconds), total / reference, rounds)},
            "each_process_of_the_pair": [arm["throughput"] for arm, _ in together]}],
        "wall_seconds": time.time() - started}
    if args.also_one_process_alone:
        alone = singles[P][0]["throughput"]
        record["shapes"][0]["arms"][f"one process of {P} policies alone on the card"] = as_arm(
            alone, alone["total_steps_per_second_millions"] / reference, rounds)
    args.output.write_text(json.dumps(record, indent=1))
    print(f"[{time.time() - started:6.1f} s] two processes of {P} policies: {total:.3f} million "
          f"steps per second together, {total / reference:.3f} of one process of {2 * P}",
          flush=True)


if __name__ == "__main__":
    main()
