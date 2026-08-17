"""Roll a run's per-unit shards up into the run-level `metrics.jsonl` and `summary.json`.

A run is one sweep, and its units may have been run by different jobs on different machines, so the
run-level files are never written by a unit — they are computed here from every `data/*.jsonl`
shard and every `slurm/jobs/*/hardware.json`, and they are what reports and document generators
read. Re-running this is safe: both output files are rewritten from the shards each time.

This is the generic layer: it collects the records, keeps the latest attempt of each, and reports
throughput and the run's headline numbers. A run that needs per-configuration science scores adds
them in its own `monitoring/aggregate.py`, which calls this first.

Run:
  PYTHONNOUSERSITE=1 <python> aggregate_run.py <run folder>
"""
import argparse
import json
from pathlib import Path


def read_shards(run_dir: Path) -> list:
    """Every record of every shard, in shard-name order, each tagged with its shard file.

    before: data/unit-1.jsonl holding job / unit_start / episode_window / unit_complete lines;
    after:  one flat list of those dictionaries, each with a "shard" field naming its file.
    """
    records = []
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["shard"] = shard.name
            records.append(record)
    return records


def read_job_hardware(run_dir: Path) -> list:
    """The hardware record of every job folder, when the run was submitted through slurm."""
    return [json.loads(path.read_text())
            for path in sorted((run_dir / "slurm" / "jobs").glob("*/hardware.json"))]


def latest_windows(records: list) -> list:
    """The episode-window records, one per (unit, window), keeping the latest attempt's line.

    before: two attempts of unit-1 both holding the window ending at iteration 200;
    after:  one record, the one written later in the shard.
    """
    latest = {}
    for record in records:
        if record.get("record") == "episode_window":
            latest[(record["unit_id"], record["last_iteration"])] = record
    return [latest[key] for key in sorted(latest)]


def aggregate(run_dir: Path) -> dict:
    """Write metrics.jsonl and summary.json from the shards; return the summary."""
    run_dir = Path(run_dir)
    records = read_shards(run_dir)
    windows = latest_windows(records)
    completions = [r for r in records if r.get("record") == "unit_complete"]
    starts = {r["unit_id"]: r for r in records if r.get("record") == "unit_start"}
    hardware = ([r for r in records if r.get("record") == "job"]
                + read_job_hardware(run_dir))

    with open(run_dir / "metrics.jsonl", "w") as out:
        for record in windows:
            out.write(json.dumps(record) + "\n")

    # throughput, always reported both ways: the whole machine's rate and the rate one copy gets,
    # plus the per-copy rate written as hours per million steps, which is the unit a reader plans in.
    # Two rates are given because the first iteration also compiles the program: the steady rate is
    # what a long run gets, the wall-clock rate is what this run actually took from start to end.
    # before: a unit record with 200 iterations, 20.6 s total of which 19.1 s was the first
    #         iteration; after: steady 0.0075 s per iteration, so 8.7e6 steps per second, against
    #         6.4e5 steps per second measured over the whole wall clock.
    copies = sum(r["copies"] for r in completions)
    seconds = max((r["seconds_total"] for r in completions), default=0.0)
    env_steps = sum(r["env_steps"] for r in completions)
    steady_iteration_seconds = (
        sum(r["seconds_per_iteration_steady"] for r in completions) / len(completions)
        if completions else None)
    steps_per_iteration = ((env_steps / sum(r["iterations"] for r in completions))
                           if completions else 0.0)
    steady_total_rate = ((steps_per_iteration / steady_iteration_seconds)
                         if steady_iteration_seconds else 0.0)
    steady_per_copy_rate = (steady_total_rate / copies) if copies else 0.0
    wall_clock_total_rate = (env_steps / seconds) if seconds else 0.0
    throughput = {
        "copies": copies,
        "seconds_total_wall_clock": seconds,
        "seconds_per_iteration_steady": steady_iteration_seconds,
        "total_env_steps": env_steps,
        "total_env_steps_per_second_steady": steady_total_rate,
        "env_steps_per_second_per_copy_steady": steady_per_copy_rate,
        "hours_per_million_steps_per_copy_steady": (
            1e6 / (3600 * steady_per_copy_rate) if steady_per_copy_rate else None),
        "total_env_steps_per_second_wall_clock": wall_clock_total_rate,
        "env_steps_per_second_per_copy_wall_clock": (
            (wall_clock_total_rate / copies) if copies else 0.0),
        "seconds_first_iteration_with_compile": [
            r["seconds_first_iteration_with_compile"] for r in completions],
    }

    # the end-of-run numbers a reader asks for first. The reward a window carries is a SUM over the
    # window, so it is divided by the window's episode count to become a mean episode return.
    # before: a last window of 200 iterations, 256 episodes per copy, per-copy sums around 512;
    # after:  a mean episode return around 2.0 per copy.
    per_unit_last = {}
    for record in windows:
        per_unit_last[record["unit_id"]] = record
    returns, coverage = [], []
    for record in per_unit_last.values():
        episodes = record["episodes_per_copy_in_window"]
        returns += [value / episodes for value in record["reward_ext_sum_per_copy"]]
        coverage += record.get("coverage_per_copy", [])
    summary = {
        "run_id": run_dir.name,
        "units_complete": len(completions),
        "units_seen": len(starts) or len({r["unit_id"] for r in records if "unit_id" in r}),
        "units": sorted(starts),
        "records": len(windows),
        "throughput": throughput,
        "last_window_episode_return_per_copy": {
            "mean": (sum(returns) / len(returns)) if returns else None,
            "min": min(returns) if returns else None,
            "max": max(returns) if returns else None,
        },
        "final_maze_coverage_per_copy": {
            "mean": (sum(coverage) / len(coverage)) if coverage else None,
            "min": min(coverage) if coverage else None,
            "max": max(coverage) if coverage else None,
        },
        "hardware": hardware,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Aggregate the run folder given on the command line and print the headline numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    summary = aggregate(Path(args.run_dir))
    print(json.dumps(summary["throughput"], indent=2))


if __name__ == "__main__":
    main()
