"""Roll a run's per-unit shards up into the run-level `metrics.jsonl` and `summary.json`.

A run is one sweep, and its units may have been run by different jobs on different machines, so the
run-level files are never written by a unit — they are computed here from every `data/*.jsonl`
shard and every `slurm/jobs/*/hardware.json`, and they are what reports and document generators
read. Re-running this is safe: both output files are rewritten from the shards each time.

Run:
  PYTHONNOUSERSITE=1 <python> aggregate_run.py <run folder>
"""
import argparse
import json
from pathlib import Path


def read_shards(run_dir: Path) -> list:
    """Every record of every shard, in shard-name order, each tagged with its shard file.

    before: data/unit_0000.jsonl holding job / iteration / unit_complete lines;
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


def aggregate(run_dir: Path) -> dict:
    """Write metrics.jsonl and summary.json from the shards; return the summary."""
    run_dir = Path(run_dir)
    records = read_shards(run_dir)
    iterations = [r for r in records if r.get("record") == "iteration"]
    completions = [r for r in records if r.get("record") == "unit_complete"]
    hardware = ([r for r in records if r.get("record") == "job"]
                + read_job_hardware(run_dir))

    # the run's metric stream: every unit's iteration records, ordered by unit and then iteration,
    # with the last record kept when a unit was attempted more than once at the same iteration
    # before: two attempts of unit_0000 both holding iteration 10; after: one line, the later one
    latest = {}
    for record in iterations:
        latest[(record["unit_id"], record["iteration"])] = record
    ordered = [latest[key] for key in sorted(latest)]
    with open(run_dir / "metrics.jsonl", "w") as out:
        for record in ordered:
            out.write(json.dumps(record) + "\n")

    # throughput, reported both ways per the throughput rule: the whole machine's rate and the rate
    # one copy gets, plus the per-copy rate written as hours per million steps
    copies = sum(r["copies"] for r in completions)
    seconds = max((r["seconds_total"] for r in completions), default=0.0)
    env_steps = sum(r["env_steps"] for r in completions)
    per_copy_rate = (env_steps / seconds / copies) if seconds and copies else 0.0
    throughput = {
        "copies": copies,
        "seconds_total": seconds,
        "seconds_per_iteration_steady": (
            sum(r["seconds_per_iteration_steady"] for r in completions) / len(completions)
            if completions else None),
        "total_env_steps": env_steps,
        "total_env_steps_per_second": (env_steps / seconds) if seconds else 0.0,
        "env_steps_per_second_per_copy": per_copy_rate,
        "hours_per_million_steps_per_copy": (1e6 / (3600 * per_copy_rate)) if per_copy_rate else None,
    }

    # the end-of-run numbers a reader asks for first: how far learning got, and how much was seen
    final = [r for r in ordered
             if r["iteration"] == max((x["iteration"] for x in ordered if
                                       x["unit_id"] == r["unit_id"]), default=-1)]
    rewards = [value for r in final for value in r["reward_ext_sum_per_copy"]]
    coverage = [value for r in final for value in r.get("coverage_per_copy", [])]
    summary = {
        "run_id": run_dir.name,
        "units_complete": len(completions),
        "units_seen": len({r["unit_id"] for r in records if "unit_id" in r}),
        "records": len(ordered),
        "throughput": throughput,
        "final_extrinsic_reward_per_copy": {
            "mean": (sum(rewards) / len(rewards)) if rewards else None,
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
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
