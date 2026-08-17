"""Turn this run's own rate probes into the table the submission planner prefers over the survey.

The shared throughput survey measures the PPO-plus-distillation trainer at 512 copies and above.
This run may want to cut a 1,024-copy unit into four chunks of 256, and nothing in the survey prices
that. `code/probe_rates.sh` runs each arm for 200 real iterations at each copy count in question on
the cards the plan might use; this module reads those shards and writes
`code/measured_cells.json`, which `plan_submission.py` uses IN PREFERENCE to the survey wherever a
cell exists — a direct measurement of this arm on this card at this copy count beats the survey's
figure for a different trainer plus a transfer factor.

Run after the probe jobs finish:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python code/measured_cells.py
"""
import json
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent
CLASS_CATALOG = Path("/p/rlprojects/.claude/skills/uva-submit-gpu-sweep/school_compute_resource"
                     "/server_introduction.json")
ROLLOUT_STEPS = 128
ENVS_PER_COPY = 4


def node_to_class() -> dict:
    """Which node class each machine belongs to, from the cluster's own catalog.

    before: the catalog's class list, serval06-09 holding serval06..serval09;
    after:  {"serval06": "serval06-09", "serval07": "serval06-09", ...}
    """
    catalog = json.load(open(CLASS_CATALOG))["classes"]
    return {node: entry["name"] for entry in catalog for node in entry["nodes"]}


def read_probe_shards(probe_dir: Path) -> list:
    """One cell per finished probe run: which arm, which card, how many copies, how fast.

    A probe shard holds a job record, a unit_start, one episode window and a unit_complete. Only a
    shard with a unit_complete is read: an interrupted probe has no steady rate to report.
    """
    classes = node_to_class()
    cells = []
    for shard in sorted((probe_dir / "data").glob("*.jsonl")):
        records = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        start = next((r for r in records if r.get("record") == "unit_start"), None)
        done = next((r for r in records if r.get("record") == "unit_complete"), None)
        job = next((r for r in records if r.get("record") == "job"), None)
        if start is None or done is None or job is None:
            continue
        copies = start["copies"]
        seconds = done["seconds_per_iteration_steady"]
        host = job["host"]
        if host not in classes:
            raise SystemExit(f"probe ran on {host}, which is in no class of {CLASS_CATALOG}")
        cells.append({
            "arm": start["bonus"],
            "node_class": classes[host],
            "copies": copies,
            "seconds_per_iteration": seconds,
            "total_steps_per_second": copies * ROLLOUT_STEPS * ENVS_PER_COPY / seconds,
            "steps_per_second_per_copy": ROLLOUT_STEPS * ENVS_PER_COPY / seconds,
            "hours_per_million_steps_per_copy":
                1e6 / (3600 * ROLLOUT_STEPS * ENVS_PER_COPY / seconds),
            # what a job pays before its first timed iteration on this card at this copy count,
            # measured here rather than transferred: the build and prime, and the first iteration
            # which also compiles the program
            "build_and_prime_seconds": done["seconds_build_and_prime"],
            "first_iteration_with_compile_seconds":
                done["seconds_first_iteration_with_compile"],
            "source": {"host": host, "slurm_job_id": job.get("slurm_job_id", ""),
                       "unit_id": start["unit_id"], "device_kind": job.get("device_kind", "")},
        })
    return cells


def main() -> None:
    """Write code/measured_cells.json and print the table it holds."""
    cells = read_probe_shards(RUN_DIR / "probe")
    if not cells:
        raise SystemExit(f"no finished probe run under {RUN_DIR / 'probe' / 'data'}")
    (RUN_DIR / "code" / "measured_cells.json").write_text(
        json.dumps({"cells": cells}, indent=2) + "\n")
    print(f"{'arm':30} {'class':14} {'copies':>7} {'s/iter':>9} {'M steps/s':>10} "
          f"{'steps/s/copy':>13} {'h/Mstep/copy':>13} {'build+prime':>12} {'first iter':>11}")
    for cell in sorted(cells, key=lambda c: (c["arm"], c["node_class"], c["copies"])):
        print(f"{cell['arm']:30} {cell['node_class']:14} {cell['copies']:>7} "
              f"{cell['seconds_per_iteration']:>9.5f} "
              f"{cell['total_steps_per_second'] / 1e6:>10.2f} "
              f"{cell['steps_per_second_per_copy']:>13.0f} "
              f"{cell['hours_per_million_steps_per_copy']:>13.4f} "
              f"{cell['build_and_prime_seconds']:>12.1f} "
              f"{cell['first_iteration_with_compile_seconds']:>11.1f}")
    print(f"\nwrote {RUN_DIR / 'code' / 'measured_cells.json'} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
