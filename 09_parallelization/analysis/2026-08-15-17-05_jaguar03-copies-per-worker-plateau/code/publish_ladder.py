"""Collect this run's side measurements into one file the report's generator reads.

The ladder rungs are already written one result file each, in the schema the report's own loader
reads, so they need no help. What this gathers is everything that is NOT a throughput setting
anyone would run — one worker alone on an otherwise empty node, the memory system's own rate, and
what the memory system had left while the training load ran — under a file name the throughput
loader does not match, so they cannot appear as extra points in the throughput tables.

Usage:
  python publish_ladder.py
"""
import json
import platform
import subprocess
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
BASE = RUN.parent.parent
RESULTS = BASE / "benchmarks" / "results"
DATA = RUN / "data"


def load(name):
    """One of this run's data files, or an empty list when the job did not get that far."""
    path = DATA / name
    return json.loads(path.read_text()) if path.exists() else []


def collect():
    """Every side measurement this run produced, in one dictionary."""
    return {
        "host": platform.node(),
        "measurement": ("side measurements of the copies-per-worker sweep on jaguar03: one worker "
                        "alone, the memory system's rate, and the memory system under load"),
        "steps_per_copy_per_iteration": 512,
        "run_folder": str(RUN),
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "alone_full_batch": load("ladder_full_batch_p1_alone_j3.json"),
        "alone_epoch_minibatch": load("ladder_epoch_minibatch_p1_alone_j3.json"),
        "memory_probe_full_batch": load("memory_probe_full_batch.json"),
        "memory_probe_epoch_minibatch": load("memory_probe_epoch_minibatch.json"),
        "memory_saturation": load("memory_saturation.json"),
        "memory_corun": load("memory_corun.json"),
        "matmul_floor": load("matmul_floor_cpu.json"),
    }


def main():
    """Write the collected side measurements to the results directory."""
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_cpu_copies_per_worker_plateau.json")
    out.write_text(json.dumps(collect(), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
