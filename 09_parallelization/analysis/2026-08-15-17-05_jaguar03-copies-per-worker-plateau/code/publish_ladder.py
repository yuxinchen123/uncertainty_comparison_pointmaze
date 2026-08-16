"""Collect this run's measurements into one file the report's generator reads.

The ladder rungs at 32 copies per worker and above are already written one file each, in the
schema the report's existing loader reads. This script gathers everything else — the rungs below
32, the two contention separations, the memory-system measurements — into a single file under a
name that loader does not match, so the report can quote them without them appearing as extra
points in the tables and figures that are about throughput settings.

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
    """Every measurement this run produced, in one dictionary."""
    return {
        "host": platform.node(),
        "measurement": "copies per worker ladder, jaguar03, independent worker processes",
        "workers": 224,
        "steps_per_copy_per_iteration": 512,
        "timing": ("each point times about 60 seconds of continuous work, so the rate is the "
                   "settled one rather than the opening seconds of the load"),
        "run_folder": str(RUN),
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "ladder_full_batch": load("ladder_full_batch_p224_plateau_j3_styleA.json"),
        "ladder_epoch_minibatch": load("ladder_epoch_minibatch_p224_plateau_j3.json"),
        "alone_full_batch": load("ladder_full_batch_p1_alone_j3_styleA.json"),
        "one_core_full_batch": load("ladder_full_batch_p112_onecore_j3_styleA.json"),
        "memory_probe_full_batch": load("memory_probe_full_batch.json"),
        "memory_probe_epoch_minibatch": load("memory_probe_epoch_minibatch.json"),
        "memory_saturation": load("memory_saturation.json"),
        "memory_corun": load("memory_corun.json"),
    }


def main():
    """Write the collected measurements to the results directory, one file per publication."""
    out = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_cpu_copies_per_worker_plateau.json")
    out.write_text(json.dumps(collect(), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
