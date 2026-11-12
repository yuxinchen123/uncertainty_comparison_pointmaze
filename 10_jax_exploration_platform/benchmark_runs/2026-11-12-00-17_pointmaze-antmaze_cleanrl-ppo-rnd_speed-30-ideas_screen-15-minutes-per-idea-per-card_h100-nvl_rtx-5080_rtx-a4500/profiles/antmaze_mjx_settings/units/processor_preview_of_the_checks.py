"""The physics checks of one model on processors, with fewer environments, while the card is in use.

It runs the same three rollouts as the card units (700, 700 and 1,200 steps) and writes one record
to processor_previews/<model in words>.json beside this folder; the report shows these records in
a section of their own. The model, the number of environments and the map come from the
environment of the submitting shell:
  ANTMAZE_PREVIEW_MODEL="max_contact_points=16,ls_iterations=4" (or "default")
  ANTMAZE_PREVIEW_ENVIRONMENTS=256   ANTMAZE_PREVIEW_MAP=umaze
Runs inside a Slurm job through the test runner, which calls a function named `main`:
  ANTMAZE_PREVIEW_MODEL=default bash <run folder>/infrastructure/submit_tests.sh \
      -o "python_functions=test_* main" -s <this file>
"""
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent
PLATFORM = FOLDER.parents[3]


def main():
    """Run the checks of the model named in the environment and write its preview record."""
    sys.path.insert(0, str(PLATFORM / "benchmarks" / "env"))
    import profile_antmaze_mjx_settings as settings
    import exploration_platform  # noqa: F401  (64-bit statistics on, as in training)
    changes = settings.parse_changes(os.environ["ANTMAZE_PREVIEW_MODEL"])
    n_envs = int(os.environ.get("ANTMAZE_PREVIEW_ENVIRONMENTS", "256"))
    map_name = os.environ.get("ANTMAZE_PREVIEW_MAP", "umaze")
    model = settings.model_name(changes)
    started = time.time()
    cell = {"kind": "checks", "name": settings.PREVIEW + settings.checks_name(model, map_name), "status": "measured",
            "measured_pacific": settings.split.pacific(started), "other_processes_on_card": []}
    machine = f"{socket.gethostname()} processors (8 cores, job {os.environ.get('SLURM_JOB_ID', 'unknown')})"
    run = {"unit": f"processor preview: {model}", "machine": machine, "host": socket.gethostname(),
           "card": {"name": "processors (preview)", "driver": "none", "uuid": ""},
           "started_pacific": settings.split.pacific(started), "unit_started_pacific": settings.split.pacific(started),
           "unit_start_epoch": started, "finished": False,
           "versions": {name: settings.importlib.metadata.version(name)
                        for name in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "numpy")},
           "python": sys.version.split()[0], "cells": [cell]}
    # GOAL: the record is written after every rollout, so a job that reaches its time limit keeps what it finished.
    slug = re.sub(r"[^a-z0-9]+", "-", f"{model}{settings.map_words(map_name)}".lower()).strip("-")
    out = FOLDER / "processor_previews" / f"{slug}.json"
    save = lambda: out.write_text(json.dumps(run, indent=1))
    settings.measure_checks(cell, changes, n_envs, save, map_name=map_name)
    run["finished"], run["end_epoch"] = True, time.time()
    save()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
