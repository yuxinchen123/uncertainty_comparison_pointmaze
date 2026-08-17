"""Write one submission's `assignment.json` and `hardware.json` into its job folder.

One sweep's jobs land on different cards, so the hardware of a run is not one fact but one per
submission. This is called by `run_unit.sh` at the start of every job, before any training.
"""
import argparse
import datetime
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path


def hardware() -> dict:
    """The card, the driver and the scheduler placement of the machine this job is on."""
    # a machine with no graphics card has no nvidia-smi at all, so ask before calling it
    query = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    smi = (subprocess.run(query, capture_output=True, text=True)
           if shutil.which("nvidia-smi") else None)
    return {
        "record": "job",
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "nvidia_smi": (smi.stdout.strip() if smi and smi.returncode == 0
                       else "no nvidia-smi on this host"),
    }


def main() -> None:
    """Write both records for the job folder named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-file", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--mode", required=True, choices=["real", "canary"])
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    unit = json.loads(Path(args.unit_file).read_text())
    assignment = {"record": "assignment", "mode": args.mode, "unit_id": unit["unit_id"],
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
                  "host": socket.gethostname(),
                  "started_at": datetime.datetime.now().astimezone().isoformat(),
                  "unit": unit}
    (job_dir / "assignment.json").write_text(json.dumps(assignment, indent=2) + "\n")
    (job_dir / "hardware.json").write_text(json.dumps(hardware(), indent=2) + "\n")


if __name__ == "__main__":
    main()
