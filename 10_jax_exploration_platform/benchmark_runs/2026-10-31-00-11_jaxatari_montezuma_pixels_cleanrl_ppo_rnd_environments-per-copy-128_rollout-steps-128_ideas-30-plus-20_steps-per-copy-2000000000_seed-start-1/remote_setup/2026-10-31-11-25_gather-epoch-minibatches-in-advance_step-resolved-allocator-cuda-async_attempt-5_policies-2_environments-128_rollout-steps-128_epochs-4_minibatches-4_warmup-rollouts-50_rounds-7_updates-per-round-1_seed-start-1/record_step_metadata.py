"""Record only named Slurm/GPU allocation fields before the unchanged approved worker."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket

ALLOWED = ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_CPUS_PER_TASK",
           "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND", "CUDA_VISIBLE_DEVICES")


def snapshot(phase, selected_gpu_uuid=None, environment=None):
    """Keep allocation names only; never serialize the inherited environment or its raw bytes."""
    environment = os.environ if environment is None else environment
    stat = Path("/proc/self/stat").read_text()
    return {"phase": phase, "time_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "process_start_time": stat.rsplit(")", 1)[1].split()[19],
        "hostname": socket.gethostname(), "cgroup": Path("/proc/self/cgroup").read_text(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "selected_gpu_uuid": selected_gpu_uuid,
        "environment": {name: environment[name] for name in ALLOWED if name in environment}}


def main():
    """Preserve a small exclusive metadata record and reject changed preflight lineage before execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "resolved", "after"), required=True)
    parser.add_argument("--selected-gpu-uuid")
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parent
    plan = json.loads((directory / "launch_plan.json").read_bytes())
    expected = plan["approval"]
    if hashlib.sha256(Path(expected["path"]).read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("the original unclaimed fifth approval changed")
    previous = json.loads(Path(plan["preflight_failure"]["path"]).read_bytes())
    if hashlib.sha256(Path(plan["preflight_failure"]["path"]).read_bytes()).hexdigest() != plan["preflight_failure"]["sha256"]:
        raise ValueError("the preserved preflight failure changed")
    for named in ("prior_lease", "prior_case"):
        receipt = previous[named]
        if hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("the held original case or fourth lease changed")
    if any(Path(path).exists() for path in previous["verified_absent_paths"]):
        raise ValueError("the existing fifth attempt was consumed before this replacement submission")
    if Path(plan["worker_output_dir"]).exists():
        raise ValueError("this replacement worker output already exists")
    observation = snapshot(arguments.phase, arguments.selected_gpu_uuid)
    observation["approval"] = expected
    observation["preflight_failure"] = plan["preflight_failure"]
    destination = directory / ("step_metadata_" + arguments.phase + ".json")
    with destination.open("x") as stream:
        json.dump(observation, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if arguments.phase == "after" and observation["environment"].get("CUDA_VISIBLE_DEVICES") != arguments.selected_gpu_uuid:
        raise ValueError("step visibility differs from the proved selected physical UUID")
    print("recorded step allocation " + arguments.phase, flush=True)


if __name__ == "__main__":
    main()
