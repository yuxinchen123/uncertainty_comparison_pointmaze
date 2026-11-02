"""Prepare an original-reference scaling component check inside its owned allocation.

These one- and eight-A4500 checks validate the committed producer and resource
shape. They contribute no hypotheses and no final optimized-scaling evidence.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_gpu_scaling as scaling

ORIGINAL_SOURCE = "d28e6735e79788f618acb0ebfde8609db1f776af"


def read(path):
    """Read explicit launch metadata without importing a numerical library."""
    return json.loads(Path(path).read_text())


def original_configuration(source, record):
    """Require the completed case's literal original full configuration, with no optimized switches."""
    defaults = scaling.diagnostic.default_configuration(source)
    tree = ast.parse((Path(source) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/paired_benchmark.py").read_text())
    execution = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "DEFAULT_EXECUTION" for target in node.targets))
    expected = json.loads(json.dumps({"learner": defaults["learner"], "environment": defaults["environment"], "execution": execution}))
    if (record.get("source_commit") != ORIGINAL_SOURCE or record.get("hardware_class") != "jaguar03_rtx_a4500"
            or record.get("copies") != 1 or record.get("case_completed") is not True
            or record.get("measurement_valid") is not True or record.get("correctness_passed") is not True
            or record.get("reference") != expected):
        raise ValueError("component checks require an actually completed original one-policy reference")
    return expected


def prepare_component(input_path):
    """Prove ownership/physical allocation and finish all immutable records before numerical startup."""
    launch = read(input_path)
    run, source = Path(launch["run_dir"]).resolve(strict=True), Path(launch["source_root"]).resolve(strict=True)
    job = os.environ.get("SLURM_JOB_ID", "")
    if (launch.get("schema_version") != 1 or launch["source_commit"] != ORIGINAL_SOURCE
            or type(launch["gpu_count"]) is not int or launch["gpu_count"] not in {1, 8} or not job.isdecimal()
            or job not in (run / "slurm/submitted_jobids_uva.txt").read_text().splitlines()
            or socket.gethostname().split(".")[0] != "jaguar03"
            or not re.search(r"(?:^|/)job_" + re.escape(job) + r"(?:/|$)", Path("/proc/self/cgroup").read_text(), re.MULTILINE)):
        raise ValueError("original scaling component checks require their exact owned jaguar03 Slurm job")
    for receipt in launch["inputs"].values():
        if hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("a committed component input changed before execution")
    configuration = original_configuration(source, read(launch["inputs"]["completed_reference"]["path"]))
    if configuration != read(launch["configuration"]):
        raise ValueError("the component configuration differs from its actual original reference")
    if "jax" in sys.modules or os.environ.get("OMP_NUM_THREADS") != "8":
        raise ValueError("metadata preparation requires no numerical runtime and eight OpenMP threads per child")
    roster = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    roster = [value.strip() for value in roster]
    proof = scaling.allocation_proof({"gpu_count": launch["gpu_count"], "gpu_uuids": roster}, os.environ, socket.gethostname())
    arguments = argparse.Namespace(source_root=source, source_commit=ORIGINAL_SOURCE,
        run_dir=run, unit_id=launch["unit_id"], configuration=Path(launch["configuration"]),
        runtime_environment=Path(launch["runtime_environment"]), asset_record=Path(launch["asset_record"]),
        jaxtari_source_root=None, gpu_uuid=roster, seed_start=1, minimum_updates_per_round=10,
        barrier_timeout_seconds=1800., python=Path(launch["python"]))
    directory = scaling.prepare(arguments)
    scope = {"schema_version": 1, "kind": "original_reference_scaling_component_check",
        "formal_scaling_credit": False, "hypothesis_count_contribution": 0, "optimization_selected": False,
        "source_commit": ORIGINAL_SOURCE, "gpu_count": launch["gpu_count"], "policies_per_gpu": 1,
        "environments_per_policy": 128, "rollout_steps": 128, "epochs": 4, "minibatches": 4,
        "normalization_warmup_iterations": 50, "rounds": 7, "updates_per_round": 10,
        "omp_threads_per_child": 8, "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "parent_cpu_affinity": sorted(os.sched_getaffinity(0)), "slurm_job_id": job,
        "physical_allocation": proof, "launch_inputs": str(Path(input_path).resolve()),
        "launch_inputs_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest()}
    scaling.selected.write_once(directory / "component_scope.json", scaling.selected.json_bytes(scope))
    return directory, launch["python"]


def main():
    """Run only the immutable staged producer after its resource/component scope has been recorded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-inputs", type=Path, required=True)
    args = parser.parse_args()
    directory, python = prepare_component(args.launch_inputs)
    os.execv(python, [python, str(directory / "code/measure_gpu_scaling.py"), "--prepared-unit", str(directory)])


if __name__ == "__main__":
    main()
