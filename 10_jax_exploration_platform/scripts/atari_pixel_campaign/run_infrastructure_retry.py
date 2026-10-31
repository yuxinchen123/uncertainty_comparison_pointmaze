"""Prepare allocator history, then call the untouched original D28 case runner.

This fresh process deliberately imports no current controller or ideas module.
The original case runner owns the numerical checks, timing, and raw result.
"""

import argparse
import csv
import gc
import importlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import infrastructure_retry as retry


SCRIPTS = Path(__file__).resolve().parent


def load_training_helper(approval):
    """Import only the committed metadata/library helper by its exact original file path."""
    path = Path(approval["helpers"]["source_root"]) / retry.PREFIX / "train_optimized.py"
    specification = importlib.util.spec_from_file_location("infrastructure_retry_training", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_original_runner(case):
    """Reject module shadowing before importing the original D28 controller, ideas and runner."""
    forbidden = ("controller", "ideas", "run_case", "paired_benchmark", "exploration_platform", "jaxtari", "jax")
    if any(name == item or name.startswith(item + ".") for name in sys.modules for item in forbidden):
        raise ValueError("the numerical retry must start before current or numerical modules are imported")
    root = Path(case["source_root"])
    sys.path.insert(0, str(root / retry.PREFIX))
    sys.path.insert(0, str(root / "10_jax_exploration_platform/src"))
    runner = importlib.import_module("run_case")
    verify_original_modules(case, names=("run_case", "controller", "ideas"))
    return runner


def verify_original_modules(case, names=("run_case", "controller", "ideas", "paired_benchmark")):
    """Check exact imported paths before accepting any original benchmark output."""
    expected = {name: str(Path(case["source_root"]) / retry.PREFIX / (name + ".py")) for name in names}
    for name, path in expected.items():
        if name not in sys.modules or Path(sys.modules[name].__file__).resolve() != Path(path).resolve():
            raise ValueError("the retry imported another version of the original module: " + name)
    package = sys.modules.get("exploration_platform")
    if package is not None and Path(package.__file__).resolve() != Path(case["source_root"]) / "10_jax_exploration_platform/src/exploration_platform/__init__.py":
        raise ValueError("the retry imported a different numerical learner package")
    for name, module in list(sys.modules.items()):
        if name.startswith("exploration_platform.atari_pixels") and getattr(module, "__file__", None):
            root = Path(case["source_root"]) / "10_jax_exploration_platform/src/exploration_platform/atari_pixels"
            if root not in Path(module.__file__).resolve().parents:
                raise ValueError("the retry imported a shadowed learner submodule: " + name)
    return expected


def verify_placement(assignment, lease, runner):
    """Prove the selected UUID while retaining every original Slurm job/step allocation field."""
    retry.validate_assignment(assignment, lease["gpu_uuid"])
    proof = assignment["allocation"]
    actual = {name: os.environ[name] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS") if os.environ.get(name)}
    if (socket.gethostname().split(".")[0] != "jaguar03" or os.environ.get("SLURM_JOB_ID") != proof["slurm_job_id"]
            or actual != proof["slurm_allocations"] or os.environ.get("CUDA_VISIBLE_DEVICES") != lease["gpu_uuid"]
            or not runner.process_matches(assignment["parent_pid"], assignment["parent_process_start_time"])):
        raise ValueError("retry child differs from its recorded full allocation, UUID or living parent")
    verify_cpu_inheritance(assignment, lease)
    gpu = runner.hardware_record(lease)
    if "A4500" not in gpu["gpu_query"]:
        raise ValueError("the retry must execute on its assigned physical RTX A4500")
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True, timeout=15)
    if any(len(row) == 2 and row[1].strip() == lease["gpu_uuid"] for row in csv.reader(rows.splitlines())):
        raise ValueError("the assigned retry GPU already has a numerical process")
    return gpu


def verify_cpu_inheritance(assignment, lease):
    """Require the original runner to inherit the worker's exact eight CPUs and original thread counts."""
    expected = retry.validate_cpu_allocation(assignment, lease["gpu_uuid"])
    proof = assignment["cpu_allocation"]
    if (sorted(os.sched_getaffinity(0)) != expected or lease.get("cpu_affinity") != expected
            or lease.get("thread_environment") != retry.THREAD_ENVIRONMENT
            or os.environ.get("SLURM_CPUS_PER_TASK") != proof["slurm_cpus_per_task"]
            or os.environ.get("SLURM_CPU_BIND_TYPE") != proof["slurm_cpu_bind_type"]
            or any(os.environ.get(name) != value for name, value in retry.THREAD_ENVIRONMENT.items())):
        raise ValueError("the original numerical runner did not inherit its assigned eight CPUs and thread runtime")


def memory_snapshot(device, gpu_uuid):
    """Read actual selected-card memory and allocator counters without assuming cached memory is returned."""
    query = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid,
        "--query-gpu=uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = list(csv.reader(query.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0].strip() != gpu_uuid:
        raise ValueError("the retry memory sample does not identify its selected GPU")
    _, total, used, free = [item.strip() for item in rows[0]]
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    sampled = [{"pid": int(row[0]), "bytes": int(row[2]) * 2 ** 20} for row in csv.reader(processes.splitlines())
               if len(row) == 3 and row[1].strip() == gpu_uuid]
    return {"time_seconds": time.time(), "device_memory": {"timestamp_seconds": time.time(), "gpu_uuid": gpu_uuid,
        "total_bytes": int(total) * 2 ** 20, "used_bytes": int(used) * 2 ** 20, "free_bytes": int(free) * 2 ** 20,
        "processes": sampled}, "allocator": {name: int(value) for name, value in device.memory_stats().items()}}


def prepare_region(runtime, arrays, snapshot):
    """Allocate, synchronize and delete the approved exact-sized array before importing learner/game code."""
    before = snapshot()
    region = arrays.zeros(tuple(retry.PREPARATION["shape"]), dtype=arrays.uint8)
    runtime.block_until_ready(region)
    nbytes, dtype = int(region.nbytes), str(region.dtype)
    if nbytes != retry.REGION_BYTES or dtype != "uint8":
        region.delete()
        raise ValueError("the retry array differs from its approved nine-GiB uint8 allocation")
    allocated = snapshot()
    region.delete()
    del region
    collected = gc.collect()
    released = snapshot()
    result = {"allocation_count": 1, "array_nbytes": nbytes, "array_dtype": dtype,
        "array_shape": retry.PREPARATION["shape"], "before": before, "allocated": allocated, "released": released,
        "unreachable_python_objects_collected": collected, "allocator_pool_release_assumed": False}
    retry.validate_region(result, before["device_memory"]["gpu_uuid"])
    return result


def execute(approval_path, assignment_path, token):
    """Validate authorization and placement, prepare memory, then preserve the original runner's raw output."""
    approval, case = retry.validate_approval(approval_path)
    expected_script = Path(approval["helpers"]["source_root"]) / retry.PREFIX / "run_infrastructure_retry.py"
    if Path(__file__).resolve() != expected_script or str(Path(sys.executable).absolute()) != approval["python"]:
        raise ValueError("execute the exact committed helper and Python recorded by the retry approval")
    if Path(retry.__file__).resolve() != expected_script.with_name("infrastructure_retry.py"):
        raise ValueError("the numerical retry imported a shadowed approval helper")
    runner = load_original_runner(case)
    run = Path(case["campaign_run_dir"])
    unit, attempt = runner.attempt_paths(run, case["unit_id"], 4)
    if any((attempt / name).exists() for name in ("result.json", "failure.json", "region_preparation.json", "infrastructure_retry_execution.json")):
        raise ValueError("this one-shot retry already has execution artifacts")
    lease = runner.wait_for_recorded_release(run, case, 4, token)
    approval_record = retry.file_record(approval_path, retry.LocalEvidence())
    if lease.get("infrastructure_retry_approval") != approval_record:
        raise ValueError("the released lease does not authorize this exact retry receipt")
    assignment = json.loads(Path(assignment_path).read_bytes())
    hardware = verify_placement(assignment, lease, runner)
    training = load_training_helper(approval)
    actual = training.enforce_runtime(retry.RUNTIME, os.environ)
    actual.update({name: os.environ[name] for name in retry.THREAD_ENVIRONMENT})
    if any(os.environ.get(name) != value for name, value in {**retry.ALLOCATOR, "JAXTARI_SKIP_SPRITE_UPDATE": "1"}.items()):
        raise ValueError("the retry must retain the original allocator and game runtime")
    assets = training.verify_assets(approval["asset_record"], os.environ)
    jaxtari_source = training.jaxtari_python_receipt(approval["jaxtari_source_root"])
    libraries = training.installed_gpu_library_receipts()
    retry.write_once(attempt / "asset_receipt.json", retry.json_bytes(assets))
    retry.write_once(attempt / "installed_backend.json", retry.json_bytes({"jaxtari_source": jaxtari_source,
        "gpu_libraries": libraries, "packages": {name: metadata.version(name) for name in training.PACKAGE_NAMES}}))
    started = time.time()
    try:
        # The only numerical intervention precedes all learner/game imports; no JAX cache is cleared.
        import jax
        import jax.numpy as arrays
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or "A4500" not in devices[0].device_kind:
            raise ValueError("the numerical retry must see exactly its assigned RTX A4500")
        region = prepare_region(jax, arrays, lambda: memory_snapshot(devices[0], lease["gpu_uuid"]))
        retry.write_once(attempt / "region_preparation.json", retry.json_bytes(region))
        runner.atomic_json(unit / "progress.json", {"state": "allocator_region_prepared", "attempt": 4,
            "unit_id": case["unit_id"], "pid": os.getpid(), "time": runner.utc_now(), "actual_array_nbytes": retry.REGION_BYTES})
        result = runner.run_case(run, case, 4, token)
        verify_cpu_inheritance(assignment, lease)
        modules = verify_original_modules(case)
        distribution = metadata.distribution("jaxtari")
        if Path(sys.modules["jaxtari"].__file__).resolve() != Path(distribution.locate_file("jaxtari/__init__.py")).resolve():
            raise ValueError("the original runner imported a shadowed JAXtari package")
        # Before: original D28 result bytes. After: a separate receipt binds them without modifying them.
        sidecar = {"schema_version": 1, "completed": True, "unit_id": case["unit_id"], "attempt": 4,
            "approval": approval_record, "lease_token": token, "helpers": approval["helpers"],
            "numerical_source": approval["numerical_source"], "runner": str(expected_script),
            "original_result": retry.file_record(attempt / "result.json", retry.LocalEvidence()),
            "region_preparation": retry.file_record(attempt / "region_preparation.json", retry.LocalEvidence()),
            "assignment": retry.file_record(assignment_path, retry.LocalEvidence()),
            "pid": os.getpid(), "process_start_time": runner.process_start_time(os.getpid()),
            "gpu_uuid": lease["gpu_uuid"], "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "slurm_allocations": assignment["allocation"]["slurm_allocations"], "runtime_environment": actual,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {name: os.environ[name] for name in retry.THREAD_ENVIRONMENT},
            "frozen_modules": modules, "started_at_seconds": started, "finished_at_seconds": time.time(),
            "loaded_gpu_libraries": training.actual_gpu_library_receipt(devices[0].client, libraries)}
        retry.write_once(attempt / "infrastructure_retry_execution.json", retry.json_bytes(sidecar))
        retry.validate_execution(attempt / "result.json", case, lease)
        return result
    except BaseException as error:
        # A raw numerical result without the completed sidecar remains an infrastructure failure.
        failure = {"case_completed": False, "failure_stage": "infrastructure_retry_runner", "retryable": False,
            "attempt": 4, "unit_id": case["unit_id"], "source_commit": retry.SOURCE_COMMIT,
            "approval": approval_record, "error": str(error), "exception_type": type(error).__name__,
            "traceback": traceback.format_exc(), "pid": os.getpid(), "time": runner.utc_now()}
        retry.write_once(attempt / "failure.json", retry.json_bytes(failure))
        runner.atomic_json(unit / "failure.json", failure)
        runner.atomic_json(unit / "progress.json", {**failure, "state": "failed"})
        raise


def main():
    """Run one released receipt; this child never claims cases or submits jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--lease-token", required=True)
    arguments = parser.parse_args()
    result = execute(arguments.approval, arguments.assignment, arguments.lease_token)
    print(json.dumps({"unit_id": result["unit_id"], "attempt": 4, "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
