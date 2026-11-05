"""Run one separately approved fifth attempt through the unchanged original D28 case runner."""

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import pregather_async_retry as retry


ENVIRONMENT_KEYS = set(retry.infra.RUNTIME) | set(retry.ALLOCATOR) | set(retry.infra.THREAD_ENVIRONMENT) | {
    "CUDA_VISIBLE_DEVICES", "JAXTARI_SKIP_SPRITE_UPDATE", "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
    "SLURM_CPUS_PER_TASK", "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND",
    "TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION"}


def initial_environment():
    """Retain only approved process-start controls; never publish raw environment bytes or credentials."""
    values = {}
    for item in Path("/proc/self/environ").read_bytes().split(b"\0"):
        name, separator, value = item.partition(b"=")
        if separator:
            key = name.decode(errors="strict")
            if key in ENVIRONMENT_KEYS:
                values[key] = value.decode(errors="strict")
    return values


def execute(approval_path, assignment_path, token):
    """Check ownership and installed bytes before importing JAX, then call the original complete case."""
    approval, case = retry.validate_approval(approval_path)
    expected = Path(approval["helpers"]["source_root"]) / retry.infra.PREFIX / "run_pregather_async_retry.py"
    if (Path(__file__).resolve() != expected or Path(retry.__file__).resolve() != expected.with_name("pregather_async_retry.py")
            or str(Path(sys.executable).absolute()) != approval["python"]):
        raise ValueError("execute the exact approved fifth-attempt helper and Python")
    retry.reject_unapproved_controls(os.environ)
    original = retry.adjacent("run_infrastructure_retry")
    if Path(getattr(original.retry, "__file__", "")).resolve() != expected.with_name("infrastructure_retry.py"):
        raise ValueError("the unchanged runner imported another infrastructure retry helper")
    runner = original.load_original_runner(case)
    run = Path(case["campaign_run_dir"])
    unit, attempt = runner.attempt_paths(run, case["unit_id"], 5)
    if any((attempt / name).exists() for name in ("result.json", "failure.json", "pregather_async_retry_execution.json")):
        raise ValueError("the separately approved fifth attempt is already consumed")
    lease = runner.wait_for_recorded_release(run, case, 5, token)
    approval_record = retry.infra.file_record(approval_path, retry.LocalEvidence())
    if lease.get("pregather_async_retry_approval") != approval_record:
        raise ValueError("the released fifth-attempt lease names another approval")
    started = time.time()
    assignment = json.loads(Path(assignment_path).read_bytes())
    hardware = original.verify_placement(assignment, lease, runner)
    owned = retry.owned_allocation(run, lease["gpu_uuid"])
    initial = initial_environment()
    if any(initial.get(name) not in (None, "") or os.environ.get(name) not in (None, "")
            for name in ("TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION")):
        raise ValueError("the fifth attempt inherited a conflicting allocator alias")
    if any(initial.get(name) != value for name, value in {**retry.infra.RUNTIME, **retry.ALLOCATOR,
            **retry.infra.THREAD_ENVIRONMENT, "CUDA_VISIBLE_DEVICES": lease["gpu_uuid"]}.items()):
        raise ValueError("the fifth attempt did not inherit its exact explicit process-start controls")
    diagnostic = retry.diagnostic_proof(approval["diagnostic"], case, retry.LocalEvidence())
    training = original.load_training_helper(approval)
    training.enforce_runtime(retry.infra.RUNTIME, os.environ)
    runtime = retry.runtime_environment(os.environ)
    retry.validate_runtime(runtime)
    source = diagnostic["parser"]["source"]
    retry.image(retry.LocalEvidence(), source["path"], source["sha256"])
    assets = training.verify_assets(approval["asset_record"], os.environ)
    public = training.jaxtari_python_receipt(approval["jaxtari_source_root"])
    libraries = training.installed_gpu_library_receipts()
    installed = {"jaxtari_source": public, "gpu_libraries": libraries}
    if installed != diagnostic["installed"]:
        raise ValueError("installed fifth-attempt backend bytes differ from the completed diagnostic")
    runner.atomic_json(attempt / "owned_allocation.json", owned)
    runner.atomic_json(attempt / "process_environment.json", {"source": "/proc/self/environ allowlist", "values": initial,
        "pid": os.getpid(), "process_start_time": runner.process_start_time(os.getpid()), "gpu_uuid": lease["gpu_uuid"]})
    runner.atomic_json(attempt / "asset_receipt.json", assets)
    runner.atomic_json(attempt / "installed_backend.json", installed)
    try:
        # The installed parser is imported only now; no learner/game initialization has happened.
        import jax
        xla_client = importlib.import_module("jaxlib.xla_client")
        if Path(xla_client.__file__).resolve() != Path(source["path"]):
            raise ValueError("the fifth attempt imported another installed allocator parser")
        parser = {"source": source, "parsed_options": xla_client.generate_pjrt_gpu_plugin_options(),
            "environment": {key: os.environ[key] for key in retry.ALLOCATOR}, "memory_preparation": None,
            "pid": os.getpid(), "process_start_time": runner.process_start_time(os.getpid()), "gpu_uuid": lease["gpu_uuid"]}
        retry.validate_parser(parser, retry.LocalEvidence())
        if parser["parsed_options"] != diagnostic["parser"]["parsed_options"]:
            raise ValueError("the actual fifth-attempt parser options differ from the completed diagnostic")
        runner.atomic_json(attempt / "actual_allocator.json", parser)
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu" or "A4500" not in devices[0].device_kind:
            raise ValueError("the fifth attempt must see only its proved physical RTX A4500")
        # No diagnostic callbacks, large-array preparation, cache clearing or numerical monkey-patching occur here.
        result = runner.run_case(run, case, 5, token)
        original.verify_cpu_inheritance(assignment, lease)
        modules = original.verify_original_modules(case)
        distribution = metadata.distribution("jaxtari")
        if Path(sys.modules["jaxtari"].__file__).resolve() != Path(distribution.locate_file("jaxtari/__init__.py")).resolve():
            raise ValueError("the fifth-attempt original runner imported a shadowed JAXtari package")
        loaded = training.actual_gpu_library_receipt(devices[0].client, libraries)
        retry.validate_libraries(installed, loaded, retry.LocalEvidence())
        sidecar = {"schema_version": 1, "completed": True, "attempt": 5, "unit_id": case["unit_id"],
            "approval": approval_record, "lease_token": token, "helpers": approval["helpers"],
            "numerical_source": approval["numerical_source"], "runner": str(expected), "memory_preparation": None,
            "original_result": retry.infra.file_record(attempt / "result.json", retry.LocalEvidence()),
            "assignment": retry.infra.file_record(assignment_path, retry.LocalEvidence()),
            "owned_allocation": retry.infra.file_record(attempt / "owned_allocation.json", retry.LocalEvidence()),
            "actual_allocator": retry.infra.file_record(attempt / "actual_allocator.json", retry.LocalEvidence()),
            "process_environment_record": retry.infra.file_record(attempt / "process_environment.json", retry.LocalEvidence()),
            "process_environment": initial, "runtime_environment": runtime, "pid": os.getpid(),
            "process_start_time": runner.process_start_time(os.getpid()), "gpu_uuid": lease["gpu_uuid"],
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "thread_environment": dict(retry.infra.THREAD_ENVIRONMENT),
            "frozen_modules": modules, "loaded_gpu_libraries": loaded,
            "started_at_seconds": started, "finished_at_seconds": time.time()}
        retry.infra.write_once(attempt / "pregather_async_retry_execution.json", retry.infra.json_bytes(sidecar))
        retry.validate_execution(attempt / "result.json", case, lease)
        return result
    except BaseException as error:
        failure = {"case_completed": False, "failure_stage": "pregather_async_retry_runner", "retryable": False,
            "attempt": 5, "unit_id": case["unit_id"], "source_commit": retry.SOURCE_COMMIT,
            "approval": approval_record, "error": str(error), "exception_type": type(error).__name__,
            "traceback": traceback.format_exc(), "pid": os.getpid(), "time": runner.utc_now()}
        retry.infra.write_once(attempt / "failure.json", retry.infra.json_bytes(failure))
        runner.atomic_json(unit / "failure.json", failure)
        runner.atomic_json(unit / "progress.json", {**failure, "state": "failed"})
        raise


def main():
    """Run one already claimed fifth attempt; never authorize, requeue or submit work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--lease-token", required=True)
    arguments = parser.parse_args()
    result = execute(arguments.approval, arguments.assignment, arguments.lease_token)
    print(json.dumps({"unit_id": result["unit_id"], "attempt": 5, "decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
