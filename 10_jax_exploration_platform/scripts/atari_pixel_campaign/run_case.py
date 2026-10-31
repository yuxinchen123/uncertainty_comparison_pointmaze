"""Run one immutable queue case and keep every attempt's complete result or failure."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import (attempt_paths, atomic_json, content_digest, process_matches,
                        process_start_time, read_json, ready_specification,
                        scientific_case_complete, utc_now, verify_source)


def wait_for_recorded_release(run_dir, case, attempt, token):
    """Keep a child away from the GPU until its exact PID and birth counter are recorded."""
    path = Path(run_dir) / "queue" / "leases" / f"{case['unit_id']}.json"
    started = time.monotonic()
    while time.monotonic() - started < 60:
        lease = read_json(path)
        if lease["token"] != token or lease["attempt"] != attempt:
            raise RuntimeError("this child no longer owns the recorded case attempt")
        if (lease.get("runner_pid") == os.getpid()
                and lease.get("runner_start_time") == process_start_time(os.getpid())
                and lease["state"] == "running"):
            return lease
        if not process_matches(lease["worker_pid"], lease["worker_start_time"]):
            raise RuntimeError("the worker ended before releasing this child for GPU execution")
        time.sleep(0.1)
    raise RuntimeError("the worker did not record and release this child within sixty seconds")


def hardware_record(lease):
    """Record the actual selected physical GPU and this exact runner's identity."""
    result = subprocess.run(["nvidia-smi", "-i", lease["gpu_uuid"],
                             "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=10)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != lease["gpu_uuid"]:
        raise ValueError("the runner's CUDA visibility differs from its physical GPU lease")
    return {"hostname": socket.gethostname(), "gpu_query": result.stdout.strip(),
            "gpu_uuid": lease["gpu_uuid"], "pid": os.getpid(),
            "process_start_time": process_start_time(os.getpid()),
            "worker_pid": lease["worker_pid"], "worker_start_time": lease["worker_start_time"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


def run_case(run_dir, case, attempt_number, lease_token):
    """Validate source and authorization before importing the numerical learner or game."""
    run_dir = Path(run_dir).resolve()
    if str(run_dir) != case["campaign_run_dir"]:
        raise ValueError("case outputs must remain in the original absolute campaign directory")
    unit, attempt = attempt_paths(run_dir, case["unit_id"], attempt_number)
    attempt.mkdir(parents=True, exist_ok=True)
    result_path = attempt / "result.json"
    digest = case["specification_digest"]
    supplied = {name: value for name, value in case.items() if name != "specification_digest"}
    if content_digest(supplied) != digest:
        raise ValueError("the immutable case specification was modified")
    if result_path.exists():
        saved = read_json(result_path)
        if saved.get("specification_digest") == digest and scientific_case_complete(saved):
            return saved
        raise ValueError("an invalid attempt already has a result; retain it and create a new attempt")
    atomic_json(attempt / "case.json", case)
    lease = wait_for_recorded_release(run_dir, case, attempt_number, lease_token)
    verify_source(case["source_root"], case["source_commit"])
    ready = ready_specification(run_dir)
    if ready is None or (ready["source_root"], ready["source_commit"]) != (case["source_root"], case["source_commit"]):
        raise ValueError("the parent source/gate authorization is no longer valid for this case")
    expected_script = (Path(case["source_root"]) /
                       "10_jax_exploration_platform/scripts/atari_pixel_campaign/run_case.py").resolve()
    if Path(__file__).resolve() != expected_script:
        raise ValueError("the case runner itself was not loaded from its immutable source checkout")
    hardware = {**hardware_record(lease), "source_root": case["source_root"],
                "source_commit": case["source_commit"], "hardware_class": case["hardware_class"]}
    atomic_json(attempt / "hardware.json", hardware)
    atomic_json(unit / "hardware.json", hardware)

    def progress(value):
        """Expose real stage/round changes in the monitoring location and the attempt record."""
        record = {**value, "unit_id": case["unit_id"], "attempt": attempt_number,
                  "pid": os.getpid(), "time": utc_now(), "source_commit": case["source_commit"]}
        atomic_json(attempt / "progress.json", record)
        atomic_json(unit / "progress.json", record)

    progress({"state": "validating_source_and_configuration"})
    # Import all numerical code from the pinned checkout only after authorization/release.
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(Path(case["source_root"]) / "10_jax_exploration_platform/src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from paired_benchmark import CorrectnessMismatch, run_paired_benchmark

    values = dict(case["configuration"])
    for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients"):
        if name in values:
            values[name] = tuple(values[name])
    config = PixelPPOConfig(**values)
    atomic_json(attempt / "config_resolved.json", asdict(config))
    atomic_json(unit / "config_resolved.json", asdict(config))
    candidate = case["candidate"]
    try:
        result = run_paired_benchmark(
            config, candidate["learner"], candidate["environment"], candidate["execution"],
            rounds=case["rounds"], iterations=case["iterations"], seed_start=case["seed_start"],
            reference_execution=case["reference_execution"], progress=progress)
    except CorrectnessMismatch as error:
        result = {**error.context, "decision": "rejected", "measurement_valid": False,
                  "correctness_passed": False, "correctness_attempted": True,
                  "failure_stage": "correctness", "error": str(error), "copies": config.n_copies}
    if result["hardware_class"] != case["hardware_class"]:
        raise ValueError("the measured GPU class differs from the queued context")
    if result["source_commit"] != case["source_commit"]:
        raise ValueError("the measured source commit differs from the immutable case")
    result.update(case_completed=True, idea=case["hypothesis"], unit_id=case["unit_id"],
                  specification_digest=digest, attempt=attempt_number,
                  additional_correctness_checks=case["additional_correctness_checks"])
    atomic_json(result_path, result)
    atomic_json(unit / "result.json", result)
    progress({"state": "completed", "decision": result["decision"], "case_completed": True})
    return result


def main():
    """Preserve a visible failure record and nonzero exit code if an attempt cannot complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--lease-token", required=True)
    args = parser.parse_args()
    case = read_json(args.case)
    if str(args.run_dir.resolve()) != case["campaign_run_dir"]:
        raise ValueError("case outputs must remain in the original absolute campaign directory")
    unit, attempt = attempt_paths(args.run_dir, case["unit_id"], args.attempt)
    try:
        result = run_case(args.run_dir, case, args.attempt, args.lease_token)
    except Exception as error:
        # This boundary records infrastructure failures and re-raises them. It never changes
        # a failed case into a successful result or silently switches source/configuration.
        failure = {"case_completed": False, "failure_stage": "runner_exception", "retryable": True,
                   "exception_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "attempt": args.attempt,
                   "pid": os.getpid(), "time": utc_now(), "source_commit": case["source_commit"]}
        atomic_json(attempt / "failure.json", failure)
        atomic_json(unit / "failure.json", failure)
        atomic_json(unit / "progress.json", {**failure, "state": "failed"})
        raise
    print(json.dumps({"unit_id": case["unit_id"], "decision": result["decision"],
                      "case_completed": True}), flush=True)


if __name__ == "__main__":
    main()
