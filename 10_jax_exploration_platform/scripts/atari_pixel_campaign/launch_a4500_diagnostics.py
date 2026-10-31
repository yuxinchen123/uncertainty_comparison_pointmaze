"""Prepare and optionally submit two separate A4500 determinism diagnostics on one card."""

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import traceback


CORE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
PIXEL_PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
NSYS = "/sw/ubuntu2204/ebu082025/software/common/core/nsight-systems/2024.7.1/bin/nsys"
ACCOUNT = "sl5nw_base"
GPU_CONTEXT = "jaguar03_rtx_a4500"
GPU_PLACEHOLDER = "SLURM_ALLOCATED_GPU_UUID"
HELPER_FILES = ("diagnose_gpu_execution.py", "launch_diagnostic.py",
                "launch_queue.py", "launch_a4500_diagnostics.py")


def load_helper(path):
    """Load only standard-library helper definitions before any numerical child starts."""
    specification = importlib.util.spec_from_file_location("a4500_diagnostic_metadata", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def committed_helpers():
    """Require exact committed helper bytes before preparing any launch artifact."""
    scripts = Path(__file__).resolve().parent
    root = Path(subprocess.check_output(["git", "-C", str(scripts), "rev-parse", "--show-toplevel"],
                                        text=True).strip())
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    identities, contents = {}, {}
    for name in HELPER_FILES:
        path = scripts / name
        relative = str(path.relative_to(root))
        expected = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative])
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError("commit the diagnostic helper before preparation: " + relative)
        identities[name] = {"path": str(path), "git_commit": commit,
                            "sha256": hashlib.sha256(actual).hexdigest()}
        contents[name] = actual
    return identities, contents


def nsight_identity(path):
    """Read the existing Nsight executable and version without creating a GPU context."""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("an existing executable Nsight Systems installation is required")
    version = subprocess.check_output([str(path), "--version"], text=True).strip()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "version": version}


def profile_command(plan, directory):
    """Capture only the separately marked seventh update, retaining the full fixed warmup."""
    directory = Path(directory)
    return [plan["nsight"]["path"], "profile", "--trace=cuda,cudnn", "--sample=none",
            "--cpuctxsw=none", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop", "--stop-on-exit=true", "--force-overwrite=false",
            "--output=" + str(directory / "trace/capture"), plan["python"],
            str(Path(plan["code_directory"]) / "diagnose_gpu_execution.py"),
            "--unit-dir", str(directory), "--trace-enabled"]


def disabled_cache_environment():
    """Disable JAX file caches explicitly and reject unrecorded XLA options or cache paths."""
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    if flags != ["--xla_gpu_exclude_nondeterministic_ops=true"]:
        raise ValueError("the diagnostic requires only the original enabled determinism XLA flag; "
                         "unexpected options or autotune-load/cache flags are not allowed")
    names = ("JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR",
             "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES")
    before = {name: os.environ.get(name) for name in names}
    if before["JAX_COMPILATION_CACHE_DIR"] or before["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"]:
        raise ValueError("the diagnostic cannot inherit a JAX compilation or XLA cache directory/policy")
    # Both new processes receive the same explicit cache policy; no XLA option is removed.
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = ""
    return {"policy": "disabled", "before": before,
            "runtime": {name: os.environ.get(name) for name in names}}


def resolved_plan(template, gpu_uuid, job_id):
    """Bind the one Slurm allocation before writing a numerical subprocess's final plan."""
    if template["gpu_uuid"] != GPU_PLACEHOLDER or template["hardware_context"] != GPU_CONTEXT:
        raise ValueError("the prepared diagnostic template has another GPU binding")
    if not gpu_uuid.startswith("GPU-") or not str(job_id).isdecimal():
        raise ValueError("the allocated GPU UUID and exact Slurm job ID are required")
    return {**template, "gpu_uuid": gpu_uuid, "slurm_job_id": str(job_id),
            "gpu_binding": "resolved from this job's one allocated physical GPU before numerical execution"}


def prepare(args):
    """Write immutable standard records and templates before optionally submitting the job."""
    if getpass.getuser() != "sl5nw" or args.source_commit != CORE_COMMIT:
        raise ValueError("this diagnostic is for sl5nw and the unchanged frozen D28 reference only")
    if not args.python.is_absolute() or not os.access(args.python, os.X_OK):
        raise ValueError("the diagnostic Python must be an existing absolute executable")
    source_root, run_dir = args.source_root.resolve(strict=True), args.run_dir.resolve(strict=True)
    identities, contents = committed_helpers()
    common = load_helper(Path(__file__).with_name("launch_diagnostic.py"))
    diagnostic = load_helper(Path(__file__).with_name("diagnose_gpu_execution.py"))
    core = diagnostic.core_identity(source_root, args.source_commit)
    nsight = nsight_identity(args.nsys)
    suffix = ("copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50"
              "_timed-rounds-2_updates-per-round-2_traced-updates-1_seed-1_source-" + CORE_COMMIT[:12]
              + "_helper-" + identities["diagnose_gpu_execution.py"]["git_commit"][:12])
    batch_id = "jaguar03_reference_determinism_diagnostics_" + suffix
    directory = run_dir / "slurm/diagnostic_jobs" / batch_id
    code_directory = directory / "code"
    publications = {code_directory / name: data for name, data in contents.items()}
    units = []
    for expected, label in ((True, "enabled"), (False, "disabled")):
        identifier = "jaguar03_diagnostic_determinism-" + label + "_" + suffix
        if len(identifier.encode()) > 255:
            raise ValueError("the complete diagnostic unit name exceeds the filesystem component limit")
        unit = run_dir / "data" / identifier
        configuration = diagnostic.default_configuration(source_root, expected)
        plan = {"schema_version": 1, "diagnostic_only": True, "formal_benchmark": False,
                "capacity_evidence": False, "unit_id": identifier, "core": core,
                "helper": identities["diagnose_gpu_execution.py"],
                "launcher": identities["launch_a4500_diagnostics.py"], "support_helpers": identities,
                "configuration": configuration, "python": str(args.python), "nsight": nsight,
                "code_directory": str(code_directory), "hardware_context": GPU_CONTEXT,
                "gpu_uuid": GPU_PLACEHOLDER, "expected_deterministic_operations": expected,
                "gpu_binding": "this Slurm job's single allocated card; resolve before the numerical child",
                "compilation_cache_policy": "disabled", "maximum_active_seconds": 1440}
        publications[unit / "diagnostic_plan_template.json"] = common.encoded_json(plan)
        publications[unit / "manifest.yaml"] = common.encoded_json({
            "schema_version": 1, "run_id": identifier, "diagnostic_only": True,
            "formal_benchmark": False, "capacity_evidence": False,
            "git": {"commit": CORE_COMMIT, "dirty": False}, "helper": plan["helper"],
            "expected_deterministic_operations": expected, "configuration": configuration,
            "allocated_gpu_binding": GPU_PLACEHOLDER, "compilation_cache_policy": "disabled"})
        publications[unit / "config_resolved.yaml"] = common.encoded_json(configuration)
        publications[unit / "command.txt"] = (shlex.join(profile_command(plan, unit)) + "\n").encode()
        publications[unit / "experiment_background.md"] = (
            "## Purpose\n\nCompare the A4500's full-size one-policy execution under the recorded "
            "determinism setting with an explicit control. These measurements are diagnostic only "
            "and are excluded from formal comparisons, capacity evidence, and research-idea counts.\n\n"
            "## Key hyperparameters\n\nOne policy, 128 environments, 128 rollout steps, four epochs, "
            "four minibatches, 50 normalization warmup rollouts, seed one. Two compilation updates, "
            "two timing rounds of two updates, then one separately traced update. Neural math uses "
            "float32 with highest matrix precision; statistics use float64. Deterministic operations "
            + ("are enabled." if expected else "are disabled only for this explicit diagnostic control.")
            + " JAX persistent caches are disabled, and XLA autotune-load/cache options are rejected.\n\n"
            "## Code and config changes\n\nThe frozen D28 learner and environment are unchanged. "
            "Two numerical subprocesses run sequentially on the same Slurm-allocated A4500; "
            "the actual UUID is bound into each immutable resolved plan before it starts. "
            "Nsight captures one later update and produces CUDA kernel and API summaries on the node. "
            "No checkpoint is saved.\n\n## Git state\n\nFrozen core commit: `" + CORE_COMMIT
            + "`. Diagnostic helper commit: `" + plan["helper"]["git_commit"]
            + "`. Original helper bytes and core SHA-256 hashes are recorded in the plan.\n").encode()
        units.append({"unit_id": identifier, "directory": str(unit),
                      "expected_deterministic_operations": expected})
    batch = {"schema_version": 1, "batch_id": batch_id, "run_dir": str(run_dir), "core": core,
             "helpers": identities, "units": units, "nsight": nsight, "account": ACCOUNT,
             "owner": "sl5nw", "hardware_context": GPU_CONTEXT, "compilation_cache_policy": "disabled"}
    publications[directory / "batch_plan.json"] = common.encoded_json(batch)
    worker = source_root / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"
    command = ["/usr/bin/python3", str(code_directory / "launch_a4500_diagnostics.py"),
               "--run-allocated", "--batch-dir", str(directory)]
    lines = ["#!/bin/bash", "set -euo pipefail", "export PYTHONDONTWRITEBYTECODE=1",
             "source " + shlex.quote(str(worker)),
             "run_directory=" + shlex.quote(str(run_dir)),
             'job_directory="$run_directory/slurm/jobs/${SLURM_JOB_ID}_${HOSTNAME}"',
             'mkdir -p "$job_directory"',
             "cp " + shlex.quote(str(directory / "job.sbatch")) + ' "$job_directory/job.sbatch"',
             "cp " + shlex.quote(str(directory / "batch_plan.json")) + ' "$job_directory/assignment.json"',
             'exec > >(tee "$job_directory/job.log") 2>&1', "exec " + shlex.join(command)]
    publications[directory / "job.sbatch"] = ("\n".join(lines) + "\n").encode()
    # Validate every collision before publishing records; prepared artifacts cannot be revised.
    for path, data in publications.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("an immutable A4500 diagnostic artifact differs: " + str(path))
    for path, data in publications.items():
        common.write_once(path, data)
    (run_dir / "slurm/jobs").mkdir(parents=True, exist_ok=True)
    for unit in units:
        (Path(unit["directory"]) / "trace").mkdir(exist_ok=True)
    result = {"prepared": True, "submitted": False, "batch_dir": str(directory), "units": units}
    if args.launch:
        result.update(submit(directory, batch))
    return result


def submit(directory, batch):
    """Submit once and save the exact returned job ID before any later action."""
    queue = load_helper(Path(directory) / "code/launch_queue.py")
    common = load_helper(Path(directory) / "code/launch_diagnostic.py")
    run_dir = Path(batch["run_dir"])
    resources = ["--account=" + ACCOUNT, "--partition=gpu", "--nodelist=jaguar03",
                 "--gres=gpu:nvidia_rtx_a4500:1", "--cpus-per-task=8", "--mem=24G",
                 "--time=01:00:00", "--nodes=1", "--ntasks=1"]
    reservation = queue.current_reservation()
    if reservation is not None:
        resources += ["--reservation=" + reservation, "--qos=csresnolim"]
    command = ["sbatch", "--parsable", "--job-name=a4500_determinism_diagnostics",
               "--output=" + str(run_dir / "slurm/jobs/a4500-diagnostics_%j.log"),
               *resources, str(Path(directory) / "job.sbatch")]
    with (Path(directory) / "submission.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (Path(directory) / "submission_attempt.json").exists():
            raise ValueError("this A4500 diagnostic already has a submission attempt; inspect its exact ledger")
        common.write_once(Path(directory) / "submission_attempt.json", common.encoded_json({
            "command": command, "time": datetime.now(timezone.utc).isoformat()}))
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        identifier = result.stdout.strip().split(";", 1)[0]
        if not identifier.isdecimal():
            raise RuntimeError("sbatch did not return an exact job identifier: " + result.stdout)
        with (run_dir / "slurm/submissions.lock").open("a+") as ledger_lock:
            fcntl.flock(ledger_lock.fileno(), fcntl.LOCK_EX)
            for path, line in ((run_dir / "slurm/submitted_jobids_uva.txt", identifier + "\n"),
                    (run_dir / "slurm/submissions.tsv", "\t".join((identifier, "a4500-diagnostics",
                     batch["core"]["source_commit"], str(Path(directory) / "job.sbatch"))) + "\n")):
                with path.open("a") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
        common.write_once(Path(directory) / "submitted_jobid.record", (identifier + "\n").encode())
    return {"submitted": True, "job_id": identifier, "command": command}


def run_allocated(directory):
    """Resolve one allocated card, then run enabled and disabled controls in fresh processes."""
    directory = Path(directory).resolve(strict=True)
    batch = json.loads((directory / "batch_plan.json").read_text())
    code = directory / "code"
    for name, identity in batch["helpers"].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("a staged A4500 diagnostic helper changed: " + name)
    common, helper = load_helper(code / "launch_diagnostic.py"), load_helper(code / "diagnose_gpu_execution.py")
    queue = load_helper(code / "launch_queue.py")
    if socket.gethostname() != "jaguar03" or getpass.getuser() != "sl5nw":
        raise ValueError("this diagnostic must execute as sl5nw on allocated jaguar03")
    if nsight_identity(batch["nsight"]["path"]) != batch["nsight"]:
        raise ValueError("the allocated host's Nsight installation differs from the recorded executable")
    gpu_uuid = queue.allocated_gpu_uuid()
    allocation = {"gpu_uuid": gpu_uuid, "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "CUDA_VISIBLE_DEVICES_before": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"),
                  "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS")}
    common.write_once(directory / "allocation.json", common.encoded_json(allocation))
    job_directory = Path(batch["run_dir"]) / "slurm/jobs" / (allocation["slurm_job_id"] + "_jaguar03")
    common.write_once(job_directory / "allocation.json", common.encoded_json(allocation))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    cache = disabled_cache_environment()
    common.write_once(directory / "executed_cache_policy.json", common.encoded_json(cache))
    statuses = []
    for item in batch["units"]:
        unit = Path(item["directory"])
        template = json.loads((unit / "diagnostic_plan_template.json").read_text())
        plan = resolved_plan(template, gpu_uuid, allocation["slurm_job_id"])
        common.write_once(unit / "diagnostic_plan.json", common.encoded_json(plan))
        runtime = common.configure_runtime(item["expected_deterministic_operations"])
        runtime["compilation_cache"] = cache
        common.write_once(unit / "executed_runtime_environment.json", common.encoded_json(runtime))
        command = profile_command(plan, unit)
        common.write_once(unit / "executed_command.json", common.encoded_json(command))
        helper.record_progress(unit, "starting_diagnostic", expected_deterministic_operations=
                               item["expected_deterministic_operations"], gpu_uuid=gpu_uuid)
        failure = None
        try:
            status = common.bounded_command(command, plan["maximum_active_seconds"])
            if status:
                helper.atomic_json(unit / "trace/analysis.json", {
                    "completed": False, "exit_code": status, "stage": "capture_process"})
            if status == 0:
                # The shared helper exports through a short local path and verifies actual CUDA rows.
                helper.summarize_trace(unit, plan["nsight"]["path"])
        except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
            # Preserve the timing output and diagnose the failed capture; still run the separate control.
            status = 1
            failure = {"error": repr(error), "traceback": traceback.format_exc(), "stage": "trace_or_execution"}
            if not (unit / "trace/analysis.json").exists():
                helper.atomic_json(unit / "trace/analysis.json", {"completed": False, **failure})
        if status:
            helper.atomic_json(unit / "execution_failure.json", {"exit_code": status,
                "diagnostic_only": True, "expected_deterministic_operations": item["expected_deterministic_operations"],
                **(failure or {"stage": "numerical_or_trace_process", "error": "diagnostic subprocess exited nonzero"})})
            helper.record_progress(unit, "failed", exit_code=status)
        statuses.append({"unit_id": item["unit_id"], "exit_code": status})
    helper.atomic_json(directory / "result.json", {"completed": all(row["exit_code"] == 0 for row in statuses),
                       "diagnostic_only": True, "units": statuses, "allocation": allocation})
    if any(row["exit_code"] for row in statuses):
        raise RuntimeError("an A4500 diagnostic failed; original unit records and trace outputs are retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit", default=CORE_COMMIT)
    parser.add_argument("--python", type=Path, default=Path(PIXEL_PYTHON))
    parser.add_argument("--nsys", type=Path, default=Path(NSYS))
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run-allocated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-dir", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.run_allocated:
        if arguments.batch_dir is None:
            parser.error("--run-allocated requires --batch-dir")
        run_allocated(arguments.batch_dir)
    else:
        if arguments.run_dir is None or arguments.source_root is None:
            parser.error("preparation requires --run-dir and --source-root")
        print(json.dumps(prepare(arguments)), flush=True)
