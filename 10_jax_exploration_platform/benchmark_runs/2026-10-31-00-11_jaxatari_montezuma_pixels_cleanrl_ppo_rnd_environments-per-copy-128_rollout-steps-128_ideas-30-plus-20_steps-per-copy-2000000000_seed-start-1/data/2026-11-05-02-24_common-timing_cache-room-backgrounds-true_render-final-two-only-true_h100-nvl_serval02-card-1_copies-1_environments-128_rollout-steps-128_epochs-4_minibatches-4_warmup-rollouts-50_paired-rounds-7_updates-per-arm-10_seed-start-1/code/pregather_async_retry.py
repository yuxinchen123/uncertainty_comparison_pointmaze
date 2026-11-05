"""Authorize only the original pregather case's fifth attempt after a completed async diagnostic.

This standard-library metadata dependency preserves the original three-attempt
case and every fourth-attempt approval/claim. A separate approval grants only
attempt five, with a fresh execution of every original numerical check.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


def adjacent(name):
    """Resolve metadata dependencies next to this file without replacing frozen numerical modules."""
    specification = importlib.util.spec_from_file_location("pregather_retry_" + name, Path(__file__).resolve().with_name(name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


infra = adjacent("infrastructure_retry")
SOURCE_COMMIT = infra.SOURCE_COMMIT
APPROVALS_PATH = "queue/infrastructure_retries/pregather_cuda_async/approvals"
CLAIMS_PATH = "queue/infrastructure_retries/pregather_cuda_async/claims"
DIAGNOSTIC_SHA256 = "aec29b68efa95935dc1abdd27b4bfe677a225db382b36b7ce4622dab7f40c7d9"
DIAGNOSTIC_RAW_SHA256 = "b36faa7ec2d16012ce366866a30e640aa612da8c83b86eaeffc0dc425c929ef6"
DIAGNOSTIC_PARSER_SHA256 = "4e13cffffd60e689743057fd962f3c08c92cd3d9f77b9fc9b6067df04f49c8b9"
DIAGNOSTIC_PLAN_SHA256 = "eebc0f156bb419ca18dc309a7cfed06d9de2d88d40c36c9abdfbb01925779964"
ALLOCATOR = {**infra.ALLOCATOR, "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async"}
RUNTIME_KEYS = set(infra.RUNTIME) | set(ALLOCATOR) | set(infra.THREAD_ENVIRONMENT) | {
    "JAX_ENABLE_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "JAXTARI_SKIP_SPRITE_UPDATE",
    "TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION"}
HELPERS = ("pregather_async_retry.py", "run_pregather_async_retry.py", "run_infrastructure_retry.py",
    "infrastructure_retry.py", "infrastructure_retry_worker.py", "controller.py", "queue_worker.py", "ideas.py",
    "measure_gpu_scaling.py", "measure_selected.py", "train_optimized.py", "diagnose_gpu_execution.py", "post_comparison_validation.py")


class LocalEvidence(infra.LocalEvidence):
    """Provide streaming image locations while retaining exact original local identities."""

    def physical_file(self, path):
        """Locate a named local file without inferring another path."""
        return Path(path)

    def file(self, path, record=True):
        """Accept append-only ledger reads without adding a mutable whole-ledger receipt."""
        return super().file(path)

    def path_identity(self, path, strict=False):
        """Retain canonical absolute paths in locally produced receipts."""
        return str(Path(path).resolve(strict=strict))


def image(evidence, path, expected):
    """Check large installed/parser bytes in bounded chunks and retain their logical receipt identity."""
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("the async retry requires complete image SHA256 values")
    digest = hashlib.sha256()
    with evidence.physical_file(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("an async retry image/parser changed: " + str(path))
    if hasattr(evidence, "receipts"):
        identity = evidence.path_identity(path)
        record = {"path": identity, "sha256": expected}
        if identity in evidence.receipts and evidence.receipts[identity] != record:
            raise ValueError("an async retry image changed during validation")
        evidence.receipts[identity] = record


def validate_runtime(actual):
    """Require the explicit async allocator with every original precision, fraction and cache setting."""
    if (not isinstance(actual, dict) or set(actual) - RUNTIME_KEYS
            or any(actual.get(name) != value for name, value in {**infra.RUNTIME, **ALLOCATOR}.items())
            or actual.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
            or actual.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != ""
            or actual.get("JAXTARI_SKIP_SPRITE_UPDATE") != "1"
            or actual.get("TF_GPU_ALLOCATOR") not in (None, "") or actual.get("XLA_CLIENT_MEM_FRACTION") not in (None, "")):
        raise ValueError("the pregather retry changed its explicit async/full-precision runtime")


def runtime_environment(environment):
    """Retain only the named operational controls, never a broad prefix or inherited secret."""
    reject_unapproved_controls(environment)
    return {name: environment[name] for name in sorted(RUNTIME_KEYS) if name in environment}


def reject_unapproved_controls(environment):
    """Reject extra execution controls before projection; report their names without inherited values."""
    unexpected = sorted(name for name, value in environment.items()
        if value and name.startswith(("JAX_", "JAXTARI_", "XLA_")) and name not in RUNTIME_KEYS)
    if unexpected:
        raise ValueError("the fifth attempt inherited unapproved execution controls: " + ", ".join(unexpected))


def validate_parser(record, evidence):
    """Verify the actual installed parser bytes and its literal reported async options."""
    source, parsed = record["source"], record["parsed_options"]
    if (source.get("function") != "generate_pjrt_gpu_plugin_options" or "cuda_async" not in source.get("allowed_allocators", [])
            or parsed.get("allocator") != "cuda_async" or parsed.get("memory_fraction") != .85
            or parsed.get("preallocate") is not False or record.get("environment") != ALLOCATOR
            or record.get("memory_preparation") is not None):
        raise ValueError("the pregather retry did not use the supported original-control async parser")
    image(evidence, source["path"], source["sha256"])


def validate_libraries(installed, loaded, evidence):
    """Bind mapped CUDA/cuDNN observations to actual installed image bytes without importing arrays."""
    expected = installed["gpu_libraries"]
    verified = loaded.get("verified_package_libraries", {})
    mapped = {path for path in loaded.get("loaded_shared_libraries", []) if Path(path).name.startswith(("libcudnn", "libcudart"))}
    if (not expected or set(verified) != mapped or not mapped or any(verified[path] != expected.get(path) for path in mapped)
            or loaded.get("platform") != "gpu" or not loaded.get("platform_version")
            or any(type(loaded.get(name)) is not int or loaded[name] <= 0 for name in ("cudnnGetVersion", "cudaRuntimeGetVersion"))
            or any(loaded.get(name) not in verified for name in ("cudnn_library", "cuda_runtime_library"))):
        raise ValueError("the async retry lacks its actual mapped/installed CUDA and cuDNN evidence")
    for path, receipt in expected.items():
        image(evidence, path, receipt["sha256"])


def validate_core(receipt, evidence):
    """Check the diagnostic's original core schema, including its recorded Git object names."""
    normalized = {**receipt, "files": {path: value["sha256"] for path, value in receipt["files"].items()}}
    infra.validate_git_receipt(normalized, evidence)
    physical = evidence.git_root(receipt["source_root"], receipt["source_commit"])
    for path, value in receipt["files"].items():
        blob = subprocess.check_output(["git", "-C", str(physical), "rev-parse", receipt["source_commit"] + ":" + path], text=True).strip()
        if blob != value["git_blob"]:
            raise ValueError("the diagnostic's original numerical Git object changed")


def diagnostic_proof(receipt, case, evidence):
    """Rebuild the completed diagnostic's original checks, seven pairs, allocator, source and hardware evidence."""
    if receipt["sha256"] != DIAGNOSTIC_SHA256:
        raise ValueError("attempt five requires the exact completed and reviewed async diagnostic")
    result = infra.bound_document(receipt, evidence)
    directory = Path(receipt["path"]).parent
    plan = evidence.read(directory / "measurement_plan.json")
    prepared = evidence.read(directory / "measurement_prepared.json")
    expected_plan_sha = hashlib.sha256(evidence.file(directory / "measurement_plan.json")).hexdigest()
    raw = infra.bound_document(result["original_result"], evidence)
    hardware = evidence.read(directory / "hardware.json")
    parser = evidence.read(directory / "actual_allocator.json")
    installed = evidence.read(directory / "installed_backend.json")
    inputs = {name: infra.bound_document(record, evidence) for name, record in plan["inputs"].items()}
    if (result.get("completed") is not True or result.get("kind") != "pregather_async_allocator_diagnosis"
            or result.get("eligible_for_formal_campaign_admission") is not False or result.get("hypothesis_count_contribution") != 0
            or result.get("source_commit") != SOURCE_COMMIT or plan.get("case") != case
            or plan.get("configuration", {}).get("learner") != case["configuration"]
            or plan.get("source_root") != case["source_root"] or plan.get("source_commit") != SOURCE_COMMIT
            or (plan.get("rounds"), plan.get("iterations"), plan.get("copy_seeds"), plan.get("latency_calls")) != (7, 1, [1, 2], 3)
            or expected_plan_sha != DIAGNOSTIC_PLAN_SHA256 or prepared != {"plan_sha256": expected_plan_sha}
            or result["original_result"]["sha256"] != DIAGNOSTIC_RAW_SHA256
            or hashlib.sha256(evidence.file(directory / "actual_allocator.json")).hexdigest() != DIAGNOSTIC_PARSER_SHA256
            or result.get("helpers") != plan.get("helpers")
            or result.get("allocator") != ALLOCATOR or plan.get("allocator") != ALLOCATOR
            or plan.get("memory_preparation") is not None or plan.get("allocator_parser") != parser.get("source")
            or result["original_result"]["path"] != str(directory / "original_paired_result.json")
            or hardware.get("hostname") != "jaguar03" or "A4500" not in hardware.get("gpu_query", "")
            or hardware.get("gpu_uuid") != plan.get("gpu_uuid") or hardware.get("cuda_visible_devices") != plan.get("gpu_uuid")
            or type(hardware.get("pid")) is not int or not str(hardware.get("process_start_time", "")).isdigit()
            or len(hardware.get("cpu_affinity", [])) != 8 or hardware.get("thread_environment") != infra.THREAD_ENVIRONMENT):
        raise ValueError("the async diagnostic is incomplete or differs from the original case/allocator/physical target")
    if (inputs.get("case") != case or inputs.get("runtime_environment") != infra.RUNTIME
            or inputs.get("asset_record") != plan["asset_record"] or inputs.get("reference", {}).get("reference") != plan["configuration"]):
        raise ValueError("the async diagnostic's original input records changed")
    if any(result.get("hardware", {}).get(key) != hardware.get(key) for key in
           ("hostname", "gpu_uuid", "pid", "process_start_time", "slurm_job_id", "cuda_visible_devices")):
        raise ValueError("the diagnostic allocator/hardware files belong to another original process")
    validate_runtime(hardware["runtime_environment"])
    validate_parser(parser, evidence)
    validate_core(plan["core"], evidence)
    for source in (plan["paired_source"], plan["helpers"]):
        infra.validate_git_receipt(source, evidence)
    validation = adjacent("post_comparison_validation")
    validation.correctness(raw)
    if (raw.get("measurement_valid") is not True or raw.get("contaminated") is not False
            or raw.get("copies") != 2 or raw.get("copy_seeds") != [1, 2] or raw.get("source_commit") != SOURCE_COMMIT
            or raw.get("hardware_class") != "jaguar03_rtx_a4500" or len(raw.get("rounds", [])) != 7
            or raw.get("reference") != plan["configuration"]):
        raise ValueError("the async diagnostic has no complete original-source paired measurement")
    candidate = {scope: {**plan["configuration"][scope], **case["candidate"][scope]} for scope in ("learner", "environment", "execution")}
    if raw.get("candidate") != candidate:
        raise ValueError("the async diagnostic changed its supplied pregather candidate")
    for index, pair in enumerate(raw["rounds"]):
        if pair.get("order") != (["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]):
            raise ValueError("the async diagnostic changed its seven alternating pairs")
        for arm in ("reference", "candidate"):
            if pair[arm].get("iterations") != 1 or pair[arm].get("memory", {}).get("gpu_uuid") != hardware["gpu_uuid"]:
                raise ValueError("the async diagnostic changed its exact per-arm work or device")
            validation.measured_round(pair[arm], 2, minimum_updates=1)
    for arm in ("reference", "candidate"):
        if raw.get("synchronized_latency", {}).get(arm, {}).get("calls") != 3:
            raise ValueError("the async diagnostic omitted its original latency calls")
    rows = [json.loads(line) for line in evidence.file(directory / "memory_snapshots.jsonl").splitlines()]
    phases = {row.get("phase") for row in rows}
    if (not {"check_fixed_inputs_entered", "check_fixed_inputs_completed", "correctness_checks_entered",
             "correctness_checks_completed", "original_benchmark_completed"} <= phases
            or any(row.get("pid") != hardware["pid"] or row.get("device_memory", {}).get("gpu_uuid") != hardware["gpu_uuid"] for row in rows)):
        raise ValueError("the async diagnostic allocator/hardware lacks its actual same-process journal binding")
    validate_libraries(installed, result["loaded_gpu_libraries"], evidence)
    if result["loaded_gpu_libraries"].get("pid") != hardware["pid"]:
        raise ValueError("the diagnostic mapped libraries came from another process")
    assets = evidence.read(directory / "asset_receipt.json")
    manifest = infra.bound_document({"path": plan["asset_record"]["manifest"], "sha256": plan["asset_record"]["sha256"]}, evidence)
    if (assets.get("root") != plan["asset_record"]["root"] or assets.get("manifest_sha256") != plan["asset_record"]["sha256"]
            or assets.get("source_commit") != manifest.get("source_commit") or assets.get("file_count") != len(manifest.get("files", {}))
            or assets.get("tree_digest") != hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()):
        raise ValueError("the async diagnostic's actual assets differ from its immutable manifest")
    return {"result": result, "plan": plan, "hardware": hardware, "parser": parser, "installed": installed, "raw": raw, "assets": assets}


def validate_approval(path, evidence=None):
    """Permit only case twenty-three's exact fifth attempt and retain all preserved fourth-attempt evidence."""
    evidence = LocalEvidence() if evidence is None else evidence
    approval = evidence.read(path)
    case = infra.validate_case(evidence.file(approval["case"]["path"]))
    run, name = Path(case["campaign_run_dir"]), case["unit_id"] + ".json"
    if (case["hypothesis_position"] != 23 or approval.get("kind") != "pregather_cuda_async_formal_retry"
            or approval.get("schema_version") != 1 or approval.get("approved_attempt") != 5 or approval.get("previous_attempt") != 4
            or approval.get("unit_id") != case["unit_id"] or approval.get("campaign_run_dir") != str(run)
            or Path(path) != run / APPROVALS_PATH / name or approval["case"] != infra.file_record(run / "queue/specifications" / name, evidence)
            or approval.get("runtime_environment") != infra.RUNTIME or approval.get("allocator") != ALLOCATOR
            or approval.get("thread_environment") != infra.THREAD_ENVIRONMENT or approval.get("memory_preparation") is not None):
        raise ValueError("attempt five approval changed its one exact case, allocator or original limit")
    previous = approval["previous"]
    old, old_case = infra.validate_approval(previous["approval"]["path"], evidence)
    infra.bound_document(previous["approval"], evidence)
    directory = run / "data" / case["unit_id"] / "attempts/attempt-0004"
    expected_paths = {"case": directory / "case.json", "lease": directory / "lease.json", "failure": directory / "failure.json",
        "hardware": directory / "hardware.json", "claim": run / infra.CLAIMS_PATH / name}
    documents = {key: infra.bound_document(previous[key], evidence) for key in expected_paths}
    if any(previous[key]["path"] != str(value) for key, value in expected_paths.items()):
        raise ValueError("attempt five does not retain the exact prior case/lease/failure/claim")
    lease, failure, claim, hardware = (documents[key] for key in ("lease", "failure", "claim", "hardware"))
    if (old_case != case or documents["case"] != case or lease.get("attempt") != 4 or lease.get("state") != "failed"
            or lease.get("infrastructure_retry_approval") != previous["approval"] or claim.get("approval") != previous["approval"]
            or claim["lease"].get("token") != lease.get("token") or failure.get("attempt") != 4
            or failure.get("case_completed") is not False or failure.get("source_commit") != SOURCE_COMMIT
            or "jit_update" not in failure.get("error", "") or "Out of memory" not in failure.get("error", "")
            or failure.get("pid") != lease.get("runner_pid") or hardware.get("pid") != lease.get("runner_pid")
            or hardware.get("process_start_time") != lease.get("runner_start_time")):
        raise ValueError("attempt five is not linked to the actual preserved fourth-attempt update allocation failure")
    if (approval.get("numerical_source") != old["numerical_source"] or any(approval.get(key) != old[key] for key in
            ("asset_record", "jaxtari_source_root", "python")) or approval["helpers"]["source_commit"] == old["helpers"]["source_commit"]
            or set(approval["helpers"]["files"]) != {infra.PREFIX + name for name in HELPERS}):
        raise ValueError("attempt five changed original assets, numerical source or its committed operational closure")
    infra.validate_git_receipt(approval["helpers"], evidence)
    infra.validate_git_receipt(approval["numerical_source"], evidence)
    diagnostic = diagnostic_proof(approval["diagnostic"], case, evidence)
    if (diagnostic["plan"]["asset_record"] != approval["asset_record"]
            or diagnostic["plan"]["inputs"]["original_failure"] != previous["failure"]
            or diagnostic["plan"]["inputs"]["case"] != previous["case"]):
        raise ValueError("attempt five changed the completed diagnostic's original case, failure or asset input")
    return approval, case


def validate_execution(result_path, case, lease=None, evidence=None):
    """Require a fresh original result plus exact attempt-five process, allocator and diagnostic linkage."""
    evidence = LocalEvidence() if evidence is None else evidence
    directory = Path(result_path).parent
    sidecar = evidence.read(directory / "pregather_async_retry_execution.json")
    infra.bound_document(sidecar["approval"], evidence)
    approval, original = validate_approval(sidecar["approval"]["path"], evidence)
    saved = evidence.read(directory / "lease.json")
    hardware = evidence.read(directory / "hardware.json")
    result = evidence.read(result_path)
    claim = evidence.read(Path(case["campaign_run_dir"]) / CLAIMS_PATH / (case["unit_id"] + ".json"))
    expected = {"case_completed": True, "attempt": 5, "unit_id": case["unit_id"], "idea": case["hypothesis"],
        "source_commit": SOURCE_COMMIT, "specification_digest": case["specification_digest"], "hardware_class": "jaguar03_rtx_a4500", "copies": 2}
    if (original != case or case["hypothesis_position"] != 23 or sidecar.get("schema_version") != 1
            or sidecar.get("unit_id") != case["unit_id"] or sidecar.get("completed") is not True or sidecar.get("attempt") != 5
            or any(result.get(key) != value for key, value in expected.items())
            or sidecar.get("original_result") != infra.file_record(result_path, evidence)
            or sidecar.get("helpers") != approval["helpers"] or sidecar.get("numerical_source") != approval["numerical_source"]
            or sidecar.get("memory_preparation") is not None or saved.get("attempt") != 5
            or sidecar.get("approval") != saved.get("pregather_async_retry_approval") or claim.get("approval") != sidecar["approval"]
            or sidecar.get("lease_token") != saved.get("token") or claim["lease"].get("token") != saved.get("token")
            or claim["lease"].get("attempt") != 5
            or directory != Path(case["campaign_run_dir"]) / "data" / case["unit_id"] / "attempts/attempt-0005"):
        raise ValueError("the fifth attempt lacks its fresh exact result, one-shot claim or separate approval")
    if saved.get("state") not in {"running", "completed"} or lease is None and saved.get("state") != "completed":
        raise ValueError("the fifth attempt has not completed its queue transition")
    if lease is not None and any(lease.get(key) != saved.get(key) for key in ("token", "attempt", "runner_pid", "runner_start_time", "pregather_async_retry_approval")):
        raise ValueError("the fifth attempt's live lease differs from its recorded execution")
    if (sidecar.get("pid") != saved.get("runner_pid") or sidecar.get("pid") != hardware.get("pid")
            or sidecar.get("process_start_time") != saved.get("runner_start_time") or sidecar.get("process_start_time") != hardware.get("process_start_time")
            or sidecar.get("gpu_uuid") != saved.get("gpu_uuid") or sidecar.get("gpu_uuid") != hardware.get("gpu_uuid")
            or any(claim["lease"].get(key) != saved.get(key) for key in ("worker_pid", "worker_start_time", "gpu_uuid", "assignment"))):
        raise ValueError("the fifth attempt's original runner belongs to another process, GPU or claim")
    assignment = infra.bound_document(sidecar["assignment"], evidence)
    infra.validate_assignment(assignment, hardware["gpu_uuid"])
    cpus = infra.validate_cpu_allocation(assignment, hardware["gpu_uuid"])
    owned = infra.bound_document(sidecar["owned_allocation"], evidence)
    if (type(assignment["expected_gpu_count"]) is not int or assignment["expected_gpu_count"] != 1
            or sidecar.get("assignment") != saved.get("assignment") or sidecar["approval"] not in assignment.get("approvals", [])
            or sidecar.get("cpu_affinity") != cpus or saved.get("cpu_affinity") != cpus
            or sidecar.get("thread_environment") != infra.THREAD_ENVIRONMENT
            or owned.get("job_id") != hardware.get("slurm_job_id") or owned.get("hostname") != "jaguar03"
            or owned.get("pid") != sidecar["pid"] or owned.get("process_start_time") != sidecar["process_start_time"]
            or owned.get("gpu_uuid") != hardware["gpu_uuid"] or owned.get("ledger_path") != str(Path(case["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt")
            or saved.get("worker_pid") != assignment.get("parent_pid") or saved.get("worker_start_time") != assignment.get("parent_process_start_time")
            or assignment["allocation"]["slurm_job_id"] != owned.get("job_id")
            or not re.search(r"(?:^|/)job_" + re.escape(owned["job_id"]) + r"(?:/|$)", owned.get("cgroup", ""), re.MULTILINE)
            or owned["job_id"] not in owned.get("ledger_snapshot", "").splitlines()):
        raise ValueError("the fifth attempt lacks its owned single-GPU/eight-CPU allocation proof")
    current_ledger = evidence.file(owned["ledger_path"], record=False).decode().splitlines()
    if current_ledger[:len(owned["ledger_snapshot"].splitlines())] != owned["ledger_snapshot"].splitlines():
        raise ValueError("the fifth attempt lost its original append-only owned UVA ledger")
    validate_runtime(sidecar["runtime_environment"])
    initial = sidecar["process_environment"]
    if any(initial.get(key) != value for key, value in {**infra.RUNTIME, **ALLOCATOR, **infra.THREAD_ENVIRONMENT, "CUDA_VISIBLE_DEVICES": hardware["gpu_uuid"]}.items()):
        raise ValueError("the fifth attempt did not inherit its exact explicit process-start controls")
    if any(initial.get(name) not in (None, "") for name in ("TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION")):
        raise ValueError("the fifth attempt inherited a conflicting allocator alias")
    initial_record = infra.bound_document(sidecar["process_environment_record"], evidence)
    if (initial_record.get("values") != initial or initial_record.get("source") != "/proc/self/environ allowlist"
            or any(initial_record.get(key) != sidecar.get(key) for key in ("pid", "process_start_time", "gpu_uuid"))
            or initial.get("SLURM_JOB_ID") != owned["job_id"]
            or any(initial.get(key) != value for key, value in assignment["allocation"]["slurm_allocations"].items())):
        raise ValueError("the fifth attempt's filtered process environment lacks its exact allocation/process binding")
    parser = infra.bound_document(sidecar["actual_allocator"], evidence)
    validate_parser(parser, evidence)
    diagnostic = diagnostic_proof(approval["diagnostic"], case, evidence)
    if (parser["source"] != diagnostic["parser"]["source"] or parser["parsed_options"] != diagnostic["parser"]["parsed_options"]
            or any(parser.get(key) != sidecar.get(key) for key in ("pid", "process_start_time", "gpu_uuid"))):
        raise ValueError("the formal fifth attempt used another installed allocator parser")
    installed = evidence.read(directory / "installed_backend.json")
    if installed != diagnostic["installed"]:
        raise ValueError("the formal fifth attempt changed the diagnostic's installed backend")
    if evidence.read(directory / "asset_receipt.json") != diagnostic["assets"]:
        raise ValueError("the formal fifth attempt changed the completed diagnostic's actual assets")
    validate_libraries(installed, sidecar["loaded_gpu_libraries"], evidence)
    if sidecar["loaded_gpu_libraries"].get("pid") != sidecar["pid"]:
        raise ValueError("the fifth attempt's mapped libraries came from another process")
    if (any(sidecar["loaded_gpu_libraries"].get(key) != diagnostic["result"]["loaded_gpu_libraries"].get(key)
            for key in ("platform", "platform_version", "cudnnGetVersion", "cudaRuntimeGetVersion"))
            or hardware["gpu_query"].split(",")[1:] != diagnostic["hardware"]["gpu_query"].split(",")[1:]
            or (sidecar["pid"], sidecar["process_start_time"]) == (diagnostic["hardware"]["pid"], diagnostic["hardware"]["process_start_time"])
            or owned["job_id"] == diagnostic["hardware"]["slurm_job_id"]):
        raise ValueError("attempt five reused the diagnostic process or changed its actual backend/physical target")
    start, end = sidecar.get("started_at_seconds"), sidecar.get("finished_at_seconds")
    expected_runner = str(Path(approval["helpers"]["source_root"]) / infra.PREFIX / "run_pregather_async_retry.py")
    if (sidecar.get("runner") != expected_runner or type(start) not in (int, float) or type(end) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(end) or not 0 < start < end):
        raise ValueError("the fifth attempt lacks its exact helper entrypoint and execution interval")
    expected_modules = {name: str(Path(case["source_root"]) / infra.PREFIX / (name + ".py"))
        for name in ("run_case", "controller", "ideas", "paired_benchmark")}
    if sidecar.get("frozen_modules") != expected_modules:
        raise ValueError("the fifth attempt imported another original numerical source module")
    return {"scope": "actual_pregather_async_fifth_attempt_runtime", "runtime_environment": dict(infra.RUNTIME),
        "actual_runtime_environment": sidecar["runtime_environment"], "approval": sidecar["approval"],
        "execution_record": infra.file_record(directory / "pregather_async_retry_execution.json", evidence),
        "helper_commit": approval["helpers"]["source_commit"], "allocator": "cuda_async", "memory_preparation": None}


def prepare(arguments):
    """Publish only the new fifth-attempt approval after every immutable prerequisite has passed."""
    run, helper = arguments.run_dir.resolve(strict=True), arguments.helper_root.resolve(strict=True)
    matches = list((run / "queue/failed").glob("idea-position-23__*jaguar03*copies-2__*.json"))
    if len(matches) != 1:
        raise ValueError("the original two-copy pregather case must be held failed before approval")
    case = infra.validate_case(matches[0].read_bytes())
    evidence = LocalEvidence()
    name = case["unit_id"] + ".json"
    prior = run / "data" / case["unit_id"] / "attempts/attempt-0004"
    saved = evidence.read(run / "queue/leases" / name)
    if saved != evidence.read(prior / "lease.json") or saved.get("attempt") != 4 or saved.get("state") != "failed":
        raise ValueError("attempt five requires the unchanged preserved failed fourth lease")
    if (run / CLAIMS_PATH / name).exists() or prior.with_name("attempt-0005").exists():
        raise ValueError("the single fifth attempt was already consumed; no sixth attempt is granted")
    old_record = saved["infrastructure_retry_approval"]
    old, _ = infra.validate_approval(old_record["path"], evidence)
    commit = subprocess.check_output(["git", "-C", str(helper), "rev-parse", "HEAD"], text=True).strip()
    previous = {key: infra.file_record(prior / filename, evidence) for key, filename in
        (("case", "case.json"), ("lease", "lease.json"), ("failure", "failure.json"), ("hardware", "hardware.json"))}
    previous.update(approval=old_record, claim=infra.file_record(run / infra.CLAIMS_PATH / name, evidence))
    approval = {"schema_version": 1, "kind": "pregather_cuda_async_formal_retry", "approved_attempt": 5, "previous_attempt": 4,
        "unit_id": case["unit_id"], "campaign_run_dir": str(run), "case": infra.file_record(run / "queue/specifications" / name, evidence),
        "previous": previous, "diagnostic": infra.file_record(arguments.diagnostic_result.resolve(strict=True), evidence),
        "helpers": infra.git_receipt(helper, commit, [infra.PREFIX + name for name in HELPERS]),
        "numerical_source": old["numerical_source"], "runtime_environment": infra.RUNTIME, "allocator": ALLOCATOR,
        "thread_environment": infra.THREAD_ENVIRONMENT, "memory_preparation": None,
        **{key: old[key] for key in ("asset_record", "jaxtari_source_root", "python")}}
    path = run / APPROVALS_PATH / name
    # Validate before publication through a reader that exposes just this not-yet-published document.
    class PlannedEvidence(LocalEvidence):
        """Let the production validator inspect a pending approval without publishing it early."""
        def read(self, candidate):
            """Keep every prerequisite read unchanged while supplying the one planned approval."""
            return approval if Path(candidate) == path else super().read(candidate)
    validate_approval(path, PlannedEvidence())
    infra.write_once(path, infra.json_bytes(approval))
    return path


def owned_allocation(run, gpu_uuid):
    """Read this exact UVA job/cgroup and physical card before any numerical context is created."""
    import socket
    run = Path(run)
    job = os.environ.get("SLURM_JOB_ID", "")
    ledger_path = run / "slurm/submitted_jobids_uva.txt"
    ledger = ledger_path.read_text()
    cgroup = Path("/proc/self/cgroup").read_text()
    if (not re.fullmatch(r"[1-9][0-9]*", job) or job not in ledger.splitlines()
            or socket.gethostname().split(".")[0] != "jaguar03"
            or not re.search(r"(?:^|/)job_" + re.escape(job) + r"(?:/|$)", cgroup, re.MULTILINE)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid):
        raise ValueError("the fifth attempt requires this campaign's exact owned UVA GPU allocation")
    text = Path(f"/proc/{os.getpid()}/stat").read_text()
    return {"job_id": job, "hostname": "jaguar03", "pid": os.getpid(), "process_start_time": text.rsplit(")", 1)[1].split()[19],
        "gpu_uuid": gpu_uuid, "cgroup": cgroup, "ledger_path": str(ledger_path), "ledger_snapshot": ledger}


def claim(run, approval_path, identity, assignment_path):
    """Consume the separate one-shot fifth approval directly from failed into running under the queue lock."""
    import controller
    run = Path(run)
    with controller.queue_lock(run):
        approval, case = validate_approval(approval_path)
        name = case["unit_id"] + ".json"
        claim_path = run / CLAIMS_PATH / name
        if claim_path.exists():
            return None
        ready = controller.ready_specification(run)
        lease_path = run / "queue/leases" / name
        previous = controller.read_json(lease_path)
        failed = run / "queue/failed" / name
        if (ready is None or (ready["source_root"], ready["source_commit"]) != (case["source_root"], SOURCE_COMMIT)
                or controller.case_states(run, case["unit_id"]) != ["failed"]
                or failed.read_bytes() != LocalEvidence().file(approval["case"]["path"])
                or previous.get("attempt") != 4 or previous.get("state") != "failed"
                or infra.digest(lease_path.read_bytes()) != approval["previous"]["lease"]["sha256"]):
            raise ValueError("the fifth attempt's prior case/lease/admission changed before claim")
        _, attempt = controller.attempt_paths(run, case["unit_id"], 5)
        if attempt.exists():
            raise ValueError("attempt five was already consumed; no sixth attempt is granted")
        record = infra.file_record(approval_path, LocalEvidence())
        lease = {**identity, "token": uuid.uuid4().hex, "unit_id": case["unit_id"], "attempt": 5,
            "state": "claimed", "claimed_at": controller.utc_now(), "runner_pid": None, "runner_start_time": None,
            "pregather_async_retry_approval": record, "assignment": infra.file_record(assignment_path, LocalEvidence())}
        infra.write_once(claim_path, infra.json_bytes({"approval": record, "lease": lease}))
        attempt.mkdir(parents=True, exist_ok=False)
        controller.atomic_json(attempt / "lease.json", lease)
        controller.atomic_json(lease_path, lease)
        failed.rename(run / "queue/running" / name)
        return approval, case, lease


def finish(run, case, lease, code):
    """Complete a proved original result, or hold a failed fifth attempt with no automatic further allowance."""
    import controller
    import queue_worker
    _, attempt = controller.attempt_paths(run, case["unit_id"], 5)
    result_path, failure_path = attempt / "result.json", attempt / "failure.json"
    complete = result_path.exists() and controller.scientific_case_complete(controller.read_json(result_path)) and not failure_path.exists()
    if complete:
        try:
            validate_execution(result_path, case, lease)
        except (ValueError, KeyError, OSError) as error:
            controller.atomic_json(failure_path, {"case_completed": False, "attempt": 5, "retryable": False,
                "failure_stage": "pregather_async_retry_proof", "error": str(error), "time": controller.utc_now()})
            complete = False
    if complete:
        return queue_worker.finish_claim(run, case, lease, code)
    with controller.queue_lock(run):
        queue = Path(run) / "queue"
        lease_path = queue / "leases" / (case["unit_id"] + ".json")
        current = controller.read_json(lease_path)
        if current["token"] != lease["token"]:
            raise ValueError("the fifth attempt no longer owns its lease")
        if not failure_path.exists():
            controller.atomic_json(failure_path, {"case_completed": False, "attempt": 5, "retryable": False,
                "failure_stage": "pregather_async_retry_exit", "exit_code": code, "time": controller.utc_now()})
        unit = attempt.parent.parent
        controller.atomic_json(unit / "failure.json", controller.read_json(failure_path))
        current.update(state="failed", runner_exit_code=code, finished_at=controller.utc_now())
        controller.atomic_json(attempt / "lease.json", current)
        controller.atomic_json(lease_path, current)
        (queue / "running" / (case["unit_id"] + ".json")).rename(queue / "failed" / (case["unit_id"] + ".json"))
        controller.atomic_json(unit / "progress.json", {"state": "failed", "attempt": 5, "unit_id": case["unit_id"],
            "retry_allowance_exhausted": True, "time": controller.utc_now()})
    return "failed"


def run_worker(arguments):
    """Use one exact allocated A4500 to execute only the single approved case, without submitting jobs."""
    import controller
    import queue_worker
    import infrastructure_retry_worker as cleanup
    import measure_gpu_scaling as allocation
    run = arguments.run_dir.resolve(strict=True)
    approval_path = arguments.approval.resolve(strict=True)
    approval, case = validate_approval(approval_path)
    expected = Path(approval["helpers"]["source_root"]) / infra.PREFIX / "pregather_async_retry.py"
    if Path(__file__).resolve() != expected:
        raise ValueError("execute the fifth-attempt worker from its exact approved helper root")
    reject_unapproved_controls(os.environ)
    gpu = arguments.gpu_uuid
    owned_allocation(run, gpu)
    if any(os.environ.get(name) not in (None, "") for name in ("TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION")):
        raise ValueError("the fifth-attempt worker inherited a conflicting allocator alias")
    identity = queue_worker.worker_identity("jaguar03_rtx_a4500", queue_worker.gpu_identity(gpu, "jaguar03_rtx_a4500"))
    proof = allocation.allocation_proof({"gpu_count": 1, "gpu_uuids": [gpu]}, os.environ, "jaguar03")
    assignment = {"schema_version": 1, "hostname": "jaguar03", "parent_pid": os.getpid(),
        "parent_process_start_time": controller.process_start_time(os.getpid()), "gpu_uuids": [gpu], "expected_gpu_count": 1,
        "allocation": proof, "cpu_allocation": infra.cpu_allocation(os.environ, [gpu], os.sched_getaffinity(0)),
        "approvals": [infra.file_record(approval_path, LocalEvidence())]}
    cpus = cleanup.bind_worker_cpus(assignment, gpu, os.environ)
    identity.update(cpu_affinity=cpus, thread_environment=dict(infra.THREAD_ENVIRONMENT))
    evidence = diagnostic_proof(approval["diagnostic"], case, LocalEvidence())
    adjacent("train_optimized").enforce_runtime(infra.RUNTIME, os.environ)
    runtime = runtime_environment({**os.environ, **ALLOCATOR})
    for name, value in ALLOCATOR.items():
        if name != "XLA_PYTHON_CLIENT_ALLOCATOR" and os.environ.get(name) != value:
            raise ValueError("the fifth attempt must retain the original fraction/preallocation controls")
    directory = arguments.output_dir.resolve()
    if run not in directory.parents or directory.exists():
        raise ValueError("use a fresh campaign child directory for the operational fifth-attempt worker")
    directory.mkdir(parents=True)
    assignment_path = directory / "assignment.json"
    infra.write_once(assignment_path, infra.json_bytes(assignment))
    for name, value in (("manifest.yaml", {"kind": "pregather_async_fifth_attempt_worker", "approval": assignment["approvals"][0],
            "helpers": approval["helpers"], "source_commit": SOURCE_COMMIT}), ("config_resolved.yaml", {
            "configuration": case["configuration"], "candidate": case["candidate"], "runtime": runtime, "allocator": ALLOCATOR})):
        infra.write_once(directory / name, infra.json_bytes(value))
    worker_background(directory, arguments, approval, case)
    with (Path("/tmp") / f"sl5nw_atari_pixel_benchmark_{gpu}.lock").open("a+") as gpu_lock, cleanup.interruption_cleanup():
        import fcntl
        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not queue_worker.gpu_readiness(gpu, "jaguar03_rtx_a4500")["ready"]:
            raise ValueError("the fifth attempt's selected physical GPU is occupied")
        claimed = claim(run, approval_path, identity, assignment_path)
        if claimed is None:
            return {"state": "already_claimed"}
        approval, case, lease = claimed
        _, attempt = controller.attempt_paths(run, case["unit_id"], 5)
        child = []
        code = -1
        try:
            command = [approval["python"], str(Path(approval["helpers"]["source_root"]) / infra.PREFIX / "run_pregather_async_retry.py"),
                "--approval", str(approval_path), "--assignment", str(assignment_path), "--lease-token", lease["token"]]
            import shlex
            infra.write_once(attempt / "command.txt", (shlex.join(command) + "\n").encode())
            with (attempt / "runner.log").open("x") as log:
                process = subprocess.Popen(command, cwd=case["source_root"], stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(gpu_lock.fileno(),), env={**os.environ, **ALLOCATOR, **infra.THREAD_ENVIRONMENT})
                owner = {"process": process, "process_start_time": None}
                child.append(owner)
                owner["process_start_time"] = controller.process_start_time(process.pid)
                if owner["process_start_time"] is None:
                    raise RuntimeError("the fifth-attempt child ended before recording its birth counter")
                lease.update(runner_pid=process.pid, runner_start_time=owner["process_start_time"], state="running")
                with controller.queue_lock(run):
                    current = controller.read_json(run / "queue/leases" / (case["unit_id"] + ".json"))
                    if current.get("token") != lease["token"] or current.get("state") != "claimed" or current.get("attempt") != 5:
                        raise ValueError("the fifth-attempt worker lost its exact claim before child release")
                    controller.atomic_json(attempt / "lease.json", lease)
                    controller.atomic_json(run / "queue/leases" / (case["unit_id"] + ".json"), lease)
                code = process.wait()
        except BaseException:
            cleanup.stop_owned(child)
            finish(run, case, lease, code)
            raise
        return {"state": finish(run, case, lease, code), "attempt": 5, "unit_id": case["unit_id"]}


def worker_background(directory, arguments, approval, case):
    """Record this metadata command separately from the original numerical child's attempt command."""
    import shlex
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--run-dir", str(arguments.run_dir.resolve()),
        "--approval", str(arguments.approval.resolve()), "--output-dir", str(directory), "--gpu-uuid", arguments.gpu_uuid]
    infra.write_once(directory / "command.txt", (shlex.join(command) + "\n").encode())
    background = (
        "## Purpose\n\nRun one separately approved fifth attempt of the original two-copy pregather case. "
        "The original retry limit remains three; every earlier attempt and approval remains preserved.\n\n"
        "## Key hyperparameters\n\nUse two independent policies with 128 environments and 128 rollout steps each, "
        "four epochs, four minibatches, fifty normalization warmup rollouts and seeds one and two. Run the original "
        "fixed-input and complete-iteration checks, seven alternating one-update pairs and three latency calls.\n\n"
        "## Code and config changes\n\nSelect the explicit cuda_async allocator with fraction 0.85 and preallocation false. "
        "Keep float32/highest neural arithmetic, float64 statistics, deterministic operations and all original numerical "
        "checks and timing. Do not prepare a nine-GiB array. This approval grants no sixth attempt.\n\n"
        "## Git state\n\nNumerical source: " + case["source_commit"] + ". Metadata helpers: "
        + approval["helpers"]["source_commit"] + ". Exact source, helper and approval hashes are in manifest.yaml.\n")
    infra.write_once(directory / "experiment_background.md", background.encode())


def main():
    """Prepare one explicit approval or run its already allocated metadata worker; submit nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("run-dir", "helper-root", "diagnostic-result"):
        preparation.add_argument("--" + name, type=Path, required=True)
    worker = commands.add_parser("worker")
    for name in ("run-dir", "approval", "output-dir"):
        worker.add_argument("--" + name, type=Path, required=True)
    worker.add_argument("--gpu-uuid", required=True)
    arguments = parser.parse_args()
    result = {"approval": str(prepare(arguments))} if arguments.command == "prepare" else run_worker(arguments)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
