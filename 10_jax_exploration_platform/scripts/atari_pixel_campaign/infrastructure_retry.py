"""Validate the five explicitly approved allocator-history retries without importing arrays.

An approval permits only attempt four of one original case. It never changes the
case's retry limit, numerical source, checks, timing window, or old attempt files.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
DIAGNOSIS_SHA256 = "0a5619b813dd3355357e2a3afdeb39b315a78d5413a5746a7d9bed1d8c3d7005"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
REGION_BYTES = 9 * 2 ** 30
PREPARATION = {"phase": "before learner/game imports and initialization", "allocation_count": 1,
               "dtype": "uint8", "shape": [9, 1024, 1024, 1024], "bytes": REGION_BYTES}
RUNTIME = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
           "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
ALLOCATOR = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"}
THREAD_ENVIRONMENT = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
APPROVED = {
    16: ("fuse_network_bias_and_activation", {"fuse_network_bias_and_activation": True},
         "52282dc139fb2e9b84e10b5bbdb5ad1c9bb6e77ba38936715b428a66b7e0c0b1"),
    17: ("cache_frozen_training_target_features", {"cache_target_features": True},
         "fe22fcb6564356127a0af5f4f3e0f74b466d021c1ef3378232b044529ed4ed73"),
    18: ("batch_rollout_intrinsic_rewards", {"hoist_rollout_bonus": True},
         "b7b361d51c54981587569bc0920217c9d46c0ae4d86889c101646fb769523abe"),
    22: ("unroll_minibatch_update_loop", {"update_unroll": 2},
         "cfc7a67483fbe7408c4f123d05b90e0f9d861417e84c82e04f5749372e10dc91"),
    23: ("gather_epoch_minibatches_in_advance", {"pre_gather_minibatches": True},
         "d661816f161c9a0b756d9cb3d376c1553e662292204dfc881095aacf4613037e"),
}
HELPERS = ("infrastructure_retry.py", "infrastructure_retry_worker.py", "run_infrastructure_retry.py",
           "controller.py", "queue_worker.py", "ideas.py", "train_optimized.py",
           "diagnose_gpu_execution.py", "post_comparison_validation.py", "measure_gpu_scaling.py", "measure_selected.py")


def digest(contents):
    """Identify the original bytes, including their JSON whitespace."""
    return hashlib.sha256(contents).hexdigest()


def json_bytes(value):
    """Write complete finite JSON as both machine metadata and resolved YAML."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def write_once(path, contents):
    """Publish new immutable bytes without replacing an earlier approval or execution record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError("an immutable retry artifact already differs: " + str(path))
        return
    # A complete temporary file is linked into place; an interruption cannot publish partial JSON.
    temporary = path.with_name(path.name + ".temporary-" + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


class LocalEvidence:
    """Use the same small reader interface as the portable final-evidence validator."""

    def file(self, path):
        """Require an absolute original artifact path."""
        if not Path(path).is_absolute():
            raise ValueError("retry evidence paths must be absolute")
        return Path(path).read_bytes()

    def read(self, path):
        """Decode one original JSON document."""
        return json.loads(self.file(path))

    def git_root(self, root, commit):
        """Keep local Git roots unchanged; a portable reader supplies explicit mappings."""
        return Path(root)


def file_record(path, evidence):
    """Retain the exact original path and hash without rewriting its logical identity."""
    return {"path": str(path), "sha256": digest(evidence.file(path))}


def bound_document(record, evidence):
    """Reject a replaced source document before using any of its fields."""
    contents = evidence.file(record["path"])
    if digest(contents) != record["sha256"]:
        raise ValueError("retry evidence bytes changed: " + record["path"])
    return json.loads(contents)


def validate_case(contents):
    """Admit only the five exact case files approved in this session."""
    case = json.loads(contents)
    position = case.get("hypothesis_position")
    if position not in APPROVED:
        raise ValueError("this case has no approved allocator-history retry")
    name, change, expected = APPROVED[position]
    if (digest(contents) != expected or case.get("hypothesis") != name
            or case.get("candidate") != {"learner": change, "environment": {}, "execution": {}}
            or case.get("source_commit") != SOURCE_COMMIT or case.get("hardware_class") != "jaguar03_rtx_a4500"
            or case.get("maximum_attempts") != 3 or (case.get("rounds"), case.get("iterations"), case.get("seed_start")) != (7, 1, 1)
            or case.get("configuration", {}).get("n_copies") != 2):
        raise ValueError("the approved original case bytes, source, limit or protocol differ")
    return case


def git_receipt(root, commit, paths, evidence=None):
    """Verify an actual clean detached Git root and every file against its committed blob."""
    evidence = LocalEvidence() if evidence is None else evidence
    physical = evidence.git_root(root, commit)
    command = ["git", "-C", str(physical)]
    top = subprocess.check_output([*command, "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output([*command, "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    if Path(top).resolve() != Path(physical).resolve() or head != commit or branch != "HEAD":
        raise ValueError("retry source requires its own exact detached Git root")
    if subprocess.check_output([*command, "status", "--porcelain", "--untracked-files=all", "--", *paths], text=True).strip():
        raise ValueError("retry source has uncommitted helper or numerical changes")
    files = {}
    for relative in paths:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("retry source paths must stay inside their Git root")
        contents = evidence.file(Path(root) / relative)
        original = subprocess.check_output([*command, "show", commit + ":" + relative])
        if contents != original:
            raise ValueError("retry source differs from its committed blob: " + relative)
        files[relative] = digest(contents)
    return {"source_root": str(root), "source_commit": commit, "files": files}


def validate_git_receipt(receipt, evidence):
    """Rebuild a source receipt rather than accepting its supplied hashes."""
    if git_receipt(receipt["source_root"], receipt["source_commit"], sorted(receipt["files"]), evidence) != receipt:
        raise ValueError("retry source receipt changed")


def numerical_source_paths(root, evidence):
    """Require the complete pinned learner and original case-runner Python source closure."""
    physical = evidence.git_root(root, SOURCE_COMMIT)
    paths = subprocess.check_output(["git", "-C", str(physical), "ls-tree", "-r", "--name-only", SOURCE_COMMIT,
        "--", "10_jax_exploration_platform/src/exploration_platform/atari_pixels", PREFIX], text=True).splitlines()
    return sorted(path for path in paths if (path.endswith(".py") and "/tests/" not in path)
                  or path == PREFIX + "worker_environment.sh")


def validate_region(record, gpu_uuid):
    """Require real before/allocated/released memory and exactly one synchronized nine-GiB array."""
    if (record.get("allocation_count") != 1 or record.get("array_nbytes") != REGION_BYTES
            or record.get("array_dtype") != "uint8" or record.get("array_shape") != PREPARATION["shape"]
            or record.get("allocator_pool_release_assumed") is not False):
        raise ValueError("retry memory preparation differs from the approved allocation")
    times = []
    for phase in ("before", "allocated", "released"):
        point = record[phase]
        timestamp = point["time_seconds"]
        memory, allocator = point["device_memory"], point["allocator"]
        observed_bytes = [memory.get(name) for name in ("total_bytes", "used_bytes", "free_bytes")]
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp <= 0
                or memory.get("gpu_uuid") != gpu_uuid or any(type(value) is not int or value < 0 for value in observed_bytes)
                or memory["total_bytes"] <= 0 or memory["used_bytes"] + memory["free_bytes"] > memory["total_bytes"]
                or allocator.get("bytes_limit", 0) <= REGION_BYTES):
            raise ValueError("retry preparation lacks actual selected-device memory observations")
        times.append(timestamp)
    if not times[0] <= times[1] <= times[2]:
        raise ValueError("retry memory observations are out of order")
    before, allocated, released = (record[name]["allocator"] for name in ("before", "allocated", "released"))
    if (before.get("bytes_in_use") != 0 or allocated.get("bytes_in_use", 0) < REGION_BYTES
            or released.get("bytes_in_use") != 0 or released.get("largest_free_block_bytes", 0) < REGION_BYTES):
        raise ValueError("the fresh retry did not allocate and release its large allocator region")


def validate_diagnosis(document, evidence):
    """Require the completed original-protocol diagnostic and its raw correctness/timing evidence."""
    path = Path(__file__).resolve().with_name("post_comparison_validation.py")
    specification = importlib.util.spec_from_file_location("allocator_retry_post_validation", path)
    validation = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(validation)
    result = bound_document(document, evidence)
    if document["sha256"] != DIAGNOSIS_SHA256:
        raise ValueError("this retry requires the exact completed and reviewed memory-region diagnosis")
    measured = result.get("paired_measurement", {})
    if (result.get("kind") != "allocator_region_history_diagnosis" or result.get("completed") is not True
            or result.get("eligible_for_formal_campaign_admission") is not False
            or result.get("hypothesis_count_contribution") != 0 or result.get("source_commit") != SOURCE_COMMIT
            or result.get("runtime_environment") != RUNTIME or measured.get("measurement_valid") is not True
            or measured.get("contaminated") is not False or measured.get("copies") != 2
            or measured.get("hardware_class") != "jaguar03_rtx_a4500" or measured.get("copy_seeds") != [1, 2]
            or len(measured.get("rounds", [])) != 7):
        raise ValueError("the memory-history diagnosis is not a completed original-protocol measurement")
    validation.correctness(measured)
    for index, pair in enumerate(measured["rounds"]):
        if pair.get("order") != (["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]):
            raise ValueError("the diagnosis did not alternate its original paired rounds")
        for arm in ("reference", "candidate"):
            if pair[arm].get("iterations") != 1:
                raise ValueError("the diagnosis changed the original complete-update window")
            validation.measured_round(pair[arm], 2, minimum_updates=1)
    validate_region(result["region_preparation"], result["hardware"]["gpu_uuid"])
    validate_git_receipt(result["helpers"], evidence)
    return result


def validate_approval_document(receipt, evidence):
    """Rebuild an approval before publishing it or trusting an already published copy."""
    case = validate_case(evidence.file(receipt["case"]["path"]))
    if (receipt.get("schema_version") != 1 or receipt.get("kind") != "allocator_region_infrastructure_retry"
            or receipt.get("approved_attempt") != 4 or receipt.get("previous_attempt") != 3
            or receipt.get("preparation") != PREPARATION or receipt.get("runtime_environment") != RUNTIME
            or receipt.get("allocator") != ALLOCATOR or receipt.get("unit_id") != case["unit_id"]
            or receipt.get("campaign_run_dir") != case["campaign_run_dir"]
            or receipt["case"] != file_record(receipt["case"]["path"], evidence)
            or Path(receipt["case"]["path"]) != Path(case["campaign_run_dir"]) / "queue/specifications" / (case["unit_id"] + ".json")):
        raise ValueError("retry authorization changed its exact case, one-shot attempt, runtime or preparation")
    prior = Path(case["campaign_run_dir"]) / "data" / case["unit_id"] / "attempts/attempt-0003"
    if receipt["previous_lease"]["path"] != str(prior / "lease.json") or receipt["previous_failure"]["path"] != str(prior / "failure.json"):
        raise ValueError("retry approval does not retain the exact previous attempt records")
    lease = bound_document(receipt["previous_lease"], evidence)
    failure = bound_document(receipt["previous_failure"], evidence)
    if (lease.get("unit_id") != case["unit_id"] or lease.get("attempt") != 3 or lease.get("state") != "failed"
            or failure.get("case_completed") is not False or failure.get("retryable") is not True
            or failure.get("attempt") != 3 or failure.get("source_commit") != SOURCE_COMMIT
            or "RESOURCE_EXHAUSTED: Out of memory" not in failure.get("error", "")):
        raise ValueError("retry approval does not describe the original held GPU allocation failure")
    validate_diagnosis(receipt["diagnosis"], evidence)
    validate_git_receipt(receipt["helpers"], evidence)
    validate_git_receipt(receipt["numerical_source"], evidence)
    if (receipt["numerical_source"]["source_root"] != case["source_root"]
            or receipt["numerical_source"]["source_commit"] != SOURCE_COMMIT
            or set(receipt["helpers"]["files"]) != {PREFIX + name for name in HELPERS}
            or not {PREFIX + name for name in ("run_case.py", "controller.py", "ideas.py", "paired_benchmark.py", "worker_environment.sh")} <= set(receipt["numerical_source"]["files"])):
        raise ValueError("retry authorization lacks its complete helper or original runner identity")
    if sorted(receipt["numerical_source"]["files"]) != numerical_source_paths(case["source_root"], evidence):
        raise ValueError("retry authorization omitted original numerical or runner source files")
    return receipt, case


def validate_approval(path, evidence=None):
    """Rebuild one immutable attempt-four authorization and all its original inputs."""
    evidence = LocalEvidence() if evidence is None else evidence
    receipt, case = validate_approval_document(evidence.read(path), evidence)
    if Path(path) != Path(case["campaign_run_dir"]) / "queue/infrastructure_retries/approvals" / (case["unit_id"] + ".json"):
        raise ValueError("the retry approval is outside its exact separate queue")
    return receipt, case


def cpu_allocation(environment, roster, available):
    """Assign disjoint eight-CPU subsets from the parent's actual allocated thread affinity."""
    declared = environment.get("SLURM_CPUS_PER_TASK", "")
    binding = environment.get("SLURM_CPU_BIND_TYPE", "")
    available = sorted(available)
    required = 8 * len(roster)
    if (not declared.isdecimal() or int(declared) < required or len(available) < required
            or len(available) > int(declared) or len(set(available)) != len(available)
            or any(type(cpu) is not int or cpu < 0 for cpu in available)
            or "threads" not in binding.split(",")):
        raise ValueError("retry parent requires allocated thread affinity and at least eight CPUs per GPU")
    return {"slurm_cpus_per_task": declared, "slurm_cpu_bind_type": binding,
        "parent_available_cpus": available, "cpus_per_gpu": 8, "thread_environment": THREAD_ENVIRONMENT,
        "gpu_cpu_affinities": {gpu: available[index * 8:(index + 1) * 8] for index, gpu in enumerate(roster)}}


def validate_cpu_allocation(assignment, gpu_uuid):
    """Reject overlapping or substituted CPU masks rather than accepting an arbitrary eight CPUs."""
    recorded = assignment["cpu_allocation"]
    expected = cpu_allocation({"SLURM_CPUS_PER_TASK": recorded["slurm_cpus_per_task"],
        "SLURM_CPU_BIND_TYPE": recorded["slurm_cpu_bind_type"]}, assignment["gpu_uuids"], recorded["parent_available_cpus"])
    if expected != recorded:
        raise ValueError("retry CPU affinity overlaps or differs from its exact per-GPU assignment")
    return recorded["gpu_cpu_affinities"][gpu_uuid]


def validate_assignment(assignment, gpu_uuid):
    """Recheck the saved full physical Slurm roster without requiring a GPU during evidence review."""
    proof, roster = assignment["allocation"], assignment["gpu_uuids"]
    if (assignment.get("hostname") != "jaguar03" or len(roster) != len(set(roster))
            or not 1 <= len(roster) <= 8 or gpu_uuid not in roster
            or assignment.get("expected_gpu_count") != len(roster)
            or not str(proof.get("slurm_job_id", "")).isdigit() or proof.get("hostname") != "jaguar03"
            or type(assignment.get("parent_pid")) is not int or not str(assignment.get("parent_process_start_time", "")).isdigit()):
        raise ValueError("retry assignment does not describe its actual parent and physical GPU roster")
    rows = list(csv.reader(proof["gpu_csv"].splitlines()))
    devices = ET.fromstring(proof["gpu_xml"]).findall("gpu")
    xml = {device.findtext("uuid"): device for device in devices}
    cards = []
    for row in rows:
        if len(row) != 3:
            raise ValueError("retry allocation has a malformed GPU row")
        index, uuid, name = [field.strip() for field in row]
        device = xml.get(uuid)
        minor = device.findtext("minor_number", "") if device is not None else ""
        if (not index.isdigit() or not minor.isdigit() or "A4500" not in name
                or device.findtext("product_name") != name):
            raise ValueError("retry allocation CSV and physical-minor XML disagree")
        cards.append({"visible_index": index, "gpu_uuid": uuid, "physical_minor": minor, "name": name})
    if cards != proof["cards"] or len(cards) != len(roster) or set(xml) != set(roster):
        raise ValueError("retry allocation changed its physical GPU roster")
    minors = {card["physical_minor"]: card["gpu_uuid"] for card in cards}
    indices = {card["visible_index"] for card in cards}
    allocations = proof["slurm_allocations"]
    if not allocations or set(allocations) - {"SLURM_JOB_GPUS", "SLURM_STEP_GPUS"}:
        raise ValueError("retry assignment lacks the original full Slurm allocation fields")
    for value in allocations.values():
        tokens = value.split(",")
        resolved = [minors.get(str(int(token))) if token.isdigit() else token for token in tokens]
        if len(tokens) != len(roster) or set(resolved) != set(roster):
            raise ValueError("a retry child replaced or changed the full Slurm physical allocation")
    visible = proof["cuda_visible_devices_original"].split(",")
    if len(visible) != len(roster) or set(visible) not in (set(roster), set(minors), indices):
        raise ValueError("retry parent visibility differs from the recorded full allocation")
    validate_cpu_allocation(assignment, gpu_uuid)


def validate_execution(result_path, case, lease=None, evidence=None):
    """Bind the untouched original result to this approved process and actual memory preparation."""
    evidence = LocalEvidence() if evidence is None else evidence
    directory = Path(result_path).parent
    sidecar = evidence.read(directory / "infrastructure_retry_execution.json")
    approval_record = sidecar["approval"]
    bound_document(approval_record, evidence)
    approval, approved_case = validate_approval(approval_record["path"], evidence)
    result = evidence.read(result_path)
    saved_lease = evidence.read(directory / "lease.json")
    hardware = evidence.read(directory / "hardware.json")
    expected_runner = str(Path(approval["helpers"]["source_root"]) / PREFIX / "run_infrastructure_retry.py")
    expected_result = {"case_completed": True, "unit_id": case["unit_id"], "idea": case["hypothesis"],
        "specification_digest": case["specification_digest"], "source_commit": SOURCE_COMMIT,
        "hardware_class": "jaguar03_rtx_a4500", "copies": 2, "attempt": 4}
    if (approved_case != case or sidecar.get("completed") is not True or sidecar.get("attempt") != 4
            or result.get("attempt") != 4 or saved_lease.get("attempt") != 4
            or sidecar.get("original_result") != file_record(result_path, evidence)
            or sidecar.get("approval") != saved_lease.get("infrastructure_retry_approval")
            or sidecar.get("lease_token") != saved_lease.get("token")
            or sidecar.get("unit_id") != case["unit_id"] or sidecar.get("helpers") != approval["helpers"]
            or sidecar.get("numerical_source") != approval["numerical_source"]
            or sidecar.get("runner") != expected_runner
            or any(result.get(name) != value for name, value in expected_result.items())
            or Path(result_path) != Path(case["campaign_run_dir"]) / "data" / case["unit_id"] / "attempts/attempt-0004/result.json"):
        raise ValueError("the original retry result lacks its exact approved attempt/helper/lease proof")
    if saved_lease.get("state") not in ("running", "completed") or (lease is None and saved_lease.get("state") != "completed"):
        raise ValueError("the retry evidence is not a completed queue attempt")
    if lease is not None and any(lease.get(name) != saved_lease.get(name) for name in
                                  ("attempt", "token", "runner_pid", "runner_start_time", "infrastructure_retry_approval")):
        raise ValueError("the retry result belongs to another worker lease")
    if (sidecar.get("pid") != hardware.get("pid") or sidecar.get("process_start_time") != hardware.get("process_start_time")
            or sidecar.get("pid") != saved_lease.get("runner_pid")
            or sidecar.get("process_start_time") != saved_lease.get("runner_start_time")
            or sidecar.get("gpu_uuid") != hardware.get("gpu_uuid") or hardware.get("gpu_uuid") != saved_lease.get("gpu_uuid")):
        raise ValueError("the retry preparation and original runner have different process or device identities")
    actual = sidecar.get("runtime_environment", {})
    if (any(actual.get(name) != value for name, value in {**RUNTIME, **ALLOCATOR}.items())
            or actual.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
            or actual.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != ""
            or actual.get("JAXTARI_SKIP_SPRITE_UPDATE") != "1"):
        raise ValueError("the retry changed its admitted precision, allocator or game runtime")
    region = bound_document(sidecar["region_preparation"], evidence)
    validate_region(region, hardware["gpu_uuid"])
    assignment = bound_document(sidecar["assignment"], evidence)
    validate_assignment(assignment, hardware["gpu_uuid"])
    if (assignment["allocation"]["slurm_job_id"] != hardware.get("slurm_job_id")
            or sidecar.get("slurm_allocations") != assignment["allocation"]["slurm_allocations"]
            or sidecar.get("cuda_visible_devices") != hardware["gpu_uuid"]
            or approval_record not in assignment.get("approvals", [])):
        raise ValueError("retry result changed its parent allocation or assigned CUDA UUID")
    expected_cpus = validate_cpu_allocation(assignment, hardware["gpu_uuid"])
    if (sidecar.get("cpu_affinity") != expected_cpus or saved_lease.get("cpu_affinity") != expected_cpus
            or sidecar.get("thread_environment") != THREAD_ENVIRONMENT
            or saved_lease.get("thread_environment") != THREAD_ENVIRONMENT
            or actual.get("OMP_NUM_THREADS") != "8" or actual.get("OPENBLAS_NUM_THREADS") != "1"
            or actual.get("MKL_NUM_THREADS") != "1"):
        raise ValueError("the original numerical runner changed its assigned eight CPUs or thread runtime")
    claim = evidence.read(Path(case["campaign_run_dir"]) / "queue/infrastructure_retries/claims" / (case["unit_id"] + ".json"))
    if (claim.get("approval") != approval_record or any(claim["lease"].get(name) != saved_lease.get(name) for name in
            ("token", "unit_id", "attempt", "worker_pid", "worker_start_time", "gpu_uuid", "slurm_job_id", "assignment",
             "cpu_affinity", "thread_environment"))
            or saved_lease.get("assignment") != sidecar["assignment"]
            or hardware.get("worker_pid") != saved_lease.get("worker_pid")
            or hardware.get("worker_start_time") != saved_lease.get("worker_start_time")):
        raise ValueError("retry execution differs from its immutable one-shot claim")
    start, finish = sidecar.get("started_at_seconds"), sidecar.get("finished_at_seconds")
    if (type(start) not in (int, float) or type(finish) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(finish)
            or not 0 < start <= region["before"]["time_seconds"] <= region["released"]["time_seconds"] < finish):
        raise ValueError("the retry execution time bounds do not contain its memory preparation")
    expected_modules = {name: str(Path(case["source_root"]) / PREFIX / (name + ".py"))
                        for name in ("run_case", "controller", "ideas", "paired_benchmark")}
    if sidecar.get("frozen_modules") != expected_modules:
        raise ValueError("the retry imported a current or shadowed original benchmark module")
    return {"scope": "actual_infrastructure_retry_runtime", "runtime_environment": dict(RUNTIME),
            "actual_runtime_environment": actual, "approval": approval_record,
            "execution_record": file_record(directory / "infrastructure_retry_execution.json", evidence),
            "helper_commit": approval["helpers"]["source_commit"], "memory_preparation_bytes": REGION_BYTES}


def prepare(arguments):
    """Prepare all five immutable approvals only after every source and held failure has validated."""
    run, helper = arguments.run_dir.resolve(strict=True), arguments.helper_root.resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(helper), "rev-parse", "HEAD"], text=True).strip()
    helpers = git_receipt(helper, commit, [PREFIX + name for name in HELPERS])
    evidence = LocalEvidence()
    diagnosis = file_record(arguments.diagnosis.resolve(strict=True), evidence)
    validate_diagnosis(diagnosis, evidence)
    assets = json.loads(arguments.asset_record.read_bytes())
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("retry preparation requires the explicit original asset record")
    approvals = []
    for position in APPROVED:
        matches = list((run / "queue/failed").glob(f"idea-position-{position}__*.json"))
        if len(matches) != 1:
            raise ValueError("each approved case must still be held in failed before preparation")
        failed = matches[0]
        case = validate_case(failed.read_bytes())
        original = run / "queue/specifications" / failed.name
        if original.read_bytes() != failed.read_bytes() or case["campaign_run_dir"] != str(run):
            raise ValueError("the held case differs from its original immutable specification")
        previous = run / "data" / case["unit_id"] / "attempts/attempt-0003"
        current_lease = json.loads((run / "queue/leases" / failed.name).read_bytes())
        if current_lease != json.loads((previous / "lease.json").read_bytes()):
            raise ValueError("the held case lease differs from its latest preserved attempt")
        numerical_paths = numerical_source_paths(case["source_root"], evidence)
        receipt = {"schema_version": 1, "kind": "allocator_region_infrastructure_retry", "unit_id": case["unit_id"],
            "campaign_run_dir": str(run), "approved_attempt": 4, "previous_attempt": 3,
            "case": file_record(original, evidence), "previous_lease": file_record(previous / "lease.json", evidence),
            "previous_failure": file_record(previous / "failure.json", evidence), "diagnosis": diagnosis,
            "helpers": helpers, "numerical_source": git_receipt(case["source_root"], SOURCE_COMMIT, numerical_paths),
            "runtime_environment": RUNTIME, "allocator": ALLOCATOR, "preparation": PREPARATION,
            "asset_record": assets, "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve(strict=True)),
            "python": str(arguments.python.absolute())}
        validate_approval_document(receipt, evidence)
        approvals.append((run / "queue/infrastructure_retries/approvals" / failed.name, receipt))
    # Before: five held cases. After: five approvals, while all five case files stay failed.
    for path, receipt in approvals:
        if path.exists() and path.read_bytes() != json_bytes(receipt):
            raise ValueError("an earlier retry approval differs")
    for path, receipt in approvals:
        write_once(path, json_bytes(receipt))
    return [str(path) for path, _ in approvals]


def main():
    """Create metadata approvals only; this command never requeues, submits or opens a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "helper-root", "diagnosis", "asset-record", "jaxtari-source-root", "python"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
