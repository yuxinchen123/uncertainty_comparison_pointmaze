"""Stage explicit final units and submit one generic UVA job per jaguar03 node.

Login preparation checks existing admission receipts and immutable bytes only.
Workers revalidate all raw gates and completion on their owned compute allocation.
Every continuation uses its unit's original detached training-helper checkout.
"""

import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_queue

PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
TRAIN_HELPERS = ("train_optimized.py", "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py",
                 "h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py",
                 "h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py",
                 "controller.py", "ideas.py",
                 "queue_worker.py", "diagnose_gpu_execution.py")
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED", "REQUEUE_FED", "SIGNALING", "STAGE_OUT"}
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
TARGET_UPDATES = 2_000_000_000 // (128 * 128)


def digest(value):
    """Bind immutable metadata to its complete JSON values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    """Hash checkpoints and source files without loading arrays or retaining large files."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    """Read one complete JSON record without interpreting numerical checkpoint payloads."""
    return json.loads(Path(path).read_text())


def process_birth(pid):
    """Identify an original supervisor or child without relying on a reused PID alone."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def atomic(path, value, *, immutable=False):
    """Publish complete metadata and preserve identical immutable records on retries."""
    path = Path(path)
    contents = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if immutable and path.exists():
        if path.read_bytes() != contents:
            raise ValueError("immutable final-training metadata differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def queue_lock(run):
    """Serialize unit claims, continuation decisions and generic job publication."""
    directory = Path(run) / "final_training_queue"
    directory.mkdir(exist_ok=True)
    with (directory / "metadata.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield directory
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_receipt(receipt):
    """Check exact original bytes named by an already computed validation receipt."""
    if not Path(receipt["path"]).is_absolute() or file_digest(receipt["path"]) != receipt["sha256"]:
        raise ValueError("an original admission/evidence file changed: " + receipt["path"])


def frozen_helpers(root, commit):
    """Pin the original runtime closure required by the committed caller source."""
    root = Path(root).resolve(strict=True)
    top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.run(["git", "-C", str(root), "symbolic-ref", "--quiet", "HEAD"], capture_output=True)
    paths = [PREFIX + name for name in TRAIN_HELPERS]
    # Old committed callers never dispatched the new retry module; retain their original receipt identity.
    callers = []
    for name in ("controller.py", "post_comparison_validation.py"):
        relative = PREFIX + name
        original = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative])
        if (root / relative).read_bytes() != original:
            raise ValueError("a pinned training helper differs from its Git blob")
        callers.append(ast.parse(original))
    if not any(isinstance(node, ast.Constant) and node.value == "pregather_async_retry.py"
               for tree in callers for node in ast.walk(tree)):
        paths.remove(PREFIX + "pregather_async_retry.py")
    h100_declared = any(isinstance(node, ast.Constant) and node.value == "h100_nvl_slurm_queue.py"
                       for tree in callers for node in ast.walk(tree))
    if not h100_declared:
        for name in ("h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py"):
            paths.remove(PREFIX + name)
    qualification_declared = False
    if h100_declared:
        # Parse only the exact already checked Git blobs; a working-tree marker cannot add dependencies.
        for name in ("h100_nvl_slurm_queue.py", "h100_nvl_slurm_evidence.py"):
            relative = PREFIX + name
            original = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative])
            if (root / relative).read_bytes() != original:
                raise ValueError("a pinned H100 helper differs from its Git blob")
            tree = ast.parse(original)
            qualification_declared |= any(isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "H100_NVL_SLURM_QUALIFICATION_MODULE" for target in node.targets)
                and isinstance(node.value, ast.Constant) and node.value.value == "h100_nvl_slurm_qualification.py" for node in tree.body)
    if not qualification_declared:
        for name in ("h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py"):
            paths.remove(PREFIX + name)
    changes = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", *paths], text=True)
    if (Path(top).resolve(strict=True) != root or actual != commit or branch.returncode != 1
            or changes.strip() or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("training helpers must retain their exact clean detached commit on every continuation")
    files = {}
    for relative in paths:
        original = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative])
        if (root / relative).read_bytes() != original:
            raise ValueError("a pinned training helper differs from its Git blob")
        files[relative] = hashlib.sha256(original).hexdigest()
    return {"source_root": str(root), "source_commit": commit, "files": files}


def module(unit, name):
    """Load a reviewed pinned standard-library helper without importing its numerical runtime."""
    path = Path(unit["helpers"]["source_root"]) / PREFIX / (name + ".py")
    specification = importlib.util.spec_from_file_location("final_training_" + name, path)
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


def expected_backend(final):
    """Require matching actual model, memory, driver and complete software evidence at the selected count."""
    context = final["specification"]["contexts"]["jaguar03_rtx_a4500"]
    selected = final["selected"]["jaguar03_rtx_a4500"]["configuration"]
    paths = [context["confirmation_record"]]
    paths += [path for path in context["packing_records"] + context["packing_confirmation_records"]
              if read(path)["configuration"]["learner"]["n_copies"] == selected["learner"]["n_copies"]]
    identities = []
    for path in paths:
        record = read(path)
        hardware = record.get("hardware")
        if not isinstance(hardware, dict) or hardware != read(Path(path).parent / "hardware.json"):
            raise ValueError("selected result hardware differs from its exact original sidecar")
        rows = list(csv.reader(hardware["gpu_query"].splitlines()))
        if len(rows) != 1 or len(rows[0]) != 4 or hardware.get("hostname") != "jaguar03":
            raise ValueError("selected final evidence lacks one actual jaguar03 GPU identity")
        gpu, name, memory, driver = (value.strip() for value in rows[0])
        match = re.fullmatch(r"(\d+) MiB", memory)
        installed = record.get("installed_backend")
        loaded = record.get("loaded_gpu_libraries", {})
        if (not gpu.startswith("GPU-") or hardware.get("gpu_uuid") != gpu or "A4500" not in name or not match or not driver
                or not isinstance(installed, dict) or set(installed) != {"jaxtari_source", "gpu_libraries", "packages"}
                or not installed["packages"] or not installed["gpu_libraries"]
                or installed != read(Path(path).parent / "installed_backend.json")
                or loaded.get("platform") != "gpu" or not loaded.get("platform_version")):
            raise ValueError("final placement needs the actual complete selected hardware/backend evidence")
        identities.append({"gpu_name": name, "total_memory_bytes": int(match.group(1)) * 2 ** 20,
            "driver_version": driver, "platform": loaded["platform"], "platform_version": loaded["platform_version"],
            "installed_backend": installed})
    if len(identities) < 3 or any(identity != identities[0] for identity in identities):
        raise ValueError("selected confirmation and independent packing repetitions used different hardware/backends")
    return identities[0]


def asset_binding(specification, final, source_commit):
    """Match complete manifest metadata to the selected actual confirmation without reading asset files."""
    confirmation = Path(final["specification"]["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"])
    plan_path, ready_path, actual_path = (confirmation.parent / name for name in (
        "measurement_plan.json", "measurement_prepared.json", "asset_receipt.json"))
    plan, actual, result = read(plan_path), read(actual_path), read(confirmation)
    if (read(ready_path) != {"plan_sha256": file_digest(plan_path)}
            or plan.get("kind") != "selected_configuration_measurement" or plan.get("mode") != "confirmation"
            or plan.get("hardware_class") != "jaguar03_rtx_a4500"
            or plan.get("source_commit") != specification["source_commit"]
            or plan.get("source_root") != specification["source_root"]
            or plan.get("configuration") != specification["configuration"]
            or plan.get("helpers") != result.get("helpers") or not plan.get("helpers", {}).get("files")):
        raise ValueError("selected asset proof differs from its immutable final confirmation plan")
    original_input = plan["inputs"]["asset_record"]
    verify_receipt(original_input)
    if read(original_input["path"]) != plan["asset_record"]:
        raise ValueError("selected confirmation asset input changed from its recorded original bytes")
    proofs = [plan_path, ready_path, actual_path, Path(original_input["path"])]
    for relative, expected in plan["helpers"]["files"].items():
        path = confirmation.parent / "code" / Path(relative).name
        if not relative.startswith(PREFIX) or ".." in Path(relative).parts or file_digest(path) != expected:
            raise ValueError("selected asset producer differs from its original raw-bound helper bytes")
        proofs.append(path)
    identities = []
    for record in (plan["asset_record"], specification["asset_record"]):
        if set(record) != {"root", "manifest", "sha256"} or not Path(record["root"]).is_absolute() or not Path(record["manifest"]).is_absolute():
            raise ValueError("final assets require explicit absolute operational roots and hashed manifest paths")
        path = Path(record["manifest"]).resolve(strict=True)
        if path.is_relative_to(Path(record["root"]).resolve()):
            raise ValueError("final asset manifests must remain outside their operational asset roots")
        if file_digest(path) != record["sha256"]:
            raise ValueError("final asset manifest bytes differ from their recorded SHA256")
        manifest = read(path)
        files = manifest.get("files")
        if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
                or manifest.get("source_commit") != source_commit or not isinstance(files, dict) or not files
                or any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                       or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                       for name, value in files.items())):
            raise ValueError("final asset manifest changed its pinned source or complete file-tree identity")
        identities.append({"manifest_sha256": record["sha256"], "source_commit": source_commit,
                           "tree_digest": digest(files), "file_count": len(files)})
        proofs.append(path)
    if (identities[0] != identities[1] or type(actual.get("file_count")) is not int
            or {key: actual.get(key) for key in identities[0]} != identities[0]
            or Path(actual.get("root", "")).resolve() != Path(plan["asset_record"]["root"]).resolve()):
        raise ValueError("final assets differ from the selected actual manifest/source/tree/file-count proof")
    return {"identity": identities[0], "proofs": [{"path": str(path.resolve()), "sha256": file_digest(path)}
                                                   for path in sorted(set(proofs))]}


def prepared_unit(run, proposal):
    """Bind one explicit contiguous seed block to existing compute admission and immutable helpers."""
    identifier = proposal["unit_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or len(identifier.encode()) > 180:
        raise ValueError("final unit names must be descriptive plain path components")
    paths = {key: Path(proposal[key]).resolve(strict=True) for key in ("final_specification", "training_admission_receipt")}
    specification, admitted = (read(paths[key]) for key in ("final_specification", "training_admission_receipt"))
    receipt = admitted.get("validation", {})
    final = read(specification["final_verification"])
    if (receipt.get("kind") != "training_admission" or receipt.get("specification_digest") != digest(specification)
            or receipt.get("source_commit") != specification["source_commit"]
            or receipt.get("configuration_digest") != digest(specification["configuration"])
            or receipt.get("selected_copies") != specification["configuration"]["learner"]["n_copies"]
            or "validation" in specification
            or any(admitted.get(key) != value for key, value in specification.items())
            or admitted.get("hardware_class") != "jaguar03_rtx_a4500"
            or final.get("kind") != "final_verification" or final.get("source_commit") != specification["source_commit"]
            or final.get("specification_digest") != digest(final["specification"])
            or Path(final["specification"]["study_registry"]).resolve() != run / "study_registry.json"
            or admitted["configuration"] != specification["configuration"]
            or admitted["copy_seeds"] != specification["copy_seeds"]
            or admitted["runtime_environment"] != specification["runtime_environment"]):
        raise ValueError("prepare requires this campaign's existing full compute-validated final admission")
    for name in ("final_combination_correctness_passed", "final_confirmation_passed", "h100_packing_complete",
                 "a4500_packing_complete", "eight_gpu_scaling_complete"):
        if final.get(name) is not True:
            raise ValueError("all fifty ideas and all final raw gates must already be validated")
    for item in receipt["admission_records"] + receipt["evidence"]:
        verify_receipt(item)
    seeds = specification["copy_seeds"]
    if not seeds or seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("each explicit final unit must own exactly one contiguous independent seed block")
    helpers = frozen_helpers(proposal["helper_root"], proposal["helper_commit"])
    python = Path(proposal["python"]).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("the explicit final Python environment is not executable")
    unit = {"schema_version": 1, "kind": "final_training_unit", "campaign_run_dir": str(run), "unit_id": identifier,
        "hardware_class": "jaguar03_rtx_a4500",
        **{key: {"path": str(path), "sha256": file_digest(path)} for key, path in paths.items()},
        "numerical_source": {"root": specification["source_root"], "commit": specification["source_commit"]},
        "helpers": helpers, "python": str(python), "copy_seeds": seeds,
        "configuration_digest": digest(specification["configuration"]),
        "runtime_environment_digest": digest(specification["runtime_environment"]),
        "checkpoint_seconds": proposal.get("checkpoint_seconds", 1200), "target_backend": expected_backend(final)}
    runner = module(unit, "train_optimized")
    runner.validate_specification(admitted)
    unit["assets"] = asset_binding(specification, final, runner.JAXTARI_COMMIT)
    if set(unit["target_backend"]["installed_backend"]["packages"]) != set(runner.PACKAGE_NAMES):
        raise ValueError("selected measurements lack the complete actual package-version map")
    if type(unit["checkpoint_seconds"]) is not int or not 0 < unit["checkpoint_seconds"] <= 1200:
        raise ValueError("final checkpoint intervals must be between one and 1200 seconds")
    return unit


def reserve_seed_blocks(queue, units):
    """Reserve explicit global copy identities once across A4500 and eventual H100 units."""
    path = Path(queue) / "seed_manifest.json"
    manifest = read(path) if path.exists() else {"schema_version": 1, "kind": "final_policy_seed_blocks", "units": {}}
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "final_policy_seed_blocks":
        raise ValueError("global final-seed manifest has an unsupported identity")
    combined = dict(manifest["units"])
    for unit in units:
        row = {key: unit[key] for key in ("unit_id", "hardware_class", "copy_seeds", "configuration_digest")}
        seeds = row["copy_seeds"]
        if (row["hardware_class"] not in {"h100", "jaguar03_rtx_a4500"} or not seeds
                or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)
                or seeds != list(range(seeds[0], seeds[0] + len(seeds)))):
            raise ValueError("global final units require one explicit contiguous unsigned seed block")
        if row["unit_id"] in combined and combined[row["unit_id"]] != row:
            raise ValueError("a reserved final seed block cannot change hardware, configuration or seeds")
        combined[row["unit_id"]] = row
    seeds = [seed for row in combined.values() for seed in row["copy_seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("A4500 and H100 final units cannot repeat any independent policy seed")
    atomic(path, {**manifest, "units": combined})
    return combined


def prepare(run, input_path):
    """Register explicit ready units outside their untouched training output directories."""
    run = Path(run).resolve(strict=True)
    if any(not (run / name).is_file() for name in (
            "experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt")):
        raise ValueError("final preparation requires all four original campaign launch records")
    proposals = read(input_path)
    if proposals.get("schema_version") != 1 or not proposals.get("units"):
        raise ValueError("supply schema_version1 and explicit final-unit proposals")
    units = [prepared_unit(run, proposal) for proposal in proposals["units"]]
    with queue_lock(run) as queue:
        old = {path.stem: read(path) for path in (queue / "units").glob("*.json")}
        # A prepared monitor policy freezes every A4500 unit available to generic workers.
        policy_path = queue / "monitor_policy.json"
        if policy_path.exists():
            policy = read(policy_path)
            expected = {name: {"path": str(queue / "units" / (name + ".json")),
                               "sha256": file_digest(queue / "units" / (name + ".json"))}
                        for name, unit in old.items() if unit["hardware_class"] == "jaguar03_rtx_a4500"}
            if (policy.get("kind") != "uva_final_training_monitor_policy" or policy.get("units") != expected
                    or any(old.get(unit["unit_id"]) != unit for unit in units
                           if unit["hardware_class"] == "jaguar03_rtx_a4500")):
                raise ValueError("a prepared final monitor policy permits only identical existing A4500 units")
        for unit in units:
            if unit["unit_id"] in old and old[unit["unit_id"]] != unit:
                raise ValueError("a final unit cannot change its source, helpers, configuration or seeds")
            old[unit["unit_id"]] = unit
        seeds = [seed for unit in old.values() for seed in unit["copy_seeds"]]
        if len(seeds) != len(set(seeds)) or len({unit["unit_id"] for unit in units}) != len(units):
            raise ValueError("explicit final units cannot duplicate names or policy seeds")
        reserve_seed_blocks(queue, units)
        for unit in units:
            atomic(queue / "units" / (unit["unit_id"] + ".json"), unit, immutable=True)
            status = queue / "status" / (unit["unit_id"] + ".json")
            if not status.exists():
                atomic(status, {"state": "ready", "unit_id": unit["unit_id"], "last_job_id": None})
    return units


def job_states(run, identifiers):
    """Query only exact final-worker IDs contained in this campaign's UVA ledger."""
    identifiers = sorted(set(identifiers))
    if not identifiers:
        return {}
    owned = set((Path(run) / "slurm/submitted_jobids_uva.txt").read_text().splitlines())
    if any(not str(identifier).isdecimal() or identifier not in owned for identifier in identifiers):
        raise ValueError("refusing a scheduler query outside this campaign's exact UVA job ledger")
    text = subprocess.check_output(["sacct", "-X", "-j", ",".join(identifiers), "--format=JobIDRaw,State", "-P", "-n"], text=True)
    states = {}
    for line in text.splitlines():
        job, state = line.split("|", 1)
        state = state.rstrip("|").split()[0].rstrip("+")
        if job not in identifiers or job in states or state not in ACTIVE | TERMINAL:
            raise ValueError("scheduler returned duplicate, foreign or unknown final-job state")
        states[job] = state
    if set(states) != set(identifiers):
        raise ValueError("previous exact final-job state is unknown; continuation requires diagnosis")
    return states


def verify_unit(unit):
    """Refuse changed input bytes or helper Git identities on every claim and continuation."""
    for key in ("final_specification", "training_admission_receipt"):
        verify_receipt(unit[key])
    for receipt in unit["assets"]["proofs"]:
        verify_receipt(receipt)
    admission = read(unit["training_admission_receipt"]["path"])["validation"]
    for receipt in admission["admission_records"] + admission["evidence"]:
        verify_receipt(receipt)
    manifest = read(Path(unit["campaign_run_dir"]) / "final_training_queue/seed_manifest.json")
    expected = {key: unit[key] for key in ("unit_id", "hardware_class", "copy_seeds", "configuration_digest")}
    seeds = [seed for row in manifest["units"].values() for seed in row["copy_seeds"]]
    if manifest["units"].get(unit["unit_id"]) != expected or len(seeds) != len(set(seeds)):
        raise ValueError("final unit no longer owns its unique reserved global seed block")
    if frozen_helpers(unit["helpers"]["source_root"], unit["helpers"]["source_commit"]) != unit["helpers"]:
        raise ValueError("the original final-unit helper identity changed")


def checkpoint_record(run, unit, *, require_terminal_result=True):
    """Read actual result, identity and checkpoint sidecar without unpickling any array."""
    directory = Path(run) / "data" / unit["unit_id"]
    progress = read(directory / "progress.json")
    attempt = progress.get("attempt")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("final progress must identify its exact current training attempt")
    if (directory / "attempts" / f"attempt-{attempt:04d}" / "failure.json").exists():
        raise ValueError("the current final attempt failed and requires diagnosis before continuation")
    if (directory / "failure.json").exists():
        failure = read(directory / "failure.json")
        if (type(failure.get("attempt")) is not int or type(progress.get("attempt")) is not int
                or failure["attempt"] >= progress["attempt"] or progress.get("state") == "failed"):
            raise ValueError("a failed final unit requires diagnosis before any continuation")
    if progress.get("state") == "failed":
        raise ValueError("the current final attempt failed and requires diagnosis before continuation")
    identity = read(directory / "training_identity.json")
    checkpoint = read(directory / "checkpoints/latest.metadata.json")
    expected = {"source_root": unit["numerical_source"]["root"], "source_commit": unit["numerical_source"]["commit"],
                "helpers": unit["helpers"], "copy_seeds": unit["copy_seeds"], "hardware_class": "jaguar03_rtx_a4500"}
    if (any(identity.get(key) != value for key, value in expected.items())
            or digest(identity["configuration"]) != unit["configuration_digest"]
            or checkpoint.get("identity_digest") != digest(identity)
            or checkpoint.get("configuration_digest") != unit["configuration_digest"]
            or checkpoint.get("copy_seeds") != unit["copy_seeds"]
            or checkpoint.get("kind") != "optimized_training_checkpoint"
            or checkpoint.get("source_commit") != unit["numerical_source"]["commit"]
            or checkpoint.get("target_updates") != TARGET_UPDATES):
        raise ValueError("actual final identity/checkpoint metadata is incomplete or inconsistent")
    path = directory / "checkpoints/latest.pkl"
    if checkpoint.get("checkpoint_path") != str(path.resolve()) or file_digest(path) != checkpoint.get("checkpoint_sha256"):
        raise ValueError("the recorded complete checkpoint bytes changed")
    iteration = checkpoint.get("iteration")
    if (type(iteration) is not int or not 0 <= iteration <= TARGET_UPDATES
            or checkpoint.get("steps_per_copy") != iteration * 128 * 128):
        raise ValueError("checkpoint update count is outside the original 2B schedule")
    expected_backend = unit["target_backend"]
    backend = identity["backend"]
    if (backend.get("driver_version") != expected_backend["driver_version"]
            or backend.get("packages") != expected_backend["installed_backend"]["packages"]
            or backend.get("installed_gpu_libraries") != expected_backend["installed_backend"]["gpu_libraries"]
            or backend.get("jaxtari_source") != expected_backend["installed_backend"]["jaxtari_source"]
            or digest({name: identity["runtime_environment"].get(name) for name in (
                "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS")}) != unit["runtime_environment_digest"]):
        raise ValueError("checkpoint backend/runtime differs from the actual selected final evidence")
    # An abrupt scheduler stop has no graceful result; resume its verified state without claiming completion.
    if not require_terminal_result:
        return None, checkpoint
    result = read(directory / "result.json")
    if (result.get("identity_digest") != digest(identity) or result.get("copy_seeds") != unit["copy_seeds"]
            or result.get("update") != iteration or result.get("target_updates") != TARGET_UPDATES
            or result.get("attempt") != attempt or result.get("requested_steps_per_copy") != 2_000_000_000
            or result.get("copies") != len(unit["copy_seeds"]) or result.get("steps_per_copy") != iteration * 128 * 128
            or progress.get("state") not in {"checkpointed", "completed"}):
        raise ValueError("actual final result/progress/checkpoint identity is incomplete or inconsistent")
    return result, checkpoint


def refresh(run, queue, states):
    """Release only terminal checkpointed attempts; keep failures or unproved completion held."""
    for path in (queue / "status").glob("*.json"):
        status = read(path)
        unit = read(queue / "units" / path.name)
        if unit["hardware_class"] != "jaguar03_rtx_a4500":
            continue
        if status["state"] in {"ready", "held"}:
            continue
        job = status.get("last_job_id")
        if status.get("last_job_cluster") != "uva":
            raise ValueError("A4500 continuation lacks its exact original UVA cluster identity")
        if job not in states:
            raise ValueError("final-unit previous job has no exact owned scheduler state")
        if states[job] in ACTIVE:
            continue
        try:
            verify_unit(unit)
            periodic = status["state"] == "running" and states[job] in {"TIMEOUT", "PREEMPTED", "NODE_FAIL"}
            result, checkpoint = checkpoint_record(run, unit, require_terminal_result=not periodic)
            if periodic:
                status.update(state="ready", resume_checkpoint_sha256=checkpoint["checkpoint_sha256"],
                              recovery_scope="verified_periodic_checkpoint_after_" + states[job])
                atomic(path, status)
                continue
            hardware = read(Path(run) / "data" / unit["unit_id"] / "hardware.json")
            if hardware.get("slurm_job_id") != job:
                raise ValueError("latest final-unit artifacts came from a different job")
            if result.get("completed") is True:
                proof = read(queue / "completion" / path.name)
                if (proof.get("kind") != "training_completion" or proof.get("completed") is not True
                        or proof.get("source_commit") != unit["numerical_source"]["commit"]
                        or proof.get("configuration_digest") != unit["configuration_digest"]
                        or proof.get("copy_seeds") != unit["copy_seeds"] or proof.get("updates") != TARGET_UPDATES
                        or proof.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]):
                    raise ValueError("completion needs the allocated worker's full raw completion validation")
                for receipt in proof["evidence"]:
                    verify_receipt(receipt)
                status.update(state="completed")
            elif result.get("stop_reason") in {"maximum_seconds", "SIGUSR1", "SIGTERM", "SIGINT"}:
                status.update(state="ready", resume_checkpoint_sha256=checkpoint["checkpoint_sha256"])
            else:
                raise ValueError("unfinished final training did not checkpoint at an authorized stop boundary")
        except (ValueError, KeyError, TypeError, OSError) as error:
            status.update(state="held", reason=f"{type(error).__name__}: {error}")
        atomic(path, status)


def status(run):
    """Refresh only owned exact job records and expose continuation/diagnosis states without timing analysis."""
    run = Path(run).resolve(strict=True)
    with queue_lock(run) as queue:
        jobs = [read(path)["job_id"] for path in (queue / "jobs").glob("uva_job-*.json")]
        states = job_states(run, jobs)
        refresh(run, queue, states)
        units = [read(path) for path in sorted((queue / "status").glob("*.json"))
                 if read(queue / "units" / path.name)["hardware_class"] == "jaguar03_rtx_a4500"]
        return {"cluster": "uva", "jobs": states, "units": units,
                "counts": {name: sum(unit["state"] == name for unit in units)
                           for name in ("ready", "running", "checkpointed", "completed", "held")}}


def tool_files():
    """Freeze only committed metadata launcher/resolver bytes separately from checkpoint helper identity."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    result = {}
    for name in ("launch_final_training.py", "launch_queue.py"):
        contents = (directory / name).read_bytes()
        original = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + PREFIX + name])
        if contents != original:
            raise ValueError("commit and push metadata launchers before any real submission")
        result[name] = contents
    return result, commit


def record_submission(run, job, commit, script):
    """Append the exact final-job ID immediately to the common UVA ledger and typed submission history."""
    slurm = Path(run) / "slurm"
    slurm.mkdir(exist_ok=True)
    with (slurm / "submissions.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        for path, line in ((slurm / "submitted_jobids_uva.txt", job + "\n"),
                           (slurm / "submissions.tsv", "\t".join((job, "final-training-worker", commit, str(script))) + "\n")):
            with path.open("a") as ledger:
                ledger.write(line)
                ledger.flush()
                os.fsync(ledger.fileno())


def submit(run, *, gpus, walltime_seconds, host_memory_gib_per_gpu, cpus_per_gpu=8, checkpoint_reserve_seconds=1800):
    """Submit one generic node job, retaining every exact ID before releasing any worker."""
    run = Path(run).resolve(strict=True)
    if (type(gpus) is not int or not 1 <= gpus <= 8 or type(walltime_seconds) is not int
            or walltime_seconds <= 2 * checkpoint_reserve_seconds or checkpoint_reserve_seconds < 1500
            or type(host_memory_gib_per_gpu) is not int or host_memory_gib_per_gpu < 1
            or type(cpus_per_gpu) is not int or cpus_per_gpu < 1):
        raise ValueError("use 1..8 GPUs, explicit host resources and walltime exceeding two checkpoint reserves")
    with queue_lock(run) as queue:
        if any(read(path).get("state") == "submitting" and read(path).get("cluster") == "uva"
               for path in (queue / "submissions").glob("*.json")):
            raise ValueError("an unresolved submission requires exact-ID diagnosis before another sbatch")
        jobs = [read(path)["job_id"] for path in (queue / "jobs").glob("uva_job-*.json")]
        states = job_states(run, jobs)
        if any(state in ACTIVE for state in states.values()):
            raise ValueError("this jaguar03 final group already has an active or pending generic job")
        refresh(run, queue, states)
        ready = [read(path) for path in (queue / "status").glob("*.json") if read(path)["state"] == "ready"]
        if not ready:
            raise ValueError("no explicit final unit is ready; held units require diagnosis")
        units = [read(queue / "units" / (row["unit_id"] + ".json")) for row in ready]
        units = [unit for unit in units if unit["hardware_class"] == "jaguar03_rtx_a4500"]
        for unit in units:
            verify_unit(unit)
        if len(units) < gpus or len({digest(unit["target_backend"]) for unit in units}) != 1:
            raise ValueError("requested GPUs need enough ready units with one exact selected A4500 backend")
        files, commit = tool_files()
        identifier = "final_training_jaguar03_gpus-" + str(gpus) + "_" + uuid.uuid4().hex
        directory = run / "slurm/final_training" / identifier
        directory.mkdir(parents=True)
        for name, contents in files.items():
            (directory / name).write_bytes(contents)
        plan = {"schema_version": 1, "campaign_run_dir": str(run), "gpus": gpus,
            "walltime_seconds": walltime_seconds, "checkpoint_reserve_seconds": checkpoint_reserve_seconds,
            "target_backend": units[0]["target_backend"], "cpus_per_gpu": cpus_per_gpu,
            "metadata_files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}}
        atomic(directory / "job_plan.json", plan, immutable=True)
        script = directory / "submit.sh"
        script.write_text("#!/bin/bash\nset -euo pipefail\nexport PYTHONNOUSERSITE=1\nexport PYTHONUNBUFFERED=1\nexec /usr/bin/python3 " + shlex.join([
            str(directory / "launch_final_training.py"), "job", "--plan", str(directory / "job_plan.json")]) + "\n")
        intent = queue / "submissions" / (identifier + ".json")
        atomic(intent, {"state": "submitting", "cluster": "uva", "script": str(script)})
        arguments = ["sbatch", "--parsable", "--partition=gpu", "--nodelist=jaguar03", "--nodes=1", "--ntasks=1",
            "--gres=gpu:nvidia_rtx_a4500:" + str(gpus), "--cpus-per-task=" + str(gpus * cpus_per_gpu),
            "--mem=" + str(gpus * host_memory_gib_per_gpu) + "G", "--time=" + str(math.ceil(walltime_seconds / 60)),
            "--signal=B:USR1@" + str(checkpoint_reserve_seconds), "--job-name=atari_final_training",
            "--output=" + str(directory / "job_%j.log")]
        reservation = launch_queue.current_reservation()
        if reservation:
            arguments += ["--reservation=" + reservation, "--qos=csresnolim"]
        result = subprocess.run([*arguments, str(script)], capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(";", 1)[0]
        if not job.isdecimal():
            raise ValueError("sbatch did not return an exact job ID; leave the submission held for diagnosis")
        # Capture the exact ID immediately; job workers wait for the final job record below.
        record_submission(run, job, commit, script)
        record = {"cluster": "uva", "job_id": job, "plan": str(directory / "job_plan.json"), "plan_sha256": file_digest(directory / "job_plan.json"),
                  "script": str(script), "gpus": gpus}
        atomic(queue / "jobs" / ("uva_job-" + job + ".json"), record, immutable=True)
        atomic(intent, {**record, "state": "submitted"})
    return record


def owned_compute(run):
    """Require an exact UVA ledger job and real compute cgroup before raw validation or GPU work."""
    job = os.environ.get("SLURM_JOB_ID", "")
    owned = set((Path(run) / "slurm/submitted_jobids_uva.txt").read_text().splitlines())
    if (not job.isdecimal() or job not in owned or socket.gethostname().split(".")[0] != "jaguar03"
            or not re.search(r"(?:^|/)job_" + re.escape(job) + r"(?:/|$)", Path("/proc/self/cgroup").read_text(), re.MULTILINE)):
        raise ValueError("final GPU work and raw validation require this campaign's owned jaguar03 UVA allocation")
    return job


def allocated_gpus(count):
    """Cross-check every physical UUID against visible rows, device minors and the complete Slurm allocation."""
    if count == 1:
        return [launch_queue.allocated_gpu_uuid()]
    rows = list(csv.reader(subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True).splitlines()))
    devices = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True)).findall("gpu")
    if len(rows) != count or len(devices) != count:
        raise ValueError("the visible physical GPU set does not equal the allocated group size")
    if any(len(row) != 3 for row in rows):
        raise ValueError("allocated GPU query has malformed physical identity rows")
    minors = {gpu.findtext("uuid"): gpu.findtext("minor_number") for gpu in devices}
    uuids = [row[1].strip() for row in rows]
    if (len(set(uuids)) != count or set(uuids) != set(minors) or any(len(row) != 3 or "A4500" not in row[2] for row in rows)
            or len(set(minors.values())) != count or any(not minor or not minor.isdecimal() for minor in minors.values())):
        raise ValueError("the allocated A4500 UUID/minor set is ambiguous")
    allowed = [set(uuids), set(minors.values())]
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if os.environ.get(name) and set(os.environ[name].split(",")) not in allowed:
            raise ValueError("Slurm physical allocation differs from the visible A4500 group")
    if not os.environ.get("SLURM_JOB_GPUS") or set(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) not in [
            *allowed, {row[0].strip() for row in rows}]:
        raise ValueError("CUDA visibility does not identify the exact allocated GPU group")
    return uuids


def end_time(job):
    """Use the actual scheduler end time so checkpoint reserve includes all startup and validation work."""
    if os.environ.get("SLURM_JOB_END_TIME", "").isdecimal():
        return float(os.environ["SLURM_JOB_END_TIME"])
    text = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, env={**os.environ, "TZ": "UTC"})
    fields = dict(token.split("=", 1) for token in text.split() if "=" in token)
    return datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=timezone.utc).timestamp()


def probe(unit, gpu):
    """Observe the selected physical card and software backend inside the pinned Python, without model arrays."""
    helper = Path(unit["helpers"]["source_root"]) / PREFIX / "train_optimized.py"
    code = '''import importlib.util,json,sys
from importlib import metadata
spec=importlib.util.spec_from_file_location("pinned_training",sys.argv[1]); training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
unit=json.load(open(sys.argv[2])); config=json.load(open(unit["final_specification"]["path"]))
training.enforce_runtime(config["runtime_environment"],__import__("os").environ)
installed={"jaxtari_source":training.jaxtari_python_receipt(config.get("jaxtari_source_root")),"gpu_libraries":training.installed_gpu_library_receipts(),"packages":{name:metadata.version(name) for name in training.PACKAGE_NAMES}}
import jax
devices=jax.devices()
assert len(devices)==1 and devices[0].platform=="gpu"
print(json.dumps({"device_kind":devices[0].device_kind,"platform":devices[0].client.platform,"platform_version":devices[0].client.platform_version,"installed_backend":installed}))
'''
    descriptor = Path(unit["campaign_run_dir"]) / "final_training_queue/units" / (unit["unit_id"] + ".json")
    observed = json.loads(subprocess.check_output([unit["python"], "-c", code, str(helper), str(descriptor)], text=True).splitlines()[-1])
    fields = next(csv.reader(subprocess.check_output(["nvidia-smi", "-i", gpu,
        "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True).splitlines()))
    actual, name, memory, driver = (value.strip() for value in fields)
    expected = unit["target_backend"]
    if (actual != gpu or name != expected["gpu_name"] or int(memory) * 2 ** 20 != expected["total_memory_bytes"]
            or driver != expected["driver_version"] or "A4500" not in observed["device_kind"]
            or any(observed[key] != expected[key] for key in ("platform", "platform_version", "installed_backend"))):
        raise ValueError("actual final GPU model/memory/driver/backend differs from its selected measurements")


def cleanup_children(children, deadline, *, process_groups=False):
    """Stop and reap only original Popen children, allowing checkpoint time before a last-resort stop."""
    active = [child for child in children if child.poll() is None]
    for child in active:
        child.send_signal(signal.SIGTERM)
    for child in active:
        try:
            child.wait(timeout=max(1., min(1200., deadline - time.time() - 30.)))
        except subprocess.TimeoutExpired:
            # Popen retains this exact unreaped child; never signal a PID recovered from an external record.
            if process_groups:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait(timeout=30)


def worker(plan_path, gpu, deadline):
    """Claim generic ready units and checkpoint before the allocation expires."""
    plan, child, stop = read(plan_path), None, {"requested": False}
    run = Path(plan["campaign_run_dir"])
    job = owned_compute(run)
    allocation = read(run / "final_training_queue/allocations" / ("uva_job-" + job + ".json"))
    if (gpu not in allocation["gpu_uuids"] or deadline != allocation["deadline"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or not math.isfinite(deadline)):
        raise ValueError("final worker differs from this job's exact physical allocation and deadline")
    def request_stop(signum, frame):
        """Forward a graceful checkpoint request only to this worker's exact numerical child."""
        stop["requested"] = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    with (Path("/tmp") / ("sl5nw_atari_pixel_benchmark_" + gpu + ".lock")).open("a+") as card:
        fcntl.flock(card.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stop["requested"] and deadline - time.time() > plan["checkpoint_reserve_seconds"]:
            with queue_lock(run) as queue:
                candidates = [path for path in sorted((queue / "status").glob("*.json"))
                              if read(path)["state"] == "ready"
                              and read(queue / "units" / path.name)["hardware_class"] == "jaguar03_rtx_a4500"]
                if not candidates:
                    return
                status_path = candidates[0]
                unit = read(queue / "units" / status_path.name)
                if unit["target_backend"] != plan["target_backend"]:
                    raise ValueError("ready final unit does not match this generic job's proved backend")
                status = read(status_path)
                unstarted = dict(status)
                if status.get("last_job_id") and status.get("last_job_cluster") != "uva":
                    raise ValueError("previous final-unit allocation was not this exact UVA cluster")
                if status.get("last_job_id") and job_states(run, [status["last_job_id"]])[status["last_job_id"]] not in TERMINAL:
                    raise ValueError("previous final unit job is not terminal")
                status.update(state="running", last_job_id=job, last_job_cluster="uva", gpu_uuid=gpu, worker_pid=os.getpid(),
                              worker_start_time=process_birth(os.getpid()))
                atomic(status_path, status)
            try:
                verify_unit(unit)
                validator = module(unit, "post_comparison_validation")
                admitted = validator.validate_training_admission(unit["final_specification"]["path"])
                if admitted != read(unit["training_admission_receipt"]["path"]):
                    raise ValueError("all-fifty raw admission changed since final-unit preparation")
                os.environ.update(admitted["runtime_environment"], CUDA_VISIBLE_DEVICES=gpu,
                    PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1",
                    JAX_ENABLE_COMPILATION_CACHE="false", JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.85",
                    JAXTARI_SKIP_SPRITE_UPDATE="1", XDG_DATA_HOME=admitted["asset_record"]["root"],
                    OMP_NUM_THREADS=str(plan["cpus_per_gpu"]), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                    LD_LIBRARY_PATH="/usr/lib/nvidia:" + os.environ.get("LD_LIBRARY_PATH", ""))
                processes = subprocess.check_output(["nvidia-smi", "-i", gpu,
                    "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                if processes:
                    raise ValueError("another numerical process already occupies this allocated physical GPU")
                probe(unit, gpu)
                maximum = deadline - time.time() - plan["checkpoint_reserve_seconds"]
                if stop["requested"]:
                    with queue_lock(run):
                        atomic(status_path, unstarted)
                    return
                if maximum <= 0:
                    raise ValueError("allocation has insufficient remaining checkpoint reserve")
                runner = Path(unit["helpers"]["source_root"]) / PREFIX / "train_optimized.py"
                log = queue / "logs" / (unit["unit_id"] + "_job-" + job + ".log")
                log.parent.mkdir(exist_ok=True)
                with log.open("x") as stream:
                    child = subprocess.Popen([unit["python"], str(runner), "--final-spec", unit["final_specification"]["path"],
                        "--run-dir", str(run), "--unit-id", unit["unit_id"], "--checkpoint-seconds", str(unit["checkpoint_seconds"]),
                        "--maximum-seconds", str(maximum)], stdout=stream, stderr=subprocess.STDOUT)
                    status.update(child_pid=child.pid, child_start_time=process_birth(child.pid))
                    with queue_lock(run):
                        atomic(status_path, status)
                    code = child.wait()
                child = None
                if code:
                    raise ValueError("numerical final training exited unsuccessfully and requires diagnosis")
                result, checkpoint = checkpoint_record(run, unit)
                hardware = read(run / "data" / unit["unit_id"] / "hardware.json")
                if (hardware.get("slurm_job_id") != job or hardware.get("pid") != status["child_pid"]
                        or hardware.get("cuda_visible_devices") != gpu):
                    raise ValueError("actual final artifacts do not identify this job's numerical child and physical GPU")
                if result.get("completed") is True:
                    completion = validator.validate_training_completion(unit["final_specification"]["path"], run / "data" / unit["unit_id"])
                    atomic(queue / "completion" / status_path.name, completion, immutable=True)
                    status.update(state="completed")
                else:
                    if result.get("stop_reason") not in {"maximum_seconds", "SIGUSR1", "SIGTERM", "SIGINT"}:
                        raise ValueError("final runner did not stop at a checkpointed continuation boundary")
                    status.update(state="checkpointed", checkpoint_sha256=checkpoint["checkpoint_sha256"])
                with queue_lock(run):
                    atomic(status_path, status)
            except BaseException as error:
                if child is not None:
                    cleanup_children([child], deadline)
                    child = None
                with queue_lock(run):
                    atomic(status_path, {**status, "state": "held", "reason": f"{type(error).__name__}: {error}"})
                raise


def run_job(plan_path):
    """Start one subprocess per actually allocated GPU under one recorded node job."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    run = Path(plan["campaign_run_dir"])
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal():
        raise ValueError("generic final workers require an exact Slurm job ID")
    record = run / "final_training_queue/jobs" / ("uva_job-" + job + ".json")
    until = time.monotonic() + 120
    while not record.exists() and time.monotonic() < until:
        time.sleep(.1)
    if not record.exists() or read(record)["plan"] != str(plan_path):
        raise ValueError("generic final job was not durably recorded before release")
    owned_compute(run)
    if file_digest(plan_path) != read(record)["plan_sha256"]:
        raise ValueError("generic final job plan changed after its durable release")
    for name, expected in plan["metadata_files"].items():
        if file_digest(plan_path.parent / name) != expected:
            raise ValueError("the frozen final worker metadata helper changed")
    gpus = allocated_gpus(plan["gpus"])
    deadline = end_time(job)
    atomic(run / "final_training_queue/allocations" / ("uva_job-" + job + ".json"),
           {"job_id": job, "gpu_uuids": gpus, "deadline": deadline}, immutable=True)
    children = []
    def stop_children(signum, frame):
        """Request checkpointing only from this job's original worker subprocesses."""
        for child in children:
            if child.poll() is None:
                child.send_signal(signum)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_children)
    try:
        for gpu in gpus:
            children.append(subprocess.Popen([sys.executable, str(plan_path.parent / "launch_final_training.py"),
                "worker", "--plan", str(plan_path), "--gpu-uuid", gpu, "--deadline", str(deadline)],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, start_new_session=True))
        codes = [child.wait() for child in children]
    except BaseException:
        cleanup_children(children, deadline, process_groups=True)
        raise
    if any(codes):
        raise RuntimeError("a final GPU worker failed; held unit records require diagnosis")


def main():
    """Separate metadata staging/submission from compute-only worker execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--units", type=Path, required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--run-dir", type=Path, required=True)
    submit_parser.add_argument("--gpus", type=int, required=True)
    submit_parser.add_argument("--walltime-seconds", type=int, required=True)
    submit_parser.add_argument("--host-memory-gib-per-gpu", type=int, required=True)
    submit_parser.add_argument("--cpus-per-gpu", type=int, default=8)
    submit_parser.add_argument("--checkpoint-reserve-seconds", type=int, default=1800)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-dir", type=Path, required=True)
    for name in ("job", "worker"):
        execution = commands.add_parser(name)
        execution.add_argument("--plan", type=Path, required=True)
        if name == "worker":
            execution.add_argument("--gpu-uuid", required=True)
            execution.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.run_dir, args.units), indent=2))
    elif args.command == "submit":
        print(json.dumps(submit(args.run_dir, **{key: value for key, value in vars(args).items()
                                               if key not in {"command", "run_dir"}}), indent=2))
    elif args.command == "status":
        print(json.dumps(status(args.run_dir), indent=2))
    elif args.command == "job":
        run_job(args.plan)
    else:
        worker(args.plan, args.gpu_uuid, args.deadline)


if __name__ == "__main__":
    main()
