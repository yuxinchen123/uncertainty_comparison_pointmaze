"""Verify deployed training completion from rebuilt admission and raw checkpoint metadata.

This validates the target-server copy count separately from the original NVL
training admission. It never imports arrays, unpickles a checkpoint, walks the
game asset tree, starts training, or submits work.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

import deployed_training_admission as admission
import portable_evidence
import train_deployed as runner
import train_optimized as training
import validate_training_deployment as deployment


def same(value, expected):
    """Compare canonical JSON identities without equating a Boolean with an integer."""
    return training.canonical_digest(value) == training.canonical_digest(expected)


def checkpoint_identity(admitted, receipt, identity, evidence):
    """Bind every numerical identity field and the deployed runner's exact committed helper closure."""
    expected_keys = {"schema_version", "source_root", "source_commit", "hardware_class", "configuration", "copy_seeds",
                     "runtime_environment", "backend", "helpers", "assets", "deployment"}
    expected = {key: admitted[key] for key in ("source_root", "source_commit", "hardware_class", "configuration", "copy_seeds")}
    if (set(identity) != expected_keys or type(identity["schema_version"]) is not int or identity["schema_version"] != 1
            or any(not same(identity.get(key), value) for key, value in expected.items())):
        raise ValueError("completed deployed identity differs from its exact admitted source/configuration/seeds")
    actual_runtime = deployment.actual_runtime(identity["runtime_environment"], admitted["runtime_environment"])
    if actual_runtime != receipt["actual_runtime_environment"] or not same(actual_runtime, identity["runtime_environment"]):
        raise ValueError("completed deployed identity used another actual compiler/runtime environment")
    expected_assets = {"root": admitted["asset_record"]["root"], **receipt["assets"]}
    expected_deployment = {"deployment_digest": admitted["validation"]["deployment_digest"],
        "specification_digest": admitted["validation"]["specification_digest"], "target": receipt["target"],
        "backend_digest": training.canonical_digest(receipt["backend"]), "assets": receipt["assets"]}
    backend = identity["backend"]
    if (not same(identity["assets"], expected_assets) or not same(identity["deployment"], expected_deployment)
            or set(backend) != {"device_kind", "driver_version", "packages", "jaxtari_source", "installed_gpu_libraries"}
            or not isinstance(backend["device_kind"], str) or not backend["device_kind"]
            or backend["driver_version"] != receipt["target"]["driver_version"]
            or not same(backend["packages"], receipt["backend"]["packages"])
            or not same(backend["installed_gpu_libraries"], receipt["backend"]["gpu_libraries"])
            or not same(backend["jaxtari_source"], {**receipt["backend"]["jaxtari_source"],
                                                   "source_root": admitted["jaxtari_source_root"]})):
        raise ValueError("completed deployed identity changed its deployment/backend/assets/public-source proof")
    required = {deployment.PREFIX + name for name in runner.HELPERS}
    deployment.committed_files(identity["helpers"], required, evidence)
    return training.canonical_digest(identity)


def completed_schedule(admitted, identity_digest, result, checkpoint):
    """Require exactly the full original update schedule and consistent report/checkpoint counters."""
    selected = admitted["configuration"]
    copies = selected["learner"]["n_copies"]
    target = 2_000_000_000 // (128 * 128)
    steps = target * 128 * 128
    execution = selected["execution"]
    pending = 0 if execution["report_readout"] == "none" else target % execution["report_every_iterations"]
    last_report = 0 if execution["report_readout"] == "none" else target - pending
    expected_result = {"completed": True, "update": target, "target_updates": target,
        "requested_steps_per_copy": 2_000_000_000, "steps_per_copy": steps, "copy_seeds": admitted["copy_seeds"],
        "copies": copies, "identity_digest": identity_digest, "pending_report_updates": pending,
        "last_report_iteration": last_report, "stop_reason": "completed"}
    if any(not same(result.get(name), value) for name, value in expected_result.items()):
        raise ValueError("deployed training has not completed the exact original per-policy update/report schedule")
    expected_checkpoint = {"schema_version": 1, "kind": "optimized_training_checkpoint", "identity_digest": identity_digest,
        "configuration_digest": training.canonical_digest(selected), "source_commit": admitted["source_commit"],
        "iteration": target, "target_updates": target, "steps_per_copy": steps, "copy_seeds": admitted["copy_seeds"],
        "pending_report_updates": pending}
    if any(not same(checkpoint.get(name), value) for name, value in expected_checkpoint.items()):
        raise ValueError("deployed checkpoint metadata does not describe the completed learner state")
    for name in ("process_elapsed_seconds", "recorded_elapsed_seconds_total"):
        admission.validation.positive(result.get(name), "completed training " + name)
    admission.validation.positive(checkpoint.get("elapsed_seconds"), "completed checkpoint runtime")
    if checkpoint["elapsed_seconds"] > result["recorded_elapsed_seconds_total"]:
        raise ValueError("completed checkpoint runtime exceeds the final recorded training runtime")
    return target, steps


def installed_attempt(attempt, admitted, receipt, evidence):
    """Check actual target placement/assets and every installed byte without asserting a mapped-library observation."""
    hardware = evidence.read(attempt / "placement_hardware.json")
    installed = evidence.read(attempt / "installed_backend.json")
    assets = evidence.read(attempt / "asset_receipt.json")
    deployment.hardware_identity(hardware, receipt["target"])
    runtime = deployment.actual_runtime(hardware.get("runtime_environment"), admitted["runtime_environment"])
    if (not same(runtime, receipt["actual_runtime_environment"])
            or not same({key: value for key, value in assets.items() if key != "root"}, receipt["assets"])
            or set(installed) != {"packages", "jaxtari_source", "gpu_libraries"}
            or not same(installed["packages"], receipt["backend"]["packages"])
            or not same(installed["gpu_libraries"], receipt["backend"]["gpu_libraries"])):
        raise ValueError("deployed attempt changed its target runtime/assets/installed backend")
    public = deployment.jaxtari_identity(installed["jaxtari_source"], installed["packages"], evidence)
    if not same(public, receipt["backend"]["jaxtari_source"]):
        raise ValueError("deployed attempt changed its installed public Python source")
    for path, record in installed["gpu_libraries"].items():
        evidence.image(path, record["sha256"])
    return hardware, installed


def actual_attempt(directory, admitted, receipt, identity, result, checkpoint, evidence, checkpoint_proof_attempt=None):
    """Validate current installed admission and the exact current/prior process that observed checkpoint libraries."""
    progress = evidence.read(directory / "progress.json")
    number = progress.get("attempt")
    if type(number) is not int or number < 1:
        raise ValueError("deployed training needs its actual process attempt")
    attempt = directory / "attempts" / f"attempt-{number:04d}"
    if not same(evidence.read(attempt / "progress.json"), progress) or progress.get("state") == "failed":
        raise ValueError("deployed progress does not describe a valid current process attempt")
    if result is not None:
        expected_state = "completed" if result["completed"] else "checkpointed"
        if (result.get("attempt") != number or not same(evidence.read(attempt / "result.json"), result)
                or progress.get("state") != expected_state or any(not same(progress.get(name), value) for name, value in result.items())):
            raise ValueError("deployed result/progress does not describe its actual finished attempt")
    hardware, installed = installed_attempt(attempt, admitted, receipt, evidence)
    if progress.get("pid") != hardware.get("pid"):
        raise ValueError("deployed progress belongs to another process")
    hardware_record = evidence.read(directory / "hardware.json")
    if (hardware_record.get("pid") != hardware["pid"] or hardware_record.get("hostname", "").split(".")[0] != hardware["hostname"]
            or hardware_record.get("device_kind") != identity["backend"]["device_kind"]
            or not same(deployment.actual_runtime(hardware_record.get("runtime_environment"), admitted["runtime_environment"]),
                        receipt["actual_runtime_environment"])):
        raise ValueError("deployed unit hardware does not match its actual process")
    process = {"attempt": number, "pid": hardware["pid"], "process_start_time": hardware["process_start_time"],
               "hostname": hardware["hostname"], "gpu_uuid": hardware["gpu_uuid"], "slurm_job_id": hardware.get("slurm_job_id", "")}
    if checkpoint["iteration"] == 0:
        if checkpoint.get("loaded_gpu_libraries") is not None or checkpoint_proof_attempt is not None:
            raise ValueError("an initial zero-update checkpoint cannot claim mapped training-library evidence")
        return {**process, "checkpoint_proof_attempt": None}
    proof_number = number if checkpoint_proof_attempt is None else checkpoint_proof_attempt
    if type(proof_number) is not int or not 1 <= proof_number <= number:
        raise ValueError("checkpoint mapped-library proof needs an explicit current or older attempt number")
    proof = directory / "attempts" / f"attempt-{proof_number:04d}"
    proof_hardware, proof_installed = installed_attempt(proof, admitted, receipt, evidence)
    loaded = evidence.read(proof / "loaded_gpu_libraries.json")
    if loaded.get("pid") != proof_hardware["pid"] or not same(loaded, checkpoint.get("loaded_gpu_libraries")):
        raise ValueError("deployed mapped libraries belong to another process/checkpoint")
    # Rebuild the small original backend-match receipt independently, retaining its original image path identities.
    backend_evidence = deployment.DeploymentEvidence(evidence.resolver)
    observed_backend = deployment.backend_identity(proof_installed, loaded, backend_evidence)
    if not same(observed_backend, receipt["backend"]):
        raise ValueError("completed deployed process used another installed or mapped backend")
    expected_match = {"kind": "training_deployment_backend_match", "deployment_digest": training.canonical_digest(receipt),
        "server": receipt["target"]["server"], "hostname": proof_hardware["hostname"], "gpu_uuid": proof_hardware["gpu_uuid"],
        "backend_digest": training.canonical_digest(observed_backend), "verified_images": backend_evidence.records()}
    match = evidence.read(proof / "deployment_backend_match.json")
    if not same(match, expected_match):
        raise ValueError("deployed mapped-backend match was replaced or belongs to another attempt")
    if proof_number == number and (not same(match, evidence.read(directory / "deployment_backend_match.json"))
            or not same(loaded, evidence.read(directory / "loaded_gpu_libraries.json"))):
        raise ValueError("current deployed mapped-library proof differs from its unit records")
    for path, expected in backend_evidence.images.items():
        evidence.image(path, expected)
    return {**process, "checkpoint_proof_attempt": proof_number,
            "checkpoint_proof_process": {"pid": proof_hardware["pid"], "process_start_time": proof_hardware["process_start_time"],
                "hostname": proof_hardware["hostname"], "gpu_uuid": proof_hardware["gpu_uuid"]}, "backend_match": match}


def bound_record(record, evidence):
    """Read a specifically named raw file and require its complete pinned byte hash."""
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("terminal ownership evidence requires explicit path/SHA256 records")
    contents = evidence.file(record["path"])
    if hashlib.sha256(contents).hexdigest() != deployment.sha256(record["sha256"]):
        raise ValueError("terminal ownership raw bytes differ from their pinned SHA256")
    return contents


def terminal_ownership(path, admitted, directory, identity, evidence):
    """Verify an exact owned Slurm claim, ledger, accounting row and complete terminal unit snapshot."""
    record = evidence.read(path)
    required = {"schema_version", "kind", "cluster", "campaign_run_dir", "job_id", "observed_at", "ledger_path",
                "ledger_snapshot", "submission_record", "accounting", "terminal_unit_snapshot"}
    if (set(record) != required or type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("kind") != "owned_slurm_terminal_receipt"
            or record.get("cluster") != "aces" or not isinstance(record.get("job_id"), str)
            or not re.fullmatch(r"[1-9][0-9]*", record["job_id"])):
        raise ValueError("checkpoint recovery needs an exact owned ACES Slurm terminal receipt")
    observed = datetime.fromisoformat(record["observed_at"].replace("Z", "+00:00"))
    if observed.tzinfo is None or observed.utcoffset() != timezone.utc.utcoffset(observed):
        raise ValueError("terminal observations require an actual UTC timestamp")
    campaign = Path(portable_evidence.original_path(record["campaign_run_dir"]))
    ledger = campaign / "slurm/submitted_jobids_aces.txt"
    if directory != campaign / "data" / admitted["unit_id"] or record["ledger_path"] != str(ledger):
        raise ValueError("terminal job ownership names another campaign/unit ledger")
    snapshot_lines = bound_record(record["ledger_snapshot"], evidence).decode().splitlines()
    current_lines = evidence.file(ledger, record=False).decode().splitlines()
    if (not snapshot_lines or any(not re.fullmatch(r"[1-9][0-9]*", item) for item in current_lines)
            or record["job_id"] not in snapshot_lines or current_lines[:len(snapshot_lines)] != snapshot_lines):
        raise ValueError("terminal job is absent from its owned append-only ledger")
    claim = portable_evidence.document(bound_record(record["submission_record"], evidence))
    training_spec = {"path": admitted["validation"]["specification_path"],
                     "sha256": hashlib.sha256(evidence.file(admitted["validation"]["specification_path"])).hexdigest()}
    if (type(claim.get("schema_version")) is not int or claim["schema_version"] != 1 or claim.get("cluster") != "aces" or claim.get("job_id") != record["job_id"]
            or claim.get("unit_id") != admitted["unit_id"] or not same(claim.get("training_specification"), training_spec)
            or not same(claim.get("helpers"), identity["helpers"])
            or type(claim.get("supervisor_pid")) is not int or claim["supervisor_pid"] < 1
            or not isinstance(claim.get("supervisor_process_start_time"), str) or not claim["supervisor_process_start_time"].isdigit()):
        raise ValueError("terminal claim does not bind this unit/specification/helpers and actual owning process")
    plan = portable_evidence.document(bound_record(claim["job_plan"], evidence))
    submitted = portable_evidence.document(bound_record(claim["generic_submission"], evidence))
    if (not same(plan.get("helpers"), identity["helpers"]) or type(submitted.get("schema_version")) is not int or submitted["schema_version"] != 1
            or submitted.get("cluster") != "aces" or submitted.get("job_id") != record["job_id"]
            or not same(submitted.get("job_plan"), claim["job_plan"])):
        raise ValueError("terminal generic submission and immutable job plan differ from the unit claim")
    accounting = record["accounting"]
    command = ["sacct", "-X", "-j", record["job_id"], "--format=JobIDRaw,State,ExitCode,End", "-P", "-n"]
    if set(accounting) != {"command", "returncode", "stdout"} or accounting["command"] != command or type(accounting["returncode"]) is not int or accounting["returncode"] != 0:
        raise ValueError("terminal accounting must be the exact owned-ID sacct query")
    rows = bound_record(accounting["stdout"], evidence).decode().strip().splitlines()
    values = rows[0].split("|") if len(rows) == 1 else []
    if (len(values) != 5 or values[0] != record["job_id"] or values[4] != ""
            or values[1] not in {"COMPLETED", "TIMEOUT", "PREEMPTED", "NODE_FAIL"}
            or values[1] == "COMPLETED" and values[2] != "0:0"
            or not re.fullmatch(r"[0-9]+:[0-9]+", values[2]) or values[3] in {"", "Unknown", "None"}):
        raise ValueError("terminal accounting is not one owned permitted terminal job row")
    datetime.fromisoformat(values[3])
    snapshot = checked_unit_snapshot(portable_evidence.document(bound_record(record["terminal_unit_snapshot"], evidence)),
                                     directory, evidence, "deployed_training_terminal_unit_snapshot")
    if snapshot["observed_at"] != record["observed_at"]:
        raise ValueError("terminal unit snapshot belongs to another observation")
    return {"receipt_path": evidence.path_identity(path), "job_id": record["job_id"], "state": values[1],
            "snapshot": snapshot, "receipt_digest": training.canonical_digest(record)}


def checked_unit_snapshot(snapshot, directory, evidence, kind):
    """Validate an explicit complete byte snapshot; absent files never come from a missing resolver mapping."""
    if (type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1 or snapshot.get("kind") != kind
            or snapshot.get("unit_directory") != str(directory)
            or type(snapshot.get("attempt")) is not int or snapshot["attempt"] < 1
            or not isinstance(snapshot.get("files"), dict) or not isinstance(snapshot.get("absent"), list)
            or len(set(snapshot["absent"])) != len(snapshot["absent"])):
        raise ValueError("terminal unit snapshot does not identify this process attempt and observation")
    observed = datetime.fromisoformat(snapshot["observed_at"].replace("Z", "+00:00"))
    if observed.tzinfo is None or observed.utcoffset() != timezone.utc.utcoffset(observed):
        raise ValueError("unit snapshot observations require actual UTC timestamps")
    for relative, file_record in snapshot["files"].items():
        parts = PurePosixPath(relative)
        if (not relative or parts.is_absolute() or ".." in parts.parts or str(parts) != relative
                or set(file_record) != {"path", "sha256"} or file_record["path"] != str(directory / relative)):
            raise ValueError("terminal unit snapshot needs exact canonical unit file paths")
        evidence.image(file_record["path"], file_record["sha256"])
    present = {row["path"] for row in snapshot["files"].values()}
    for absent in snapshot["absent"]:
        original = portable_evidence.original_path(absent)
        if not Path(original).is_relative_to(directory) or original in present:
            raise ValueError("terminal absence proof conflicts with the actual unit snapshot")
        if evidence.resolver is not None and original in evidence.resolver.files:
            raise ValueError("unit snapshot marked an explicitly mapped file absent: " + original)
        if evidence.resolver is None and Path(original).exists():
            raise ValueError("terminal unit snapshot omitted a file that actually exists")
    if evidence.resolver is None:
        actual = {str(item.relative_to(directory)) for item in directory.rglob("*") if item.is_file()}
        if actual != set(snapshot["files"]):
            raise ValueError("terminal unit snapshot does not include all actual unit files")
    else:
        mapped = {path for path in evidence.resolver.files if Path(path).is_relative_to(directory)}
        if mapped != present:
            raise ValueError("unit snapshot does not include exactly all mapped unit files")
    return snapshot


def optional_unit_record(path, evidence, terminal=None):
    """Distinguish a recorded absence from an unmapped historical file without falling back to local paths."""
    if terminal is not None:
        snapshot = terminal["snapshot"]
        if str(path) in {item["path"] for item in snapshot["files"].values()}:
            return evidence.read(path)
        if str(path) in snapshot["absent"]:
            return None
        raise ValueError("terminal snapshot lacks explicit presence/absence for " + str(path))
    if evidence.resolver is not None:
        # A missing resolver mapping is an error, never evidence that a failure/result was absent.
        physical = evidence.physical_file(path)
        return evidence.read(path) if physical.is_file() else None
    return evidence.read(path) if path.is_file() else None


def checkpoint_schedule(admitted, digest, checkpoint):
    """Validate an actual zero/partial/full checkpoint and its original unpadded report position."""
    iteration = checkpoint.get("iteration")
    target = 2_000_000_000 // (128 * 128)
    if type(iteration) is not int or not 0 <= iteration <= target:
        raise ValueError("deployed checkpoint update count is outside the original schedule")
    execution = admitted["configuration"]["execution"]
    pending = 0 if execution["report_readout"] == "none" else iteration % execution["report_every_iterations"]
    expected = {"schema_version": 1, "kind": "optimized_training_checkpoint", "identity_digest": digest,
        "configuration_digest": training.canonical_digest(admitted["configuration"]), "source_commit": admitted["source_commit"],
        "iteration": iteration, "target_updates": target, "steps_per_copy": iteration * 128 * 128,
        "copy_seeds": admitted["copy_seeds"], "pending_report_updates": pending}
    if any(not same(checkpoint.get(key), value) for key, value in expected.items()):
        raise ValueError("deployed checkpoint metadata does not describe its exact admitted state")
    admission.validation.positive(checkpoint.get("elapsed_seconds"), "checkpoint runtime")
    return iteration, target, pending


def validate_deployed_training_checkpoint(specification_path, unit_directory, resolver=None, terminal_receipt=None,
                                         checkpoint_proof_attempt=None, unit_snapshot=None, *, historical_read_only=False):
    """Validate full completion or one safely resumable checkpoint from raw owned terminal evidence."""
    admitted = admission.validate_deployed_training_admission(specification_path, resolver=resolver,
                                                              historical_read_only=historical_read_only)
    evidence = deployment.DeploymentEvidence(resolver)
    directory = Path(evidence.path_identity(unit_directory, strict=True))
    if directory.name != admitted["unit_id"]:
        raise ValueError("completed deployed unit directory differs from its reserved unit identifier")
    receipt = evidence.read(admitted["deployment_verification"])
    if training.canonical_digest(receipt) != admitted["validation"]["deployment_digest"]:
        raise ValueError("the completed deployment receipt changed after admission")
    identity = evidence.read(directory / "training_identity.json")
    checkpoint = evidence.read(directory / "checkpoints/latest.metadata.json")
    digest = checkpoint_identity(admitted, receipt, identity, evidence)
    iteration, target, pending = checkpoint_schedule(admitted, digest, checkpoint)
    saved_specification = evidence.read(directory / "final_specification.json")
    if (not same(runner.immutable_specification(saved_specification), runner.immutable_specification(admitted))
            or not same(evidence.read(directory / "validation_receipt.json"), admitted["validation"])
            or not same(evidence.read(directory / "training_unit.json"), {"kind": "validated_optimized_training",
                "specification_digest": training.canonical_digest(runner.immutable_specification(admitted))})):
        raise ValueError("completed deployed unit changed its original specification/admission identity")
    if terminal_receipt is not None and unit_snapshot is not None:
        raise ValueError("use one exact terminal or graceful unit snapshot, not both")
    terminal = terminal_ownership(terminal_receipt, admitted, directory, identity, evidence) if terminal_receipt is not None else None
    snapshot_proof = terminal
    if unit_snapshot is not None:
        raw_snapshot = evidence.file(unit_snapshot)
        snapshot_proof = {"snapshot": checked_unit_snapshot(portable_evidence.document(raw_snapshot), directory, evidence,
                                                           "deployed_training_unit_snapshot"),
            "path": evidence.path_identity(unit_snapshot), "sha256": hashlib.sha256(raw_snapshot).hexdigest()}
    result = optional_unit_record(directory / "result.json", evidence, snapshot_proof)
    progress = evidence.read(directory / "progress.json")
    number = progress.get("attempt")
    if type(number) is not int or number < 1 or snapshot_proof is not None and snapshot_proof["snapshot"]["attempt"] != number:
        raise ValueError("checkpoint recovery does not identify the current actual attempt")
    attempt = directory / "attempts" / f"attempt-{number:04d}"
    if result is not None and result.get("attempt") != number:
        old = result.get("attempt")
        if type(old) is not int or not 1 <= old < number or not same(evidence.read(directory / "attempts" / f"attempt-{old:04d}" / "result.json"), result):
            raise ValueError("a stale unit result does not match its exact older process attempt")
        result = None
    completed = result is not None and result.get("completed") is True
    if completed:
        completed_schedule(admitted, digest, result, checkpoint)
    else:
        if optional_unit_record(attempt / "failure.json", evidence, snapshot_proof) is not None:
            raise ValueError("application-failed training is not eligible for automatic checkpoint continuation")
        if result is None:
            if (terminal is None or terminal["state"] not in {"TIMEOUT", "PREEMPTED", "NODE_FAIL"}
                    or optional_unit_record(attempt / "result.json", evidence, snapshot_proof) is not None):
                raise ValueError("a periodic checkpoint without a current result needs exact owned terminal infrastructure evidence")
        else:
            expected = {"completed": False, "update": iteration, "target_updates": target, "requested_steps_per_copy": 2_000_000_000,
                "steps_per_copy": iteration * 128 * 128, "copy_seeds": admitted["copy_seeds"],
                "copies": admitted["configuration"]["learner"]["n_copies"], "identity_digest": digest,
                "pending_report_updates": pending, "last_report_iteration": 0 if admitted["configuration"]["execution"]["report_readout"] == "none" else iteration - pending}
            if (iteration == target or any(not same(result.get(key), value) for key, value in expected.items())
                    or result.get("stop_reason") not in {"SIGUSR1", "SIGTERM", "maximum_seconds"}):
                raise ValueError("partial deployed result is not an exact graceful checkpoint boundary")
            for key in ("process_elapsed_seconds", "recorded_elapsed_seconds_total"):
                admission.validation.positive(result.get(key), "checkpointed training runtime")
            if checkpoint["elapsed_seconds"] > result["recorded_elapsed_seconds_total"]:
                raise ValueError("checkpoint runtime exceeds its graceful boundary result")
    process = actual_attempt(directory, admitted, receipt, identity, result, checkpoint, evidence, checkpoint_proof_attempt)
    if terminal is not None and process["slurm_job_id"] != terminal["job_id"]:
        raise ValueError("the terminal owned job differs from the actual training process allocation")
    path = directory / "checkpoints/latest.pkl"
    if checkpoint.get("checkpoint_path") != evidence.path_identity(path):
        raise ValueError("completed deployed checkpoint is not at its recorded original path")
    evidence.image(path, checkpoint.get("checkpoint_sha256"))
    evidence.recheck_images()
    return {"schema_version": 1, "kind": "deployed_training_checkpoint", "completed": completed,
        "source_root": admitted["source_root"], "source_commit": admitted["source_commit"], "hardware_class": "h100",
        "unit_id": admitted["unit_id"], "copy_seeds": admitted["copy_seeds"], "updates": iteration, "target_updates": target,
        "steps_per_copy": iteration * 128 * 128, "checkpoint_proof_attempt": process["checkpoint_proof_attempt"],
        "configuration_digest": training.canonical_digest(admitted["configuration"]), "identity_digest": digest,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"], "deployment_digest": admitted["validation"]["deployment_digest"],
        "admission": admitted["validation"], "process": process, "terminal": terminal,
        "unit_snapshot": snapshot_proof if terminal is None else None, "evidence": evidence.records()}


def validate_deployed_training_completion(specification_path, unit_directory, resolver=None, terminal_receipt=None,
                                         checkpoint_proof_attempt=None, unit_snapshot=None, *, historical_read_only=False):
    """Require true full-budget completion from the same strict checkpoint/admission implementation."""
    result = validate_deployed_training_checkpoint(specification_path, unit_directory, resolver, terminal_receipt,
                                                   checkpoint_proof_attempt, unit_snapshot,
                                                   historical_read_only=historical_read_only)
    if result["completed"] is not True:
        raise ValueError("deployed training has not completed the exact original per-policy update/report schedule")
    return {**result, "kind": "deployed_training_completion"}


def main():
    """Write an immutable completion receipt only after all original raw evidence has been revalidated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--unit-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--checkpoint-proof-attempt", type=int)
    parser.add_argument("--unit-snapshot", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--evidence-manifest-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--historical-read-only", action="store_true")
    args = parser.parse_args()
    if args.evidence_manifest_sha256 and args.evidence_manifest is None:
        parser.error("an expected manifest hash requires --evidence-manifest")
    resolver = portable_evidence.ExactPathResolver(args.evidence_manifest, args.evidence_manifest_sha256) if args.evidence_manifest else None
    validate = validate_deployed_training_checkpoint if args.checkpoint else validate_deployed_training_completion
    result = validate(args.specification, args.unit_directory, resolver, args.terminal_receipt,
                      args.checkpoint_proof_attempt, args.unit_snapshot, historical_read_only=args.historical_read_only)
    contents = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if args.output is None:
        print(contents.decode(), end="")
    elif args.output.exists():
        if args.output.read_bytes() != contents:
            raise ValueError("an existing deployed completion receipt is immutable")
    else:
        with args.output.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
