"""Classify the failed exact accepted-evidence reference graph without opening payload bytes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign/accepted_evidence_snapshot.py"
CAMPAIGN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
PACKET = CAMPAIGN / "remote_setup/2026-11-05-00-54_accepted-direct-evidence_cpu-snapshot_maximum-check-seconds-285_cpu-step-seconds-300_processor-count-1_host-memory-mib-512_gpu-count-0"
FAILED = CAMPAIGN / "data" / PACKET.name
FAILED_METADATA = {FAILED / name for name in ("snapshot_attempt.json", "snapshot_stdout.raw", "snapshot_stderr.raw", "snapshot_return.json")}
STARTED = time.monotonic()
STEP = os.environ["SLURM_STEP_ID"]
assert os.environ.get("SLURM_JOB_ID") == "6783003" and STEP.isdecimal() and os.getresuid() == (2618919, 2618919, 2618919)
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]
SOURCE_SHA256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
start_record = Path(__file__).with_name("metadata_classification_start_" + STEP + ".json")
start_record.write_text(json.dumps({"job_id": "6783003", "step_id": STEP, "candidate_source_sha256": SOURCE_SHA256,
    "base_commit": "7d07da121c41c531d51cbd25b73579735212ead8", "metadata_only": True}, sort_keys=True) + "\n")
specification = importlib.util.spec_from_file_location("exact_unmodified_failed_snapshot_source", SOURCE)
snapshot = importlib.util.module_from_spec(specification)
specification.loader.exec_module(snapshot)
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("no numerical import in reference classification")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system", "subprocess.Popen"}:
        raise RuntimeError("no network, library or tool operation in reference classification")
    if event == "open" and isinstance(values[0], (str, bytes)):
        path = Path(os.fsdecode(values[0]))
        if str(path).startswith(("/dev/", "/tmp/sl5nw_serval02_card1.lock")):
            raise RuntimeError("no device or standing-lock open")
        if path.suffix not in {".json", ".py"} and path not in FAILED_METADATA and not path.is_relative_to(Path(__file__).parent):
            raise RuntimeError("no payload open during metadata-only classification: " + str(path))


sys.addaudithook(audit)
preserved = {}
preservation = Path(__file__).with_name("preserved_failed_actual_attempt")
preservation.mkdir(exist_ok=True)
for name in ("snapshot_attempt.json", "snapshot_stdout.raw", "snapshot_stderr.raw", "snapshot_return.json"):
    original = FAILED / name
    assert original.stat().st_size < 1024 * 1024
    raw = original.read_bytes()
    if not (preservation / name).exists():
        with (preservation / name).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    else:
        assert (preservation / name).read_bytes() == raw
    preserved[str(original)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "copy": str(preservation / name)}
inputs_raw = (PACKET / "inputs.json").read_bytes()
assert hashlib.sha256(inputs_raw).hexdigest() == "9a69ba96aaeb5560fced8e6299674cf9ffa548d1466a062e7bbd8bb1e45c5c0f"
inputs = snapshot.document(inputs_raw)
with snapshot.ValidationFiles(285) as files:
    inventory = snapshot.AcceptedInventory(files)
    qualification = snapshot.accepted_role(files, inventory, inputs["qualification_review"], "qualification")
    qualification_review = files.json(inputs["qualification_review"])
    verification = files.json(inputs["qualification_verification"])
    assert verification["passed"] is True and verification["unchanged_full_consumer_passed"] is True
    inventory.add(inputs["qualification_verification"]); inventory.references(verification)
    correctness = [snapshot.accepted_role(files, inventory, value, "correctness", qualification_review, qualification["qualified_asset_receipt"])
        for value in inputs["correctness_reviews"]]
    inventory.process_metadata()
    captures = {}
    for row in inventory.logical_mappings:
        captures.setdefault((row["original_path"], row["sha256"]), set()).add(row["captured_path"])
    rows = []
    for original, expected in sorted(inventory.pending_blobs):
        key = (original, expected)
        if key in captures:
            kind, resolved = "captured_logical_path_and_hash", sorted(captures[key])
        elif key in inventory.physical_captures:
            kind, resolved = "explicit_declared_physical_capture", [original]
        elif key in inventory.shared_metadata:
            kind, resolved = "exact_shared_metadata_declaration", [original]
        else:
            kind, resolved = "UNRESOLVED", []
        rows.append({"original_path": original, "sha256": expected, "classification": kind, "resolved_paths": resolved,
            "shared_metadata": inventory.shared_metadata.get(key), "physical_capture_role": inventory.physical_captures.get(key)})
    annotation_references = {(row["source_annotation"]["path"], row["source_annotation"]["sha256"])
        for row in inventory.historical_source_annotations}
    required_count = len(rows)
    for original, expected in sorted(annotation_references):
        assert (original, expected) not in inventory.pending_blobs
        rows.append({"original_path": original, "sha256": expected, "classification": "HASHED_HISTORICAL_ANNOTATION_NOT_EXECUTION_DEPENDENCY",
            "resolved_paths": [], "current_execution_dependency": False})
    missing = {(row["original_path"], row["sha256"]): row for row in rows if row["classification"] == "UNRESOLVED"}
    def visit(value, pointer, document_reference):
        if type(value) is dict:
            if type(value.get("path")) is str and type(value.get("sha256")) is str and (value["path"], value["sha256"]) in missing:
                missing[(value["path"], value["sha256"])].setdefault("references", []).append({"document": document_reference, "field": pointer})
            if type(value.get("source_root")) is str and type(value.get("files")) is dict:
                for relative, item in value["files"].items():
                    expected = item.get("sha256") if type(item) is dict else item
                    key = (str(Path(value["source_root"]) / relative), expected)
                    if key in missing:
                        missing[key].setdefault("references", []).append({"document": document_reference, "field": pointer + "/files/" + relative})
            if type(value.get("verified_package_libraries")) is dict:
                for path, item in value["verified_package_libraries"].items():
                    key = (path, item["sha256"])
                    if key in missing:
                        missing[key].setdefault("references", []).append({"document": document_reference, "field": pointer + "/verified_package_libraries/" + path})
            for name, item in value.items():
                visit(item, pointer + "/" + name, document_reference)
        elif type(value) is list:
            for number, item in enumerate(value):
                visit(item, pointer + "/" + str(number), document_reference)
    for entry in files.physical.values():
        aliases = [name for name in entry["aliases"] if name.endswith(".json")]
        if aliases and entry["metadata_bytes"] is not None:
            visit(snapshot.document(entry["metadata_bytes"]), "", {"path": aliases[0], "sha256": entry["sha256"]})
    record = {"schema_version": 1, "kind": "actual_failed_snapshot_complete_reference_classification_without_payload_reads",
        "job_id": "6783003", "step_id": STEP, "base_commit": "7d07da121c41c531d51cbd25b73579735212ead8",
        "candidate_source_sha256": SOURCE_SHA256,
        "elapsed_seconds": time.monotonic() - STARTED, "metadata_physical_files_read": len(files.physical),
        "metadata_bytes_hashed": files.bytes_hashed, "pending_reference_count": len(rows), "unresolved_count": len(missing),
        "first_unresolved": next(iter(missing.values()), None), "all_unresolved": list(missing.values()), "classifications": rows,
        "historical_payload_files_opened": 0, "actual_snapshot_produced": False, "old_raw_gates_reexecuted": False,
        "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)), "gpu_operations": 0, "scientific_credit": 0,
        "complete_formal_evidence": False}
    record["required_execution_reference_count"] = required_count
    record["historical_annotation_reference_count"] = len(annotation_references)
    record["historical_annotation_instances"] = inventory.historical_source_annotations
    if not missing:
        inventory.resolve_blobs()
        record["exact_actual_resolver_passed_without_payload_reads"] = True
    record["preserved_failed_actual_attempt"] = preserved
path = Path(__file__).with_name("actual_reference_classification_" + STEP + ".json")
with path.open("x") as stream:
    json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
print(json.dumps({"receipt": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "unresolved_count": record["unresolved_count"],
    "first_unresolved": record["first_unresolved"], "elapsed_seconds": record["elapsed_seconds"]}, sort_keys=True))
