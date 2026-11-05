"""Reuse genuine accepted direct evidence after one bounded CPU file snapshot.

The snapshot hashes every distinct physical file once, including different files
with the same expected content. Descriptor identities belong only to that CPU
host. GPU workers consume an independently reviewed historical fact; their live
source, assets, backend and CUDA checks remain separate.
"""
import argparse
import atexit
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import stat
import sys
import time

NAME = "accepted_evidence_snapshot.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
NUMERICAL_SOURCE = "d28e6735e79788f618acb0ebfde8609db1f776af"
ORIGINAL_HELPER_SOURCE = "0a880984b4a03a2ff7335337cc18576628bc9fc1"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
ACCEPTED_REVIEW_SHA256 = {"qualification": "8c61ec70059475a479a2e0b395bdc6cd2d5cb2a0e062ae05ef51e066c4a4d01a",
    "1": "22f4ec55120c987fe8d44b55e6427ab75e25db61785e29d7cb6bdcbc55df8818",
    "2": "b0aa9a0953a4079e592dd86c4c692c0b09b3958c891760a26103c7e3a7fa1bf6"}
PROOF_KIND = "complete_accepted_direct_evidence_cpu_snapshot"
REVIEW_KIND = "independent_complete_accepted_direct_evidence_cpu_snapshot_review"
PROTOCOL = {"schema_version": 1, "capability": "genuine_accepted_direct_evidence_with_complete_bounded_cpu_snapshot",
    "maximum_check_seconds": 300, "reuse_scope": "historical_evidence_only",
    "cross_host_descriptor_identity_asserted": False, "operational_admission": False}
IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid", "st_gid", "st_nlink")
MAXIMUM_METADATA_BYTES = 16 * 1024 * 1024
HISTORICAL_METADATA_FIELDS = {"registry", "study_registry", "admission", "exit_observations", "exit_plan", "exit_status", "worker_stop"}
_CONSUMERS = {}
_SOURCE_FILES = {}


def require(condition, message):
    """Refuse incomplete source, input, file or accepted-review evidence."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Serialize finite metadata with stable keys for exact review references."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify exact supplied bytes without importing or loading a library."""
    return hashlib.sha256(raw).hexdigest()


def document(raw):
    """Refuse duplicate keys and nonfinite JSON instead of accepting ambiguous metadata."""
    def unique(pairs):
        """Require one value per explicit metadata key."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate accepted-evidence metadata key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite accepted-evidence metadata")))


def exact(value, expected):
    """Keep Boolean, integer and floating-point metadata fields distinct."""
    return encoded(value) == encoded(expected)


def reference_shape(value):
    """Require one exact absolute file path and a complete lowercase SHA-256 hash."""
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and type(value["path"]) is str and Path(value["path"]).is_absolute()
        and str(Path(value["path"])) == value["path"] and ".." not in Path(value["path"]).parts
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "an exact accepted-evidence path and SHA-256 reference are required")
    return value


def identity(value):
    """Retain stable descriptor and pathname fields, excluding read-access timestamps."""
    return {name: getattr(value, name) for name in IDENTITY_FIELDS}


def ordinary(path):
    """Keep evidence paths canonical and refuse symbolic links or another file kind."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
        "accepted evidence requires one exact ordinary file path")
    return path


class ValidationFiles:
    """Hash each pinned physical file once during one explicitly bounded validation."""

    def __init__(self, maximum_seconds=285, *, clock=time.monotonic, maximum_open_files=None, metadata_only=False):
        """Keep a fresh descriptor cache and refuse limits beyond the user's five-minute bound."""
        require(type(maximum_seconds) in {int, float} and math.isfinite(maximum_seconds)
            and 0 < maximum_seconds <= 300, "each validation must finish within five minutes")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        budget = min(65536, soft - 64) if maximum_open_files is None else maximum_open_files
        require(type(budget) is int and 0 < budget <= soft - 64,
            "file descriptor budget requires bounded shards instead of silently exceeding the limit")
        self.clock, self.started, self.maximum_seconds = clock, clock(), maximum_seconds
        self.maximum_open_files, self.descriptor_limits = budget, [soft, hard]
        self.physical, self.paths, self.bytes_hashed = {}, {}, 0
        self.failed, self.closed, self.completed = False, False, False
        self.metadata_only = metadata_only

    def begin_metadata_reuse(self):
        """Bound each later identity check while retaining only already verified small metadata buffers."""
        require(self.metadata_only and self.completed and not self.failed and not self.closed
            and all(entry["metadata_bytes"] is not None for entry in self.physical.values()),
            "only a complete pinned metadata reference cache may be reused")
        self.started = self.clock()
        for entry in self.physical.values():
            self.check_entry(entry)

    def check_deadline(self):
        """Stop incomplete work before it can exceed the declared validation limit."""
        require(not self.closed and self.clock() - self.started < self.maximum_seconds,
            "accepted-evidence validation reached its time limit without a complete proof")

    def check_entry(self, entry):
        """Recheck the pinned descriptor and every declared alias without hashing the bytes again."""
        self.check_deadline()
        require(identity(os.fstat(entry["descriptor"])) == entry["identity"], "a pinned evidence descriptor changed")
        for name in entry["aliases"]:
            path = ordinary(name)
            require(identity(path.stat(follow_symlinks=False)) == entry["identity"],
                "an accepted-evidence pathname was replaced or changed")

    def file(self, path, expected_sha256, *, keep_bytes=False):
        """Read a new physical file once, or recheck all identities before reusing its digest."""
        descriptor = None
        try:
            self.check_deadline()
            require(type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
                "every evidence file needs its already declared complete content hash")
            path = ordinary(path); name = str(path)
            if name in self.paths:
                entry = self.physical[self.paths[name]]
                self.check_entry(entry)
                require(entry["sha256"] == expected_sha256, "one evidence path has conflicting expected content")
                return entry
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(descriptor)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == 2618919,
                "an evidence descriptor must be an ordinary file owned by this campaign")
            require(not self.metadata_only or keep_bytes and info.st_size <= MAXIMUM_METADATA_BYTES,
                "the GPU consumer can validate small reviewed references, never rehash historical payloads")
            observed = identity(info); key = (info.st_dev, info.st_ino)
            require(identity(path.stat(follow_symlinks=False)) == observed, "evidence pathname changed during descriptor capture")
            if key in self.physical:
                entry = self.physical[key]
                require(entry["identity"] == observed and entry["sha256"] == expected_sha256,
                    "a physical evidence alias changed identity or expected content")
                self.check_entry(entry)
                entry["aliases"].append(name); self.paths[name] = key
                return entry
            require(len(self.physical) < self.maximum_open_files,
                "too many unique files require bounded shards and complete proof composition")
            require(not keep_bytes or info.st_size <= MAXIMUM_METADATA_BYTES, "metadata exceeds its bounded in-memory size")
            value, blocks, total = hashlib.sha256(), [], 0
            while True:
                self.check_deadline()
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                value.update(block); total += len(block); self.bytes_hashed += len(block)
                if keep_bytes:
                    blocks.append(block)
            require(total == info.st_size and value.hexdigest() == expected_sha256,
                "an accepted evidence file differs from its declared complete bytes")
            entry = {"descriptor": descriptor, "identity": observed, "aliases": [name],
                "sha256": expected_sha256, "bytes": total, "metadata_bytes": b"".join(blocks) if keep_bytes else None}
            self.physical[key] = entry; self.paths[name] = key; descriptor = None
            self.check_entry(entry)
            return entry
        except BaseException:
            self.failed = True
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def small_bytes(self, value):
        """Reuse a validated metadata buffer, never recomputing its content digest."""
        reference_shape(value)
        entry = self.file(value["path"], value["sha256"], keep_bytes=True)
        require(entry["bytes"] <= MAXIMUM_METADATA_BYTES, "accepted metadata is too large")
        if entry["metadata_bytes"] is None:
            # A later metadata interpretation may read the pinned bytes again; its digest remains the one actual hash.
            self.check_entry(entry)
            entry["metadata_bytes"] = os.pread(entry["descriptor"], entry["bytes"], 0)
            require(len(entry["metadata_bytes"]) == entry["bytes"], "pinned metadata ended before its verified size")
            self.check_entry(entry)
        return entry["metadata_bytes"]

    def json(self, value):
        """Parse only a fully hashed accepted metadata reference."""
        return document(self.small_bytes(value))

    def finish(self, expected_paths):
        """Require exact total coverage and final descriptor/path checks before returning a complete snapshot."""
        try:
            self.check_deadline()
            require(not self.failed and set(self.paths) == set(expected_paths) and self.paths,
                "partial, failed or missing evidence coverage cannot form a complete snapshot")
            for entry in self.physical.values():
                self.check_entry(entry)
            self.check_deadline(); self.completed = True
            return {"complete": True, "producer_hostname": socket.gethostname().split(".")[0],
                "descriptor_identity_scope": "this CPU host and this validation only",
                "maximum_seconds": self.maximum_seconds, "elapsed_seconds": self.clock() - self.started,
                "descriptor_limits": self.descriptor_limits, "maximum_open_files": self.maximum_open_files,
                "distinct_paths": len(self.paths), "distinct_physical_files": len(self.physical),
                "bytes_hashed": self.bytes_hashed, "physical_files": [{"identity": entry["identity"],
                    "sha256": entry["sha256"], "bytes": entry["bytes"], "aliases": sorted(entry["aliases"])}
                    for unused, entry in sorted(self.physical.items())]}
        except BaseException:
            self.failed = True
            raise

    def close(self):
        """Close only this validation's own descriptors, including after a failed check."""
        if not self.closed:
            for entry in self.physical.values():
                os.close(entry["descriptor"])
            self.closed = True

    def __enter__(self):
        """Keep descriptor custody explicit for the whole validation."""
        return self

    def __exit__(self, error_type, error, traceback):
        """Never leave validation descriptors open after success or failure."""
        self.close()


def compute_context(run):
    """Require the real owned numeric CPU step before building any accepted-evidence snapshot."""
    require(os.getuid() == os.geteuid() == 2618919, "snapshot production requires this campaign owner")
    require(not {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}.intersection(sys.modules),
        "accepted-evidence snapshot production cannot inherit numerical initialization")
    require(Path(run).is_absolute() and Path(run).resolve(strict=True) == Path(run) and Path(run).is_dir(),
        "the exact ordinary campaign directory is required")
    job, step = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[0-9]+", step),
        "accepted-evidence snapshots require an owned numeric compute step, never a login host")
    require(Path("/proc/self/cgroup").read_text().splitlines() == [
        "0::/system.slice/slurmstepd.scope/job_" + job + "/step_" + step + "/user/task_0"], "snapshot kernel ownership changed")
    uids = [line.split()[1:] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Uid:")]
    require(uids == [["2618919"] * 4] and (Path(run) / "slurm/submitted_jobids_uva.txt").read_text().splitlines().count(job) == 1,
        "snapshot process or ledger is outside the exact campaign owner")
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"hostname": socket.gethostname().split(".")[0], "job_id": job, "step_id": step,
        "pid": os.getpid(), "parent_pid": int(fields[1]), "process_start_time": fields[19], "uids": [2618919] * 4,
        "argv": [value.decode() for value in Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").split(b"\0")]}


class AcceptedInventory:
    """Collect exactly declared accepted references and all historical manifest payloads."""

    def __init__(self, files):
        """Keep a fresh declared path map and separate logical mapping coverage."""
        self.files, self.declared, self.pending_json, self.visited_json = files, {}, [], set()
        self.manifests, self.logical_mappings = {}, []
        self.pending_blobs, self.resolved_logical_references = set(), []
        self.inline_historical_records = []

    def add(self, value, *, metadata=None, captured=False):
        """Require consistent declared hashes before considering a file or metadata reference."""
        reference_shape(value); path, sha = value["path"], value["sha256"]
        if metadata is None:
            metadata = Path(path).suffix == ".json"
        if not metadata and not captured:
            self.pending_blobs.add((path, sha))
            return
        require(path not in self.declared or self.declared[path] == sha, "accepted references disagree about one exact physical file")
        self.declared[path] = sha
        if metadata and path not in self.visited_json:
            self.pending_json.append(value)

    def references(self, value, field=None):
        """Read only explicit absolute path/hash pairs and known source-file maps."""
        if isinstance(value, dict):
            original, captured = value.get("original"), value.get("captured")
            if type(original) is dict and type(captured) is dict and all(type(item.get("path")) is str
                    and type(item.get("sha256")) is str for item in (original, captured)):
                original = {key: original[key] for key in ("path", "sha256")}
                captured = {key: captured[key] for key in ("path", "sha256")}
                reference_shape(original); reference_shape(captured)
                require(original["sha256"] == captured["sha256"], "accepted driver/backend capture changed its exact original bytes")
                self.add(captured, metadata=Path(captured["path"]).suffix == ".json", captured=True)
                self.logical_mappings.append({"accepted_capture": True, "original_path": original["path"],
                    "captured_path": captured["path"], "sha256": captured["sha256"]})
            if type(value.get("path")) is str and Path(value["path"]).is_absolute() and type(value.get("sha256")) is str:
                if value["path"].startswith("/proc/"):
                    require(type(value.get("raw")) is str and digest(value["raw"].encode()) == value["sha256"],
                        "historical process data needs its captured inline bytes, never a live process read")
                    self.inline_historical_records.append({"path": value["path"], "sha256": value["sha256"],
                        "scope": "captured inline historical process bytes; no live process queried"})
                else:
                    require(not value["path"].startswith("/dev/"), "historical evidence cannot open a live GPU or device file")
                    self.add({"path": value["path"], "sha256": value["sha256"]},
                        metadata=Path(value["path"]).suffix == ".json" and field not in HISTORICAL_METADATA_FIELDS)
            if type(value.get("source_root")) is str and isinstance(value.get("files"), dict):
                for relative, item in value["files"].items():
                    sha = item.get("sha256") if isinstance(item, dict) else item
                    require(type(relative) is str and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                        "a source receipt contains another root or relative traversal")
                    self.add({"path": str(Path(value["source_root"]) / relative), "sha256": sha}, metadata=False)
            if isinstance(value.get("verified_package_libraries"), dict):
                for path, item in value["verified_package_libraries"].items():
                    self.add({"path": path, "sha256": item["sha256"]}, metadata=False)
            for key, item in value.items():
                self.references(item, key)
        elif isinstance(value, list):
            for item in value:
                self.references(item, field)

    def process_metadata(self):
        """Discover metadata dependencies before streaming the large declared payloads."""
        while self.pending_json:
            value = self.pending_json.pop(0); path = value["path"]
            if path in self.visited_json:
                continue
            self.visited_json.add(path)
            metadata = self.files.json(value)
            self.references(metadata)
            if isinstance(metadata, dict) and isinstance(metadata.get("files"), list) and isinstance(metadata.get("git_roots"), list):
                mappings = metadata["files"]; seen = set()
                for row in mappings:
                    require(type(row) is dict and set(row) == {"original_path", "relative_path", "sha256"}
                        and type(row["original_path"]) is str and Path(row["original_path"]).is_absolute()
                        and row["original_path"] not in seen, "historical payload mapping is duplicate or malformed")
                    relative = row["relative_path"]
                    require(type(relative) is str and not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                        "historical payload escaped its declared capture directory")
                    captured = str(Path(path).parent / relative)
                    self.add({"path": captured, "sha256": row["sha256"]}, metadata=False, captured=True)
                    self.logical_mappings.append({"manifest": value, "original_path": row["original_path"],
                        "captured_path": captured, "sha256": row["sha256"]})
                    seen.add(row["original_path"])
                self.manifests[path] = {"reference": value, "logical_mapping_count": len(mappings),
                    "git_root_declarations": metadata["git_roots"]}

    def resolve_blobs(self):
        """Resolve declared historical logical references without opening today's original paths."""
        captures = {}
        for row in self.logical_mappings:
            captures.setdefault((row["original_path"], row["sha256"]), set()).add(row["captured_path"])
        for original, expected in sorted(self.pending_blobs):
            resolved = sorted(captures.get((original, expected), {original}))
            for path in resolved:
                require(path not in self.declared or self.declared[path] == expected,
                    "accepted evidence has conflicting content for one physical path")
                self.declared[path] = expected
            self.resolved_logical_references.append({"declared_path": original, "sha256": expected,
                "actual_paths_hashed": resolved, "used_historical_capture": (original, expected) in captures})

    def hash_complete(self):
        """Hash all still-unread distinct physical files and refuse incomplete coverage."""
        self.process_metadata()
        self.resolve_blobs()
        for path, expected in sorted(self.declared.items()):
            self.files.file(path, expected)
        return self.files.finish(self.declared)


def zero_credit(value):
    """Require a genuine review to retain its original zero-credit admission boundary."""
    require(value.get("admission_granted") is False and value.get("worker_admitted") is False,
        "accepted historical evidence cannot admit a new worker")
    for key in ("scientific_credit", "case_credit", "idea_credit", "site_qualification_credit", "final_training_credit"):
        require(type(value.get(key)) is int and value[key] == 0, "accepted review changed its zero-credit field: " + key)


def source_contract(files, reference_value, qualified_plan, qualified_directory):
    """Retain the complete original55/raw53 statement provenance without replaying those gates."""
    contract = files.json(reference_value)
    require(contract.get("kind") == "exact_fresh_direct_qualification_raw_reader_two_review_statement_difference"
        and type(contract.get("original_body_statement_count")) is int and contract["original_body_statement_count"] == 55
        and type(contract.get("retained_body_statement_count")) is int and contract["retained_body_statement_count"] == 53
        and contract.get("removed_statement_indices") == [5, 50]
        and contract.get("staged_source", {}).get("path") == str(qualified_directory / "code/direct_measurement.py")
        and contract["staged_source"]["sha256"] == qualified_plan["helpers"]["files"][PREFIX + "direct_measurement.py"],
        "accepted qualification lost the exact original55/raw53 source contract")
    original_tree = ast.parse(files.small_bytes(contract["staged_source"]))
    raw_tree = ast.parse(files.small_bytes(contract["new_raw_reader"]))
    original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_qualification_inputs")
    raw = next(node for node in raw_tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_unreviewed_actual_raw_qualification")
    retained = [ast.dump(node, include_attributes=False) for index, node in enumerate(original.body) if index not in (5, 50)]
    observed = [ast.dump(node, include_attributes=False) for node in raw.body]
    require(len(original.body) == 55 and len(raw.body) == 53 and retained == observed
        and ast.dump(original.args, include_attributes=False) == ast.dump(raw.args, include_attributes=False)
        and contract["removed_statements"] == [ast.unparse(original.body[index]) for index in (5, 50)]
        and contract["removed_statements"][0] == "review = identity.read_receipt(inputs['review'])"
        and "independent actual qualification review is required" in contract["removed_statements"][1]
        and contract["retained_body_ast_sha256"] == digest("\n".join(observed).encode()),
        "accepted qualification changed an original raw gate or removed more than the review dependency")
    return {"reference": reference_value, "original_statements": 55, "retained_raw_statements": 53,
        "retained_body_ast_sha256": contract["retained_body_ast_sha256"],
        "raw_gates_reexecuted_by_snapshot": False, "genuine_prior_validation_reused": True}


def accepted_role(files, inventory, review_reference, role, qualification_review=None):
    """Bind a genuine accepted review to its exact result, plan, source and copy seeds."""
    inventory.add(review_reference); review = files.json(review_reference)
    accepted_key = "qualification" if role == "qualification" else str(review.get("copies"))
    require(review_reference["sha256"] == ACCEPTED_REVIEW_SHA256.get(accepted_key),
        "only these already genuine accepted original qualification and correctness reviews may be reused")
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("approved") is True and review.get("independent_of_producer") is True
        and type(review.get("actual_returncode")) is int and review["actual_returncode"] == 0,
        "a genuine complete independent accepted review is required")
    zero_credit(review)
    result_reference = review["result"]; result = files.json(result_reference)
    binding = result["resolved_device_binding"]
    plan_reference = review["plan"] if role == "correctness" else binding["plan"]
    plan = files.json(plan_reference); unit = Path(result_reference["path"]).parent
    require(plan_reference["path"] == str(unit / "measurement_plan.json")
        and unit == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and result.get("source_commit") == plan.get("source_commit") == NUMERICAL_SOURCE
        and plan["helpers"]["source_commit"] == ORIGINAL_HELPER_SOURCE
        and len(plan["helpers"]["files"]) == 26
        and exact(result.get("helpers"), plan["helpers"])
        and plan.get("hardware_class") == "h100" and binding.get("gpu_uuid") == CARD_ONE,
        "accepted evidence changed its original source, real target or exact unit")
    require(exact(result.get("runtime_environment"), plan["runtime_environment"])
        and exact(result.get("asset_record"), plan["asset_record"])
        and exact(result.get("copy_seeds"), plan["copy_seeds"]),
        "accepted evidence changed its exact original runtime, assets or random streams")
    # These fixed files are authenticated by values already inside the accepted result.
    inventory.add({"path": str(unit / "direct_device_binding.json"), "sha256": digest(encoded(binding))})
    inventory.add({"path": str(unit / "direct_process_metadata.json"), "sha256": binding["metadata_sha256"]})
    if role == "qualification":
        require(review.get("kind") == "independent_full_original_direct_qualification_review"
            and review.get("raw_validation_passed_before_review_existed") is True
            and review.get("placeholder_or_partial_approval_used") is False
            and result.get("kind") == "direct_serval02_card1_full_original_qualification"
            and result.get("completed") is True and result.get("all_original_gates_passed") is True,
            "only the genuine full-original qualification can be reused")
        raw_provenance = source_contract(files, review["source_contract"], plan, unit)
    else:
        require(review.get("kind") == "independent_actual_direct_common_configuration_correctness_review"
            and review.get("passed") is True and review.get("unresolved_anomalies") == []
            and result.get("measurement_mode") == "correctness" and result.get("measurement_valid") is False
            and result.get("correctness_attempted") is True and result.get("correctness_passed") is True
            and result.get("implementation_distinct") is True,
            "timing or partial data cannot replace separately accepted correctness")
        raw = files.json(review["raw_audit"])
        require(raw.get("kind") == "independent_actual_direct_common_correctness_raw_validation"
            and raw.get("passed") is True and raw.get("unchanged_staged_consumer_passed") is True
            and raw.get("result") == result_reference and raw.get("plan") == plan_reference,
            "accepted correctness lost its complete genuine raw-consumer validation")
        for key in ("source_commit", "helper_commit", "configuration", "reference_configuration", "copies", "copy_seeds",
                "gpu_uuid", "actual_returncode", "terminal_process", "hostname", "pid", "process_start_time"):
            require(key in raw and key in review and exact(raw[key], review[key]), "accepted correctness raw/review join changed: " + key)
        require(exact(raw.get("execution_identity"), result["execution_identity"])
            and all(exact(review[key], result["execution_identity"][key]) for key in ("hostname", "pid", "process_start_time")),
            "accepted correctness changed its actual completed execution identity")
        require(exact(review["configuration"], plan["configuration"])
            and exact(review["reference_configuration"], plan["reference_configuration"])
            and exact(review["copy_seeds"], plan["copy_seeds"])
            and review["source_commit"] == NUMERICAL_SOURCE and review["helper_commit"] == ORIGINAL_HELPER_SOURCE
            and review["gpu_uuid"] == CARD_ONE and raw["qualification_source_contract"] == qualification_review["source_contract"],
            "accepted correctness changed its original configuration, target or qualification provenance")
        zero_credit(raw)
        count = plan["configuration"]["learner"]["n_copies"]
        require(type(count) is int and count in (1, 2) and type(review["copies"]) is int and review["copies"] == count
            and plan["copy_seeds"] == list(range(1, count + 1)) and all(type(seed) is int for seed in plan["copy_seeds"]),
            "accepted correctness used different independent-copy seed identities")
        require(result["execution_identity"]["finished_at_seconds"] <= review["reviewed_at_seconds"],
            "the genuine correctness review predates its actual completed run")
        raw_provenance = {"reference": review["raw_audit"], "unchanged_staged_consumer_passed": True,
            "raw_gates_reexecuted_by_snapshot": False, "genuine_prior_validation_reused": True}
    inventory.add(result_reference); inventory.add(plan_reference)
    inventory.references(review); inventory.references(result); inventory.references(plan)
    original_history = files.json(plan["direct_binding"]["frozen_inputs"])
    manifest_reference = original_history["evidence_manifest"]
    manifest = files.json(manifest_reference)
    require(manifest.get("kind") == "portable_raw_evidence" and type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1 and type(manifest.get("files")) is list and len(manifest["files"]) == 693
        and type(manifest.get("git_roots")) is list and len(manifest["git_roots"]) == 8,
        "an accepted original history must retain its complete 693 mappings and eight source declarations")
    return {"review": review_reference, "result": result_reference, "plan": plan_reference,
        "campaign_run_dir": plan["campaign_run_dir"],
        "source_commit": plan["source_commit"], "helper_commit": plan["helpers"]["source_commit"],
        "runtime_environment": plan["runtime_environment"], "asset_record": plan["asset_record"],
        "copy_seeds": plan["copy_seeds"], "configuration": plan["configuration"],
        "reference_configuration": plan.get("reference_configuration"), "direct_profile": plan["direct_binding"]["profile"],
        "core": plan["core"], "paired_source": plan["paired_source"], "source_root": plan["source_root"], "python": plan["python"],
        "backend_preparation": plan["inputs"]["direct_backend_image_record"], "backend_expected": plan["direct_backend_image"]["expected"],
        "historical_inputs": plan["direct_binding"]["frozen_inputs"], "historical_manifest": manifest_reference,
        "raw_validation_provenance": raw_provenance}


def build_snapshot(run, inputs, producer_sha256, maximum_seconds=285):
    """Build one complete reviewed-input snapshot without replaying accepted numerical or raw-gate cases."""
    context = compute_context(run)
    require(type(inputs) is dict and set(inputs) == {"qualification_review", "qualification_verification", "correctness_reviews"}
        and isinstance(inputs["correctness_reviews"], list) and len(inputs["correctness_reviews"]) == 2,
        "snapshot production needs exactly the genuine qualification and both accepted correctness reviews")
    for value in (inputs["qualification_review"], inputs["qualification_verification"], *inputs["correctness_reviews"]):
        reference_shape(value)
    with ValidationFiles(maximum_seconds) as files:
        files.file(str(Path(__file__).resolve(strict=True)), producer_sha256)
        inventory = AcceptedInventory(files)
        inventory.add({"path": str(Path(__file__).resolve(strict=True)), "sha256": producer_sha256}, metadata=False, captured=True)
        qualification = accepted_role(files, inventory, inputs["qualification_review"], "qualification")
        qualification_review = files.json(inputs["qualification_review"])
        verification = files.json(inputs["qualification_verification"])
        require(verification.get("passed") is True and verification.get("raw_checks_before_review") is True
            and verification.get("unchanged_full_consumer_passed") is True
            and verification.get("independent_review") == inputs["qualification_review"]
            and verification.get("result") == qualification["result"]
            and verification.get("source_contract") == qualification_review["source_contract"],
            "the genuine qualification lacks its completed raw-first and unchanged full-consumer verification")
        inventory.add(inputs["qualification_verification"]); inventory.references(verification)
        correctness = [accepted_role(files, inventory, value, "correctness", qualification_review)
            for value in inputs["correctness_reviews"]]
        by_count = {row["configuration"]["learner"]["n_copies"]: row for row in correctness}
        require(set(by_count) == {1, 2} and len(by_count) == len(correctness), "both exact accepted copy counts are required")
        for row in correctness:
            for key in ("source_commit", "runtime_environment", "asset_record", "direct_profile", "core", "paired_source", "source_root", "python"):
                require(exact(row[key], qualification[key]), "accepted correctness changed the original qualified " + key)
        require(all(row["campaign_run_dir"] == str(Path(run)) for row in (qualification, *correctness)),
            "the accepted evidence and exact owned CPU ledger must belong to this same campaign")
        actual = inventory.hash_complete()
        return {"schema_version": 1, "kind": PROOF_KIND, "complete": True, "protocol": PROTOCOL,
            "inputs": inputs, "accepted": {"qualification": qualification, "correctness": {str(key): row for key, row in by_count.items()}},
            "producer": context, "producer_source_sha256": producer_sha256, "created_at_seconds": time.time(),
            "coverage_scope": "all explicit accepted references and all declared historical captured payloads",
            "declared_files": [{"path": path, "sha256": expected} for path, expected in sorted(inventory.declared.items())],
            "historical_manifests": inventory.manifests, "logical_mapping_count": len(inventory.logical_mappings),
            "resolved_logical_references": inventory.resolved_logical_references, "file_snapshot": actual,
            "inline_historical_records": inventory.inline_historical_records,
            "previous_gpu_or_raw_gate_cases_reexecuted": False, "live_training_inputs_validated_by_this_snapshot": False,
            "cross_host_descriptor_identity_asserted": False, "admission_granted": False, "worker_admitted": False,
            "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "throughput_credit": 0, "packing_point_credit": 0,
            "final_training_credit": 0, "site_qualification_credit": 0}


def metadata_reference(path):
    """Bind a small metadata file without opening old payloads or a device."""
    path = ordinary(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        observed = identity(os.fstat(descriptor))
        require(stat.S_ISREG(observed["st_mode"]) and observed["st_uid"] == 2618919
            and observed["st_size"] <= MAXIMUM_METADATA_BYTES, "reviewed snapshot metadata exceeds its ordinary owned-file bound")
        raw = os.read(descriptor, MAXIMUM_METADATA_BYTES + 1)
        require(len(raw) == observed["st_size"] and identity(os.fstat(descriptor)) == observed
            and identity(path.stat(follow_symlinks=False)) == observed, "a metadata reference changed during capture")
        return {"path": str(path), "sha256": digest(raw)}
    finally:
        os.close(descriptor)


def required_helper_names():
    """Derive the declared source set from the two unchanged literal dependency lists."""
    names = []
    for filename in ("measure_selected.py", "serval02_card1_binding.py"):
        tree = ast.parse(Path(__file__).with_name(filename).read_bytes())
        assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "HELPERS" for target in node.targets)]
        require(len(assignments) == 1, "one exact literal helper dependency list is required")
        values = ast.literal_eval(assignments[0])
        require(type(values) is tuple and all(type(value) is str and Path(value).name == value for value in values),
            "the helper source set cannot contain another namespace or a computed dependency")
        names.extend(values)
    names.extend(("direct_measurement.py", "configured_measurement.py", "direct_backend_image.py",
        "direct_common_confirmation.py", "direct_historical_consumer.py", NAME))
    result = tuple(dict.fromkeys(names))
    require(len(result) == 29, "the new reviewed-snapshot closure must retain its exact twenty-nine source files")
    return result


def history_document(plan):
    """Keep only current explicit references to an already reviewed complete historical fact."""
    return {"schema_version": 1, "kind": "reviewed_accepted_direct_history_inputs", "protocol": PROTOCOL,
        "accepted_evidence_snapshot_inputs": plan["accepted_evidence_snapshot_inputs"],
        "campaign_run_dir": plan["campaign_run_dir"], "source_commit": plan["source_commit"],
        "configuration": plan["configuration"], "reference_configuration": plan["reference_configuration"],
        "copy_seeds": plan["copy_seeds"]}


def verify_helper_sources(plan):
    """Hash each current small helper once and recheck its local identity on later calls."""
    names = required_helper_names()
    require(set(plan["helpers"]["files"]) == {PREFIX + name for name in names},
        "accepted reuse needs its exact complete helper source set, not a relaxed file count")
    key = (os.getpid(), digest(encoded(plan["helpers"]["files"])), str(Path(__file__).parent))
    if key not in _SOURCE_FILES:
        files = ValidationFiles(60, metadata_only=True)
        try:
            for name in names:
                files.file(str(Path(__file__).with_name(name)), plan["helpers"]["files"][PREFIX + name], keep_bytes=True)
            files.finish(files.paths)
        except BaseException:
            files.close()
            raise
        _SOURCE_FILES[key] = files
        atexit.register(files.close)
    else:
        files = _SOURCE_FILES[key]
        files.begin_metadata_reuse(); files.finish(files.paths)


def publish(path, raw):
    """Create and synchronize a new ordinary metadata record without replacing an earlier attempt."""
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve(strict=True) == path.parent and not path.exists()
        and not path.is_symlink(), "an existing or indirect metadata output cannot be replaced")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stage_history_inputs(plan, path):
    """Stage a small same-unit pointer, without copying or revalidating old payloads."""
    expected = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"] / "direct_history" / "inputs.json"
    require(Path(path) == expected, "accepted history must use the exact reserved current-unit path")
    expected.parent.mkdir(parents=True, exist_ok=True)
    raw = encoded(history_document(plan))
    if expected.exists():
        require(ordinary(expected).read_bytes() == raw, "the reserved history input already records another attempt")
    else:
        publish(expected, raw)
    return metadata_reference(expected)


def configure(plan, proof_path, review_path):
    """Declare two actual reviewed references before direct history or qualification consumption."""
    require(proof_path is not None and review_path is not None, "new common confirmation needs an actual CPU snapshot and independent review")
    plan["accepted_evidence_snapshot_protocol"] = dict(PROTOCOL)
    plan["accepted_evidence_snapshot_inputs"] = {"proof": metadata_reference(proof_path), "review": metadata_reference(review_path)}
    return plan


def require_plan(plan):
    """Limit accepted-history reuse to this exact original-source two-rendering common comparison."""
    require(plan.get("kind") == "selected_configuration_measurement" and plan.get("mode") == "confirmation"
        and plan.get("hardware_class") == "h100" and plan.get("gpu_uuid") == CARD_ONE
        and plan.get("source_commit") == NUMERICAL_SOURCE and "direct_binding" in plan
        and exact(plan.get("accepted_evidence_snapshot_protocol"), PROTOCOL)
        and not any(key in plan for key in ("configured_isolation", "configured_group", "direct_qualification", "target_qualification",
            "final_protocol", "final_confirmation_protocol", "completed_study", "device_memory_limit_protocol", "scaling_protocol", "native_process_control")),
        "accepted history cannot admit another source, target, adaptive, final, capacity or scaling operation")
    require(all(type(plan.get(key)) is int and plan[key] == value for key, value in (
        ("rounds", 7), ("iterations", 10), ("minimum_updates_per_round", 10))), "accepted reuse retains seven pairs of ten complete updates")
    verify_helper_sources(plan)
    require(all(set(plan["configuration"][scope]) == set(plan["reference_configuration"][scope])
        for scope in ("learner", "environment", "execution")), "accepted reuse changed configuration fields")
    changes = [(scope, key, plan["reference_configuration"][scope][key], value)
        for scope in ("learner", "environment", "execution") for key, value in plan["configuration"][scope].items()
        if not exact(value, plan["reference_configuration"][scope][key])]
    require(sorted(changes) == [("environment", "cache_room_backgrounds", False, True),
        ("environment", "render_final_two_only", False, True)] and all(type(configuration["environment"][key]) is bool
            for configuration in (plan["configuration"], plan["reference_configuration"])
            for key in ("cache_room_backgrounds", "render_final_two_only")), "accepted reuse changed more than its two reviewed rendering options")
    inputs = plan.get("accepted_evidence_snapshot_inputs")
    require(type(inputs) is dict and set(inputs) == {"proof", "review"}, "the actual CPU snapshot and genuine independent review are required")
    for value in inputs.values():
        reference_shape(value)
    return inputs


def validate_complete_file_proof(proof):
    """Verify complete recorded CPU coverage without applying CPU descriptor identities on this host."""
    actual = proof["file_snapshot"]
    require(actual.get("complete") is True and actual.get("producer_hostname") == proof["producer"]["hostname"]
        and actual.get("descriptor_identity_scope") == "this CPU host and this validation only"
        and type(actual.get("elapsed_seconds")) in {int, float} and math.isfinite(actual["elapsed_seconds"])
        and 0 <= actual["elapsed_seconds"] < actual["maximum_seconds"] <= 300,
        "CPU snapshot lacks a complete bounded validation on its declared host")
    require(type(actual.get("maximum_seconds")) in {int, float} and math.isfinite(actual["maximum_seconds"])
        and type(actual.get("bytes_hashed")) is int and type(actual.get("distinct_paths")) is int
        and type(actual.get("distinct_physical_files")) is int and type(actual.get("physical_files")) is list,
        "complete CPU coverage needs exact finite timing and integer file counters")
    expected = {reference_shape(value)["path"]: value["sha256"] for value in proof["declared_files"]}
    require(len(expected) == len(proof["declared_files"]) and expected, "CPU snapshot repeats or omits declared coverage")
    paths, identities, total = {}, set(), 0
    for value in actual["physical_files"]:
        observed = value["identity"]
        require(set(observed) == set(IDENTITY_FIELDS) and all(type(item) is int for item in observed.values())
            and observed["st_ino"] > 0 and observed["st_size"] >= 0 and stat.S_ISREG(observed["st_mode"])
            and observed["st_uid"] == 2618919 and type(value["bytes"]) is int and value["bytes"] == observed["st_size"], "CPU snapshot physical-file record changed")
        key = (observed["st_dev"], observed["st_ino"])
        require(key not in identities and type(value["aliases"]) is list and value["aliases"], "CPU snapshot repeats one physical file")
        identities.add(key); total += value["bytes"]
        for path in value["aliases"]:
            require(path not in paths and expected.get(path) == value["sha256"], "CPU snapshot aliases do not cover the exact declared files")
            paths[path] = value["sha256"]
    require(paths == expected and len(paths) == actual["distinct_paths"]
        and len(identities) == actual["distinct_physical_files"] and total == actual["bytes_hashed"],
        "CPU snapshot is partial or rehashes a physical file instead of complete exact coverage")
    return actual


class ReviewedSnapshot:
    """Consume one genuine immutable CPU fact while retaining live training gates elsewhere."""

    def __init__(self, plan):
        """Pin only small snapshot/review/current-reference metadata on this host."""
        self.plan_hash = digest(encoded(plan)); self.inputs = require_plan(plan)
        self.files = ValidationFiles(60, metadata_only=True)
        try:
            self.proof = self.files.json(self.inputs["proof"])
            self.review = self.files.json(self.inputs["review"])
            proof, review = self.proof, self.review
            require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
                and proof.get("kind") == PROOF_KIND and proof.get("complete") is True and exact(proof.get("protocol"), PROTOCOL)
                and proof.get("producer_source_sha256") == plan["helpers"]["files"][PREFIX + NAME]
                and proof.get("previous_gpu_or_raw_gate_cases_reexecuted") is False
                and proof.get("live_training_inputs_validated_by_this_snapshot") is False
                and proof.get("cross_host_descriptor_identity_asserted") is False,
                "only the complete same-source historical CPU snapshot may be consumed")
            zero_credit(proof)
            require(all(type(proof.get(key)) is int and proof[key] == 0 for key in ("throughput_credit", "packing_point_credit"))
                and type(proof.get("created_at_seconds")) in {int, float} and math.isfinite(proof["created_at_seconds"])
                and proof["created_at_seconds"] > 0, "a historical CPU snapshot cannot grant timing, capacity or final credit")
            require(type(review.get("schema_version")) is int and review["schema_version"] == 1
                and review.get("kind") == REVIEW_KIND and review.get("approved") is True and review.get("passed") is True
                and review.get("independent_of_producer") is True and review.get("snapshot") == self.inputs["proof"]
                and review.get("producer_source_sha256") == proof["producer_source_sha256"]
                and review.get("complete_coverage_verified") is True and review.get("unresolved_anomalies") == []
                and type(review.get("reviewer_identity")) is str and bool(review["reviewer_identity"].strip())
                and type(review.get("reviewed_at_seconds")) in {int, float} and math.isfinite(review["reviewed_at_seconds"])
                and proof["created_at_seconds"] <= review["reviewed_at_seconds"]
                and review.get("admission_granted") is False and review.get("worker_admitted") is False
                and type(review.get("scientific_credit")) is int and review["scientific_credit"] == 0,
                "a genuine independent complete CPU-snapshot review is required")
            validate_complete_file_proof(proof)
            self.qual = proof["accepted"]["qualification"]
            count = plan["configuration"]["learner"]["n_copies"]
            require(type(count) is int and count in (1, 2), "accepted history supports only the existing matched copy counts")
            self.correctness = proof["accepted"]["correctness"][str(count)]
            require(set(proof["accepted"]["correctness"]) == {"1", "2"}
                and self.qual["review"]["sha256"] == ACCEPTED_REVIEW_SHA256["qualification"]
                and all(proof["accepted"]["correctness"][key]["review"]["sha256"] == ACCEPTED_REVIEW_SHA256[key]
                    for key in ("1", "2")), "the snapshot changed an already accepted historical review")
            declared = {value["path"]: value["sha256"] for value in proof["declared_files"]}
            historical_manifests = set()
            for accepted in (self.qual, *proof["accepted"]["correctness"].values()):
                for name in ("review", "result", "plan", "historical_inputs", "historical_manifest"):
                    value = reference_shape(accepted[name])
                    require(declared.get(value["path"]) == value["sha256"],
                        "the complete CPU snapshot omitted an explicit accepted historical reference")
                actual_review = self.files.json(accepted["review"])
                actual_plan = self.files.json(accepted["plan"])
                require(actual_review["result"] == accepted["result"] and accepted["helper_commit"] == ORIGINAL_HELPER_SOURCE
                    and actual_plan["helpers"]["source_commit"] == ORIGINAL_HELPER_SOURCE,
                    "accepted snapshot changed its genuine reviewed result or original helper source")
                for key in ("campaign_run_dir", "source_root", "source_commit", "core", "paired_source", "python",
                        "runtime_environment", "asset_record", "configuration", "copy_seeds"):
                    require(exact(accepted[key], actual_plan[key]), "accepted snapshot changed its actual original plan field: " + key)
                require(exact(accepted["reference_configuration"], actual_plan.get("reference_configuration"))
                    and accepted["backend_expected"] == actual_plan["direct_backend_image"]["expected"]
                    and accepted["backend_preparation"] == actual_plan["inputs"]["direct_backend_image_record"]
                    and accepted["historical_inputs"] == actual_plan["direct_binding"]["frozen_inputs"],
                    "accepted snapshot substituted its original plan's backend or historical input references")
                require(exact(accepted["direct_profile"], actual_plan["direct_binding"]["profile"]),
                    "accepted snapshot substituted the actual approved historical direct profile")
                original_history = self.files.json(accepted["historical_inputs"])
                manifest_reference = accepted["historical_manifest"]
                require(original_history.get("evidence_manifest") == manifest_reference,
                    "the accepted historical manifest differs from its exact captured input")
                manifest = self.files.json(manifest_reference)
                require(manifest.get("kind") == "portable_raw_evidence" and type(manifest.get("schema_version")) is int
                    and manifest["schema_version"] == 1 and len(manifest["files"]) == 693 and len(manifest["git_roots"]) == 8,
                    "complete accepted history requires all original captured mappings and sources")
                historical_manifests.add(manifest_reference["path"])
                for row in manifest["files"]:
                    require(type(row) is dict and set(row) == {"original_path", "relative_path", "sha256"}
                        and type(row["relative_path"]) is str and not Path(row["relative_path"]).is_absolute()
                        and ".." not in Path(row["relative_path"]).parts,
                        "the accepted historical capture mapping is malformed")
                    captured = str(Path(manifest_reference["path"]).parent / row["relative_path"])
                    require(declared.get(captured) == row["sha256"],
                        "the reviewed CPU snapshot omitted a declared captured physical payload")
            require(set(proof["historical_manifests"]) == historical_manifests and len(historical_manifests) == 3,
                "the complete snapshot must bind exactly the genuine qualification and both correctness histories")
            require(self.qual["review"] == plan["direct_qualification_inputs"]["review"]
                and self.qual["result"] == plan["direct_qualification_inputs"]["result"]
                and self.correctness["review"] == plan["common_correctness_inputs"]["review"]
                and self.correctness["result"] == plan["common_correctness_inputs"]["result"], "the current plan changed its actual accepted reviews")
            require(exact(self.correctness["configuration"], plan["configuration"])
                and exact(self.correctness["reference_configuration"], plan["reference_configuration"])
                and exact(self.correctness["copy_seeds"], plan["copy_seeds"]), "accepted correctness changed configuration or independent random streams")
            for role in (self.qual, self.correctness):
                for key in ("campaign_run_dir", "source_root", "source_commit", "core", "paired_source", "python", "runtime_environment", "asset_record"):
                    require(exact(role[key], plan[key]), "accepted snapshot differs from the current exact " + key)
                require(exact(role["direct_profile"], plan["direct_binding"]["profile"]), "accepted snapshot changed its approved direct physical target")
                require(role["backend_preparation"] == plan["inputs"]["direct_backend_image_record"],
                    "accepted history changed its exact prepared backend package record")
                previous_backend = self.files.json(role["backend_expected"])
                current_backend = self.files.json(plan["direct_backend_image"]["expected"])
                require(all(exact(previous_backend[key], current_backend[key]) for key in
                    ("selected_python", "site_packages", "image", "distribution", "producer_source_sha256")),
                    "accepted history changed the prepared backend image, owning distribution or exact producer")
                # These are small actual referenced records, not the historical binary payload set.
                self.files.json(role["review"]); self.files.json(role["plan"])
            historical = plan["direct_binding"]["frozen_inputs"]
            require(reference_shape(historical)["path"] == str(Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
                / "direct_history/inputs.json") and exact(self.files.json(historical), history_document(plan)),
                "the small current history pointer differs from this complete accepted snapshot")
            self.qualification_result = self.files.json(self.qual["result"])
            self.correctness_result = self.files.json(self.correctness["result"])
            self.files.finish(self.files.paths)
        except BaseException:
            self.files.close()
            raise
        atexit.register(self.files.close)

    def check(self, plan):
        """Refuse changed local metadata references without rehashing their already pinned bytes."""
        require(self.plan_hash == digest(encoded(plan)), "the frozen current plan changed after accepted evidence consumption")
        self.files.begin_metadata_reuse()
        self.files.finish(self.files.paths)
        return self


def consume(plan):
    """Keep one process-local exact-reference consumer, never a stale cross-host descriptor cache."""
    require_plan(plan)
    key = (os.getpid(), digest(encoded(plan)))
    if key not in _CONSUMERS:
        _CONSUMERS[key] = ReviewedSnapshot(plan)
    return _CONSUMERS[key].check(plan)


def qualification(plan):
    """Return the genuinely accepted original qualification, not a replayed or fabricated success flag."""
    return consume(plan).qualification_result


def common_validation(plan, plan_sha256):
    """Bind accepted correctness and qualification to this unchanged common comparison only."""
    selected = consume(plan)
    return {"schema_version": 1, "kind": "reviewed_accepted_direct_common_correctness_validation", "passed": True,
        "plan_sha256": plan_sha256, "common_correctness_inputs": plan["common_correctness_inputs"],
        "accepted_evidence_snapshot_inputs": plan["accepted_evidence_snapshot_inputs"],
        "source_commit": plan["source_commit"], "prior_correctness_identity": selected.correctness_result["execution_identity"],
        "qualification_source_contract": selected.qual["raw_validation_provenance"], "parent_pid": os.getppid(),
        "finished_at_seconds": time.time(), "historical_raw_gates_recomputed": False,
        "cpu_snapshot_reused_as_reviewed_historical_fact": True, "gpu_time_historical_byte_equality_asserted": False,
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False}


def history(plan):
    """Return explicitly reviewed past history without claiming to recompute120 cases or current GPU bytes."""
    selected = consume(plan)
    return {"kind": "reviewed_accepted_direct_history_fact", "inputs": plan["direct_binding"]["frozen_inputs"],
        "snapshot": selected.inputs["proof"], "independent_review": selected.inputs["review"],
        "comparisons_previously_validated": 120, "comparisons_recomputed": 0,
        "historical_payloads_rehashed_on_gpu": False, "cpu_descriptor_identity_asserted_on_gpu": False}


def main():
    """Produce only a fresh bounded CPU snapshot; independent review is a separate required action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=float, default=285)
    arguments = parser.parse_args()
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "snapshot input declarations exceed their small metadata bound")
    result = build_snapshot(arguments.run_dir, document(raw), arguments.producer_sha256, arguments.maximum_seconds)
    publish(arguments.output, encoded(result))
    print(json.dumps({"kind": PROOF_KIND, "snapshot": metadata_reference(arguments.output), "complete": True,
        "distinct_physical_files": result["file_snapshot"]["distinct_physical_files"],
        "bytes_hashed": result["file_snapshot"]["bytes_hashed"], "scientific_credit": 0}, allow_nan=False))


if __name__ == "__main__":
    main()
