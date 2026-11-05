"""Produce the existing four-context metadata fact without approving or admitting it."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import time


HELPER_COMMIT = "d1469fd063a44167ec56760729eb0beea7ecf7d3"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
NAME = "configured_profile_prerequisites.py"


def load_helper(root):
    """Load the exact published metadata reader from the declared immutable source."""
    # The only executed project module is the exact published pure metadata reader.
    root = Path(root).absolute()
    path = root / (PREFIX + NAME)
    before = path.lstat()
    if root.resolve(strict=True) != root or path.resolve(strict=True) != path or before.st_uid != os.getuid() \
            or not stat.S_ISREG(before.st_mode):
        raise ValueError("the helper root and source must be exact owned paths")
    if before.st_size > 1024 * 1024:
        raise ValueError("the metadata helper source is oversized")
    raw = path.read_bytes()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    completed = subprocess.run(["git", "-C", str(root), "show", HELPER_COMMIT + ":" + PREFIX + NAME],
        capture_output=True, check=True, timeout=15, env=environment)
    after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns, item.st_mode, item.st_uid, item.st_gid)
    if raw != completed.stdout or completed.stderr or identity(before) != identity(after):
        raise ValueError("the metadata reader differs from its published source")
    specification = importlib.util.spec_from_file_location("common_profile_fact_reader", path)
    helper = importlib.util.module_from_spec(specification)
    exec(compile(raw, str(path), "exec"), helper.__dict__)
    return helper, {"commit": HELPER_COMMIT, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def produce_fact(helper, request, files):
    """Reuse all four accepted-context and original-snapshot checks before making a fact."""
    # Missing contexts fail before any old reader, large result or output is opened.
    helper.require(type(request) is dict and set(request) == {"proposed_plan", "contexts",
        "initial_snapshot_inputs", "native_process_qualification"}, "the fact request has unexpected fields")
    entries = request["contexts"]
    helper.require(type(entries) is list and len(entries) == 4 and all(type(row) is dict
        and set(row) == {"hardware_class", "copies", "review"} and type(row["copies"]) is int for row in entries)
        and {(row["hardware_class"], row["copies"]) for row in entries} == helper.CONTEXTS,
        "all four distinct accepted common performance contexts, including two-copy H100, are required")
    proposed = files.document(request["proposed_plan"])
    helper.require(type(proposed) is dict and proposed.get("source_commit") == helper.SOURCE
        and proposed.get("hardware_class") == "jaguar03_rtx_a4500"
        and type(proposed["configuration"]["learner"]["n_copies"]) is int
        and proposed["configuration"]["learner"]["n_copies"] in (1, 2),
        "the proposed metadata must describe the original one- or two-copy A4500 configuration")
    contexts = [helper.reviewed_context(files, entry, proposed) for entry in entries]
    selected = next(row for row in contexts if row["hardware_class"] == "jaguar03_rtx_a4500"
        and row["copies"] == proposed["configuration"]["learner"]["n_copies"])
    helper.require(selected["plan"] == request["proposed_plan"],
        "the proposed metadata must be the exact accepted A4500 context plan")

    # The accepted native component remains process-only; no native library is loaded.
    native = request["native_process_qualification"]
    helper.require(type(native) is dict and set(native) == {"diagnostic_plan", "diagnostic_result",
        "independent_review", "target_contract"}
        and native["independent_review"]["sha256"] == helper.NATIVE_REVIEW,
        "the actual accepted A4500 process-component references are required")
    documents = {key: files.document(value) for key, value in native.items()}
    review = documents["independent_review"]
    helper.require(review.get("kind") == "independent_configured_native_process_review"
        and review.get("approved") is True and review.get("passed") is True
        and review.get("independent_of_producer") is True
        and review.get("plan") == native["diagnostic_plan"] and review.get("result") == native["diagnostic_result"]
        and review.get("target_contract") == native["target_contract"]
        and review.get("profile_admission_granted") is False,
        "the accepted native process review differs from the exact requested references")

    # Read the already accepted original qualification, never its original raw payloads.
    target = selected["target_qualification_inputs"]
    helper.require(all(helper.equal(row["target_qualification_inputs"], target) for row in contexts
        if row["hardware_class"] == "jaguar03_rtx_a4500"),
        "both accepted A4500 contexts must retain the same exact original target qualification")
    helper.require(type(target) is dict and target["review"]["sha256"] == helper.QUALIFICATION_REVIEW
        and target["result"]["sha256"] == helper.QUALIFICATION_RESULT,
        "the original accepted A4500 target qualification must remain exact")
    target_review = files.document(target["review"])
    helper.require(target_review.get("kind") == "independent_full_original_target_qualification_review"
        and target_review.get("approved") is True and target_review.get("result") == target["result"]
        and target_review.get("actual_terminal_state") == "COMPLETED" and target_review.get("actual_exit_code") == "0:0"
        and target_review.get("admission_granted") is False,
        "the original target qualification lacks its actual independent acceptance")
    historical = next(row["historical_plan"] for row in contexts
        if row["hardware_class"] == "h100" and row["copies"] == 1)
    snapshot = helper.initial_snapshot(files, request["initial_snapshot_inputs"], historical)

    # Before: reviewed context summaries; after: the existing fact's three-key entries.
    fact = dict(helper.PROTOCOL)
    fact.update(campaign_run_dir=proposed["campaign_run_dir"], source_commit=helper.SOURCE,
        contexts=copy.deepcopy(sorted(entries, key=lambda row: (row["hardware_class"], row["copies"]))),
        initial_snapshot_inputs=copy.deepcopy(request["initial_snapshot_inputs"]),
        target_qualification_inputs=copy.deepcopy(target), native_process_qualification=copy.deepcopy(native),
        scientific_credit=0, admission_granted=False, worker_admitted=False)
    return fact, snapshot


def write_once(path, raw):
    """Save one new record durably without replacing an earlier attempt."""
    # An existing output is an earlier attempt, never a path to overwrite.
    path = Path(path).absolute()
    parent = path.parent
    if parent.resolve(strict=True) != parent or parent.stat().st_uid != os.getuid():
        raise ValueError("the output directory must already be an exact owned path")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv=None):
    """Default to no action and write only on an exact owned CPU metadata step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--owned-job-id", required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.execute:
        print(json.dumps({"action_taken": False, "independent_review_created": False, "admission_granted": False}))
        return 0

    # This is metadata computation on one exact owned CPU step, not target validation.
    started = time.monotonic()
    job, step = arguments.owned_job_id, os.environ.get("SLURM_STEP_ID", "")
    cgroup = Path("/proc/self/cgroup").read_text()
    if not re.fullmatch(r"[1-9][0-9]*", job) or os.environ.get("SLURM_JOB_ID") != job \
            or not re.fullmatch(r"[0-9]+", step) or "/job_" + job + "/" not in cgroup \
            or "/step_" + step + "/" not in cgroup:
        raise ValueError("the fact producer requires the declared exact owned CPU job and step")
    outputs = [Path(arguments.proof).absolute(), Path(arguments.audit).absolute()]
    if outputs[0] == outputs[1] or any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("both new output paths must be distinct and unused")
    helper, source = load_helper(arguments.helper_root)
    files = helper.MetadataFiles()
    request_ref = files.reference(arguments.request)
    fact, snapshot = produce_fact(helper, files.document(request_ref), files)
    raw = helper.encoded(fact)
    helper.require(time.monotonic() - started < 285, "the complete fact production exceeded its metadata bound")
    audit = {"schema_version": 1, "kind": "common_profile_prerequisite_fact_production_audit",
        "request": request_ref, "reader_source": source, "producer_source": files.reference(__file__),
        "fact": {"path": str(outputs[0]), "sha256": hashlib.sha256(raw).hexdigest()},
        "actual_pid": os.getpid(), "actual_uid": os.getuid(), "hostname": socket.gethostname(),
        "owned_job_id": job, "owned_step_id": step, "cgroup": cgroup,
        "metadata_files": files.records, "initial_snapshot": snapshot,
        "elapsed_seconds": time.monotonic() - started, "independent_review_created": False,
        "scientific_credit": 0, "admission_granted": False, "worker_admitted": False,
        "GPU_numerical_qualification": False, "current_physical_identity_verified": False}
    # The audit precedes the fact; a failed partial attempt can never be reused.
    write_once(outputs[1], helper.encoded(audit))
    write_once(outputs[0], raw)
    print(json.dumps({"fact": audit["fact"], "independent_review_created": False, "admission_granted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
