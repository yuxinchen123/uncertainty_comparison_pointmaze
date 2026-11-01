"""Adopt one returned held ACES ID locally; never contact a cluster or release a job."""

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


PLAN_SHA = "14797308244bd53c70da15d4c1284820e92e2105d9cf7106e86e20bdf09ad58f"
DRIVER_SHA = "be81fb18909a933592e5cdb6824698b5e94439f19495f053305c0505d2e1e64b"
BOUNDARY = {"workers_permitted": False, "campaign_claims_permitted": False,
    "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 0, "admission_credit": 0}
ARTIFACTS = {"submission_receipt.json", "held_submission_intent.json", "held_submission_stdout.raw",
    "held_submission_stderr.raw", "held_submission_return.json", "submitted_jobids_aces.txt", "submissions_aces.tsv"}
MAX_BYTES = 2 * 1024 * 1024


def require(condition, message):
    """Reject missing or contradictory local evidence before writing anything."""
    if not condition:
        raise ValueError(message)


def ordinary(path, directory=False):
    """Keep metadata and ledger paths at their exact ordinary locations."""
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True)
        and (path.is_dir() if directory else path.is_file()), "an exact ordinary adoption path is required")
    return path


def read(path):
    """Bound every local metadata read without importing packages or querying anything."""
    with ordinary(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "returned held metadata exceeds its explicit bound")
    return raw


def digest(raw):
    """Identify original bytes without reserializing remote JSON."""
    return hashlib.sha256(raw).hexdigest()


def boundary(value):
    """Reject Boolean credit substitutes and any admission permission."""
    require(all(type(value.get(key)) is type(expected) and value[key] == expected for key, expected in BOUNDARY.items()),
        "held submission cannot grant admission or scientific credit")


def fsync_directory(path):
    """Persist metadata names before the next adoption step."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, raw):
    """Publish complete exact bytes atomically without replacing a prior record."""
    path = Path(path)
    ordinary(path.parent, True)
    descriptor, temporary = tempfile.mkstemp(prefix=".submission-adopt-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        fsync_directory(path.parent)


def json_bytes(value):
    """Write new local operational evidence with literal JSON types."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def validate_returned(plan, received):
    """Verify all seven original returned artifacts against the preserved wrapper stdout."""
    received = ordinary(received, True)
    require(received == Path(plan["receipt_directory"]) / "received", "adopt only this wrapper's exact returned directory")
    for item in plan["local_files"]:
        raw = read(item["path"])
        require(len(raw) == item["bytes"] and digest(raw) == item["sha256"], "an original prepared local input changed before adoption")
    transport = json.loads(read(received.parent / "transport_result.json"))
    require(type(transport.get("returncode")) is int and transport["returncode"] == 0,
        "the held submission wrapper did not return successfully")
    raw_stdout = read(received.parent / "stdout.raw")
    require(transport["stdout"] == {"path": str(received.parent / "stdout.raw"), "sha256": digest(raw_stdout), "bytes": len(raw_stdout)},
        "the wrapper's original raw stdout changed")
    raw_stderr = read(received.parent / "stderr.raw")
    require(transport["stderr"] == {"path": str(received.parent / "stderr.raw"), "sha256": digest(raw_stderr), "bytes": len(raw_stderr)},
        "the wrapper's original raw stderr changed")
    response = json.loads(raw_stdout)
    require(set(response) == {"receipt", "files"} and len(response["files"]) == len(ARTIFACTS)
        and {item["name"] for item in response["files"]} == ARTIFACTS, "the exact seven native returned artifacts are required")
    values, hashes = {}, {}
    for item in response["files"]:
        name = item["name"]
        raw = base64.b64decode(item["contents_base64"], validate=True)
        parent = Path(plan["native_run"]) / "slurm" if name in {"submitted_jobids_aces.txt", "submissions_aces.tsv"} else Path(plan["native_packet"])
        require(item["path"] == str(parent / name) and type(item["bytes"]) is int and item["bytes"] == len(raw)
            and item["sha256"] == digest(raw) and read(received / name) == raw,
            "a native returned artifact changed its path, bytes or hash")
        values[name], hashes[name] = raw, digest(raw)
    receipt = json.loads(values["submission_receipt.json"])
    require(receipt == response["receipt"], "the native receipt differs from its returned original bytes")
    boundary(receipt)
    identifier = receipt.get("job_id")
    require(isinstance(identifier, str) and re.fullmatch(r"[1-9][0-9]*", identifier), "the held job ID is invalid")
    row = [identifier, Path(plan["native_packet"]).name, plan["helpers"]["source_commit"], plan["native_run"]]
    require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
        and receipt.get("kind") == "owned_aces_full_site_qualification_submission" and receipt.get("cluster") == "aces"
        and receipt.get("held_command") == plan["held_command"] and receipt.get("submission_row") == row
        and receipt.get("ledger_path") == str(Path(plan["native_run"]) / "slurm/submitted_jobids_aces.txt")
        and receipt.get("data_unit") == plan["data_unit"] and receipt.get("submission_plan_sha256") == PLAN_SHA
        and receipt.get("submission_driver_sha256") == DRIVER_SHA
        and isinstance(receipt.get("published_preparation_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", receipt["published_preparation_commit"])
        and receipt.get("automatic_retry_permitted") is False
        and all(receipt.get(key) == plan[key] for key in ("plan_sha256", "manifest_sha256", "job_script_sha256")),
        "the native submission receipt identifies another packet or wrapper")
    require(values["held_submission_stdout.raw"] in ((identifier + "\n").encode(), identifier.encode())
        and values["held_submission_stdout.raw"] == receipt["stdout"].encode()
        and values["held_submission_stderr.raw"] == receipt["stderr"].encode(), "native submission stdout/stderr changed")
    intent = json.loads(values["held_submission_intent.json"])
    boundary(intent)
    require(type(intent.get("schema_version")) is int and intent["schema_version"] == 1
        and intent.get("kind") == "aces_full_site_held_submission_intent"
        and intent.get("driver_sha256") == DRIVER_SHA and intent.get("submission_plan_sha256") == PLAN_SHA
        and intent.get("published_preparation_commit") == receipt["published_preparation_commit"]
        and intent.get("held_command") == plan["held_command"] and intent.get("automatic_retry_permitted") is False,
        "the original native intent identifies another submission")
    returned = json.loads(values["held_submission_return.json"])
    require(type(returned.get("returncode")) is int and returned["returncode"] == 0
        and returned.get("automatic_retry_permitted") is False, "the native held submission did not succeed")
    for suffix in ("stdout", "stderr"):
        name = "held_submission_" + suffix + ".raw"
        require(returned[suffix] == {"path": str(Path(plan["native_packet"]) / name), "sha256": hashes[name], "bytes": len(values[name])},
            "the native durable raw return receipt changed")
    require(values["submitted_jobids_aces.txt"].decode().splitlines().count(identifier) == 1
        and [line for line in values["submissions_aces.tsv"].decode().splitlines() if line.split("\t", 1)[0] == identifier] == ["\t".join(row)],
        "the exact held ID and row are not durable in both returned native ACES ledgers")
    local_intent_raw = read(Path(plan["local_packet"]) / "held_submission_intent.json")
    local_intent = json.loads(local_intent_raw)
    boundary(local_intent)
    require(local_intent.get("kind") == "aces_full_site_held_submission_local_intent"
        and local_intent.get("driver_sha256") == DRIVER_SHA and local_intent.get("submission_plan_sha256") == PLAN_SHA
        and local_intent.get("native_packet") == plan["native_packet"]
        and local_intent.get("held_command") == plan["held_command"]
        and local_intent.get("published_preparation_commit") == receipt["published_preparation_commit"]
        and local_intent_raw == read(received.parent / "local_intent.json"), "the local wrapper intent changed")
    return receipt, row, values, hashes, local_intent_raw


def adopt(plan, received):
    """Fsync exactly the new local ID and row before publishing the exact native receipt."""
    receipt, row, values, hashes, original_intent = validate_returned(plan, received)
    packet, run = ordinary(plan["local_packet"], True), ordinary(plan["local_campaign_run_dir"], True)
    names = ("submission_receipt.json", "remote_held_submission_intent.json", "submission_adoption_intent.json",
        "submission_adoption_receipt.json", "submission_adoption_failure.json")
    require(not any((packet / name).exists() or (packet / name).is_symlink() for name in names),
        "adoption was already attempted or a receipt already exists; automatic retry is forbidden")
    streams = []
    stage = "validation"
    try:
        for name in ("submitted_jobids_aces.txt", "submissions_aces.tsv"):
            path = ordinary(run / "slurm" / name)
            stream = os.fdopen(os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW), "r+", encoding="utf-8")
            streams.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = []
        for stream in streams:
            stream.seek(0); raw = stream.read()
            require(not raw or raw.endswith("\n"), "a local ACES ledger has an incomplete final line")
            previous.append(raw.splitlines())
        require(all(re.fullmatch(r"[1-9][0-9]*", item) for item in previous[0])
            and len(previous[0]) == len(set(previous[0]))
            and all(re.fullmatch(r"[1-9][0-9]*", line.split("\t", 1)[0]) for line in previous[1])
            and len(previous[1]) == len({line.split("\t", 1)[0] for line in previous[1]}), "the local ACES ledgers contain invalid identifiers")
        identifier = receipt["job_id"]
        require(identifier not in previous[0] and not any(line.split("\t", 1)[0] == identifier for line in previous[1]),
            "the local held ID or submission row already exists; do not duplicate or infer recovery")
        stage = "intent"
        publish(packet / "submission_adoption_intent.json", json_bytes({"schema_version": 1,
            "kind": "aces_full_site_local_adoption_intent", "at_utc": datetime.now(timezone.utc).isoformat(),
            "job_id": identifier, "submission_row": row, "native_artifact_sha256": hashes,
            "wrapper_plan_sha256": PLAN_SHA, "wrapper_source_sha256": DRIVER_SHA,
            "automatic_retry_permitted": False, **BOUNDARY}))
        for name, stream, line in zip(("submitted_jobids_aces.txt", "submissions_aces.tsv"), streams, (identifier, "\t".join(row))):
            stage = "append " + name
            stream.write(line + "\n"); stream.flush(); os.fsync(stream.fileno())
        fsync_directory(run / "slurm")
        stage = "publish native intent"
        publish(packet / "remote_held_submission_intent.json", values["held_submission_intent.json"])
        stage = "publish native receipt"
        publish(packet / "submission_receipt.json", values["submission_receipt.json"])
        require(read(packet / "held_submission_intent.json") == original_intent, "the original local wrapper intent changed during adoption")
        stage = "complete"
        proof = {"schema_version": 1, "kind": "aces_full_site_local_adoption_receipt", "passed": True,
            "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": identifier, "submission_row": row,
            "native_artifact_sha256": hashes, "local_ledgers_fsynced_before_receipt": True,
            "native_receipt_bytes_preserved": True, "local_wrapper_intent_unchanged": True,
            "submission_or_release_performed": False, **BOUNDARY}
        publish(packet / "submission_adoption_receipt.json", json_bytes(proof))
        return proof
    except Exception as error:
        if (packet / "submission_adoption_intent.json").exists() and not (packet / "submission_adoption_failure.json").exists():
            publish(packet / "submission_adoption_failure.json", json_bytes({"kind": "aces_full_site_local_adoption_failure",
                "stage": stage, "job_id": receipt["job_id"], "error_type": type(error).__name__, "error": str(error),
                "automatic_retry_permitted": False}))
        raise
    finally:
        for stream in reversed(streams):
            stream.close()


def main():
    """Validate the frozen wrapper plan; adoption requires the explicit root action flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--adopt", action="store_true")
    arguments = parser.parse_args()
    raw = read(arguments.plan)
    require(digest(raw) == PLAN_SHA, "the exact reviewed wrapper plan is required")
    plan = json.loads(raw)
    if arguments.adopt:
        print(json.dumps(adopt(plan, Path(plan["receipt_directory"]) / "received")))
    else:
        print(json.dumps({"wrapper_plan_sha256": PLAN_SHA, "wrapper_source_sha256": DRIVER_SHA,
            "received_directory": str(Path(plan["receipt_directory"]) / "received"), "local_writes": 0,
            "network_calls": 0, "submissions": 0, "release_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
