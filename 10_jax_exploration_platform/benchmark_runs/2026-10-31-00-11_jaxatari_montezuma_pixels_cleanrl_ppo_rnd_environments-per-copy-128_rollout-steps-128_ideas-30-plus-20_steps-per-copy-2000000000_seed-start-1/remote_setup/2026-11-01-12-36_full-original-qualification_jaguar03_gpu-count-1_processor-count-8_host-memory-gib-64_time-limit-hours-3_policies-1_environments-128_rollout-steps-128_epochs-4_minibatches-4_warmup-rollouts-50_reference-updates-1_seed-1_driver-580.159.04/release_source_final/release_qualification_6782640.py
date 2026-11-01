"""Release only the actual held qualification, then publish its real release inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


PACKET = Path(__file__).absolute().parent
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
REPOSITORY = RUN.parents[2]
JOB = "6782640"
SOURCE = "43eade68238a8d010ad1d1ea681e348022a00060"
INITIAL_PUBLICATION = "4c93083501d4201fc4eea7b830df45daf7a7a5df"
SUBMISSION_SHA = "5dd4d7ab3d9b952e2401373627b371f64da96a857df6ffcb25e54cf36c8ac711"
PLAN_SHA = "f262656ab49a3730f2447fbb797aedc9b5eece8bad7cafbd80bbb60e3c759585"
SCRIPT_SHA = "9fa0eedd0971197c026e61d79a4b6a71dd658a527be1bb8d9169fefd481db306"
PRE_RELEASE_SHA = "ec4170bd0cc03f76d17e65a643830cc8f67b4ff7998cfef92f43122d0b24023a"
HELD_BINDING_SHA = "b0c63df88b5c63dcacae7b38e6f7e4c4aa77c117d5ad3ac062cc9feef63085de"
HELD_PROOF_SHA = "b215111134d0c6e923b69ec43e7b336d7f47490243bb2881b54a35f0497ca0ab"
FROZEN_INPUTS_SHA = 'b71f1641d485af4178ce1186f39441760bd563cf3861d5cb75fe6b23bc843ee5'
COMMAND = ["scontrol", "release", JOB]
NUMERICAL_MODULES = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy"}
BOUNDARY = {"scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
            "packing_point_credit": 0, "throughput_credit": 0, "site_qualification_credit": 0,
            "admission_granted": False, "worker_admitted": False}
ATTEMPT_FILES = ("release_intent.json", "release_receipt.json", "release_preflight_return.json",
    "release_preflight_stdout.raw", "release_preflight_stderr.raw", "release_return.json",
    "release_stdout.raw", "release_stderr.raw", "release_final_inputs_verification.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def contents(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
        and not path.is_symlink(), "an exact ordinary metadata artifact is required")
    raw = path.read_bytes()
    require(len(raw) <= 4 * 1024 * 1024, "release metadata exceeds its bound")
    return raw


def document(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate release metadata key")
            result[key] = value
        return result
    return json.loads(contents(path), object_pairs_hook=unique)


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(contents(path))}


def checked_ref(value):
    require(set(value) == {"path", "sha256"} and ref(value["path"]) == value,
        "an immutable release input changed")
    return document(value["path"])


def put(path, raw, *, atomic=False):
    """Publish new fsynced bytes without replacing any attempted output."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent and not path.exists()
        and not path.is_symlink(), "preserve the previous release attempt")
    temporary = path.with_name("." + path.name + ".pending") if atomic else path
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if atomic:
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def field(raw, name):
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", raw)
    require(len(values) == 1, "missing or repeated owned job field: " + name)
    return values[0]


def validate_held(raw):
    expected = {"JobId": JOB, "UserId": "sl5nw(2618919)", "Account": "sl5nw_base",
        "Partition": "gpu", "QOS": "csresnolim", "JobState": "PENDING", "Reason": "JobHeldUser",
        "ReqNodeList": "jaguar03", "Reservation": "sl5nw_168", "Requeue": "0", "Restarts": "0",
        "NumCPUs": "8", "NumTasks": "1", "CPUs/Task": "8", "MinMemoryNode": "64G",
        "TimeLimit": "03:00:00", "Command": str(PACKET / "job.sbatch"),
        "TresPerNode": "gres/gpu:nvidia_rtx_a4500:1"}
    require(all(field(raw, name) == value for name, value in expected.items())
        and field(raw, "NumNodes") in {"1", "1-1"}, "the actual held qualification identity or request changed")
    pairs = [entry.split("=", 1) for entry in field(raw, "ReqTRES").split(",")]
    require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs),
        "invalid or repeated requested resources")
    values = dict(pairs)
    require(values.get("cpu") == "8" and values.get("mem") == "64G" and values.get("node") == "1"
        and values.get("gres/gpu") == "1" and all(key in {"cpu", "mem", "node", "billing", "gres/gpu", "gres/gpu:nvidia_rtx_a4500"} for key in values)
        and values.get("gres/gpu:nvidia_rtx_a4500", "1") == "1", "the requested single typed A4500 resources changed")


def core(plan):
    path = UNIT / "code/uva_slurm_configured_isolation.py"
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + path.name
    require(sha(contents(path)) == plan["helpers"]["files"][relative], "the exact staged input reader changed")
    specification = importlib.util.spec_from_file_location("qualification_release_owned_input_reader", path)
    module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
    module.validate_plan(plan); module.source_identity(plan)
    return module


def validate_pre_release(plan):
    value = document(UNIT / "configured_isolation_pre_release_inputs.json")
    require(sha(contents(UNIT / "configured_isolation_pre_release_inputs.json")) == PRE_RELEASE_SHA
        and set(value) == {"schema_version", "kind", "job_id", "plan", "ledger", "submissions", "submission", "submission_row"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["kind"] == "uva_configured_isolation_pre_release_inputs" and value["job_id"] == JOB,
        "the actual immutable pre-release input envelope changed")
    require(value["plan"] == ref(UNIT / "measurement_plan.json") and checked_ref(value["plan"]) == plan,
        "the actual held plan changed")
    row = [JOB, plan["kind"], SOURCE, str(PACKET / "job.sbatch")]
    require(value["submission_row"] == row and value["submission"] == ref(PACKET / "submission_receipt.json")
        and value["submission"]["sha256"] == SUBMISSION_SHA, "the actual submission identity changed")
    for key, filename, snapshot_name, member in (("ledger", "submitted_jobids_uva.txt", "configured_isolation_owned_job_ids.txt", JOB),
            ("submissions", "submissions.tsv", "configured_isolation_owned_submissions.tsv", "\t".join(row))):
        record = value[key]
        require(set(record) == {"snapshot", "source_path"} and record["source_path"] == str(RUN / "slurm" / filename)
            and record["snapshot"]["path"] == str(UNIT / snapshot_name) and ref(record["snapshot"]["path"]) == record["snapshot"],
            "the copied exact owned ledger changed")
        for path in (record["snapshot"]["path"], record["source_path"]):
            lines = contents(path).decode("utf-8", errors="strict").splitlines()
            require(lines.count(member) == 1, "the exact owned submission is missing or duplicated")
            require(len(lines) == len(set(lines)) if key == "ledger" else [line for line in lines if line.split("\t", 1)[0] == JOB] == [member],
                "the exact owned ledger contains duplicate or conflicting identity")
    require(contents(RUN / "slurm/submitted_jobids.txt").decode().splitlines().count(JOB) == 1,
        "the actual owned ID is absent or duplicated in the global ledger")
    return value


def validate_proof(proof):
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "owned_uva_configured_isolation_actual_full_held_monitor_proof"
        and proof.get("passed") is True and proof.get("job_id") == JOB
        and proof.get("state") == "PENDING" and proof.get("reason") == "JobHeldUser"
        and proof.get("publication_commit") == INITIAL_PUBLICATION and proof.get("helper_source_commit") == SOURCE,
        "the actual full-monitor held proof is missing or belongs to another job")
    observed = datetime.fromisoformat(proof["at_utc"].replace("Z", "+00:00"))
    require(observed.tzinfo is not None and observed.utcoffset().total_seconds() == 0
        and 0 <= (datetime.now(timezone.utc) - observed).total_seconds() < 1200, "the held full-monitor proof is stale or not UTC")
    require(proof["submission"] == ref(PACKET / "submission_receipt.json") and proof["submission"]["sha256"] == SUBMISSION_SHA
        and proof["plan"] == ref(UNIT / "measurement_plan.json") and proof["plan"]["sha256"] == PLAN_SHA
        and proof["job_script"] == ref(PACKET / "job.sbatch") and proof["job_script"]["sha256"] == SCRIPT_SHA
        and proof["pre_release_inputs"] == ref(UNIT / "configured_isolation_pre_release_inputs.json")
        and proof["pre_release_inputs"]["sha256"] == PRE_RELEASE_SHA and proof["held_binding"] == ref(PACKET / "held_binding.json"),
        "the actual held proof changed its submitted inputs")
    require(all(type(proof[key]) is int and proof[key] == 0 for key in ("scientific_credit", "case_credit", "idea_credit"))
        and proof["admission_granted"] is False and proof["released"] is False
        and proof["frozen_physical_identity_remains_unresolved"] is True,
        "the held proof cannot grant credit,admission or resolved physical identity")
    monitor = proof["monitor"]
    require(monitor["job_id"] == "6781231" and monitor["source_commit"] == "fb3071f3772ed836144f5876dbdda1113a5339d5"
        and type(monitor["cadence_seconds"]) is int and monitor["cadence_seconds"] == 1200
        and monitor["persistent_running_verified"] is True,
        "the exact full monitoring or zero-credit scope changed")
    schedule = proof["recurring_session_task"]
    require(schedule["id"] == "aa2fb837-0978-49e6-ba42-5431efd8a9b6" and schedule["cron"] == "*/20 * * * *"
        and type(schedule["cadence_seconds"]) is int and schedule["cadence_seconds"] == 1200
        and schedule["exact_prompt_and_schedule_verified_before_old_deleted"] is True,
        "the exact recurring monitor verification changed")
    verification = checked_ref(monitor["full_verification"])
    status = checked_ref(monitor["full_status"])
    require(ref(monitor["full_report"]["path"]) == monitor["full_report"]
        and verification["kind"] == "owned_full_scheduled_monitor_check" and verification["passed"] is True
        and type(verification["returncode"]) is int and verification["returncode"] == 0
        and verification["monitor_job_id"] == monitor["job_id"] and verification["step_id"] == monitor["step_id"]
        and verification["source_commit"] == monitor["source_commit"]
        and verification["report_sha256"] == monitor["full_report"]["sha256"]
        and verification["status_sha256"] == monitor["full_status"]["sha256"]
        and [row for row in status["jobs"] if row[0] == JOB] == [monitor["row"]]
        and monitor["row"] == [JOB, "PENDING", "", "gres/gpu:nvidia_rtx_a4500:1", "(JobHeldUser)"],
        "the actual full report does not contain this exact held qualification")
    query = checked_ref(proof["owned_query_return"])
    require(type(query.get("schema_version")) is int and query["schema_version"] == 1
        and query.get("kind") == "actual_owned_qualification_held_scheduler_query" and query.get("job_id") == JOB
        and query["command"] == ["scontrol", "show", "job", JOB, "--oneliner"]
        and type(query["returncode"]) is int and query["returncode"] == 0,
        "the saved exact owned held query changed")
    require(ref(proof["owned_query_stdout"]["path"]) == proof["owned_query_stdout"]
        and ref(proof["owned_query_stderr"]["path"]) == proof["owned_query_stderr"]
        and query["stdout_sha256"] == proof["owned_query_stdout"]["sha256"]
        and query["stderr_sha256"] == proof["owned_query_stderr"]["sha256"]
        and contents(proof["owned_query_stderr"]["path"]) == b"", "the saved owned query bytes changed")
    validate_held(contents(proof["owned_query_stdout"]["path"]).decode("utf-8", errors="strict"))


def checked(publication=None):
    """Default checks are local metadata reads, with no scheduler command or intent."""
    require(socket.gethostname().split(".")[0] == "portal05" and os.getuid() == 2618919
        and not NUMERICAL_MODULES.intersection(sys.modules), "wrong release host,owner or numerical import")
    frozen_path = PACKET / "release_frozen_inputs_6782640.json"
    require(FROZEN_INPUTS_SHA is not None and sha(contents(frozen_path)) == FROZEN_INPUTS_SHA,
        "the reviewed actual release input snapshot is not final")
    frozen = document(frozen_path)
    require(frozen["job_id"] == JOB and frozen["source_commit"] == SOURCE and frozen["packet"] == str(PACKET)
        and frozen["data_unit"] == str(UNIT), "the fixed release scope changed")
    for path, expected in frozen["files"].items():
        raw = contents(path)
        require(sha(raw) == expected["sha256"] and len(raw) == expected["bytes"], "an immutable release prerequisite changed: " + path)
    plan = document(UNIT / "measurement_plan.json")
    require(sha(contents(UNIT / "measurement_plan.json")) == PLAN_SHA and plan["helpers"]["source_commit"] == SOURCE
        and plan["gpu_uuid"] is None and plan["configured_isolation"]["physical_identity"]["gpu_uuid"] is None,
        "the original unresolved plan changed")
    identity = core(plan)
    pre = validate_pre_release(plan)
    submission = checked_ref(pre["submission"])
    require(submission["kind"] == "owned_uva_configured_isolation_submission" and submission["cluster"] == "uva"
        and submission["job_id"] == JOB and type(submission["returncode"]) is int and submission["returncode"] == 0
        and submission["plan_sha256"] == PLAN_SHA and submission["script_sha256"] == SCRIPT_SHA
        and submission["data_unit"] == str(UNIT) and submission["packet"] == str(PACKET)
        and submission["submission_row"] == pre["submission_row"] and submission["publication_commit"] == INITIAL_PUBLICATION
        and submission["ledgers_and_submission_row_fsynced_before_this_receipt"] is True and submission["released"] is False,
        "the actual held submission chain changed")
    require(submission["held_command"] == document(PACKET / "root_pre_submission_review.json")["held_command"]
        and "--hold" in submission["held_command"] and "--no-requeue" in submission["held_command"]
        and submission["held_command"][-1] == str(PACKET / "job.sbatch") and sha(contents(PACKET / "job.sbatch")) == SCRIPT_SHA,
        "the exact original held request changed")
    require(sha(contents(PACKET / "held_binding.json")) == HELD_BINDING_SHA,
        "the actual pre-release binding changed")
    binding = document(PACKET / "held_binding.json")
    require(binding["job_id"] == JOB and binding["submission_receipt"] == pre["submission"]
        and binding["plan"] == pre["plan"] and binding["pre_release_inputs"] == ref(UNIT / "configured_isolation_pre_release_inputs.json")
        and binding["copied_ledgers"] == {key: pre[key] for key in ("ledger", "submissions")}
        and binding["no_frozen_plan_change"] is True and binding["physical_gpu_identity_unresolved"] is True,
        "the held ID is not bound to the original immutable inputs")
    require(sha(contents(PACKET / "actual_held_monitor_proof.json")) == HELD_PROOF_SHA, "the exact held-monitor proof changed")
    validate_proof(document(PACKET / "actual_held_monitor_proof.json"))
    review = document(PACKET / "release_implementation_review_6782640.json")
    require(review["job_id"] == JOB and review["source_sha256"] == sha(contents(Path(__file__).absolute()))
        and review["release_executed"] is False and review["maximum_startup_wait_seconds"] == 180,
        "the reviewed release implementation changed")
    for name in ATTEMPT_FILES:
        require(not (PACKET / name).exists() and not (PACKET / name).is_symlink(), "a release attempt already exists; never retry")
    final = UNIT / "owned_allocation_inputs.json"
    require(not final.exists() and not final.is_symlink() and not final.with_name("." + final.name + ".pending").exists(),
        "retain the previous final input publication attempt")
    if publication is not None:
        require(re.fullmatch(r"[0-9a-f]{40}", publication), "the full published release commit is required")
        subprocess.run(["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", publication,
            "refs/remotes/origin/merge-atari-cleanrl"], check=True)
        paths = [Path(path) for path in frozen["files"]] + [Path(__file__).absolute(), frozen_path,
            PACKET / "release_implementation_review_6782640.json"]
        for path in paths:
            require(path.is_relative_to(REPOSITORY) and contents(path) == subprocess.check_output([
                "git", "-C", str(REPOSITORY), "cat-file", "blob", publication + ":" + str(path.relative_to(REPOSITORY))]),
                "an exact release prerequisite was not published: " + str(path))
    return {"schema_version": 1, "kind": "owned_uva_full_original_qualification_release_intent", "cluster": "uva",
        "job_id": JOB, "command": COMMAND, "at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_commit": publication, "submission_receipt_sha256": SUBMISSION_SHA, "plan_sha256": PLAN_SHA,
        "data_unit": str(UNIT), "pre_release_inputs_sha256": PRE_RELEASE_SHA,
        "frozen_inputs_sha256": FROZEN_INPUTS_SHA, "held_monitor_review": ref(PACKET / "actual_held_monitor_proof.json"),
        "source_sha256": review["source_sha256"], "source_commit": SOURCE,
        "automatic_retry_permitted": False, **BOUNDARY}


def capture(label, command):
    """Retain the exact owned command result before any parse or state check."""
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
    except BaseException as error:
        put(PACKET / (label + "_stdout.raw"), getattr(error, "stdout", None) or b"")
        put(PACKET / (label + "_stderr.raw"), getattr(error, "stderr", None) or b"")
        put(PACKET / (label + "_return.json"), encoded({"command": command, "returncode": None,
            "error_type": type(error).__name__, "error": str(error), "automatic_retry_permitted": False}))
        raise
    put(PACKET / (label + "_stdout.raw"), result.stdout)
    put(PACKET / (label + "_stderr.raw"), result.stderr)
    put(PACKET / (label + "_return.json"), encoded({"command": command,
        "returncode": result.returncode, "automatic_retry_permitted": False}))
    require(result.returncode == 0 and result.stderr == b"", "owned scheduler command failed; preserve evidence and never retry")
    return result.stdout


def release_once(intent):
    require(intent["publication_commit"] is not None, "published exact release inputs are required before action")
    put(PACKET / "release_intent.json", encoded(intent))
    before = capture("release_preflight", ["scontrol", "show", "job", "-o", JOB])
    validate_held(before.decode("utf-8", errors="strict"))
    stdout = capture("release", COMMAND)
    release = {"schema_version": 1, "kind": "owned_uva_configured_isolation_release", "cluster": "uva",
        "job_id": JOB, "command": COMMAND, "returncode": 0, "stdout": stdout.decode("utf-8", errors="strict"),
        "stderr": "", "at_utc": datetime.now(timezone.utc).isoformat(), "data_unit": str(UNIT),
        "plan_sha256": PLAN_SHA, "submission_receipt_sha256": SUBMISSION_SHA,
        "publication_commit": intent["publication_commit"], "release_source_sha256": intent["source_sha256"],
        "held_monitor_review": intent["held_monitor_review"], "frozen_inputs_sha256": FROZEN_INPUTS_SHA,
        "automatic_retry_permitted": False, **BOUNDARY}
    put(PACKET / "release_receipt.json", encoded(release))
    pre = document(UNIT / "configured_isolation_pre_release_inputs.json")
    require(sha(contents(UNIT / "configured_isolation_pre_release_inputs.json")) == PRE_RELEASE_SHA,
        "the immutable pre-release envelope changed during release")
    final = {**pre, "kind": "uva_configured_isolation_owned_inputs", "release": ref(PACKET / "release_receipt.json")}
    put(UNIT / "owned_allocation_inputs.json", encoded(final), atomic=True)
    identity = core(document(UNIT / "measurement_plan.json"))
    require(identity.owned_inputs(document(UNIT / "measurement_plan.json")) == final,
        "the actual successful release chain failed the exact core reader")
    verification = {"schema_version": 1, "kind": "actual_uva_release_final_inputs_verification", "job_id": JOB,
        "release": ref(PACKET / "release_receipt.json"), "final_inputs": ref(UNIT / "owned_allocation_inputs.json"),
        "actual_core_reader_passed": True, "frozen_plan_sha256": PLAN_SHA, "maximum_startup_wait_seconds": 180,
        "at_utc": datetime.now(timezone.utc).isoformat(), **BOUNDARY}
    put(PACKET / "release_final_inputs_verification.json", encoded(verification))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-commit")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args()
    require(not arguments.release or arguments.publication_commit is not None,
        "publication is required before the one release")
    intent = checked(arguments.publication_commit)
    print(json.dumps(release_once(intent) if arguments.release else {"validated": True, "job_id": JOB,
        "command": COMMAND, "scheduler_queries": 0, "release_calls": 0, "final_inputs_written": False}))


if __name__ == "__main__":
    main()
