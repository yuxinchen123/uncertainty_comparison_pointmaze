"""Submit one diagnostic held, then release its recorded owned identifier exactly once."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

PACKET = Path(__file__).resolve(strict=True).parent
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
INTEGRATION = RUN.parents[2]
KIND = "nsight_process_identity_diagnostic"
OWNER_UID = 2618919


def require(value, message):
    """Refuse unsupported or previously attempted actions without a fallback."""
    if not value:
        raise ValueError(message)


def now():
    """Record an actual UTC timestamp rather than deriving one from a filename."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ordinary_bytes(path):
    """Read one exact ordinary owned file through a pinned descriptor."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "an exact ordinary artifact path is required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        status = os.fstat(descriptor)
        require(stat.S_ISREG(status.st_mode) and status.st_uid == OWNER_UID and status.st_size <= 4 * 1024 * 1024,
                "the launch artifact is not an owned bounded ordinary file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read()
        require(len(raw) == status.st_size, "the launch artifact changed during its read")
        return raw
    finally:
        os.close(descriptor)


def reference(path):
    """Return exactly the two fields required by the unchanged diagnostic runtime."""
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(ordinary_bytes(path)).hexdigest()}


def document(path):
    """Reject duplicate JSON keys rather than accepting an ambiguous launch record."""
    def unique(pairs):
        """Retain exactly one value per record field."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate launch-record field")
            result[key] = value
        return result
    return json.loads(ordinary_bytes(path), object_pairs_hook=unique)


def new_path(path):
    """Preserve every completed, partial or dangling earlier publication attempt."""
    path = Path(path)
    require(not os.path.lexists(path) and not os.path.lexists(path.with_name(path.name + ".prepared")),
            "preserve the previous attempt: " + path.name)


def publish(path, value):
    """Publish a complete artifact atomically without overwriting an earlier one."""
    path = Path(path)
    new_path(path)
    require(path.parent.resolve(strict=True) == path.parent, "the artifact parent must be exact")
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".prepared")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
        temporary.unlink()
        os.fsync(directory)
    finally:
        os.close(directory)
    return reference(path)


def checked_packet():
    """Validate the complete original packet and unchanged one-card diagnostic shape."""
    original = document(PACKET / "immutable_artifacts.json")["files"]
    require(len(original) == 17, "the complete seventeen-file prepared diagnostic is required")
    for name, expected in original.items():
        path = Path(name)
        require(path.is_relative_to(PACKET) or path.is_relative_to(UNIT), "an initial artifact escaped its packet")
        raw = ordinary_bytes(path)
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"]
                and oct(path.stat().st_mode & 0o777) == expected["mode"], "a frozen diagnostic artifact changed")
    plan = document(UNIT / "measurement_plan.json")
    require(plan["kind"] == KIND and plan["packet"] == str(PACKET) and plan["campaign_run_dir"] == str(RUN)
            and plan["unit_id"] == PACKET.name and plan["job_id"] is None and plan["gpu_uuid"] is None,
            "the plan is not this exact unbound diagnostic")
    require(plan["resources"] == {"partition": "gpu", "account": "sl5nw_base", "qos": "normal", "reservation": None,
        "node": "jaguar03", "nodes": 1, "tasks": 1, "requested_cpus": 1, "host_memory_gib": 2,
        "gres": "gpu:nvidia_rtx_a4500:1", "wall_seconds": 300, "requeue": False}, "the approved diagnostic shape changed")
    command = document(PACKET / "held_submission_command.json")
    require(command == ["sbatch", "--hold", "--parsable", "--no-requeue", "--partition=gpu", "--account=sl5nw_base",
        "--qos=normal", "--nodelist=jaguar03", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G",
        "--gres=gpu:nvidia_rtx_a4500:1", "--time=00:05:00", str(PACKET / "job.sbatch")],
        "the exact held unreserved submission command changed")
    return plan, command


def check_publication(commit, release=False):
    """Require independently reviewed exact source and packet bytes in the actual Git commit."""
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "the real full publication commit is required")
    freeze_path = PACKET / "launcher_source_freeze.json"
    freeze = document(freeze_path)
    review_path = PACKET / "launcher_independent_review.json"
    review = document(review_path)
    require(review.get("passed") is True and review.get("source_freeze") == reference(freeze_path),
            "the exact launcher source has no independent approval")
    paths = dict(freeze["files"])
    initial = document(PACKET / "immutable_artifacts.json")["files"]
    mandatory = set(initial) | {str(PACKET / "immutable_artifacts.json"), str(PACKET / "packet_static_verification.json"),
                               str(PACKET / "launch_diagnostic_once.py"), str(PACKET / "test_launch_diagnostic_once.py")}
    require(mandatory <= set(paths), "the publication freeze omitted a mandatory source or prepared artifact")
    paths.update({str(freeze_path): reference(freeze_path)["sha256"], str(review_path): reference(review_path)["sha256"]})
    if release:
        binding_path = PACKET / "release_binding.json"
        release_review_path = PACKET / "release_independent_review.json"
        release_review = document(release_review_path)
        require(release_review.get("passed") is True and release_review.get("binding") == reference(binding_path),
                "the actual fixed-identifier release has no independent approval")
        for path in (binding_path, release_review_path, PACKET / "submission_receipt.json", PACKET / "submission_ledgers_fsynced.json"):
            paths[str(path)] = reference(path)["sha256"]
    for name, expected in paths.items():
        path = Path(name)
        require(path.is_relative_to(INTEGRATION) and reference(path)["sha256"] == expected,
                "a reviewed source or packet input changed")
        raw = subprocess.check_output(["git", "-C", str(INTEGRATION), "cat-file", "blob", commit + ":" + str(path.relative_to(INTEGRATION))])
        require(hashlib.sha256(raw).hexdigest() == expected, "a reviewed input is not the published Git byte sequence")


def actor():
    """Require the campaign owner's actual user identity before any scheduler call."""
    require(os.getuid() == os.geteuid() == OWNER_UID, "only the campaign owner may act on this packet")


def scheduler(command, stem):
    """Retain a single actual native command return, with no automatic retry."""
    actor()
    for suffix in ("_stdout.raw", "_stderr.raw", "_return.json"):
        new_path(PACKET / (stem + suffix))
    # Before: SBATCH_RESERVATION could override an omitted option; after: only recorded arguments and headers apply.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    environment["TZ"] = "UTC"
    started = now()
    try:
        result = subprocess.run(command, capture_output=True, env=environment, timeout=30)
    except subprocess.TimeoutExpired as error:
        publish(PACKET / (stem + "_stdout.raw"), error.stdout or b"")
        publish(PACKET / (stem + "_stderr.raw"), error.stderr or b"")
        publish(PACKET / (stem + "_return.json"), {"command": command, "started_at_utc": started,
            "finished_at_utc": now(), "returncode": None, "timed_out": True, "automatic_retry_permitted": False})
        raise
    publish(PACKET / (stem + "_stdout.raw"), result.stdout)
    publish(PACKET / (stem + "_stderr.raw"), result.stderr)
    publish(PACKET / (stem + "_return.json"), {"command": command, "started_at_utc": started,
        "finished_at_utc": now(), "returncode": result.returncode, "automatic_retry_permitted": False,
        "inherited_SBATCH_variables_removed": sorted(key for key in os.environ if key.startswith("SBATCH_"))})
    require(type(result.returncode) is int and result.returncode == 0, "the actual native command failed; preserve this attempt")
    return result


def identifier(value):
    """Accept only a real positive decimal identifier, never the CPU monitor."""
    require(type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) and value != "6783003",
            "the actual diagnostic identifier is invalid")
    return value


def submission_row(job, plan):
    """Construct the exact diagnostic row required by its unchanged owned-allocation reader."""
    return [identifier(job), KIND, plan["helpers"]["source_commit"], str(PACKET / "job.sbatch")]


def append_member(path, member):
    """Append and fsync one new exact ledger member through an owned pinned descriptor."""
    path = Path(path)
    before = ordinary_bytes(path)
    require(member not in before.decode().splitlines(), "the returned identifier or exact row is already recorded")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        status = os.fstat(descriptor)
        require(status.st_uid == OWNER_UID and stat.S_ISREG(status.st_mode), "the ledger is not an owned ordinary file")
        with os.fdopen(descriptor, "a", closefd=False) as stream:
            stream.write(member + "\n"); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    require(ordinary_bytes(path) == before + (member + "\n").encode(), "a ledger changed during its single append")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return reference(path)


def owned_submission(plan):
    """Validate an actual successful held receipt against all three current own ledgers."""
    value = document(PACKET / "submission_receipt.json")
    job = identifier(value["job_id"])
    row = submission_row(job, plan)
    require(value["kind"] == "owned_nsight_process_diagnostic_submission" and type(value["returncode"]) is int
            and value["returncode"] == 0 and value["plan_sha256"] == reference(UNIT / "measurement_plan.json")["sha256"]
            and value["data_unit"] == str(UNIT) and value["submission_row"] == row, "the actual held submission receipt changed")
    for path, member in ((RUN / "slurm/submitted_jobids.txt", job), (RUN / "slurm/submitted_jobids_uva.txt", job),
                         (RUN / "slurm/submissions.tsv", "\t".join(row))):
        require(ordinary_bytes(path).decode().splitlines().count(member) == 1, "the diagnostic lacks a unique owned ledger member")
    rows = [line.split("\t") for line in ordinary_bytes(RUN / "slurm/submissions.tsv").decode().splitlines()
            if line.split("\t")[0] == job]
    require(rows == [row], "a different command is recorded for this actual identifier")
    require(document(PACKET / "submission_ledgers_fsynced.json")["submission_row"] == row,
            "the actual three-ledger synchronization record is missing")
    return value


def submit_held_once(commit):
    """Run one held submission, then durably record its real identifier before any release."""
    plan, command = checked_packet()
    check_publication(commit)
    for name in ("submission_intent.json", "submission_receipt.json", "submission_ledgers_fsynced.json"):
        new_path(PACKET / name)
    actor()
    lock = RUN / "queue/metadata.lock"
    with lock.open("r+") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        publish(PACKET / "submission_intent.json", {"kind": "one_nsight_diagnostic_held_submission_attempt",
            "at_utc": now(), "publication": commit, "command": command, "plan": reference(UNIT / "measurement_plan.json"),
            "automatic_retry_permitted": False})
        actual = scheduler(command, "actual_submission")
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", actual.stdout.decode().strip())
        require(parsed is not None, "the actual parsable submission output has no unique identifier")
        job = identifier(parsed.group(1)); row = submission_row(job, plan)
        ledgers = [append_member(RUN / "slurm/submitted_jobids.txt", job),
                   append_member(RUN / "slurm/submitted_jobids_uva.txt", job),
                   append_member(RUN / "slurm/submissions.tsv", "\t".join(row))]
        publish(PACKET / "submission_ledgers_fsynced.json", {"kind": "actual_diagnostic_three_ledgers_fsynced",
            "job_id": job, "submission_row": row, "ledgers": ledgers, "at_utc": now()})
        receipt = {"kind": "owned_nsight_process_diagnostic_submission", "job_id": job, "returncode": 0,
            "plan_sha256": reference(UNIT / "measurement_plan.json")["sha256"], "data_unit": str(UNIT),
            "submission_row": row, "submitted_held": True, "publication": commit, "at_utc": now()}
        publish(PACKET / "submission_receipt.json", receipt)
        job_directory = RUN / "slurm/jobs" / job
        require(not os.path.lexists(job_directory), "preserve the earlier per-job directory")
        job_directory.mkdir()
        for name in ("job.sbatch", "manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
            publish(job_directory / name, ordinary_bytes(PACKET / name))
        publish(job_directory / "assignment.json", {"job_id": job, "unit": str(UNIT), "submission_row": row,
            "state": "held", "actual_gpu_uuid": None, "submission": reference(PACKET / "submission_receipt.json")})
    return receipt


def query_held(job, stem):
    """Query only the already ledger-owned identifier and verify its exact held resource request."""
    job = identifier(job)
    result = scheduler(["scontrol", "show", "job", "-o", job], stem)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)", result.stdout.decode()))
    require(fields.get("JobId") == job and fields.get("UserId") == "sl5nw(" + str(OWNER_UID) + ")"
            and fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
            and fields.get("Requeue") == "0" and fields.get("Restarts") == "0"
            and fields.get("TimeLimit") == "00:05:00" and fields.get("Partition") == "gpu"
            and fields.get("Account") == "sl5nw_base" and fields.get("QOS") == "normal"
            and fields.get("ReqNodeList") == "jaguar03" and fields.get("NumNodes") in {"1", "1-1"}
            and fields.get("CPUs/Task") == "1" and fields.get("NumTasks") == "1"
            and fields.get("Command") == str(PACKET / "job.sbatch")
            and fields.get("Reservation", "") in {"", "(null)", "None"}, "the actual held diagnostic shape or owner changed")
    resources = dict(item.split("=", 1) for item in fields["ReqTRES"].split(","))
    # One requested processor may occupy both logical threads on this documented two-thread-per-core node.
    require(resources.get("cpu") in {"1", "2"} and fields.get("NumCPUs") in {"1", "2"}
            and resources.get("mem") == "2G" and resources.get("gres/gpu") == "1"
            and fields.get("TresPerNode") == "gres/gpu:nvidia_rtx_a4500:1", "the actual requested resources changed")
    return reference(PACKET / (stem + "_return.json"))


def prepare_release(commit):
    """Bind a release command only after the held job and actual durable ledgers exist."""
    plan, unused_command = checked_packet()
    check_publication(commit)
    new_path(PACKET / "release_binding.json")
    actual = owned_submission(plan)
    query = query_held(actual["job_id"], "release_preparation_query")
    binding = {"kind": "actual_owned_nsight_diagnostic_fixed_identifier_release_binding", "job_id": actual["job_id"],
        "command": ["scontrol", "release", actual["job_id"]], "submission": reference(PACKET / "submission_receipt.json"),
        "plan": reference(UNIT / "measurement_plan.json"), "held_query": query,
        "at_utc": now(), "actual_release_performed": False}
    return publish(PACKET / "release_binding.json", binding)


def release_once(commit):
    """Release only the published reviewed identifier and atomically publish actual ownership afterward."""
    plan, unused_command = checked_packet()
    check_publication(commit, release=True)
    for path in (PACKET / "release_intent.json", PACKET / "release_receipt.json", UNIT / "owned_launch_inputs.json"):
        new_path(path)
    actual = owned_submission(plan); binding = document(PACKET / "release_binding.json"); job = actual["job_id"]
    require(binding["job_id"] == job and binding["command"] == ["scontrol", "release", job]
            and binding["submission"] == reference(PACKET / "submission_receipt.json")
            and binding["plan"] == reference(UNIT / "measurement_plan.json"), "the fixed release binding changed")
    query_held(job, "release_final_query")
    publish(PACKET / "release_intent.json", {"kind": "one_actual_nsight_diagnostic_release_attempt", "job_id": job,
        "at_utc": now(), "publication": commit, "binding": reference(PACKET / "release_binding.json"),
        "automatic_retry_permitted": False})
    scheduler(binding["command"], "actual_release")
    receipt = {"kind": "owned_nsight_process_diagnostic_release", "job_id": job, "returncode": 0,
        "plan_sha256": reference(UNIT / "measurement_plan.json")["sha256"], "data_unit": str(UNIT),
        "publication": commit, "at_utc": now()}
    release = publish(PACKET / "release_receipt.json", receipt)
    ownership = {"schema_version": 1, "job_id": job, "plan": reference(UNIT / "measurement_plan.json"),
        "submission_row": submission_row(job, plan), "submission": reference(PACKET / "submission_receipt.json"),
        "release": release}
    final = publish(UNIT / "owned_launch_inputs.json", ownership)
    publish(PACKET / "release_completed.json", {"job_id": job, "release": release, "owned_launch_inputs": final,
        "at_utc": now(), "released_once": True, "scientific_credit": 0})
    return receipt


def main(argv=None):
    """Default to a publication check; every actual scheduler action needs an explicit operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--submit-held-once", action="store_true")
    operations.add_argument("--prepare-release", action="store_true")
    operations.add_argument("--release-once", action="store_true")
    parser.add_argument("--publication-commit", required=True)
    arguments = parser.parse_args(argv)
    if arguments.submit_held_once:
        value = submit_held_once(arguments.publication_commit)
    elif arguments.prepare_release:
        value = prepare_release(arguments.publication_commit)
    elif arguments.release_once:
        value = release_once(arguments.publication_commit)
    else:
        plan, command = checked_packet(); check_publication(arguments.publication_commit)
        value = {"kind": "nsight_diagnostic_launcher_default_check", "passed": True,
                 "action_performed": False, "held_submission_command": command,
                 "plan": reference(UNIT / "measurement_plan.json"), "scientific_credit": 0}
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
