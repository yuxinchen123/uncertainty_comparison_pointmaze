"""Submit and release one independently reviewed CPU monitor replacement without changing the old monitor."""
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
OWNER_UID = 2618919
COMMIT = "b431573a272ccec354cfbccb2b7528b4b846e7b9"
KIND = "monitor"

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
        require(stat.S_ISREG(status.st_mode) and status.st_uid == OWNER_UID and status.st_size <= 32 * 1024 * 1024,
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



def checked_packet():
    """Require the unchanged prepared source, five standard records and actual independent full-candidate proof."""
    frozen = document(PACKET / "static_source_freeze.json")
    require(frozen["source_commit"] == COMMIT and len(frozen["files"]) == 322,
            "the complete reviewed 322-artifact prepared monitor is required")
    for name, expected in frozen["files"].items():
        path = Path(name)
        require(path.is_relative_to(PACKET) or path.is_relative_to(UNIT), "a prepared monitor artifact escaped its packet")
        raw = ordinary_bytes(path)
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"]
                and oct(stat.S_IMODE(path.stat().st_mode)) == expected["mode"], "a prepared monitor artifact changed")
    plan = document(PACKET / "monitor_replacement_plan.json")
    shape = {"account": "sl5nw_base", "partition": "nolim", "qos": "normal", "nodes": 1, "ntasks": 1,
        "cpus_per_task": 1, "host_memory_gib": 2, "gpus_per_node": 0, "time": "7-00:00:00",
        "interval_seconds": 1200, "requeue": False}
    require(plan["kind"] == "persistent_pixel_study_monitor_replacement" and plan["helper_commit"] == COMMIT
            and plan["campaign_run_dir"] == str(RUN) and plan["job_shape"] == shape
            and plan["source_file_count"] == 287 and plan["old_monitor_job_id"] == "6783003"
            and all(plan[name] is None for name in ("job_id", "gpu_uuid", "reservation"))
            and plan["data_unit"] == {"name": UNIT.name, "path": str(UNIT)}, "the prepared monitor identity or shape changed")
    command = ["sbatch", "--parsable", "--hold", "--no-requeue", "--account=sl5nw_base", "--partition=nolim",
        "--qos=normal", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G", "--gpus-per-node=0",
        "--time=7-00:00:00", "--job-name=pixel_study_monitor",
        "--output=" + str(RUN / "slurm/jobs/monitor_%j.log"), str(PACKET / "monitor.sbatch")]
    require(plan["submission_command"] == command and plan["job_script"] == reference(PACKET / "monitor.sbatch"),
            "the exact reviewed held submission command or script changed")
    for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json"):
        require(ordinary_bytes(PACKET / name) == ordinary_bytes(UNIT / name), "packet and unit standard records differ")
    manifest = document(PACKET / "manifest.yaml")
    require(manifest["plan"] == reference(PACKET / "monitor_replacement_plan.json")
            and manifest["submission_command"] == command and manifest["source_commit"] == COMMIT,
            "the prepared manifest differs from its exact plan")
    prepared = document(PACKET / "independent_prepared_packet_review.json")
    require(prepared["passed"] is True and prepared["source_commit"] == COMMIT and prepared["source_files_verified"] == 287
            and prepared["frozen_artifacts_verified"] == 322 and prepared["source_freeze"] == reference(PACKET / "static_source_freeze.json")
            and prepared["plan"] == reference(PACKET / "monitor_replacement_plan.json")
            and prepared["actual_new_monitor_runtime_verified"] is False and prepared["actual_new_first_report_verified"] is False,
            "the actual independent prepared monitor proof is missing or changed")
    candidate = document(PACKET / "candidate_after_reporting_index_full_report_receipt.json")
    require(prepared["candidate_full_report"] == reference(PACKET / "candidate_after_reporting_index_full_report_receipt.json")
            and candidate["passed"] is True and candidate["full_tick_completed"] is True
            and candidate["source_commit"] == COMMIT and candidate["job_id"] == "6783003",
            "the actual full candidate report was not independently verified")
    return plan, command


def published_bytes(commit, paths):
    """Check all exact reviewed Git byte sequences in one bounded read-only batch."""
    expressions = [commit + ":" + str(Path(name).relative_to(INTEGRATION)) for name in paths]
    result = subprocess.run(["git", "-C", str(INTEGRATION), "cat-file", "--batch"],
        input=("\n".join(expressions) + "\n").encode(), capture_output=True, check=True)
    offset = 0
    for name, expected in paths.items():
        end = result.stdout.find(b"\n", offset)
        require(end >= offset, "the published Git batch is incomplete")
        header = result.stdout[offset:end].split()
        require(len(header) == 3 and header[1] == b"blob" and header[2].isdigit(), "a reviewed input is not a published Git blob")
        size = int(header[2]); start = end + 1; offset = start + size + 1
        require(size <= 32 * 1024 * 1024 and result.stdout[offset - 1:offset] == b"\n"
                and hashlib.sha256(result.stdout[start:start + size]).hexdigest() == expected,
                "a reviewed input differs from its published Git byte sequence")
    require(offset == len(result.stdout), "the published Git batch has unexpected extra output")


def check_publication(commit, release=False):
    """Require the actual published prepared proof and independently reviewed launcher before any action."""
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "the real full publication commit is required")
    freeze_path = PACKET / "monitor_launcher_source_freeze.json"
    review_path = PACKET / "monitor_launcher_independent_review.json"
    freeze, review = document(freeze_path), document(review_path)
    require(review.get("passed") is True and review.get("source_freeze") == reference(freeze_path),
            "the exact monitor launcher has no independent approval")
    paths = dict(freeze["files"])
    original = document(PACKET / "static_source_freeze.json")["files"]
    mandatory = set(original) | {str(PACKET / name) for name in (
        "static_source_freeze.json", "independent_prepared_packet_review.json", "launch_monitor_once.py",
        "test_launch_monitor_once.py", "run_monitor_launcher_metadata_tests.py", "monitor_launch_policy.json")}
    require(mandatory <= set(paths), "the launcher freeze omitted a mandatory prepared proof or source")
    paths.update({str(freeze_path): reference(freeze_path)["sha256"], str(review_path): reference(review_path)["sha256"]})
    if release:
        binding_path = PACKET / "monitor_release_binding.json"
        release_review = PACKET / "monitor_release_independent_review.json"
        reviewed = document(release_review)
        require(reviewed.get("passed") is True and reviewed.get("binding") == reference(binding_path),
                "the actual fixed monitor identifier has no independently reviewed release")
        for path in (binding_path, release_review, PACKET / "monitor_submission_receipt.json",
                     PACKET / "monitor_submission_ledgers_fsynced.json"):
            paths[str(path)] = reference(path)["sha256"]
    for name, expected in paths.items():
        path = Path(name)
        require(path.is_relative_to(INTEGRATION) and reference(path)["sha256"] == expected,
                "a reviewed monitor source or input changed")
    published_bytes(commit, paths)


def submission_row(job, plan):
    """Use the exact real identifier and source/script row required by the unchanged runtime guard."""
    return [identifier(job), "monitor", plan["helper_commit"], str(PACKET / "monitor.sbatch")]


def owned_submission(plan):
    """Verify one actual held receipt against all three live owned ledgers before any job query."""
    value = document(PACKET / "monitor_submission_receipt.json")
    job = identifier(value["job_id"])
    row = submission_row(job, plan)
    require(value["kind"] == "owned_persistent_cpu_monitor_submission" and type(value["returncode"]) is int
            and value["returncode"] == 0 and value["plan"] == reference(PACKET / "monitor_replacement_plan.json")
            and value["data_unit"] == str(UNIT) and value["submission_row"] == row and value["submitted_held"] is True,
            "the actual held monitor submission receipt changed")
    for path, member in ((RUN / "slurm/submitted_jobids.txt", job), (RUN / "slurm/submitted_jobids_uva.txt", job),
                         (RUN / "slurm/submissions.tsv", "\t".join(row))):
        require(ordinary_bytes(path).decode().splitlines().count(member) == 1, "the monitor lacks a unique owned ledger member")
    recorded = [line.split("\t") for line in ordinary_bytes(RUN / "slurm/submissions.tsv").decode().splitlines()
                if line.split("\t")[0] == job]
    require(recorded == [row], "another row uses the actual monitor identifier")
    synced = document(PACKET / "monitor_submission_ledgers_fsynced.json")
    require(synced["job_id"] == job and synced["submission_row"] == row and len(synced["ledgers"]) == 3,
            "the actual three-ledger synchronization record is missing")
    return value


def submit_held_once(commit):
    """Make one held attempt and fsync its exact identifier and row before publishing the actual receipt."""
    plan, command = checked_packet()
    check_publication(commit)
    for name in ("monitor_submission_intent.json", "monitor_submission_receipt.json", "monitor_submission_ledgers_fsynced.json"):
        new_path(PACKET / name)
    actor()
    descriptor = os.open(RUN / "queue/metadata.lock", os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "r+") as guard:
        require(os.fstat(guard.fileno()).st_uid == OWNER_UID and stat.S_ISREG(os.fstat(guard.fileno()).st_mode),
                "the campaign metadata lock is not an owned ordinary file")
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        publish(PACKET / "monitor_submission_intent.json", {"kind": "one_owned_cpu_monitor_held_submission_attempt",
            "at_utc": now(), "publication": commit, "command": command,
            "plan": reference(PACKET / "monitor_replacement_plan.json"), "automatic_retry_permitted": False})
        actual = scheduler(command, "monitor_actual_submission")
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", actual.stdout.decode().strip())
        require(parsed is not None, "the actual parsable submission output lacks one unique identifier")
        job = identifier(parsed.group(1))
        row = submission_row(job, plan)
        ledgers = [append_member(path, member) for path, member in (
            (RUN / "slurm/submitted_jobids.txt", job), (RUN / "slurm/submitted_jobids_uva.txt", job),
            (RUN / "slurm/submissions.tsv", "\t".join(row)))]
        publish(PACKET / "monitor_submission_ledgers_fsynced.json", {"kind": "actual_monitor_three_ledgers_fsynced",
            "job_id": job, "submission_row": row, "ledgers": ledgers, "at_utc": now()})
        value = {"kind": "owned_persistent_cpu_monitor_submission", "job_id": job, "returncode": actual.returncode,
            "plan": reference(PACKET / "monitor_replacement_plan.json"), "data_unit": str(UNIT),
            "submission_row": row, "submitted_held": True, "publication": commit, "at_utc": now(),
            "old_monitor_stopped_or_adopted": False}
        publish(PACKET / "monitor_submission_receipt.json", value)
        job_folder = RUN / "slurm/jobs" / job
        job_folder.mkdir()
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json"):
            publish(job_folder / name, ordinary_bytes(PACKET / name))
        publish(job_folder / "monitor.sbatch", ordinary_bytes(PACKET / "monitor.sbatch"))
        publish(job_folder / "assignment.json", {"job_id": job, "kind": "monitor", "unit_id": UNIT.name,
            "source_commit": COMMIT, "plan": reference(PACKET / "monitor_replacement_plan.json"),
            "old_monitor_to_preserve": "6783003", "actual_new_runtime_independent_review_required": True})
        return value


def query_held(job, stem):
    """Query only the already recorded monitor identifier and check its exact held CPU request."""
    job = identifier(job)
    plan, unused = checked_packet()
    require(owned_submission(plan)["job_id"] == job, "the query identifier is not this exact ledger-owned monitor")
    result = scheduler(["scontrol", "show", "job", "-o", job], stem)
    require(len(result.stdout.splitlines()) == 1, "the exact owned held query did not return one row")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)", result.stdout.decode()))
    wanted = {"JobId": job, "UserId": "sl5nw(2618919)", "JobState": "PENDING", "Reason": "JobHeldUser",
        "Requeue": "0", "Restarts": "0", "TimeLimit": "7-00:00:00", "Partition": "nolim", "Account": "sl5nw_base",
        "QOS": "normal", "CPUs/Task": "1", "NumTasks": "1", "MinMemoryNode": "2G",
        "Command": str(PACKET / "monitor.sbatch"), "TresPerNode": "gres/gpu:0"}
    require(all(fields.get(name) == expected for name, expected in wanted.items())
            and fields.get("NumNodes") in {"1", "1-1"} and fields.get("NumCPUs") in {"1", "2"}
            and fields.get("Reservation", "") in {"", "(null)", "None"}, "the actual held monitor shape or owner changed")
    pairs = [item.split("=", 1) for item in fields["ReqTRES"].split(",")]
    require(len({key for key, value in pairs}) == len(pairs), "the held monitor resource record repeats a key")
    resources = dict(pairs)
    require(resources.get("cpu") == "1" and resources.get("node") == "1" and resources.get("mem") == "2G"
            and all(not key.startswith("gres/gpu") or value == "0" for key, value in resources.items()),
            "the held monitor requested extra CPUs, memory, nodes or GPUs")
    return reference(PACKET / (stem + "_return.json"))


def prepare_release(commit):
    """Bind the single future release to the actual returned and durably ledger-owned held identifier."""
    plan, unused = checked_packet()
    check_publication(commit)
    new_path(PACKET / "monitor_release_binding.json")
    actual = owned_submission(plan)
    query = query_held(actual["job_id"], "monitor_release_preparation_query")
    value = {"kind": "actual_owned_monitor_fixed_identifier_release_binding", "job_id": actual["job_id"],
        "command": ["scontrol", "release", actual["job_id"]], "submission": reference(PACKET / "monitor_submission_receipt.json"),
        "plan": reference(PACKET / "monitor_replacement_plan.json"), "held_query": query, "at_utc": now(),
        "actual_release_performed": False, "old_monitor_stopped_or_adopted": False}
    return publish(PACKET / "monitor_release_binding.json", value)


def release_once(commit):
    """Release the independently reviewed actual identifier once and preserve every unsuccessful attempt."""
    plan, unused = checked_packet()
    check_publication(commit, release=True)
    actual = owned_submission(plan)
    binding = document(PACKET / "monitor_release_binding.json")
    require(binding["job_id"] == actual["job_id"] and binding["command"] == ["scontrol", "release", actual["job_id"]]
            and binding["submission"] == reference(PACKET / "monitor_submission_receipt.json")
            and binding["plan"] == reference(PACKET / "monitor_replacement_plan.json")
            and binding["actual_release_performed"] is False, "the actual fixed monitor release binding changed")
    for name in ("monitor_release_intent.json", "monitor_release_receipt.json", "monitor_release_complete.json"):
        new_path(PACKET / name)
    fresh = query_held(actual["job_id"], "monitor_release_fresh_query")
    publish(PACKET / "monitor_release_intent.json", {"kind": "one_owned_cpu_monitor_release_attempt",
        "job_id": actual["job_id"], "command": binding["command"], "binding": reference(PACKET / "monitor_release_binding.json"),
        "fresh_held_query": fresh, "publication": commit, "at_utc": now(), "automatic_retry_permitted": False})
    result = scheduler(binding["command"], "monitor_actual_release")
    released = {"kind": "owned_persistent_cpu_monitor_release", "job_id": actual["job_id"], "returncode": result.returncode,
        "plan": reference(PACKET / "monitor_replacement_plan.json"), "binding": reference(PACKET / "monitor_release_binding.json"),
        "submission": reference(PACKET / "monitor_submission_receipt.json"), "at_utc": now(), "publication": commit,
        "old_monitor_stopped_or_adopted": False}
    publish(PACKET / "monitor_release_receipt.json", released)
    publish(PACKET / "monitor_release_complete.json", {"job_id": actual["job_id"], "release": reference(PACKET / "monitor_release_receipt.json"),
        "at_utc": now(), "actual_new_runtime_and_first_full_independent_review_required": True,
        "old_monitor_to_preserve": "6783003", "old_auxiliary_step_drain_required_before_any_old_stop": True,
        "canonical_actions_or_provenance_changed": False, "scientific_credit": 0})
    return released


def main(arguments=None):
    """Default to read-only publication verification; require one explicit irreversible one-shot action flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-commit", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit-held-once", action="store_true")
    action.add_argument("--prepare-release", action="store_true")
    action.add_argument("--release-once", action="store_true")
    args = parser.parse_args(arguments)
    if args.submit_held_once:
        value = submit_held_once(args.publication_commit)
    elif args.prepare_release:
        value = prepare_release(args.publication_commit)
    elif args.release_once:
        value = release_once(args.publication_commit)
    else:
        checked_packet()
        check_publication(args.publication_commit)
        value = {"passed": True, "action_performed": False, "new_monitor_job_id": None,
                 "old_monitor_to_preserve": "6783003", "actual_monitoring_verification_required_in_submission_turn": True}
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
