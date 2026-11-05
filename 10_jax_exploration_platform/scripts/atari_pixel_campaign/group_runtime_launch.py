"""One-shot root-facing held submission, exact release and terminal capture for group plans.

Every operation defaults to a published-packet check. Actual identifiers are
recorded and fsynced before release; no attempted submission or release retries.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_runtime as runtime

group = runtime.group


def now():
    """Retain the actual UTC time of one native operation."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def packet_for(plan):
    """Return the exact preparation directory, independently of its future job identifier."""
    return Path(plan["campaign_run_dir"]) / "remote_setup" / plan["unit_id"]


def publish_raw(path, raw):
    """Atomically retain exact native bytes, including incomplete or failed returns."""
    temporary = path.with_name(path.name + ".prepared")
    group.require(type(raw) is bytes and not os.path.lexists(path) and not os.path.lexists(temporary),
        "preserve every previous native byte record")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor); temporary.unlink(); os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return group.receipt(path)


def scheduler(plan, command, stem):
    """Execute one literal native command after owner proof, retaining its full actual return without retries."""
    group.require(os.getuid() == os.geteuid() == runtime.OWNER, "only this campaign owner may act on its jobs")
    packet = packet_for(plan)
    for suffix in ("_stdout.raw", "_stderr.raw", "_return.json"):
        group.require(not os.path.lexists(packet / (stem + suffix)), "preserve the previous native action attempt")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    environment["TZ"] = "UTC"
    started = now()
    try:
        result = subprocess.run(command, capture_output=True, env=environment, timeout=30)
    except subprocess.TimeoutExpired as error:
        publish_raw(packet / (stem + "_stdout.raw"), error.stdout or b"")
        publish_raw(packet / (stem + "_stderr.raw"), error.stderr or b"")
        group.publish(packet / (stem + "_return.json"), {"command": command, "started_at_utc": started,
            "finished_at_utc": now(), "returncode": None, "timed_out": True, "automatic_retry_permitted": False})
        raise
    publish_raw(packet / (stem + "_stdout.raw"), result.stdout)
    publish_raw(packet / (stem + "_stderr.raw"), result.stderr)
    group.publish(packet / (stem + "_return.json"), {"command": command, "started_at_utc": started,
        "finished_at_utc": now(), "returncode": result.returncode, "automatic_retry_permitted": False})
    group.require(type(result.returncode) is int and result.returncode == 0, "the native command failed; do not reuse this attempt")
    return result


def checked_plan(directory):
    """Read a prepared unresolved group without requiring a future ownership sidecar."""
    directory = Path(directory).resolve(strict=True)
    plan = group.read_receipt(group.receipt(directory / "scaling_plan.json"))
    runtime.validate_plan(plan)
    group.require(directory == group.unit_directory(plan) and group.read_receipt(group.receipt(directory / "scaling_prepared.json")) == {
        "plan_sha256": group.receipt(directory / "scaling_plan.json")["sha256"]}, "the frozen group preparation changed")
    runtime.require_launch_review(plan)
    group.source_identity(plan)
    return plan


def published_packet(plan, commit, *, release=False):
    """Require exact staged sources, standards and independent reviews in the actual integration commit."""
    group.require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "a full actual publication commit is required")
    unit, packet = group.unit_directory(plan), packet_for(plan)
    integration = Path(plan["campaign_run_dir"]).parents[2]
    paths = [unit / "scaling_plan.json", unit / "scaling_prepared.json", unit / "backend_image_expected.json",
        packet / "job.sbatch", packet / "held_submission_command.json", Path(plan["runtime_launch_review"]["path"])]
    paths += [unit / "code" / name for name in runtime.helper_names()]
    paths += [directory / name for directory in (unit, packet) for name in (
        "experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json")]
    if release:
        paths += [packet / name for name in ("group_submission.json", "submission_ledgers_fsynced.json",
            "release_binding.json", "release_independent_review.json")]
        review = group.read_receipt(group.receipt(packet / "release_independent_review.json"))
        group.require(type(review.get("schema_version")) is int and review["schema_version"] == 2
            and review.get("kind") == "independent_configured_group_fixed_release_review"
            and review.get("approved") is True and review.get("binding") == group.receipt(packet / "release_binding.json")
            and review.get("job_id") == group.read_receipt(group.receipt(packet / "release_binding.json"))["job_id"],
            "the actual fixed group release has no independent approval")
    for path in paths:
        group.require(path.is_relative_to(integration), "the required published group input must be in integration")
        raw = subprocess.check_output(["git", "-C", str(integration), "cat-file", "blob", commit + ":" + str(path.relative_to(integration))])
        group.require(hashlib.sha256(raw).hexdigest() == group.receipt(path)["sha256"],
            "a required reviewed group source, standard record or release input was not published")
    return paths


def row_for(plan, job):
    """Bind one exact returned job to this script and helper commit."""
    group.require(type(job) is str and re.fullmatch(r"[1-9][0-9]*", job), "an actual positive job identifier is required")
    return [job, plan["kind"], plan["helpers"]["source_commit"], str(packet_for(plan) / "job.sbatch")]


def append_fsynced(path, member):
    """Append one unique actual identifier or row before exposing a submission receipt."""
    before = group.single.ordinary(path).read_bytes()
    group.require(member not in before.decode().splitlines(), "the actual owned identifier or exact row is already recorded")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write((member + "\n").encode()); stream.flush(); os.fsync(stream.fileno())
    group.require(Path(path).read_bytes() == before + (member + "\n").encode(), "owned ledger changed during its append")
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return group.receipt(path)


def submit_held_once(directory, commit):
    """Submit one published script held and durably register the real returned identifier before any release."""
    plan = checked_plan(directory)
    published_packet(plan, commit)
    packet, run = packet_for(plan), Path(plan["campaign_run_dir"])
    group.require((run / "slurm/jobs").is_dir() and (run / "slurm/jobs").resolve(strict=True) == run / "slurm/jobs",
        "the campaign's ordinary per-submission jobs directory must exist before native submission")
    command = runtime.held_submission_command(plan, packet / "job.sbatch")
    group.require(group.read_receipt(group.receipt(packet / "held_submission_command.json")) == command,
        "the prepared held command differs from the exact frozen resource shape")
    for name in ("submission_intent.json", "group_submission.json", "submission_ledgers_fsynced.json"):
        group.require(not os.path.lexists(packet / name), "never reuse a prior attempted group submission")
    with (run / "queue/metadata.lock").open("r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        group.publish(packet / "submission_intent.json", {"at_utc": now(), "publication": commit,
            "held_command": command, "automatic_retry_permitted": False})
        actual = scheduler(plan, command, "actual_submission")
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", actual.stdout.decode().strip())
        group.require(parsed is not None, "the real native submission returned no unique parsable identifier")
        job = parsed.group(1); row = row_for(plan, job)
        snapshots = [append_fsynced(run / "slurm/submitted_jobids.txt", job),
            append_fsynced(run / "slurm/submitted_jobids_uva.txt", job), append_fsynced(run / "slurm/submissions.tsv", "\t".join(row))]
        group.publish(packet / "submission_ledgers_fsynced.json", {"job_id": job, "submission_row": row,
            "ledgers": snapshots, "at_utc": now()})
        result = {"schema_version": 1, "kind": "owned_uva_configured_group_submission", "cluster": "uva",
            "job_id": job, "data_unit": str(group.unit_directory(plan)), "returncode": 0,
            "plan_sha256": group.receipt(group.unit_directory(plan) / "scaling_plan.json")["sha256"],
            "held_command": command, "submission_row": row, "publication": commit, "submitted_held": True, "at_utc": now()}
        group.publish(packet / "group_submission.json", result)
        destination = run / "slurm/jobs" / job
        destination.mkdir(mode=0o700)
        for name in ("job.sbatch", "manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
            publish_raw(destination / name, (packet / name).read_bytes())
        group.publish(destination / "assignment.json", {"job_id": job, "unit": str(group.unit_directory(plan)),
            "submission_row": row, "held": True, "physical_gpu_identities": None})
    return result


def submitted(plan):
    """Prove the actual held receipt and all three matching live ledgers before any query or release."""
    packet, run = packet_for(plan), Path(plan["campaign_run_dir"])
    value = group.read_receipt(group.receipt(packet / "group_submission.json"))
    job = value["job_id"]; row = row_for(plan, job)
    group.require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "owned_uva_configured_group_submission" and value.get("cluster") == "uva"
        and value.get("submitted_held") is True
        and value.get("submission_row") == row and value.get("data_unit") == str(group.unit_directory(plan))
        and type(value.get("returncode")) is int and value["returncode"] == 0
        and value.get("plan_sha256") == group.receipt(group.unit_directory(plan) / "scaling_plan.json")["sha256"],
        "the actual group submission receipt changed")
    for path, member in ((run / "slurm/submitted_jobids.txt", job), (run / "slurm/submitted_jobids_uva.txt", job),
            (run / "slurm/submissions.tsv", "\t".join(row))):
        group.require(path.read_text().splitlines().count(member) == 1, "the actual group lacks one unique matching owned ledger entry")
    group.require(group.read_receipt(group.receipt(packet / "submission_ledgers_fsynced.json"))["submission_row"] == row,
        "group ledgers were not durably synchronized before its receipt")
    return value


def held_query(plan, stem):
    """Query only the already ledger-owned held identifier and compare its complete literal request."""
    value = submitted(plan); job = value["job_id"]
    result = scheduler(plan, ["scontrol", "show", "job", "-o", job], stem)
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)", result.stdout.decode()))
    shape, place = plan["configured_group"]["resources"]["job"], plan["placement"]
    group.require(fields.get("JobId") == job and fields.get("UserId") == "sl5nw(2618919)"
        and fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
        and fields.get("RunTime") == "00:00:00" and fields.get("NodeList") in {"", "(null)"}
        and fields.get("Command") == str(packet_for(plan) / "job.sbatch")
        and fields.get("Partition") == place["partition"] and fields.get("Account") == place["account"]
        and fields.get("QOS") == place["qos"] and fields.get("Reservation", "(null)") == (place["reservation"] or "(null)")
        and fields.get("ReqNodeList") == "jaguar03" and fields.get("NumNodes") in {"1", "1-1"}
        and fields.get("NumTasks") == "1" and fields.get("CPUs/Task") == str(shape["requested_cpus"])
        and fields.get("Requeue") == "0" and fields.get("Restarts") == "0"
        and fields.get("TresPerNode") == "gres/gpu:nvidia_rtx_a4500:" + str(plan["gpu_count"]),
        "the actual held owner, command, placement or typed resource request changed")
    requested = group.single.resources(fields["ReqTRES"])
    group.require(requested.get("cpu") == str(shape["requested_cpus"]) and requested.get("node") == "1"
        and group.memory_mib(requested["mem"]) == shape["host_memory_mib"]
        and requested.get("gres/gpu") == str(plan["gpu_count"]), "held CPU, memory or GPU count differs from the frozen request")
    limit = fields.get("TimeLimit", "")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", limit)
    group.require(match is not None and sum(int(amount or 0) * scale for amount, scale in zip(match.groups(),
        (86400, 3600, 60, 1))) == plan["wall_seconds"], "held wall time differs from the frozen request")
    return group.receipt(packet_for(plan) / (stem + "_return.json"))


def prepare_release(directory, commit):
    """Bind one actual held identifier after durable owned ledgers and native request verification."""
    plan = checked_plan(directory)
    target = packet_for(plan) / "release_binding.json"
    group.require(not os.path.lexists(target) and not os.path.lexists(target.with_name(target.name + ".prepared")),
        "preserve the previous attempted fixed group binding")
    published_packet(plan, commit)
    submission = submitted(plan)
    query = held_query(plan, "release_preparation_query")
    binding = {"kind": "owned_configured_group_fixed_identifier_release_binding", "job_id": submission["job_id"],
        "command": ["scontrol", "release", submission["job_id"]],
        "submission": group.receipt(packet_for(plan) / "group_submission.json"), "held_query": query,
        "plan": group.receipt(group.unit_directory(plan) / "scaling_plan.json"), "at_utc": now(), "release_performed": False}
    return group.publish(packet_for(plan) / "release_binding.json", binding)


def release_once(directory, commit):
    """Release the independently approved fixed identifier and publish exact ownership only afterward."""
    plan = checked_plan(directory); published_packet(plan, commit, release=True)
    packet, unit, run = packet_for(plan), group.unit_directory(plan), Path(plan["campaign_run_dir"])
    for name in ("release_intent.json", "group_release.json"):
        group.require(not os.path.lexists(packet / name), "never reuse a previous attempted group release")
    group.require(not os.path.lexists(unit / "owned_group_inputs.json"), "the actual ownership sidecar already exists")
    submission = submitted(plan); job = submission["job_id"]
    binding = group.read_receipt(group.receipt(packet / "release_binding.json"))
    group.require(binding["job_id"] == job and binding["command"] == ["scontrol", "release", job]
        and binding["submission"] == group.receipt(packet / "group_submission.json")
        and binding["plan"] == group.receipt(unit / "scaling_plan.json"), "the published fixed release binding changed")
    held_query(plan, "release_final_query")
    group.publish(packet / "release_intent.json", {"job_id": job, "publication": commit, "binding": group.receipt(packet / "release_binding.json"),
        "at_utc": now(), "automatic_retry_permitted": False})
    scheduler(plan, binding["command"], "actual_release")
    released = {"schema_version": 1, "kind": "owned_uva_configured_group_release", "cluster": "uva", "job_id": job,
        "data_unit": str(unit), "plan_sha256": group.receipt(unit / "scaling_plan.json")["sha256"], "returncode": 0,
        "command": binding["command"], "submission_receipt_sha256": group.receipt(packet / "group_submission.json")["sha256"],
        "publication": commit, "at_utc": now()}
    release_ref = group.publish(packet / "group_release.json", released)
    owned = {"schema_version": 1, "kind": "uva_configured_group_owned_inputs", "job_id": job,
        "plan": group.receipt(unit / "scaling_plan.json"), "submission_row": row_for(plan, job),
        "submission": group.receipt(packet / "group_submission.json"), "release": release_ref}
    for key, filename, snapshot in (("ledger", "submitted_jobids_uva.txt", "group_owned_job_ids.txt"),
            ("submissions", "submissions.tsv", "group_owned_submissions.tsv")):
        source = run / "slurm" / filename
        publish_raw(unit / snapshot, source.read_bytes())
        owned[key] = {"source_path": str(source), "snapshot": group.receipt(unit / snapshot)}
    group.publish(unit / "owned_group_inputs.json", owned)
    group.owned_inputs(plan)
    group.publish(packet / "release_completed.json", {"job_id": job, "release": release_ref,
        "owned_inputs": group.receipt(unit / "owned_group_inputs.json"), "at_utc": now()})
    return released


def capture_terminal(directory):
    """Capture only the exact already owned job's terminal accounting after execution."""
    plan = checked_plan(directory)
    job = group.owned_inputs(plan)["job_id"]
    result = scheduler(plan, ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"],
        "terminal_observation_" + str(time.time_ns()))
    value = {"command": ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"],
        "returncode": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode(),
        "capture_status": "completed", "byte_limit": 32768, "stdout_scope": "complete_raw_utf8"}
    for name, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
        group.require(len(raw) <= 32768, "owned terminal accounting exceeds the source-bound group limit")
        value[name + "_captured_bytes"] = len(raw)
        value[name + "_full_sha256"] = hashlib.sha256(raw).hexdigest()
    group.single.validate_query(value, value["command"])
    rows = [line.split("|") for line in value["stdout"].splitlines() if line.strip()]
    parent = [row for row in rows if row[0] == job]
    group.require(len(parent) == 1 and parent[0][1] not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"},
        "the owned group has not reached a terminal state; preserve this observation and capture later explicitly")
    group.publish(group.unit_directory(plan) / "terminal_accounting.json", value)
    return value


def main(argv=None):
    """Default to reviewed publication checks; actual operations must be explicitly selected by root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-unit", required=True)
    parser.add_argument("--publication", required=True)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--submit-held-once", action="store_true")
    choices.add_argument("--prepare-release", action="store_true")
    choices.add_argument("--release-once", action="store_true")
    choices.add_argument("--capture-terminal", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.submit_held_once:
        value = submit_held_once(arguments.prepared_unit, arguments.publication)
    elif arguments.prepare_release:
        value = prepare_release(arguments.prepared_unit, arguments.publication)
    elif arguments.release_once:
        value = release_once(arguments.prepared_unit, arguments.publication)
    elif arguments.capture_terminal:
        value = capture_terminal(arguments.prepared_unit)
    else:
        plan = checked_plan(arguments.prepared_unit)
        published_packet(plan, arguments.publication)
        value = {"passed": True, "actual_scheduler_action": False}
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
