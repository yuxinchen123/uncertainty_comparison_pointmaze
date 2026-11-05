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
import shlex
import subprocess

PACKET = Path(__file__).resolve(strict=True).parent
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
INTEGRATION = RUN.parents[2]
KIND = "configured_native_nsight_process_diagnostic"
PACKET_NAME = "2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0"
PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
HELPERS = ("diagnose_configured_native_process.py", "nsight_process_identity.py", "configured_native_process.py",
    "linux_pidfd_process_control.py", "native_process_evidence.py", "direct_tool_descendants.py", "direct_profile_process.py")
BUDGET = {"schema_version": 1, "work_seconds": 240, "cleanup_seconds": 60, "total_seconds": 300}
ZERO = {"scientific_credit": 0, "profile_credit": 0, "case_credit": 0, "idea_credit": 0,
    "environment_transitions": 0, "training_updates": 0, "admission_granted": False,
    "worker_admitted": False, "physical_gres_mapping_verified": False,
    "kernel_device_enforcement_verified": False, "standing_gpu_descriptor_held": False,
    "whole_tool_library_or_device_activity_quantified": False}
RESOURCES = {"partition": "gpu", "account": "sl5nw_base", "qos": "csresnolim", "reservation": "sl5nw_168",
    "node": "jaguar03", "nodes": 1, "tasks": 1, "requested_cpus": 1, "host_memory_gib": 2,
    "gres": "gpu:nvidia_rtx_a4500:1", "wall_seconds": 300, "requeue": False}
OWNER_UID = 2618919
ORIGIN_URL = "git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git"
PUBLICATION_REF = "refs/remotes/origin/merge-atari-cleanrl"
ORIGIN_FETCH = "+refs/heads/*:refs/remotes/origin/*"


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
    require(stat.S_ISREG(path.lstat().st_mode), "the launch artifact must be regular before opening")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        status = os.fstat(descriptor)
        require(stat.S_ISREG(status.st_mode) and status.st_uid == OWNER_UID and status.st_size <= 4 * 1024 * 1024,
                "the launch artifact is not an owned bounded ordinary file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read()
        after = os.fstat(descriptor)
        require(len(raw) == status.st_size and (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "the launch artifact changed during its read")
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


def same(left, right):
    """Require exact JSON types, including integer rather than Boolean counts."""
    if type(left) is not type(right): return False
    if type(right) is dict: return left.keys() == right.keys() and all(same(left[key], right[key]) for key in right)
    if type(right) is list: return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def expected_initial_paths():
    """Derive every canonical seven-helper/standard-record input without a future digest."""
    return {UNIT / "code" / name for name in HELPERS} | {UNIT / name for name in (
        "measurement_plan.json", "measurement_prepared.json", "manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md")} | {
        PACKET / name for name in ("job.sbatch", "manifest.yaml", "config_resolved.yaml", "command.txt",
            "experiment_background.md", "progress.json", "source_checksums.sha256", "held_submission_command.json")}


def expected_job_script():
    """Return the exact reviewed one-card wrapper; it performs only the staged diagnostic."""
    lines = ["#!/bin/bash", "#SBATCH --job-name=selected-python-native-process", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1", "#SBATCH --mem=2G", "#SBATCH --time=00:05:00", "#SBATCH --partition=gpu",
        "#SBATCH --nodelist=jaguar03", "#SBATCH --account=sl5nw_base", "#SBATCH --reservation=sl5nw_168",
        "#SBATCH --qos=csresnolim", "#SBATCH --gres=gpu:nvidia_rtx_a4500:1", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(RUN / "slurm/jobs/native-process-observation-%j.out"),
        "#SBATCH --error=" + str(RUN / "slurm/jobs/native-process-observation-%j.err"), "set -euo pipefail", "umask 077",
        "unset PYTHONPATH PYTHONHOME", "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1",
        "cd " + shlex.quote(str(RUN)), "sha256sum --check --strict " + shlex.quote(str(PACKET / "source_checksums.sha256")),
        "exec " + shlex.join(["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--cpu-bind=cores",
            "--gres=gpu:nvidia_rtx_a4500:1", "--kill-on-bad-exit=1", PYTHON, "-I", "-S", "-B",
            str(UNIT / "code/diagnose_configured_native_process.py"), "--prepared-unit", str(UNIT)])]
    return ("\n".join(lines) + "\n").encode()


def prepared_records():
    """Bind actual canonical prepared bytes only; no target import, tool or scheduler is involved."""
    require(PACKET.name == PACKET_NAME and PACKET.parent.name == "remote_setup"
        and RUN.resolve(strict=True) == RUN and UNIT == RUN / "data" / PACKET_NAME, "the exact new diagnostic packet is required")
    original_path = PACKET / "immutable_artifacts.json"
    original = document(original_path)
    require(type(original) is dict and set(original) == {"files"} and type(original["files"]) is dict
        and set(original["files"]) == {str(path) for path in expected_initial_paths()},
        "the complete canonical prepared inventory is required")
    for name, expected in original["files"].items():
        require(type(expected) is dict and set(expected) == {"sha256", "bytes", "mode"}
            and type(expected["bytes"]) is int and expected["bytes"] >= 0,
            "typed immutable artifact size/mode/hash records are required")
        path = Path(name); raw = ordinary_bytes(path)
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"]
            and oct(path.stat().st_mode & 0o777) == expected["mode"], "a frozen diagnostic artifact changed")
    require(ordinary_bytes(PACKET / "job.sbatch") == expected_job_script(), "the exact staged diagnostic Slurm wrapper changed")
    plan_path = UNIT / "measurement_plan.json";plan = document(plan_path)
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 1
        and plan.get("kind") == KIND and plan.get("packet") == str(PACKET) and plan.get("campaign_run_dir") == str(RUN)
        and plan.get("unit_id") == PACKET_NAME and plan.get("gpu_uuid") is None and plan.get("job_id") is None
        and plan.get("python") == PYTHON and plan.get("hardware_class") == "jaguar03_rtx_a4500"
        and "direct_binding" not in plan and "configured_isolation" not in plan
        and type(plan.get("maximum_active_seconds")) is int and plan["maximum_active_seconds"] == 240
        and same(plan.get("diagnostic_time_budget"), BUDGET) and same(plan.get("diagnostic_boundary"), ZERO),
        "the actual canonical selected-Python zero-credit diagnostic plan is required")
    helpers = plan.get("helpers")
    require(type(helpers) is dict and set(helpers) == {"source_commit", "files"}
        and type(helpers["source_commit"]) is str and re.fullmatch("[0-9a-f]{40}", helpers["source_commit"])
        and type(helpers["files"]) is dict and set(helpers["files"]) == {PREFIX + name for name in HELPERS},
        "the actual full source commit and exact seven-helper closure are required")
    for name in HELPERS:
        expected = helpers["files"][PREFIX + name]
        require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected)
            and reference(UNIT / "code" / name)["sha256"] == expected, "a canonical staged helper differs")
        raw = subprocess.check_output(["git", "-C", str(INTEGRATION), "cat-file", "blob", helpers["source_commit"] + ":" + PREFIX + name])
        require(hashlib.sha256(raw).hexdigest() == expected, "a staged helper is not the declared source commit byte sequence")
    native = {"schema_version": 1, "backend": "linux-x86-64-libc-syscall-pidfd-version-1",
        "mode": "observe_configured_target_for_zero_credit_diagnostic", "expected_target_contract": None,
        "qualification": False, "python": PYTHON,
        "helper_hashes": {name: helpers["files"][PREFIX + name] for name in ("linux_pidfd_process_control.py", "native_process_evidence.py")}}
    configured = {"schema_version": 1, "protocol": "configured-allocation-native-process-control-version-1",
        "hardware_class": "jaguar03_rtx_a4500", "standing_gpu_descriptor_held": False,
        "physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False,
        "configured_isolation_trust": True,
        "helper_hashes": {name: helpers["files"][PREFIX + name] for name in HELPERS if name != "diagnose_configured_native_process.py"}}
    require(same(plan.get("native_process_control"), native) and same(plan.get("configured_native_process_control"), configured),
        "the exact explicit diagnostic-only native declarations are required")
    require(document(UNIT / "measurement_prepared.json") == {"plan_sha256": reference(plan_path)["sha256"]}
        and same(document(UNIT / "manifest.yaml"), plan)
        and same(document(UNIT / "config_resolved.yaml"), {**ZERO, "budget": BUDGET}), "the canonical preparation sidecars differ")
    tool = plan.get("profile_tool")
    require(type(tool) is dict and set(tool) == {"path", "version", "sha256"} and type(tool["path"]) is str
        and Path(tool["path"]).is_absolute() and type(tool["version"]) is str and tool["version"]
        and type(tool["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", tool["sha256"]), "the actual tool metadata is incomplete")
    expected_command = [PYTHON, "-I", "-S", "-B", str(UNIT / "code/diagnose_configured_native_process.py"), "--prepared-unit", str(UNIT)]
    require(ordinary_bytes(UNIT / "command.txt") == (shlex.join(expected_command) + "\n").encode(), "the canonical producer command changed")
    for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
        require(ordinary_bytes(PACKET / name) == ordinary_bytes(UNIT / name), "packet and canonical standard records differ")
    command = document(PACKET / "held_submission_command.json")
    require(command == ["sbatch", "--hold", "--parsable", "--no-requeue", "--partition=gpu", "--account=sl5nw_base",
        "--qos=csresnolim", "--reservation=sl5nw_168", "--nodelist=jaguar03", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G",
        "--gres=gpu:nvidia_rtx_a4500:1", "--time=00:05:00", str(PACKET / "job.sbatch")], "the exact held submission command changed")
    expected_checksums = {str(UNIT / "code" / name): helpers["files"][PREFIX + name] for name in HELPERS}
    expected_checksums.update({str(path): reference(path)["sha256"] for path in (plan_path, UNIT / "measurement_prepared.json")})
    lines = ordinary_bytes(PACKET / "source_checksums.sha256").decode().splitlines()
    require(len(lines) == len(expected_checksums) and all(re.fullmatch(r"[0-9a-f]{64}  /.+", line) for line in lines)
        and len({line[66:] for line in lines}) == len(lines)
        and {line[66:]: line[:64] for line in lines} == expected_checksums, "source checksums omit or substitute canonical bytes")
    binding = {"kind": "actual_configured_native_diagnostic_prepared_binding", "schema_version": 1,
        "packet": str(PACKET), "data_unit": str(UNIT), "diagnostic_kind": KIND,
        "plan": reference(plan_path), "source_commit": helpers["source_commit"], "resources": RESOURCES,
        "immutable_artifacts": reference(original_path),
        "canonical_inputs": {str(path): reference(path) for path in sorted(expected_initial_paths())},
        "submission_command": command, "actual_job_id": None, "actual_release_performed": False,
        "scientific_credit": 0, "profile_admission_granted": False}
    return plan, command, binding


def prepare_binding_once():
    """Publish only a real prepared metadata binding; it performs no scheduler operation."""
    actor()
    for path in (PACKET / "prepared_launch_binding.json", PACKET / "submission_intent.json", PACKET / "submission_receipt.json"):
        new_path(path)
    unused_plan, unused_command, binding = prepared_records()
    return publish(PACKET / "prepared_launch_binding.json", binding)


def checked_packet():
    """Require the exact later immutable binding; placeholders never authorize a launch."""
    plan, command, binding = prepared_records()
    require(same(document(PACKET / "prepared_launch_binding.json"), binding), "the actual canonical prepared binding changed")
    return plan, command


def require_published_commit(commit):
    """Require existing exact origin-tracking ancestry; root verifies its pushed tip separately."""
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "the real full publication commit is required")
    prefix = ["git", "-C", str(INTEGRATION)]
    try:
        origin = subprocess.check_output([*prefix, "config", "--get-all", "remote.origin.url"], stderr=subprocess.PIPE, timeout=30).decode().splitlines()
        fetch = subprocess.check_output([*prefix, "config", "--get-all", "remote.origin.fetch"], stderr=subprocess.PIPE, timeout=30).decode().splitlines()
        tip = subprocess.check_output([*prefix, "rev-parse", "--verify", PUBLICATION_REF + "^{commit}"], stderr=subprocess.PIPE, timeout=30).decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("the actual configured origin and exact publication tracking reference are required") from error
    require(origin == [ORIGIN_URL] and fetch == [ORIGIN_FETCH] and re.fullmatch(r"[0-9a-f]{40}", tip),
        "the configured origin or its exact branch-tracking mapping differs")
    result = subprocess.run([*prefix, "merge-base", "--is-ancestor", commit, tip], capture_output=True, timeout=30)
    require(type(result.returncode) is int and result.returncode == 0,
        "the supplied local commit is not included in the exact origin/merge-atari-cleanrl history")
    return tip


def check_publication(commit, release=False):
    """Require independently reviewed exact source and packet bytes in the actual Git commit."""
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "the real full publication commit is required")
    require_published_commit(commit)
    freeze_path = PACKET / "launcher_source_freeze.json"
    freeze = document(freeze_path)
    review_path = PACKET / "launcher_independent_review.json"
    review = document(review_path)
    require(review.get("passed") is True and review.get("kind") == "independent_configured_native_diagnostic_prepared_launcher_review"
            and review.get("source_freeze") == reference(freeze_path)
            and review.get("prepared_binding") == reference(PACKET / "prepared_launch_binding.json"),
            "the exact launcher source has no independent approval")
    require(type(freeze) is dict and type(freeze.get("files")) is dict, "the exact launcher publication roster is required")
    static = document(PACKET / "packet_static_verification.json")
    require(static.get("passed") is True and static.get("prepared_binding") == reference(PACKET / "prepared_launch_binding.json"),
        "the exact prepared binding has no static packet verification")
    paths = dict(freeze["files"])
    initial = document(PACKET / "immutable_artifacts.json")["files"]
    mandatory = set(initial) | {str(PACKET / "immutable_artifacts.json"), str(PACKET / "packet_static_verification.json"),
                               str(PACKET / "launch_diagnostic_once.py"), str(PACKET / "test_launch_diagnostic_once.py"),
                               str(PACKET / "prepared_launch_binding.json"),
                               str(PACKET / "schema/diagnose_configured_native_process_approved.py")}
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
    require(type(value) is str and re.fullmatch(r"[1-9][0-9]*", value) and value != "6821437",
            "the actual diagnostic identifier is invalid")
    return value


def submission_row(job, plan):
    """Construct the exact diagnostic row required by its unchanged owned-allocation reader."""
    return [identifier(job), KIND, plan["helpers"]["source_commit"], str(PACKET / "job.sbatch")]


def append_member(path, member):
    """Append and fsync one new exact ledger member through an owned pinned descriptor."""
    path = Path(path)
    before = ordinary_bytes(path)
    require(not before or before.endswith(b"\n"), "an existing ledger lacks its final newline")
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
    require(value["kind"] == "owned_configured_native_process_diagnostic_submission" and type(value["returncode"]) is int
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
        publish(PACKET / "submission_intent.json", {"kind": "one_configured_native_diagnostic_held_submission_attempt",
            "at_utc": now(), "publication": commit, "command": command, "plan": reference(UNIT / "measurement_plan.json"),
            "automatic_retry_permitted": False})
        actual = scheduler(command, "actual_submission")
        parsed = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", actual.stdout.decode().strip())
        require(parsed is not None, "the actual parsable submission output has no unique identifier")
        job = identifier(parsed.group(1)); row = submission_row(job, plan)
        require(not any(line.split("\t")[0] == job for line in ordinary_bytes(RUN / "slurm/submissions.tsv").decode().splitlines()),
            "the returned identifier already has a campaign submission row")
        ledgers = [append_member(RUN / "slurm/submitted_jobids.txt", job),
                   append_member(RUN / "slurm/submitted_jobids_uva.txt", job),
                   append_member(RUN / "slurm/submissions.tsv", "\t".join(row))]
        publish(PACKET / "submission_ledgers_fsynced.json", {"kind": "actual_configured_native_diagnostic_three_ledgers_fsynced",
            "job_id": job, "submission_row": row, "ledgers": ledgers, "at_utc": now()})
        receipt = {"kind": "owned_configured_native_process_diagnostic_submission", "job_id": job, "returncode": 0,
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


def query_held(job, stem, plan):
    """Query only the already ledger-owned identifier and verify its exact held resource request."""
    job = identifier(job)
    require(owned_submission(plan)["job_id"] == job, "the query identifier is not this exact owned submission")
    result = scheduler(["scontrol", "show", "job", "-o", job], stem)
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)", result.stdout.decode())
    require(len({key for key, unused in pairs}) == len(pairs), "duplicate native held-job fields are not accepted")
    fields = dict(pairs)
    require(fields.get("JobId") == job and fields.get("UserId") == "sl5nw(" + str(OWNER_UID) + ")"
            and fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
            and fields.get("Requeue") == "0" and fields.get("Restarts") == "0"
            and fields.get("TimeLimit") == "00:05:00" and fields.get("Partition") == "gpu"
            and fields.get("Account") == "sl5nw_base" and fields.get("QOS") == "csresnolim"
            and fields.get("ReqNodeList") == "jaguar03" and fields.get("NumNodes") in {"1", "1-1"}
            and fields.get("CPUs/Task") == "1" and fields.get("NumTasks") == "1"
            and fields.get("Command") == str(PACKET / "job.sbatch")
            and fields.get("Reservation", "") == "sl5nw_168", "the actual held diagnostic shape or owner changed")
    resource_pairs = [item.split("=", 1) for item in fields["ReqTRES"].split(",")]
    require(all(len(pair) == 2 for pair in resource_pairs) and len({pair[0] for pair in resource_pairs}) == len(resource_pairs),
        "duplicate or malformed requested resources are not accepted")
    resources = dict(resource_pairs)
    # One requested processor may occupy both logical threads on this documented two-thread-per-core node.
    require(resources.get("cpu") in {"1", "2"} and fields.get("NumCPUs") in {"1", "2"}
            and resources.get("mem") == "2G" and resources.get("node") == "1" and resources.get("gres/gpu") == "1"
            and fields.get("TresPerNode") == "gres/gpu:nvidia_rtx_a4500:1", "the actual requested resources changed")
    return reference(PACKET / (stem + "_return.json"))


def prepare_release(commit):
    """Bind a release command only after the held job and actual durable ledgers exist."""
    plan, unused_command = checked_packet()
    check_publication(commit)
    new_path(PACKET / "release_binding.json")
    actual = owned_submission(plan)
    query = query_held(actual["job_id"], "release_preparation_query", plan)
    binding = {"kind": "actual_owned_configured_native_diagnostic_fixed_identifier_release_binding", "job_id": actual["job_id"],
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
    query_held(job, "release_final_query", plan)
    publish(PACKET / "release_intent.json", {"kind": "one_actual_configured_native_diagnostic_release_attempt", "job_id": job,
        "at_utc": now(), "publication": commit, "binding": reference(PACKET / "release_binding.json"),
        "automatic_retry_permitted": False})
    scheduler(binding["command"], "actual_release")
    receipt = {"kind": "owned_configured_native_process_diagnostic_release", "job_id": job, "returncode": 0,
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
    operations.add_argument("--prepare-binding-once", action="store_true")
    operations.add_argument("--submit-held-once", action="store_true")
    operations.add_argument("--prepare-release", action="store_true")
    operations.add_argument("--release-once", action="store_true")
    parser.add_argument("--publication-commit")
    arguments = parser.parse_args(argv)
    if arguments.prepare_binding_once:
        require(arguments.publication_commit is None, "metadata binding makes no publication claim")
        value = prepare_binding_once()
    elif arguments.submit_held_once:
        value = submit_held_once(arguments.publication_commit)
    elif arguments.prepare_release:
        value = prepare_release(arguments.publication_commit)
    elif arguments.release_once:
        value = release_once(arguments.publication_commit)
    else:
        plan, command = checked_packet(); check_publication(arguments.publication_commit)
        value = {"kind": "configured_native_diagnostic_launcher_default_check", "passed": True,
                 "action_performed": False, "held_submission_command": command,
                 "plan": reference(UNIT / "measurement_plan.json"), "scientific_credit": 0}
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
