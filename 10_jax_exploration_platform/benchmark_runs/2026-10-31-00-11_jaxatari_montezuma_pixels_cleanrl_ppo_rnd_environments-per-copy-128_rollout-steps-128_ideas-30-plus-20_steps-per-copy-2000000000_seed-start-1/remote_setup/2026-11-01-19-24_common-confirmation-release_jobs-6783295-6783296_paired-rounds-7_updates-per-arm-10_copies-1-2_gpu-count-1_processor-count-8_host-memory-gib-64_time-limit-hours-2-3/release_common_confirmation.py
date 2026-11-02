"""Release one exact held common-candidate confirmation job once, then save real ownership inputs."""
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

UTILITY = Path(__file__).absolute().parent
RUN = UTILITY.parent.parent
REPOSITORY = RUN.parents[2]
SOURCE = "9fcc43a478b93d731df780805fd7c3c514cdf67f"
INITIAL_PUBLICATION = "4ef27e7e46bce2b64ea6b8050935f88986ded12e"
FROZEN_INPUTS_SHA = 'ae959ea98a6daba71b0c7a651a8342e4c959c05d6b269d3af7c31ed6ed540a9b'
MONITOR_JOB = "6783003"
MONITOR_SOURCE = "79a01e851ff92ddba7d0e12eaf8164cab2028878"
SCOPES = {'1': {'job_id': '6783295', 'packet': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-54_common_cache-room-backgrounds-true_render-final-two-only-true_confirmation_paired-rounds-7_updates-per-arm-10_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-2_jaguar03', 'unit': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-18-54_common_cache-room-backgrounds-true_render-final-two-only-true_confirmation_paired-rounds-7_updates-per-arm-10_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-2_jaguar03'}, '2': {'job_id': '6783296', 'packet': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-54_common_cache-room-backgrounds-true_render-final-two-only-true_confirmation_paired-rounds-7_updates-per-arm-10_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3_jaguar03', 'unit': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-18-54_common_cache-room-backgrounds-true_render-final-two-only-true_confirmation_paired-rounds-7_updates-per-arm-10_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3_jaguar03'}}
PACKET = UNIT = JOB = COMMAND = None
PLAN_SHA = SCRIPT_SHA = PRE_RELEASE_SHA = SUBMISSION_SHA = HELD_BINDING_SHA = HELD_PROOF_SHA = None
SELECTED = None
REQUESTED_HOURS = None
NUMERICAL_MODULES = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy"}
BOUNDARY = {"scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
            "packing_point_credit": 0, "throughput_credit": 0, "site_qualification_credit": 0,
            "admission_granted": False, "worker_admitted": False}
ATTEMPT_FILES = ("release_intent.json", "release_receipt.json", "release_preflight_return.json",
    "release_preflight_stdout.raw", "release_preflight_stderr.raw", "release_return.json",
    "release_stdout.raw", "release_stderr.raw", "release_final_inputs_verification.json", "release_failure.json")

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
    require(len(raw) <= 64 * 1024 * 1024, "release metadata exceeds its bound")
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
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]*)", raw)
    require(len(values) == 1, "missing or repeated owned job field: " + name)
    return values[0]


def validate_held(raw):
    expected = {"JobId": JOB, "UserId": "sl5nw(2618919)", "Account": "sl5nw_base",
        "Partition": "gpu", "QOS": "csresnolim", "JobState": "PENDING", "Reason": "JobHeldUser",
        "ReqNodeList": "jaguar03", "Reservation": "sl5nw_168", "Requeue": "0", "Restarts": "0",
        "NumCPUs": "8", "NumTasks": "1", "CPUs/Task": "8", "MinMemoryNode": "64G",
        "TimeLimit": str(REQUESTED_HOURS).zfill(2) + ":00:00", "Command": str(PACKET / "job.sbatch"),
        "TresPerNode": "gres/gpu:nvidia_rtx_a4500:1"}
    require(all(field(raw, name) == value for name, value in expected.items())
        and field(raw, "NumNodes") in {"1", "1-1"}, "the actual held qualification identity or request changed")
    require(field(raw, "AllocTRES") == "(null)" and field(raw, "NodeList") in {"", "(null)"},
        "the held job must remain unallocated")
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


def select(copies):
    """Resolve only one literal actual identifier; no other packet or job is accepted."""
    global PACKET, UNIT, JOB, COMMAND, PLAN_SHA, SCRIPT_SHA, PRE_RELEASE_SHA, SUBMISSION_SHA
    global HELD_BINDING_SHA, HELD_PROOF_SHA, SELECTED, REQUESTED_HOURS
    require(type(copies) is int and copies in (1, 2), "choose exactly copies one or two")
    require(FROZEN_INPUTS_SHA is not None and ref(UTILITY / "release_frozen_inputs.json")["sha256"] == FROZEN_INPUTS_SHA,
        "the actual held proof and immutable release inputs are not frozen")
    frozen = document(UTILITY / "release_frozen_inputs.json")
    require(type(frozen.get("schema_version")) is int and frozen["schema_version"] == 1
        and frozen.get("kind") == "fixed_common_confirmation_actual_release_inputs"
        and frozen.get("source_commit") == SOURCE and frozen.get("initial_publication") == INITIAL_PUBLICATION
        and set(frozen["jobs"]) == {"1", "2"}, "the exact release source or selectors changed")
    item = frozen["jobs"][str(copies)]
    require({key: item[key] for key in ("job_id", "packet", "unit")} == SCOPES[str(copies)], "the literal job scope changed")
    PACKET, UNIT, JOB = Path(item["packet"]), Path(item["unit"]), item["job_id"]
    COMMAND = ["scontrol", "release", JOB]
    PLAN_SHA, SCRIPT_SHA, PRE_RELEASE_SHA = (item[key]["sha256"] for key in ("plan", "script", "pre_release_inputs"))
    SUBMISSION_SHA, HELD_BINDING_SHA, HELD_PROOF_SHA = (item[key]["sha256"] for key in ("submission", "held_binding", "held_proof"))
    SELECTED = item
    REQUESTED_HOURS = {1: 2, 2: 3}[copies]
    return frozen


def utc_observation(value, label):
    """Require a genuine UTC observation less than twenty minutes old."""
    observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(observed.tzinfo is not None and observed.utcoffset().total_seconds() == 0
        and 0 <= (datetime.now(timezone.utc) - observed).total_seconds() < 1200,
        label + " is stale, future-dated or not UTC")
    return observed


def validate_proof(proof):
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "owned_uva_configured_isolation_actual_full_held_monitor_proof"
        and proof.get("passed") is True and proof.get("job_id") == JOB
        and proof.get("state") == "PENDING" and proof.get("reason") == "JobHeldUser"
        and proof.get("publication_commit") == INITIAL_PUBLICATION and proof.get("helper_source_commit") == SOURCE,
        "the actual full-monitor held proof is missing or belongs to another job")
    observed = utc_observation(proof["at_utc"], "the actual held query")
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
    require(monitor["job_id"] == MONITOR_JOB and monitor["source_commit"] == MONITOR_SOURCE
        and type(monitor["cadence_seconds"]) is int and monitor["cadence_seconds"] == 1200
        and monitor["persistent_running_verified"] is True,
        "the exact full monitoring or zero-credit scope changed")
    runtime_review = checked_ref(monitor["runtime_and_first_full_report_review"])
    require(type(runtime_review.get("schema_version")) is int and runtime_review["schema_version"] == 1
        and runtime_review.get("kind") == "independent_new_persistent_cpu_monitor_runtime_and_first_full_report_approval"
        and runtime_review.get("passed") is True and runtime_review.get("job_id") == MONITOR_JOB
        and runtime_review.get("source_commit") == MONITOR_SOURCE
        and type(runtime_review.get("gpu_count")) is int and runtime_review["gpu_count"] == 0
        and type(runtime_review.get("source_git_archive_sha_and_executable_files_verified")) is int
        and runtime_review["source_git_archive_sha_and_executable_files_verified"] == 265
        and type(runtime_review.get("numerical_or_gpu_operations")) is int and runtime_review["numerical_or_gpu_operations"] == 0,
        "the actual replacement monitor runtime or source approval changed")
    schedule = proof["recurring_session_task"]
    require(schedule["id"] == SELECTED["recurring_session_task_id"] and schedule["cron"] == "*/20 * * * *"
        and type(schedule["cadence_seconds"]) is int and schedule["cadence_seconds"] == 1200
        and schedule["exact_prompt_and_schedule_verified_before_old_deleted"] is True,
        "the exact recurring monitor verification changed")
    verification = checked_ref(monitor["full_verification"])
    monitor_observed = utc_observation(verification["checked_at_utc"], "the actual full monitor")
    require(monitor_observed <= observed, "the full monitor cannot follow the proof's held query")
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
    require(set(query) == {"command", "at_utc", "returncode", "stdout", "stderr", "kind", "job_id", "started_at_utc", "job_action_performed"}
        and query["kind"] == "actual_exact_owned_held_release_query" and query["job_id"] == JOB
        and query["job_action_performed"] is False
        and query["command"] == ["scontrol", "show", "job", "-o", JOB]
        and type(query["returncode"]) is int and query["returncode"] == 0
        and query["at_utc"] == proof["at_utc"], "the saved exact owned held query changed")
    require(utc_observation(query["started_at_utc"], "the actual held-query start") <= observed,
        "the held-query return cannot precede its start")
    require(query["stdout"] == proof["owned_query_stdout"] and query["stderr"] == proof["owned_query_stderr"]
        and ref(proof["owned_query_stdout"]["path"]) == proof["owned_query_stdout"]
        and ref(proof["owned_query_stderr"]["path"]) == proof["owned_query_stderr"]
        and contents(proof["owned_query_stderr"]["path"]) == b"", "the saved owned query bytes changed")
    validate_held(contents(proof["owned_query_stdout"]["path"]).decode("utf-8", errors="strict"))


def validate_frozen_files(frozen):
    """Check shared static bytes and only this target's mutable prelaunch bytes."""
    shared = frozen["shared_files"]
    own = SELECTED["files"]
    require(isinstance(shared, dict) and isinstance(own, dict) and not set(shared).intersection(own),
        "shared and target-owned release inputs must be separate")
    for path, expected in {**shared, **own}.items():
        raw = contents(path)
        require(sha(raw) == expected["sha256"] and len(raw) == expected["bytes"],
            "an immutable shared or selected-target release prerequisite changed: " + path)


def require_unattempted_release():
    """Reject every prior release output, including a dangling pending-input link."""
    for name in ATTEMPT_FILES:
        require(not (PACKET / name).exists() and not (PACKET / name).is_symlink(),
            "a release attempt already exists; never retry")
    final = UNIT / "owned_allocation_inputs.json"
    pending = final.with_name("." + final.name + ".pending")
    require(not final.exists() and not final.is_symlink() and not pending.exists()
        and not pending.is_symlink(), "retain the previous final input publication attempt")


def checked(copies, publication=None):
    """Default checks are local metadata reads, with no scheduler command or intent."""
    require(socket.gethostname().split(".")[0] == "portal05" and os.getuid() == 2618919
        and not NUMERICAL_MODULES.intersection(sys.modules), "wrong release host,owner or numerical import")
    frozen = select(copies)
    frozen_path = UTILITY / "release_frozen_inputs.json"
    require(FROZEN_INPUTS_SHA is not None and sha(contents(frozen_path)) == FROZEN_INPUTS_SHA,
        "the reviewed actual release input snapshot is not final")
    validate_frozen_files(frozen)
    plan = document(UNIT / "measurement_plan.json")
    require(sha(contents(UNIT / "measurement_plan.json")) == PLAN_SHA and plan["helpers"]["source_commit"] == SOURCE
        and plan["kind"] == "selected_configuration_measurement" and plan["mode"] == "confirmation"
        and type(plan["configuration"]["learner"]["n_copies"]) is int and plan["configuration"]["learner"]["n_copies"] == copies
        and plan["copy_seeds"] == list(range(1, copies + 1)) and all(type(seed) is int for seed in plan["copy_seeds"])
        and plan["gpu_uuid"] is None and plan["configured_isolation"]["physical_identity"]["gpu_uuid"] is None,
        "the original unresolved plan changed")
    require(all(type(plan.get(key)) is int and plan[key] == value for key, value in
        (("rounds", 7), ("iterations", 10), ("minimum_updates_per_round", 10)))
        and "common_correctness_inputs" in plan
        and not any(key in plan for key in ("final_confirmation_protocol", "completed_study", "direct_binding", "target_qualification")),
        "only the reviewed intermediate common confirmation is supported")
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
    require(submission["held_command"] == document(PACKET / "held_submission_command.json")["command"]
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
    require(sha(contents(SELECTED["held_proof"]["path"])) == HELD_PROOF_SHA, "the exact held-monitor proof changed")
    validate_proof(document(SELECTED["held_proof"]["path"]))
    review = document(UTILITY / "author_review.json")
    require(review["jobs"] == {"1": "6783295", "2": "6783296"} and review["source_sha256"] == sha(contents(Path(__file__).absolute()))
        and review["release_executed"] is False and review["maximum_startup_wait_seconds"] == 180,
        "the reviewed release implementation changed")
    require_unattempted_release()
    if publication is not None:
        require(re.fullmatch(r"[0-9a-f]{40}", publication), "the full published release commit is required")
        subprocess.run(["git", "-C", str(REPOSITORY), "merge-base", "--is-ancestor", publication,
            "refs/remotes/origin/merge-atari-cleanrl"], check=True)
        paths = [Path(path) for path in {**frozen["shared_files"], **SELECTED["files"]}] + [Path(__file__).absolute(), frozen_path,
            UTILITY / "author_review.json", UTILITY / "test_release_common_confirmation.py"]
        for path in paths:
            require(path.is_relative_to(REPOSITORY) and contents(path) == subprocess.check_output([
                "git", "-C", str(REPOSITORY), "cat-file", "blob", publication + ":" + str(path.relative_to(REPOSITORY))]),
                "an exact release prerequisite was not published: " + str(path))
    return {"schema_version": 1, "kind": "owned_uva_common_confirmation_release_intent", "cluster": "uva",
        "job_id": JOB, "copies": copies, "command": COMMAND, "at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_commit": publication, "submission_receipt_sha256": SUBMISSION_SHA, "plan_sha256": PLAN_SHA,
        "data_unit": str(UNIT), "pre_release_inputs_sha256": PRE_RELEASE_SHA,
        "frozen_inputs_sha256": FROZEN_INPUTS_SHA, "held_monitor_review": ref(Path(SELECTED["held_proof"]["path"])),
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
    require(intent["job_id"] == JOB and intent["command"] == COMMAND, "the selected literal release intent changed")
    require_unattempted_release()
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", type=int, choices=(1, 2), required=True)
    parser.add_argument("--publication-commit")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args(argv)
    require(not arguments.release or arguments.publication_commit is not None,
        "publication is required before the one release")
    intent = checked(arguments.copies, arguments.publication_commit)
    print(json.dumps(release_once(intent) if arguments.release else {"validated": True,
        "copies": arguments.copies, "job_id": JOB, "command": COMMAND, "scheduler_queries": 0,
        "release_calls": 0, "final_inputs_written": False}))


if __name__ == "__main__":
    main()
