"""Submit only this reviewed CPU monitor held, exactly once; never release it."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess

PACKET = Path(__file__).resolve(strict=True).parent
SOURCE = "79a01e851ff92ddba7d0e12eaf8164cab2028878"
PLAN_SHA = "18376135e3c342c9be628a1ab8cf022a34a7942ab76f2e986ae9553b4f3dcb69"
INPUTS_SHA = "fd8138d4df9d1deb02468c47e6f70677d66915df742b1c6b394a1e3da683b609"
ATTEMPTS = ("submission_intent.json", "submission_stdout.raw", "submission_stderr.raw",
    "submission_return.json", "submission_failure.json", "submission_receipt.json", "held_binding.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "exact ordinary path required")
    require(stat.S_ISREG(path.lstat().st_mode), "ordinary file required")
    return path


def read_bytes(path):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "ordinary opened file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        require(len(raw) <= 64 * 1024 * 1024, "metadata/source artifact exceeds bound")
        return raw
    finally:
        os.close(fd)


def receipt(path):
    return {"path": str(path), "sha256": digest(read_bytes(path))}


def checked(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
        and receipt(value["path"]) == value, "exact input bytes changed")
    return read_bytes(value["path"])


def document(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(read_bytes(path).decode("utf-8", errors="strict"), object_pairs_hook=pairs)


def publish_bytes(path, raw):
    path = Path(path); require(path.parent.resolve(strict=True) == path.parent, "ordinary output parent required")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as out:
            out.write(raw); out.flush(); os.fsync(out.fileno())
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return receipt(path)


def publish(path, value):
    return publish_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def unattempted(campaign):
    require(not any((PACKET / name).exists() or (PACKET / name).is_symlink() for name in ATTEMPTS),
        "this immutable packet already has a submission attempt or receipt; never retry")
    for line in read_bytes(campaign / "slurm/submissions.tsv").decode().splitlines():
        require(line.split("\t")[-1] != str(PACKET / "monitor.sbatch"), "a submission row already names this exact packet")


def validate_recovery(inputs):
    """Use reviewed static recovery evidence; no prelaunch full report is claimed."""
    require(inputs["recovery_basis"] == receipt(PACKET / "recovery_basis.json"), "recovery basis changed")
    basis = document(inputs["recovery_basis"]["path"])
    require(type(basis.get("schema_version")) is int and basis["schema_version"] == 1
        and basis["kind"] == "explicit_persistent_cpu_monitor_recovery_basis" and basis["source_commit"] == SOURCE
        and basis["old_failed_job_id"] == "6782816" and basis["unreleased_superseded_job_id"] == "6782952"
        and basis["prelaunch_full_candidate_available"] is False and basis["prelaunch_full_candidate_executed"] is False
        and basis["prelaunch_actual_full_report_available"] is False and basis["old_monitor_running_required"] is False
        and basis["actual_first_full_report_required_after_start"] is True
        and basis["actual_source_and_runtime_review_required_after_start"] is True
        and basis["future_work_blocked_until_first_full_report_independent_review"] is True
        and type(basis["scientific_credit"]) is int and basis["scientific_credit"] == 0 and basis["worker_admitted"] is False,
        "explicit limited recovery requirements changed")
    for value in basis["source_validation"].values():
        checked(value)
        proof = document(value["path"])
        require(proof["passed"] is True and type(proof["tests"]) is int and proof["tests"] == 23
            and proof["source_before"] == proof["source_after"] and proof["protected_before"] == proof["protected_after"]
            and proof["numerical_import_attempts"] == []
            and all(type(proof[key]) is int and proof[key] == 0 for key in
                ("live_scheduler_commands", "gpu_library_calls", "network_actions")),
            "the actual reviewed hostname source/fixture evidence differs")
        manifest = document(PACKET / "source_snapshot.json")
        require(len(proof["source_after"]) == 3
            and all(manifest["files"].get(name, {}).get("sha256") == expected
                for name, expected in proof["source_after"].items()),
            "the tested hostname code and actual registry fixture differ from the published archive")
    require(len(basis["source_validation"]) == 2, "both independent and author static source records are required")
    checked(basis["old_terminal_evidence"])
    failure = document(basis["old_terminal_evidence"]["path"])
    require(failure["job_id"] == "6782816" and failure["old_allocation_usable"] is False
        and failure["actual_current_old_allocation_validation"] is False and failure["new_allocation_identity"] is None,
        "the old failed allocation cannot qualify this recovery")
    for key in ("terminal_query", "terminal_stdout", "terminal_stderr", "historical_runtime"):
        checked(failure[key])
    query = document(failure["terminal_query"]["path"])
    require(type(query["returncode"]) is int and query["returncode"] == 0 and "--jobs=6782816" in query["command"],
        "the saved terminal query is not exact owned evidence")
    rows = [row.split("|") for row in checked(failure["terminal_stdout"]).decode().splitlines()]
    for job in ("6782816", "6782816.batch"):
        found = [row for row in rows if row[0] == job]
        require(len(found) == 1 and found[0][2:4] == ["FAILED", "1:0"], "the preceding monitor is not recorded terminal")
    checked(basis["native_cron"])
    native = document(basis["native_cron"]["path"])["root_reported_current_task"]
    current = document(Path(inputs["campaign_run_dir"]) / "next_actions.json")["monitoring_session_cron"]
    require(native["id"] == current["id"] == "ca8af776-26dd-4ee8-8c00-45ddc8bbc0cd"
        and native["cron"] == current["cron"] == "*/20 * * * *"
        and native["exact_prompt_and_schedule_verified"] is True and current["exact_prompt_and_schedule_verified"] is True,
        "root's actual current recurring monitor recovery basis changed")
    return basis


def recovery_root_approval():
    path = PACKET / "root_recovery_approval.json"
    value = document(path)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "root_approved_static_cpu_monitor_recovery"
        and value.get("approved") is True and value.get("source_commit") == SOURCE and value.get("packet") == str(PACKET)
        and value.get("static_freeze") == receipt(PACKET / "static_source_freeze.json")
        and value.get("prelaunch_full_candidate_available") is False
        and value.get("first_full_report_review_required_after_start") is True,
        "a real published root recovery approval is required")
    return receipt(path)


def validate_inputs():
    require(receipt(PACKET / "held_submission_inputs.json")["sha256"] == INPUTS_SHA, "fixed submission inputs changed")
    inputs = document(PACKET / "held_submission_inputs.json"); campaign = Path(inputs["campaign_run_dir"])
    require(type(inputs["schema_version"]) is int and inputs["schema_version"] == 1
        and inputs["kind"] == "fixed_packet_monitor_held_submission_inputs" and inputs["packet"] == str(PACKET)
        and inputs["source_commit"] == SOURCE and inputs["old_monitor_job_id"] == "6782816",
        "unexpected monitor submission inputs")
    for value in inputs["required_files"]:
        checked(value)
    require(inputs["plan"] == receipt(PACKET / "monitor_replacement_plan.json")
        and inputs["plan"]["sha256"] == PLAN_SHA, "reviewed static recovery plan changed")
    plan = document(inputs["plan"]["path"])
    validate_recovery(inputs)
    roster = document(PACKET / "static_publication_roster.json")
    require(roster["source_file_count"] == 265 and roster["source_commit"] == SOURCE
        and roster["prelaunch_full_candidate_available"] is False, "exact recovery source roster changed")
    for entry in roster["files"]:
        path = ordinary(entry["path"])
        require(digest(read_bytes(path)) == entry["sha256"] and path.stat().st_size == entry["bytes"]
            and format(stat.S_IMODE(path.stat().st_mode), "04o") == entry["mode"], "a reviewed static source byte or mode changed")
    expected_shape = {"account": "sl5nw_base", "partition": "nolim", "qos": "normal", "nodes": 1, "ntasks": 1,
        "cpus_per_task": 1, "host_memory_gib": 2, "gpus_per_node": 0, "time": "7-00:00:00", "interval_seconds": 1200, "requeue": False}
    command = ["sbatch", "--parsable", "--hold", "--no-requeue", "--account=sl5nw_base", "--partition=nolim",
        "--qos=normal", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G", "--gpus-per-node=0",
        "--time=7-00:00:00", "--job-name=pixel_study_monitor",
        "--output=" + str(campaign / "slurm/jobs/monitor_%j.log"), str(PACKET / "monitor.sbatch")]
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1
        and plan["kind"] == "persistent_pixel_study_monitor_replacement" and plan["job_id"] is None
        and plan["campaign_run_dir"] == str(campaign) and plan["helper_commit"] == SOURCE
        and plan["old_monitor_job_id"] == "6782816" and plan["job_shape"] == expected_shape
        and plan["submission_command"] == command and plan["job_script"] == receipt(PACKET / "monitor.sbatch")
        and plan["required_submission_row"] == ["<exact_returned_job_id>", "monitor", SOURCE, command[-1]],
        "the exact zero-GPU held monitor request changed")
    require(plan["prelaunch_full_candidate_available"] is False and plan["actual_first_full_report_review_required_after_start"] is True
        and plan["recovery_basis"] == inputs["recovery_basis"], "actual first-report recovery gate changed")
    unattempted(campaign)
    return inputs, plan


def validate_publication(inputs, commit):
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "one exact published commit is required")
    approval = recovery_root_approval()
    repository = Path(inputs["repository"])
    subprocess.run(["git", "-C", str(repository), "merge-base", "--is-ancestor", commit, "origin/merge-atari-cleanrl"], check=True)
    refs = [approval, *inputs["required_files"], receipt(PACKET / "held_submission_inputs.json"), receipt(Path(__file__)),
        receipt(PACKET / "test_submit_monitor_held.py"), receipt(PACKET / "held_submission_author_review.json")]
    for value in refs:
        path = Path(value["path"]); relative = path.relative_to(repository).as_posix()
        raw = subprocess.check_output(["git", "-C", str(repository), "show", commit + ":" + relative])
        require(digest(raw) == value["sha256"] and receipt(path) == value, "required exact bytes are not in the published commit")
    return commit


def append_ledgers(campaign, job_id, row):
    paths = [campaign / "slurm/submitted_jobids.txt", campaign / "slurm/submitted_jobids_uva.txt", campaign / "slurm/submissions.tsv"]
    with ExitStack() as stack:
        streams = []
        for path in paths:
            ordinary(path)
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
            require(stat.S_ISREG(os.fstat(fd).st_mode), "ordinary ledger required")
            stream = stack.enter_context(os.fdopen(fd, "r+", encoding="utf-8"))
            fcntl.flock(fd, fcntl.LOCK_EX); streams.append(stream)
        before = [stream.read() for stream in streams]
        require(all(job_id not in text.splitlines() for text in before[:2])
            and not any(line.split("\t")[0] == job_id or line.split("\t")[-1] == row[-1] for line in before[2].splitlines()),
            "returned identifier or packet already exists; preserve this partial attempt")
        additions = [job_id, job_id, "\t".join(row)]
        for stream, text, addition in zip(streams, before, additions):
            require(not text or text.endswith("\n"), "ledger is missing its final newline")
            stream.seek(0, os.SEEK_END); stream.write(addition + "\n"); stream.flush(); os.fsync(stream.fileno())
        return [{"path": str(path), "before_sha256": digest(text.encode()), "after_sha256": digest(read_bytes(path))}
            for path, text in zip(paths, before)]


def submit_once(publication_commit):
    inputs, plan = validate_inputs(); validate_publication(inputs, publication_commit)
    require(os.getuid() == os.geteuid() == 2618919 and socket.gethostname().split(".")[0] == "portal05",
        "only the exact campaign owner on portal05 may perform the reviewed submission")
    campaign = Path(inputs["campaign_run_dir"])
    intent = publish(PACKET / "submission_intent.json", {"schema_version": 1, "kind": "exclusive_held_cpu_monitor_submission_intent",
        "at_utc": datetime.now(timezone.utc).isoformat(), "command": plan["submission_command"],
        "publication_commit": publication_commit, "inputs": receipt(PACKET / "held_submission_inputs.json"),
        "old_monitor_preserved": "6782816", "automatic_retry_permitted": False})
    try:
        command = plan["submission_command"]; returned = {"command": command, "at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            process = subprocess.run(command, capture_output=True, timeout=90)
            stdout, stderr = process.stdout, process.stderr
            returned.update(returncode=process.returncode, completed=True)
        except BaseException as error:
            stdout, stderr = getattr(error, "stdout", None) or b"", getattr(error, "stderr", None) or b""
            returned.update(returncode=None, completed=False, error_type=type(error).__name__, error=str(error))
        out_ref = publish_bytes(PACKET / "submission_stdout.raw", stdout)
        err_ref = publish_bytes(PACKET / "submission_stderr.raw", stderr)
        returned.update(stdout=out_ref, stderr=err_ref)
        raw_return = publish(PACKET / "submission_return.json", returned)
        match = re.fullmatch(rb"([1-9][0-9]*)\n?", stdout)
        require(returned["completed"] is True and type(returned["returncode"]) is int and returned["returncode"] == 0
            and stderr == b"" and match is not None, "sbatch returned failure or an ambiguous identifier; never retry")
        job_id = match[1].decode("ascii"); row = [job_id, "monitor", SOURCE, command[-1]]
        ledgers = append_ledgers(campaign, job_id, row)
        record = {"schema_version": 1, "kind": "owned_uva_persistent_monitor_submission", "cluster": "uva",
            "job_id": job_id, "source_commit": SOURCE, "published_preparation_commit": publication_commit,
            "packet": str(PACKET), "data_unit": plan["data_unit"], "submission_row": row, "held": True, "released": False,
            "plan_sha256": inputs["plan"]["sha256"], "manifest_sha256": receipt(PACKET / "manifest.yaml")["sha256"],
            "job_script_sha256": plan["job_script"]["sha256"], "held_command": command, "at_utc": datetime.now(timezone.utc).isoformat(),
            "submission_intent": intent, "submission_return": raw_return, "ledger_appends": ledgers,
            "both_uva_and_global_id_ledgers_fsynced_before_receipt": True, "exact_tsv_row_fsynced_before_receipt": True,
            "automatic_retry_permitted": False, "old_monitor_job_id": "6782816", "old_monitor_changed": False, "prelaunch_full_candidate_available": False, "first_full_report_independent_review_required_after_start": True,
            "scientific_hypothesis_credit": 0, "worker_admission": False}
        submission = publish(PACKET / "submission_receipt.json", record)
        publish(PACKET / "held_binding.json", {"schema_version": 1, "kind": "held_owned_cpu_monitor_binding",
            "job_id": job_id, "source_commit": SOURCE, "packet": str(PACKET), "submission_receipt": submission,
            "plan": inputs["plan"], "command": command, "submission_row": row, "requested_gpu_count": 0,
            "old_monitor_preserved": "6782816", "released": False, "scientific_credit": 0, "worker_admitted": False})
        return record
    except BaseException as error:
        publish(PACKET / "submission_failure.json", {"error_type": type(error).__name__, "error": str(error),
            "intent": intent, "automatic_retry_permitted": False, "old_monitor_changed": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.submit:
        result = submit_once(args.publication_commit)
        print(json.dumps({"job_id": result["job_id"], "held": True, "released": False, "ledgers_fsynced": True}), flush=True)
    else:
        inputs, plan = validate_inputs(); validate_publication(inputs, args.publication_commit)
        print(json.dumps({"validated": True, "submitted": False, "command": plan["submission_command"]}), flush=True)


if __name__ == "__main__":
    main()

