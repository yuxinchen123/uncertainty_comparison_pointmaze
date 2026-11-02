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
SOURCE = "7fd40de24201b5e1efce398f49549db26c1e94d9"
PLAN_SHA = "0233969400188d7506077921711880eb73661462d18fa50cf2749ce07e87cd44"
INPUTS_SHA = "e1363e27ad6615277ff959e131231642f69281f8d003339977b8e04254a16cc3"
STATIC_SHA = "d322a67ed7b9422beecd45b0c6063df71e418c122cfbfa1e74aac6bc94d3ba91"
CANDIDATE_SHA = "53d2b06fed87889cdb75c02f51fc0cc510487fee3dacf0aab8878973d3e5483e"
ROOT_APPROVAL_SHA = "28d93b8a2c1a3b1523a5485d4ddd9b47d12d987d53eddb811cda4a7c9e30691d"
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


def validate_candidate(proof):
    """Validate the actual successful full tick, without re-reading mutable live reports."""
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "frozen_published_monitor_candidate_full_tick_verification"
        and proof.get("passed") is True and proof.get("source_commit") == SOURCE
        and proof.get("actual_job_id") == "6782816" and proof.get("actual_step_id") == "34"
        and proof.get("new_monitor_job_id") is None and proof.get("candidate_full_tick_ran") is True
        and proof.get("complete_study") is False and proof.get("old_monitor_preserved") == "6782816",
        "the actual full candidate tick identity is unrelated or incomplete")
    expected = {"completed": 120, "failed": 0, "pending": 0, "running": 0}
    for field in ("queue_counts_before", "queue_counts_after"):
        counts = proof.get(field)
        require(isinstance(counts, dict) and counts == expected
            and all(type(value) is int for value in counts.values()), "candidate study counts changed")
    before, after = proof.get("protected_before"), proof.get("protected_after")
    campaign = PACKET.parent.parent
    require(isinstance(before, dict) and before == after
        and set(before) == {str(campaign / name) for name in
                           ("next_actions.json", "campaign_provenance.json", "SOURCE_AND_GATES_READY.json")}
        and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in before.values()),
        "the actual candidate protected before/after maps changed")
    visible = proof.get("source_visibility", {})
    require(visible.get("source_commit") == SOURCE and type(visible.get("files_verified")) is int
        and visible["files_verified"] == 263 and visible.get("numerical_imports") is False,
        "candidate source visibility proof differs")
    require(proof.get("blocked_action_attempts") == [] and proof.get("numerical_imports") is False
        and proof.get("network_connections") is False and proof.get("worker_admission") is False,
        "candidate attempted a forbidden action or admission")
    for key in ("new_jobs_submitted", "jobs_released_or_cancelled", "scientific_case_credit",
                "scientific_hypothesis_credit", "final_training_credit"):
        require(type(proof.get(key)) is int and proof[key] == 0, "candidate grants action or scientific credit")
    for key in ("all_five_status_sections_present", "report_and_status_are_this_process_exact_real_writes",
                "printed_table_final_newline_removed_once_for_report_framing",
                "latest_global_report_still_matches_candidate", "latest_global_status_still_matches_candidate"):
        require(proof.get(key) is True, "candidate full report/status observation is incomplete")
    require(proof.get("write_observer_changes_bytes") is False,
        "the candidate observer cannot alter actual report bytes")
    return proof


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
        and inputs["plan"]["sha256"] == PLAN_SHA and inputs["static_review"]["sha256"] == STATIC_SHA
        and inputs["candidate_proof"]["sha256"] == CANDIDATE_SHA, "reviewed packet/proof binding changed")
    plan = document(inputs["plan"]["path"]); static_review = document(inputs["static_review"]["path"])
    proof = document(inputs["candidate_proof"]["path"])
    require(type(static_review.get("schema_version")) is int and static_review["schema_version"] == 1
        and static_review.get("kind") == "independent_direct_reader_monitor_static_source_and_callback_review"
        and static_review.get("approved") is True and static_review["packet"] == str(PACKET)
        and static_review["source_commit"] == SOURCE and static_review["source_files"] == 263
        and static_review.get("all301static_files_modes_preserved") is True,
        "the independent static review is unrelated")
    validate_candidate(proof)
    approval = document(inputs["root_candidate_approval"]["path"])
    require(inputs["root_candidate_approval"] == receipt(PACKET / "root_candidate_approval.json")
        and inputs["root_candidate_approval"]["sha256"] == ROOT_APPROVAL_SHA
        and type(approval.get("schema_version")) is int and approval["schema_version"] == 1
        and approval.get("kind") == "root_approved_actual_frozen_monitor_candidate"
        and approval.get("approved") is True and approval["source_commit"] == SOURCE
        and approval["actual_job_id"] == "6782816" and approval["actual_step_id"] == "34"
        and approval["static_independent_review"] == inputs["static_review"]
        and approval["actual_candidate_verification"] == inputs["candidate_proof"]
        and approval["source_file_count"] == 263 and approval["all_five_status_sections"] is True
        and approval["protected_before"] == approval["protected_after"] == proof["protected_before"]
        and approval["own_process_write_capture_exact"] is True
        and approval["current_source_admission_changed"] is False
        and type(approval["jobs_submitted_released_cancelled"]) is int and approval["jobs_submitted_released_cancelled"] == 0
        and type(approval["scientific_credit"]) is int and approval["scientific_credit"] == 0
        and approval["worker_admitted"] is False and approval["admission_granted"] is False,
        "the root's actual candidate approval changed or is unrelated")
    for value in approval["actual_artifacts"].values():
        checked(value)
    roster = document(PACKET / "static_publication_roster.json")
    require(len(roster["files"]) == 301 and roster["source_files"] == 263 and roster["source_commit"] == SOURCE,
        "the exact static packet/source roster changed")
    for entry in roster["files"]:
        path = ordinary(entry["path"])
        require(digest(read_bytes(path)) == entry["sha256"] and path.stat().st_size == entry["bytes"]
            and format(stat.S_IMODE(path.stat().st_mode), "04o") == entry["mode"],
            "a static packet/source byte, size or executable mode changed")
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
    unattempted(campaign)
    return inputs, plan


def validate_publication(inputs, commit):
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "one exact published commit is required")
    repository = Path(inputs["repository"])
    subprocess.run(["git", "-C", str(repository), "merge-base", "--is-ancestor", commit, "origin/merge-atari-cleanrl"], check=True)
    refs = [*inputs["required_files"], receipt(PACKET / "held_submission_inputs.json"), receipt(Path(__file__)),
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
            "automatic_retry_permitted": False, "old_monitor_job_id": "6782816", "old_monitor_changed": False,
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

