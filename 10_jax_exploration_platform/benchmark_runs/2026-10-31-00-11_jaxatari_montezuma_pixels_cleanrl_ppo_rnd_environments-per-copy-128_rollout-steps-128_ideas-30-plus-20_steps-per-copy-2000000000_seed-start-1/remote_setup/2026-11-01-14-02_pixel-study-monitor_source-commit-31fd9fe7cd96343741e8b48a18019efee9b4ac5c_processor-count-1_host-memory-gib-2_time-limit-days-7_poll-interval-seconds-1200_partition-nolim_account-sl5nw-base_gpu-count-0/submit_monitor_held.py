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
SOURCE = "31fd9fe7cd96343741e8b48a18019efee9b4ac5c"
PLAN_SHA = "909a59d9ad51c0000858ad04c2e464d5a42efbcfbf162d945a91a1fddbc9025a"
INPUTS_SHA = "b56ea6e71f888d70e3959a92b612d96f9d8290aa14cb321d334ef3db29e0fd17"
STATIC_SHA = "43d8b11bb1a735af4374d34be8ae4577a17661a7bb16f1ebb0fe0d8ddf5b0b58"
CANDIDATE_SHA = "2e2b994c327dc8174a9beeb2fa4e2666ed75ef703b8bfb23995bef1cc6876366"
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


def validate_inputs():
    require(receipt(PACKET / "held_submission_inputs.json")["sha256"] == INPUTS_SHA, "fixed submission inputs changed")
    inputs = document(PACKET / "held_submission_inputs.json"); campaign = Path(inputs["campaign_run_dir"])
    require(type(inputs["schema_version"]) is int and inputs["schema_version"] == 1
        and inputs["kind"] == "fixed_packet_monitor_held_submission_inputs" and inputs["packet"] == str(PACKET)
        and inputs["source_commit"] == SOURCE and inputs["old_monitor_job_id"] == "6781231",
        "unexpected monitor submission inputs")
    for value in inputs["required_files"]:
        checked(value)
    checked(inputs["original_candidate_failure_preserved"])
    require(inputs["plan"] == receipt(PACKET / "monitor_replacement_plan.json")
        and inputs["plan"]["sha256"] == PLAN_SHA and inputs["static_review"]["sha256"] == STATIC_SHA
        and inputs["candidate_proof"]["sha256"] == CANDIDATE_SHA, "reviewed packet/proof binding changed")
    plan = document(inputs["plan"]["path"]); static_review = document(inputs["static_review"]["path"])
    proof = document(inputs["candidate_proof"]["path"])
    require(static_review["approved_static_only"] is True and static_review["packet"] == str(PACKET)
        and static_review["source_commit"] == SOURCE and static_review["source_files"] == 260,
        "the independent static review is unrelated")
    require(type(proof["schema_version"]) is int and proof["schema_version"] == 1
        and proof["kind"] == "existing_candidate_monitor_exact_framing_independent_verification"
        and proof["passed"] is True and proof["source_commit"] == SOURCE
        and proof["candidate_job_id"] == "6781231" and proof["candidate_step_id"] == "119"
        and proof["check_count"] == 33 and proof["five_sections_present"] is True
        and proof["actual_counts"] == {"completed": 120, "failed": 0, "pending": 0, "running": 0}
        and proof["candidate_inputs_before"] == proof["candidate_inputs_after"]
        and proof["original_wrapper_verification_failed"] is True and proof["full_tick_rerun"] is False
        and proof["candidate_protected_before_after_hash_maps_saved"] is False
        and proof["actual_candidate_protected_equality_assertion_passed_before_failure"] is True
        and proof["monitor_tick_returned_before_failure"] is True and proof["monitor_complete_return"] is False
        and proof["arbitrary_whitespace_normalization"] is False,
        "the actual reviewed candidate framing proof is incomplete")
    for key in ("new_jobs", "release_calls", "cancel_calls", "network_calls", "scientific_credit", "case_credit", "idea_credit"):
        require(type(proof[key]) is int and proof[key] == 0, "candidate proof grants actions or scientific credit")
    require(proof["worker_admitted"] is False and proof["admission_granted"] is False and proof["numerical_imports"] == 0,
        "candidate is not an inactive metadata-only monitor check")
    for value in proof["candidate_inputs_before"].values():
        checked(value)
    expected_shape = {"account": "sl5nw_base", "partition": "nolim", "qos": "normal", "nodes": 1, "ntasks": 1,
        "cpus_per_task": 1, "host_memory_gib": 2, "gpus_per_node": 0, "time": "7-00:00:00", "interval_seconds": 1200, "requeue": False}
    command = ["sbatch", "--parsable", "--hold", "--no-requeue", "--account=sl5nw_base", "--partition=nolim",
        "--qos=normal", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G", "--gpus-per-node=0",
        "--time=7-00:00:00", "--job-name=pixel_study_monitor",
        "--output=" + str(campaign / "slurm/jobs/monitor_%j.log"), str(PACKET / "monitor.sbatch")]
    require(type(plan["schema_version"]) is int and plan["schema_version"] == 1
        and plan["kind"] == "persistent_pixel_study_monitor_replacement" and plan["job_id"] is None
        and plan["campaign_run_dir"] == str(campaign) and plan["helper_commit"] == SOURCE
        and plan["old_monitor_job_id"] == "6781231" and plan["job_shape"] == expected_shape
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
        "old_monitor_preserved": "6781231", "automatic_retry_permitted": False})
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
            "automatic_retry_permitted": False, "old_monitor_job_id": "6781231", "old_monitor_changed": False,
            "scientific_hypothesis_credit": 0, "worker_admission": False}
        submission = publish(PACKET / "submission_receipt.json", record)
        publish(PACKET / "held_binding.json", {"schema_version": 1, "kind": "held_owned_cpu_monitor_binding",
            "job_id": job_id, "source_commit": SOURCE, "packet": str(PACKET), "submission_receipt": submission,
            "plan": inputs["plan"], "command": command, "submission_row": row, "requested_gpu_count": 0,
            "old_monitor_preserved": "6781231", "released": False, "scientific_credit": 0, "worker_admitted": False})
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

