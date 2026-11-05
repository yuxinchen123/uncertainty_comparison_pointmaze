"""Prepare one reviewed-source persistent CPU monitor packet without submitting it."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import tarfile

PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
COMMIT = "548cbcaef4c1fc28f80817ec9ba93f1e6b2fee4e"
REPOSITORY = Path("/p/rlprojects/RND_worktrees/2026-11-05-09-23_pixel-study-monitor_source-548cbcaef4c1fc28f80817ec9ba93f1e6b2fee4e_processor-count-1_host-memory-gib-2_gpu-count-0")
PREFIXES = ["10_jax_exploration_platform/src", "10_jax_exploration_platform/scripts"]


def write(path, raw, mode=0o644):
    """Fsync fresh packet metadata without replacing any prior preparation output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def document(path, value):
    """Write complete JSON records, also valid YAML, using exclusive creation."""
    write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def receipt(path):
    """Return the exact bytes hash of one prepared ordinary file."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git(*arguments):
    """Read the fixed detached source without modifying its index or references."""
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(REPOSITORY), *arguments],
                                   env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})


def seconds(value):
    """Read the exact partition duration, including an unlimited partition."""
    if value in {"UNLIMITED", "INFINITE"}:
        return None
    matched = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if matched is None:
        raise ValueError("unsupported actual partition duration")
    days, hours, minutes, second = map(int, [matched.group(1) or 0, *matched.groups()[1:]])
    if minutes >= 60 or second >= 60 or matched.group(1) is not None and hours >= 24:
        raise ValueError("invalid actual partition duration")
    return days * 86400 + hours * 3600 + minutes * 60 + second


def functions(raw):
    """Compare the already reviewed guard and verifier function bodies without metadata constants."""
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
            if isinstance(node, ast.FunctionDef)}


def main():
    """Freeze committed source and the existing resource guard before a later actual full candidate tick."""
    if os.environ.get("SLURM_JOB_ID") != "6821437" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("preparation must run in an actual numeric step of owned CPU allocation 6821437")
    if os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("the actual campaign owner is required")
    if git("rev-parse", "HEAD").decode().strip() != COMMIT or git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("the fixed source checkout must be clean at the exact requested commit")
    if subprocess.run(["git", "-C", str(REPOSITORY), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 1:
        raise ValueError("the fixed source checkout must be detached")

    # Preserve the standing monitor and verify only its exact existing owned ledger member.
    actions = json.loads((RUN / "next_actions.json").read_text())
    current = actions["persistent_monitor"]
    if str(current["job_id"]) != "6821437":
        raise ValueError("the currently reviewed monitor changed before preparation")
    old = Path(current["packet"])
    ledger = RUN / "slurm/submitted_jobids_uva.txt"
    rows = [line.split("\t") for line in (RUN / "slurm/submissions.tsv").read_text().splitlines()
            if line.split("\t")[0] == "6821437"]
    if ledger.read_text().splitlines().count("6821437") != 1 or rows != [
            ["6821437", "monitor", current["source_commit"], str(old / "monitor.sbatch")]]:
        raise ValueError("the standing monitor lacks its exact unique owned ledger row")
    query = ["scontrol", "show", "job", "6821437", "-o"]
    actual = subprocess.run(query, capture_output=True, text=True, check=True, timeout=20, env={**os.environ, "TZ": "UTC"})
    fields = dict(token.split("=", 1) for token in actual.stdout.split() if "=" in token)
    if fields.get("JobId") != "6821437" or fields.get("UserId") != "sl5nw(2618919)" or fields.get("JobState") != "RUNNING":
        raise ValueError("the exact old owned monitor is not currently running")
    document(PACKET / "standing_monitor_basis.json", {"job_id": "6821437", "packet": str(old),
        "source_commit": current["source_commit"], "command": query, "returncode": actual.returncode,
        "stdout": actual.stdout, "stderr": actual.stderr, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": receipt(old / "runtime_allocation.json"),
        "runtime_and_first_full_review": {"path": current["runtime_and_first_full_report_review"],
            "sha256": current["runtime_and_first_full_report_review_sha256"]},
        "old_monitor_kept_running": True, "new_monitor_job_id": None})

    # Before: commit tree entries; after: an exact ordinary-file source snapshot with Git modes and hashes.
    archive = git("archive", "--format=tar", "--prefix=" + COMMIT + "/", COMMIT, *PREFIXES)
    write(PACKET / "source_git_archive.tar", archive)
    entries = {}
    for entry in git("ls-tree", "-rz", COMMIT, "--", *PREFIXES).split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            mode, kind, blob = header.decode().split()
            if mode not in {"100644", "100755"} or kind != "blob":
                raise ValueError("the monitor snapshot requires ordinary committed files")
            entries[name.decode()] = {"git_mode": mode, "git_blob": blob}
    source = PACKET / "source_snapshot" / COMMIT
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for member in stream:
            parts = Path(member.name).parts
            if not parts or parts[0] != COMMIT or ".." in parts or member.name.startswith("/"):
                raise ValueError("the committed source archive contains an invalid path")
            if member.isdir():
                continue
            name = Path(*parts[1:]).as_posix()
            if not member.isfile() or name not in entries or name in seen:
                raise ValueError("the committed source archive member is unsupported or repeated")
            raw = stream.extractfile(member).read()
            record = entries[name]
            if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != record["git_blob"]:
                raise ValueError("an archived source differs from its committed Git blob")
            write(source / name, raw, 0o755 if record["git_mode"] == "100755" else 0o644)
            record.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            seen.add(name)
    if seen != set(entries):
        raise ValueError("the committed source snapshot is incomplete")
    document(PACKET / "source_snapshot.json", {"schema_version": 1,
        "kind": "complete_committed_monitor_source_archive", "source_commit": COMMIT,
        "source_root": str(source), "source_prefixes": PREFIXES, "files": entries,
        "archive": receipt(PACKET / "source_git_archive.tar"), "working_tree_bytes_used": False,
        "clean_detached_source_root": str(REPOSITORY)})

    # The resource contract is unchanged; only the exact new packet and source constants differ.
    guard_old = (old / "allocation_guard.py").read_text()
    guard = guard_old.replace(str(old), str(PACKET)).replace(current["source_commit"], COMMIT)
    if functions(guard_old) != functions(guard):
        raise ValueError("the reviewed allocation guard functions changed")
    write(PACKET / "allocation_guard.py", guard.encode())
    partition_command = ["scontrol", "show", "partition", "nolim", "-o"]
    partition = subprocess.run(partition_command, capture_output=True, text=True, check=True, timeout=20)
    partition_fields = dict(token.split("=", 1) for token in partition.stdout.split() if "=" in token)
    maximum, requested = seconds(partition_fields["MaxTime"]), seconds("7-00:00:00")
    if partition_fields["State"] != "UP" or maximum is not None and requested > maximum:
        raise ValueError("the actual partition does not admit a seven-day request")
    document(PACKET / "duration_cap_proof.json", {"command": partition_command, "returncode": partition.returncode,
        "stdout": partition.stdout, "stderr": partition.stderr, "actual_partition_max_time": partition_fields["MaxTime"],
        "actual_partition_max_seconds": maximum, "requested_time_limit": "7-00:00:00",
        "requested_seconds": requested, "within_actual_cap": True})
    shape = {"account": "sl5nw_base", "partition": "nolim", "qos": "normal", "nodes": 1, "ntasks": 1,
        "cpus_per_task": 1, "host_memory_gib": 2, "gpus_per_node": 0, "time": "7-00:00:00",
        "interval_seconds": 1200, "requeue": False}
    cpu_policy = {"requested_cpus": 1, "accepted_allocated_logical_cpus": [1, 2],
        "two_requires_actual_node_threads_per_core": 2, "requested_and_allocated_nodes": 1,
        "requested_and_allocated_memory": "2G", "requested_and_allocated_gpus": 0}
    monitor = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py"
    loop = ["/usr/bin/python3", "-B", str(monitor), "--run-dir", str(RUN), "--loop", "--interval", "1200"]
    held = ["sbatch", "--parsable", "--hold", "--no-requeue", "--account=sl5nw_base", "--partition=nolim",
        "--qos=normal", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G",
        "--gpus-per-node=0", "--time=7-00:00:00", "--job-name=pixel_study_monitor",
        "--output=" + str(RUN / "slurm/jobs/monitor_%j.log"), str(PACKET / "monitor.sbatch")]
    config = {"campaign_run_dir": str(RUN), "source_commit": COMMIT, "old_monitor_job_id": "6821437",
        "job_id": None, "gpu_uuid": None, "reservation": None, **shape, "python": "/usr/bin/python3",
        "monitor_command": loop, "cpu_allocation_policy": cpu_policy, "component_only": True,
        "duration_cap_proof": receipt(PACKET / "duration_cap_proof.json"),
        "scientific_case_credit": 0, "scientific_hypothesis_credit": 0, "final_training_credit": 0,
        "worker_admission": False}
    document(PACKET / "config_resolved.yaml", config)
    background = ("## Purpose\n\nPrepare a replacement for the standing persistent CPU monitor from the reviewed source, "
        "including the corrected reporting of remaining validation work. The old monitor stays running.\n\n"
        "## Key hyperparameters\n\nOne requested CPU, two GiB host memory, zero GPUs, seven days, no requeue, "
        "and one complete report every 1,200 seconds on the nolim partition under sl5nw_base with normal quality of service.\n\n"
        "## Code and config changes\n\nBefore: the standing monitor uses source " + current["source_commit"] +
        ". After: this packet uses source " + COMMIT + ". The reviewed allocation guard functions and resource shape are unchanged. "
        "The candidate is checked on the existing CPU allocation. A future actual replacement must pass independent runtime and first-full-report review "
        "before the old monitor is stopped, after its auxiliary steps finish. This infrastructure work grants zero experiment credit.\n\n"
        "## Git state\n\nCommit: `" + COMMIT + "`. The preparation source is a clean detached checkout.\n")
    write(PACKET / "experiment_background.md", background.encode())
    write(PACKET / "command.txt", (shlex.join(held) + "\n").encode())
    expected = {name: hashlib.sha256((PACKET / name).read_bytes()).hexdigest()
        for name in ("source_snapshot.json", "config_resolved.yaml", "allocation_guard.py", "source_git_archive.tar")}
    verifier_old = (old / "verify_source.py").read_text()
    verifier = re.sub(r"^PACKET=.*$", "PACKET=Path(" + repr(str(PACKET)) + ")", verifier_old, flags=re.M)
    verifier = re.sub(r"^EXPECTED=.*$", "EXPECTED=" + repr(expected), verifier, flags=re.M)
    if functions(verifier_old) != functions(verifier):
        raise ValueError("the reviewed source verifier functions changed")
    write(PACKET / "verify_source.py", verifier.encode())
    shell = ("#!/bin/bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n"
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1\n"
        "printf '%s  %s\\n' " + shlex.quote(hashlib.sha256(verifier.encode()).hexdigest()) + " " +
        shlex.quote(str(PACKET / "verify_source.py")) + " | sha256sum --check --status\n" +
        shlex.join(["/usr/bin/python3", "-B", str(PACKET / "verify_source.py")]) + "\n" +
        shlex.join(["/usr/bin/python3", "-B", str(PACKET / "allocation_guard.py")]) + "\nexec " + shlex.join(loop) + "\n")
    write(PACKET / "monitor.sbatch", shell.encode(), 0o755)
    plan = {"schema_version": 1, "kind": "persistent_pixel_study_monitor_replacement",
        "state": "prepared, not submitted; full candidate proof pending",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "campaign_run_dir": str(RUN),
        "helper_commit": COMMIT, "source": str(source / "10_jax_exploration_platform"),
        "source_manifest": receipt(PACKET / "source_snapshot.json"), "source_file_count": len(entries),
        "job_shape": shape, "job_id": None, "gpu_uuid": None, "reservation": None,
        "data_unit": {"name": UNIT.name, "path": str(UNIT)}, "job_script": receipt(PACKET / "monitor.sbatch"),
        "submission_command": held, "old_monitor_job_id": "6821437", "cadence_seconds": 1200,
        "cpu_allocation_policy": cpu_policy, "duration_cap_proof": receipt(PACKET / "duration_cap_proof.json"),
        "prelaunch_full_candidate_available": True, "prelaunch_full_candidate_executed": False,
        "actual_first_full_report_review_required_after_start": True,
        "required_submission_row": ["<exact_returned_job_id>", "monitor", COMMIT, str(PACKET / "monitor.sbatch")],
        "resource_basis": {"current_monitor_runtime": receipt(old / "runtime_allocation.json"),
            "current_monitor_plan": receipt(old / "monitor_replacement_plan.json"),
            "standing_monitor_basis": receipt(PACKET / "standing_monitor_basis.json")},
        "scientific_case_credit": 0, "scientific_hypothesis_credit": 0,
        "final_training_credit": 0, "worker_admission": False}
    document(PACKET / "monitor_replacement_plan.json", plan)
    manifest = {"schema_version": 1, "kind": "prepared_persistent_pixel_study_monitor_replacement",
        "source_commit": COMMIT, "plan": receipt(PACKET / "monitor_replacement_plan.json"),
        "submission_command": held, "job_id": None, "gpu_uuid": None, "reservation": None,
        "scientific_case_credit": 0, "scientific_hypothesis_credit": 0, "final_training_credit": 0, "worker_admission": False}
    document(PACKET / "manifest.yaml", manifest)
    progress = {"kind": "persistent_cpu_monitor_replacement", "state": "planned", "job_id": None,
        "source_commit": COMMIT, "old_monitor_job_id": "6821437",
        "scientific_case_credit": 0, "scientific_hypothesis_credit": 0,
        "final_training_credit": 0, "worker_admission": False}
    document(PACKET / "progress.json", progress)
    for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json"):
        write(UNIT / name, (PACKET / name).read_bytes())
    document(PACKET / "handover_plan.json", {"old_monitor_job_id": "6821437", "new_monitor_job_id": None,
        "old_monitor_preserved": True, "root_only_submission_and_release": True,
        "required_before_adoption": ["actual runtime and exact source proof", "actual first full report",
            "independent complete runtime and first-report review"],
        "required_before_stopping_old": ["independent replacement approval", "fresh exact old owned auxiliary-step drain proof"],
        "canonical_next_actions_and_provenance_not_changed": True, "scientific_credit": 0})
    subprocess.run(["bash", "-n", str(PACKET / "monitor.sbatch")], check=True)
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("the detached source changed during preparation")
    document(PACKET / "preparation_receipt.json", {"schema_version": 1, "passed": True,
        "kind": "published_source_cpu_monitor_preparation", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"], "host": socket.gethostname(),
        "source_commit": COMMIT, "clean_detached_source_root": str(REPOSITORY), "source_files": len(entries),
        "source_manifest": receipt(PACKET / "source_snapshot.json"), "plan": receipt(PACKET / "monitor_replacement_plan.json"),
        "manifest": receipt(PACKET / "manifest.yaml"), "job_script": receipt(PACKET / "monitor.sbatch"),
        "unchanged_allocation_guard_functions": True, "unchanged_source_verifier_functions": True,
        "new_jobs_submitted": 0, "old_monitor_to_preserve": "6821437",
        "canonical_next_actions_and_provenance_not_changed": True, "candidate_full_tick_pending": True})
    print(json.dumps({"packet": str(PACKET), "unit": str(UNIT), "source_files": len(entries),
                      "plan": receipt(PACKET / "monitor_replacement_plan.json"), "submitted": False}))


if __name__ == "__main__":
    main()

