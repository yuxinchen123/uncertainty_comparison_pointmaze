"""Check the prepared CPU monitor source, exact resource records and actual candidate report metadata."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile

PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
COMMIT = "b431573a272ccec354cfbccb2b7528b4b846e7b9"


def digest(path):
    """Read only one small prepared metadata or source artifact at a time."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    """Describe the exact prepared artifact inspected by this check."""
    return {"path": str(path), "sha256": digest(path)}


def require(value, message):
    """Stop on any mismatch rather than changing a prepared source or record."""
    if not value:
        raise ValueError(message)


def functions(raw):
    """Retain every guard and source-verifier function body for an exact reviewed comparison."""
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
            if isinstance(node, ast.FunctionDef)}


def main():
    """Verify the fixed packet on owned compute, without scheduling, GPU or scientific operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-freeze-sha256", required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ.get("SLURM_STEP_ID", "").isdigit(),
            "the exact existing owned CPU allocation and numeric step are required")
    require(not os.path.lexists(args.receipt), "preserve the earlier monitor metadata receipt")
    freeze_path = PACKET / "static_source_freeze.json"
    require(digest(freeze_path) == args.source_freeze_sha256, "the exact prepared source freeze changed")
    frozen = json.loads(freeze_path.read_text())
    checked = 0
    for name, expected in frozen["files"].items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path,
                "a frozen path is not an exact ordinary file")
        require(digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"]
                and oct(stat.S_IMODE(path.stat().st_mode)) == expected["mode"], "a frozen prepared artifact changed")
        checked += 1
    manifest = json.loads((PACKET / "source_snapshot.json").read_text())
    require(manifest["source_commit"] == COMMIT and manifest["working_tree_bytes_used"] is False,
            "the source snapshot must come only from the fixed committed source")
    source = Path(manifest["source_root"])
    actual_paths = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    require(actual_paths == set(manifest["files"]), "the prepared source snapshot has an extra or missing file")
    blobs = 0
    for name, expected in manifest["files"].items():
        path = source / name
        raw = path.read_bytes()
        require(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == expected["git_blob"]
                and hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"],
                "the prepared source differs from its committed Git blob")
        require(bool(path.stat().st_mode & 0o111) == (expected["git_mode"] == "100755"), "a source executable mode changed")
        blobs += 1
    archive_members = set()
    with tarfile.open(PACKET / "source_git_archive.tar", "r:") as archive:
        for item in archive:
            if item.isdir():
                continue
            parts = Path(item.name).parts
            require(parts[0] == COMMIT and ".." not in parts and item.isfile(), "invalid committed archive member")
            name = Path(*parts[1:]).as_posix()
            require(name in manifest["files"] and name not in archive_members, "extra or repeated archive member")
            require(hashlib.sha256(archive.extractfile(item).read()).hexdigest() == manifest["files"][name]["sha256"],
                    "the archived bytes differ from the prepared source")
            archive_members.add(name)
    require(archive_members == set(manifest["files"]), "the archive is incomplete")
    plan = json.loads((PACKET / "monitor_replacement_plan.json").read_text())
    config = json.loads((PACKET / "config_resolved.yaml").read_text())
    resource_shape = {"account": "sl5nw_base", "partition": "nolim", "qos": "normal", "nodes": 1, "ntasks": 1,
        "cpus_per_task": 1, "host_memory_gib": 2, "gpus_per_node": 0, "time": "7-00:00:00",
        "interval_seconds": 1200, "requeue": False}
    require(plan["helper_commit"] == config["source_commit"] == COMMIT and plan["job_shape"] == resource_shape,
            "the reviewed exact source or resource shape changed")
    require(all(config[name] == value for name, value in resource_shape.items()), "the resolved resource shape changed")
    require(all(plan[name] is None and config[name] is None for name in ("job_id", "gpu_uuid", "reservation")),
            "preparation cannot invent an allocation or GPU binding")
    require(plan["old_monitor_job_id"] == config["old_monitor_job_id"] == "6783003", "the standing monitor changed")
    command = plan["submission_command"]
    expected = ["sbatch", "--parsable", "--hold", "--no-requeue", "--account=sl5nw_base", "--partition=nolim",
        "--qos=normal", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G", "--gpus-per-node=0",
        "--time=7-00:00:00", "--job-name=pixel_study_monitor",
        "--output=" + str(RUN / "slurm/jobs/monitor_%j.log"), str(PACKET / "monitor.sbatch")]
    require(command == expected, "the held CPU submission command changed")
    import shlex
    require(shlex.split((PACKET / "command.txt").read_text()) == expected, "the standard command record differs")
    standard = ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt", "progress.json")
    for name in standard:
        require((PACKET / name).read_bytes() == (UNIT / name).read_bytes(), "packet and unit standard records differ")
    background = (PACKET / "experiment_background.md").read_text()
    require([line for line in background.splitlines() if line.startswith("## ")] ==
            ["## Purpose", "## Key hyperparameters", "## Code and config changes", "## Git state"],
            "the four required background sections changed")
    basis = json.loads((PACKET / "standing_monitor_basis.json").read_text())
    old = Path(basis["packet"])
    require(functions((PACKET / "allocation_guard.py").read_text()) == functions((old / "allocation_guard.py").read_text()),
            "the reviewed allocation guard functions changed")
    require(functions((PACKET / "verify_source.py").read_text()) == functions((old / "verify_source.py").read_text()),
            "the reviewed source verifier functions changed")
    handover = json.loads((PACKET / "handover_plan.json").read_text())
    require(handover["old_monitor_preserved"] is True and handover["root_only_submission_and_release"] is True
            and handover["canonical_next_actions_and_provenance_not_changed"] is True
            and handover["new_monitor_job_id"] is None, "the required independent monitor handover policy changed")
    candidate_path = PACKET / "candidate_after_reporting_index_full_report_receipt.json"
    candidate = json.loads(candidate_path.read_text())
    require(candidate["passed"] is True and candidate["full_tick_completed"] is True
            and candidate["source_commit"] == COMMIT and candidate["job_id"] == "6783003",
            "the actual full candidate did not pass on the exact existing owned allocation")
    require(candidate["report_sections"] == ["TRAINING", "WAITING", "HELD", "FINISHED", "PLANNED"],
            "the full candidate is missing a required status section")
    require(candidate["queue_counts"] == {"completed": 120, "pending": 0, "running": 0, "failed": 0}
            and candidate["phase"] == "post_initial_validation_pending", "the actual full candidate state changed")
    for name in ("stdout", "stderr", "report", "status"):
        require(reference(Path(candidate[name]["path"])) == candidate[name], "an actual full candidate artifact changed")
    require(not candidate["numerical_modules_imported"] and candidate["new_jobs_submitted"] == 0
            and candidate["actual_new_monitor_runtime_verified"] is False and candidate["actual_new_first_report_verified"] is False,
            "the candidate must not claim new-allocation runtime or scientific authority")
    status = json.loads(Path(candidate["status"]["path"]).read_text())
    requirements = {row["name"]: row for row in status["campaign"]["post_initial_reporting_requirements"]}
    require(requirements["common_correctness"]["missing"] == []
            and requirements["common_confirmation"]["missing"] == [
                {"hardware_class": "h100", "copies": 1}, {"hardware_class": "h100", "copies": 2}]
            and len(requirements["kernel_profiles"]["missing"]) == len(requirements["packing"]["missing"]) == 2
            and all(not row["errors"] and row["scientific_credit"] == 0 and row["admission_authority"] is False
                    for row in requirements.values()), "the candidate display coverage or remaining work changed")
    value = {"schema_version": 1, "kind": "prepared_cpu_monitor_packet_metadata_check", "passed": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
        "step_id": os.environ["SLURM_STEP_ID"], "source_commit": COMMIT, "source_files_verified": blobs,
        "archive_members_verified": len(archive_members), "frozen_artifacts_verified": checked,
        "source_freeze": reference(freeze_path), "plan": reference(PACKET / "monitor_replacement_plan.json"),
        "candidate_full_report": reference(candidate_path), "allocation_guard_functions_unchanged": True,
        "source_verifier_functions_unchanged": True, "standard_records_match": True,
        "actual_new_monitor_runtime_verified": False, "actual_new_first_report_verified": False,
        "old_monitor_to_preserve": "6783003", "actual_job_actions": 0, "numerical_imports": False, "scientific_credit": 0}
    with args.receipt.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"passed": True, "receipt": reference(args.receipt), "source_files": blobs,
        "frozen_artifacts": checked, "step_id": value["step_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()

