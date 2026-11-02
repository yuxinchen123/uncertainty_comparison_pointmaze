"""Finalize only the new controller preparation records from actual metadata results."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

P = Path(__file__).absolute().parent
record = json.loads((P / "controller_metadata_tests/author_6782816_30.json").read_text())
assert record["passed"] is True and record["tests_run"] == 18 and record["job_id"] == "6782816" and record["step_id"] == "30"
assert record["original_66_before"] == record["original_66_after"]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    raw = value.encode() if isinstance(value, str) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
def receipt(path):
    return {"path": str(path), "sha256": sha(path)}

write(P / "controller_record_finalization_intent.json", {
    "kind": "metadata_only_controller_record_finalization", "at_utc": datetime.now(timezone.utc).isoformat(),
    "original_packet_modified": False, "actual_target_process_started": False})
archive = P / "controller_source_versions/initial_controller_standard_records"
archive.mkdir()
standard = P / "controller_preparation"
for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
    target = archive / name
    write(target, (standard / name).read_text()); target.chmod(0o444)
boundary = {"scientific_credit": 0, "idea_credit": 0, "case_credit": 0, "worker_admitted": False,
            "admission_granted": False, "qualification_performed": False}
base = {"schema_version": 1, "kind": "direct_qualification_controller_metadata_preparation",
    "actual_direct_contact": False, "actual_target_process_started": False, "original_66_unchanged": True,
    "helper_commit": "44f7d99db22c01bfd8beefefc5f548f5517d5348",
    "actual_metadata_verification": receipt(P / "controller_metadata_tests/author_6782816_30.json"),
    "initial_failed_metadata_verification": receipt(P / "controller_metadata_tests/author_6782816_26.json"),
    "focused_fixture_correction_verification": receipt(P / "controller_metadata_tests/author_6782816_27.json"),
    "controller": receipt(P / "direct_qualification_controller.py"),
    "tests": receipt(P / "test_direct_qualification_controller.py"),
    "original_packet_freeze": receipt(P / "packet_metadata_freeze.json"), **boundary}
configuration = {**base, "actual_target": {"hostname": "serval02", "uid": 2618919, "gpu_count": 1,
    "gpu_uuid": "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3", "actual_cpu_affinity_count": 8,
    "cpu_indices": "chosen only from the actual controller's available mask at launch",
    "exclusive_cpu_claim": False, "slurm_allocation_claim": False,
    "controller_owns_gpu_standing_descriptor": False, "producer_owns_gpu_standing_descriptor": True,
    "actual_pid": None, "actual_birth": None},
    "metadata_test_work": {"full_corrected_tests": 18, "allocation": "6782816", "step": "30",
        "private_cpu_child_spawns": len(record["private_child_actions"]),
        "actual_fixture_cpu_counts": sorted({len(value["observed_child"]["affinity"])
            for fixture in record["private_cpu_fixtures"] for name, value in fixture["files"].items()
            if name == "direct_controller_spawn.json"}),
        "numerical_work": False, "real_target_runtime_proved": False}}
python = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
command = "# Actual corrected CPU metadata verification; do not rerun solely to replace a receipt.\n"
command += shlex.join(["srun", "--jobid=6782816", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", python,
    "-I", "-B", str(P / "run_direct_controller_metadata_tests.py")]) + "\n"
command += "# Local default check only; starts no process and performs no host/scheduler/device query.\n"
command += shlex.join([python, "-I", "-B", str(P / "direct_qualification_controller.py")]) + "\n"
command += "# FUTURE TARGET ONLY: requires independent root review, exact publication and separate root launch authorization.\n"
command += "# Set CUDA_VISIBLE_DEVICES to the approved UUID in the controller's environment; do not rewrite the frozen producer argv.\n"
command += "# " + shlex.join([python, "-I", "-B", str(P / "direct_qualification_controller.py"),
    "--launch", "--publication-commit", "FULL_ROOT_REVIEWED_PUBLICATION_COMMIT"]) + "\n"
background = """This separate controller prepares one full-original direct H100 qualification. Its default reads fixed local metadata and starts no child. Its actual mode is limited to serval02, sl5nw UID 2618919 and the approved physical GPU UUID. The immutable 66-file packet and approved 44f producer remain unchanged.

The controller records a durable attempt before Popen, without a future child identity. It starts the exact frozen producer argv with an explicit initial environment. Only after Popen returns does it record the actual child PID, birth, all four user IDs, command and inherited affinity. It writes the actual campaign process row under the existing processes.lock and synchronizes its immutable copied ledger before publishing the complete launch record. The unchanged producer checks that record before numerical imports or device work. Missing data causes failure; no startup barrier, wrapper identity substitution or automatic retry was added.

The producer itself owns the standing GPU lock throughout its selected work. The controller owns no such descriptor. It retains its actual child object, waits for exactly that process, synchronizes the actual integer return and only writes the separate terminal proof for an ordinary zero exit. Its timeout can signal only the spawned PID after rechecking owner and birth. No successor is started.

All 18 corrected metadata tests passed on actual owned CPU step 6782816.30. These tests used only private temporary CPU processes and files, including the protected launch-input and terminal readers. The first run had 15 passes and one test-record parser error on deliberately invalid JSON; its exact files and failed receipt are preserved. The corrected collision test then passed separately before the final 18-case run. Those were separate runs.

The metadata fixtures use a synthetic serval02 profile label to test the protected reader; their actual CPU host, PID, affinity and wait are saved separately. They prove no CUDA, ABI, native GPU, lock custody or qualification result. Actual target execution, independent full qualification review and any admission remain pending.
"""
for name, value in (("manifest.yaml", base), ("config_resolved.yaml", configuration),
                    ("command.txt", command), ("experiment_background.md", background)):
    temporary = standard / (name + ".final")
    write(temporary, value)
    os.replace(temporary, standard / name)
    descriptor = os.open(standard, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
write(P / "controller_standard_records_finalization.json", {
    "schema_version": 1, "kind": "actual_metadata_based_controller_standard_records",
    "files": {name: receipt(standard / name) for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md")},
    "original_66_unchanged": True, **boundary})

