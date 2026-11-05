"""Adapt the reviewed one-shot diagnostic launcher to the explicitly selected active reservation."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
OLD = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-04-18-33_nsight-process-observation_jaguar03_gpu-count-1_processor-count-1_host-memory-gib-2_time-limit-minutes-5_placement-unreserved_qos-normal_tool-version-2024.7.1.84_environment-transitions-0")


def reference(path):
    """Record the exact ordinary source or metadata bytes."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    """Preserve every completed or partial preparation artifact."""
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def substitute(source, changes):
    """Apply only the exact declared literal substitutions and retain their counts."""
    records = []
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError("the reviewed source substitution is not unique")
        source = source.replace(before, after, 1)
        records.append({"before": before, "after": after, "count": 1})
    ast.parse(source)
    return source.encode(), records


def main():
    """Write the narrow reservation source changes, verify the actual packet and freeze its exact inputs."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("an actual numeric step of owned CPU monitor 6783003 is required")
    old_source = (OLD / "launch_diagnostic_once.py").read_text()
    if reference(OLD / "launch_diagnostic_once.py")["sha256"] != "b9f781e00542e007c19b46c8fe4e2d52bdfc64c95bf5b96494eb09f7de7be967":
        raise ValueError("the reviewed normal-quality-of-service launcher changed")
    source, source_changes = substitute(old_source, [
        ('"qos": "normal", "reservation": None', '"qos": "csresnolim", "reservation": "sl5nw_168"'),
        ('"--qos=normal", "--nodelist=jaguar03"', '"--qos=csresnolim", "--reservation=sl5nw_168", "--nodelist=jaguar03"'),
        ('fields.get("QOS") == "normal"', 'fields.get("QOS") == "csresnolim"'),
        ('fields.get("Reservation", "") in {"", "(null)", "None"}', 'fields.get("Reservation", "") == "sl5nw_168"'),
        ('the exact held unreserved submission command changed', 'the exact held reserved submission command changed')
    ])
    write(PACKET / "launch_diagnostic_once.py", source)
    old_tests = (OLD / "test_launch_diagnostic_once.py").read_text()
    if reference(OLD / "test_launch_diagnostic_once.py")["sha256"] != "447b271277f1a055f9015a065db26ef1cd2bad8be361378556fb55697d9c7b47":
        raise ValueError("the reviewed private diagnostic fixtures changed")
    tests, test_changes = substitute(old_tests, [
        ('TimeLimit=00:05:00 Partition=gpu Account=sl5nw_base QOS=normal ', 'TimeLimit=00:05:00 Partition=gpu Account=sl5nw_base QOS=csresnolim Reservation=sl5nw_168 '),
        ('"ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=2", "Reservation=sl5nw_168"))',
         '"ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=2", "Reservation=synthetic_other_reservation"))'),
        ('else: text += replacement', 'else: text = text.replace("Reservation=sl5nw_168", replacement)')
    ])
    write(PACKET / "test_launch_diagnostic_once.py", tests)
    runner = (OLD / "run_launcher_metadata_tests.py").read_bytes()
    if hashlib.sha256(runner).hexdigest() != "de292bb9d86507b9d3daecb951af9ce91d8776f446d6b0ddc3df44b1cc0fc82f":
        raise ValueError("the reviewed metadata runner changed")
    write(PACKET / "run_launcher_metadata_tests.py", runner)
    spec = importlib.util.spec_from_file_location("reserved_diagnostic_metadata_launcher", PACKET / "launch_diagnostic_once.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan, command = module.checked_packet()
    if plan["resources"]["reservation"] != "sl5nw_168" or plan["resources"]["qos"] != "csresnolim":
        raise ValueError("the actual reserved plan has a different reservation or quality of service")
    if command.count("--reservation=sl5nw_168") != 1 or command.count("--qos=csresnolim") != 1:
        raise ValueError("the actual held command differs from the explicitly selected reservation")
    if (PACKET / "job.sbatch").read_text().count("#SBATCH --reservation=sl5nw_168\n") != 1:
        raise ValueError("the actual job script lacks the exact selected reservation")
    for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
        if (PACKET / name).read_bytes() != (UNIT / name).read_bytes():
            raise ValueError("the actual packet and unit standard records differ")
    for name, expected in plan["helpers"]["files"].items():
        committed = subprocess.check_output(["git", "-C", plan["helpers"]["source_root"], "cat-file", "blob",
                                            plan["helpers"]["source_commit"] + ":" + name])
        if hashlib.sha256(committed).hexdigest() != expected or committed != (UNIT / "code" / Path(name).name).read_bytes():
            raise ValueError("the staged producers differ from unchanged committed source")
    evidence = RUN / "remote_setup/own_reservation_168_exact_utc_2026-11-05-03-36.json"
    reservation = json.loads(evidence.read_text())
    if not (reservation["environment_override"] == {"TZ": "UTC"} and reservation["returncode"] == 0
            and reservation["fields"]["State"] == "ACTIVE" and reservation["fields"]["Users"] == "sl5nw"
            and "jaguar03" in reservation["fields"]["Nodes"].split(",")
            and reservation["fields"]["EndTime"] == "2026-11-17T05:00:01"):
        raise ValueError("the parent's exact current reservation evidence does not support this placement")
    write(PACKET / "packet_static_verification.json", {"schema_version": 1, "passed": True,
        "kind": "actual_reserved_nsight_diagnostic_packet_static_verification", "at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"],
        "plan": reference(UNIT / "measurement_plan.json"), "script": reference(PACKET / "job.sbatch"),
        "immutable_artifacts": reference(PACKET / "immutable_artifacts.json"), "initial_files_verified": 17,
        "standard_records_per_directory": 5, "reservation_evidence": reference(evidence),
        "reservation": "sl5nw_168", "reservation_expiry_utc": "2026-11-17T05:00:01+00:00",
        "helper_commit": plan["helpers"]["source_commit"], "staged_producers_unchanged": True,
        "actual_tool_hash_rerun": False, "actual_tool_execution": False, "actual_jobs_submitted": 0,
        "scientific_credit": 0})
    write(PACKET / "reserved_launcher_substitution_proof.json", {"schema_version": 1, "passed": True,
        "kind": "exact_reserved_diagnostic_launcher_literal_substitution_proof",
        "original_reviewed_launcher": reference(OLD / "launch_diagnostic_once.py"),
        "new_launcher": reference(PACKET / "launch_diagnostic_once.py"), "launcher_substitutions": source_changes,
        "original_reviewed_tests": reference(OLD / "test_launch_diagnostic_once.py"),
        "new_tests": reference(PACKET / "test_launch_diagnostic_once.py"), "test_substitutions": test_changes,
        "metadata_runner_byte_identical": True, "staged_producers_unchanged": True,
        "reservation_evidence": reference(evidence), "actual_scheduler_or_tool_or_gpu_actions": 0, "scientific_credit": 0})
    paths = [Path(name) for name in json.loads((PACKET / "immutable_artifacts.json").read_text())["files"]]
    paths += [PACKET / name for name in ("immutable_artifacts.json", "packet_static_verification.json",
        "launch_diagnostic_once.py", "test_launch_diagnostic_once.py", "run_launcher_metadata_tests.py")]
    freeze = {"schema_version": 1, "kind": "reserved_nsight_diagnostic_launcher_source_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "actual_scheduler_actions": 0, "scientific_credit": 0}
    write(PACKET / "launcher_source_freeze.json", freeze)
    print(json.dumps({"packet": str(PACKET), "source_freeze": reference(PACKET / "launcher_source_freeze.json"),
        "substitution_proof": reference(PACKET / "reserved_launcher_substitution_proof.json"),
        "plan": reference(UNIT / "measurement_plan.json"), "initial_files": 17, "frozen_files": len(paths)}))


if __name__ == "__main__":
    main()

