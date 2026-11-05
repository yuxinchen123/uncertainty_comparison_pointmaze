"""Copy the unchanged reviewed reserved launcher and check one fresh system-Python diagnostic packet."""
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
OLD = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-04-19-37_nsight-process-observation_jaguar03_gpu-count-1_processor-count-1_host-memory-gib-2_time-limit-minutes-5_reservation-sl5nw_168_qos-csresnolim_tool-version-2024.7.1.84_environment-transitions-0")
HELPER = Path("/p/rlprojects/RND_worktrees/2026-11-04-20-08_nsight-process-diagnostic-source_system-python-3.10_required-process-api-precheck_commit-7aff76d9d163d01de805364de917cc558e783535")
COMMIT = "7aff76d9d163d01de805364de917cc558e783535"


def reference(path):
    """Record the exact current ordinary metadata bytes."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    """Fsync only a new artifact and preserve every previous packet or partial attempt."""
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def main():
    """Reuse byte-identical approved source; check new input records without repeating fixture tests."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("the exact existing owned CPU allocation and numeric step are required")
    expected = {"launch_diagnostic_once.py": "80d65582290bb19bd4743f997034724be894c0efe88b0e5433552cb05e266e81",
        "test_launch_diagnostic_once.py": "9c3bb633cafcaa7d56696375593a7212978c64fa7d989e1ea4bc3fe7bd3be343",
        "run_launcher_metadata_tests.py": "de292bb9d86507b9d3daecb951af9ce91d8776f446d6b0ddc3df44b1cc0fc82f"}
    approval_path = OLD / "launcher_independent_review.json"
    tests_path = OLD / "launcher_root_independent_metadata_tests.json"
    if reference(approval_path)["sha256"] != "12071da948e15d2b1bbcad400cadd814c8fe3ec1c4a1417a41cce83726125475":
        raise ValueError("the actual original reserved launcher approval changed")
    approval, tests = json.loads(approval_path.read_text()), json.loads(tests_path.read_text())
    if not (approval["passed"] is True and approval["actual_test_receipt"] == reference(tests_path)
            and reference(tests_path)["sha256"] == "a4f5669130cb221c51a02eadfcea9e88aa0a6babf28895e0d29d42ead82fb037"
            and tests["passed"] is True and tests["tests_run"] == 11 and tests["step_id"] == "305"
            and tests["source_files_unchanged"] is True):
        raise ValueError("the actual prior source controls do not support source-only reuse")
    copied = {}
    for name, expected_sha in expected.items():
        raw = (OLD / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("an approved reserved launcher, fixture or runner changed")
        write(PACKET / name, raw)
        copied[name] = {"original": reference(OLD / name), "new": reference(PACKET / name), "byte_identical": True}
    spec = importlib.util.spec_from_file_location("fresh_system_python_reserved_launcher", PACKET / "launch_diagnostic_once.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan, held = module.checked_packet()
    if not (plan["python"] == "/usr/bin/python3" and plan["helpers"]["source_root"] == str(HELPER)
            and plan["helpers"]["source_commit"] == COMMIT and plan["job_id"] is None and plan["gpu_uuid"] is None):
        raise ValueError("the fresh system-Python producer configuration or source changed")
    if not (plan["resources"]["reservation"] == "sl5nw_168" and plan["resources"]["qos"] == "csresnolim"
            and held.count("--reservation=sl5nw_168") == 1 and held.count("--qos=csresnolim") == 1):
        raise ValueError("the fresh reserved resource records disagree")
    if subprocess.check_output(["git", "-C", str(HELPER), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("the actual helper checkout is not the published source")
    if subprocess.check_output(["git", "-C", str(HELPER), "status", "--porcelain", "--untracked-files=all"]):
        raise ValueError("the published helper checkout is no longer clean")
    if subprocess.run(["git", "-C", str(HELPER), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 1:
        raise ValueError("the helper checkout is no longer detached")
    helper_review_path = HELPER / ".review_metadata/nsight_interpreter_api_precheck_root_review.json"
    helper_review = json.loads(helper_review_path.read_text())
    if reference(helper_review_path)["sha256"] != "adf28c36ad0e706ac40ebfc6ba85f3446a1686f3b5a924e1c7e07fe0754ee093" or helper_review["passed"] is not True:
        raise ValueError("the published API-precheck source review changed")
    full_source = json.loads((HELPER / ".review_metadata/nsight_interpreter_api_precheck_full_source_freeze.json").read_text())
    approved_hashes = {Path(name).name: sha for name, sha in full_source["files"].items()}
    for name, expected_sha in plan["helpers"]["files"].items():
        raw = subprocess.check_output(["git", "-C", str(HELPER), "cat-file", "blob", COMMIT + ":" + name])
        staged = UNIT / "code" / Path(name).name
        if not (raw == staged.read_bytes() and hashlib.sha256(raw).hexdigest() == expected_sha
                and expected_sha == approved_hashes[Path(name).name]):
            raise ValueError("the fresh staged producer differs from the reviewed committed source")
    for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"):
        if (PACKET / name).read_bytes() != (UNIT / name).read_bytes():
            raise ValueError("the two actual sets of standard records differ")
    script = (PACKET / "job.sbatch").read_text()
    if script.count("#SBATCH --reservation=sl5nw_168\n") != 1 or "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python" in script:
        raise ValueError("the actual reserved script uses another reservation or interpreter")
    for name in ("manifest.yaml", "config_resolved.yaml"):
        if json.loads((PACKET / name).read_text())["python"] != "/usr/bin/python3":
            raise ValueError("a fresh standard record uses the scientific interpreter")
    write(PACKET / "packet_static_verification.json", {"schema_version": 1, "passed": True,
        "kind": "fresh_reserved_system_python_nsight_packet_static_check", "at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"],
        "plan": reference(UNIT / "measurement_plan.json"), "script": reference(PACKET / "job.sbatch"),
        "immutable_artifacts": reference(PACKET / "immutable_artifacts.json"), "initial_files_verified": 17,
        "standard_records_per_directory": 5, "source_commit": COMMIT, "source_checkout_clean_and_detached": True,
        "source_api_precheck_review": reference(helper_review_path), "selected_python_path": "/usr/bin/python3",
        "actual_target_python_version_established": False, "target_python_version_recorded_only_at_runtime": True,
        "staged_producers_match_published_and_reviewed_bytes": True, "actual_tool_hash_rerun": False,
        "actual_tool_executions": 0, "actual_job_actions": 0, "actual_kernel_pidfd_calls": 0,
        "scientific_credit": 0, "admission_granted": False})
    write(PACKET / "launcher_source_copy_proof.json", {"schema_version": 1, "passed": True,
        "kind": "byte_identical_reserved_launcher_source_reuse_proof", "copied_files": copied,
        "original_independent_launcher_approval": reference(approval_path),
        "original_actual_eleven_control_receipt": reference(tests_path), "original_actual_control_step": "6783003.305",
        "fixture_controls_repeated_for_this_copy": False,
        "scope": "Only the three copied launcher/fixture/runner byte sequences inherit their prior source review. New packet and API-precheck helper inputs require this packet's separate review; no target, tool, profile or scientific admission.",
        "actual_tool_or_job_or_gpu_actions": 0, "scientific_credit": 0})
    paths = [Path(name) for name in json.loads((PACKET / "immutable_artifacts.json").read_text())["files"]]
    paths += [PACKET / name for name in ("immutable_artifacts.json", "packet_static_verification.json",
        "launch_diagnostic_once.py", "test_launch_diagnostic_once.py", "run_launcher_metadata_tests.py")]
    write(PACKET / "launcher_source_freeze.json", {"schema_version": 1,
        "kind": "fresh_reserved_system_python_nsight_launcher_source_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "actual_job_id": None, "actual_gpu_uuid": None, "scientific_credit": 0})
    print(json.dumps({"source_freeze": reference(PACKET / "launcher_source_freeze.json"),
        "static_check": reference(PACKET / "packet_static_verification.json"), "source_copy_proof": reference(PACKET / "launcher_source_copy_proof.json"),
        "plan": reference(UNIT / "measurement_plan.json"), "initial_files": 17, "frozen_files": len(paths),
        "actual_target_python_version_established": False, "fixture_controls_repeated": False}))


if __name__ == "__main__":
    main()

