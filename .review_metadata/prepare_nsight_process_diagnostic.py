"""Prepare one source-bound, zero-credit Nsight process observation without running the tool."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

RELATIVE = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
FILES = ("diagnose_nsight_process_tree.py", "nsight_process_identity.py")
KIND = "nsight_process_identity_diagnostic"
PYTHON = "/usr/bin/python3"
ZERO = {"scientific_credit": 0, "profile_credit": 0, "case_credit": 0, "idea_credit": 0,
        "environment_credit": 0, "learner_credit": 0, "whole_tool_library_or_device_activity_quantified": False,
        "admission_granted": False, "worker_admitted": False, "physical_gres_mapping_verified": False}


def checked(value, message):
    """Reject an unsupported source or an attempted output path before preparing files."""
    if not value: raise ValueError(message)


def git(root, *arguments):
    """Read only the exact proposed detached helper source."""
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def put(path, raw):
    """Create and fsync a fresh artifact without replacing an earlier attempt."""
    if not isinstance(raw, bytes): raw = (json.dumps(raw, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def prepare(root, commit, run, name, tool_path, *, reservation=None, unreserved=False):
    """Stage only two committed standard-library modules and five ordinary unit records."""
    # Before: an omitted choice silently used an old reservation; after: one explicit choice is required.
    checked(type(unreserved) is bool, "the explicit unreserved selection must be Boolean")
    if unreserved:
        checked(reservation is None, "choose an explicit reservation or an explicitly unreserved job, never both")
    else:
        checked(type(reservation) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", reservation),
                "choose an explicit reservation name or an explicitly unreserved job")
    placement_description = ("An unreserved job was explicitly selected." if unreserved
                             else "The caller explicitly selected reservation " + reservation + ".")
    reservation_arguments = [] if unreserved else ["--reservation=" + reservation]
    reservation_directive = "" if unreserved else "#SBATCH --reservation=" + reservation + "\n"
    qos = "normal" if unreserved else "csresnolim"
    checked(re.fullmatch(r"[0-9a-f]{40}", commit) and root.is_absolute() and root.resolve(strict=True) == root,
            "an exact published helper root and full commit are required")
    checked(git(root, "rev-parse", "HEAD").decode().strip() == commit
            and subprocess.run(["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 1
            and git(root, "status", "--porcelain=v1", "--untracked-files=all") == b"", "the helper root must be detached and clean")
    checked(name.startswith("2026-") and "/" not in name and len(name.encode()) <= 255
            and run.is_absolute() and run.resolve(strict=True) == run, "an exact new campaign unit name is required")
    packet, unit = run / "remote_setup" / name, run / "data" / name
    checked(not os.path.lexists(packet) and not os.path.lexists(unit), "never reuse an attempted or partial diagnostic unit")
    raw_files = {}
    for filename in FILES:
        path = root / (RELATIVE + filename); raw = git(root, "show", commit + ":" + RELATIVE + filename)
        checked(path.resolve(strict=True) == path and path.read_bytes() == raw, "committed diagnostic source differs from disk")
        raw_files[filename] = raw
    tool = json.loads(tool_path.read_text())
    checked(set(tool) == {"path", "version", "sha256"} and tool["sha256"] ==
            "48e4c308f538c8971b8f62c389834372d986f00f200743f1eae90918f2f04e1d"
            and tool["version"] == "NVIDIA Nsight Systems version 2024.7.1.84-247135125610v0",
            "use the genuine saved actual-tool record, with no tool execution during preparation")
    packet.mkdir(); unit.mkdir(); (unit / "code").mkdir()
    for filename, raw in raw_files.items(): put(unit / "code" / filename, raw)
    helpers = {"source_root": str(root), "source_commit": commit,
        "files": {RELATIVE + filename: hashlib.sha256(raw).hexdigest() for filename, raw in raw_files.items()}}
    plan = {"schema_version": 1, "kind": KIND, "unit_id": name, "campaign_run_dir": str(run), "packet": str(packet),
        "python": PYTHON, "helpers": helpers, "profile_tool": tool, "gpu_uuid": None, "job_id": None,
        "resources": {"partition": "gpu", "account": "sl5nw_base", "qos": qos, "reservation": reservation,
            "node": "jaguar03", "nodes": 1, "tasks": 1, "requested_cpus": 1, "host_memory_gib": 2,
            "gres": "gpu:nvidia_rtx_a4500:1", "wall_seconds": 300, "requeue": False},
        "work": {"learner_updates": 0, "warmup_rollouts": 0, "cuda_profiler_api_calls_by_payload": 0,
            "numpy_or_jax_or_torch_imports": 0, "actual_nsight_profile_command_count": 1},
        "capture_process_protocol": "nsight-owned-descendant-observation-version-1", **ZERO}
    plan_ref = put(unit / "measurement_plan.json", plan)
    put(unit / "measurement_prepared.json", {"plan_sha256": plan_ref["sha256"]})
    command = [PYTHON, "-I", "-S", "-B", str(unit / "code/diagnose_nsight_process_tree.py"), "--prepared-unit", str(unit)]
    background = ("## Purpose\n\nObserve the actual process relationship created by the installed Nsight tool before any learner work. "
        "The previous two profile attempts failed before numerical release and remain unchanged.\n\n"
        "## Key hyperparameters\n\nOne typed A4500, one requested processor, two GiB of host memory and five minutes. "
        "The payload uses only the standard library, with no learner updates, warmups or profiler API calls. "
        "Nsight itself may load its native tracing libraries on the assigned GPU.\n\n"
        "## Code and config changes\n\nThe exact existing Nsight launch options run a metadata waiter and one metadata child. "
        "The parent saves positive kernel-reported process links, user IDs, births, commands and the numeric step. "
        "Different process groups are recorded. There is no scientific, profile, case or idea credit.\n\n"
        f"## Placement\n\n{placement_description}\n\n"
        f"## Git state\n\nClean detached helper commit: `{commit}`. Two exact committed source files are staged.\n").encode()
    for directory in (packet, unit):
        put(directory / "manifest.yaml", plan); put(directory / "config_resolved.yaml", plan)
        put(directory / "experiment_background.md", background)
        put(directory / "command.txt", (shlex.join(command) + "\n").encode())
        put(directory / "progress.json", {"state": "prepared_not_submitted", "submitted": False,
            "job_id": None, "gpu_uuid": None, "learner_updates": 0, **ZERO})
    launch = ["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--cpu-bind=cores",
              "--gres=gpu:nvidia_rtx_a4500:1", "--kill-on-bad-exit=1", *command]
    shell = ("#!/bin/bash\n#SBATCH --job-name=pixel-nsight-process-observation\n"
        f"{reservation_directive}"
        f"#SBATCH --output={run}/slurm/jobs/nsight-process-observation-%j.out\n"
        f"#SBATCH --error={run}/slurm/jobs/nsight-process-observation-%j.err\n"
        f"# Placement: {placement_description}\n"
        "set -euo pipefail\numask 077\nunset PYTHONPATH PYTHONHOME\n"
        "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1\n"
        "export LD_LIBRARY_PATH=\"/usr/lib/nvidia:${LD_LIBRARY_PATH:-}\"\n"
        f"cd {shlex.quote(str(run))}\n"
        f"sha256sum --check --strict {shlex.quote(str(packet / 'source_checksums.sha256'))}\n"
        "exec " + shlex.join(launch) + "\n").encode()
    script = put(packet / "job.sbatch", shell)
    held = ["sbatch", "--hold", "--parsable", "--no-requeue", "--partition=gpu", "--account=sl5nw_base",
        "--qos=" + qos, *reservation_arguments, "--nodelist=jaguar03", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=1", "--mem=2G", "--gres=gpu:nvidia_rtx_a4500:1", "--time=00:05:00", str(packet / "job.sbatch")]
    put(packet / "held_submission_command.json", held)
    checked_files = [*sorted((unit / "code").iterdir()), unit / "measurement_plan.json", unit / "measurement_prepared.json"]
    put(packet / "source_checksums.sha256", "".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p) + "\n" for p in checked_files).encode())
    freeze = {str(path): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                         "mode": oct(path.stat().st_mode & 0o777)}
              for directory in (packet, unit) for path in sorted(directory.rglob("*")) if path.is_file()}
    put(packet / "immutable_artifacts.json", {"files": freeze})
    return {"prepared": True, "packet": str(packet), "unit": str(unit), "plan": plan_ref, "script": script,
            "files": len(freeze), "actual_jobs_submitted": 0, "actual_tool_executions": 0, **ZERO}


def main(argv=None):
    """Require an explicit published source and a new unit, with no submission operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("helper-root", "campaign-run", "tool-record"): parser.add_argument("--" + name, type=Path, required=True)
    for name in ("helper-commit", "unit-id"): parser.add_argument("--" + name, required=True)
    placement = parser.add_mutually_exclusive_group(required=True)
    placement.add_argument("--reservation", help="Explicit reservation name; availability is checked before submission.")
    placement.add_argument("--unreserved", action="store_true", help="Explicitly request no reservation.")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.helper_root, args.helper_commit, args.campaign_run, args.unit_id, args.tool_record,
                             reservation=args.reservation, unreserved=args.unreserved), indent=2))


if __name__ == "__main__": main()
