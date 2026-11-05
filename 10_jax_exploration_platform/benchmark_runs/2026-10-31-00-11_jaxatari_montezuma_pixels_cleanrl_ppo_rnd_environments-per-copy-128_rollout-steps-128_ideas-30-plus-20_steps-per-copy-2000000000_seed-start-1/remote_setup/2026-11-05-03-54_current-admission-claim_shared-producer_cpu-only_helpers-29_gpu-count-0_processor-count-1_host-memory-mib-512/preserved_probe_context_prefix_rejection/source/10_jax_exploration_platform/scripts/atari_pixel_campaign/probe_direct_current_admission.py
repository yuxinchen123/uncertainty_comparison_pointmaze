"""Read a direct admission claim on owned CPU compute before any target launch.

Prepared-unit mode uses the actual staged helper. Fixture mode is explicitly
nonoperational and accepts only a separately stored source-receipt fixture.
Neither mode checks the CPU host as a GPU target or grants worker admission.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"


def require(value, message):
    """Refuse incomplete metadata without attempting execution or repair."""
    if not value:
        raise ValueError(message)


def read(path):
    """Read one ordinary immutable metadata file and return its exact hash."""
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
        "an exact ordinary metadata path is required")
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def git(root, *arguments):
    """Read only the declared local helper checkout through Git."""
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def probe(path, expected_hash, *, fixture=False, canonical_source=False):
    """Execute the actual shared claim producer with exact new helper bytes."""
    plan, reference = read(path)
    require(reference["sha256"] == expected_hash, "the requested plan bytes changed")
    campaign = Path(plan["campaign_run_dir"])
    job, step = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    require(job.isdecimal() and step.isdecimal() and os.getuid() == os.geteuid() == 2618919
        and job in (campaign / "slurm/submitted_jobids_uva.txt").read_text().split(),
        "the claim probe requires an exact owned numerical CPU step")
    cgroup = Path("/proc/self/cgroup").read_text()
    require("job_" + job in cgroup and "step_" + step in cgroup,
        "the current CPU process must belong to the recorded owned step")
    helper = plan["helpers"]; root = Path(helper["source_root"])
    require(git(root, "rev-parse", "HEAD").decode().strip() == helper["source_commit"]
        and not git(root, "status", "--porcelain"), "the exact helper checkout must be clean")
    require(len(helper["files"]) == 29 and "accepted_evidence_snapshot_protocol" in plan,
        "this probe requires the exact new 29-helper history protocol")
    unit = campaign / "data" / plan["unit_id"]
    if not fixture:
        require(Path(path).absolute() == unit / "measurement_plan.json", "the actual prepared plan path changed")
        ready, _ = read(unit / "measurement_prepared.json")
        require(ready == {"plan_sha256": expected_hash}, "the actual preparation receipt changed")
        require(not any((unit / name).exists() for name in ("direct_controller_outcome.json",
            "direct_controller_spawn.json", "direct_current_launch.json", "result.json", "failure.json")),
            "an attempted unit cannot be a fresh prelaunch admission check")
    directory = root / PREFIX if fixture else unit / "code"
    for relative, expected in helper["files"].items():
        require(relative.startswith(PREFIX) and Path(relative).name == relative[len(PREFIX):],
            "the helper roster contains an unexpected path")
        canonical = root / relative
        staged = directory / Path(relative).name
        blob = git(root, "show", helper["source_commit"] + ":" + relative)
        require(all(hashlib.sha256(raw).hexdigest() == expected for raw in
            (blob, canonical.read_bytes(), staged.read_bytes())), "a declared helper byte changed")
    binding = directory / "serval02_card1_binding.py"
    require(hashlib.sha256(binding.read_bytes()).hexdigest() == hashlib.sha256(
        Path(__file__).with_name(binding.name).read_bytes()).hexdigest(),
        "the canonical probe and selected claim producer must be from the same reviewed source")
    specification = importlib.util.spec_from_file_location("actual_direct_admission_probe_binding", binding)
    core = importlib.util.module_from_spec(specification); specification.loader.exec_module(core)
    core.source_identity(plan)
    core.validate_history(plan)
    claim = core.current_admission_claim(plan)
    require(claim["frozen_admission"]["sha256"] == core.ORIGINAL_ADMISSION_SHA256,
        "the shared claim producer did not retain the original admission")
    if canonical_source:
        core.check_live_admission(claim)
    return {"schema_version": 1, "kind": "cpu_shared_direct_current_admission_probe", "passed": True,
        "plan": reference, "helpers": helper, "claim": claim, "fixture_only": fixture,
        "actual_prepared_unit_checked": not fixture, "canonical_source_checked": canonical_source,
        "cpu_step": {"job": job, "step": step, "pid": os.getpid(), "cgroup": cgroup},
        "gpu_target_checked": False, "worker_admission_granted": False,
        "environment_transitions": 0, "scientific_credit": 0}


def main():
    """Print read-only claim evidence; the caller saves stdout outside helper source."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepared-unit", type=Path)
    mode.add_argument("--fixture-plan", type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--check-canonical-source", action="store_true")
    args = parser.parse_args()
    path = args.fixture_plan or args.prepared_unit / "measurement_plan.json"
    print(json.dumps(probe(path, args.expected_plan_sha256, fixture=args.fixture_plan is not None,
        canonical_source=args.check_canonical_source), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
