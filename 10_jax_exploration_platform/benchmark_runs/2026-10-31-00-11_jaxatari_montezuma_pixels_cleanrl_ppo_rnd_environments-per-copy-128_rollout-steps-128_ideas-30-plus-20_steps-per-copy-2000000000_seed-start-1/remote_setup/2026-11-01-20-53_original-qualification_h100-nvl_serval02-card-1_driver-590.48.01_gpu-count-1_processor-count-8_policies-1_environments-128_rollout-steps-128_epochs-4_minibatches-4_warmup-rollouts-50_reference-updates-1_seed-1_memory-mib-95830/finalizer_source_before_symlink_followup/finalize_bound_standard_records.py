"""Publish standard records from actual owned-CPU preparation; start no target process."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).absolute().parent


def encoded(value):
    """Encode finite complete metadata deterministically."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def receipt(path):
    """Bind an ordinary small metadata record to its exact bytes."""
    path = Path(path)
    assert path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
    assert path.stat().st_size <= 1024 * 1024
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    """Read one small metadata object without numerical imports."""
    receipt(path)
    return json.loads(Path(path).read_bytes())


def write(path, raw):
    """Create one durable new metadata record; never replace a prior attempt."""
    path = Path(path)
    assert path.parent.resolve(strict=True) == path.parent
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    """Preserve the initial records and derive the new records only from actual proofs."""
    assert os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ.get("SLURM_STEP_ID", "").isdecimal()
    assert os.getresuid() == (2618919, 2618919, 2618919)
    assert not (P / "final_standard_records_receipt.json").exists()
    assert not any((P / (name + ".bound")).exists() for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"))
    request = read(P / "preparation_request.json")
    prepared = read(P / "metadata_preparation_result.json")
    software = read(P / "shared_software_assets_staged_verification.json")
    unit = Path(request["data_unit"]); plan = read(unit / "measurement_plan.json")
    assert prepared["passed"] is software["passed"] is True
    assert prepared["plan"] == software["plan"] == receipt(unit / "measurement_plan.json")
    assert plan["helpers"]["source_commit"] == "0a880984b4a03a2ff7335337cc18576628bc9fc1"
    assert len(plan["helpers"]["files"]) == 26 and prepared["original_test_cases"] == 48
    assert prepared["history"]["comparisons"] == 120 and prepared["history"]["stop"] == {
        "direct_processes": 62, "jobs": 25, "source": "immutable historical snapshots only"}
    assert len(software["installed_library_images"]) == 11 and software["assets"]["file_count"] == 4531
    assert plan["direct_backend_image"]["schema_version"] == 2
    assert software["backend_expected"] == prepared["backend_expected"] == plan["direct_backend_image"]["expected"]
    assert prepared["backend_preparation"] == request["direct_backend_image_record"]
    assert receipt(plan["direct_backend_image"]["expected"]["path"]) == plan["direct_backend_image"]["expected"]
    assert receipt(request["direct_backend_image_record"]["path"]) == request["direct_backend_image_record"]
    assert not any((unit / name).exists() for name in ("direct_current_launch.json", "direct_device_binding.json",
        "direct_process_metadata.json", "direct_terminal_process.json", "result.json", "direct_qualification_result.json"))
    old = request["initial_standard_data_records"]
    assert set(old) == {"manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"}
    for reference in old.values():
        assert receipt(reference["path"]) == reference
    for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
        assert receipt(P / name)["sha256"] == old[name]["sha256"]
    manifest = {"schema_version": 1, "kind": "bound_full_original_direct_backend_package_qualification_metadata",
        "state": "metadata prepared; reviewed controller publication and actual qualification pending",
        "at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": plan["helpers"]["source_commit"],
        "numerical_source_commit": plan["source_commit"], "plan": prepared["plan"],
        "prepared": prepared["prepared"], "metadata_preparation": receipt(P / "metadata_preparation_result.json"),
        "shared_software_assets": receipt(P / "shared_software_assets_staged_verification.json"),
        "backend_preparation": request["direct_backend_image_record"], "backend_expected": plan["direct_backend_image"]["expected"],
        "actual_process": None, "actual_physical_binding": None, "producer_started": False,
        "host_contact": False, "standing_gpu_lock_opened": False, "qualification_performed": False,
        "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False,
        "command_scope": "exact prepared producer argv; reviewed controller publication required before execution"}
    configuration = {"reference": plan["reference_configuration"], "candidate": plan["configuration"],
        "runtime_environment": plan["runtime_environment"], "declared_qualification_shape": request["shape"],
        "backend_expected": plan["direct_backend_image"]["expected"], "actual_gpu_runtime_verified": False}
    background = """## Purpose

Run one new full original H100 qualification after requiring the actual backend package image to be prepared before GPU work. Owned CPU preparation has checked all 120 historical comparisons, 25 original numerical-job exits and 62 direct-process exits using fresh copies of the original immutable capture. This unit has no GPU result or approval yet.

## Key hyperparameters

One independent policy, 128 environments, 128 rollout steps, four epochs, four minibatches, fifty normalization warmups, one full reference update and seed 1. Networks and Adam use float32; statistics use float64; matrix precision is highest and operations are deterministic. The intended target is the approved serval02 physical card 1 H100 NVL with driver 590.48.01, 95830 MiB and eight observed nonexclusive CPU-affinity entries, for at most three hours. Actual execution must verify these values. Controlled future work is 819200 normalization transitions, 16384 training transitions and sixteen optimizer minibatch updates. Synthetic and partial original test work has no invented whole-job total.

## Code and config changes

The new direct package-image protocol pins the uniquely owning installed distribution, the exact image bytes, METADATA and RECORD before the new run. The actual process must preserve mapped-file and same-process evidence before and after all original work. The separate eleven-image CUDA and cuDNN receipts retain their original meanings. CPU metadata preparation checked the exact 26 staged helpers, original 48 test definitions, 33 rooms, 132 renderer variants, eleven synthetic pixel cases, package metadata and 4531 asset files. No numerical tests or GPU operations ran during preparation. The producer alone holds the standing GPU-lock descriptor; the external controller registers and waits on its exact actual child and owns no GPU descriptor. Every timeout or intervention fails. Root must independently review and publish that controller before starting one child. This packet grants zero scientific, timing, packing, idea, case or worker-admission credit.

## Git state

Helper source: 0a880984b4a03a2ff7335337cc18576628bc9fc1. Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af. Owned CPU preparation verified the new private detached helper was clean; exact source and staged byte receipts are saved in the plan and metadata preparation record. The numerical source and its computation order are unchanged.
"""
    records = {"manifest.yaml": encoded(manifest), "config_resolved.yaml": encoded(configuration),
        "command.txt": (unit / "command.txt").read_bytes(), "experiment_background.md": background.encode()}
    for name, raw in records.items():
        temporary = P / (name + ".bound")
        write(temporary, raw); os.replace(temporary, P / name)
    descriptor = os.open(P, os.O_RDONLY | os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
    assert all((unit / name).is_file() for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md", "progress.json"))
    write(P / "final_standard_records_receipt.json", encoded({"schema_version": 1,
        "kind": "actual_bound_qualification_standard_record_finalization", "initial_records_preserved": old,
        "bound_packet_records": {name: receipt(P / name) for name in records},
        "standard_data_records": {name: receipt(unit / name) for name in (*records, "progress.json")},
        "metadata_preparation": receipt(P / "metadata_preparation_result.json"),
        "shared_software_assets": receipt(P / "shared_software_assets_staged_verification.json"),
        "actual_qualification_started": False, "scientific_credit": 0, "admission_granted": False}))


if __name__ == "__main__":
    main()
