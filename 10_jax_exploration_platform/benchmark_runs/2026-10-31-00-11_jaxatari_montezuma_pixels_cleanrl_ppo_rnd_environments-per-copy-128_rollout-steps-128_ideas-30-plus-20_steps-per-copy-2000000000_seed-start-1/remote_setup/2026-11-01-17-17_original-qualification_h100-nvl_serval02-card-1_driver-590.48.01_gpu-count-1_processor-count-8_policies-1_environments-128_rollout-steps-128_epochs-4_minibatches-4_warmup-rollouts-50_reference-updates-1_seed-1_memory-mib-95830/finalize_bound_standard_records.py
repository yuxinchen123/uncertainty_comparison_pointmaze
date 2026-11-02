"""Finalize the bound qualification records from actual preparation and software checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex

P = Path(__file__).absolute().parent
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def receipt(path):
    return {"path": str(path), "sha256": digest(path)}
def raw_write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

request = json.loads((P / "preparation_request.json").read_bytes())
prepared = json.loads((P / "metadata_preparation_result.json").read_bytes())
software = json.loads((P / "shared_software_assets_staged_verification.json").read_bytes())
unit = Path(request["data_unit"]); plan = json.loads((unit / "measurement_plan.json").read_bytes())
assert prepared["passed"] is software["passed"] is True
assert prepared["plan"] == software["plan"] == receipt(unit / "measurement_plan.json")
assert plan["helpers"]["source_commit"] == "6066d9ea35fbc9c045855144550971c2ac5895f9"
assert len(plan["helpers"]["files"]) == 25 and prepared["original_test_cases"] == 48
assert prepared["history"]["comparisons"] == 120 and prepared["history"]["stop"] == {
    "direct_processes": 62, "jobs": 25, "source": "immutable historical snapshots only"}
assert len(software["installed_library_images"]) == 11 and software["assets"]["file_count"] == 4531
assert not any((unit / name).exists() for name in ("direct_current_launch.json", "direct_device_binding.json",
    "direct_process_metadata.json", "direct_terminal_process.json", "result.json", "direct_qualification_result.json"))
initial = P / "initial_planned_standard_records"; initial.mkdir()
for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
    raw_write(initial / name, (P / name).read_bytes()); (initial / name).chmod(0o444)
manifest = {"schema_version": 1, "kind": "bound_full_original_direct_qualification_metadata_packet",
    "state": "metadata prepared; actual root-reviewed launch and qualification remain pending",
    "at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": plan["helpers"]["source_commit"],
    "numerical_source_commit": plan["source_commit"], "plan": prepared["plan"],
    "prepared": prepared["prepared"], "metadata_preparation": receipt(P / "metadata_preparation_result.json"),
    "actual_shared_software_assets": receipt(P / "shared_software_assets_staged_verification.json"),
    "actual_direct_process": None, "current_physical_binding": None, "producer_started": False,
    "server_contact": False, "gpu_standing_lock_probed": False, "qualification_performed": False,
    "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}
configuration = {"reference": plan["reference_configuration"], "candidate": plan["configuration"],
    "runtime_environment": plan["runtime_environment"], "declared_qualification_shape": request["shape"],
    "actual_gpu_runtime_verified": False, "actual_cpu_affinity_at_future_launch": None}
command = "# Actual metadata preparation only; a producer launch requires separate root review and publication.\n"
for program in ("prepare_bound_qualification.py", "verify_shared_software_assets_staged.py"):
    command += shlex.join(["srun", "--jobid=6783003", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        request["python"], "-I", "-B", str(P / program)]) + "\n"
command += "# The exact future producer argv is frozen separately in the data-unit command.txt.\n"
background = """Prepare a fresh full-original H100 NVL qualification from the reviewed 6066 helper and D28 numerical source. The formal metadata preparation replayed all 120 original comparisons, 25 original numerical-job exits and 62 direct-process exits using the original immutable history. The new staged closure contains 25 exact helper files. Static gate contracts retain 48 original test definitions, 33 rooms, 132 renderer variants and 11 synthetic pixel cases; these numerical gates were not executed during preparation.

The shared-software check verified 11 installed library images, CPU Torch 2.6.0+cpu distribution metadata, JAXtari source and 4531 immutable asset files. It performed metadata and byte reads under the campaign CPU-artifact read lock. Actual H100 CUDA/context/PCI/minor/driver proof remains a future requirement in the actual process.

The planned qualification uses one independent policy, 128 environments, 128 rollout steps, four epochs, four minibatches, 50 normalization warmups, one reference update and seed 1. Network and optimizer arithmetic uses float32; running statistics use float64; matrix precision is highest. Controlled future work is 819200 normalization transitions and 16384 training transitions, with 16 optimizer minibatch updates. Original synthetic/partial test work has no invented whole-job transition total. The ordinary seven-round/ten-update measurement fields are unused in this standalone qualification branch.

The selected producer owns its original GPU standing-lock descriptor through synchronized GPU work and persistence. Its external controller records the actual child after Popen, publishes the actual ledger/launch identity durably, and waits for exactly that child. The controller owns no GPU descriptor. Future execution must prove eight actual CPU-affinity entries without claiming exclusive CPUs, preserve the approved physical UUID selector and reject any timeout or missing launch proof. No new direct PID, birth, physical binding, terminal proof or qualification result is created here. This packet grants zero scientific, throughput, packing, case, idea or worker-admission credit.
"""
for name, raw in (("manifest.yaml", encoded(manifest)), ("config_resolved.yaml", encoded(configuration)),
                  ("command.txt", command.encode()), ("experiment_background.md", background.encode())):
    temporary = P / (name + ".bound")
    raw_write(temporary, raw); os.replace(temporary, P / name)
    descriptor = os.open(P, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
raw_write(P / "final_standard_records_receipt.json", encoded({
    "schema_version": 1, "kind": "actual_bound_qualification_standard_record_finalization",
    "initial_generic_records_preserved": {name: receipt(initial / name)
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt")},
    "current_bound_records": {name: receipt(P / name)
        for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt")},
    "metadata_preparation": receipt(P / "metadata_preparation_result.json"),
    "software_assets": receipt(P / "shared_software_assets_staged_verification.json"),
    "actual_qualification_started": False, "scientific_credit": 0}))

