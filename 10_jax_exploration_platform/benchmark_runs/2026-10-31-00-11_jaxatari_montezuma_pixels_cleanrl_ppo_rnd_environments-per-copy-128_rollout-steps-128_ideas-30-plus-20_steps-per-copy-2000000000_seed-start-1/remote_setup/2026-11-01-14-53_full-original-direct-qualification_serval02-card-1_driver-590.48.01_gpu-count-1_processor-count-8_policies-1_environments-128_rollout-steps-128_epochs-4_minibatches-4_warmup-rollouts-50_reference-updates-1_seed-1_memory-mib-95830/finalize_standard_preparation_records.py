"""Finalize only the new packet's metadata documents, preserving every initial planned record."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
from zoneinfo import ZoneInfo

P = Path(__file__).resolve(strict=True).parent
read = lambda name: json.loads((P / name).read_bytes())
request = read("preparation_request.json"); prepared = read("metadata_preparation_result.json")
software = read("shared_software_assets_staged_verification.json")
assert prepared["passed"] is True and software["passed"] is True
unit = Path(request["data_unit"]); plan = json.loads((unit / "measurement_plan.json").read_bytes())
receipt = lambda path: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
archive = P / "initial_planned_standard_records"; archive.mkdir(exist_ok=False)
for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
    with (archive / name).open("xb") as out:
        out.write((P / name).read_bytes()); out.flush(); os.fsync(out.fileno())
    (archive / name).chmod(0o444)

scope = {"schema_version": 1, "kind": "full_original_direct_selected_qualification_execution_scope",
    "plan": prepared["plan"], "helper_source_commit": request["helper_source_commit"],
    "selected_producer_owns_standing_descriptor": True, "external_controller_owns_standing_descriptor": False,
    "profiling_has_separate_supervisor_owned_inherited_descriptor": True,
    "selected_producer_scope": "card_admission encloses numerical initialization, complete queued work, synchronized readout, evidence persistence and return from _compute; the external controller serializes successors until actual Popen.wait and terminal proof",
    "current_actual_direct_process": None, "actual_target_runtime_verified": False,
    "future_controller_protocol": [
        "On owned compute before contact, replay this exact frozen 120-case/exit proof and verify published plan/source/software/assets; refuse any previous intent or actual launch record.",
        "Only after separate launcher review/publication and root launch authorization, validate the direct host/UID and record a durable launch intent; do not invent a child identifier or birth.",
        "Read the controller's own actual available CPU affinity. Require at least eight allowed logical CPUs, choose exactly eight from that observed mask, set/record the controller mask, and let the exact-command child inherit it. Claim no Slurm CPU allocation or exclusive CPU ownership.",
        "Call Popen once with exactly the frozen data-unit producer argv, shell=False and an explicit initial child environment. Preserve the declared approved UUID and precision/determinism/cache controls without a taskset/env/spawn wrapper.",
        "Only after Popen returns, read this actual own child's kernel PID/birth/all four UIDs/argv/affinity and the controller's own birth/UID. If any identity cannot be proved, retain failure and actual wait evidence; never invent or reuse a prior PID.",
        "Under the exact owned process-ledger lock, append/synchronize the actual current row, then write/synchronize the immutable unit ledger snapshot and direct_current_launch.json binding the exact plan and returned child identity.",
        "The unchanged producer validates the actual launch record in collect_metadata before numerical import or device use. Missing, malformed, wrong or racing metadata fails before CUDA; preserve the packet and do not retry automatically.",
        "The producer itself acquires its single original standing-lock descriptor; the external Popen controller does not acquire, inherit or claim that descriptor. Fresh native context/UUID/PCI/minor and own-only windows remain required.",
        "The controller waits on exactly that spawned child. Save actual integer return/waited PID, exact producer argv and before/after controller UID/birth, then publish terminal proof only for ordinary zero exit.",
        "Start no successor before the actual wait/terminal proof. Full raw qualification review must succeed before writing a genuine independent review and invoking the unchanged full consumer."
    ],
    "future_launch_record_files": [str(unit / name) for name in ("direct_current_process_ledger_snapshot.json",
        "direct_current_launch.json", "direct_process_wait_return.json", "direct_terminal_process.json")],
    "current_launch_record_files_created": False,
    "ordinary_timing_fields_are_inert_in_this_standalone_branch": {"rounds": plan["rounds"], "iterations": plan["iterations"]},
    "controlled_future_work_only": {"normalization_environment_transitions": 819200,
        "training_environment_transitions": 16384, "full_reference_updates": 1, "optimizer_minibatch_updates": 16},
    "numerical_test_work_totals": "not inferred; actual original test fixtures are separate",
    "qualification_performed": False, "scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
    "worker_admitted": False, "admission_granted": False}
def write_new(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    return receipt(path)
scope_ref = write_new(P / "selected_custody_and_launch_protocol.json", scope)
write_new(unit / "direct_qualification_scope.json", scope)
manifest = {"schema_version": 1, "kind": "prepared_full_original_direct_qualification_packet",
    "state": "metadata prepared and checked; no actual direct launch or qualification",
    "campaign_run_dir": request["campaign_run_dir"], "unit_id": request["unit_id"], "data_unit": str(unit),
    "git": {"commit": request["helper_source_commit"], "source_root": request["helper_source_root"],
        "dirty": False, "verified_by": receipt(P / "metadata_preparation_result.json")},
    "numerical_source_commit": plan["source_commit"], "numerical_source_root": plan["source_root"],
    "shape": request["shape"], "plan": prepared["plan"], "prepared": prepared["prepared"],
    "history": prepared["history"], "logical_file_mappings": 693, "physical_payloads": 646,
    "metadata_verification": receipt(P / "shared_software_assets_staged_verification.json"),
    "current_physical_binding": plan["direct_binding"]["physical_identity"], "launch_protocol": scope_ref,
    "actual_direct_process": None, "actual_qualification_success": False, "scientific_credit": 0,
    "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}
command = ["srun", "--jobid=6782816", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
    request["python"], "-I", "-B", str(P / "prepare_direct_qualification_metadata.py")]
when = datetime.fromisoformat(prepared["at_utc"]).astimezone(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")
background = """## Purpose

Prepare a separate full-original Montezuma pixel qualification on the one approved H100 NVL card of serval02. Metadata preparation passed on owned CPU step 6782816.2. No direct server contact, lock probe, device access, numerical test, learning update or actual target qualification has occurred.

## Key hyperparameters

One policy uses 128 environments, 128 rollout steps, four optimization epochs and four minibatches. Retain fifty full normalization warmup rollouts and one controlled reference update with seed one. Networks and optimizer state use float32; running statistics use float64 and highest matrix precision. The authorized target is the known physical card with driver 590.48.01 and 95,830 mebibytes; fresh actual child context, UUID, PCI address and minor remain unresolved.

All forty-eight original test definitions, thirty-three rooms, 132 renderer variants and eleven synthetic pixel cases remain required. The controlled future warmup contains 819,200 environment transitions; the controlled future training update contains 16,384 transitions and sixteen optimizer minibatch updates. Work done by original numerical tests is separate and no whole-job transition total is inferred. Ordinary seven-round/ten-update fields in the generic measurement plan are not executed by this standalone qualification branch.

## Code and config changes

Use only published helper commit """ + request["helper_source_commit"] + """ and unchanged numerical source """ + plan["source_commit"] + """. Metadata checks rebuilt all 120 original comparisons, twenty-five terminal numerical jobs and sixty-two birth-qualified direct processes from 646 read-only payloads, 693 logical file mappings and eight exact historical Git roots. The current shared software, eleven installed CUDA/cuDNN image hashes and all 4,531 assets match the completed original shared-software reference; this proves no target CUDA runtime or driver ABI.

The selected producer owns its standing-lock descriptor through its measured GPU work and synchronized readout. Its external exact-command controller owns no such descriptor and must wait for actual child termination before a successor. Profiling has a separate supervisor-owned inherited-descriptor protocol. The future launcher must derive and record eight actual allowed CPU identifiers without a wrapper or exclusive-CPU claim. Child PID, birth, all UIDs, argv and affinity are recorded only after Popen returns and synchronized before the producer can pass its launch-record guard. A missing or invalid record fails before CUDA without automatic retry.

The post-context startup check retains the used-memory limit below 1,024 mebibytes, disabled multi-instance/confidential-computing modes, own-only assigned-device samples and at least fifteen percent actual free memory. Pre-context occupancy and startup peaks remain unobserved. No previous direct qualification, accepted combined baseline or completed-fifty proof is required. Actual launch, terminal/raw review, genuine independent review and any later admission remain separate.

## Git state

The detached helper checkout was clean at metadata preparation """ + when + """. Its twenty-five helper files and the unchanged original source are bound by the saved plan and verification records. Numerical source and current source admission are unchanged. The initial planned standard records are retained verbatim in initial_planned_standard_records/. This command records only the completed metadata preparation; it is not a direct server launch.
"""
outputs = {"manifest.yaml": (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    "config_resolved.yaml": (unit / "config_resolved.yaml").read_bytes(),
    "command.txt": ("# Actual metadata preparation only; do not rerun an attempted packet.\n" + shlex.join(command) + "\n").encode(),
    "experiment_background.md": background.encode()}
for name, raw in outputs.items():
    temporary = P / (name + ".finalizing")
    write_new(temporary, raw); os.replace(temporary, P / name)
descriptor = os.open(P, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
write_new(P / "final_standard_records_receipt.json", {"schema_version": 1,
    "kind": "direct_qualification_final_standard_metadata_records", "files": [receipt(P / name) for name in outputs],
    "initial_records_preserved": [receipt(archive / name) for name in outputs], "launch_protocol": scope_ref,
    "actual_target_process": None, "launch_performed": False, "qualification_credit": 0})
print(json.dumps({"finalized_metadata_only": True, "plan_sha256": prepared["plan"]["sha256"],
    "launch_protocol": scope_ref, "standard_files": len(outputs)}))

