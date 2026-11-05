"""Independently review the actual bound first-copy timing packet on owned CPU."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
R = P.parent.parent
U = R / "data" / P.name
STARTED = time.monotonic()
CHECKS = []


def checked(condition, description):
    """Record a successful required check and stop at its first failure."""
    # Retain each actual assertion rather than an invented check count.
    assert condition, description
    CHECKS.append(description)


def read(name):
    """Read one declared JSON metadata file."""
    return json.loads(Path(name).read_text())


def ref(name):
    """Bind one ordinary regular metadata file to its exact bytes."""
    path = Path(name)
    checked(path.is_file() and not path.is_symlink(), "ordinary metadata path: " + str(path))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(name, value):
    """Write a new review record once and make it durable."""
    # A second invocation must not replace a first review.
    with (P / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


# Require the actual admitted CPU parent and bounded metadata-only resources.
checked(os.environ.get("SLURM_JOB_ID") == "6783003", "exact owned CPU allocation")
checked(os.environ.get("SLURM_STEP_ID", "").isdigit(), "actual numeric CPU step")
checked(os.getresuid() == (2618919, 2618919, 2618919), "all actual UID values")
checked(socket.gethostname().split(".")[0] == "heartpiece", "current owned allocation host")
checked("6783003" in (R / "slurm/submitted_jobids_uva.txt").read_text().split(), "owned identifier ledger contains parent")

# Verify actual source freeze and every frozen packet byte before execution.
freeze = read(P / "controller_source_freeze.json")
checked(ref(P / "controller_source_freeze.json")["sha256"] == "1dba3762c597733b6edf1fddd63ce87ffdd254817480424430ac06570e33bc95", "exact author source freeze")
checked(len(freeze["files"]) == 42, "all 42 actual packet files frozen")
for path, expected in freeze["files"].items():
    checked(Path(path).is_relative_to(P) and ref(path)["sha256"] == expected, "unchanged frozen packet file: " + Path(path).name)
checked(freeze["canonical_source_commit"] == "765480d990cff5ae556914e3b74afba0472f64ea", "approved canonical controller commit")
mapping = read(P / "canonical_controller_source_mapping.json")
for original, item in mapping["files"].items():
    checked(ref(original)["sha256"] == item["sha256"] == ref(item["staged"])["sha256"], "exact canonical-to-staged source: " + Path(original).name)
checked(mapping["copied_preparer_is_archival_only"] is True, "archival preparer cannot be mistaken for the canonical command")
source_review = read(P / "controller_source_independent_approval.json")
checked(source_review["passed"] is True and source_review["approved"] is True and source_review["independent_of_author"] is True, "genuine independent corrected source approval")
checked(source_review["tests_run"] == 8 and source_review["actual_cpu_step_id"] == "450", "exact completed source-review receipt")

# Verify actual operation returns, cold validation, software proof and frozen configuration.
for phase in ("prepare", "verify", "bind"):
    actual = read(P / ("root_actual_" + phase + "_return.json"))
    checked(actual["returncode"] == 0 and actual["local_wait_timed_out"] is False and 0 <= actual["elapsed_seconds"] < 300, "one real bounded successful " + phase)
plan = read(U / "measurement_plan.json")
request = read(P / "preparation_request.json")
cold = read(P / "cold_metadata_validation.json")
software = read(P / "shared_software_assets_staged_verification.json")
checked(cold["passed"] is True and cold["plan_sha256"] == ref(U / "measurement_plan.json")["sha256"], "real cold metadata child joins actual plan")
checked(cold["numerical_imports"] == [] and cold["gpu_operations"] == 0 and cold["historical_raw_gates_recomputed"] is False, "cold child did no numerical or GPU work")
checked(cold["metadata_compute_context"]["job_id"] == "6783003" and cold["metadata_compute_context"]["step_id"] == "461", "real prior cold child step")
checked(software["passed"] is True and software["helper_count"] == 29 and software["numerical_imports"] == [] and software["gpu_operations"] == 0, "actual complete metadata software proof")
checked(software["plan"] == ref(U / "measurement_plan.json"), "software proof joins actual plan")
checked(plan["mode"] == "confirmation" and type(plan["rounds"]) is int and plan["rounds"] == 7 and plan["iterations"] == plan["minimum_updates_per_round"] == 10, "seven alternating paired rounds and ten full updates per arm")
checked(type(plan["configuration"]["learner"]["n_copies"]) is int and plan["configuration"]["learner"]["n_copies"] == request["serial_order"] == 1 and plan["copy_seeds"] == [1], "strict first-copy serial position and seed")
reference = read(P / "reference_configuration.json")
candidate = read(P / "candidate_configuration.json")
expected = json.loads(json.dumps(reference))
expected["environment"]["cache_room_backgrounds"] = True
expected["environment"]["render_final_two_only"] = True
checked(candidate == expected == plan["configuration"] and reference == plan["reference_configuration"], "only background caching and final-two-frame rendering change")
checked(plan["source_commit"] == "d28e6735e79788f618acb0ebfde8609db1f776af" and len(plan["helpers"]["files"]) == 29 and plan["helpers"]["source_commit"] == "f7e8f5c8f0a2d1847fff4f93726fce62205c23aa", "exact original numerical source and reviewed 29 helpers")
checked(plan["gpu_uuid"] == "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3", "only approved serval02 physical card-one UUID")
checked(plan["runtime_environment"] == {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest", "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}, "unchanged float64-statistics deterministic highest-precision runtime")
checked(ref(request["accepted_evidence_snapshot_inputs"]["proof"]["path"]) == request["accepted_evidence_snapshot_inputs"]["proof"] and ref(request["accepted_evidence_snapshot_inputs"]["review"]["path"]) == request["accepted_evidence_snapshot_inputs"]["review"], "real completed reviewed historical CPU proof")
for key in ("direct_qualification_inputs", "common_correctness_inputs"):
    for item in request[key].values():
        checked(ref(item["path"]) == item, "unchanged accepted " + key)

# Review the proposed transport without contacting any host or inventing a commit.
commands = read(P / "prospective_launch_commands.json")
ssh = commands["one_shot_ssh_argv_template"]
checked(ssh[:8] == ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=15", "serval02"], "normal single direct SSH route with no proxy override")
checked(len(ssh) == 9 and "ProxyCommand" not in " ".join(ssh), "no inherited external-cluster proxy rule")
checked(commands["publication_commit"] is None and commands["marker_is_deliberately_invalid_for_controller"] is True, "unknown future publication commit remains explicit")
checked(commands["controller_owns_standing_descriptor"] is False and commands["producer_owns_standing_descriptor"] is True, "only numerical producer owns standing GPU descriptor")
checked(commands["automatic_retry_permitted"] is False and commands["c2_must_remain_unbound"] is True, "one attempt and no premature second-copy binding")

# Execute the actual default metadata reader once; it must not launch a producer.
argv = ["/usr/bin/python3", "-I", "-S", "-B", str(P / "direct_qualification_controller.py")]
completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
with (P / "independent_bound_default_stdout.raw").open("xb") as handle:
    handle.write(completed.stdout)
with (P / "independent_bound_default_stderr.raw").open("xb") as handle:
    handle.write(completed.stderr)
checked(completed.returncode == 0, "unchanged actual bound default reader returned zero: " + completed.stderr.decode())
result = json.loads(completed.stdout)
checked(result["passed"] is True and result["action_performed"] is False and result["actual_child_pid"] is None, "default reader performs no action and invents no process")
checked(result["controller_owns_gpu_standing_descriptor"] is False and result["producer_owns_gpu_standing_descriptor"] is True, "actual reader retains descriptor ownership boundary")
checked(result["plan"] == ref(U / "measurement_plan.json"), "actual default reader joins this exact plan")
checked(not any((U / name).exists() for name in ("direct_controller_intent.json", "direct_controller_spawn.json", "direct_current_launch.json", "result.json", "failure.json")), "unit remains actually unattempted")
for path, expected in freeze["files"].items():
    checked(ref(path)["sha256"] == expected, "frozen input remains unchanged after reader: " + Path(path).name)
checked(not ({"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"} & set(sys.modules)), "review has no numerical imports")

# Publish a genuine prospective approval; publication and replacement runtime remain root's prerequisites.
review = {"schema_version": 1, "kind": "direct_qualification_controller_root_independent_review", "passed": True, "launch_approved": True, "independent_of_author": True, "reviewer": "/root/review_bound_h100_timing_launch", "controller_freeze": ref(P / "controller_source_freeze.json"), "packet_freeze": ref(P / "packet_metadata_freeze.json"), "plan": ref(U / "measurement_plan.json"), "controller_owns_gpu_standing_descriptor": False, "producer_owns_gpu_standing_descriptor": True, "review_inputs": ref(P / "prospective_controller_review_inputs.json"), "launch_commands": ref(P / "prospective_launch_commands.json"), "actual_metadata_reader_stdout": ref(P / "independent_bound_default_stdout.raw"), "actual_metadata_reader_stderr": ref(P / "independent_bound_default_stderr.raw"), "actual_cpu_job_id": os.environ["SLURM_JOB_ID"], "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "review_process": {"pid": os.getpid(), "hostname": socket.gethostname(), "uids": list(os.getresuid()), "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19], "cgroup": Path("/proc/self/cgroup").read_text()}, "source_review_repeated": False, "large_historical_payloads_rehashed": False, "historical_proof_cpu_identity_claimed_on_gpu": False, "required_before_actual_launch": ["Publish exact prepared source, packet and this independent review.", "Run publication-aware default reader once with the real published commit.", "Verify replacement monitor actual runtime and first full report before GPU work.", "Root alone executes the reviewed single SSH command once; only producer holds standing descriptor.", "Keep copy two unbound until actual first-copy timing normal wait and both exact exits."], "actual_gpu_action": False, "numerical_imports": [], "gpu_operations": 0, "remote_actions": 0, "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "packing_credit": 0, "worker_admission": False, "checked": CHECKS, "elapsed_seconds": time.monotonic() - STARTED, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
save("controller_root_independent_review.json", review)
save("independent_prospective_launch_approval.json", {"kind": "independent_prospective_first_copy_common_timing_launch_approval", "passed": True, "independent_review": ref(P / "controller_root_independent_review.json"), "commands": ref(P / "prospective_launch_commands.json"), "publication_commit": None, "actual_launch_performed": False, "required_before_actual_launch": review["required_before_actual_launch"], "scientific_credit": 0})
print(json.dumps({"passed": True, "review": ref(P / "controller_root_independent_review.json"), "check_count": len(CHECKS), "elapsed_seconds": time.monotonic() - STARTED, "actual_cpu_step_id": os.environ["SLURM_STEP_ID"]}))
