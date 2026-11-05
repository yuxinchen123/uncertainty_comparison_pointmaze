"""Independently approve the fresh bound packet and its actual shared claim proof."""
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
START = time.monotonic()
CHECKS = []


def require(condition, statement):
    """Record each actual required assertion without inventing a check count."""
    assert condition, statement
    CHECKS.append(statement)


def ref(path):
    """Bind a small ordinary source or metadata input to its exact bytes."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ordinary reviewed input: " + str(path))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    """Read one exact small JSON document."""
    return json.loads(Path(path).read_text())


# Bind only actual owned CPU review and actual newly frozen packet metadata.
require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit(), "current exact owned CPU step")
require(os.getresuid() == (2618919, 2618919, 2618919), "actual reviewer UID values")
require("/job_6821437/step_" + os.environ["SLURM_STEP_ID"] + "/" in Path("/proc/self/cgroup").read_text(), "reviewer current exact kernel job/step path")
freeze_path = P / "controller_source_freeze.json"
require(ref(freeze_path)["sha256"] == "5e8c6c362599e5548031677009f2ea7d356a55984b588180967959febb0c3828", "actual held fresh source freeze")
freeze = read(freeze_path)
require(len(freeze["files"]) == 51, "all 51 actual frozen packet and additional closure inputs")
for name, expected in freeze["files"].items():
    require(ref(name)["sha256"] == expected, "unchanged actual frozen input: " + Path(name).name)
plan = read(U / "measurement_plan.json")
request = read(P / "preparation_request.json")
require(ref(U / "measurement_plan.json")["sha256"] == "31e5dc09ec2a8050b7043d0e69842d5785c1996939e83f61aa23dc7422b5301b", "actual new prepared plan")
require(plan["helpers"]["source_commit"] == "69f1b5075530418adbe02c1e433c5647848a61ba" and len(plan["helpers"]["files"]) == 29, "exact reviewed current 29-helper namespace")
require(plan["helpers"]["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py"] == "af72d3de3e84f7b8be08a73f90f5becf143b1d32c90083c1773b10343751361b", "actual shared claim fix is staged")
require(plan["helpers"]["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/accepted_evidence_snapshot.py"] == "94dc00524bfc5d02ebc2f7b8f2b88444516938fe583967ea16e414a1e2e4e54a", "historical CPU proof producer bytes remain unchanged")
require(plan["source_commit"] == "d28e6735e79788f618acb0ebfde8609db1f776af" and plan["copy_seeds"] == [1], "original numerical source and independent first-copy seed")
learner = plan["configuration"]["learner"]
for key, value in {"n_copies": 1, "n_envs": 128, "num_steps": 128, "update_epochs": 4, "num_minibatches": 4, "observation_warmup_iterations": 50, "total_timesteps": 2000000000}.items():
    require(type(learner[key]) is int and learner[key] == value, "original strict learner configuration: " + key)
require(plan["mode"] == "confirmation" and plan["rounds"] == 7 and plan["iterations"] == plan["minimum_updates_per_round"] == 10, "seven paired rounds with ten complete updates per arm")
expected = json.loads(json.dumps(plan["reference_configuration"]))
expected["environment"]["cache_room_backgrounds"] = True
expected["environment"]["render_final_two_only"] = True
require(expected == plan["configuration"], "only two accepted environment rendering flags differ")
require(plan["accepted_evidence_snapshot_inputs"]["proof"]["sha256"] == "88791d3f7e602523c547650d41bb2147d101a3793db0d969d34872ded8388a0e" and plan["accepted_evidence_snapshot_inputs"]["review"]["sha256"] == "ac50aa8d71e4e1fd7665161486cae76f7b9fa38e045af52559182037a81733db", "exact already accepted historical proof and genuine review")

# Verify real prepare, cold verification, shared claim probe and bind each succeeded once.
for phase in ("prepare", "verify", "current_admission_claim_probe", "bind"):
    result = read(P / ("root_actual_" + phase + "_return.json"))
    require(result["returncode"] == 0 and result["local_wait_timed_out"] is False and 0 <= result["elapsed_seconds"] < 300, "actual bounded successful phase: " + phase)
probe = read(P / "root_actual_current_admission_claim_probe_stdout.raw")
require(probe["kind"] == "cpu_shared_direct_current_admission_probe" and probe["passed"] is True and probe["fixture_only"] is False and probe["actual_prepared_unit_checked"] is True, "actual newly prepared-unit shared claim producer was exercised, not a source fixture")
require(probe["plan"] == ref(U / "measurement_plan.json") and probe["helpers"] == plan["helpers"], "actual claim proof joins this exact new plan and staged/canonical/Git helper declaration")
require(probe["canonical_source_checked"] is True and probe["gpu_target_checked"] is False and probe["worker_admission_granted"] is False and probe["environment_transitions"] == probe["scientific_credit"] == 0, "canonical shared source proof grants no GPU target or scientific admission")
context = probe["cpu_step"]
require(context["job"] == "6821437" and context["step"] == "41" and "/job_6821437/step_41/" in context["cgroup"] and type(context["pid"]) is int and context["pid"] > 0, "actual claim proof has exact positive owned CPU-step identity")
claim = probe["claim"]
require(set(claim) == {"schema_version", "kind", "canonical_path", "frozen_admission", "source_root", "source_commit"} and claim["kind"] == "current_direct_admission_matches_frozen_original" and claim["schema_version"] == 1, "unchanged exact original current-admission claim schema")
require(claim["canonical_path"] == str(R / "SOURCE_AND_GATES_READY.json") and claim["source_root"] == plan["source_root"] and claim["source_commit"] == plan["source_commit"] and claim["frozen_admission"]["sha256"] == "4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c", "actual new claim retains original authority, source root and original frozen admission")
require(ref(claim["frozen_admission"]["path"]) == claim["frozen_admission"], "exact original qualified frozen admission bytes")
probe_intent = read(P / "root_actual_current_admission_claim_probe_intent.json")
require("--prepared-unit" in probe_intent["argv"] and "--fixture-plan" not in probe_intent["argv"] and "--check-canonical-source" in probe_intent["argv"] and str(U) in probe_intent["argv"], "actual probe used prepared-unit and canonical-source modes")
probe_return = read(P / "root_actual_current_admission_claim_probe_return.json")
bind_intent = read(P / "root_actual_bind_intent.json")
require(datetime.datetime.fromisoformat(probe_return["finished_at_utc"]) <= datetime.datetime.fromisoformat(bind_intent["at_utc"]), "actual claim producer passed before binding")

# Preserve the genuine scientific predecessor and separately close the failed metadata-only attempt.
require(ref(P / "serial_predecessor_exit_inputs.json")["sha256"] == "936b7b53758e6f02c0f803decccd89a3c943137bd24b60ee910e8716838e8027", "genuine old second-copy correctness predecessor bytes unchanged")
closed = read(P / "closed_failed_startup_controller_additional_facts.json")
require(closed["actual_controller_pid"] == 321183 and closed["actual_controller_birth_counter"] == "83855111" and closed["actual_controller_ended"] is True and closed["actual_numerical_child_started"] is False and closed["old_attempt_remains_failed"] is True, "separate actual failed-controller closure creates no fake numerical success")
for value in closed["records"].values():
    require(ref(value["path"]) == value, "unchanged additional failed-startup record")
exit_record = read(closed["records"]["root_failed_startup_controller_exit_observation.json"]["path"])
require(exit_record["actual_predecessor_ended"] is True and exit_record["process_directory_exists"] is False and exit_record["process_signals_sent"] == 0 and exit_record["pid"] == 321183 and exit_record["expected_birth_counter"] == "83855111", "actual root observed exact prior controller exit without signals")
old_outcome = read(closed["records"]["direct_controller_outcome.json"]["path"])
require(old_outcome["passed"] is False and old_outcome["actual_child_started"] is False and old_outcome["actual_wait_saved"] is False, "old failed attempt has no invented normal numerical wait")
restored = read(closed["records"]["root_metadata_lock_permission_restore.json"]["path"])
require(restored["restored"] is True and restored["standing_gpu_lock_touched"] is False and all(row["same_inode_and_bytes"] is True for row in restored["files"]), "own metadata permissions were restored on the same inodes after failed-controller closure")
procedure = read(P / "controller_launch_review_procedure.json")
require(procedure["actual_inner_publication"] is None and procedure["actual_launch_performed"] is False and procedure["root_starter_process_and_custody_functions_must_remain_unchanged"] is True and procedure["root_starter_reference"]["sha256"] == "060ee49018531e797d9a7dd23bf8ab9f49978de48a713b3956f1923289980fdc", "future transport must derive the accepted detached starter after real inner publication")

# Execute only the bound default metadata reader once, never the claim producer or numerical lifecycle again.
argv = ["/usr/bin/python3", "-I", "-S", "-B", str(P / "direct_qualification_controller.py")]
completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
for name, raw in (("independent_bound_default_stdout.raw", completed.stdout), ("independent_bound_default_stderr.raw", completed.stderr)):
    with (P / name).open("xb") as handle:
        handle.write(raw)
require(completed.returncode == 0, "fresh exact bound default returned zero: " + completed.stderr.decode())
default = json.loads(completed.stdout)
require(default["passed"] is True and default["action_performed"] is False and default["actual_child_pid"] is None and default["plan"] == ref(U / "measurement_plan.json"), "fresh bound source/config/predecessor/claim namespace is valid without action or invented PID")
require(not any((U / name).exists() for name in ("direct_controller_intent.json", "direct_controller_spawn.json", "direct_current_launch.json", "result.json", "failure.json")) and not (P / "root_actual_controller_spawn.json").exists(), "actual fresh timing remains unattempted")
for name, expected_hash in freeze["files"].items():
    require(ref(name)["sha256"] == expected_hash, "held input unchanged after independent reader: " + Path(name).name)
require(not {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"}.intersection(sys.modules), "no numerical imports in reviewer")
review = {"schema_version": 1, "kind": "direct_qualification_controller_root_independent_review", "passed": True, "launch_approved": True,
          "independent_of_author": True, "reviewer": "/root/review_bound_h100_timing_launch", "controller_freeze": ref(freeze_path), "packet_freeze": ref(P / "packet_metadata_freeze.json"), "plan": ref(U / "measurement_plan.json"),
          "controller_owns_gpu_standing_descriptor": False, "producer_owns_gpu_standing_descriptor": True,
          "actual_prepared_unit_claim_probe": ref(P / "root_actual_current_admission_claim_probe_stdout.raw"), "actual_prepared_unit_claim_probe_fixture_only": False,
          "additional_failed_startup_facts": ref(P / "closed_failed_startup_controller_additional_facts.json"), "scientific_predecessor_unchanged": ref(P / "serial_predecessor_exit_inputs.json"),
          "actual_owned_cpu_job_id": "6821437", "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "actual_cpu_process": {"pid": os.getpid(), "hostname": socket.gethostname(), "uids": list(os.getresuid()), "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19], "cgroup": Path("/proc/self/cgroup").read_text()},
          "checks": CHECKS, "elapsed_seconds": time.monotonic() - START, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "required_before_actual_target_launch": ["Publish the exact actual source freeze, prepared records and this review to obtain a real inner controller publication.", "Derive the detached root starter from unchanged060eeprocess/barrier/custody functions with only the real inner publication literal changed; inspect and publish that transport separately.", "Run a publication-aware default metadata check with the real inner commit.", "Root alone executes one reviewed SSH after own metadata permission protection; sole numerical producer owns standing GPU descriptor.", "Keep second-copy timing unbound until actual first-copy timing normal zero wait and both exact exits."],
          "actual_target_or_gpu_or_remote_action": False, "actual_shared_claim_producer_or_causal_cases_repeated_by_reviewer": False, "large_historical_payloads_replayed": False,
          "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admission": False}
out = P / "controller_root_independent_review.json"
with out.open("x") as handle:
    json.dump(review, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"passed": True, "review": ref(out), "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "checks": len(CHECKS), "elapsed_seconds": review["elapsed_seconds"]}))
