"""Independently approve the fresh bound packet and its actual shared claim proof."""
import datetime
import ast
import copy
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
require(ref(freeze_path)["sha256"] == "d86d80ad584e2491c45a5851726c0b96e369920dc64f695b5b51191ae04ee468", "actual held fresh source freeze")
freeze = read(freeze_path)
require(len(freeze["files"]) == 87, "all87 actual frozen packet, provenance and closure inputs")
for name, expected in freeze["files"].items():
    require(ref(name)["sha256"] == expected, "unchanged actual frozen input: " + Path(name).name)
plan = read(U / "measurement_plan.json")
request = read(P / "preparation_request.json")
require(ref(U / "measurement_plan.json")["sha256"] == "bdcf18f09d88667e28919ee07b15f66dfdc52dfa40f289b1cabfde57af2d3e48", "actual new prepared plan")
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
    require(result["returncode"] == 0 and 0 <= result["wall_seconds"] < 300, "actual bounded successful phase: " + phase)
probe = read(P / "root_actual_current_admission_claim_probe_stdout.raw")
require(probe["kind"] == "cpu_shared_direct_current_admission_probe" and probe["passed"] is True and probe["fixture_only"] is False and probe["actual_prepared_unit_checked"] is True, "actual newly prepared-unit shared claim producer was exercised, not a source fixture")
require(probe["plan"] == ref(U / "measurement_plan.json") and probe["helpers"] == plan["helpers"], "actual claim proof joins this exact new plan and staged/canonical/Git helper declaration")
require(probe["canonical_source_checked"] is True and probe["gpu_target_checked"] is False and probe["worker_admission_granted"] is False and probe["environment_transitions"] == probe["scientific_credit"] == 0, "canonical shared source proof grants no GPU target or scientific admission")
context = probe["cpu_step"]
require(context["job"] == "6821437" and context["step"] == "73" and "/job_6821437/step_73/" in context["cgroup"] and type(context["pid"]) is int and context["pid"] > 0, "actual claim proof has exact positive owned CPU-step identity")
claim = probe["claim"]
require(set(claim) == {"schema_version", "kind", "canonical_path", "frozen_admission", "source_root", "source_commit"} and claim["kind"] == "current_direct_admission_matches_frozen_original" and claim["schema_version"] == 1, "unchanged exact original current-admission claim schema")
require(claim["canonical_path"] == str(R / "SOURCE_AND_GATES_READY.json") and claim["source_root"] == plan["source_root"] and claim["source_commit"] == plan["source_commit"] and claim["frozen_admission"]["sha256"] == "4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c", "actual new claim retains original authority, source root and original frozen admission")
require(ref(claim["frozen_admission"]["path"]) == claim["frozen_admission"], "exact original qualified frozen admission bytes")
probe_intent = read(P / "root_actual_current_admission_claim_probe_intent.json")
require("--prepared-unit" in probe_intent["argv"] and "--fixture-plan" not in probe_intent["argv"] and "--check-canonical-source" in probe_intent["argv"] and str(U) in probe_intent["argv"], "actual probe used prepared-unit and canonical-source modes")
probe_return = read(P / "root_actual_current_admission_claim_probe_return.json")
bind_intent = read(P / "root_actual_bind_intent.json")
require(datetime.datetime.fromisoformat(probe_return["at_utc"]) <= datetime.datetime.fromisoformat(bind_intent["at_utc"]), "actual claim producer passed before binding")

# Preserve the genuine scientific predecessor and separately close the failed metadata-only attempt.
require(ref(P / "serial_predecessor_exit_inputs.json")["sha256"] == "936b7b53758e6f02c0f803decccd89a3c943137bd24b60ee910e8716838e8027", "genuine old second-copy correctness predecessor bytes unchanged")
closed = read(P / "separate_failed_attempt_closure_facts.json")
require(closed["kind"] == "actual_failed_attempt_closures_separate_from_accepted_scientific_predecessor" and len(closed["attempts"]) == 2 and closed["scientific_credit"] == 0 and closed["new_process_or_liveness_query"] is False, "separate historical failed-attempt closure facts, no new liveness query or credit")
for index, attempt in enumerate(closed["attempts"]):
    require(attempt["remains_failed"] is True and attempt["is_scientific_serial_predecessor"] is False, "failed attempt remains failed and is not the scientific predecessor")
    require(ref(attempt["outcome"]["path"]) == attempt["outcome"], "exact original failed outcome bytes")
    outcome = read(attempt["outcome"]["path"])
    require(outcome["passed"] is False and outcome["actual_child_started"] is attempt["actual_child_started"] and outcome["actual_wait_saved"] is attempt["actual_wait_saved"], "actual failed outcome joins separate closure facts")
    for value in attempt["facts"].values():
        require(ref(value["path"]) == value, "unchanged additional failed-startup record")
    if index == 0:
        require(attempt["actual_child_started"] is False and attempt["actual_wait_saved"] is False, "old02-24 failed before numerical child and contains no invented wait")
        exit_record = read(attempt["facts"]["root_failed_startup_controller_exit_observation.json"]["path"])
        require(exit_record["actual_predecessor_ended"] is True and exit_record["process_directory_exists"] is False and exit_record["process_signals_sent"] == 0 and exit_record["pid"] == 321183 and exit_record["expected_birth_counter"] == "83855111", "actual root observed exact old02-24 controller exit without signals")
    else:
        require(attempt["actual_child_started"] is True and attempt["actual_wait_saved"] is True, "old04-07 did start and wait for actual failed child")
        exit_record = read(attempt["facts"]["root_failed_identity_startup_exact_exit_observation.json"]["path"])
        require(exit_record["all_original_processes_ended"] is True and exit_record["signals_sent"] == 0 and {(row["pid"],row["expected_birth_counter"]) for row in exit_record["observations"]} == {(351185,"84237458"),(351650,"84242978")}, "actual root observed both exact old04-07 controller and child ended without signals")
        audit = read(attempt["facts"]["independent_actual_child_identity_failure_audit.json"]["path"])
        require(audit["audit_passed"] is True and audit["scientific_attempt_passed"] is False and audit["exact_initial_failed_subcondition"] == "unproved" and audit["actual_child"]["normal_wait_status"] == 1, "actual failed first-child audit retains unproved subcondition and normal wait one")
    restored = read(attempt["facts"]["root_metadata_lock_permission_restore.json"]["path"])
    require(restored["restored"] is True and restored["standing_gpu_lock_touched"] is False, "own metadata permissions restored after each failed attempt; no standing descriptor touched")
require(closed["scientific_predecessor_unchanged"]["sha256"] == ref(P / "serial_predecessor_exit_inputs.json")["sha256"], "all failed-startup facts remain separate from unchanged genuine scientific predecessor")

# Validate only the reviewed observation-only controller changes and their actual CPU proof.
require(request["wrapper_source_commit"] == "e16c015ac7c9ca674f768240cad35a372eb18e87", "actual independently reviewed observation-only wrapper source commit")
require(ref(P / "direct_qualification_controller.py")["sha256"] == "09a1c944116541b631e12e465f5e782641429e305e9a135da4dc8e5379f9a5b6", "exact approved observation-only controller bytes staged")
require(ref(P / "test_initial_child_identity.py")["sha256"] == "4b3312a8e2437b0bc776ead943bb3237ad45bdce0d1e788e124203d4a1089e5d", "new affected identity observation test is included and frozen")
source_review = read(P / "initial_child_identity_source_independent_review.json")
require(ref(P / "initial_child_identity_source_independent_review.json")["sha256"] == "db15bd9b3ce07c9913473aebc3e3756e7d027f5e25ca02be1eb903cd5825702f" and source_review["approved"] is True and source_review["passed"] is True, "genuine independent observation-only source review included")
require(ref(P / "initial_child_identity_corrected_source_freeze.json")["sha256"] == "63c3186658242633d913b8302c1b0dae38adb8ee72faf311c64b693d324665cc", "exact corrected source-freeze provenance included")
proof_ref = read(P / "prior_actual_CPU_identity_diagnostic_review.json")
require(ref(proof_ref["path"])["sha256"] == proof_ref["sha256"] == "16357840abe1d5692c1041369f9f5058554f351878a91e7e8573dda4b6de271a" and proof_ref["gpu_admission"] is False and proof_ref["scientific_credit"] == 0, "actual eight-CPU diagnostic independent review joined without GPU admission")
proof = read(proof_ref["path"])
require(proof["passed"] is True and proof["actual_diagnostic_accepted"] is True and proof["historical_failure_cause_established"] is False and proof["worker_or_GPU_or_numerical_admission"] is False, "CPU diagnostic does not establish historical cause or numerical qualification")
old_packet = next((R / "remote_setup").glob("2026-11-05-04-07_common-timing*"))
old_source = (old_packet / "direct_qualification_controller.py").read_text()
new_source = (P / "direct_qualification_controller.py").read_text()
old_functions = {node.name:node for node in ast.parse(old_source).body if isinstance(node,ast.FunctionDef)}
new_functions = {node.name:node for node in ast.parse(new_source).body if isinstance(node,ast.FunctionDef)}
require({name for name in old_functions if ast.dump(old_functions[name],include_attributes=False) != ast.dump(new_functions[name],include_attributes=False)} == {"supervise"} and set(new_functions)-set(old_functions) == {"child_identity_observation"}, "only supervise observation publications plus one pure helper changed; other process functions unchanged")
class RemoveObservationPublications(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id == "publish" and any(isinstance(value,ast.Constant) and value.value in {"direct_controller_initial_child_observation.json","direct_controller_pre_registration_child_observation.json"} for value in ast.walk(node)):
            return None
        return node
retained = RemoveObservationPublications().visit(copy.deepcopy(new_functions["supervise"]))
require(ast.dump(retained,include_attributes=False) == ast.dump(old_functions["supervise"],include_attributes=False), "removing only two new observation publications yields exact original supervise AST, including all identity/wait/registration checks")
visibility = read(P / "root_portal_plan_visibility_before_claim_probe.json")
require(visibility["prepare_and_verify_completed_once"] is True and visibility["claim_probe_not_attempted"] is True and visibility["bind_not_attempted"] is True and visibility["plan_visible_now"] is True and visibility["actual_prepare_result"] == ref(U / "measurement_plan.json"), "initial portal visibility error retained before once-only claim and bind; no phase rerun")
require(datetime.datetime.fromisoformat(visibility["at_utc"]) < datetime.datetime.fromisoformat(probe_intent["at_utc"]), "saved visibility observation precedes actual once-only claim probe")
source_mapping = read(P / "canonical_controller_source_mapping.json")
require(source_mapping["copied_preparer_is_archival_only"] is True and source_mapping["legacy_names_admit_no_correctness_operations"] is True, "canonical current source is used; copied legacy names grant no old operations")
for original, value in source_mapping["files"].items():
    require(ref(original)["sha256"] == ref(value["staged"])["sha256"] == value["sha256"], "actual canonical source equals frozen staged metadata source")

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
          "additional_failed_startup_facts": ref(P / "separate_failed_attempt_closure_facts.json"), "observation_only_controller_source_commit": request["wrapper_source_commit"], "observation_only_source_review": ref(P / "initial_child_identity_source_independent_review.json"), "actual_CPU_identity_diagnostic_review": ref(proof_ref["path"]), "all_original_identity_checks_preserved": True, "all25_process_functions_unchanged_claimed": False, "two_observation_publications_added_to_supervise": True, "scientific_predecessor_unchanged": ref(P / "serial_predecessor_exit_inputs.json"),
          "actual_owned_cpu_job_id": "6821437", "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "actual_cpu_process": {"pid": os.getpid(), "hostname": socket.gethostname(), "uids": list(os.getresuid()), "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19], "cgroup": Path("/proc/self/cgroup").read_text()},
          "checks": CHECKS, "elapsed_seconds": time.monotonic() - START, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "required_before_actual_target_launch": ["Publish the exact actual source freeze, prepared records and this review to obtain a real inner controller publication.", "Derive the detached root starter from unchanged060ee process/barrier/custody functions; any path/publication substitutions must be inspected and published separately.", "Run a publication-aware default metadata check with the real inner commit.", "Root alone executes one reviewed SSH after own metadata permission protection; sole numerical producer owns standing GPU descriptor.", "Keep second-copy timing unbound until actual first-copy timing normal zero wait and both exact exits."],
          "actual_target_or_gpu_or_remote_action": False, "actual_shared_claim_producer_or_causal_cases_repeated_by_reviewer": False, "large_historical_payloads_replayed": False,
          "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admission": False}
out = P / "controller_root_independent_review.json"
with out.open("x") as handle:
    json.dump(review, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"passed": True, "review": ref(out), "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "checks": len(CHECKS), "elapsed_seconds": review["elapsed_seconds"]}))
