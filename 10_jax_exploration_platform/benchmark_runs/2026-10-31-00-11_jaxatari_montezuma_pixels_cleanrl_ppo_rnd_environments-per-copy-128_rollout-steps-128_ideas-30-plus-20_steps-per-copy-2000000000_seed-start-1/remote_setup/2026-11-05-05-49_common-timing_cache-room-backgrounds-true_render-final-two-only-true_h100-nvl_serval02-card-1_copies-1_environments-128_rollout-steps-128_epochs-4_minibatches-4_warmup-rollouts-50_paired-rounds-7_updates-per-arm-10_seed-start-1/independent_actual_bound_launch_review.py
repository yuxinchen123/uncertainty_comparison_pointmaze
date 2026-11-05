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
require(ref(freeze_path)["sha256"] == "9236545885b07c2a2764decbc4c75657e031f72aeb52593c4130c415c3847ad1", "actual held fresh source freeze")
freeze = read(freeze_path)
require(len(freeze["files"]) == 87, "all87 actual frozen packet, provenance and closure inputs")
for name, expected in freeze["files"].items():
    require(ref(name)["sha256"] == expected, "unchanged actual frozen input: " + Path(name).name)
plan = read(U / "measurement_plan.json")
request = read(P / "preparation_request.json")
require(ref(U / "measurement_plan.json")["sha256"] == "4dd7ea9dcc7f70ebc85170780b36c9564790017cd06ce7d3fbf8e1b45a7d952c", "actual new prepared plan")
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
require(context["job"] == "6821437" and context["step"] == "93" and "/job_6821437/step_93/" in context["cgroup"] and type(context["pid"]) is int and context["pid"] > 0, "actual claim proof has exact positive owned CPU-step identity")
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
closed = read(P / "separate_failed_startup_closures.json")
require(closed["kind"] == "three_actual_closed_failed_startups_separate_from_scientific_predecessor" and len(closed["attempts"]) == 3 and closed["new_process_queries"] is False and closed["success_or_wait_records_fabricated"] is False, "all three actual failed startups remain closed and separate from accepted scientific predecessor")
for index, attempt in enumerate(closed["attempts"]):
    require(attempt["remains_failed"] is True and attempt["is_accepted_scientific_predecessor"] is False, "failed startup remains failed and is not substituted for scientific predecessor")
    require(ref(attempt["failed_outcome"]["path"]) == attempt["failed_outcome"], "exact original failed outcome bytes")
    outcome = read(attempt["failed_outcome"]["path"])
    require(outcome["passed"] is False and outcome["actual_child_started"] is attempt["actual_child_started"] and outcome["actual_wait_saved"] is attempt["actual_wait_saved"], "actual failed outcome matches recorded real child/wait flags")
    for value in attempt["facts"].values():
        require(ref(value["path"]) == value, "unchanged actual failed-startup closure fact")
    if index == 0:
        require(attempt["actual_child_started"] is False and attempt["actual_wait_saved"] is False, "old02-24 remains failed before any numerical child; no invented wait")
        exit_record = read(attempt["facts"]["root_failed_startup_controller_exit_observation.json"]["path"])
        require(exit_record["actual_predecessor_ended"] is True and exit_record["process_directory_exists"] is False and exit_record["process_signals_sent"] == 0 and exit_record["pid"] == 321183 and exit_record["expected_birth_counter"] == "83855111", "actual old02-24 controller exit was observed without signals")
    else:
        wait = read(attempt["facts"]["actual_child_wait"]["path"])
        expected_pid, expected_birth = (351650,"84242978") if index == 1 else (371111,"84596712")
        require(wait["pid"] == wait["waited_pid"] == expected_pid and wait["process_start_time"] == expected_birth and wait["returncode"] == 1 and wait["exited_normally"] is True and wait["timeouts"] == [], "actual failed numerical child has normal exact wait one, no intervention")
        if index == 1:
            exits = read(attempt["facts"]["root_failed_identity_startup_exact_exit_observation.json"]["path"])
            require(exits["all_original_processes_ended"] is True and exits["signals_sent"] == 0 and {(row["pid"],row["expected_birth_counter"]) for row in exits["observations"]} == {(351185,"84237458"),(351650,"84242978")}, "actual old04-07 controller and child both ended")
        else:
            terminal = read(attempt["facts"]["independent_actual_empty_command_terminal_addendum.json"]["path"])
            require(ref(attempt["facts"]["independent_actual_empty_command_terminal_addendum.json"]["path"])["sha256"] == "933b444d68d29d0305480ae9b0db7b5b26a8d477643e75323b42ee79fd9326d2" and terminal["passed"] is True and terminal["attempt_remains_failed"] is True and terminal["both_exact_process_exits_verified"] is True and terminal["metadata_restore_after_actual_wait_and_both_exits_verified"] is True, "genuine old05-10 terminal addendum proves failed wait, both exits and same-file metadata restoration")
require(closed["scientific_predecessor_unchanged"]["sha256"] == ref(P / "serial_predecessor_exit_inputs.json")["sha256"], "genuine oldC2 scientific predecessor bytes unchanged across all failed closures")

# Verify the explicit thirty-second empty-only protocol and exact previously reviewed source.
require(request["wrapper_source_commit"] == "d2253fa853290cb8fcd0a9212e332dd33a63f2f9", "exact reviewed bounded-startup controller source commit")
require(ref(P / "direct_qualification_controller.py")["sha256"] == "8e567998fc5209f5ff4370338fe62bf78391b7c5637adf989572b42486adcb0e", "actual strict startup controller source8e56 staged exactly")
for name, expected_hash in {"test_empty_command_startup.py":"0393d6ad44041cb2fe8db64f0103c83138351334a0e4275b389ea22f65b64cc5", "test_initial_child_identity.py":"6b6d89f2b5d01e08521680e23bea314287652f8e921350d25a3bb756d38dbd3d", "bounded_startup_source_freeze.json":"9fa36c9cef144d1008d9c91af41a301b75d44da863dcb00999706383f16052d1", "bounded_startup_source_independent_review.json":"8b4f58210ba5557b5b0ae71d7f6d069839814444c883909748ca256f19e69a42"}.items():
    require(ref(P / name)["sha256"] == expected_hash and str(P / name) in freeze["files"], "new affected protocol/source/test proof is in actual frozen closure: " + name)
source_review = read(P / "bounded_startup_source_independent_review.json")
require(source_review["approved"] is source_review["passed"] is True and source_review["maximum_startup_wait_seconds"] == 30 and source_review["nonempty_command_requires_complete_exact_raw_bytes"] is True and source_review["all_UID_PID_birth_PPID_affinity_rechecked_each_observation"] is True and source_review["GPU_or_worker_admission"] is False, "genuine independent CPU source review covers bounded empty-only readiness with no GPU admission")
protocol = read(P / "controller_startup_protocol.json")
require(protocol["controller_source_commit"] == request["wrapper_source_commit"] and protocol["controller_source_sha256"] == ref(P / "direct_qualification_controller.py")["sha256"] and type(protocol["maximum_startup_seconds"]) is int and protocol["maximum_startup_seconds"] == 30, "new startup protocol explicitly binds actual source and strict30second limit")
require(all(protocol[key] is True for key in ("wait_only_for_truly_empty_raw_command_bytes", "every_snapshot_requires_same_actual_owned_PID_birth_parent_affinity_and_raw_hashes", "exact_complete_expected_raw_command_required_before_registration", "wrong_nonempty_or_changed_identity_or_exit_or_deadline_refuses", "numerical_and_helper_source_unchanged")) and protocol["new_scientific_or_hardware_admission"] is False, "only genuinely empty raw bytes may wait; exact original identity and complete expected argv still required before registration")
old_packet = next((R / "remote_setup").glob("2026-11-05-05-10_common-timing*"))
old_source = (old_packet / "direct_qualification_controller.py").read_text()
new_source = (P / "direct_qualification_controller.py").read_text()
old_functions = {node.name:node for node in ast.parse(old_source).body if isinstance(node,ast.FunctionDef)}
new_functions = {node.name:node for node in ast.parse(new_source).body if isinstance(node,ast.FunctionDef)}
require({name for name in old_functions if ast.dump(old_functions[name],include_attributes=False) != ast.dump(new_functions[name],include_attributes=False)} == {"supervise"} and set(new_functions)-set(old_functions) == {"wait_for_exact_child_startup"}, "only supervise plus one strict startup observer changed from09a; first empty-command refusal timing explicitly changes")
class RemoveStartupObserver(ast.NodeTransformer):
    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call) and (any(isinstance(target,ast.Name) and target.id == "startup_started" for target in node.targets) or isinstance(node.value.func,ast.Name) and node.value.func.id == "wait_for_exact_child_startup"):
            return None
        return node
retained = RemoveStartupObserver().visit(copy.deepcopy(new_functions["supervise"]))
require(ast.dump(retained,include_attributes=False) == ast.dump(old_functions["supervise"],include_attributes=False), "outside explicit startup clock/observer all original checks, observations, registration/publication and exact wait remain unchanged")
source_mapping = read(P / "canonical_controller_source_mapping.json")
require(source_mapping["copied_preparer_is_archival_only"] is True and source_mapping["legacy_names_admit_no_correctness_operations"] is True, "actual canonical metadata preparer used; copied legacy names are archival only")
for original, value in source_mapping["files"].items():
    require(ref(original)["sha256"] == ref(value["staged"])["sha256"] == value["sha256"], "actual canonical source equals frozen staged metadata source")
prepare_stdout = read(P / "root_actual_prepare_stdout.raw")
require(prepare_stdout["operation"] == "prepare" and prepare_stdout["result"] == ref(U / "measurement_plan.json") and prepare_stdout["jobs_submitted"] == prepare_stdout["remote_actions"] == prepare_stdout["gpu_operations"] == 0, "actual successful prepare receipt itself provides exact new plan hash; no target action")
require(probe_intent["argv"][probe_intent["argv"].index("--expected-plan-sha256")+1] == prepare_stdout["result"]["sha256"], "once-only probe expected hash is actual prepare stdout receipt, not an inferred future or portal-cached value")

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
          "additional_failed_startup_facts": ref(P / "separate_failed_startup_closures.json"), "bounded_startup_controller_source_commit": request["wrapper_source_commit"], "bounded_startup_source_review": ref(P / "bounded_startup_source_independent_review.json"), "explicit_new_startup_protocol": ref(P / "controller_startup_protocol.json"), "first_empty_command_refusal_timing_changed_explicitly": True, "exact_complete_raw_command_comparison_scope": "startup ready observation only; later pre-registration uses unchanged decoded-command, birth, parent and affinity checks", "all_complete_expected_argv_and_UID_PID_birth_parent_affinity_checks_required_before_registration": True, "all_initial_refusal_behavior_unchanged_claimed": False, "all25_process_functions_unchanged_claimed": False, "scientific_predecessor_unchanged": ref(P / "serial_predecessor_exit_inputs.json"),
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
